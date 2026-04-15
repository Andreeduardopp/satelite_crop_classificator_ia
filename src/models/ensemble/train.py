"""
Treina um backbone específico com a cabeça temporal (FiLM + Attention).

Uso:
    python train.py --backbone efficientnet_b0
    python train.py --backbone resnet50
    python train.py --backbone convnext_tiny
    python train.py --all                        # Treina os 3 sequencialmente

Segue o mesmo pipeline de 2 fases do V7:
    Fase 1: Backbone congelado, treina FiLM + attention + head
    Fase 2: Fine-tuning das últimas camadas do backbone
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

from data import (
    carregar_dados, TemporalCulturaDataset,
    CLASSES, DB_TREINO, DEVICE, USE_AMP, SEED, MAX_SEQ_LEN
)
from model import TemporalCulturaModel, BACKBONE_CONFIG

# -- Configurações de treino ---------------------------------------------------
_HERE          = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR  = os.path.join(_HERE, 'artifacts')
METRICS_DIR    = os.path.join(_HERE, 'metrics')
LOG_DIR        = os.path.join(_HERE, 'logs')

BATCH_SIZE       = 64
EPOCHS_FASE1     = 10
EPOCHS_FASE2     = 15
LR_FASE1         = 1e-3
LR_FASE2         = 1e-5
GRAD_CLIP        = 1.0
LABEL_SMOOTHING  = 0.1
NUM_WORKERS      = 4

BACKBONES_DISPONIVEIS = list(BACKBONE_CONFIG.keys())

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True


def setup_logging(backbone_name: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(LOG_DIR, f'treino_{backbone_name}_{ts}.txt')

    # Limpar handlers existentes para evitar duplicação entre backbones
    root = logging.getLogger()
    root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_file


# -- Loop de treino (idêntico ao V7) ------------------------------------------

def treinar_fase(
    modelo, loader_treino, loader_val, optimizer, criterion,
    epochs, class_weight_tensor, fase_nome, patience, scaler=None,
):
    best_val_loss = float('inf')
    best_state = None
    epochs_sem_melhora = 0

    for epoch in range(epochs):
        modelo.train()
        total_loss = corretos = total = 0

        for images, dias, mes, mask, labels in loader_treino:
            images = images.to(DEVICE, non_blocking=True)
            dias   = dias.to(DEVICE, non_blocking=True)
            mes    = mes.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = modelo(images, dias, mes, mask)
                loss = (criterion(logits, labels) * class_weight_tensor[labels]).mean()

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in modelo.parameters() if p.requires_grad),
                    max_norm=GRAD_CLIP,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in modelo.parameters() if p.requires_grad),
                    max_norm=GRAD_CLIP,
                )
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            corretos += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = total_loss / total
        train_acc = corretos / total

        # Validação
        modelo.eval()
        vl_loss = vl_corr = vl_tot = 0
        with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=USE_AMP):
            for images, dias, mes, mask, labels in loader_val:
                images = images.to(DEVICE, non_blocking=True)
                dias   = dias.to(DEVICE, non_blocking=True)
                mes    = mes.to(DEVICE, non_blocking=True)
                mask   = mask.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                logits = modelo(images, dias, mes, mask)
                vl_loss += criterion(logits, labels).mean().item() * images.size(0)
                vl_corr += (logits.argmax(1) == labels).sum().item()
                vl_tot += images.size(0)

        val_loss = vl_loss / vl_tot
        val_acc = vl_corr / vl_tot

        logging.info(
            f"[{fase_nome}] Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in modelo.state_dict().items()}
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= patience:
                logging.info(f"Early stopping apos {patience} epochs sem melhora.")
                break

    if best_state:
        modelo.load_state_dict(best_state)


# -- Pipeline completo para um backbone ---------------------------------------

def treinar_backbone(backbone_name: str):
    """Treina um backbone do zero e salva pesos + métricas."""
    log_file = setup_logging(backbone_name)
    logging.info("="*70)
    logging.info(f"ENSEMBLE: Treinando {backbone_name}")
    logging.info(f"Device: {DEVICE} | AMP: {USE_AMP}")
    logging.info("="*70)

    # 1. Carregar dados
    logging.info(f"Carregando dados de: {DB_TREINO}")
    registros, labels, meses = carregar_dados(DB_TREINO)
    logging.info(f"Total: {len(registros)} talhoes, {sum(len(r) for r in registros)} imagens")

    if not registros:
        logging.error("Nenhum dado encontrado!")
        return

    for i, c in enumerate(CLASSES):
        logging.info(f"  {c}: {labels.count(i)} talhoes")

    # 2. Split
    reg_tr, reg_val, lab_tr, lab_val, mes_tr, mes_val = train_test_split(
        registros, labels, meses, test_size=0.2, stratify=labels, random_state=SEED
    )
    logging.info(f"Treino: {len(reg_tr)} | Validacao: {len(reg_val)}")

    # 3. DataLoaders
    ds_tr  = TemporalCulturaDataset(reg_tr, lab_tr, mes_tr)
    ds_val = TemporalCulturaDataset(reg_val, lab_val, mes_val)
    pin = DEVICE.type == 'cuda'
    loader_tr  = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=NUM_WORKERS > 0)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=NUM_WORKERS > 0)

    # 4. Class weights (tolerante a classes ausentes no dataset)
    y_arr = np.array(lab_tr)
    present_classes = np.unique(y_arr)
    pesos_present = compute_class_weight('balanced', classes=present_classes, y=y_arr)
    cw_tensor = torch.ones(len(CLASSES), dtype=torch.float32).to(DEVICE)
    for idx, cls_id in enumerate(present_classes):
        cw_tensor[int(cls_id)] = pesos_present[idx]
    logging.info(f"Class weights: {dict(zip(CLASSES, cw_tensor.cpu().numpy().round(3)))}")
    logging.info(f"Classes com dados: {[CLASSES[c] for c in present_classes]}")

    # 5. Modelo
    logging.info(f"Criando modelo: {backbone_name} + FiLM + Temporal Attention")
    modelo = TemporalCulturaModel(backbone_name, len(CLASSES)).to(DEVICE)
    logging.info(f"Feature dim: {modelo.feature_dim}")

    total_p = sum(p.numel() for p in modelo.parameters())
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Params: {total_p:,} total | {train_p:,} treinaveis")

    scaler = GradScaler(device=DEVICE.type) if USE_AMP else None
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=LABEL_SMOOTHING)

    # 6. Fase 1: Backbone congelado
    t_train_start = time.perf_counter()
    logging.info("=== Fase 1: Cabeca + FiLM + Attention (base congelada) ===")
    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE1
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, criterion,
                 EPOCHS_FASE1, cw_tensor, "Fase1", patience=3, scaler=scaler)

    # 7. Fase 2: Fine-tuning
    ft_layers = BACKBONE_CONFIG[backbone_name]['fine_tune_layers']
    logging.info(f"=== Fase 2: Fine-tuning (ultimas {ft_layers} camadas) ===")
    modelo.descongelar_ultimas_camadas(ft_layers)
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Parametros treinaveis: {train_p:,}")

    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE2
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, criterion,
                 EPOCHS_FASE2, cw_tensor, "Fase2", patience=4, scaler=scaler)

    t_train_total = time.perf_counter() - t_train_start
    train_time_mins = t_train_total / 60.0
    logging.info(f"Tempo total de treinamento (Fases 1 + 2): {train_time_mins:.2f} minutos")

    # 8. Salvar pesos
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    peso_path = os.path.join(ARTIFACTS_DIR, f'pesos_{backbone_name}.pt')
    torch.save(modelo.state_dict(), peso_path)
    logging.info(f"Pesos salvos: {peso_path}")

    # 9. Avaliação final
    modelo.eval()
    y_true, y_pred = [], []
    t_start = time.perf_counter()

    with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=USE_AMP):
        for images, dias, mes, mask, labels in loader_val:
            images = images.to(DEVICE, non_blocking=True)
            dias   = dias.to(DEVICE, non_blocking=True)
            mes    = mes.to(DEVICE, non_blocking=True)
            mask   = mask.to(DEVICE, non_blocking=True)

            logits = modelo(images, dias, mes, mask)
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())

    t_total = time.perf_counter() - t_start

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    logging.info(f"Resultado {backbone_name} — Acc: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    for cls_name, f1_cls in zip(CLASSES, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_true, y_pred, target_names=CLASSES, zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    logging.info("Confusion Matrix:")
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    logging.info(header)
    for cls, row in zip(CLASSES, cm):
        logging.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    tempo_medio = (t_total / len(y_true)) * 1000
    logging.info(f"Tempo medio: {tempo_medio:.2f} ms/talhao")

    # 10. Salvar métricas
    os.makedirs(METRICS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    m_path = os.path.join(METRICS_DIR, f'{backbone_name}_{ts}.json')
    with open(m_path, 'w') as f:
        json.dump({
            'backbone': backbone_name,
            'feature_dim': modelo.feature_dim,
            'f1_macro': float(f1_macro),
            'accuracy': float(acc),
            'f1_per_class': {c: float(v) for c, v in zip(CLASSES, f1_per_class)},
            'tempo_medio_ms': float(tempo_medio),
            'tempo_treino_minutos': float(train_time_mins),
        }, f, indent=4)

    logging.info(f"Metricas salvas: {m_path}")
    logging.info(f"Tempo total de treino de {backbone_name}: {train_time_mins:.2f} minutos")
    logging.info(f"Log completo: {log_file}")
    logging.info("="*70 + "\n")

    return acc, f1_macro, train_time_mins


def main():
    parser = argparse.ArgumentParser(description='Treina backbone para o ensemble')
    parser.add_argument('--backbone', type=str, choices=BACKBONES_DISPONIVEIS,
                        help='Backbone para treinar')
    parser.add_argument('--all', action='store_true',
                        help='Treinar todos os 3 backbones sequencialmente')
    args = parser.parse_args()

    if not args.all and not args.backbone:
        parser.error('Especifique --backbone NOME ou --all')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    backbones = BACKBONES_DISPONIVEIS if args.all else [args.backbone]
    resultados = {}

    for bb in backbones:
        logging.info(f"\n{'#'*70}")
        logging.info(f"# TREINANDO: {bb}")
        logging.info(f"{'#'*70}\n")
        acc, f1, train_time_mins = treinar_backbone(bb)
        resultados[bb] = {'accuracy': acc, 'f1_macro': f1, 'tempo_treino_minutos': train_time_mins}

    # Resumo final
    if len(resultados) > 1:
        print("\n" + "="*70)
        print("RESUMO DO ENSEMBLE — Resultados Individuais")
        print("="*70)
        print(f"{'Backbone':<25} {'Accuracy':>10} {'F1-macro':>10} {'Tempo Treino (min)':>18}")
        print("-"*75)
        for bb, r in resultados.items():
            print(f"{bb:<25} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r.get('tempo_treino_minutos', 0):>18.2f}")
        print("="*75)
        print("Proximo passo: python extrator.py")


if __name__ == '__main__':
    main()
