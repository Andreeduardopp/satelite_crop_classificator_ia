"""
Treina um classificador de culturas (milho, soja, trigo) usando
ViT-Small/16 (timm, PyTorch) com informação multi-temporal.

Diferenças em relação à v1:
    - PyTorch nativo via timm (sem dependência de HuggingFace TF)
    - vit_small_patch16_224 (22M params vs 86M do vit_base)
    - O dia após plantio (d21, d31, d56...) é extraído do nome do arquivo
      e alimentado ao modelo como feature adicional, permitindo ao modelo
      aprender padrões específicos por estágio de crescimento
    - Suporta ablação multi-temporal: treinar com todas as datas ou
      filtrar por janela temporal específica (--dias 21 31)

Fases:
    1. Base congelada — treina apenas a cabeça de classificação
    2. Fine-tuning  — descongela os últimos N blocos do encoder ViT

Uso:
    # Treino com todas as datas (modelo aprende a usar a info temporal)
    python src/treinamento/treinar_classificador_vit_v2.py

    # Ablação: treinar apenas com imagens de d56
    python src/treinamento/treinar_classificador_vit_v2.py --dias 56

    # Ablação: treinar apenas com d21 e d31
    python src/treinamento/treinar_classificador_vit_v2.py --dias 21 31
"""

import os
import re
import sys
import ast
import time
import logging
import argparse
import sqlite3
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

import timm

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f'treino_vit_v2_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)

# ── Configurações ─────────────────────────────────────────────────────────────
DB_PATH          = './sample_treino.db'
TABELA           = 'culturas'
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32
EPOCHS_FASE1     = 10
EPOCHS_FASE2     = 15
LR_FASE1         = 1e-3
LR_FASE2         = 1e-5
MODELO_SAIDA     = './modelos/classificador_cultura_vit_v2'
CLASSES          = ['milho', 'soja', 'trigo']
SEED             = 42
VIT_MODEL_NAME   = 'vit_small_patch16_224'
FINE_TUNE_BLOCOS = 2
MAX_DIA          = 100.0   # para normalizar o dia em [0, 1]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dados ─────────────────────────────────────────────────────────────────────

def extrair_dia(caminho: str) -> int:
    """Extrai o dia após plantio do nome do arquivo (e.g. _d56.png → 56)."""
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_dados(db_path: str, filtro_dias: list[int] | None = None
                   ) -> tuple[list[list[tuple[str, int]]], list[int]]:
    """
    Retorna (registros, labels) onde cada registro é uma lista de
    (caminho, dia) pertencentes ao MESMO talhão.

    Se filtro_dias for fornecido, mantém apenas imagens dos dias especificados.
    """
    classe_para_id = {c: i for i, c in enumerate(CLASSES)}
    registros = []
    labels = []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT cultura, imagens_processadas FROM {TABELA}"
        ).fetchall()

    for cultura, imgs_str in rows:
        if cultura not in classe_para_id:
            continue
        try:
            paths = ast.literal_eval(imgs_str)
        except (ValueError, SyntaxError):
            continue

        validos = []
        for p in paths:
            if not os.path.exists(p):
                continue
            dia = extrair_dia(p)
            if filtro_dias and dia not in filtro_dias:
                continue
            validos.append((p, dia))

        if validos:
            registros.append(validos)
            labels.append(classe_para_id[cultura])

    return registros, labels


def expandir_registros(registros: list[list[tuple[str, int]]], labels: list[int]
                       ) -> tuple[list[str], list[int], list[int]]:
    """Expande registros em listas planas de (caminhos, dias, labels)."""
    caminhos = []
    dias = []
    labels_exp = []
    for items, label in zip(registros, labels):
        for caminho, dia in items:
            caminhos.append(caminho)
            dias.append(dia)
            labels_exp.append(label)
    return caminhos, dias, labels_exp


# ── Dataset PyTorch ───────────────────────────────────────────────────────────

class CulturaDataset(Dataset):
    def __init__(self, caminhos: list[str], dias: list[int], labels: list[int]):
        self.caminhos = caminhos
        self.dias = dias
        self.labels = labels

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        img = cv2.imread(self.caminhos[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)

        # HWC uint8 → CHW float32 normalizado para ImageNet
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW

        dia_norm = self.dias[idx] / MAX_DIA

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(dia_norm, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── Modelo ────────────────────────────────────────────────────────────────────

class ViTTemporalClassificador(nn.Module):
    """
    ViT-Small/16 + embedding temporal do dia após plantio.

    Arquitetura:
        Imagem → vit_small (384-dim)  ─┐
                                        ├─ concat → Dense(256) → ReLU → Dropout → Dense(3)
        Dia    → Linear(32-dim)       ─┘
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = timm.create_model(VIT_MODEL_NAME, pretrained=True, num_classes=0)
        self.vit_dim = self.vit.num_features  # 384 for vit_small

        # Congela backbone inicialmente
        for param in self.vit.parameters():
            param.requires_grad = False

        # Branch temporal
        self.dia_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
        )

        # Cabeça de classificação
        self.head = nn.Sequential(
            nn.Linear(self.vit_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img: torch.Tensor, dia: torch.Tensor) -> torch.Tensor:
        features = self.vit(img)                          # (B, 384)
        dia_feat = self.dia_embed(dia.unsqueeze(-1))      # (B, 32)
        combined = torch.cat([features, dia_feat], dim=1)  # (B, 416)
        return self.head(combined)                         # (B, num_classes)

    def descongelar_ultimos_blocos(self, n_blocos: int):
        """Descongela os últimos n blocos do encoder ViT para fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = False

        for block in self.vit.blocks[-n_blocos:]:
            for param in block.parameters():
                param.requires_grad = True


# ── Treino ────────────────────────────────────────────────────────────────────

def treinar_fase(modelo, loader_treino, loader_val, optimizer, criterion,
                 epochs, class_weight_tensor, fase_nome, patience):
    best_val_loss = float('inf')
    best_state = None
    epochs_sem_melhora = 0

    for epoch in range(epochs):
        # ── Train ──
        modelo.train()
        total_loss = 0
        corretos = 0
        total = 0

        for imgs, dias, labels in loader_treino:
            imgs, dias, labels = imgs.to(DEVICE), dias.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = modelo(imgs, dias)

            # Aplica pesos de classe por amostra
            pesos_amostra = class_weight_tensor[labels]
            loss = (criterion(logits, labels) * pesos_amostra).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            corretos += (logits.argmax(dim=1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc = corretos / total

        # ── Validação ──
        modelo.eval()
        val_loss_total = 0
        val_corretos = 0
        val_total = 0

        with torch.no_grad():
            for imgs, dias, labels in loader_val:
                imgs, dias, labels = imgs.to(DEVICE), dias.to(DEVICE), labels.to(DEVICE)
                logits = modelo(imgs, dias)
                loss = criterion(logits, labels).mean()
                val_loss_total += loss.item() * imgs.size(0)
                val_corretos += (logits.argmax(dim=1) == labels).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_corretos / val_total

        logging.info(
            f"[{fase_nome}] Epoch {epoch+1}/{epochs} — "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in modelo.state_dict().items()}
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= patience:
                logging.info(f"Early stopping após {patience} épocas sem melhora.")
                break

    if best_state:
        modelo.load_state_dict(best_state)


def avaliar(modelo, loader, class_names):
    """Avalia o modelo e imprime métricas completas."""
    modelo.eval()
    y_true = []
    y_pred = []
    n_samples = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for imgs, dias, labels in loader:
            imgs, dias = imgs.to(DEVICE), dias.to(DEVICE)
            logits = modelo(imgs, dias)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
            n_samples += imgs.size(0)

    t_total = time.perf_counter() - t_start

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    acc = np.mean(np.array(y_true) == np.array(y_pred))

    logging.info(f"Validação final — Accuracy: {acc:.4f}")
    logging.info(f"F1-score (macro): {f1_macro:.4f}")
    for cls_name, f1_cls in zip(class_names, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_true, y_pred, target_names=class_names
    ))

    tempo_medio = (t_total / n_samples) * 1000
    logging.info(f"Tempo médio de inferência: {tempo_medio:.2f} ms/amostra ({n_samples} amostras)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Treinar ViT v2 com info multi-temporal')
    parser.add_argument('--dias', nargs='+', type=int, default=None,
                        help='Filtrar por dias específicos (ex: --dias 21 56). Omitir = usar todos.')
    args = parser.parse_args()

    filtro_dias = args.dias
    if filtro_dias:
        logging.info(f"ABLAÇÃO: treinando apenas com dias {filtro_dias}")
    else:
        logging.info("Treinando com TODOS os dias (modelo multi-temporal)")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Carregar dados (agrupados por registro/talhão)
    logging.info("Carregando dados...")
    registros, labels = carregar_dados(DB_PATH, filtro_dias=filtro_dias)
    total_imgs = sum(len(r) for r in registros)
    logging.info(f"Total: {len(registros)} registros, {total_imgs} imagens válidas")

    for i, c in enumerate(CLASSES):
        n = labels.count(i)
        logging.info(f"  {c}: {n} registros")

    # 2. Split por registro (evita data leakage)
    reg_treino, reg_val, lab_reg_treino, lab_reg_val = train_test_split(
        registros, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    cam_treino, dias_treino, lab_treino = expandir_registros(reg_treino, lab_reg_treino)
    cam_val, dias_val, lab_val = expandir_registros(reg_val, lab_reg_val)
    logging.info(f"Treino: {len(cam_treino)} imgs ({len(reg_treino)} regs) | "
                 f"Validação: {len(cam_val)} imgs ({len(reg_val)} regs)")

    # 3. DataLoaders
    ds_treino = CulturaDataset(cam_treino, dias_treino, lab_treino)
    ds_val = CulturaDataset(cam_val, dias_val, lab_val)
    loader_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 4. Class weights
    pesos = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=np.array(lab_treino))
    class_weight_tensor = torch.tensor(pesos, dtype=torch.float32).to(DEVICE)
    logging.info(f"Class weights: {dict(zip(CLASSES, pesos))}")

    # 5. Modelo
    logging.info(f"Carregando {VIT_MODEL_NAME} (pretrained ImageNet)...")
    modelo = ViTTemporalClassificador(len(CLASSES)).to(DEVICE)

    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Parâmetros: {total_params:,} total | {trainable_params:,} treináveis")

    # ── Fase 1: Base congelada ────────────────────────────────────────────
    logging.info("=== Fase 1: Treinando cabeça (base congelada) ===")
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()), lr=LR_FASE1
    )
    treinar_fase(modelo, loader_treino, loader_val, optimizer, criterion,
                 EPOCHS_FASE1, class_weight_tensor, "Fase 1", patience=3)

    # ── Fase 2: Fine-tuning ──────────────────────────────────────────────
    logging.info(f"=== Fase 2: Fine-tuning (últimos {FINE_TUNE_BLOCOS} blocos) ===")
    modelo.descongelar_ultimos_blocos(FINE_TUNE_BLOCOS)

    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Parâmetros treináveis após descongelamento: {trainable_params:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()), lr=LR_FASE2
    )
    treinar_fase(modelo, loader_treino, loader_val, optimizer, criterion,
                 EPOCHS_FASE2, class_weight_tensor, "Fase 2", patience=4)

    # ── Salvar ────────────────────────────────────────────────────────────
    os.makedirs(MODELO_SAIDA, exist_ok=True)
    sufixo = f"_d{'_'.join(map(str, filtro_dias))}" if filtro_dias else "_all"
    peso_path = os.path.join(MODELO_SAIDA, f'pesos{sufixo}.pt')
    torch.save(modelo.state_dict(), peso_path)
    logging.info(f"Pesos salvos em: {peso_path}")

    # ── Avaliação final ──────────────────────────────────────────────────
    avaliar(modelo, loader_val, CLASSES)


if __name__ == '__main__':
    main()
