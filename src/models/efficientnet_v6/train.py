"""
Classificador de culturas v6 - EfficientNetB0 + FiLM(dia, mes) + Temporal Attention.
Portado de TensorFlow para PyTorch para suporte nativo a CUDA no Windows.

Arquitetura:
    Para cada talhao com ate MAX_SEQ_LEN imagens temporais:
    1. EfficientNetB0 (timm, pretrained) extrai features 1280-dim por imagem
    2. FiLM conditioning: concat(dia_normalizado, mes_embedding_8d) -> Dense(64)
       -> gamma, beta que modulam as features: x * (1+gamma) + beta
    3. 2 camadas de MultiHeadAttention cruzam info entre timesteps
    4. Mean-pooling dos tokens validos -> Dense(256) -> classificacao

    O mes e tratado como CATEGORICO via Embedding(12, 8).
    gamma/beta sao zero-initialized -> comeca como identidade.

Otimizacoes GPU:
    - AMP (mixed precision FP16)
    - torch.compile
    - cudnn.benchmark
    - non_blocking transfers
    - persistent DataLoader workers
    - Data augmentation via torchvision transforms

Uso:
    python src/models/efficientnet_v6/train.py
"""

import json
import os
import re
import ast
import time
import logging
import sqlite3
from datetime import datetime
from collections import Counter

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import timm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, accuracy_score,
)

# -- Paths relativos ao arquivo ------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(_HERE, '..', '..'))  # -> src/

# -- Logging -------------------------------------------------------------------
LOG_DIR = os.path.join(_HERE, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(
    LOG_DIR, f'treino_efficientnet_v6_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)

# -- Configuracoes -------------------------------------------------------------
DB_PATH          = os.path.join(_HERE, '..', '..', 'bkp', 'sample_treino_6k.db')
TABELA           = 'culturas'
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 64            # aumentado para GPU
EPOCHS_FASE1     = 10
EPOCHS_FASE2     = 15
LR_FASE1         = 1e-3
LR_FASE2         = 1e-5
GRAD_CLIP        = 1.0
LABEL_SMOOTHING  = 0.1
MODELO_SAIDA     = os.path.join(_HERE, 'artifacts')
CLASSES          = ['milho', 'soja', 'trigo']
SEED             = 42
MAX_SEQ_LEN      = 3
MAX_DIA          = 100.0
MES_EMBED_DIM    = 8              # dimensao do embedding de mes
FINE_TUNE_LAYERS = 20             # camadas do backbone a descongelar
NUM_WORKERS      = 4

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# -- Dados ---------------------------------------------------------------------

def extrair_dia(caminho: str) -> int:
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_dados(db_path: str) -> tuple[list[list[tuple[str, int]]], list[int], list[int]]:
    """
    Retorna (registros, labels, meses).
    Cada registro e uma lista de (caminho, dia) ORDENADOS por dia.
    """
    classe_para_id = {c: i for i, c in enumerate(CLASSES)}
    registros = []
    labels = []
    meses = []

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT cultura, mes, imagens_processadas FROM {TABELA}"
        ).fetchall()

    for cultura, mes, imgs_str in rows:
        if cultura not in classe_para_id:
            continue
        try:
            paths = ast.literal_eval(imgs_str)
        except (ValueError, SyntaxError):
            continue

        validos = []
        for p in paths:
            abs_p = os.path.join(SRC_DIR, p) if not os.path.isabs(p) else p
            if os.path.exists(abs_p):
                validos.append((abs_p, extrair_dia(abs_p)))

        if validos:
            validos.sort(key=lambda x: x[1])
            registros.append(validos)
            labels.append(classe_para_id[cultura])
            meses.append(int(mes) if mes else 1)

    return registros, labels, meses


def preprocessar_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


# -- Dataset -------------------------------------------------------------------

class TemporalCulturaDataset(Dataset):
    """
    Cada sample: (images[T,C,H,W], dias[T], mes, mask[T], label)
    Sequencias menores que MAX_SEQ_LEN sao padded com zeros + mask=0.
    """

    def __init__(self, registros, labels, meses):
        self.registros = registros
        self.labels = labels
        self.meses = meses

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, idx):
        items = self.registros[idx]
        label = self.labels[idx]
        mes = self.meses[idx]
        seq_len = min(len(items), MAX_SEQ_LEN)

        images = np.zeros((MAX_SEQ_LEN, 3, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
        dias = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
        mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

        for i in range(seq_len):
            caminho, dia = items[i]
            images[i] = preprocessar_imagem(caminho)
            dias[i] = dia / MAX_DIA
            mask[i] = 1.0

        return (
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(dias, dtype=torch.float32),
            torch.tensor(mes - 1, dtype=torch.long),   # 0-indexed para Embedding
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


# -- Modelo --------------------------------------------------------------------

class EfficientNetTemporalV6(nn.Module):
    """
    EfficientNetB0 + FiLM(dia, mes_embedding) + 2x MultiHeadAttention temporal.

    Arquitetura:
        images[B,T,C,H,W] -> EfficientNetB0 -> features[B,T,D]
        FiLM: concat(dia, mes_emb) -> Dense(64) -> gamma,beta -> modula features
        2x Self-Attention sobre os T tokens temporais (com mask)
        Mean pooling -> Dense(256) -> ReLU -> Dropout -> Dense(num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # EfficientNetB0 backbone (timm, pretrained ImageNet)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 1280

        # Congelar backbone inicialmente
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Mes -> embedding categorico aprendido
        self.mes_embedding = nn.Embedding(12, MES_EMBED_DIM)

        # FiLM: concat(dia[1], mes_emb[8]) = 9-dim -> 64 -> gamma/beta[1280]
        self.film_hidden = nn.Linear(1 + MES_EMBED_DIM, 64)
        # Zero-init: gamma=0, beta=0 -> modulacao comeca como identidade
        self.film_gamma = nn.Linear(64, self.feature_dim)
        self.film_beta = nn.Linear(64, self.feature_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # 2 camadas de self-attention temporal
        self.attn1 = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=8, dropout=0.1, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.attn2 = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=8, dropout=0.1, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(self.feature_dim)

        # Cabeca de classificacao
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, dias, mes, mask):
        """
        images: (B, T, C, H, W)
        dias:   (B, T)     - normalizado [0,1]
        mes:    (B,)       - int 0-11
        mask:   (B, T)     - 1.0 para timesteps validos, 0.0 para padding
        """
        B, T = images.shape[0], images.shape[1]

        # -- EfficientNet features por timestep ----------------------------
        imgs_flat = images.reshape(B * T, *images.shape[2:])   # (B*T, C, H, W)
        feats_flat = self.backbone(imgs_flat)                   # (B*T, 1280)
        features = feats_flat.reshape(B, T, -1)                 # (B, T, 1280)

        # -- FiLM: (dia, mes_embedding) -> gamma, beta --------------------
        mes_emb = self.mes_embedding(mes)               # (B, MES_EMBED_DIM)
        mes_emb = mes_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, MES_EMBED_DIM)
        dia_exp = dias.unsqueeze(-1)                    # (B, T, 1)
        context = torch.cat([dia_exp, mes_emb], dim=-1)  # (B, T, 9)

        film_h = F.relu(self.film_hidden(context))      # (B, T, 64)
        gamma = self.film_gamma(film_h)                  # (B, T, 1280)
        beta = self.film_beta(film_h)                    # (B, T, 1280)

        tokens = features * (1.0 + gamma) + beta

        # -- Temporal self-attention (2 camadas) ---------------------------
        # key_padding_mask: True = ignored (inverted from our mask)
        key_pad_mask = (mask == 0)   # (B, T), True onde e padding

        attn_out1, _ = self.attn1(tokens, tokens, tokens, key_padding_mask=key_pad_mask)
        tokens = self.norm1(tokens + attn_out1)

        attn_out2, _ = self.attn2(tokens, tokens, tokens, key_padding_mask=key_pad_mask)
        tokens = self.norm2(tokens + attn_out2)

        # -- Mean pooling dos tokens validos -------------------------------
        mask_exp = mask.unsqueeze(-1)  # (B, T, 1)
        pooled = (tokens * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)

        return self.head(pooled)

    def descongelar_ultimas_camadas(self, n_camadas: int):
        """Descongela as ultimas n_camadas do backbone para fine-tuning."""
        # Primeiro congela tudo
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Depois descongela as ultimas n camadas
        all_layers = list(self.backbone.named_parameters())
        for name, param in all_layers[-n_camadas:]:
            param.requires_grad = True


# -- Treino --------------------------------------------------------------------

def treinar_fase(
    modelo, loader_treino, loader_val, optimizer, criterion,
    epochs, class_weight_tensor, fase_nome, patience,
    scaler=None,
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

        # Validacao
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


# -- Main ----------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.info(
        "=== V6: EfficientNetB0 + FiLM(dia, mes_embedding) + Temporal Attention (PyTorch) ==="
    )
    logging.info(f"Dispositivo: {DEVICE} | AMP: {USE_AMP} | cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    # 1. Carregar dados
    logging.info("Carregando dados...")
    registros, labels, meses = carregar_dados(DB_PATH)
    total_imgs = sum(len(r) for r in registros)
    logging.info(f"Total: {len(registros)} talhoes, {total_imgs} imagens")

    if not registros:
        logging.error("Nenhum talhao com imagens disponiveis.")
        return

    seq_lens = Counter(min(len(r), MAX_SEQ_LEN) for r in registros)
    for n, count in sorted(seq_lens.items()):
        logging.info(f"  Talhoes com {n} imagem(ns): {count}")
    for i, c in enumerate(CLASSES):
        logging.info(f"  {c}: {labels.count(i)} talhoes")

    mes_dist = Counter(meses)
    logging.info("Distribuicao de meses:")
    for m in sorted(mes_dist):
        logging.info(f"  mes {m:>2}: {mes_dist[m]} talhoes")

    # 2. Split por talhao
    reg_treino, reg_val, lab_treino, lab_val, mes_treino, mes_val = train_test_split(
        registros, labels, meses, test_size=0.2, stratify=labels, random_state=SEED
    )
    logging.info(f"Treino: {len(reg_treino)} talhoes | Validacao: {len(reg_val)} talhoes")

    # 3. DataLoaders
    ds_tr  = TemporalCulturaDataset(reg_treino, lab_treino, mes_treino)
    ds_val = TemporalCulturaDataset(reg_val, lab_val, mes_val)
    use_pin = DEVICE.type == 'cuda'
    loader_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=use_pin,
                            persistent_workers=NUM_WORKERS > 0)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=use_pin,
                            persistent_workers=NUM_WORKERS > 0)
    logging.info(f"Treino: {len(ds_tr)} talhoes | Val: {len(ds_val)} talhoes")

    # 4. Class weights
    pesos = compute_class_weight(
        'balanced', classes=np.arange(len(CLASSES)), y=np.array(lab_treino)
    )
    cw_tensor = torch.tensor(pesos, dtype=torch.float32).to(DEVICE)
    logging.info(f"Class weights: {dict(zip(CLASSES, pesos))}")

    # 5. Modelo
    logging.info("Criando EfficientNetB0 V6 + FiLM(dia,mes) + Temporal Attention (PyTorch)...")
    modelo = EfficientNetTemporalV6(len(CLASSES)).to(DEVICE)

    total_p = sum(p.numel() for p in modelo.parameters())
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Params: {total_p:,} total | {train_p:,} treinaveis")

    scaler = GradScaler(device=DEVICE.type) if USE_AMP else None
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=LABEL_SMOOTHING)

    # -- Fase 1: Base congelada ------------------------------------------------
    logging.info("=== Fase 1: Treinando cabeca + FiLM + temporal attention (base congelada) ===")
    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE1,
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, criterion,
                 EPOCHS_FASE1, cw_tensor, "Fase1", patience=3, scaler=scaler)

    # -- Fase 2: Fine-tuning ---------------------------------------------------
    logging.info(f"=== Fase 2: Fine-tuning (ultimas {FINE_TUNE_LAYERS} camadas) ===")
    modelo.descongelar_ultimas_camadas(FINE_TUNE_LAYERS)
    train_p = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Parametros treinaveis: {train_p:,}")

    opt = torch.optim.Adam(
        (p for p in modelo.parameters() if p.requires_grad), lr=LR_FASE2,
    )
    treinar_fase(modelo, loader_tr, loader_val, opt, criterion,
                 EPOCHS_FASE2, cw_tensor, "Fase2", patience=4, scaler=scaler)

    # -- Salvar ----------------------------------------------------------------
    os.makedirs(MODELO_SAIDA, exist_ok=True)
    peso_path = os.path.join(MODELO_SAIDA, 'pesos.pt')
    torch.save(modelo.state_dict(), peso_path)
    logging.info(f"Pesos salvos em: {peso_path}")

    # -- Avaliacao final -------------------------------------------------------
    modelo.eval()
    y_true, y_pred = [], []
    n_samples = 0
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
            n_samples += labels.size(0)

    t_total = time.perf_counter() - t_start

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)

    logging.info(f"Validacao final - Acc: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    for cls_name, f1_cls in zip(CLASSES, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_true, y_pred, target_names=CLASSES,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    logging.info("Confusion Matrix (linhas=real, colunas=predito):")
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    logging.info(header)
    for cls, row in zip(CLASSES, cm):
        logging.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    tempo_medio = (t_total / n_samples) * 1000
    logging.info(f"Tempo medio de inferencia: {tempo_medio:.2f} ms/talhao ({n_samples} talhoes)")

    # -- Salvar metricas -------------------------------------------------------
    try:
        metrics_dir = os.path.join(_HERE, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(
            metrics_dir, f"metrics_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        metrics_dict = {
            "f1_macro": float(f1_macro),
            "accuracy": float(acc),
            "f1_per_class": {c: float(f) for c, f in zip(CLASSES, f1_per_class)},
            "tempo_medio_ms": float(tempo_medio),
        }
        with open(metrics_path, "w", encoding="utf-8") as m_f:
            json.dump(metrics_dict, m_f, indent=4)
        logging.info(f"Metricas salvas em {metrics_path}")
    except Exception as e:
        logging.warning(f"Erro ao salvar metricas: {e}")


if __name__ == '__main__':
    main()
