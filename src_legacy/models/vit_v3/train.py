import json
"""
Treina um classificador de culturas (milho, soja, trigo) usando
ViT-Small/16 (timm, PyTorch) com SEQUÊNCIA multi-temporal.

Diferença em relação à v2:
    A v2 classifica cada imagem independentemente (1 imagem → 1 predição).
    A v3 classifica o TALHÃO inteiro combinando todas as imagens temporais
    de um campo (até 3 datas) em uma única predição.

    O modelo aprende como a cultura EVOLUI ao longo do tempo:
        [img_d21, img_d31, img_d56] → ViT (compartilhado) → features por data
            → Temporal Transformer → predição única do talhão

Arquitetura:
    1. Cada imagem → vit_small (pesos compartilhados) → 384-dim features
    2. Adiciona embedding temporal baseado no dia normalizado
    3. Sequência de features → Temporal Transformer Encoder (2 camadas)
    4. Mean pooling dos tokens válidos → Dense → classificação

    Campos com menos de 3 imagens são padded com zeros + attention mask.

Fases:
    1. Base congelada — treina cabeça + temporal transformer
    2. Fine-tuning  — descongela últimos N blocos do ViT

Uso:
    python src/treinamento/treinar_classificador_vit_v3.py
"""

import os
import re
import ast
import time
import logging
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
from collections import Counter

import timm

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f'treino_vit_v3_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)

# ── Configurações ─────────────────────────────────────────────────────────────
DB_PATH          = './sample_treino_v2.db'
TABELA           = 'culturas'
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 16           # menor que v2 pois cada sample tem até 3 imagens
EPOCHS_FASE1     = 10
EPOCHS_FASE2     = 15
LR_FASE1         = 1e-4
LR_FASE2         = 1e-5
GRAD_CLIP        = 1.0
MODELO_SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
CLASSES          = ['milho', 'soja', 'trigo']
SEED             = 42
VIT_MODEL_NAME   = 'vit_small_patch16_224'
FINE_TUNE_BLOCOS = 2
MAX_DIA          = 100.0
MAX_SEQ_LEN      = 3            # máximo de imagens temporais por talhão

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalização ImageNet
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# ── Dados ─────────────────────────────────────────────────────────────────────

def extrair_dia(caminho: str) -> int:
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_dados(db_path: str) -> tuple[list[list[tuple[str, int]]], list[int]]:
    """
    Retorna (registros, labels) onde cada registro é uma lista de
    (caminho, dia) ORDENADOS por dia, pertencentes ao mesmo talhão.
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
            if os.path.exists(p):
                validos.append((p, extrair_dia(p)))

        if validos:
            validos.sort(key=lambda x: x[1])  # ordenar por dia
            registros.append(validos)
            labels.append(classe_para_id[cultura])

    return registros, labels


def preprocessar_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return img


# ── Dataset PyTorch ───────────────────────────────────────────────────────────

class CulturaTemporalDataset(Dataset):
    """
    Cada sample é um talhão com até MAX_SEQ_LEN imagens temporais.

    Retorna:
        images:  (MAX_SEQ_LEN, 3, H, W) — padded com zeros
        dias:    (MAX_SEQ_LEN,)          — dias normalizados, 0 para padding
        mask:    (MAX_SEQ_LEN,)          — True para posições válidas
        label:   int
    """

    def __init__(self, registros: list[list[tuple[str, int]]], labels: list[int]):
        self.registros = registros
        self.labels = labels

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, idx):
        items = self.registros[idx]
        seq_len = min(len(items), MAX_SEQ_LEN)

        images = torch.zeros(MAX_SEQ_LEN, 3, IMG_SIZE[0], IMG_SIZE[1])
        dias = torch.zeros(MAX_SEQ_LEN)
        mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)

        for i in range(seq_len):
            caminho, dia = items[i]
            images[i] = torch.tensor(preprocessar_imagem(caminho))
            dias[i] = dia / MAX_DIA
            mask[i] = True

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return images, dias, mask, label


# ── Modelo ────────────────────────────────────────────────────────────────────

class ViTSequencialClassificador(nn.Module):
    """
    ViT-Small/16 + Temporal Transformer para classificação multi-temporal.

    Fluxo:
        Para cada timestep t no talhão:
            img_t → vit_small (compartilhado) → 384-dim
            dia_t → Linear → 384-dim
            token_t = vit_features + dia_embedding

        [token_1, token_2, token_3] → TransformerEncoder (2 camadas)
                                    → mean pool (tokens válidos)
                                    → Dense(256) → Dense(3)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # ViT backbone (compartilhado entre timesteps)
        self.vit = timm.create_model(VIT_MODEL_NAME, pretrained=True, num_classes=0)
        self.vit_dim = self.vit.num_features  # 384

        for param in self.vit.parameters():
            param.requires_grad = False

        # Embedding temporal: dia → 384-dim (mesmo que ViT features)
        self.dia_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.vit_dim),
        )

        # Temporal Transformer: combina features de diferentes datas
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.vit_dim,
            nhead=6,          # 384 / 6 = 64 dim por head
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Cabeça de classificação
        self.head = nn.Sequential(
            nn.Linear(self.vit_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor, dias: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, T, 3, H, W)
            dias:   (B, T)
            mask:   (B, T) — True para posições válidas
        """
        B, T = images.shape[:2]

        # Extrair features de cada imagem com ViT (compartilhado)
        images_flat = images.view(B * T, 3, IMG_SIZE[0], IMG_SIZE[1])
        vit_features = self.vit(images_flat)        # (B*T, 384)
        vit_features = vit_features.view(B, T, -1)  # (B, T, 384)

        # Embedding temporal
        dia_features = self.dia_embed(dias.unsqueeze(-1))  # (B, T, 384)

        # Combinar: features visuais + info temporal
        tokens = vit_features + dia_features  # (B, T, 384)

        # Temporal Transformer
        # Nota: como sample_treino_v2.db filtra para exatamente 3 imagens por talhão,
        # todos os tokens são válidos. Não passamos src_key_padding_mask para evitar
        # o bug de nested tensor do TransformerEncoder no PyTorch.
        tokens = self.temporal_transformer(tokens)

        # Mean pooling dos tokens válidos
        mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)
        pooled = (tokens * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return self.head(pooled)

    def descongelar_ultimos_blocos(self, n_blocos: int):
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
        modelo.train()
        total_loss = 0
        corretos = 0
        total = 0

        for images, dias, mask, labels in loader_treino:
            images = images.to(DEVICE)
            dias = dias.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = modelo(images, dias, mask)

            pesos_amostra = class_weight_tensor[labels]
            loss = (criterion(logits, labels) * pesos_amostra).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, modelo.parameters()),
                max_norm=GRAD_CLIP,
            )
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            corretos += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss = total_loss / total
        train_acc = corretos / total

        modelo.eval()
        val_loss_total = 0
        val_corretos = 0
        val_total = 0

        with torch.no_grad():
            for images, dias, mask, labels in loader_val:
                images = images.to(DEVICE)
                dias = dias.to(DEVICE)
                mask = mask.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = modelo(images, dias, mask)
                loss = criterion(logits, labels).mean()
                val_loss_total += loss.item() * images.size(0)
                val_corretos += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_corretos / val_total

        logging.info(
            f"[{fase_nome}] Epoch {epoch+1}/{epochs} — "
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
                logging.info(f"Early stopping após {patience} épocas sem melhora.")
                break

    if best_state:
        modelo.load_state_dict(best_state)


def avaliar(modelo, loader, class_names):
    modelo.eval()
    y_true = []
    y_pred = []
    n_samples = 0
    t_start = time.perf_counter()

    with torch.no_grad():
        for images, dias, mask, labels in loader:
            images = images.to(DEVICE)
            dias = dias.to(DEVICE)
            mask = mask.to(DEVICE)

            logits = modelo(images, dias, mask)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
            n_samples += images.size(0)

    t_total = time.perf_counter() - t_start

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)

    logging.info(f"Validação final — Accuracy: {acc:.4f}")
    logging.info(f"F1-score (macro): {f1_macro:.4f}")
    for cls_name, f1_cls in zip(class_names, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_true, y_pred, target_names=class_names
    ))

    tempo_medio = (t_total / n_samples) * 1000
    logging.info(f"Tempo médio de inferência: {tempo_medio:.2f} ms/talhão ({n_samples} talhões)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.info("Modelo V3: Classificação multi-temporal por sequência")

    # 1. Carregar dados (agrupados por talhão, ordenados por dia)
    logging.info("Carregando dados...")
    registros, labels = carregar_dados(DB_PATH)
    total_imgs = sum(len(r) for r in registros)
    logging.info(f"Total: {len(registros)} talhões, {total_imgs} imagens")

    seq_lens = Counter(len(r) for r in registros)
    for n, count in sorted(seq_lens.items()):
        logging.info(f"  Talhões com {n} imagem(ns): {count}")

    for i, c in enumerate(CLASSES):
        n = labels.count(i)
        logging.info(f"  {c}: {n} talhões")

    # 2. Split por talhão (evita data leakage)
    reg_treino, reg_val, lab_treino, lab_val = train_test_split(
        registros, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    logging.info(f"Treino: {len(reg_treino)} talhões | Validação: {len(reg_val)} talhões")

    # 3. DataLoaders
    ds_treino = CulturaTemporalDataset(reg_treino, lab_treino)
    ds_val = CulturaTemporalDataset(reg_val, lab_val)
    loader_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 4. Class weights
    pesos = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=np.array(lab_treino))
    class_weight_tensor = torch.tensor(pesos, dtype=torch.float32).to(DEVICE)
    logging.info(f"Class weights: {dict(zip(CLASSES, pesos))}")

    # 5. Modelo
    logging.info(f"Carregando {VIT_MODEL_NAME} + Temporal Transformer...")
    modelo = ViTSequencialClassificador(len(CLASSES)).to(DEVICE)

    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"Parâmetros: {total_params:,} total | {trainable_params:,} treináveis")

    # ── Fase 1: ViT congelado ─────────────────────────────────────────────
    logging.info("=== Fase 1: Treinando cabeça + temporal transformer (ViT congelado) ===")
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()), lr=LR_FASE1
    )
    treinar_fase(modelo, loader_treino, loader_val, optimizer, criterion,
                 EPOCHS_FASE1, class_weight_tensor, "Fase 1", patience=3)

    # ── Fase 2: Fine-tuning ──────────────────────────────────────────────
    logging.info(f"=== Fase 2: Fine-tuning (últimos {FINE_TUNE_BLOCOS} blocos ViT) ===")
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
    peso_path = os.path.join(MODELO_SAIDA, 'pesos.pt')
    torch.save(modelo.state_dict(), peso_path)
    logging.info(f"Pesos salvos em: {peso_path}")

    # ── Avaliação final ──────────────────────────────────────────────────
    avaliar(modelo, loader_val, CLASSES)



    # Salvar metricas
    try:
        metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, f"metrics_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        metrics_dict = {
            "f1_macro": float(f1_macro),
            "f1_per_class": {c: float(f) for c, f in zip(CLASSES, f1_per_class)},
            "tempo_medio_ms": float(tempo_medio)
        }
        with open(metrics_path, "w", encoding="utf-8") as m_f:
            json.dump(metrics_dict, m_f, indent=4)
        logging.info(f"Metricas salvas em {metrics_path}")
    except Exception as e:
        print("Erro ao salvar metricas:", e)

if __name__ == '__main__':
    main()
