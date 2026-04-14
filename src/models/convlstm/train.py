import json
"""
Treina um classificador de culturas (milho, soja, trigo) usando
ConvLSTM2D para capturar padrões espaço-temporais diretamente da
sequência de imagens de satélite.

Diferença em relação aos modelos ViT/EfficientNet:
    Não usa backbone pré-treinado. A ConvLSTM processa a sequência
    de imagens como um "vídeo curto" de 3 frames, aprendendo padrões
    espaciais e temporais simultaneamente com convolução recorrente.

Arquitetura:
    (3, 224, 224, 3) → ConvLSTM2D(32) → BatchNorm → ConvLSTM2D(64) → BatchNorm
                     → GlobalAvgPool → Dense(128) → Dropout → Dense(3)

Uso:
    python src/treinamento/treinar_classificador_convlstm.py
"""

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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f'treino_convlstm_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)

# ── Configurações ─────────────────────────────────────────────────────────────
DB_PATH       = './sample_treino_v2.db'
TABELA        = 'culturas'
IMG_SIZE      = (112, 112)       # menor que 224 — ConvLSTM é mais pesada em memória
BATCH_SIZE    = 16
EPOCHS        = 25
LR            = 1e-3
MODELO_SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
CLASSES       = ['milho', 'soja', 'trigo']
SEED          = 42
MAX_SEQ_LEN   = 3
MAX_DIA       = 100.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# ── Dados ─────────────────────────────────────────────────────────────────────

def extrair_dia(caminho: str) -> int:
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_dados(db_path: str) -> tuple[list[list[tuple[str, int]]], list[int]]:
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
            validos.sort(key=lambda x: x[1])
            registros.append(validos)
            labels.append(classe_para_id[cultura])

    return registros, labels


def preprocessar_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img


# ── Dataset ───────────────────────────────────────────────────────────────────

class CulturaSequenciaDataset(Dataset):
    """
    Cada sample: sequência de imagens de um talhão.

    Retorna:
        images: (T, C, H, W) — sequência temporal em formato CHW
        label:  int
    """

    def __init__(self, registros, labels):
        self.registros = registros
        self.labels = labels

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, idx):
        items = self.registros[idx]
        seq_len = min(len(items), MAX_SEQ_LEN)

        # (T, H, W, C)
        images = np.zeros((MAX_SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

        for i in range(seq_len):
            caminho, dia = items[i]
            images[i] = preprocessar_imagem(caminho)

        # HWC → CHW para PyTorch: (T, C, H, W)
        images = np.transpose(images, (0, 3, 1, 2))

        return (
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── ConvLSTM Cell ─────────────────────────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    Uma célula ConvLSTM: substitui as multiplicações matriciais do LSTM
    por convoluções 2D, preservando a estrutura espacial.

    Equações (mesmas do LSTM, mas com conv2d):
        i = sigmoid(conv(x) + conv(h) + b)   — input gate
        f = sigmoid(conv(x) + conv(h) + b)   — forget gate
        g = tanh(conv(x) + conv(h) + b)      — candidate
        o = sigmoid(conv(x) + conv(h) + b)   — output gate
        c = f * c_prev + i * g
        h = o * tanh(c)
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Uma conv para os 4 gates (i, f, g, o) juntos
        self.gates = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=padding, bias=True
        )

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Args:
            x: (B, C_in, H, W) — input de um timestep
            state: (h, c) cada (B, hidden, H, W), ou None para inicializar com zeros

        Returns:
            h: (B, hidden, H, W)
            (h, c): novo estado
        """
        B, _, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, (h, c)


class ConvLSTM(nn.Module):
    """Empilha ConvLSTMCells e retorna o hidden state final."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            h_last: (B, hidden, H, W) — hidden state do último timestep
        """
        B, T, C, H, W = x.shape
        state = None

        for t in range(T):
            h, state = self.cell(x[:, t], state)

        return h


# ── Modelo ────────────────────────────────────────────────────────────────────

class ConvLSTMClassificador(nn.Module):
    """
    Classificador multi-temporal usando ConvLSTM.

    Fluxo:
        (B, 3, 3, 112, 112)
          → ConvLSTM(3→32)  → BatchNorm → ReLU
          → ConvLSTM(32→64) → BatchNorm → ReLU
          → AdaptiveAvgPool(1x1) → Flatten
          → Dense(128) → ReLU → Dropout(0.3) → Dense(3)

    A ConvLSTM vê a sequência temporal como um vídeo curto.
    Cada "frame" é processado por convoluções recorrentes que
    mantêm a estrutura espacial — o modelo pode aprender que
    uma região específica do campo mudou de verde para amarelo.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.convlstm1 = ConvLSTM(in_channels=3, hidden_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.convlstm2 = ConvLSTM(in_channels=32, hidden_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) — sequência de imagens
        """
        # Camada 1: (B, T, 3, H, W) → (B, 32, H, W)
        h1 = self.convlstm1(x)
        h1 = torch.relu(self.bn1(h1))

        # Preparar para camada 2: precisa de (B, T, 32, H, W)
        # Alimenta o output espacial de cada timestep na segunda ConvLSTM
        B, T = x.shape[:2]
        h1_seq = torch.zeros(B, T, 32, h1.shape[2], h1.shape[3], device=x.device)

        # Re-processar cada timestep pela camada 1 para obter todos os hidden states
        state = None
        for t in range(T):
            h_t, state = self.convlstm1.cell(x[:, t], state)
            h1_seq[:, t] = torch.relu(self.bn1(h_t))

        # Camada 2: (B, T, 32, H, W) → (B, 64, H, W)
        h2 = self.convlstm2(h1_seq)
        h2 = torch.relu(self.bn2(h2))

        # Pool → classificação
        pooled = self.pool(h2).flatten(1)  # (B, 64)
        return self.classifier(pooled)


# ── Treino ────────────────────────────────────────────────────────────────────

def treinar(modelo, loader_treino, loader_val, optimizer, criterion,
            epochs, class_weight_tensor, patience):
    best_val_loss = float('inf')
    best_state = None
    epochs_sem_melhora = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    for epoch in range(epochs):
        modelo.train()
        total_loss = 0
        corretos = 0
        total = 0

        for images, labels in loader_treino:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = modelo(images)

            pesos_amostra = class_weight_tensor[labels]
            loss = (criterion(logits, labels) * pesos_amostra).mean()

            loss.backward()
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
            for images, labels in loader_val:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = modelo(images)
                loss = criterion(logits, labels).mean()
                val_loss_total += loss.item() * images.size(0)
                val_corretos += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_corretos / val_total
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(
            f"Epoch {epoch+1}/{epochs} — "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
            f"lr: {current_lr:.1e}"
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
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = modelo(images)
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

    total_params = sum(p.numel() for p in modelo.parameters())
    size_mb = total_params * 4 / (1024 * 1024)
    logging.info(f"Parâmetros: {total_params:,} ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logging.info("Modelo ConvLSTM: Classificação espaço-temporal por sequência")

    # 1. Carregar dados
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

    # 2. Split por talhão
    reg_treino, reg_val, lab_treino, lab_val = train_test_split(
        registros, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    logging.info(f"Treino: {len(reg_treino)} talhões | Validação: {len(reg_val)} talhões")

    # 3. DataLoaders
    ds_treino = CulturaSequenciaDataset(reg_treino, lab_treino)
    ds_val = CulturaSequenciaDataset(reg_val, lab_val)
    loader_treino = DataLoader(ds_treino, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 4. Class weights
    pesos = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=np.array(lab_treino))
    class_weight_tensor = torch.tensor(pesos, dtype=torch.float32).to(DEVICE)
    logging.info(f"Class weights: {dict(zip(CLASSES, pesos))}")

    # 5. Modelo
    logging.info("Criando ConvLSTM...")
    modelo = ConvLSTMClassificador(len(CLASSES)).to(DEVICE)

    total_params = sum(p.numel() for p in modelo.parameters())
    logging.info(f"Parâmetros: {total_params:,}")

    # 6. Treinar
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(modelo.parameters(), lr=LR)

    treinar(modelo, loader_treino, loader_val, optimizer, criterion,
            EPOCHS, class_weight_tensor, patience=5)

    # 7. Salvar
    os.makedirs(MODELO_SAIDA, exist_ok=True)
    peso_path = os.path.join(MODELO_SAIDA, 'pesos.pt')
    torch.save(modelo.state_dict(), peso_path)
    logging.info(f"Pesos salvos em: {peso_path}")

    # 8. Avaliação final
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
