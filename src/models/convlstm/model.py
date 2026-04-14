"""
Preditor de culturas usando ConvLSTM multi-temporal.

Recebe todas as imagens de um talhão (até 3) e faz uma predição
baseada na evolução espaço-temporal.

Uso como script:
    python src/preditor_convlstm.py ./processadas/mascara_xxx_v_d21.png \
                                     ./processadas/mascara_xxx_v_d31.png \
                                     ./processadas/mascara_xxx_v_d56.png

Uso como módulo:
    from preditor_convlstm import PreditorConvLSTM
    preditor = PreditorConvLSTM()
    cultura, confianca = preditor.predizer([
        './processadas/mascara_xxx_v_d21.png',
        './processadas/mascara_xxx_v_d31.png',
        './processadas/mascara_xxx_v_d56.png',
    ])
"""

import os
import re
import sys

import numpy as np
import cv2
import torch
import torch.nn as nn

# ── Configurações (idênticas ao treinamento) ─────────────────────────────────
CLASSES     = ['milho', 'soja', 'trigo']
IMG_SIZE    = (112, 112)
PESOS_PATH  = './modelos/classificador_cultura_convlstm/pesos.pt'
MAX_SEQ_LEN = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# ── Modelo (idêntico ao treinamento) ─────────────────────────────────────────

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size=kernel_size, padding=padding, bias=True
        )

    def forward(self, x, state=None):
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
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        state = None
        for t in range(T):
            h, state = self.cell(x[:, t], state)
        return h


class ConvLSTMClassificador(nn.Module):
    def __init__(self, num_classes):
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

    def forward(self, x):
        B, T = x.shape[:2]
        state = None
        h1_seq = torch.zeros(B, T, 32, x.shape[3], x.shape[4], device=x.device)
        for t in range(T):
            h_t, state = self.convlstm1.cell(x[:, t], state)
            h1_seq[:, t] = torch.relu(self.bn1(h_t))

        h2 = self.convlstm2(h1_seq)
        h2 = torch.relu(self.bn2(h2))
        pooled = self.pool(h2).flatten(1)
        return self.classifier(pooled)


# ── Preditor ──────────────────────────────────────────────────────────────────

class PreditorConvLSTM:
    def __init__(self, pesos_path: str = PESOS_PATH):
        self._modelo = ConvLSTMClassificador(len(CLASSES))
        self._modelo.load_state_dict(torch.load(pesos_path, map_location=DEVICE, weights_only=True))
        self._modelo.to(DEVICE)
        self._modelo.eval()

    @staticmethod
    def _extrair_dia(caminho: str) -> int:
        match = re.search(r'_d(\d+)\.png$', caminho)
        if match:
            return int(match.group(1))
        raise ValueError(f"Não foi possível extrair o dia do nome: {caminho}")

    @staticmethod
    def _preprocessar(caminho: str) -> np.ndarray:
        img = cv2.imread(caminho)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada ou corrompida: {caminho}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return np.transpose(img, (2, 0, 1))  # CHW

    def predizer(self, caminhos: list[str]) -> tuple[str, float]:
        """
        Prediz a cultura de um talhão a partir de suas imagens temporais.

        Args:
            caminhos: lista de paths (1 a 3, qualquer ordem — será ordenado por dia)

        Retorna:
            (classe, confiança)
        """
        items = [(c, self._extrair_dia(c)) for c in caminhos]
        items.sort(key=lambda x: x[1])

        seq_len = min(len(items), MAX_SEQ_LEN)
        images = np.zeros((1, MAX_SEQ_LEN, 3, IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)

        for i in range(seq_len):
            caminho, _ = items[i]
            images[0, i] = self._preprocessar(caminho)

        images_t = torch.tensor(images, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = self._modelo(images_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])

    def predizer_lote(self, lista_talhoes: list[list[str]]) -> list[dict]:
        resultados = []
        for caminhos in lista_talhoes:
            try:
                cultura, confianca = self.predizer(caminhos)
                dias = sorted(self._extrair_dia(c) for c in caminhos)
                resultados.append({
                    'caminhos': caminhos, 'dias': dias,
                    'cultura': cultura, 'confianca': confianca, 'erro': None,
                })
            except Exception as e:
                resultados.append({
                    'caminhos': caminhos, 'dias': None,
                    'cultura': None, 'confianca': None, 'erro': str(e),
                })
        return resultados


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python src/preditor_convlstm.py <img_d21.png> [img_d31.png] [img_d56.png]")
        sys.exit(1)

    preditor = PreditorConvLSTM()
    caminhos = sys.argv[1:]

    cultura, confianca = preditor.predizer(caminhos)
    dias = sorted(preditor._extrair_dia(c) for c in caminhos)
    print(f"{cultura}  {confianca:.1%}  dias={dias}  ({len(caminhos)} imagens)")
