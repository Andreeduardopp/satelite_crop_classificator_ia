"""
Preditor de culturas usando ViT-Small/16 + Temporal Transformer (v3).

Recebe TODAS as imagens de um talhão e faz uma única predição
combinando a evolução temporal da cultura.

Uso como script:
    # Passar todas as imagens de um talhão (ordem não importa, será ordenado por dia)
    python src/preditor_vit_v3.py ./processadas/mascara_7e86fe1d_v_d21.png \
                                   ./processadas/mascara_7e86fe1d_v_d31.png \
                                   ./processadas/mascara_7e86fe1d_v_d56.png

Uso como módulo:
    from preditor_vit_v3 import PreditorViTV3
    preditor = PreditorViTV3()

    # Predizer um talhão (lista de caminhos)
    cultura, confianca = preditor.predizer([
        './processadas/mascara_xxx_v_d21.png',
        './processadas/mascara_xxx_v_d31.png',
        './processadas/mascara_xxx_v_d56.png',
    ])

    # Também funciona com 1 ou 2 imagens (mas 3 é melhor)
    cultura, confianca = preditor.predizer(['./processadas/mascara_xxx_v_d21.png'])
"""

import os
import re
import sys

import numpy as np
import cv2
import torch
import torch.nn as nn
import timm

# ── Configurações (idênticas ao treinamento) ─────────────────────────────────
CLASSES        = ['milho', 'soja', 'trigo']
IMG_SIZE       = (224, 224)
PESOS_PATH     = './modelos/classificador_cultura_vit_v3/pesos.pt'
VIT_MODEL_NAME = 'vit_small_patch16_224'
MAX_DIA        = 100.0
MAX_SEQ_LEN    = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


# ── Modelo (idêntico ao treinamento) ─────────────────────────────────────────

class ViTSequencialClassificador(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = timm.create_model(VIT_MODEL_NAME, pretrained=False, num_classes=0)
        self.vit_dim = self.vit.num_features

        self.dia_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.vit_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.vit_dim, nhead=6, dim_feedforward=512,
            dropout=0.1, batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Sequential(
            nn.Linear(self.vit_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, dias, mask):
        B, T = images.shape[:2]
        images_flat = images.view(B * T, 3, IMG_SIZE[0], IMG_SIZE[1])
        vit_features = self.vit(images_flat).view(B, T, -1)
        dia_features = self.dia_embed(dias.unsqueeze(-1))
        tokens = vit_features + dia_features
        tokens = self.temporal_transformer(tokens, src_key_padding_mask=~mask)
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (tokens * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.head(pooled)


# ── Preditor ──────────────────────────────────────────────────────────────────

class PreditorViTV3:
    def __init__(self, pesos_path: str = PESOS_PATH):
        self._modelo = ViTSequencialClassificador(len(CLASSES))
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
        return np.transpose(img, (2, 0, 1))

    def predizer(self, caminhos: list[str]) -> tuple[str, float]:
        """
        Prediz a cultura de um talhão a partir de suas imagens temporais.

        Args:
            caminhos: lista de paths de imagens do MESMO talhão
                      (1 a 3 imagens, em qualquer ordem)

        Retorna:
            (classe, confiança) — ex.: ('soja', 0.87)
        """
        # Extrair dias e ordenar por dia
        items = [(c, self._extrair_dia(c)) for c in caminhos]
        items.sort(key=lambda x: x[1])

        seq_len = min(len(items), MAX_SEQ_LEN)

        images = torch.zeros(1, MAX_SEQ_LEN, 3, IMG_SIZE[0], IMG_SIZE[1])
        dias = torch.zeros(1, MAX_SEQ_LEN)
        mask = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.bool)

        for i in range(seq_len):
            caminho, dia = items[i]
            images[0, i] = torch.tensor(self._preprocessar(caminho))
            dias[0, i] = dia / MAX_DIA
            mask[0, i] = True

        images = images.to(DEVICE)
        dias = dias.to(DEVICE)
        mask = mask.to(DEVICE)

        with torch.no_grad():
            logits = self._modelo(images, dias, mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])

    def predizer_lote(self, lista_talhoes: list[list[str]]) -> list[dict]:
        """
        Prediz a cultura de uma lista de talhões.

        Args:
            lista_talhoes: lista de listas de caminhos
                           ex: [['d21.png', 'd31.png', 'd56.png'], ['d21.png'], ...]

        Retorna lista de dicts com: caminhos, dias, cultura, confianca, erro.
        """
        resultados = []
        for caminhos in lista_talhoes:
            try:
                cultura, confianca = self.predizer(caminhos)
                dias = sorted(self._extrair_dia(c) for c in caminhos)
                resultados.append({
                    'caminhos':  caminhos,
                    'dias':      dias,
                    'cultura':   cultura,
                    'confianca': confianca,
                    'erro':      None,
                })
            except Exception as e:
                resultados.append({
                    'caminhos':  caminhos,
                    'dias':      None,
                    'cultura':   None,
                    'confianca': None,
                    'erro':      str(e),
                })
        return resultados


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python src/preditor_vit_v3.py <img_d21.png> [img_d31.png] [img_d56.png]")
        print("Passar todas as imagens de UM talhão como argumentos.")
        sys.exit(1)

    preditor = PreditorViTV3()
    caminhos = sys.argv[1:]

    cultura, confianca = preditor.predizer(caminhos)
    dias = sorted(preditor._extrair_dia(c) for c in caminhos)
    print(f"{cultura}  {confianca:.1%}  dias={dias}  ({len(caminhos)} imagens)")
