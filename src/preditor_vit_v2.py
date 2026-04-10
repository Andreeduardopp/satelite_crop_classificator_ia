"""
Preditor de culturas usando ViT-Small/16 (timm, PyTorch) com info multi-temporal.

Extrai automaticamente o dia após plantio do nome do arquivo (e.g. _d56.png → 56)
e o alimenta ao modelo junto com a imagem.

Uso como script:
    python src/preditor_vit_v2.py ./processadas/mascara_xxx_v_d21.png
    python src/preditor_vit_v2.py ./processadas/mascara_a.png ./processadas/mascara_b.png

Uso como módulo:
    from preditor_vit_v2 import PreditorViTV2
    preditor = PreditorViTV2()
    cultura, confianca = preditor.predizer('./processadas/mascara_xxx_v_d56.png')
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
PESOS_PATH     = './modelos/classificador_cultura_vit_v2/pesos_all.pt'
VIT_MODEL_NAME = 'vit_small_patch16_224'
MAX_DIA        = 100.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Modelo (idêntico ao treinamento) ─────────────────────────────────────────

class ViTTemporalClassificador(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = timm.create_model(VIT_MODEL_NAME, pretrained=False, num_classes=0)
        self.vit_dim = self.vit.num_features

        self.dia_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(self.vit_dim + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASSES)),
        )

    def forward(self, img: torch.Tensor, dia: torch.Tensor) -> torch.Tensor:
        features = self.vit(img)
        dia_feat = self.dia_embed(dia.unsqueeze(-1))
        combined = torch.cat([features, dia_feat], dim=1)
        return self.head(combined)


# ── Preditor ──────────────────────────────────────────────────────────────────

class PreditorViTV2:
    def __init__(self, pesos_path: str = PESOS_PATH):
        self._modelo = ViTTemporalClassificador(len(CLASSES))
        self._modelo.load_state_dict(torch.load(pesos_path, map_location=DEVICE, weights_only=True))
        self._modelo.to(DEVICE)
        self._modelo.eval()

    @staticmethod
    def _extrair_dia(caminho: str) -> int:
        """Extrai o dia após plantio do nome do arquivo."""
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

        # Normalização ImageNet (CHW float32)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return img

    def predizer(self, caminho: str, dia: int | None = None) -> tuple[str, float]:
        """
        Prediz a cultura de uma imagem.

        Args:
            caminho: path da imagem processada
            dia: dia após plantio (se None, extrai do nome do arquivo)

        Retorna:
            (classe, confiança) — ex.: ('soja', 0.91)
        """
        if dia is None:
            dia = self._extrair_dia(caminho)

        img = self._preprocessar(caminho)
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dia_t = torch.tensor(dia / MAX_DIA, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self._modelo(img_t, dia_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])

    def predizer_lote(self, caminhos: list[str]) -> list[dict]:
        """
        Prediz a cultura de uma lista de imagens.

        Retorna lista de dicts com: caminho, dia, cultura, confianca, erro.
        """
        resultados = []
        for caminho in caminhos:
            try:
                dia = self._extrair_dia(caminho)
                cultura, confianca = self.predizer(caminho, dia=dia)
                resultados.append({
                    'caminho':   caminho,
                    'dia':       dia,
                    'cultura':   cultura,
                    'confianca': confianca,
                    'erro':      None,
                })
            except Exception as e:
                resultados.append({
                    'caminho':   caminho,
                    'dia':       None,
                    'cultura':   None,
                    'confianca': None,
                    'erro':      str(e),
                })
        return resultados


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python src/preditor_vit_v2.py <imagem1.png> [imagem2.png ...]")
        sys.exit(1)

    preditor = PreditorViTV2()

    for r in preditor.predizer_lote(sys.argv[1:]):
        if r['erro']:
            print(f"ERRO    {r['caminho']}: {r['erro']}")
        else:
            print(f"{r['cultura']:<6}  {r['confianca']:>6.1%}  d{r['dia']:<3}  {r['caminho']}")
