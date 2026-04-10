"""
Carrega o modelo ViT treinado e prediz a cultura de imagens processadas.

Uso como script:
    python src/preditor_vit.py ./processadas/mascara_xxx.png
    python src/preditor_vit.py ./processadas/mascara_a.png ./processadas/mascara_b.png

Uso como módulo:
    from preditor_vit import PreditorViT
    preditor = PreditorViT()
    cultura, confianca = preditor.predizer('./processadas/mascara_xxx.png')
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from transformers import TFViTModel

# ── Configurações (devem ser idênticas ao treinamento) ────────────────────────
CLASSES        = ['milho', 'soja', 'trigo']
IMG_SIZE       = (224, 224)
PESOS_PATH     = './modelos/classificador_cultura_vit/pesos.weights.h5'
VIT_PRETRAINED = 'google/vit-base-patch16-224-in21k'


# ── Arquitetura (idêntica à usada no treinamento) ─────────────────────────────
class ViTClassificador(tf.keras.Model):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained(VIT_PRETRAINED, use_safetensors=False)
        self.vit.trainable = False
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])  # NHWC -> NCHW
        outputs = self.vit(pixel_values=inputs, training=training)
        cls_output = outputs.last_hidden_state[:, 0, :]   # token [CLS]
        x = self.dropout(cls_output, training=training)
        x = self.dense(x)
        return self.classifier(x)


# ── Preditor ──────────────────────────────────────────────────────────────────
class PreditorViT:
    def __init__(self, pesos_path: str = PESOS_PATH):
        self._modelo = ViTClassificador(len(CLASSES))
        # Constrói o grafo antes de carregar os pesos
        self._modelo(tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3)))
        self._modelo.load_weights(pesos_path)

    def _preprocessar(self, caminho: str) -> np.ndarray:
        img = cv2.imread(caminho)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada ou corrompida: {caminho}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 127.5 - 1.0   # normalização ViT: [-1, 1]
        return img

    def predizer(self, caminho: str) -> tuple[str, float]:
        """
        Prediz a cultura de uma única imagem.

        Retorna:
            (classe, confiança) — ex.: ('soja', 0.91)
        """
        img = self._preprocessar(caminho)
        batch = np.expand_dims(img, axis=0)          # (1, H, W, C)
        probs = self._modelo(batch, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])

    def predizer_lote(self, caminhos: list[str]) -> list[dict]:
        """
        Prediz a cultura de uma lista de imagens.

        Retorna lista de dicts com chaves: caminho, cultura, confianca, erro.
        """
        resultados = []
        for caminho in caminhos:
            try:
                cultura, confianca = self.predizer(caminho)
                resultados.append({
                    'caminho':   caminho,
                    'cultura':   cultura,
                    'confianca': confianca,
                    'erro':      None,
                })
            except Exception as e:
                resultados.append({
                    'caminho':   caminho,
                    'cultura':   None,
                    'confianca': None,
                    'erro':      str(e),
                })
        return resultados


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python src/preditor_vit.py <imagem1.png> [imagem2.png ...]")
        sys.exit(1)

    preditor = PreditorViT()

    for r in preditor.predizer_lote(sys.argv[1:]):
        if r['erro']:
            print(f"ERRO    {r['caminho']}: {r['erro']}")
        else:
            print(f"{r['cultura']:<6}  {r['confianca']:>6.1%}  {r['caminho']}")
