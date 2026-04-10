"""
Treina um classificador de culturas (milho, soja, trigo) usando
EfficientNetB0 com SEQUÊNCIA multi-temporal (TensorFlow/Keras).

Diferença em relação à v1:
    A v1 classifica cada imagem independentemente.
    A v2 classifica o TALHÃO inteiro combinando todas as imagens temporais
    de um campo (até 3 datas) em uma única predição.

Arquitetura:
    1. Cada imagem → EfficientNetB0 (compartilhado) → GlobalAvgPool → 1280-dim
    2. Dia normalizado → Dense → 1280-dim embedding temporal
    3. Soma features visuais + temporal por timestep
    4. Sequência → MultiHeadAttention (2 camadas) → mean pool → Dense → classificação

Fases:
    1. Base congelada — treina cabeça + temporal attention
    2. Fine-tuning  — descongela as últimas 20 camadas do EfficientNet

Uso:
    python src/treinamento/treinar_classificador_v2.py
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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f'treino_efficientnet_v2_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

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
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
EPOCHS_FASE1  = 10
EPOCHS_FASE2  = 15
LR_FASE1      = 1e-3
LR_FASE2      = 1e-5
MODELO_SAIDA  = './modelos/classificador_cultura_efficientnet_v2'
CLASSES       = ['milho', 'soja', 'trigo']
SEED          = 42
MAX_SEQ_LEN   = 3
MAX_DIA       = 100.0
FEATURE_DIM   = 1280    # EfficientNetB0 output dim


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
            validos.sort(key=lambda x: x[1])
            registros.append(validos)
            labels.append(classe_para_id[cultura])

    return registros, labels


def preprocessar_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


# ── Dataset ───────────────────────────────────────────────────────────────────

def criar_dataset(registros: list[list[tuple[str, int]]], labels: list[int],
                  treino: bool) -> tf.data.Dataset:
    """
    Cria um tf.data.Dataset onde cada sample é um talhão com até MAX_SEQ_LEN
    imagens temporais, dias normalizados, e uma mask de posições válidas.
    """

    def gerador():
        for items, label in zip(registros, labels):
            seq_len = min(len(items), MAX_SEQ_LEN)

            images = np.zeros((MAX_SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            dias = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
            mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

            for i in range(seq_len):
                caminho, dia = items[i]
                images[i] = preprocessar_imagem(caminho)
                dias[i] = dia / MAX_DIA
                mask[i] = 1.0

            yield (images, dias, mask), label

    ds = tf.data.Dataset.from_generator(
        gerador,
        output_signature=(
            (
                tf.TensorSpec(shape=(MAX_SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.float32),
                tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if treino:
        ds = ds.shuffle(1024, seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Modelo ────────────────────────────────────────────────────────────────────

class EfficientNetTemporalModel(tf.keras.Model):
    """
    EfficientNetB0 + Temporal Attention para classificação multi-temporal.

    Fluxo:
        Para cada timestep t:
            img_t → EfficientNetB0 → GlobalAvgPool → 1280-dim
            dia_t → Dense(64) → ReLU → Dense(1280)
            token_t = features + dia_embedding

        [token_1, token_2, token_3] → MultiHeadAttention × 2
                                    → mean pool (mascarado)
                                    → Dense(256) → Dense(3)
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)

        # EfficientNetB0 backbone (compartilhado entre timesteps)
        self.backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        )
        self.backbone.trainable = False
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        # Embedding temporal
        self.dia_dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dia_dense2 = tf.keras.layers.Dense(FEATURE_DIM)

        # Temporal attention (2 camadas)
        self.attn1 = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=FEATURE_DIM // 8, dropout=0.1
        )
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn2 = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=FEATURE_DIM // 8, dropout=0.1
        )
        self.norm2 = tf.keras.layers.LayerNormalization()

        # Cabeça de classificação
        self.head_dense = tf.keras.layers.Dense(256, activation='relu')
        self.head_dropout = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        images, dias, mask = inputs  # (B, T, H, W, 3), (B, T), (B, T)

        B = tf.shape(images)[0]
        T = MAX_SEQ_LEN

        # Extrair features de cada imagem (backbone compartilhado)
        images_flat = tf.reshape(images, (-1, IMG_SIZE[0], IMG_SIZE[1], 3))
        features_flat = self.backbone(images_flat, training=training)
        features_flat = self.pool(features_flat)          # (B*T, 1280)
        features = tf.reshape(features_flat, (B, T, -1))  # (B, T, 1280)

        # Embedding temporal
        dia_embed = self.dia_dense1(tf.expand_dims(dias, -1))  # (B, T, 64)
        dia_embed = self.dia_dense2(dia_embed)                  # (B, T, 1280)

        # Combinar
        tokens = features + dia_embed  # (B, T, 1280)

        # Attention mask: (B, T) → (B, 1, 1, T) para Keras MHA
        # 0.0 no mask = posição de padding → large negative attention logit
        attn_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
        # Converter para aditivo: 0 → -1e9, 1 → 0
        attn_bias = (1.0 - attn_mask) * -1e9

        # Camada 1
        attn_out1 = self.attn1(tokens, tokens, attention_mask=attn_bias, training=training)
        tokens = self.norm1(tokens + attn_out1)

        # Camada 2
        attn_out2 = self.attn2(tokens, tokens, attention_mask=attn_bias, training=training)
        tokens = self.norm2(tokens + attn_out2)

        # Mean pooling dos tokens válidos
        mask_expanded = tf.expand_dims(mask, -1)  # (B, T, 1)
        pooled = tf.reduce_sum(tokens * mask_expanded, axis=1) / tf.maximum(
            tf.reduce_sum(mask_expanded, axis=1), 1.0
        )

        # Classificação
        x = self.head_dense(pooled)
        x = self.head_dropout(x, training=training)
        return self.classifier(x)

    def descongelar_ultimas_camadas(self, n_camadas: int = 20):
        self.backbone.trainable = True
        for layer in self.backbone.layers[:-n_camadas]:
            layer.trainable = False


# ── Treino ────────────────────────────────────────────────────────────────────

def main() -> None:
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    logging.info("Modelo EfficientNetB0 V2: Classificação multi-temporal por sequência")

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

    # 2. Split por talhão (evita data leakage)
    reg_treino, reg_val, lab_treino, lab_val = train_test_split(
        registros, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    logging.info(f"Treino: {len(reg_treino)} talhões | Validação: {len(reg_val)} talhões")

    # 3. Datasets
    ds_treino = criar_dataset(reg_treino, lab_treino, treino=True)
    ds_val = criar_dataset(reg_val, lab_val, treino=False)

    # 4. Class weights
    pesos = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=np.array(lab_treino))
    class_weight = {i: w for i, w in enumerate(pesos)}
    logging.info(f"Class weights: {class_weight}")

    # 5. Modelo
    logging.info("Criando EfficientNetB0 + Temporal Attention...")
    modelo = EfficientNetTemporalModel(len(CLASSES))

    # Build
    for batch in ds_treino.take(1):
        modelo(batch[0], training=False)
    modelo.summary(print_fn=logging.info)

    # ── Fase 1: Base congelada ────────────────────────────────────────────
    logging.info("=== Fase 1: Treinando cabeça + temporal attention (base congelada) ===")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FASE1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks_fase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3, restore_best_weights=True
        ),
    ]

    modelo.fit(
        ds_treino,
        validation_data=ds_val,
        epochs=EPOCHS_FASE1,
        class_weight=class_weight,
        callbacks=callbacks_fase1,
    )

    # ── Fase 2: Fine-tuning ──────────────────────────────────────────────
    logging.info("=== Fase 2: Fine-tuning (últimas 20 camadas EfficientNet) ===")
    modelo.descongelar_ultimas_camadas(20)

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FASE2),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks_fase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7
        ),
    ]

    modelo.fit(
        ds_treino,
        validation_data=ds_val,
        epochs=EPOCHS_FASE2,
        class_weight=class_weight,
        callbacks=callbacks_fase2,
    )

    # ── Salvar ────────────────────────────────────────────────────────────
    os.makedirs(MODELO_SAIDA, exist_ok=True)
    peso_path = os.path.join(MODELO_SAIDA, 'pesos.weights.h5')
    modelo.save_weights(peso_path)
    logging.info(f"Pesos salvos em: {peso_path}")

    # ── Avaliação final ──────────────────────────────────────────────────
    loss, acc = modelo.evaluate(ds_val)
    logging.info(f"Validação final — Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    y_true = []
    y_pred = []
    n_samples = 0
    t_start = time.perf_counter()

    for batch_inputs, batch_labels in ds_val:
        preds = modelo(batch_inputs, training=False)
        y_pred.extend(tf.argmax(preds, axis=1).numpy())
        y_true.extend(batch_labels.numpy())
        n_samples += batch_labels.shape[0]

    t_total = time.perf_counter() - t_start

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    logging.info(f"F1-score (macro): {f1_macro:.4f}")
    for cls_name, f1_cls in zip(CLASSES, f1_per_class):
        logging.info(f"  F1 {cls_name}: {f1_cls:.4f}")

    logging.info("Classification Report:\n" + classification_report(
        y_true, y_pred, target_names=CLASSES
    ))

    tempo_medio = (t_total / n_samples) * 1000
    logging.info(f"Tempo médio de inferência: {tempo_medio:.2f} ms/talhão ({n_samples} talhões)")


if __name__ == '__main__':
    main()
