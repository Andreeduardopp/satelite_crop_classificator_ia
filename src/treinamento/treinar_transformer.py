"""
Treina um classificador de culturas (milho, soja, trigo) usando
Vision Transformer (ViT-B16) com transfer learning sobre imagens de
satélite processadas (BANDAS_RBN mascaradas ao talhão).

Fases:
    1. Base congelada — treina apenas a cabeça de classificação
    2. Fine-tuning  — descongela as últimas 20 camadas da base

Uso:
    python treinar_transformer.py
"""

import os
import sqlite3
import ast
import logging

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from vit_keras import vit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ── Configurações ─────────────────────────────────────────────────────────────

DB_PATH       = './sample_treino.db'
TABELA        = 'culturas'
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16          # ViT é mais pesado em VRAM, batch menor
EPOCHS_FASE1  = 10
EPOCHS_FASE2  = 15
LR_FASE1      = 1e-3
LR_FASE2      = 1e-5
MODELO_SAIDA  = './modelos/classificador_cultura_vit.keras'
CLASSES       = ['milho', 'soja', 'trigo']
SEED          = 42

# ── Dados ─────────────────────────────────────────────────────────────────────

def carregar_dados(db_path: str) -> tuple[list[str], list[int]]:
    """Retorna (caminhos, labels_int) expandindo imagens_processadas."""
    classe_para_id = {c: i for i, c in enumerate(CLASSES)}
    caminhos = []
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
        for p in paths:
            if os.path.exists(p):
                caminhos.append(p)
                labels.append(classe_para_id[cultura])

    return caminhos, labels


def carregar_imagem(caminho: str) -> np.ndarray:
    """Lê e redimensiona uma imagem para o tamanho esperado pelo modelo."""
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img


def criar_dataset(caminhos: list[str], labels: list[int], treino: bool) -> tf.data.Dataset:
    """Cria um tf.data.Dataset a partir de listas de caminhos e labels."""

    def gerador():
        for c, l in zip(caminhos, labels):
            img = carregar_imagem(c)
            yield img, l

    ds = tf.data.Dataset.from_generator(
        gerador,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    # ViT espera pixels em [0, 1]
    def preprocessar(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = ds.map(preprocessar, num_parallel_calls=tf.data.AUTOTUNE)

    if treino:
        ds = ds.shuffle(1024, seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Modelo ────────────────────────────────────────────────────────────────────

def criar_modelo(num_classes: int) -> tf.keras.Model:
    """ViT-B16 com cabeça de classificação customizada."""
    base = vit.vit_b16(
        image_size=IMG_SIZE[0],
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    base.trainable = False

    modelo = tf.keras.Sequential([
        base,
        tf.keras.layers.Lambda(lambda x: x[:, 0]),  # token [CLS]
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    return modelo


# ── Treino ────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Carregar dados
    logging.info("Carregando dados...")
    caminhos, labels = carregar_dados(DB_PATH)
    logging.info(f"Total de imagens válidas: {len(caminhos)}")

    for i, c in enumerate(CLASSES):
        n = labels.count(i)
        logging.info(f"  {c}: {n}")

    # 2. Split treino / validação (80/20, estratificado)
    cam_treino, cam_val, lab_treino, lab_val = train_test_split(
        caminhos, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    logging.info(f"Treino: {len(cam_treino)} | Validação: {len(cam_val)}")

    # 3. Criar datasets
    ds_treino = criar_dataset(cam_treino, lab_treino, treino=True)
    ds_val    = criar_dataset(cam_val, lab_val, treino=False)

    # 4. Class weights (balancear classes)
    pesos = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=np.array(lab_treino))
    class_weight = {i: w for i, w in enumerate(pesos)}
    logging.info(f"Class weights: {class_weight}")

    # 5. Criar modelo
    modelo = criar_modelo(len(CLASSES))
    modelo.summary(print_fn=logging.info)

    # ── Fase 1: Base congelada ────────────────────────────────────────────
    logging.info("=== Fase 1: Treinando cabeça (base congelada) ===")
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
    logging.info("=== Fase 2: Fine-tuning (últimas 20 camadas) ===")
    base = modelo.layers[0]
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

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
    os.makedirs(os.path.dirname(MODELO_SAIDA), exist_ok=True)
    modelo.save(MODELO_SAIDA)
    logging.info(f"Modelo salvo em: {MODELO_SAIDA}")

    # ── Avaliação final ──────────────────────────────────────────────────
    loss, acc = modelo.evaluate(ds_val)
    logging.info(f"Validação final — Loss: {loss:.4f} | Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()
