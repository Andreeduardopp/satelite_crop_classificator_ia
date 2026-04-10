"""
Treina um classificador de culturas (milho, soja, trigo) usando
EfficientNetB0 com transfer learning sobre imagens de satélite
processadas (BANDAS_RBN mascaradas ao talhão).

Fases:
    1. Base congelada — treina apenas a cabeça de classificação
    2. Fine-tuning  — descongela as últimas 20 camadas da base

Uso:
    python treinar_classificador.py
"""

import os
import sqlite3
import ast
import logging
import time
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f'treino_efficientnet_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)

# ── Configurações ─────────────────────────────────────────────────────────────

DB_PATH       = './sample_treino.db'
TABELA        = 'culturas'
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FASE1  = 10
EPOCHS_FASE2  = 15
LR_FASE1      = 1e-3
LR_FASE2      = 1e-5
MODELO_SAIDA  = './modelos/classificador_cultura.keras'
CLASSES       = ['milho', 'soja', 'trigo']
SEED          = 42

# ── Dados ─────────────────────────────────────────────────────────────────────

def carregar_dados(db_path: str) -> tuple[list[list[str]], list[int]]:
    """
    Retorna (registros, labels_int) onde cada registro é uma lista de
    caminhos de imagem pertencentes ao MESMO talhão/campo.

    O split treino/val deve ser feito nesta granularidade (por registro)
    para evitar data leakage — imagens do mesmo talhão em datas diferentes
    são visualmente quase idênticas.
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
        validos = [p for p in paths if os.path.exists(p)]
        if validos:
            registros.append(validos)
            labels.append(classe_para_id[cultura])

    return registros, labels


def expandir_registros(registros: list[list[str]], labels: list[int]) -> tuple[list[str], list[int]]:
    """Expande registros agrupados em listas planas de (caminhos, labels)."""
    caminhos = []
    labels_exp = []
    for paths, label in zip(registros, labels):
        for p in paths:
            caminhos.append(p)
            labels_exp.append(label)
    return caminhos, labels_exp


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

    # Preprocessamento do EfficientNet: escala pixels para [-1, 1]
    def preprocessar(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    ds = ds.map(preprocessar, num_parallel_calls=tf.data.AUTOTUNE)

    if treino:
        ds = ds.shuffle(1024, seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Modelo ────────────────────────────────────────────────────────────────────

def criar_modelo(num_classes: int) -> tf.keras.Model:
    """EfficientNetB0 com cabeça de classificação customizada."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False

    modelo = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    return modelo


# ── Treino ────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Carregar dados
    logging.info("Carregando dados...")
    registros, labels = carregar_dados(DB_PATH)
    total_imgs = sum(len(r) for r in registros)
    logging.info(f"Total: {len(registros)} registros, {total_imgs} imagens válidas")

    for i, c in enumerate(CLASSES):
        n = labels.count(i)
        logging.info(f"  {c}: {n} registros")

    # 2. Split treino / validação POR REGISTRO (80/20, estratificado)
    #    Todas as imagens de um mesmo talhão ficam no mesmo lado do split,
    #    evitando data leakage entre campos fotografados em datas diferentes.
    reg_treino, reg_val, lab_reg_treino, lab_reg_val = train_test_split(
        registros, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    cam_treino, lab_treino = expandir_registros(reg_treino, lab_reg_treino)
    cam_val, lab_val = expandir_registros(reg_val, lab_reg_val)
    logging.info(f"Treino: {len(cam_treino)} imgs ({len(reg_treino)} registros) | "
                 f"Validação: {len(cam_val)} imgs ({len(reg_val)} registros)")

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

    # Predições para F1-score e tempo de inferência
    y_true = []
    y_pred = []
    n_samples = 0
    t_start = time.perf_counter()

    for batch_imgs, batch_labels in ds_val:
        preds = modelo(batch_imgs, training=False)
        y_pred.extend(tf.argmax(preds, axis=1).numpy())
        y_true.extend(batch_labels.numpy())
        n_samples += batch_imgs.shape[0]

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
    logging.info(f"Tempo médio de inferência: {tempo_medio:.2f} ms/amostra ({n_samples} amostras)")


if __name__ == '__main__':
    main()
