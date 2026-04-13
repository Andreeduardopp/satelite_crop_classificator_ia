"""
Treina um classificador de culturas (milho, soja, trigo) usando
EfficientNetB0 com SEQUÊNCIA multi-temporal (TensorFlow/Keras) — versão v6.

Diferenças em relação ao v3:
    1. FiLM condicionado em (dia, mês) em vez de apenas dia.
       O XGBoost mostrou que `mes` (10.3% importance) e `last_dia` (22.1%)
       são os dois features mais discriminantes para milho vs soja.
       O v3 ignorava `mes` completamente — o v6 corrige isso.
    2. Mês é tratado como CATEGÓRICO via Embedding(12, 8), não como
       escalar normalizado. Meses não são linearmente ordenados para
       fenologia agrícola (dez é mais perto de jan do que de jul).
    3. FiLM recebe concat(dia_scalar, mes_embedding) → 9-dim → Dense(64)
       → γ, β. Mesma estrutura do v3, só muda a dimensão de entrada.

Tudo o resto é idêntico ao v3:
    - Mesmo backbone (EfficientNetB0 ImageNet)
    - Mesma augmentation
    - Mesmo label smoothing 0.1
    - Mesmo DB (sample_treino_6k.db, que já tem coluna `mes`)
    - Mesmas fases de treino (congelado → fine-tune 20 camadas)

Uso:
    python src/treinamento/treinar_classificador_v6.py
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = './logs'
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

# ── Configurações ─────────────────────────────────────────────────────────────
DB_PATH         = './sample_treino_6k.db'
TABELA          = 'culturas'
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
EPOCHS_FASE1    = 10
EPOCHS_FASE2    = 15
LR_FASE1        = 1e-3
LR_FASE2        = 1e-5
LABEL_SMOOTHING = 0.1
MODELO_SAIDA    = './modelos/classificador_cultura_efficientnet_v6'
CLASSES         = ['milho', 'soja', 'trigo']
SEED            = 42
MAX_SEQ_LEN     = 3
MAX_DIA         = 100.0
FEATURE_DIM     = 1280    # EfficientNetB0 output dim
MES_EMBED_DIM   = 8       # dimensão do embedding de mês


# ── Dados ─────────────────────────────────────────────────────────────────────

def extrair_dia(caminho: str) -> int:
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_dados(db_path: str) -> tuple[list[list[tuple[str, int]]], list[int], list[int]]:
    """
    Retorna (registros, labels, meses).

    Cada registro é uma lista de (caminho, dia) ORDENADOS por dia.
    `meses[i]` é o mês de plantio (1-12) do talhão i.
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
            if os.path.exists(p):
                validos.append((p, extrair_dia(p)))

        if validos:
            validos.sort(key=lambda x: x[1])
            registros.append(validos)
            labels.append(classe_para_id[cultura])
            meses.append(int(mes) if mes else 1)

    return registros, labels, meses


def carregar_imagem_raw(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

def criar_dataset(registros, labels, meses, treino: bool) -> tf.data.Dataset:
    """
    Cada sample: ((images, dias, mes, mask), label_onehot)

    `mes` é um escalar float32 (1.0-12.0) — convertido para int dentro
    do modelo antes do Embedding lookup.
    """
    num_classes = len(CLASSES)

    def gerador():
        for items, label, mes in zip(registros, labels, meses):
            seq_len = min(len(items), MAX_SEQ_LEN)

            images = np.zeros((MAX_SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
            dias = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
            mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

            for i in range(seq_len):
                caminho, dia = items[i]
                images[i] = carregar_imagem_raw(caminho)
                dias[i] = dia / MAX_DIA
                mask[i] = 1.0

            label_onehot = np.zeros(num_classes, dtype=np.float32)
            label_onehot[label] = 1.0

            yield (images, dias, np.float32(mes), mask), label_onehot

    ds = tf.data.Dataset.from_generator(
        gerador,
        output_signature=(
            (
                tf.TensorSpec(shape=(MAX_SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),       # mes (escalar)
                tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
        ),
    )

    if treino:
        ds = ds.shuffle(1024, seed=SEED)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Modelo ────────────────────────────────────────────────────────────────────

class EfficientNetTemporalModelV6(tf.keras.Model):
    """
    EfficientNetB0 + FiLM(dia, mês) + Temporal Attention.

    Única mudança vs v3: o FiLM agora recebe concat(dia, mes_embedding)
    em vez de apenas dia. O mes_embedding é um Embedding(12, 8) aprendido,
    tratando o mês como variável categórica.
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)

        # Data augmentation (só ativa em training=True)
        self.augment = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.15),
                tf.keras.layers.RandomBrightness(0.1),
            ],
            name="augment",
        )

        # EfficientNetB0 backbone (compartilhado entre timesteps)
        self.backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        )
        self.backbone.trainable = False
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        # Mês → embedding aprendido (categórico, 12 valores)
        # Índices 0-11 internamente (mes - 1)
        self.mes_embedding = tf.keras.layers.Embedding(
            input_dim=12, output_dim=MES_EMBED_DIM, name='mes_emb'
        )

        # FiLM: (dia, mes_emb) → γ e β
        # Input dim: 1 (dia) + MES_EMBED_DIM (8) = 9
        self.film_hidden = tf.keras.layers.Dense(64, activation='relu', name='film_hidden')
        self.film_gamma = tf.keras.layers.Dense(
            FEATURE_DIM, kernel_initializer='zeros', bias_initializer='zeros',
            name='film_gamma',
        )
        self.film_beta = tf.keras.layers.Dense(
            FEATURE_DIM, kernel_initializer='zeros', bias_initializer='zeros',
            name='film_beta',
        )

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
        images, dias, mes, mask = inputs
        # images: (B, T, H, W, 3)
        # dias:   (B, T)
        # mes:    (B,)  — float, 1.0-12.0
        # mask:   (B, T)

        B = tf.shape(images)[0]
        T = MAX_SEQ_LEN

        # ── EfficientNet features ─────────────────────────────────────
        images_flat = tf.reshape(images, (-1, IMG_SIZE[0], IMG_SIZE[1], 3))
        images_flat = self.augment(images_flat, training=training)
        images_flat = tf.keras.applications.efficientnet.preprocess_input(images_flat)
        features_flat = self.backbone(images_flat, training=training)
        features_flat = self.pool(features_flat)          # (B*T, 1280)
        features = tf.reshape(features_flat, (B, T, -1))  # (B, T, 1280)

        # ── FiLM: (dia, mes_embedding) → γ, β ────────────────────────
        # Mes embedding: (B,) → int → (B, MES_EMBED_DIM) → tile → (B, T, MES_EMBED_DIM)
        mes_idx = tf.cast(mes - 1.0, tf.int32)                         # (B,) 0-indexed
        mes_emb = self.mes_embedding(mes_idx)                           # (B, MES_EMBED_DIM)
        mes_emb = tf.tile(tf.expand_dims(mes_emb, 1), [1, T, 1])       # (B, T, MES_EMBED_DIM)

        # Dia: (B, T) → (B, T, 1)
        dia_expanded = tf.expand_dims(dias, -1)

        # Concat → (B, T, 1 + MES_EMBED_DIM) = (B, T, 9)
        context = tf.concat([dia_expanded, mes_emb], axis=-1)

        film_h = self.film_hidden(context)       # (B, T, 64)
        gamma = self.film_gamma(film_h)           # (B, T, 1280)
        beta = self.film_beta(film_h)             # (B, T, 1280)

        # Modulação (γ, β zero-init → começa como identidade)
        tokens = features * (1.0 + gamma) + beta

        # ── Temporal attention ────────────────────────────────────────
        attn_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
        attn_bias = (1.0 - attn_mask) * -1e9

        attn_out1 = self.attn1(tokens, tokens, attention_mask=attn_bias, training=training)
        tokens = self.norm1(tokens + attn_out1)

        attn_out2 = self.attn2(tokens, tokens, attention_mask=attn_bias, training=training)
        tokens = self.norm2(tokens + attn_out2)

        # Mean pooling dos tokens válidos
        mask_expanded = tf.expand_dims(mask, -1)
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

    logging.info(
        "Modelo EfficientNetB0 V6: v3 + FiLM(dia, mes_embedding) — "
        "testando se `mes` fecha o gap para o XGBoost (91%)"
    )

    # 1. Carregar dados
    logging.info("Carregando dados...")
    registros, labels, meses = carregar_dados(DB_PATH)
    total_imgs = sum(len(r) for r in registros)
    logging.info(f"Total: {len(registros)} talhões, {total_imgs} imagens")

    seq_lens = Counter(len(r) for r in registros)
    for n, count in sorted(seq_lens.items()):
        logging.info(f"  Talhões com {n} imagem(ns): {count}")

    for i, c in enumerate(CLASSES):
        n = labels.count(i)
        logging.info(f"  {c}: {n} talhões")

    mes_dist = Counter(meses)
    logging.info("Distribuição de meses:")
    for m in sorted(mes_dist):
        logging.info(f"  mes {m:>2}: {mes_dist[m]} talhões")

    # 2. Split por talhão (evita data leakage)
    reg_treino, reg_val, lab_treino, lab_val, mes_treino, mes_val = train_test_split(
        registros, labels, meses, test_size=0.2, stratify=labels, random_state=SEED
    )
    logging.info(f"Treino: {len(reg_treino)} talhões | Validação: {len(reg_val)} talhões")

    # 3. Datasets
    ds_treino = criar_dataset(reg_treino, lab_treino, mes_treino, treino=True)
    ds_val = criar_dataset(reg_val, lab_val, mes_val, treino=False)

    # 4. Class weights
    pesos = compute_class_weight(
        'balanced', classes=np.arange(len(CLASSES)), y=np.array(lab_treino)
    )
    class_weight = {i: w for i, w in enumerate(pesos)}
    logging.info(f"Class weights: {class_weight}")

    # 5. Modelo
    logging.info("Criando EfficientNetB0 V6 + FiLM(dia, mes) + Temporal Attention...")
    modelo = EfficientNetTemporalModelV6(len(CLASSES))

    # Build
    for batch in ds_treino.take(1):
        modelo(batch[0], training=False)
    modelo.summary(print_fn=logging.info)

    # ── Fase 1: Base congelada ────────────────────────────────────────────
    logging.info("=== Fase 1: Treinando cabeça + FiLM(dia,mes) + temporal attention (base congelada) ===")
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FASE1),
        loss=loss_fn,
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
        loss=loss_fn,
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
        y_true.extend(tf.argmax(batch_labels, axis=1).numpy())
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

    logging.info("Confusion Matrix (linhas=real, colunas=predito):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    logging.info(header)
    for cls, row in zip(CLASSES, cm):
        logging.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    tempo_medio = (t_total / n_samples) * 1000
    logging.info(f"Tempo médio de inferência: {tempo_medio:.2f} ms/talhão ({n_samples} talhões)")


if __name__ == '__main__':
    main()
