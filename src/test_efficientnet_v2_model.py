"""
Testa o modelo EfficientNetB0 V2 (multi-temporal) usando sample_teste.db.

Diferença para test_vit_model.py:
    O modelo V2 classifica o TALHÃO inteiro (até 3 imagens temporais).
    Portanto, a avaliação agrupa imagens por registro, e cada predição
    consome a sequência completa do talhão.

Uso:
    python src/test_efficientnet_v2_model.py
"""

import os
import re
import ast
import sqlite3
import logging
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from treinamento.treinar_classificador_v2 import (
    EfficientNetTemporalModel,
    IMG_SIZE,
    MAX_SEQ_LEN,
    MAX_DIA,
    CLASSES,
)

# ── Configurações ─────────────────────────────────────────────────────────────
DB_TESTE    = './sample_teste.db'
TABELA      = 'culturas'
PESOS_PATH  = './modelos/classificador_cultura_efficientnet_v2/pesos.weights.h5'
LOG_DIR     = './logs'

os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(
    LOG_DIR, f'test_efficientnet_v2_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)


# ── Carregar dados (agrupados por talhão) ────────────────────────────────────
def extrair_dia(caminho: str) -> int:
    match = re.search(r'_d(\d+)\.png$', caminho)
    return int(match.group(1)) if match else 0


def carregar_registros(db_path: str) -> tuple[list[list[tuple[str, int]]], list[str]]:
    """
    Retorna (registros, culturas) onde cada registro é uma lista de
    (caminho, dia) ORDENADOS por dia, pertencentes ao mesmo talhão.
    """
    registros = []
    culturas = []
    ignorados = 0

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT cultura, imagens_processadas FROM {TABELA}"
        ).fetchall()

    for cultura, imgs_str in rows:
        if cultura not in CLASSES:
            ignorados += 1
            continue
        try:
            paths = ast.literal_eval(imgs_str)
        except (ValueError, SyntaxError):
            ignorados += 1
            continue

        validos = [(p, extrair_dia(p)) for p in paths if os.path.exists(p)]
        if not validos:
            ignorados += 1
            continue

        validos.sort(key=lambda x: x[1])
        registros.append(validos)
        culturas.append(cultura)

    logging.info(f"Talhões encontrados: {len(registros)}")
    logging.info(f"Registros ignorados: {ignorados}")
    return registros, culturas


# ── Pré-processamento (idêntico ao treino) ───────────────────────────────────
def preprocessar_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def montar_sample(items: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta (images, dias, mask) para um único talhão."""
    seq_len = min(len(items), MAX_SEQ_LEN)
    images = np.zeros((MAX_SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    dias = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
    mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)

    for i in range(seq_len):
        caminho, dia = items[i]
        images[i] = preprocessar_imagem(caminho)
        dias[i] = dia / MAX_DIA
        mask[i] = 1.0

    return images, dias, mask


# ── Modelo ────────────────────────────────────────────────────────────────────
def carregar_modelo() -> EfficientNetTemporalModel:
    logging.info(f"Carregando modelo: {PESOS_PATH}")
    modelo = EfficientNetTemporalModel(len(CLASSES))

    # Build do modelo com um input dummy antes de carregar os pesos
    dummy_images = tf.zeros((1, MAX_SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    dummy_dias = tf.zeros((1, MAX_SEQ_LEN), dtype=tf.float32)
    dummy_mask = tf.ones((1, MAX_SEQ_LEN), dtype=tf.float32)
    _ = modelo((dummy_images, dummy_dias, dummy_mask), training=False)

    modelo.load_weights(PESOS_PATH)
    logging.info("Pesos carregados com sucesso.")
    return modelo


# ── Avaliação ─────────────────────────────────────────────────────────────────
def avaliar(registros: list[list[tuple[str, int]]], culturas: list[str],
            modelo: EfficientNetTemporalModel, batch_size: int = 16) -> None:
    y_true = []
    y_pred = []
    erros = []
    total = len(registros)

    logging.info(f"Iniciando predições em {total} talhões (batch={batch_size})...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_items = registros[start:end]
        batch_culturas = culturas[start:end]

        batch_images = []
        batch_dias = []
        batch_mask = []

        for items in batch_items:
            try:
                images, dias, mask = montar_sample(items)
            except Exception as e:
                erros.append((items[0][0] if items else '?', str(e)))
                images = np.zeros((MAX_SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
                dias = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
                mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
            batch_images.append(images)
            batch_dias.append(dias)
            batch_mask.append(mask)

        images_t = tf.convert_to_tensor(np.stack(batch_images), dtype=tf.float32)
        dias_t = tf.convert_to_tensor(np.stack(batch_dias), dtype=tf.float32)
        mask_t = tf.convert_to_tensor(np.stack(batch_mask), dtype=tf.float32)

        preds = modelo((images_t, dias_t, mask_t), training=False)
        pred_idx = tf.argmax(preds, axis=1).numpy()

        for cultura_real, idx in zip(batch_culturas, pred_idx):
            y_true.append(cultura_real)
            y_pred.append(CLASSES[idx])

        logging.info(f"  [{end}/{total}] talhões processados")

    if erros:
        logging.warning(f"{len(erros)} talhões com erro durante o preprocessamento:")
        for caminho, msg in erros[:10]:
            logging.warning(f"  {caminho}: {msg}")

    if not y_true:
        logging.error("Nenhuma predição realizada. Verifique os dados e o modelo.")
        return

    # ── Métricas ──────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average='macro',    labels=CLASSES, zero_division=0)
    f1_micro    = f1_score(y_true, y_pred, average='micro',    labels=CLASSES, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=CLASSES, zero_division=0)
    f1_each     = f1_score(y_true, y_pred, average=None,       labels=CLASSES, zero_division=0)

    logging.info("=" * 55)
    logging.info("RESULTADOS")
    logging.info("=" * 55)
    logging.info(f"Total avaliado : {len(y_true)} talhões")
    logging.info(f"Acurácia       : {acc:.4f}  ({acc:.1%})")
    logging.info(f"F1 macro       : {f1_macro:.4f}")
    logging.info(f"F1 micro       : {f1_micro:.4f}")
    logging.info(f"F1 weighted    : {f1_weighted:.4f}")
    logging.info("")

    logging.info("F1 por classe:")
    for cls, f1 in zip(CLASSES, f1_each):
        logging.info(f"  {cls:<8}: {f1:.4f}")
    logging.info("")

    logging.info("Classification Report:")
    logging.info("\n" + classification_report(
        y_true, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0
    ))

    logging.info("Confusion Matrix (linhas=real, colunas=predito):")
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
    logging.info(header)
    for cls, row in zip(CLASSES, cm):
        logging.info(f"{cls:>10}" + "".join(f"{v:>10}" for v in row))

    logging.info("=" * 55)
    logging.info(f"Log salvo em: {log_filename}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logging.info(f"Carregando amostras de teste: {DB_TESTE}")
    registros, culturas = carregar_registros(DB_TESTE)

    if not registros:
        logging.error("Nenhum talhão válido encontrado. Encerrando.")
        return

    modelo = carregar_modelo()
    avaliar(registros, culturas, modelo)


if __name__ == '__main__':
    main()
