"""
Testa o modelo ViT treinado usando os dados de sample_teste.db.

Para cada imagem referenciada no banco, chama o PreditorViT e compara
o resultado com o rótulo real armazenado na coluna `cultura`.

Uso:
    python src/test_vit_model.py
"""

import sqlite3
import ast
import os
import logging
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preditor_vit import PreditorViT

# ── Configurações ─────────────────────────────────────────────────────────────
DB_TESTE   = './sample_teste.db'
TABELA     = 'culturas'
CLASSES    = ['milho', 'soja', 'trigo']
LOG_DIR    = './logs'

os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f'test_vit_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)


# ── Carregar dados ────────────────────────────────────────────────────────────
def carregar_amostras(db_path: str) -> list[tuple[str, str]]:
    """
    Retorna lista de (caminho_imagem, cultura_real) expandindo imagens_processadas.
    Ignora imagens que não existem no filesystem.
    """
    amostras = []
    ignoradas = 0

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT cultura, imagens_processadas FROM {TABELA}"
        ).fetchall()

    for cultura, imgs_str in rows:
        try:
            caminhos = ast.literal_eval(imgs_str)
        except (ValueError, SyntaxError):
            continue

        for caminho in caminhos:
            if os.path.exists(caminho):
                amostras.append((caminho, cultura))
            else:
                ignoradas += 1

    logging.info(f"Imagens encontradas: {len(amostras)}")
    logging.info(f"Imagens ignoradas (path não encontrado): {ignoradas}")
    return amostras


# ── Avaliar ───────────────────────────────────────────────────────────────────
def avaliar(amostras: list[tuple[str, str]], preditor: PreditorViT) -> None:
    y_true = []
    y_pred = []
    erros = []

    logging.info(f"Iniciando predições em {len(amostras)} imagens...")

    for i, (caminho, cultura_real) in enumerate(amostras, 1):
        try:
            cultura_pred, confianca = preditor.predizer(caminho)
            y_true.append(cultura_real)
            y_pred.append(cultura_pred)

            if i % 50 == 0:
                logging.info(f"  [{i}/{len(amostras)}] processadas")
        except Exception as e:
            erros.append((caminho, str(e)))

    if erros:
        logging.warning(f"{len(erros)} imagens falharam durante a predição:")
        for caminho, msg in erros:
            logging.warning(f"  {caminho}: {msg}")

    if not y_true:
        logging.error("Nenhuma predição realizada. Verifique os dados e o modelo.")
        return

    # ── Métricas ──────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average='macro',    labels=CLASSES, zero_division=0)
    f1_micro  = f1_score(y_true, y_pred, average='micro',    labels=CLASSES, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=CLASSES, zero_division=0)
    f1_each   = f1_score(y_true, y_pred, average=None,       labels=CLASSES, zero_division=0)

    logging.info("=" * 55)
    logging.info("RESULTADOS")
    logging.info("=" * 55)
    logging.info(f"Total avaliado : {len(y_true)}")
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
    logging.info("\n" + classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASSES, zero_division=0))

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
    amostras = carregar_amostras(DB_TESTE)

    if not amostras:
        logging.error("Nenhuma amostra válida encontrada. Encerrando.")
        return

    logging.info("Carregando modelo ViT...")
    preditor = PreditorViT()

    avaliar(amostras, preditor)


if __name__ == '__main__':
    main()
