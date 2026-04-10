"""
Avalia as imagens do banco de amostra para verificar integridade, dimensões
e disponibilidade antes do treinamento.

Uso:
    python avaliar_imagens.py
"""

import sqlite3
import ast
import os
import cv2
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = './sample_treino.db'
TABELA  = 'culturas'


def carregar_caminhos(db_path: str) -> list[tuple[str, str]]:
    """Retorna lista de (caminho_imagem, cultura) expandindo imagens_processadas."""
    pares = []
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
            pares.append((caminho, cultura))

    return pares


def avaliar(pares: list[tuple[str, str]]) -> None:
    total = len(pares)
    encontradas = 0
    nao_encontradas = 0
    corrompidas = 0
    shapes = Counter()
    dtypes = Counter()
    por_cultura = Counter()
    por_cultura_valida = Counter()

    for caminho, cultura in pares:
        por_cultura[cultura] += 1

        if not os.path.exists(caminho):
            nao_encontradas += 1
            continue

        img = cv2.imread(caminho)
        if img is None:
            corrompidas += 1
            continue

        encontradas += 1
        shapes[img.shape] += 1
        dtypes[img.dtype.name] += 1
        por_cultura_valida[cultura] += 1

    logging.info("=== Relatório de Avaliação ===")
    logging.info(f"Total de imagens referenciadas: {total}")
    logging.info(f"Encontradas e válidas:          {encontradas}")
    logging.info(f"Não encontradas (path errado):  {nao_encontradas}")
    logging.info(f"Corrompidas (cv2 falhou):       {corrompidas}")
    logging.info("")

    logging.info("--- Imagens por cultura (total / válidas) ---")
    for cultura in sorted(por_cultura):
        logging.info(f"  {cultura}: {por_cultura[cultura]} / {por_cultura_valida[cultura]}")

    logging.info("")
    logging.info("--- Dimensões encontradas (top 10) ---")
    for shape, count in shapes.most_common(10):
        h, w, c = shape
        logging.info(f"  {w}x{h}x{c}: {count} imagens")

    logging.info("")
    logging.info("--- Dtypes ---")
    for dtype, count in dtypes.most_common():
        logging.info(f"  {dtype}: {count}")


def main() -> None:
    logging.info(f"Carregando caminhos de {DB_PATH}...")
    pares = carregar_caminhos(DB_PATH)

    if not pares:
        logging.error("Nenhuma imagem encontrada no banco.")
        return

    logging.info(f"{len(pares)} imagens referenciadas. Avaliando...")
    avaliar(pares)


if __name__ == '__main__':
    main()
