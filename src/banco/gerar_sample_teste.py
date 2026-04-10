"""
Gera um banco SQLite de amostra para testes do modelo de classificação de culturas.

Lê da base real (bkp/26-04-06.dados.db) e extrai 100 registros de cada cultura,
excluindo registros cujo imagens_processadas já esteja em sample_treino.db
(garantindo que os conjuntos de treino e teste não se sobreponham).

Uso:
    python banco/gerar_sample_teste.py
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_ORIGEM   = './src/bkp/26-04-06.dados.db'
DB_TREINO   = './sample_treino_6k.db'
DB_DESTINO  = './sample_teste.db'
TABELA      = 'culturas'
CULTURAS    = ['milho', 'soja', 'trigo']
N_AMOSTRAS  = 100


def criar_tabela(cursor: sqlite3.Cursor) -> None:
    cursor.execute(f"DROP TABLE IF EXISTS {TABELA}")
    cursor.execute(f"""
        CREATE TABLE {TABELA} (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            cultura             TEXT    NOT NULL,
            mes                 INTEGER,
            imagens_processadas TEXT
        )
    """)


def copiar_amostra(origem: sqlite3.Connection, destino: sqlite3.Connection) -> None:
    dest_cur = destino.cursor()
    criar_tabela(dest_cur)

    origem.execute(f"ATTACH DATABASE '{DB_TREINO}' AS treino")

    for cultura in CULTURAS:
        rows = origem.execute(
            f"""
            SELECT cultura, mes, imagens_processadas
            FROM   {TABELA}
            WHERE  cultura = ?
              AND  imagens_processadas IS NOT NULL
              AND  imagens_processadas != '[]'
              AND  imagens_processadas NOT IN (
                       SELECT imagens_processadas FROM treino.{TABELA}
                   )
            ORDER BY RANDOM()
            LIMIT  {N_AMOSTRAS}
            """,
            (cultura,)
        ).fetchall()

        dest_cur.executemany(
            f"INSERT INTO {TABELA} (cultura, mes, imagens_processadas) VALUES (?, ?, ?)",
            rows
        )
        logging.info(f"{cultura}: {len(rows)} registros inseridos.")

    destino.commit()
    origem.execute("DETACH DATABASE treino")


def main() -> None:
    with sqlite3.connect(DB_ORIGEM) as origem, \
         sqlite3.connect(DB_DESTINO) as destino:
        copiar_amostra(origem, destino)

    # Verificação rápida
    with sqlite3.connect(DB_DESTINO) as conn:
        counts = conn.execute(
            f"SELECT cultura, COUNT(*) FROM {TABELA} GROUP BY cultura"
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) FROM {TABELA}").fetchone()[0]

    logging.info("--- Resultado ---")
    for cultura, n in counts:
        logging.info(f"  {cultura}: {n} registros")
    logging.info(f"  Total: {total} registros")
    logging.info(f"Banco salvo em: {DB_DESTINO}")


if __name__ == '__main__':
    main()
