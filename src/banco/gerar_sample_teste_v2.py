"""
Gera um banco SQLite de amostra v2 para testes/treino do modelo de classificação de culturas.

Lê da base real e extrai 6000 registros de cada cultura (milho, soja, trigo, feijão),
excluindo registros que já estejam no DB_TREINO.

Uso:
    python src/banco/gerar_sample_teste_v2.py
"""

import sqlite3
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Adjust the paths to be relative to the script location or project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_ORIGEM   = os.path.join(BASE_DIR, 'src', 'bkp', '26-04-06.dados.db')
DB_TREINO   = os.path.join(BASE_DIR, 'sample_treino_6k.db')
# Creating the new database where the user asked: sample_db_6k_v2.db
DB_DESTINO  = os.path.join(BASE_DIR, 'sample_db_6k_v2.db')
TABELA      = 'culturas'
CULTURAS    = ['milho', 'soja', 'trigo', 'feijão', 'feijao'] # Added both variations of feijão just in case
N_AMOSTRAS  = 6000


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
    
    # We will attach the memory db or check if DB_TREINO exists
    if os.path.exists(DB_TREINO):
        origem.execute(f"ATTACH DATABASE '{DB_TREINO}' AS treino")
        tem_treino = True
    else:
        logging.warning(f"DB_TREINO {DB_TREINO} não encontrado. Não vamos excluir registros de treino.")
        tem_treino = False

    for cultura in CULTURAS:
        # Se for para usar apenas 'feijão' ou 'feijao', e obtermos os valores.
        # Nós usamos um limite de 6000.
        
        where_treino = "AND imagens_processadas NOT IN (SELECT imagens_processadas FROM treino.{})".format(TABELA) if tem_treino else ""
        
        query = f"""
            SELECT cultura, mes, imagens_processadas
            FROM   {TABELA}
            WHERE  cultura = ?
              AND  imagens_processadas IS NOT NULL
              AND  imagens_processadas != '[]'
              {where_treino}
            ORDER BY RANDOM()
            LIMIT  {N_AMOSTRAS}
        """
        
        try:
            rows = origem.execute(query, (cultura,)).fetchall()
            
            if len(rows) > 0:
                dest_cur.executemany(
                    f"INSERT INTO {TABELA} (cultura, mes, imagens_processadas) VALUES (?, ?, ?)",
                    rows
                )
                logging.info(f"{cultura}: {len(rows)} registros inseridos.")
            else:
                logging.info(f"{cultura}: Nenhum registro encontrado.")
        except sqlite3.OperationalError as e:
            logging.error(f"Erro ao consultar {cultura}: {e}")

    destino.commit()
    if tem_treino:
        origem.execute("DETACH DATABASE treino")


def main() -> None:
    # Verificações de caminho
    if not os.path.exists(DB_ORIGEM):
        logging.error(f"Banco de origem {DB_ORIGEM} não encontrado.")
        return

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
