"""
Gera um banco SQLite de amostra para testes de treinamento do modelo de
classificação de culturas.

Lê da base real (bkp/26-04-06.dados.db) e extrai 6000 registros de cada
cultura selecionada, filtrando apenas registros com exatamente 3 imagens
(sequência temporal completa: d21/d31/d56 ou d26/d32/d47) E com todos os
arquivos PNG efetivamente presentes no filesystem.

Uso:
    python banco/gerar_sample_treino.py
"""

import os
import sqlite3
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_ORIGEM   = './src/bkp/26-04-06.dados.db'
DB_DESTINO  = './sample_treino_6k.db'
TABELA      = 'culturas'
CULTURAS    = ['milho', 'soja', 'trigo']
N_AMOSTRAS  = 6000
N_IMAGENS   = 3      # exigir sequência temporal completa


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


def parse_paths(imgs_str: str) -> list[str] | None:
    """Parse seguro do campo imagens_processadas em lista de caminhos."""
    try:
        paths = ast.literal_eval(imgs_str)
        if isinstance(paths, list):
            return paths
    except (ValueError, SyntaxError):
        pass
    return None


def registro_valido(imgs_str: str, n: int) -> bool:
    """
    Valida que imagens_processadas:
      1. contém exatamente n caminhos
      2. todos os n arquivos existem no filesystem
    """
    paths = parse_paths(imgs_str)
    if paths is None or len(paths) != n:
        return False
    return all(os.path.exists(p) for p in paths)


def copiar_amostra(origem: sqlite3.Connection, destino: sqlite3.Connection) -> None:
    dest_cur = destino.cursor()
    criar_tabela(dest_cur)

    for cultura in CULTURAS:
        # Busca candidatos (todos os registros não-vazios — filtramos em Python)
        rows = origem.execute(
            f"""
            SELECT cultura, mes, imagens_processadas
            FROM   {TABELA}
            WHERE  cultura = ?
              AND  imagens_processadas IS NOT NULL
              AND  imagens_processadas != '[]'
            ORDER BY RANDOM()
            """,
            (cultura,)
        ).fetchall()

        # Filtrar: exatamente N_IMAGENS paths E todos existem em disco
        filtrados = []
        descartados_contagem = 0
        descartados_disco = 0
        for r in rows:
            paths = parse_paths(r[2])
            if paths is None or len(paths) != N_IMAGENS:
                descartados_contagem += 1
                continue
            if not all(os.path.exists(p) for p in paths):
                descartados_disco += 1
                continue
            filtrados.append(r)
            if len(filtrados) >= N_AMOSTRAS:
                break

        dest_cur.executemany(
            f"INSERT INTO {TABELA} (cultura, mes, imagens_processadas) VALUES (?, ?, ?)",
            filtrados
        )
        logging.info(
            f"{cultura}: {len(filtrados)} registros inseridos "
            f"(alvo={N_AMOSTRAS}, descartados por contagem={descartados_contagem}, "
            f"por disco={descartados_disco})"
        )

    destino.commit()


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
