"""
Gera bancos SQLite de treino e teste para o modelo de classificação de culturas.

Lê da base real (bkp/26-04-06.dados.db) e:
  1. Extrai 5000 registros de cada cultura para o banco de TREINO
  2. Extrai 1000 registros (diferentes) de cada cultura para o banco de TESTE

Culturas: SOJA, MILHO, TRIGO, AVEIA, FEIJÃO (e variante 'feijao' sem acento)

Filtra apenas registros com exatamente 3 imagens (sequência temporal completa)
e com todos os arquivos PNG presentes no filesystem.

Uso:
    python src/banco/gerar_sample_treino.py
"""

import os
import sqlite3
import ast
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_ORIGEM    = os.path.join(BASE_DIR, 'src', 'bkp', '26-04-06.dados.db')
DB_TREINO    = os.path.join(BASE_DIR, 'sample_treino_max9000.db')
DB_TESTE     = os.path.join(BASE_DIR, 'sample_teste_250.db')

TABELA       = 'culturas'

# Culturas alvo — incluímos ambas as grafias de feijão por segurança
CULTURAS     = ['soja', 'milho', 'trigo', 'aveia', 'feijão', 'feijao']

N_MAX_TREINO = 9000   # até 9000 para treino
N_TESTE      = 250    # por cultura (reservado primeiro)
N_IMAGENS    = 3      # exigir sequência temporal completa


# ── Helpers ───────────────────────────────────────────────────────────────

def parse_paths(imgs_str: str) -> list[str] | None:
    """Parse seguro do campo imagens_processadas em lista de caminhos."""
    try:
        paths = ast.literal_eval(imgs_str)
        if isinstance(paths, list):
            return paths
    except (ValueError, SyntaxError):
        pass
    return None


def registro_valido(imgs_str: str) -> bool:
    """
    Valida que imagens_processadas:
      1. contém exatamente N_IMAGENS caminhos
      2. todos os arquivos existem no filesystem
    """
    paths = parse_paths(imgs_str)
    if paths is None or len(paths) != N_IMAGENS:
        return False
    return all(os.path.exists(p) for p in paths)


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


# ── Geração dos bancos ───────────────────────────────────────────────────

def gerar_bancos(origem: sqlite3.Connection,
                 destino_treino: sqlite3.Connection,
                 destino_teste: sqlite3.Connection) -> None:
    cur_treino = destino_treino.cursor()
    cur_teste  = destino_teste.cursor()
    criar_tabela(cur_treino)
    criar_tabela(cur_teste)

    for cultura in CULTURAS:
        logging.info(f"Processando cultura: {cultura}")

        # Busca todos os candidatos (ORDER BY RANDOM para embaralhar)
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

        # Filtra registros válidos (3 imagens na sequência temporal)
        valid_rows = []
        descartados_contagem = 0

        for r in rows:
            paths = parse_paths(r[2])
            if paths is None or len(paths) != N_IMAGENS:
                descartados_contagem += 1
                continue
            valid_rows.append(r)

        # Separa garantindo as 200 amostras para teste
        teste_rows = valid_rows[:N_TESTE]
        treino_rows = valid_rows[N_TESTE : N_TESTE + N_MAX_TREINO]

        # Insere no banco de treino
        cur_treino.executemany(
            f"INSERT INTO {TABELA} (cultura, mes, imagens_processadas) VALUES (?, ?, ?)",
            treino_rows,
        )
        # Insere no banco de teste
        cur_teste.executemany(
            f"INSERT INTO {TABELA} (cultura, mes, imagens_processadas) VALUES (?, ?, ?)",
            teste_rows,
        )

        logging.info(
            f"  {cultura}: treino={len(treino_rows)} (max {N_MAX_TREINO}), "
            f"teste={len(teste_rows)} (alvo {N_TESTE}), "
            f"descartados={descartados_contagem}"
        )

    destino_treino.commit()
    destino_teste.commit()


# ── Verificação ──────────────────────────────────────────────────────────

def verificar_banco(db_path: str, label: str) -> None:
    with sqlite3.connect(db_path) as conn:
        counts = conn.execute(
            f"SELECT cultura, COUNT(*) FROM {TABELA} GROUP BY cultura"
        ).fetchall()
        total = conn.execute(f"SELECT COUNT(*) FROM {TABELA}").fetchone()[0]

    logging.info(f"--- {label} ({db_path}) ---")
    for cultura, n in counts:
        logging.info(f"  {cultura}: {n} registros")
    logging.info(f"  Total: {total} registros")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(DB_ORIGEM):
        logging.error(f"Banco de origem não encontrado: {DB_ORIGEM}")
        return

    logging.info(f"Origem: {DB_ORIGEM}")
    logging.info(f"Treino → {DB_TREINO}  (max {N_MAX_TREINO}/cultura)")
    logging.info(f"Teste  → {DB_TESTE}  ({N_TESTE}/cultura)")

    with sqlite3.connect(DB_ORIGEM) as origem, \
         sqlite3.connect(DB_TREINO) as treino, \
         sqlite3.connect(DB_TESTE)  as teste:
        gerar_bancos(origem, treino, teste)

    verificar_banco(DB_TREINO, "TREINO")
    verificar_banco(DB_TESTE,  "TESTE")

    logging.info("Pronto!")


if __name__ == '__main__':
    main()
