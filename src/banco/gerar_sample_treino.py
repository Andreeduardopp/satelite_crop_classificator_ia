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
DIR_TREINO   = os.path.join(BASE_DIR, 'datasets', 'dataset_treino')
DIR_TESTE    = os.path.join(BASE_DIR, 'datasets', 'dataset_teste')

TABELA       = 'culturas'

# Culturas alvo — incluímos ambas as grafias de feijão por segurança
CULTURAS     = ['soja', 'milho', 'trigo', 'aveia', 'feijão', 'feijao']

N_DATASETS   = 5
N_MAX_TREINO = 9000   # até 9000 para treino
N_TESTE      = 250    # por cultura por dataset (reservado primeiro)
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

def gerar_datasets_multiplos() -> None:
    import random
    os.makedirs(DIR_TREINO, exist_ok=True)
    os.makedirs(DIR_TESTE, exist_ok=True)

    # Prepara as conexões
    conexoes = []
    for i in range(1, N_DATASETS + 1):
        db_treino = os.path.join(DIR_TREINO, f"sample_treino_max9000_v{i}.db")
        db_teste  = os.path.join(DIR_TESTE, f"sample_teste_250_v{i}.db")
        
        conn_treino = sqlite3.connect(db_treino)
        conn_teste  = sqlite3.connect(db_teste)
        
        criar_tabela(conn_treino.cursor())
        criar_tabela(conn_teste.cursor())
        
        conexoes.append({
            'treino': conn_treino,
            'teste': conn_teste,
            'path_treino': db_treino,
            'path_teste': db_teste
        })

    with sqlite3.connect(DB_ORIGEM) as origem:
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

            # Para evitar vazamento de dados, separamos globalmente testes e treinos:
            total_teste_necessario = N_TESTE * N_DATASETS
            
            if len(valid_rows) < total_teste_necessario:
                logging.warning(
                    f"Atenção: cultura {cultura} possui {len(valid_rows)} registros válidos. "
                    f"Menos que os {total_teste_necessario} ideais para testes exclusivos."
                )
            
            pool_teste = valid_rows[:total_teste_necessario]
            pool_treino = valid_rows[total_teste_necessario:]

            # Distribuir para cada dataset
            for i in range(N_DATASETS):
                idx_inicio_teste = i * N_TESTE
                idx_fim_teste = (i + 1) * N_TESTE
                teste_rows = pool_teste[idx_inicio_teste:idx_fim_teste]
                
                # Para o treino, randomizamos o pool restante para ter variação entre os 5 datasets
                if pool_treino:
                    random.shuffle(pool_treino)
                treino_rows = pool_treino[:N_MAX_TREINO]
                
                conn_treino = conexoes[i]['treino']
                conn_teste = conexoes[i]['teste']
                
                if treino_rows:
                    conn_treino.executemany(
                        f"INSERT INTO {TABELA} (cultura, mes, imagens_processadas) VALUES (?, ?, ?)",
                        treino_rows,
                    )
                if teste_rows:
                    conn_teste.executemany(
                        f"INSERT INTO {TABELA} (cultura, mes, imagens_processadas) VALUES (?, ?, ?)",
                        teste_rows,
                    )

            logging.info(
                f"  {cultura}: total_valido={len(valid_rows)}, "
                f"descartados={descartados_contagem}. "
                f"Distribuição concluída para {N_DATASETS} datasets."
            )

    # Fechar, comitar e verificar conexões
    for i, cx in enumerate(conexoes):
        cx['treino'].commit()
        cx['teste'].commit()
        cx['treino'].close()
        cx['teste'].close()
        verificar_banco(cx['path_treino'], f"TREINO v{i+1}")
        verificar_banco(cx['path_teste'],  f"TESTE v{i+1}")


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
    logging.info(f"Gerando {N_DATASETS} datasets...")
    logging.info(f"Treinos → {DIR_TREINO}  (max {N_MAX_TREINO}/cultura)")
    logging.info(f"Testes  → {DIR_TESTE}  ({N_TESTE}/cultura exclusivo por dataset)")

    gerar_datasets_multiplos()

    logging.info("Pronto!")


if __name__ == '__main__':
    main()
