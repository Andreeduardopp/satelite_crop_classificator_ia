import sqlite3
import pandas as pd

import logging
import ast

from dados.processamento_sentinel_Hub import request_sentinel_hub
from dados.processamento_imagens import aplica_mascara, calcular_area2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def criar_tabela_sqlite(nome_banco, nome_tabela):
    """
    Cria uma tabela no banco de dados SQLite com um esquema pré-definido.
    """
    try:
        conn = sqlite3.connect(nome_banco)
        cursor = conn.cursor()

        # Remove a tabela se ela já existir para garantir um estado limpo
        cursor.execute(f"DROP TABLE IF EXISTS {nome_tabela}")

        # Cria a tabela com os tipos de dados apropriados
        cursor.execute(f"""
        CREATE TABLE {nome_tabela} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cultura TEXT,
            ref_infra_v TEXT,
            ref_rgb TEXT,
            data TEXT,
            mes INTEGER,
            path TEXT,
            area REAL,
            imagens_baixadas TEXT,
            imagens_processadas TEXT
        )
        """)

        conn.commit()
        conn.close()
        logging.info(f"Tabela '{nome_tabela}' criada com sucesso no banco de dados '{nome_banco}'.")

    except Exception as e:
        logging.error(f"Ocorreu um erro ao criar a tabela: {e}")


def salvar_csv_no_sqlite(caminho_csv, nome_banco, nome_tabela):
    """
    Lê um arquivo CSV e salva seus dados em uma tabela SQLite.
    """
    try:
        conn = sqlite3.connect(nome_banco)
        df = pd.read_csv(caminho_csv)

        # Remove a primeira coluna
        df = df.iloc[:, 2:]

        # Itera sobre as linhas do DataFrame e insere cada uma no banco de dados
        for index, row in df.iterrows():

            try:
                area = row['area']
                if area == 0:
                    partes = row['path'].split('/')
                    path = f"./culturas/{partes[len(partes)-2]}/{partes[len(partes)-1]}"
                    row['area'] = calcular_area2(path)

            except FileNotFoundError:
                logging.error(f"Erro: Ao calcular area de '{path}'.")

            # Tratamento específico para a coluna 'imagens_baixadas'
            imagens_baixadas_raw = row['imagens_baixadas']
            if imagens_baixadas_raw == '0' or imagens_baixadas_raw == '':
                continue

            try:
                # Tenta avaliar a string como um literal Python (ex: '["path1", "path2"]')
                imagens_lista = ast.literal_eval(imagens_baixadas_raw)

                # Verifica se é uma lista de listas e achata se necessário
                if any(isinstance(i, list) for i in imagens_lista):
                    imagens_lista = [item for sublist in imagens_lista for item in sublist]

                # Converte a lista de volta para uma string para salvar no DB
                row['imagens_baixadas'] = str(imagens_lista)
            except (ValueError, SyntaxError):
                # Se não for uma lista literal (já é uma string simples), usa o valor como está
                pass

            cursor = conn.cursor()
            # As colunas no CSV devem corresponder às colunas da tabela, exceto 'id'
            colunas = ', '.join(row.index)
            placeholders = ', '.join(['?'] * len(row))
            sql = f"INSERT INTO {nome_tabela} ({colunas}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(row))

        conn.commit()
        conn.close()
        logging.info(f"Dados do '{caminho_csv}' salvos com sucesso na tabela '{nome_tabela}'.")

    except FileNotFoundError:
        logging.error(f"Erro: O arquivo '{caminho_csv}' não foi encontrado.")
    except Exception as e:
        logging.error(f"Ocorreu um erro ao salvar o CSV no SQLite: {e}")
        logging.error(tuple(row))


def selecionar_dados_e_exportar_para_csv(db_nome, tabela_nome, csv_export_path):
    """
    Seleciona todos os dados de uma tabela SQLite e os exporta para um arquivo CSV.
    """
    try:
        conn = sqlite3.connect(db_nome)
        query = f"SELECT * FROM {tabela_nome}"

        df = pd.read_sql_query(query, conn)
        conn.close()

        df.to_csv(csv_export_path, index=False)
        logging.info(f"Dados da tabela '{tabela_nome}' exportados com sucesso para '{csv_export_path}'.")
    except Exception as e:
        logging.error(f"Ocorreu um erro ao selecionar dados e exportar para CSV: {e}")


if __name__ == '__main__':
    db_nome = 'dados.db'
    tabela_nome = 'culturas'
    csv_path = 'dataframe_processado_final.csv'

    # criar_tabela_sqlite(db_nome, tabela_nome)
    #salvar_csv_no_sqlite(csv_path, db_nome, tabela_nome)

    csv_export_path = 'dataframe_processado.csv'
    selecionar_dados_e_exportar_para_csv(db_nome, tabela_nome, csv_export_path)
