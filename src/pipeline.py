import os
import cv2
import pandas as pd
from time import sleep
from datetime import datetime, timedelta
import uuid
import numpy as np
import logging
# import tensorflow as tf
import ast
import requests
import sqlite3

from dados.processamento_sentinel_Hub import request_sentinel_hub
from dados.processamento_imagens import aplica_mascara, treshold_indice, calcular_area2
# from main import prediz_sigmoide

PASTA_LOGS = './logs'
os.makedirs(PASTA_LOGS, exist_ok=True)
ARQUIVO_LOG_ERRO = os.path.join(PASTA_LOGS, 'error_pipeline.txt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_error_file_handler = logging.FileHandler(ARQUIVO_LOG_ERRO)
_error_file_handler.setLevel(logging.WARNING)
_error_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(_error_file_handler)



def normaliza(valor, minimo, maximo):
    """
    Normaliza um valor para o intervalo [0, 1].

    Argumentos:
        valor (float): O valor a ser normalizado.
        minimo (float): O valor mínimo no intervalo original.
        maximo (float): O valor máximo no intervalo original.

    Retorna:
        float: O valor normalizado no intervalo [0, 1].
    """
    return(valor - minimo)/(maximo - minimo)



def lista_datas_cultura(cultura):
    culturas = {
        "feijão": [21, 31, 56],
        "feijao": [21, 31, 56],
        "soja": [21, 31, 56],
        "aveia": [29, 44, 64],
        "trigo": [26, 32, 47],
        "milho": [21, 31, 56],
        "cafe": [30, 50, 75, 100],
        "café": [30, 50, 75, 100],
        "arroz": [15, 30, 95]
    }
    return culturas[cultura]


# Colocar função para montar dataframe a partir do lista arquivos

# lista_arquivos['cultura'] = 'split 0' # A cultura vai ser y^: a variavel que o modelo tem que acertar
# lista_arquivos['ref_infra_v'] = 'split 1' # Add numero aleatorio para evitar duplicidade + infra
# lista_arquivos['ref_rgb'] = 'split 1' # Add numero aleatorio para evitar duplicidade + RGB
# lista_arquivos['data'] = 'split 3' # Ajustar de acordo com o ZARC por cultura
# lista_arquivos['mes'] = 'split 3.2'
# lista_arquivos['path'] = 'string toda'

DB_NOME = 'dados.db'
DB_TABELA = 'culturas'
PASTA_IMAGENS = './imagens_20k'
CULTURAS_PERMITIDAS = {'aveia', 'trigo', 'feijão', 'feijao', 'milho', 'soja'}


def inicializar_db():
    conn = sqlite3.connect(DB_NOME)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_TABELA} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cultura TEXT,
            ref_infra_v TEXT UNIQUE,
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


def ja_processado(ref_infra_v):
    conn = sqlite3.connect(DB_NOME)
    row = conn.execute(
        f"SELECT imagens_baixadas FROM {DB_TABELA} WHERE ref_infra_v = ?", (ref_infra_v,)
    ).fetchone()
    conn.close()
    if row is None:
        return False
    imagens = ast.literal_eval(row[0]) if isinstance(row[0], str) else row[0]
    return bool(imagens)


def inserir_registro(row, area, imagens_baixadas, imagens_processadas):
    conn = sqlite3.connect(DB_NOME)
    conn.execute(f"""
        INSERT INTO {DB_TABELA} (cultura, ref_infra_v, ref_rgb, data, mes, path, area, imagens_baixadas, imagens_processadas)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ref_infra_v) DO UPDATE SET
            area=excluded.area,
            imagens_baixadas=excluded.imagens_baixadas,
            imagens_processadas=excluded.imagens_processadas
    """, (
        row['cultura'], row['ref_infra_v'], row['ref_rgb'],
        str(row['data']), int(row['mes']), row['path'],
        area, str(imagens_baixadas), str(imagens_processadas)
    ))
    conn.commit()
    conn.close()


def baixar_imagens():
    inicializar_db()
    data_frame = pd.read_csv('dataframe_processado.csv')

    try:
        for _, row in data_frame.iterrows():
            ref_infra_v = row['ref_infra_v']
            cultura = row['cultura']

            if str(cultura).strip().lower() not in CULTURAS_PERMITIDAS:
                continue

            if ja_processado(ref_infra_v):
                logging.info(f"skipping {ref_infra_v}...")
                continue

            arquivo = row['path']
            plantio_base = datetime.strptime(str(row['data']), "%Y-%m-%d")

            logging.info(f"processando {ref_infra_v} -> {arquivo}")
            area = calcular_area2(arquivo)

            imagens_baixadas = []
            imagens_processadas = []

            for dap in lista_datas_cultura(cultura):
                imagem_mask = ref_infra_v + '_d' + str(dap)
                data_dap = plantio_base + timedelta(days=dap)

                try:
                    nome, _, _ = request_sentinel_hub(data_dap, arquivo, imagem_mask, pasta_imagens=PASTA_IMAGENS)

                    mask = f"./mascaras/mascara_{imagem_mask}.png"
                    path = f"{PASTA_IMAGENS}/{imagem_mask}/{nome[0]}/response.png"
                    processada = f"./processadas/mascara_{imagem_mask}.png"
                    aplica_mascara(mask, path, processada, arquivo)

                    imagens_baixadas.append(path)
                    imagens_processadas.append(processada)
                except FileNotFoundError:
                    logging.warning(f"Arquivo não encontrado: {imagem_mask}, DAP {dap}")
                    continue
                except Exception as ex:
                    logging.error(f"Erro no DAP {dap} para {ref_infra_v}: {ex}")
                    continue

            inserir_registro(row, area, imagens_baixadas, imagens_processadas)
            logging.info(f"Salvo no DB: {ref_infra_v}")

    except Exception as ex:
        logging.error(f"Erro inesperado: {ex}")
        logging.info("Finalizado.")


def request_mlserver(ref_infra: str, caminho_imagem: str):
    # imagem = caminho_imagem
    # tamanho_default = (224, 224)
    # payload = tf.image.resize(imagem, tamanho_default)
    # file = {'imagem': (imagem, open(imagem, 'rb'))}

    url = "http://localhost:8080/sigmoides/"
    
    data = {
        "ref_infra_v": ref_infra,
        "mes": 0.5,
        "path_image": caminho_imagem
    }
    response = requests.post(url, files=None, data=data)
    
    return response


def selecionar_dados(cultura, limit=10):
    """
    Seleciona todos os dados de uma tabela SQLite e os exporta para um arquivo CSV.
    """
    try:
        db_nome = '/home/softfocus/tf_serving_4090/serving_gpu/dados.db'
        tabela_nome = 'culturas'

        conn = sqlite3.connect(db_nome)
        query = f"SELECT * FROM {tabela_nome} where cultura='{cultura}' and imagens_baixadas <> '[]' and sigmoides_iv IS NULL limit '{limit}';"
        df = pd.read_sql_query(query, conn)        
        conn.close()

        return df
                
    except Exception as e:
        logging.error(f"Ocorreu um erro ao selecionar dados: {e}")


def executar_request():
    # dataframe_atualizado = pd.read_csv("dataframe_processado.csv")

    limit = 150
    dataframe_milho = selecionar_dados("milho", limit)
    dataframe_feijao = selecionar_dados("feijão", limit)
    dataframe_soja = selecionar_dados("soja", limit)
    dataframe_trigo = selecionar_dados("trigo", limit)
    dataframe_cafe = selecionar_dados("café", limit)
    dataframe_arroz = selecionar_dados("arroz", limit)

    dataframe_atualizado = pd.concat([
        dataframe_milho, 
        dataframe_feijao, 
        dataframe_soja, 
        dataframe_trigo, 
        dataframe_cafe, 
        dataframe_arroz
    ], ignore_index=True)

    dataframe_atualizado["imagens_processadas"] = dataframe_atualizado["imagens_processadas"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    ref_infra = dataframe_atualizado['ref_infra_v'].tolist()
    imagens_processadas = dataframe_atualizado['imagens_processadas'].tolist()
   
    for i in range(len(dataframe_atualizado)):
        index = i

        for j in range(len(imagens_processadas[index])):            
            sig = request_mlserver(ref_infra[index], imagens_processadas[index][j])
            print(f"Reg: {index} - sig: {sig.content}")


if __name__ == '__main__':
    baixar_imagens()

