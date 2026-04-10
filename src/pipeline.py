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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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

def baixar_imagens():
    data_frame = pd.read_csv('dataframe_processado.csv')

    lista_datas = data_frame['data'].to_list()
    lista_path = data_frame['path'].to_list()
    lista_imagem = data_frame['ref_infra_v'].to_list()

    # indice_aleatorio = np.random.randint(1024, 4096)
    # indice_aleatorio = 200

    cultura_l = data_frame['cultura'].tolist()
    arquivo_l = data_frame['path'].tolist()
    plantio_l = data_frame['data'].tolist()
    imagem_l = data_frame['ref_infra_v'].tolist()

    imagens_baixadas_df = data_frame['imagens_baixadas'].tolist()
    imagens_processadas_df = data_frame['imagens_processadas'].tolist()

    areas_li = [0 for i in range(len(lista_path))]
    imagens_baixadas_li = [0 for i in range(len(lista_path))]
    imagens_processadas_li = [0 for i in range(len(lista_path))]

    try:
        count_requests = 0
        for i in range(len(lista_path)):
            cultura = cultura_l[i]
            arquivo = arquivo_l[i]
            plantio = plantio_l[i]
            imagem = imagem_l[i]

            if len(imagens_baixadas_df[i]) > 2:
                logging.info(f"skipping loop {i}...")
                imagens_baixadas_li[i] = imagens_baixadas_df[i]
                imagens_processadas_li[i] = imagens_processadas_df[i]
                continue

            logging.info(f"processando loop {i} -> {arquivo}")
            area = calcular_area2(arquivo)
            areas_li[i] = area
            plantio = datetime.strptime(plantio, "%Y-%m-%d")

            imagens_baixadas = []
            imagens_processadas = []

            datas_cultura = lista_datas_cultura(cultura)

            for j in range(len(datas_cultura)):
                # Primeiro: Para cada kml baixar a imagem e renomear com a ref
                imagem_mask = imagem +'_d'+ str(datas_cultura[j])
                plantio = plantio + timedelta(days=datas_cultura[j])

                try:
                    nome, val2, val3 = request_sentinel_hub(plantio, arquivo, imagem_mask)

                    mask = f"./mascaras/mascara_{imagem_mask}.png"
                    path = f"./imagens/{imagem_mask}/{nome[0]}/response.png"
                    processada = f"./processadas/mascara_{imagem_mask}.png"
                    aplica_mascara(mask, path, processada, arquivo)

                    imagens_baixadas.append(path)
                    imagens_processadas.append(processada)
                except FileNotFoundError:
                    continue
                except Exception as ex:
                    raise ex

            imagens_baixadas_li[i] = imagens_baixadas
            imagens_processadas_li[i] = imagens_processadas
    except Exception as ex:
        print(ex)

    finally:
        data_frame["area"] = areas_li
        data_frame['imagens_baixadas'] = imagens_baixadas_li
        data_frame['imagens_processadas'] = imagens_processadas_li
        data_frame.to_csv('dataframe_processado_final.csv')


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
    executar_request()

