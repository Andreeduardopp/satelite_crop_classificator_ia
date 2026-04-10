import os
import shutil
import pandas as pd
import logging
from .processamento_imagens import calcular_area2

# Configuração do logging para salvar em arquivo txt
logging.basicConfig(filename='logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# diretoriodeorigem = './processadas'
# diretoriodestino = './processadas'
# def mover_pasta():
#     for index, row in df.iterrows():
#         cultura = row['cultura']
#         ref_infra_v = row['ref_infra_v']
#         for filename in os.listdir(diretoriodeorigem):
#             if ref_infra_v in filename:
#                 caminho_origem = os.path.join(diretoriodeorigem, filename)
#                 caminho_destino = os.path.join(pasta_cultura, filename)
#                 shutil.move(caminho_origem, caminho_destino)
#                 print(f"Arquivo {filename} movido para {pasta_cultura}")


# def data_arquivos():
#     df = pd.read_csv('dataframe.csv')
#     for index, row in df.iterrows():
#         ref_infra_v = row['ref_infra_v']
#         lista_arquivos = buscar_arquivo_por_nome(termo=ref_infra_v)

#     df['imagens_processadas'] = [lista_arquivos]
#     df.to_csv('dataframe_processado.csv')    


def buscar_pngs_em_hashes(caminho_base):
    """
    Verifica se existe a pasta imagens/{ref_infra_v}/
    e retorna o caminho completo dos arquivos .png dentro das subpastas (hashes).
    """
    pasta_ref = os.path.join(caminho_base)
    encontrados = []

    # verifica se a pasta existe
    if not os.path.isdir(pasta_ref):
        return None # pode retornar None se preferir

    # percorre as subpastas (hashes)
    for hash_dir in os.listdir(pasta_ref):
        caminho_hash = os.path.join(pasta_ref, hash_dir)
        if os.path.isdir(caminho_hash):
            for arquivo in os.listdir(caminho_hash):
                if arquivo.endswith(".png"):
                    encontrados.append(os.path.join(caminho_hash, arquivo))

    return encontrados


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


def data_arquivos():
    logging.info("Iniciando processamento...")
    df = pd.read_csv('dataframe.csv')
    
    culturas = df['cultura'].tolist()
    all_areas = [0 for i in range(len(culturas))]
    all_imagens_baixadas = [0 for i in range(len(culturas))]
    all_imagens_processadas = [0 for i in range(len(culturas))]

    count_row = 0
    for index, row in df.iterrows():
        cultura = row['cultura']
        ref_infra_v = row['ref_infra_v']
        kml = row['path']
        
        datas_cultura = lista_datas_cultura(cultura)
        # imagem_mask = imagem +'_d'+ str(datas_cultura[j])
        # path = f"./imagens/{imagem_mask}/{nome[0]}/response.png"

        lista_por_data = []
        lista_por_data_pross = []

        for j in range(len(datas_cultura)):

            folder_imagem = ref_infra_v +'_d'+ str(datas_cultura[j])
            lista_arquivos = buscar_pngs_em_hashes(f"./imagens/{folder_imagem}")        

            if lista_arquivos:
                logging.info(f"Imagens baixadas para {ref_infra_v}: {lista_arquivos}")
                lista_por_data.append(lista_arquivos)

                imagem_processada = f"./processadas/mascara_{folder_imagem}.png"

                lista_por_data_pross.append(imagem_processada)
                logging.info(f"Imagens processadas para {ref_infra_v}: {lista_arquivos}")
                

            else:
                logging.info(f"Imagens não baixadas para {ref_infra_v}")

        logging.info("calculando area: "+ kml)
        area = calcular_area2(kml)
        all_areas[count_row] = area                 

        all_imagens_baixadas[count_row] = lista_por_data
        all_imagens_processadas[count_row] = lista_por_data_pross
        count_row += 1

    logging.info("Atribuindo listas..")

    df['area'] = all_areas
    df['imagens_baixadas'] = all_imagens_baixadas
    df['imagens_processadas'] = all_imagens_processadas
    df.to_csv('dataframe_processado.csv', index=False) # index=False is a good practice

    logging.info("Finalizado..")

data_arquivos()


