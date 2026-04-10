# Lê ARROZ_519339805-1_plantio_01-12-24_colheita_01-04-25

import os
import cv2
import pandas as pd
import datetime
import uuid

# from processamento_sentinel_Hub import request_sentinel_hub
# from processamento_imagens import aplica_mascara, treshold_indice
# from main import prediz_sigmoide


def get_uuid():
    return str(uuid.uuid4()).split('-')[0]

# lista_arquivos = os.listdir('pasta_kmls')

# Colocar função para montar dataframe a partir do lista arquivos

# lista_arquivos['cultura'] = 'split 0' # A cultura vai ser y^: a variavel que o modelo tem que acertar
# lista_arquivos['ref_infra_v'] = 'split 1' # Add numero aleatorio para evitar duplicidade + infra
# lista_arquivos['ref_rgb'] = 'split 1' # Add numero aleatorio para evitar duplicidade + RGB
# lista_arquivos['data'] = 'split 3' # Ajustar de acordo com o ZARC por cultura
# lista_arquivos['mes'] = 'split 3.2'
# lista_arquivos['path'] = 'string toda'

pasta_culturas = '/home/henrique/Projetos/treinamento/kmls/culturas'
dataframe = []

for root, dirs, files in os.walk(pasta_culturas):
    qtd_arquivos = 0

    for file in files:

        if qtd_arquivos == 9840:
            break

        if file.endswith('.kml'):
            # ARROZ_517267258-1_plantio_01-01-24_colheita_15-05-24.kml
            path_completo = os.path.join(root, file)
            nome = os.path.splitext(file)[0]
            partes = nome.split('_')

            try:
                if len(partes) >= 6:
                    cultura = partes[0]
                    ref = partes[1]
                    # plantio = datetime.datetime.strptime(partes[3], '%d-%m-%y').date()
                    # colheita = datetime.datetime.strptime(partes[5], '%d-%m-%y').date()
                    data = datetime.datetime.strptime(partes[3], '%d-%m-%y').date()
                    mes = data.month

                    uuid_str = get_uuid()
                    dataframe.append({
                        'cultura': cultura.lower(),
                        'ref_infra_v': uuid_str + "_v",  # Ajustar conforme necessário
                        'ref_rgb': uuid_str + "_rgb",      # Ajustar conforme necessário
                        'data': data,
                        'mes': mes,
                        'path': path_completo
                    })

                    qtd_arquivos += 1
            except Exception as e:
                print(f"Erro em arquivo {nome}:", e)
                continue


df = pd.DataFrame(dataframe)
df.to_csv('dataframe.csv', index=False)


# Primeiro: Para cada kml baixar a imagem e renomear com a ref

# request_sentinel_hub()

# Segundo: Aplicar mascara

# aplica_mascara()

# Terceiro: Fazer limpeza

# treshold_indice()

# Quarto: Normalizar a imagem

# cv2.resize(imagem, 224, 224)

# Quinto: Chama os modelos com o main.py
# lista_arquivos['vetor_infra_v'] = Vai ser o resultado do quinto processo
# lista_arquivos['vetor_rgb'] = Vai ser o resultado do quinto processo

# prediz_sigmoide()

# Sexto: Salvar o dataframe completo (usado para treinar o novo modelo)