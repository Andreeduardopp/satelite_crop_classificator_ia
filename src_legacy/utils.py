import cv2
import os
import shutil
import datetime
import time
import numpy as np


def analisar_cobertura_de_nuvens(caminhos_imagens, limiar=230):
    porcentagens_imagens = []

    for caminho in caminhos_imagens:
        # Carregar a imagem
        imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

        # Verificar se a imagem foi carregada
        if imagem is None:
            print(f"Erro ao carregar a imagem {caminho}")
            continue

        # Contar o número de pixels acima do limiar
        pixels_nuvens = np.sum(imagem >= limiar)
        total_pixels = imagem.size

        # Calcular a porcentagem de pixels de nuvens
        porcentagem = (pixels_nuvens / total_pixels) * 100
        porcentagens_imagens.append((caminho, porcentagem))

    # Ordenar as imagens pela porcentagem de pixels de nuvens
    porcentagens_imagens.sort(key=lambda x: x[1])

    # Separar as tuplas em duas listas
    caminhos_ordenados, porcentagens_ordenadas = zip(*porcentagens_imagens)

    return list(caminhos_ordenados), list(porcentagens_ordenadas)

def limpar_task_temp(task_id : str):
    DIRETORY_TEMP = "app/temp"
    if os.path.exists(f'{DIRETORY_TEMP}/{task_id}'):
        # Deleta a pasta inteira
        shutil.rmtree(f'{DIRETORY_TEMP}/{task_id}')
        print(f"Pasta '{f'{DIRETORY_TEMP}/{task_id}'}' deletada com sucesso!")
    else:
        print(f"A pasta '{f'{DIRETORY_TEMP}/{task_id}'}' não existe.")
