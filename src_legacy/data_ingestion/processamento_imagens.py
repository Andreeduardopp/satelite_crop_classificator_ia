#!usr/bin/env python3.8
#-*- coding: utf-8 -*-
'''
Created on Fri Oct 7 10:41:26 2022

@author: fabianodicheti
'''

import sys
import os
import time
import asyncio

path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(path)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import matplotlib

from scipy.interpolate import interp1d
from PIL import Image
from datetime import datetime, timedelta
# from osgeo import gdal
from pykml import parser
from pykml.factory import KML_ElementMaker as KML #nao remover
import xml.etree.ElementTree as ET

from google_engine import baixa_geotif, atualiza_metadados, geo_json
from processamento_sentinel_Hub import request_sentinel_hub

matplotlib.use('SVG')

def converter(xml_original):
    """
    Função para converter o arquivo KML do formato 1.x para o formato 2.2

    Precisa dessa fun para KML quebrados
    """

    root = ET.fromstring(xml_original)

    kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, 'Document')

    for placemark in root.findall('.//{http://www.opengis.net/kml/2.2}Placemark'):
        new_placemark = ET.SubElement(document, 'Placemark')
        polygon = placemark.find('.//{http://www.opengis.net/kml/2.2}Polygon')
        new_placemark.append(polygon)

    return ET.tostring(kml, encoding='utf-8').decode()


def converte_kml(original):
    """
    Método para substituir o arquivo KML que esteja no formato antigo pelo novo.

    Precisa dessa fun para KML quebrados (Segundo tipo)
    """
    novo = original.replace('.kml', 'conv.kml')
    # Lê o conteúdo do arquivo KML com as informações originais
    with open(original, 'r', encoding='utf-8') as file:
        xml_original = file.read()

    # Converte o arquivo KML para o formato 2.2
    xml_novo = converter(xml_original)

    # Escreve o conteúdo no novo arquivo KML
    with open(novo, 'w', encoding='utf-8') as file:
        file.write(xml_novo)

    return novo


def calcular_area(coordenadas):
    '''
    Calcula a área de um polígono definido pelas coordenadas fornecidas, usando a fórmula de Gauss.

    Vamos testar um modelo com a area em mt2 como variável de entrada
    '''

    raio_terrestre = 6378137  # raio da Terra em metros
    area_total = 0.0
    ultima_coord = coordenadas[-1]
    for coord in coordenadas:
        lat1, lon1 = ultima_coord
        lat2, lon2 = coord
        delta_lat = (lat2 - lat1) * math.pi / 180
        delta_lon = (lon2 - lon1) * math.pi / 180
        a = math.sin(delta_lat / 2) * math.sin(delta_lat / 2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(delta_lon / 2) * math.sin(delta_lon / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        area_total += raio_terrestre * raio_terrestre * c
        ultima_coord = coord
    return area_total / 2



def calcular_area2(kml):
    lista_lat, lista_long = geo_json(kml)
    lista_lat = [math.radians(lat) for lat in lista_lat]
    lista_long = [math.radians(long) for long in lista_long]

    raio_terrestre = 6378137  # raio da Terra em metros
    area_total = 0.0

    for c in range(len(lista_long)):
        d = (c + 1) % len(lista_long)
        area_total += (lista_long[d] - lista_long[c]) * (2 + math.sin(lista_lat[c]) + math.sin(lista_lat[d]))

    return abs(area_total) * (raio_terrestre**2) / 2


def area_kml(kml):
    """
    Calcula a área total em hectares contida em um arquivo KML.

    Aplica a fun calcular_area()
    """

    with open(kml, 'rb') as f:
        doc = parser.parse(f).getroot()

    # Encontra todos os polígonos no arquivo KML
    poligonos = doc.findall('.//{http://www.opengis.net/kml/2.2}Polygon')

    # Calcula a área total em metros quadrados
    area_total = 0.0
    for poligono in poligonos:
        aneis = poligono.outerBoundaryIs.LinearRing.coordinates.text.split()
        coordenadas = [(float(x.split(',')[0]), float(x.split(',')[1])) for x in aneis]
        area_total += calcular_area(coordenadas)

        hectares = round(area_total / 10000, 2)

    return hectares


def cria_mascara(nome_kml, caminho_destino):
    """
    Cria e salva a máscara utilizada na limpeza de imagens.

    Primeiro passo do processamento de imagem
    """
    os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)
    lista_lat, lista_long = geo_json(nome_kml)

    eixo_x = [lista_lat[i] for i in range(len(lista_lat))]
    eixo_y = [lista_long[j] for j in range(len(lista_long))]
    #os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)

    plt.plot(eixo_x,eixo_y, 'k')
    plt.fill(eixo_x, eixo_y, 'k')
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.savefig(caminho_destino)
    plt.close()


def aplica_mascara(nome_mascara, caminho_fonte, caminho_destino, caminho_kml):
    """
    Aplica a limpeza na imagem através da máscara.

    Args:
    -----------
        nome_mascara (str): caminho completo e nome do arquivo com a máscara a ser aplicada na imagem.
        caminho_fonte (str): caminho completo e nome do arquivo de origem da imagem.
        caminho_destino (str): caminho completo e nome do arquivo de destino para salvar a imagem processada.

    Aplica a fun cria_mascara()
    """
    os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)
    cria_mascara(caminho_kml, nome_mascara)

    imagem = cv2.imread(caminho_fonte)
    mascara = cv2.imread(nome_mascara)
    imagem = cv2.resize(imagem, (mascara.shape[1], mascara.shape[0]))

    mascara = cv2.cvtColor(mascara, cv2.COLOR_RGB2GRAY)
    mascara_inversa = cv2.bitwise_not(mascara)
    kernel = np.ones((5,5))
    mascara_erosao = cv2.erode(mascara_inversa, kernel, iterations=5)
    filtro = cv2.bitwise_or(imagem, imagem, mask=mascara_erosao)

    cv2.imwrite(caminho_destino, filtro)



def treshold_indice(caminho_fonte, nome_mascara, caminho_destino):
    """
    Parameters
    ----------
    caminho_fonte :
        imagem com area de cultivo.
    nome_mascara :
        nome do arquivo de imagem com a mascara do kml
    caminho_destino :
        nome do arquivo de imagem final que devera ser salvo.

    Grava imagem com o limiar aplicado.
    Limpeza da imagem: Aplica corrosão e limpesa de limiar, e grava a imagem
    """

    imagem = cv2.imread(caminho_fonte, 0)

    limite = round(imagem.max()*0.23,0).astype('int')
    lim, thresh = cv2.threshold(imagem, limite, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5))
    erosao = cv2.erode(thresh, kernel, iterations=1)

    imagem_inteira = cv2.imread(caminho_fonte)

    frente = cv2.bitwise_or(imagem_inteira, imagem_inteira, mask=erosao)

    imagem = frente
    mascara = cv2.imread(nome_mascara)
    mascara = cv2.resize(mascara,(imagem.shape[1],imagem.shape[0]))

    mascara = cv2.cvtColor(mascara, cv2.COLOR_RGB2GRAY)
    mascara_inversa = cv2.bitwise_not(mascara)
    kernel = np.ones((5,5))
    mascara_erosao = cv2.erode(mascara_inversa, kernel, iterations=10)
    filtro = cv2.bitwise_or(imagem, imagem, mask=mascara_erosao)

    cv2.imwrite(caminho_destino, filtro)
