#!usr/bin/env python3.8
#-*- coding: utf-8 -*-
'''
Created on Fri Oct 7 10:41:26 2022

@author: fabianodicheti
'''
import os
import datetime
import ee
# import geemap
from fastkml import kml
from fastkml import geometry
from geojson import Polygon
# from osgeo import gdal
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon as Poly_shapely
from bs4 import BeautifulSoup as bs

def simplifica_kml(kml_entrada, kml_saida):
    '''
    Remove marcaÃ§Ãµes desnecessÃ¡rias e detalhes de formataÃ§Ã£o de um kml
    importante remover para evitar erros de contorno no momento de gerar o
    geojson;
    Este mÃ©todo recebe o kml, e retorna o mesmo arquivo, porÃ©m, sem as formataÃ§Ãµes
    adicionais.
    '''

    tree = ET.parse(kml_entrada)
    root = tree.getroot()

    for elem in root.iter():
        elem.tag = elem.tag.split('}')[-1]

    kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, 'Document')

    for placemark in root.iter('Placemark'):
        new_placemark = ET.Element('Placemark')
        for polygon in placemark.iter('Polygon'):
            new_placemark.append(polygon)
        document.append(new_placemark)

    tree = ET.ElementTree(kml)
    tree.write(kml_saida, xml_declaration=True, encoding='utf-8')


def geo_json2(kml_caminho):

    kml_filename = kml_caminho
    with open(kml_filename, encoding="utf-8") as kml_file:
        doc = kml_file.read().replace("<fill>false</fill>", "<fill>0</fill>").encode('utf-8')
        doc = kml_file.read().encode('utf-8')
        k = kml.KML()
        k.from_string(doc)
        poligono = []
        for feature0 in k.features():
            #print("{}, {}".format(feature0.name, feature0.description))
            for feature1 in feature0.features():
                if isinstance(feature1.geometry, geometry.Polygon):
                    polygon = feature1.geometry
                    for coord in polygon.exterior.coords:
                        tupla = (coord[0],coord[1])
                        #  long, lat tuplas
                        poligono.append(tupla)

    poligono = Polygon([poligono])

    return poligono



def atualiza_metadados(caminho_arquivo, data, evento, produto):
    """
    Atualiza os metadados de uma imagem com informaÃ§Ãµes do CSV de onde o KML da imagem foi extraÃ­do.

    Args:
    -----------
        caminho_arquivo (str): O caminho para o arquivo da imagem.
        data (str): A data da captura da imagem no formato 'AAAA-MM-DD'.
        evento (str): O nome do evento para o qual a imagem foi capturada.
        produto (str): O resultado do processamento da imagem.

    Retorna:
    -----------
        None
    """
    meta_imagem = gdal.Open(caminho_arquivo, gdal.GA_ReadOnly)
    meta_original = meta_imagem.GetMetadata()
    meta_imagem.SetMetadata({'AREA_OR_POINT':meta_original['AREA_OR_POINT'],
                             'data':data,
                             'evento':evento,
                             'resultado':produto})


def mascara_nuvens(image):
    """
    Aplica uma mÃ¡scara nas nuvens em uma imagem do Google Earth Engine.

    Args:
    -----------
        image: Uma imagem do Google Earth Engine contendo a banda 'QA60'.

    Retorna:
    -----------
        A imagem com as nuvens e cirros mascarados.

    """
    qa60 = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa60.bitwiseAnd(cloud_bit_mask).eq(0)
    mask = qa60.bitwiseAnd(cirrus_bit_mask).eq(0)

    return image.updateMask(mask).divide(10000)



def get_dates(collection, month, year, bounds):

    if month == 12:
        virada = 1
        month2 = 1
    else:
        virada = 0
        month2 = month +1

    filtered = (collection.filter(ee.Filter.calendarRange(year, year+virada, 'year'))
                .filter(ee.Filter.calendarRange(month, month2, 'month'))
                .filterBounds(bounds))

    def format_date(img):
        return ee.Image(img).date().format()

    date_list = filtered.toList(filtered.size()).map(format_date)
    return date_list


def filtrar_por_intervalo(lista_hist, data_inicio, data_fim):
    # Convertendo as datas de string para objetos datetime
    #data_inicio = datetime.datetime.strptime(data_inicio, '%Y-%m-%d')
    #data_fim = datetime.datetime.strptime(data_fim, '%Y-%m-%d')


    # Filtrando as datas dentro do intervalo
    datas_filtradas = []
    for data_hora_str in lista_hist:
        data_hora = datetime.datetime.strptime(data_hora_str, '%Y-%m-%dT%H:%M:%S')
        if data_inicio <= data_hora <= data_fim:
            datas_filtradas.append(data_hora_str)

    return datas_filtradas


def baixa_geotif(data_inicio, data_fim, nome_kml, caminho_destino):
    """
    Baixa imagens do Google Earth Engine e exporta em formato GeoTIFF.

    Args:
    -----------
        data_inicio: A data de inÃ­cio da busca das imagens.
        data_fim: A data de fim da busca das imagens.
        nome_kml: O caminho do arquivo KML contendo o polÃ­gono de interesse.
        caminho_destino: O caminho para salvar o arquivo GeoTIFF.

    Retorna:
    -----------
        None.
    """

    nome_kml_simplificado = nome_kml.replace('.kml', '_simplificado.kml')
    simplifica_kml(nome_kml, nome_kml_simplificado)

    poligono = geo_json2(nome_kml_simplificado)
    #os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)

    service_account = 'experimentos@experimentos-370501.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, './app/priv.json')
    ee.Initialize(credentials)

    imagem = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(data_inicio, data_fim)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
              .map(mascara_nuvens).filterBounds(poligono)
              .sort('CLOUD_COVER').first())

    img_bandas = ee.Image(imagem).select(['B2','B4','B8'])

    asset = imagem.getInfo()

    asset_id = asset['properties']['system:index']

    asset_id = f'COPERNICUS/S2_SR/{asset_id}'

    collection = ee.ImageCollection('COPERNICUS/S2')


    data_str = data_inicio.strftime('%Y-%m-%d')
    month = int(data_str[5:7])
    year = int(data_str[0:4])

    datas = get_dates(collection, month, year, poligono)

    datas_list = datas.getInfo()


    arquivo = caminho_destino
    geemap.ee_export_image(img_bandas,
                           filename=arquivo,
                           scale=9,
                           crs='EPSG:4326',
                           region=poligono,
                           file_per_band=False)

    data_satelite = filtrar_por_intervalo(datas_list, data_inicio, data_fim)

    atualiza_metadados(arquivo, data_satelite, 'place holder', 'place holder')

    os.remove(nome_kml_simplificado)

    return data_satelite, asset_id


def geo_json(kml_caminho):
    kml_filename = kml_caminho
    with open(kml_filename, encoding="utf-8") as kml_file:
        doc = kml_file.read().replace("<fill>false</fill>", "<fill>0</fill>").encode('utf-8')

    return converte_poligono_do_kml_em_objeto_bs(doc)


def converte_poligono_do_kml_em_objeto_bs(poligonos: str):
    soup = bs(poligonos, "xml")
    geometrias_validas = []

    for placemark in soup.find_all("Placemark"):
        polygons = []
        multi = placemark.find("MultiGeometry")
        if multi:
            polygons = multi.find_all("Polygon")
        else:
            poly = placemark.find("Polygon")
            if poly:
                polygons = [poly]

        lista_lat = []
        lista_long = []
        for polygon_element in polygons:
            # 1. Encontra e processa o anel externo (outerBoundaryIs)
            outer_boundary = polygon_element.find("outerBoundaryIs")
            if not outer_boundary:
                continue

            outer_coords_text = outer_boundary.find("coordinates").text
            points_outer = []

            for pt in outer_coords_text.strip().split():
                lon_lat = pt.split(",")
                if len(lon_lat) < 2:
                    continue
                lon, lat = map(float, lon_lat[:2])
                points_outer.append((lon, lat))
                lista_lat.append(lat)
                lista_long.append(lon)

            # 2. Encontra e processa os anéis internos (innerBoundaryIs)
            inner_boundaries = polygon_element.find_all("innerBoundaryIs")
            points_inners = []
            for inner_boundary in inner_boundaries:
                coordinates_inners = inner_boundary.find_all("coordinates")
                for cordinate in coordinates_inners:
                    inner_coords_text = cordinate.text
                    points_inner = []
                    for pt in inner_coords_text.strip().split():
                        lon_lat = pt.split(",")
                        if len(lon_lat) < 2:
                            continue
                        lon, lat = map(float, lon_lat[:2])
                        points_inner.append((lon, lat))
                    points_inners.append(points_inner)

            if points_outer:
                # Cria um Shapely Polygon com o anel externo e todos os internos
                poly_shapely = Poly_shapely(shell=points_outer, holes=points_inners)

                # Validação e correção, se necessário
                if not poly_shapely.is_valid:
                    poly_shapely = poly_shapely.buffer(0)

                if poly_shapely.is_valid:
                    # Converte polígono do Shapely → WKT → GEOSGeometry (Django)
                    geometrias_validas.append(poly_shapely)

    if not geometrias_validas:
        return None

    return  lista_lat, lista_long