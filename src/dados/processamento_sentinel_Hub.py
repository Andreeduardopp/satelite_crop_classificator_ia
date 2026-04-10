# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:16:42 2024

@author: Fabiano Dicheti

#client = 66d895ef-6894-4c21-8a7c-0e0ec2c41537
#secret = erzOD5se8OwefvOHxoJMDs4QBq5gYwwp

print("Localização do arquivo de configuração:", SHConfig.get_config_location())
print(f"Client ID: {config.sh_client_id}")
print(f"Client Secret: {config.sh_client_secret}")
print(f"Instance ID: {config.instance_id}")

produtos@softfocus.com.br
!YcYcMbFAGnR5k3X

https://apps.sentinel-hub.com/dashboard/#/
"""

import os
import shutil
from datetime import datetime, timedelta
import numpy as np

from sentinelhub import SHConfig, CRS, BBox, DataCollection
from sentinelhub import WcsRequest, MimeType

import xml.etree.ElementTree as ET
from utils import analisar_cobertura_de_nuvens


config = SHConfig()
#config.sh_client_id = '66d895ef-6894-4c21-8a7c-0e0ec2c41537'
#config.sh_client_secret = 'erzOD5se8OwefvOHxoJMDs4QBq5gYwwp'
#config.instance_id = 'b7a41a97-15af-46db-9d6f-cb08f98dbf0f'
#config.save('main_test')

config.sh_client_id = '8103d553-6466-4a14-8861-a59f11dc7387'
config.sh_client_secret = '4Ewmc2uMHghjubNUnWfVkhDlGztPo4o8'
config.instance_id = 'b3969a49-37a6-41e1-982f-0c8a15c8a00f'
config.save('main_test')


def request_sentinel_hub(data, caminho_kml: str, r_imagem: str, sentinel_layer = 'BANDAS_RBN', png = True ) -> str:
    if png:
        mime = MimeType.PNG
        extensao = '.png'
    else:
        mime = MimeType.TIFF
        extensao = '.tiff'

    # TRUE-COLOR-DEM, TRUE-COLOR-S2-L2A

    ref_imagem = './imagens/' + r_imagem

    data_final = data
    data_inicial = data - timedelta(days=5)

    data_inicial = data_inicial.strftime('%Y-%m-%d')
    data_final = data_final.strftime('%Y-%m-%d')

    with open(caminho_kml, 'r') as file:
        kml_content = file.read()

    tree = ET.fromstring(kml_content)
    coordinates = tree.findall('.//{http://www.opengis.net/kml/2.2}coordinates')[0].text

    coords_list = [tuple(map(float, coord.split(',')[:2])) for coord in coordinates.strip().split()]

    min_x = min(coord[0] for coord in coords_list)  # longitude mínima
    max_x = max(coord[0] for coord in coords_list)  # longitude máxima
    min_y = min(coord[1] for coord in coords_list)  # latitude mínima
    max_y = max(coord[1] for coord in coords_list)  # latitude máxima


    bbox = BBox(bbox=[min_x, min_y, max_x, max_y], crs=CRS.WGS84)

    # layer personalizado
    wcs_request = WcsRequest(
        data_collection=DataCollection.SENTINEL2_L2A,
        layer=sentinel_layer,
        bbox=bbox,
        time=(data_inicial, data_final),
        resx='10m',  # Resolução em x
        resy='10m',  # Resolução em y
        image_format=mime,
        config=config,
        data_folder= ref_imagem  # Caminho para salvar os dados
    )


    imagens = wcs_request.get_data(save_data=True)
    dates = wcs_request.get_dates()

    ordem = os.listdir(ref_imagem)
    ordem.sort(key=lambda x: os.path.getmtime(os.path.join(ref_imagem, x)))

    # ASSET_ID = [f"SENTINEL_HUB_XOFFSET{str(min_x)}{str(max_x)}_YOFFSET{str(min_y)}{str(max_y)}_FILE{ordem[i]}_TIMESTAMP{dates[i]}" for i in range(len(dates))]
    ASSET_ID = [f"{ordem[i]}" for i in range(len(dates))]

    lista_imagens = [os.path.join(ordem[r], 'response' + extensao) for r in range(len(ordem))]

    destino_imagens = [os.path.join(ref_imagem, lista_imagens[a]) for a in range(len(ordem))]

    lista_nuvens, _percent = analisar_cobertura_de_nuvens(destino_imagens)

    return ASSET_ID[:1], lista_nuvens[:1], dates[:1]

# data = datetime.strptime("2025-05-05", "%Y-%m-%d")
# val1, val2, val3 = request_sentinel_hub(data, './kmls/ARROZ_517267258-1_plantio_01-01-24_colheita_15-05-24.kml',
#                      'result_sentinel', sentinel_layer='BANDAS_RBN', png=True)

# print(val1)
# print(val2)
# print(val3)