"""
Wrapper baixo-nível do Sentinel Hub Processing API para o V9.

Substitui o request_sentinel_hub do V7/V8 (WCS + layer BANDAS_RBN).
Uma chamada baixa um único TIFF FLOAT32 com 6 bandas, escolhido pelo Sentinel
Hub via mosaicking 'leastCC' dentro de uma janela temporal.

Diferenças vs request_sentinel_hub original:
  - Processing API com evalscript versionado em arquivo (não depende de layer
    pré-configurada no dashboard).
  - leastCC server-side: não baixamos várias e escolhemos cliente; o servidor
    já entrega a melhor.
  - Janela default de 15 dias (vs 5 do original) — combinada com leastCC dá
    cobertura quase certa em qualquer época.
  - Salva como TIFF FLOAT32 (não PNG uint8).
  - Captura erros de rede separadamente para retry no caller.
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from sentinelhub import (
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS,
    bbox_to_dimensions,
    SHConfig,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
EVALSCRIPT_PATH = os.path.join(_HERE, 'evalscript_s2_6bands.js')

with open(EVALSCRIPT_PATH, encoding='utf-8') as _f:
    EVALSCRIPT_S2_6BANDS = _f.read()


# Defaults — overridable via parâmetros
JANELA_DIAS_DEFAULT     = 15      # janela [data - N, data]
MAX_CLOUD_COVER_DEFAULT = 0.30    # filtro server-side por tile
RESOLUCAO_M_DEFAULT     = 10      # resolução de saída (B11/B12 são reamostradas)


def bbox_from_kml(kml_path: str) -> BBox:
    """Extrai bbox WGS84 do polígono em um arquivo KML."""
    with open(kml_path, 'r', encoding='utf-8') as f:
        kml = f.read()
    tree = ET.fromstring(kml)
    coordinates = tree.findall('.//{http://www.opengis.net/kml/2.2}coordinates')[0].text
    pares = [
        tuple(map(float, c.split(',')[:2]))
        for c in coordinates.strip().split()
    ]
    min_x = min(p[0] for p in pares)
    max_x = max(p[0] for p in pares)
    min_y = min(p[1] for p in pares)
    max_y = max(p[1] for p in pares)
    return BBox(bbox=[min_x, min_y, max_x, max_y], crs=CRS.WGS84)


def baixar_tiff_s2(
    data_alvo: datetime,
    kml_path: str,
    tag: str,
    output_dir: str,
    config: SHConfig,
    janela_dias: int = JANELA_DIAS_DEFAULT,
    max_cloud_cover: float = MAX_CLOUD_COVER_DEFAULT,
    resolucao_m: int = RESOLUCAO_M_DEFAULT,
):
    """
    Baixa um TIFF S2 6-bandas para a janela [data_alvo - janela_dias, data_alvo].

    O Sentinel Hub aplica o evalscript no servidor e devolve a melhor imagem
    da janela (mosaicking leastCC). O resultado é salvo em
        output_dir/<tag>/<hash>/response.tiff

    Args:
        data_alvo: data alvo (final da janela).
        kml_path: caminho do KML do talhão (define a bbox).
        tag: identificador único do download (ex.: 'a3f1b2c4_v_d21').
        output_dir: pasta raiz onde o subdir <tag> será criado.
        config: SHConfig já autenticado.
        janela_dias: tamanho da janela temporal anterior a data_alvo.
        max_cloud_cover: filtro server-side de cobertura do tile (0..1).
        resolucao_m: resolução de saída em metros.

    Returns:
        (tiff_path, metadata) ou (None, None) se nenhuma imagem foi retornada.

    Levanta:
        Erros de rede (`requests.RequestException`, `ConnectionError`,
        `TimeoutError`) sobem; o caller decide se faz retry.
    """
    bbox = bbox_from_kml(kml_path)
    data_inicio = data_alvo - timedelta(days=janela_dias)
    time_interval = (
        data_inicio.strftime('%Y-%m-%dT00:00:00Z'),
        data_alvo.strftime('%Y-%m-%dT23:59:59Z'),
    )

    sub_dir = os.path.join(output_dir, tag)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_S2_6BANDS,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                maxcc=max_cloud_cover,
                mosaicking_order='leastCC',
            ),
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF),
        ],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=resolucao_m),
        config=config,
        data_folder=sub_dir,
    )

    # save_data() retorna lista de paths gravados em disco
    saved = request.save_data()
    if not saved:
        return None, None

    tiff_path = saved[0] if isinstance(saved, list) else saved
    metadata = {
        'tag': tag,
        'data_alvo': data_alvo.isoformat(),
        'time_interval': time_interval,
        'bbox': tuple(bbox),
        'resolucao_m': resolucao_m,
        'max_cloud_cover': max_cloud_cover,
    }
    return tiff_path, metadata


def localizar_tiff_existente(tag: str, output_dir: str):
    """
    Procura um response.tiff já baixado em output_dir/<tag>/<hash>/.
    Usado pelo skip-if-exists do pipeline_tiff.py.
    """
    import glob
    pattern = os.path.join(output_dir, tag, '*', 'response.tiff')
    matches = glob.glob(pattern)
    return matches[0] if matches else None
