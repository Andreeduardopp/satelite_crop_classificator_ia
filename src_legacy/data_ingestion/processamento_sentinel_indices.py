"""
Serviço de download de índices espectrais (NDVI, NDWI, NDBI) do Sentinel-2.

Diferenças em relação a `processamento_sentinel_Hub.py`:
    - Usa `SentinelHubRequest` (Process API) em vez de `WcsRequest` (depreciado)
    - Retorna TIFF float32 multi-banda em vez de PNG uint8
    - Calcula 3 índices a partir das bandas B03, B04, B08, B11 via evalscript
    - Preserva a precisão real dos índices (-1.0 a 1.0 em float32)

Índices calculados:
    NDVI = (B08 - B04) / (B08 + B04)  → vegetação / biomassa
    NDWI = (B03 - B08) / (B03 + B08)  → conteúdo de água foliar
    NDBI = (B11 - B08) / (B11 + B08)  → umidade / estrutura do canopy

Saída:
    ./imagens_indices/{ref_destino}/indices_{data}.tiff
    → TIFF float32 com shape (3, H, W) — bandas [NDVI, NDWI, NDBI]
    → pixels fora do dataMask do Sentinel são NaN

Uso como script (teste em 1 KML):
    python src/dados/processamento_sentinel_indices.py

Uso como módulo:
    from processamento_sentinel_indices import baixar_indices_espectrais
    caminho, data_eff = baixar_indices_espectrais(
        data=datetime(2025, 5, 5),
        caminho_kml='./src/kmls/SOJA_XXX.kml',
        ref_destino='soja_xxx_d21',
    )
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import tifffile

from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)

# Adiciona o diretório atual ao path para importar google_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from google_engine import geo_json


# ── Configuração ──────────────────────────────────────────────────────────────
# Reutiliza o perfil já salvo por processamento_sentinel_Hub.py
config = SHConfig('main_test')


# ── Evalscript: NDVI + NDWI + NDBI em um só request ──────────────────────────
EVALSCRIPT_INDICES = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B03", "B04", "B08", "B11", "dataMask"]
        }],
        output: [
            { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
            { id: "ndwi", bands: 1, sampleType: "FLOAT32" },
            { id: "ndbi", bands: 1, sampleType: "FLOAT32" }
        ]
    };
}

function evaluatePixel(sample) {
    if (sample.dataMask === 0) {
        return {
            ndvi: [NaN],
            ndwi: [NaN],
            ndbi: [NaN]
        };
    }
    return {
        ndvi: [index(sample.B08, sample.B04)],
        ndwi: [index(sample.B03, sample.B08)],
        ndbi: [index(sample.B11, sample.B08)]
    };
}
"""


# ── Utilitários KML → BBox ───────────────────────────────────────────────────

def _bbox_from_kml(caminho_kml: str) -> BBox:
    """
    Lê o polígono do KML via `geo_json()` e retorna o BBox envolvente em WGS84.

    O `geo_json()` de google_engine.py retorna (lista_lat, lista_long).
    """
    resultado = geo_json(caminho_kml)
    if resultado is None:
        raise ValueError(f"KML sem geometria válida: {caminho_kml}")

    lista_lat, lista_long = resultado
    if not lista_lat or not lista_long:
        raise ValueError(f"KML com listas de coordenadas vazias: {caminho_kml}")

    min_lon = min(lista_long)
    max_lon = max(lista_long)
    min_lat = min(lista_lat)
    max_lat = max(lista_lat)

    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)


# ── Download principal ───────────────────────────────────────────────────────

def baixar_indices_espectrais(
    data: datetime,
    caminho_kml: str,
    ref_destino: str,
    pasta_saida: str = './imagens_indices',
    janela_dias: int = 5,
    max_cloud_coverage: float = 0.3,
) -> tuple[str, datetime] | None:
    """
    Baixa NDVI/NDWI/NDBI para um talhão e data, salva como TIFF float32 3-bandas.

    Args:
        data: data-alvo (usa janela retroativa de `janela_dias` para mosaico)
        caminho_kml: caminho do KML do talhão
        ref_destino: nome da subpasta de saída (ex: 'soja_000203e6_d21')
        pasta_saida: raiz onde salvar os TIFFs
        janela_dias: quantos dias para trás olhar a partir de `data`
        max_cloud_coverage: filtro de nuvens (0.0 a 1.0)

    Retorna:
        (caminho_tiff_final, data_alvo) em caso de sucesso
        None se nenhuma imagem disponível na janela
    """
    data_fim = data
    data_inicio = data - timedelta(days=janela_dias)

    bbox = _bbox_from_kml(caminho_kml)
    size = bbox_to_dimensions(bbox, resolution=10)  # 10 m/pixel

    destino_dir = os.path.join(pasta_saida, ref_destino)
    os.makedirs(destino_dir, exist_ok=True)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_INDICES,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(
                    data_inicio.strftime('%Y-%m-%d'),
                    data_fim.strftime('%Y-%m-%d'),
                ),
                mosaicking_order='mostRecent',
                maxcc=max_cloud_coverage,
            ),
        ],
        responses=[
            SentinelHubRequest.output_response('ndvi', MimeType.TIFF),
            SentinelHubRequest.output_response('ndwi', MimeType.TIFF),
            SentinelHubRequest.output_response('ndbi', MimeType.TIFF),
        ],
        bbox=bbox,
        size=size,
        config=config,
        data_folder=destino_dir,
    )

    resposta = request.get_data(save_data=False)
    if not resposta:
        return None

    # `resposta` é uma lista: [{ 'ndvi.tif': np.ndarray, 'ndwi.tif': ..., 'ndbi.tif': ... }]
    # Quando há múltiplos outputs, retorna dict por request
    primeiro = resposta[0]
    if not isinstance(primeiro, dict):
        # Fallback: se vier como lista de arrays (ordem definida pelo evalscript)
        arrays = [np.asarray(a, dtype=np.float32) for a in primeiro]
    else:
        # Chaves esperadas: ndvi.tif, ndwi.tif, ndbi.tif (ou ndvi, ndwi, ndbi)
        def _pega(chave: str) -> np.ndarray:
            for k in (chave, f"{chave}.tif", f"{chave}.tiff"):
                if k in primeiro:
                    return np.asarray(primeiro[k], dtype=np.float32)
            raise KeyError(f"Índice '{chave}' não encontrado na resposta: {list(primeiro.keys())}")

        arrays = [_pega('ndvi'), _pega('ndwi'), _pega('ndbi')]

    # Empilhar em (3, H, W)
    stacked = np.stack(arrays, axis=0)

    # Se as bandas vierem com shape (H, W, 1), achatar
    if stacked.ndim == 4 and stacked.shape[-1] == 1:
        stacked = stacked[..., 0]

    caminho_tiff = os.path.join(destino_dir, f'indices_{data.strftime("%Y-%m-%d")}.tiff')
    tifffile.imwrite(
        caminho_tiff,
        stacked,
        photometric='minisblack',
        metadata={
            'bands': 'NDVI,NDWI,NDBI',
            'source': 'Sentinel-2 L2A',
            'kml': os.path.basename(caminho_kml),
            'data_alvo': data.strftime('%Y-%m-%d'),
        },
    )

    return caminho_tiff, data


# ── CLI de teste ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Uso: python src/dados/processamento_sentinel_indices.py <caminho_kml> <YYYY-MM-DD> [ref_destino]")
        print("Exemplo:")
        print("  python src/dados/processamento_sentinel_indices.py ./src/kmls/SOJA_XXX.kml 2025-05-05 teste_soja")
        sys.exit(1)

    caminho_kml = sys.argv[1]
    data_str = sys.argv[2]
    ref_destino = sys.argv[3] if len(sys.argv) > 3 else 'teste_indices'

    data_alvo = datetime.strptime(data_str, "%Y-%m-%d")

    print(f"Baixando índices espectrais para {caminho_kml} na data {data_str}...")
    resultado = baixar_indices_espectrais(
        data=data_alvo,
        caminho_kml=caminho_kml,
        ref_destino=ref_destino,
    )

    if resultado is None:
        print("Nenhuma imagem disponível na janela temporal.")
        sys.exit(1)

    caminho, _ = resultado
    print(f"TIFF salvo: {caminho}")

    # Inspeção rápida
    arr = tifffile.imread(caminho)
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    for i, nome in enumerate(['NDVI', 'NDWI', 'NDBI']):
        banda = arr[i]
        valid = banda[~np.isnan(banda)]
        if valid.size > 0:
            print(f"  {nome}: min={valid.min():.3f} max={valid.max():.3f} mean={valid.mean():.3f}")
        else:
            print(f"  {nome}: sem pixels válidos")
