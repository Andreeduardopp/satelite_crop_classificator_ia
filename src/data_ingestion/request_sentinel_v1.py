import json
import logging
import requests
from sentinelhub import SHConfig

logger = logging.getLogger(__name__)


class SentinelHubService:
    def __init__(self) -> None:
        config = SHConfig()
        config.sh_client_id = '8103d553-6466-4a14-8861-a59f11dc7387'
        config.sh_client_secret = '4Ewmc2uMHghjubNUnWfVkhDlGztPo4o8'
        
        self.url_base = "https://services.sentinel-hub.com"
        self.client_id = config.sh_client_id
        self.secret_id = config.sh_client_secret

        self.url_autenticacao = f'{self.url_base}/auth/realms/main/protocol/openid-connect/token'

    def autenticar(self):
        try:
            data = {'grant_type': 'client_credentials', 'client_id': self.client_id, 'client_secret': self.secret_id}
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            resposta = requests.post(
                url=self.url_autenticacao, timeout=25, headers=headers, data=data
            )
            resposta.raise_for_status()
            return f"Bearer {resposta.json()['access_token']}"
        except Exception as e:
            logger.exception(e)
            raise e

    def consultar_imagens_tiff(self, access_token, coordenadas, data_inicio, data_fim, width, height):
        try:
            self.headers = {
                'Authorization': access_token,
                'Content-Type': 'application/json',
                'Accept': 'application/tar',
            }
            payload = json.dumps(self._get_imagens_tiff_payload(coordenadas, data_inicio, data_fim, width, height))
            url = f'{self.url_base}/api/v1/process'
            resposta = requests.post(
                url=url, timeout=25, headers=self.headers, data=payload
            )
            resposta.raise_for_status()
            return resposta.content
        except Exception as e:
            raise e

    def _get_imagens_tiff_payload(self, coordenadas, data_inicio, data_fim, width, height):
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                // B05/B06/B07/B8A added for red-edge chlorophyll indices (NDRE, CIRE,
                // MTCI) and B11 reused for NDMI; these separate spectrally-similar
                // cereals (e.g. AVEIA vs TRIGO) that NDVI cannot.
                input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "dataMask"],
                output: [
                    { id: "ndvi_tiff", bands: 1, sampleType: "FLOAT32" },
                    { id: "evi_tiff",  bands: 1, sampleType: "FLOAT32" }, // New EVI output
                    { id: "ndwi_tiff", bands: 1, sampleType: "FLOAT32" },
                    { id: "ndbi_tiff", bands: 1, sampleType: "FLOAT32" },
                    { id: "ndre_tiff", bands: 1, sampleType: "FLOAT32" },
                    { id: "cire_tiff", bands: 1, sampleType: "FLOAT32" },
                    { id: "mtci_tiff", bands: 1, sampleType: "FLOAT32" },
                    { id: "psri_tiff", bands: 1, sampleType: "FLOAT32" },
                    { id: "ndmi_tiff", bands: 1, sampleType: "FLOAT32" }
                ]
            };
        }

        function safeDiv(a, b) { return (b === 0.0) ? 0.0 : a / b; }
        function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

        function evaluatePixel(sample) {
            if (sample.dataMask === 0) {
                return {
                    ndvi_tiff: [NaN], evi_tiff: [NaN], ndwi_tiff: [NaN],
                    ndbi_tiff: [NaN], ndre_tiff: [NaN], cire_tiff: [NaN],
                    mtci_tiff: [NaN], psri_tiff: [NaN], ndmi_tiff: [NaN]
                };
            }

            let ndvi = index(sample.B08, sample.B04);
            let ndwi = index(sample.B03, sample.B08);
            let ndbi = index(sample.B11, sample.B08);
            let ndre = index(sample.B8A, sample.B05);   // red-edge chlorophyll
            let ndmi = index(sample.B08, sample.B11);   // canopy water / drydown

            // Mathematical computation of the Enhanced Vegetation Index (EVI)
            let denominator = sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0;
            let evi = (denominator === 0) ? 0 : 2.5 * (sample.B08 - sample.B04) / denominator;

            let cire = clamp(safeDiv(sample.B07, sample.B05) - 1.0, -1.0, 20.0);
            let mtci = clamp(safeDiv(sample.B06 - sample.B05, sample.B05 - sample.B04), -10.0, 20.0);
            let psri = clamp(safeDiv(sample.B04 - sample.B02, sample.B06), -2.0, 2.0);

            return {
                ndvi_tiff: [ndvi], evi_tiff: [evi], ndwi_tiff: [ndwi],
                ndbi_tiff: [ndbi], ndre_tiff: [ndre], cire_tiff: [cire],
                mtci_tiff: [mtci], psri_tiff: [psri], ndmi_tiff: [ndmi]
            };
        }
        """
        return {
            'input': {
                'bounds': {'geometry': {'type': 'Polygon', 'coordinates': coordenadas}},
                'data': [
                    {
                        'dataFilter': {
                            'timeRange': {'from': f'{str(data_inicio)}T00:00:00Z', 'to': f'{str(data_fim)}T23:59:59Z'},
                        },
                        'type': 'sentinel-2-l2a',
                    }
                ],
            },
            'output': {
                'width': width,
                'height': height,
                'responses': [
                    {'identifier': 'ndvi_tiff', 'format': {'type': 'image/tiff'}},
                    {'identifier': 'evi_tiff',  'format': {'type': 'image/tiff'}},
                    {'identifier': 'ndwi_tiff', 'format': {'type': 'image/tiff'}},
                    {'identifier': 'ndbi_tiff', 'format': {'type': 'image/tiff'}},
                    {'identifier': 'ndre_tiff', 'format': {'type': 'image/tiff'}},
                    {'identifier': 'cire_tiff', 'format': {'type': 'image/tiff'}},
                    {'identifier': 'mtci_tiff', 'format': {'type': 'image/tiff'}},
                    {'identifier': 'psri_tiff', 'format': {'type': 'image/tiff'}},
                    {'identifier': 'ndmi_tiff', 'format': {'type': 'image/tiff'}},
                ],
            },
            'evalscript': evalscript.strip(),
        }