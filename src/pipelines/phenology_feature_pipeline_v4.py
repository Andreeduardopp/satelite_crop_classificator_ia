import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import io
import re
import json
import math
import time
import tarfile
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests
import numpy as np
import tifffile
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon
from pyproj import Geod
import xml.etree.ElementTree as ET

from src.data_ingestion.request_sentinel_v1 import SentinelHubService

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# -- Constants -----------------------------------------------------------------

STAGES = ["baseline", "emergence", "vegetative", "flowering", "grain_fill", "maturity"]

OPTICAL_INDICES = ["NDVI", "NDWI", "EVI"]
STATS = ["mean", "median", "std", "p10", "p90"]

SAR_INDICES = ["VV", "VH", "CR", "RVI"]

# Sentinel-1 revisits ~12 days in Brazil; 24-day minimum guarantees >= 1 pass
MIN_SAR_WINDOW_DAYS = 24

# Sentinel-2 revisits ~5 days; 20-day minimum guarantees ~4 passes,
# greatly increasing the chance of at least one cloud-free scene.
MIN_OPTICAL_WINDOW_DAYS = 20

CLEAR_SCL = frozenset({2, 4, 5, 6, 7})

FIXED_COLS = [
    "field_id", "crop_label", "planting_date",
    "area_hectares", "latitude", "longitude", "stages_covered",
]
TEXT_COLS = {"field_id", "crop_label", "planting_date"}

OPTICAL_FEATURE_COLS = [
    f"{idx}_{stat}_{stage}"
    for idx in OPTICAL_INDICES
    for stage in STAGES
    for stat in STATS
]

SAR_FEATURE_COLS = [
    f"{idx}_{stat}_{stage}"
    for idx in SAR_INDICES
    for stage in STAGES
    for stat in STATS
]

ALL_COLS = FIXED_COLS + OPTICAL_FEATURE_COLS + SAR_FEATURE_COLS

CROP_STAGE_WINDOWS = {
    "SOJA": {
        "baseline":   (-15, 0),
        "emergence":  (0, 20),
        "vegetative": (20, 55),
        "flowering":  (55, 75),
        "grain_fill": (75, 100),
        "maturity":   (100, 130),
    },
    "MILHO": {
        "baseline":   (-15, 0),
        "emergence":  (0, 25),
        "vegetative": (25, 55),
        "flowering":  (55, 85),
        "grain_fill": (85, 115),
        "maturity":   (115, 140),
    },
    "FEIJAO": {
        "baseline":   (-15, 0),
        "emergence":  (0, 10),
        "vegetative": (10, 25),
        "flowering":  (25, 45),
        "grain_fill": (45, 65),
        "maturity":   (65, 90),
    },
    "ARROZ": {
        "baseline":   (-15, 0),
        "emergence":  (0, 25),
        "vegetative": (25, 60),
        "flowering":  (60, 85),
        "grain_fill": (85, 115),
        "maturity":   (115, 145),
    },
    "TRIGO": {
        "baseline":   (-15, 0),
        "emergence":  (0, 20),
        "vegetative": (20, 50),
        "flowering":  (50, 75),
        "grain_fill": (75, 105),
        "maturity":   (105, 135),
    },
    "AVEIA": {
        "baseline":   (-15, 0),
        "emergence":  (0, 20),
        "vegetative": (20, 45),
        "flowering":  (45, 70),
        "grain_fill": (70, 100),
        "maturity":   (100, 130),
    },
    "CAFE": {
        "baseline":   (-15, 0),
        "emergence":  (0, 30),
        "vegetative": (30, 90),
        "flowering":  (90, 150),
        "grain_fill": (150, 210),
        "maturity":   (210, 270),
    },
}

DEFAULT_STAGE_WINDOWS = {
    "baseline":   (-15, 0),
    "emergence":  (0, 20),
    "vegetative": (20, 50),
    "flowering":  (50, 80),
    "grain_fill": (80, 110),
    "maturity":   (110, 140),
}

FILENAME_REGEX = re.compile(
    r"(.+?)_(.+)_plantio_(\d{2}-\d{2}-\d{2})_colheita_(\d{2}-\d{2}-\d{2})\.kml",
    re.IGNORECASE,
)

CROP_NAME_NORMALIZE = {
    "SOJA": "SOJA", "MILHO": "MILHO", "ARROZ": "ARROZ",
    "TRIGO": "TRIGO", "AVEIA": "AVEIA",
    "CAFE": "CAFE", "CAFÉ": "CAFE",
    "FEIJAO": "FEIJAO", "FEIJÃO": "FEIJAO",
}

# -- Evalscripts --------------------------------------------------------------

EVALSCRIPT_PROCESS = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04", "B08", "B11", "SCL", "dataMask"],
        output: [
            { id: "ndvi_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "evi_tiff",  bands: 1, sampleType: "FLOAT32" },
            { id: "ndwi_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "scl_tiff",  bands: 1, sampleType: "UINT8" }
        ]
    };
}

function evaluatePixel(sample) {
    if (sample.dataMask === 0) {
        return {
            ndvi_tiff: [NaN],
            evi_tiff:  [NaN],
            ndwi_tiff: [NaN],
            scl_tiff:  [0]
        };
    }

    let ndvi = index(sample.B08, sample.B04);
    let ndwi = index(sample.B03, sample.B08);

    let denom = sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0;
    let evi = (denom === 0) ? 0 : 2.5 * (sample.B08 - sample.B04) / denom;

    return {
        ndvi_tiff: [ndvi],
        evi_tiff:  [evi],
        ndwi_tiff: [ndwi],
        scl_tiff:  [Math.round(sample.SCL)]
    };
}
""".strip()

EVALSCRIPT_OPTICAL_STATS = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04", "B08", "B11", "SCL", "dataMask"],
        output: [
            { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
            { id: "evi",  bands: 1, sampleType: "FLOAT32" },
            { id: "ndwi", bands: 1, sampleType: "FLOAT32" },
            { id: "dataMask", bands: 1 }
        ]
    };
}

function evaluatePixel(sample) {
    var scl = sample.SCL;
    var isClear = (scl == 2 || scl == 4 || scl == 5 || scl == 6 || scl == 7);
    var mask = (sample.dataMask && isClear) ? 1 : 0;

    var ndvi = index(sample.B08, sample.B04);
    var ndwi = index(sample.B03, sample.B08);

    var denom = sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0;
    var evi = (denom === 0) ? 0 : 2.5 * (sample.B08 - sample.B04) / denom;

    return {
        ndvi: [ndvi],
        evi:  [evi],
        ndwi: [ndwi],
        dataMask: [mask]
    };
}
""".strip()

EVALSCRIPT_SAR_STATS = """
//VERSION=3
function setup() {
    return {
        input: ["VV", "VH", "dataMask"],
        output: [
            { id: "vv",  bands: 1, sampleType: "FLOAT32" },
            { id: "vh",  bands: 1, sampleType: "FLOAT32" },
            { id: "cr",  bands: 1, sampleType: "FLOAT32" },
            { id: "rvi", bands: 1, sampleType: "FLOAT32" },
            { id: "dataMask", bands: 1 }
        ]
    };
}

function evaluatePixel(sample) {
    var mask = sample.dataMask;
    var vv = Math.max(sample.VV, 1e-10);
    var vh = Math.max(sample.VH, 1e-10);

    var vv_db = 10.0 * Math.log10(vv);
    var vh_db = 10.0 * Math.log10(vh);
    var cr = vh / vv;
    var rvi = 4.0 * vh / (vv + vh);

    return {
        vv:  [vv_db],
        vh:  [vh_db],
        cr:  [cr],
        rvi: [rvi],
        dataMask: [mask]
    };
}
""".strip()

_GEOD = Geod(ellps="WGS84")
_KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


# -- Helpers -------------------------------------------------------------------

def expand_window(day_start: int, day_end: int,
                  min_days: int) -> tuple[int, int]:
    """Expand a stage window symmetrically so it covers >= min_days."""
    span = day_end - day_start
    if span >= min_days:
        return day_start, day_end
    gap = min_days - span
    half = gap // 2
    return day_start - half, day_end + (gap - half)


def parse_kml_polygon(kml_path: str) -> list[list[float]] | None:
    try:
        tree = ET.parse(kml_path)
        coord_node = tree.getroot().find(".//kml:coordinates", _KML_NS)
        if coord_node is None:
            return None
        raw = coord_node.text.strip()
        coords = [
            [float(p.split(",")[0]), float(p.split(",")[1])]
            for p in raw.split()
            if len(p.split(",")) >= 2
        ]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        return coords if len(coords) >= 4 else None
    except Exception:
        return None


def parse_kml_content(kml_content: str | bytes) -> list[list[float]] | None:
    """Parse KML polygon from in-memory string/bytes (for API use)."""
    try:
        if isinstance(kml_content, bytes):
            kml_content = kml_content.decode("utf-8")
        root = ET.fromstring(kml_content)
        coord_node = root.find(".//kml:coordinates", _KML_NS)
        if coord_node is None:
            return None
        raw = coord_node.text.strip()
        coords = [
            [float(p.split(",")[0]), float(p.split(",")[1])]
            for p in raw.split()
            if len(p.split(",")) >= 2
        ]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        return coords if len(coords) >= 4 else None
    except Exception:
        return None


def polygon_centroid(coords: list[list[float]]) -> tuple[float, float]:
    n = len(coords)
    lon = sum(c[0] for c in coords) / n
    lat = sum(c[1] for c in coords) / n
    return lat, lon


def polygon_area_hectares(coords: list[list[float]]) -> float:
    poly = ShapelyPolygon(coords)
    area_m2, _ = _GEOD.geometry_area_perimeter(poly)
    return abs(area_m2) / 10_000


def _normalize_crop_name(raw: str) -> str | None:
    clean = raw.upper().strip()
    if clean in CROP_NAME_NORMALIZE:
        return CROP_NAME_NORMALIZE[clean]
    ascii_only = "".join(ch for ch in clean if ord(ch) < 128)
    for canonical in CROP_NAME_NORMALIZE.values():
        if ascii_only.startswith(canonical[:3]) and len(ascii_only) >= 3:
            return canonical
    return None

def parse_metadata_from_filename(filename: str) -> dict | None:
    match = FILENAME_REGEX.match(filename)
    if not match:
        return None
    crop_raw, raw_id, plantio_str, colheita_str = match.groups()
    try:
        dt_plantio = datetime.strptime(plantio_str, "%d-%m-%y")
        dt_colheita = datetime.strptime(colheita_str, "%d-%m-%y")
    except ValueError:
        return None
    crop_key = _normalize_crop_name(crop_raw)
    if crop_key is None:
        return None
    return {
        "crop_label": crop_key,
        "field_id": f"{crop_key}_{raw_id}",
        "planting_date": dt_plantio,
        "harvest_date": dt_colheita,
    }


def compute_zonal_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {s: np.nan for s in STATS}
    return {
        "mean":   float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "std":    float(np.nanstd(values)),
        "p10":    float(np.nanpercentile(values, 10)),
        "p90":    float(np.nanpercentile(values, 90)),
    }


# -- Pipeline ------------------------------------------------------------------

class PhenologyFeaturePipeline:

    def __init__(
        self,
        sentinel_service: SentinelHubService,
        output_dir: str = "output",
        request_delay: float = 0.3,
        max_workers: int = 2,
        use_stats_api: bool = True,
        fetch_sar: bool = True,
    ):
        self.service = sentinel_service
        self.output_dir = output_dir
        self.request_delay = request_delay
        self.max_workers = max_workers
        self.use_stats_api = use_stats_api
        self.fetch_sar = fetch_sar

        self._token: str | None = None
        self._token_time: float = 0.0
        self._token_lock = threading.Lock()

        os.makedirs(self.output_dir, exist_ok=True)

    # -- Token refresh --------------------------------------------------------

    def _get_token(self) -> str:
        with self._token_lock:
            if self._token is None or time.time() - self._token_time > 250:
                logger.info("Refreshing Sentinel Hub token...")
                self._token = self.service.autenticar()
                self._token_time = time.time()
            return self._token

    # -- Adaptive resolution --------------------------------------------------

    @staticmethod
    def _compute_resolution(
        coords: list[list[float]], target_m_per_px: int = 10,
    ) -> tuple[int, int]:
        poly = ShapelyPolygon(coords)
        minx, miny, maxx, maxy = poly.bounds
        mid_lat = (miny + maxy) / 2.0
        m_per_deg_lon = 111_320 * math.cos(math.radians(mid_lat))
        m_per_deg_lat = 110_540
        w_m = (maxx - minx) * m_per_deg_lon
        h_m = (maxy - miny) * m_per_deg_lat
        width = max(32, min(2500, int(w_m / target_m_per_px)))
        height = max(32, min(2500, int(h_m / target_m_per_px)))
        return width, height

    # -- Optical: Process API (raster download) -------------------------------

    def _build_process_payload(self, coords, date_start, date_end, width, height):
        return {
            "input": {
                "bounds": {
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                },
                "data": [{
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date_start}T00:00:00Z",
                            "to": f"{date_end}T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                    "type": "sentinel-2-l2a",
                }],
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [
                    {"identifier": "ndvi_tiff", "format": {"type": "image/tiff"}},
                    {"identifier": "evi_tiff",  "format": {"type": "image/tiff"}},
                    {"identifier": "ndwi_tiff", "format": {"type": "image/tiff"}},
                    {"identifier": "scl_tiff",  "format": {"type": "image/tiff"}},
                ],
            },
            "evalscript": EVALSCRIPT_PROCESS,
        }

    def _fetch_stage_rasters(self, coords, date_start, date_end, width, height):
        token = self._get_token()
        payload = self._build_process_payload(coords, date_start, date_end, width, height)
        headers = {
            "Authorization": token,
            "Content-Type": "application/json",
            "Accept": "application/tar",
        }
        try:
            resp = http_requests.post(
                f"{self.service.url_base}/api/v1/process",
                timeout=60, headers=headers, data=json.dumps(payload),
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Process API failed [%s to %s]: %s", date_start, date_end, e)
            return None

        rasters = {}
        try:
            with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r") as tar:
                for member in tar.getmembers():
                    if not (member.isfile() and member.name.endswith(".tif")):
                        continue
                    key = member.name.split("_")[0]
                    raw = tar.extractfile(member).read()
                    img = tifffile.imread(io.BytesIO(raw))
                    if img.ndim == 3:
                        img = img[0] if img.shape[0] < img.shape[-1] else img[:, :, 0]
                    rasters[key] = img
        except Exception as e:
            logger.warning("Tar parse failed [%s to %s]: %s", date_start, date_end, e)
            return None

        if not {"ndvi", "evi", "ndwi", "scl"}.issubset(rasters):
            return None
        return rasters

    @staticmethod
    def _compute_features_from_rasters(rasters):
        scl = rasters["scl"].astype(np.uint8)
        clear_mask = np.isin(scl, list(CLEAR_SCL))
        total_px = scl.size
        usable_px = int(clear_mask.sum())
        usable_pct = usable_px / total_px if total_px > 0 else 0.0

        features = {}
        for idx_name, rk in [("NDVI", "ndvi"), ("NDWI", "ndwi"), ("EVI", "evi")]:
            arr = rasters[rk].astype(np.float32)
            valid = arr[clear_mask & ~np.isnan(arr)]
            for stat_name, val in compute_zonal_stats(valid).items():
                features[f"{idx_name}_{stat_name}"] = val

        return features, usable_pct

    # -- Optical: Statistical API ---------------------------------------------

    def _build_optical_stats_payload(self, coords, date_start, date_end):
        dt_start = datetime.strptime(date_start, "%Y-%m-%d")
        dt_end = datetime.strptime(date_end, "%Y-%m-%d")
        interval_days = max(1, (dt_end - dt_start).days)

        return {
            "input": {
                "bounds": {
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326",
                    },
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date_start}T00:00:00Z",
                            "to": f"{date_end}T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                }],
            },
            "aggregation": {
                "timeRange": {
                    "from": f"{date_start}T00:00:00Z",
                    "to": f"{date_end}T23:59:59Z",
                },
                "aggregationInterval": {"of": f"P{interval_days}D"},
                "resx": 10,
                "resy": 10,
                "evalscript": EVALSCRIPT_OPTICAL_STATS,
            },
            "calculations": {
                "default": {
                    "statistics": {
                        "default": {
                            "percentiles": {"k": [10, 50, 90]},
                        }
                    }
                }
            },
        }

    def _fetch_stats_api(self, payload, date_start, date_end, _retries=3):
        token = self._get_token()
        headers = {
            "Authorization": token,
            "Content-Type": "application/json",
        }

        for attempt in range(_retries):
            try:
                resp = http_requests.post(
                    f"{self.service.url_base}/api/v1/statistics",
                    timeout=60, headers=headers, data=json.dumps(payload),
                )
                if resp.status_code in (403, 404):
                    logger.warning(
                        "Statistical API HTTP %d [%s to %s]",
                        resp.status_code, date_start, date_end,
                    )
                    return None
                if resp.status_code in (400, 429, 500, 502, 503) and attempt < _retries - 1:
                    wait = 2 ** attempt + 0.5
                    logger.info(
                        "    Stats API HTTP %d, retry %d/%d in %.1fs...",
                        resp.status_code, attempt + 1, _retries, wait,
                    )
                    time.sleep(wait)
                    token = self._get_token()
                    headers["Authorization"] = token
                    continue
                resp.raise_for_status()
            except http_requests.exceptions.HTTPError as e:
                logger.warning("Stats API HTTP error [%s to %s]: %s", date_start, date_end, e)
                return None
            except Exception as e:
                if attempt < _retries - 1:
                    time.sleep(2 ** attempt + 0.5)
                    continue
                logger.warning("Stats API error [%s to %s]: %s", date_start, date_end, e)
                return None

            return resp.json()

        return None

    def _fetch_optical_stats(self, coords, date_start, date_end):
        payload = self._build_optical_stats_payload(coords, date_start, date_end)
        body = self._fetch_stats_api(payload, date_start, date_end)
        if body is None:
            return None
        try:
            return self._parse_optical_stats(body)
        except Exception as e:
            logger.warning("Optical stats parse error: %s", e)
            return None

    @staticmethod
    def _parse_optical_stats(body):
        data = body.get("data", [])
        if not data:
            return None

        outputs = data[0].get("outputs", {})
        features = {}
        total_samples = 0
        total_nodata = 0

        for idx_name, key in [("NDVI", "ndvi"), ("NDWI", "ndwi"), ("EVI", "evi")]:
            stats = (
                outputs.get(key, {})
                .get("bands", {})
                .get("B0", {})
                .get("stats", {})
            )
            sample_count = stats.get("sampleCount", 0)
            nodata_count = stats.get("noDataCount", 0)
            total_samples += sample_count
            total_nodata += nodata_count

            if sample_count == 0:
                for s in STATS:
                    features[f"{idx_name}_{s}"] = np.nan
                continue

            pcts = stats.get("percentiles", {})
            features[f"{idx_name}_mean"]   = stats.get("mean", np.nan)
            features[f"{idx_name}_median"] = pcts.get("50.0", np.nan)
            features[f"{idx_name}_std"]    = stats.get("stDev", np.nan)
            features[f"{idx_name}_p10"]    = pcts.get("10.0", np.nan)
            features[f"{idx_name}_p90"]    = pcts.get("90.0", np.nan)

        grand_total = total_samples + total_nodata
        usable_pct = total_samples / grand_total if grand_total > 0 else 0.0
        return features, usable_pct

    # -- SAR: Statistical API (Sentinel-1 GRD) --------------------------------

    def _build_sar_stats_payload(self, coords, date_start, date_end):
        dt_start = datetime.strptime(date_start, "%Y-%m-%d")
        dt_end = datetime.strptime(date_end, "%Y-%m-%d")
        interval_days = max(1, (dt_end - dt_start).days)

        return {
            "input": {
                "bounds": {
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326",
                    },
                },
                "data": [{
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date_start}T00:00:00Z",
                            "to": f"{date_end}T23:59:59Z",
                        },
                    },
                    "processing": {
                        "backCoeff": "GAMMA0_TERRAIN",
                        "orthorectify": True,
                    },
                }],
            },
            "aggregation": {
                "timeRange": {
                    "from": f"{date_start}T00:00:00Z",
                    "to": f"{date_end}T23:59:59Z",
                },
                "aggregationInterval": {"of": f"P{interval_days}D"},
                "resx": 10,
                "resy": 10,
                "evalscript": EVALSCRIPT_SAR_STATS,
            },
            "calculations": {
                "default": {
                    "statistics": {
                        "default": {
                            "percentiles": {"k": [10, 50, 90]},
                        }
                    }
                }
            },
        }

    def _fetch_sar_stats(self, coords, date_start, date_end):
        payload = self._build_sar_stats_payload(coords, date_start, date_end)
        body = self._fetch_stats_api(payload, date_start, date_end)
        if body is None:
            return None
        try:
            return self._parse_sar_stats(body)
        except Exception as e:
            logger.warning("SAR stats parse error: %s", e)
            return None

    @staticmethod
    def _parse_sar_stats(body):
        data = body.get("data", [])
        if not data:
            return None

        outputs = data[0].get("outputs", {})
        features = {}
        total_samples = 0
        total_nodata = 0

        for idx_name, key in [("VV", "vv"), ("VH", "vh"), ("CR", "cr"), ("RVI", "rvi")]:
            stats = (
                outputs.get(key, {})
                .get("bands", {})
                .get("B0", {})
                .get("stats", {})
            )
            sample_count = stats.get("sampleCount", 0)
            nodata_count = stats.get("noDataCount", 0)
            total_samples += sample_count
            total_nodata += nodata_count

            if sample_count == 0:
                for s in STATS:
                    features[f"{idx_name}_{s}"] = np.nan
                continue

            pcts = stats.get("percentiles", {})
            features[f"{idx_name}_mean"]   = stats.get("mean", np.nan)
            features[f"{idx_name}_median"] = pcts.get("50.0", np.nan)
            features[f"{idx_name}_std"]    = stats.get("stDev", np.nan)
            features[f"{idx_name}_p10"]    = pcts.get("10.0", np.nan)
            features[f"{idx_name}_p90"]    = pcts.get("90.0", np.nan)

        grand_total = total_samples + total_nodata
        usable_pct = total_samples / grand_total if grand_total > 0 else 0.0
        return features, usable_pct

    # -- Stage dispatch -------------------------------------------------------

    def _process_stage(self, coords, stage_name, base_date, stage_windows,
                       width, height, save_tiffs, field_meta, stagger=0.0):
        if stagger > 0:
            time.sleep(stagger)

        day_start, day_end = stage_windows[stage_name]

        # -- Optical (expanded window for cloud-free coverage) -----------
        opt_day_start, opt_day_end = expand_window(
            day_start, day_end, MIN_OPTICAL_WINDOW_DAYS,
        )
        opt_window_start = base_date + timedelta(days=opt_day_start)
        opt_window_end = base_date + timedelta(days=opt_day_end)
        ds = opt_window_start.strftime("%Y-%m-%d")
        de = opt_window_end.strftime("%Y-%m-%d")

        optical_result = None

        if self.use_stats_api and not save_tiffs:
            optical_result = self._fetch_optical_stats(coords, ds, de)

        if optical_result is None:
            rasters = self._fetch_stage_rasters(coords, ds, de, width, height)
            if rasters is not None:
                if save_tiffs:
                    tiff_dir = os.path.join(
                        self.output_dir, "processed",
                        str(base_date.year), field_meta["crop_label"].lower(),
                        field_meta["field_id"], stage_name,
                    )
                    os.makedirs(tiff_dir, exist_ok=True)
                    for k in ("ndvi", "evi", "ndwi"):
                        tifffile.imwrite(
                            os.path.join(tiff_dir, f"{k}.tif"),
                            rasters[k].astype(np.float32),
                        )
                optical_result = self._compute_features_from_rasters(rasters)

        # -- SAR (expanded window to guarantee S1 coverage) --------------
        sar_result = None
        if self.fetch_sar:
            sar_day_start, sar_day_end = expand_window(
                day_start, day_end, MIN_SAR_WINDOW_DAYS,
            )
            sar_window_start = base_date + timedelta(days=sar_day_start)
            sar_window_end = base_date + timedelta(days=sar_day_end)
            sar_ds = sar_window_start.strftime("%Y-%m-%d")
            sar_de = sar_window_end.strftime("%Y-%m-%d")
            sar_result = self._fetch_sar_stats(coords, sar_ds, sar_de)

        # -- Merge --------------------------------------------------------
        merged_features = {}
        usable_pct = 0.0

        if optical_result is not None:
            opt_features, usable_pct = optical_result
            merged_features.update(opt_features)

        if sar_result is not None:
            sar_features, sar_pct = sar_result
            merged_features.update(sar_features)

        if not merged_features:
            return stage_name, None

        return stage_name, (merged_features, usable_pct)

    # -- Field processing -----------------------------------------------------

    def _process_field(self, meta, coords, width, height, save_tiffs):
        stage_windows = CROP_STAGE_WINDOWS.get(meta["crop_label"], DEFAULT_STAGE_WINDOWS)
        base_date = meta["planting_date"]

        lat, lon = polygon_centroid(coords)

        row = {
            "field_id": meta["field_id"],
            "crop_label": meta["crop_label"],
            "planting_date": base_date.strftime("%Y-%m-%d"),
            "area_hectares": round(polygon_area_hectares(coords), 2),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
        }

        stages_covered = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._process_stage, coords, sn, base_date,
                    stage_windows, width, height, save_tiffs, meta,
                    stagger=i * 0.4,
                ): sn
                for i, sn in enumerate(STAGES)
            }

            for future in as_completed(futures):
                stage_name = futures[future]
                try:
                    _, result = future.result()
                except Exception as e:
                    logger.warning("    %s: exception -- %s", stage_name, e)
                    result = None

                if result is None:
                    for idx in OPTICAL_INDICES:
                        for stat in STATS:
                            row[f"{idx}_{stat}_{stage_name}"] = np.nan
                    if self.fetch_sar:
                        for idx in SAR_INDICES:
                            for stat in STATS:
                                row[f"{idx}_{stat}_{stage_name}"] = np.nan
                    logger.info("    %s: no data", stage_name)
                else:
                    features, usable_pct = result
                    has_data = any(
                        not np.isnan(float(v)) for k, v in features.items() if "mean" in k
                    )
                    if has_data:
                        stages_covered += 1
                    for fk, fv in features.items():
                        row[f"{fk}_{stage_name}"] = fv
                    sar_tag = " +SAR" if any(k.startswith(("VV_", "VH_")) for k in features) else ""
                    logger.info(
                        "    %s: usable %.0f%%%s %s",
                        stage_name, usable_pct * 100, sar_tag,
                        "OK" if has_data else "-> null",
                    )

        row["stages_covered"] = stages_covered
        return row

    # -- DB management --------------------------------------------------------

    @staticmethod
    def _init_db(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)

        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(phenology_features)")}

        if not existing_cols:
            cols_def = []
            for col in ALL_COLS:
                dtype = "TEXT" if col in TEXT_COLS else "REAL"
                cols_def.append(f'"{col}" {dtype}')
            cols_def.append('"sar_backfill_done" INTEGER DEFAULT 0')
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS phenology_features ({', '.join(cols_def)})"
            )
            conn.commit()
            return conn

        for col in ALL_COLS:
            if col not in existing_cols:
                dtype = "TEXT" if col in TEXT_COLS else "REAL"
                conn.execute(f'ALTER TABLE phenology_features ADD COLUMN "{col}" {dtype}')
                logger.info("Migrated: added column '%s'", col)

        if "sar_backfill_done" not in existing_cols:
            conn.execute('ALTER TABLE phenology_features ADD COLUMN "sar_backfill_done" INTEGER DEFAULT 0')
            logger.info("Migrated: added column 'sar_backfill_done'")

        conn.commit()
        return conn

    @staticmethod
    def _insert_row(conn: sqlite3.Connection, row: dict):
        placeholders = ", ".join(["?"] * len(ALL_COLS))
        cols_str = ", ".join([f'"{c}"' for c in ALL_COLS])
        sql = f"INSERT INTO phenology_features ({cols_str}) VALUES ({placeholders})"

        values = []
        for col in ALL_COLS:
            v = row.get(col)
            if col in TEXT_COLS:
                values.append(v)
            elif v is None:
                values.append(None)
            else:
                try:
                    f = float(v)
                    values.append(None if math.isnan(f) else f)
                except (TypeError, ValueError):
                    values.append(None)
        conn.execute(sql, values)
        conn.commit()

    @staticmethod
    def _load_existing_ids(db_path: str) -> set[str]:
        if not os.path.exists(db_path):
            return set()
        try:
            with sqlite3.connect(db_path) as conn:
                return {r[0] for r in conn.execute(
                    "SELECT field_id FROM phenology_features"
                )}
        except Exception:
            return set()

    # -- Main processing loop -------------------------------------------------

    def process_directory(
        self,
        root_dir: str,
        max_per_crop: int | None = None,
        save_tiffs: bool = False,
        width: int | None = None,
        height: int | None = None,
        planting_year: int | None = None,
    ) -> pd.DataFrame:
        db_path = os.path.join(self.output_dir, "features.db")

        existing_ids = self._load_existing_ids(db_path)
        if existing_ids:
            logger.info(
                "Resuming -- %d fields already in DB will be skipped",
                len(existing_ids),
            )

        conn = self._init_db(db_path)

        kml_files_raw = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.lower().endswith(".kml"):
                    kml_files_raw.append((dirpath, fn))

        def _planting_sort_key(item):
            _, fn = item
            meta = parse_metadata_from_filename(fn)
            if meta is None:
                return (1, datetime.min)
            year = meta["planting_date"].year
            return (0 if year >= 2025 else 1, meta["planting_date"])

        kml_files = sorted(kml_files_raw, key=_planting_sort_key)

        year_tag = f" | filtering planting_year={planting_year}" if planting_year else ""
        logger.info("Found %d KML files in %s (2025+ prioritized%s)", len(kml_files), root_dir, year_tag)

        crop_counts: dict[str, int] = {}
        processed = 0

        for file_idx, (dirpath, fn) in enumerate(kml_files, 1):
            meta = parse_metadata_from_filename(fn)
            if meta is None:
                continue

            crop = meta["crop_label"]

            if meta["field_id"] in existing_ids:
                continue

            if planting_year is not None and meta["planting_date"].year != planting_year:
                continue

            if max_per_crop is not None and crop_counts.get(crop, 0) >= max_per_crop:
                continue

            kml_path = os.path.join(dirpath, fn)
            coords = parse_kml_polygon(kml_path)
            if coords is None:
                logger.warning("Skipping %s -- no valid polygon", fn)
                continue

            if width is not None and height is not None:
                w, h = width, height
            else:
                w, h = self._compute_resolution(coords)

            logger.info(
                "[%d/%d] %s | %s | %dx%d px",
                file_idx, len(kml_files), meta["field_id"], crop, w, h,
            )

            row = self._process_field(meta, coords, w, h, save_tiffs)

            self._insert_row(conn, row)
            existing_ids.add(meta["field_id"])
            processed += 1
            crop_counts[crop] = crop_counts.get(crop, 0) + 1

            logger.info(
                "  -> %d/%d stages | written to DB",
                row["stages_covered"], len(STAGES),
            )

            time.sleep(self.request_delay)

        conn.close()

        with sqlite3.connect(db_path) as read_conn:
            df = pd.read_sql("SELECT * FROM phenology_features", read_conn)

        logger.info(
            "Done -- %d new + %d resumed = %d total rows, %d columns",
            processed, len(df) - processed, len(df), len(df.columns),
        )
        return df

    # -- SAR Backfill ---------------------------------------------------------
    # Iterates existing rows that have NOT been attempted yet (sar_backfill_done=0
    # AND first SAR col IS NULL), fetches Sentinel-1 with expanded windows,
    # and marks every row as done regardless of whether data was found.

    def backfill_sar(self, kml_root: str):
        db_path = os.path.join(self.output_dir, "features.db")
        conn = self._init_db(db_path)

        first_sar_col = f"VV_mean_{STAGES[0]}"
        rows = conn.execute(
            f'SELECT rowid, field_id, crop_label, planting_date '
            f'FROM phenology_features '
            f'WHERE "{first_sar_col}" IS NULL '
            f'AND (sar_backfill_done IS NULL OR sar_backfill_done = 0)'
        ).fetchall()

        if not rows:
            logger.info("No rows need SAR backfill -- all populated or already attempted")
            conn.close()
            return

        logger.info("SAR backfill: %d rows to process", len(rows))

        kml_index = {}
        for dirpath, _, filenames in os.walk(kml_root):
            for fn in filenames:
                if fn.lower().endswith(".kml"):
                    meta = parse_metadata_from_filename(fn)
                    if meta:
                        kml_index[meta["field_id"]] = os.path.join(dirpath, fn)

        logger.info("KML index: %d files", len(kml_index))

        sar_cols = SAR_FEATURE_COLS
        update_sql = (
            "UPDATE phenology_features SET "
            + ", ".join([f'"{c}" = ?' for c in sar_cols])
            + ', "sar_backfill_done" = 1'
            + " WHERE rowid = ?"
        )

        updated = 0
        skipped = 0

        for i, (rowid, field_id, crop_label, planting_date_str) in enumerate(rows, 1):
            kml_path = kml_index.get(field_id)
            if kml_path is None:
                skipped += 1
                continue

            coords = parse_kml_polygon(kml_path)
            if coords is None:
                skipped += 1
                continue

            planting_date = datetime.strptime(planting_date_str, "%Y-%m-%d")
            stage_windows = CROP_STAGE_WINDOWS.get(crop_label, DEFAULT_STAGE_WINDOWS)

            logger.info(
                "[%d/%d] SAR backfill: %s | %s",
                i, len(rows), field_id, crop_label,
            )

            sar_features = {}
            any_data = False

            for stage_name in STAGES:
                day_start, day_end = stage_windows[stage_name]
                sar_day_start, sar_day_end = expand_window(day_start, day_end, MIN_SAR_WINDOW_DAYS)

                window_start = planting_date + timedelta(days=sar_day_start)
                window_end = planting_date + timedelta(days=sar_day_end)
                ds = window_start.strftime("%Y-%m-%d")
                de = window_end.strftime("%Y-%m-%d")

                expanded = (sar_day_start != day_start or sar_day_end != day_end)
                win_label = f"{ds} to {de}"
                if expanded:
                    orig_span = day_end - day_start
                    new_span = sar_day_end - sar_day_start
                    win_label += f" (expanded {orig_span}d -> {new_span}d)"

                result = self._fetch_sar_stats(coords, ds, de)

                if result is not None:
                    features, sar_pct = result
                    for fk, fv in features.items():
                        sar_features[f"{fk}_{stage_name}"] = fv
                    any_data = True
                    logger.info("    %s: SAR OK (%.0f%%) | %s", stage_name, sar_pct * 100, win_label)
                else:
                    for idx in SAR_INDICES:
                        for stat in STATS:
                            sar_features[f"{idx}_{stat}_{stage_name}"] = np.nan
                    logger.info("    %s: SAR no data | %s", stage_name, win_label)

                time.sleep(self.request_delay)

            values = []
            for col in sar_cols:
                v = sar_features.get(col)
                if v is None:
                    values.append(None)
                else:
                    try:
                        f = float(v)
                        values.append(None if math.isnan(f) else f)
                    except (TypeError, ValueError):
                        values.append(None)
            values.append(rowid)

            conn.execute(update_sql, values)
            conn.commit()
            updated += 1

            if any_data:
                logger.info("  -> SAR written")
            else:
                logger.info("  -> no SAR data (marked done, won't retry)")

        conn.close()
        logger.info("SAR backfill complete -- %d updated, %d skipped (no KML)", updated, skipped)


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phenology Feature Pipeline v4 (expanded optical + SAR windows)")
    parser.add_argument("--mode", choices=["full", "backfill-sar"], default="full",
                        help="'full' = process new KMLs (optical+SAR), "
                             "'backfill-sar' = add SAR to existing rows with expanded windows")
    parser.add_argument("--max-per-crop", type=int, default=500)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--year", type=int, default=None,
                        help="Only process fields with this planting year (e.g. 2025)")
    parser.add_argument("--no-sar", action="store_true",
                        help="Skip SAR fetching in full mode")
    args = parser.parse_args()

    service = SentinelHubService()
    kml_root = os.path.join("src", "data", "dataset_split", "train")
    output_dir = os.path.join("src", "data", "features")

    pipeline = PhenologyFeaturePipeline(
        service,
        output_dir=output_dir,
        max_workers=args.max_workers,
        use_stats_api=True,
        fetch_sar=not args.no_sar,
    )

    if args.mode == "backfill-sar":
        pipeline.backfill_sar(kml_root)
    else:
        df = pipeline.process_directory(
            root_dir=kml_root,
            max_per_crop=args.max_per_crop,
            save_tiffs=False,
            planting_year=args.year,
        )

        if not df.empty:
            print(f"\nShape: {df.shape}")
            print(f"Columns ({len(df.columns)}):")
            for c in df.columns:
                print(f"  {c}")
            print(f"\nstages_covered distribution:")
            print(df["stages_covered"].value_counts().sort_index().to_string())
            print(f"\nNull rate per stage (optical):")
            for stage in STAGES:
                stage_cols = [c for c in df.columns
                              if c.endswith(f"_{stage}") and c.startswith(("NDVI", "NDWI", "EVI"))]
                if stage_cols:
                    null_rate = df[stage_cols].isna().all(axis=1).mean()
                    print(f"  {stage}: {null_rate:.0%} fields fully null")
            print(f"\nNull rate per stage (SAR):")
            for stage in STAGES:
                stage_cols = [c for c in df.columns
                              if c.endswith(f"_{stage}") and c.startswith(("VV_", "VH_", "CR_", "RVI_"))]
                if stage_cols:
                    null_rate = df[stage_cols].isna().all(axis=1).mean()
                    print(f"  {stage}: {null_rate:.0%} fields fully null")
