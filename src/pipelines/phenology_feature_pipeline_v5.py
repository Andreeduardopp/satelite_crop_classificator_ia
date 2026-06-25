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

# Optical index -> evalscript output id (lowercase). Beyond the broadband trio,
# red-edge indices (NDRE, CIRE, MTCI) track chlorophyll/LAI without the NDVI
# saturation that makes spectrally-similar cereals (AVEIA vs TRIGO) inseparable;
# PSRI captures senescence timing and NDMI canopy-water/drydown signal.
OPTICAL_INDEX_KEYS = {
    "NDVI": "ndvi",
    "NDWI": "ndwi",
    "EVI":  "evi",
    "NDRE": "ndre",
    "CIRE": "cire",
    "MTCI": "mtci",
    "PSRI": "psri",
    "NDMI": "ndmi",
}
OPTICAL_INDICES = list(OPTICAL_INDEX_KEYS)
# Indices that are unbounded ratios; clamped server-side, so no Python clip needed.
STATS = ["mean", "median", "std", "p10", "p90"]

SAR_INDICES = ["VV", "VH", "CR", "RVI"]

# Sentinel-1 revisits ~12 days in Brazil; 24-day minimum guarantees >= 1 pass
MIN_SAR_WINDOW_DAYS = 24

# Sentinel-2 revisits ~5 days; 20-day minimum guarantees ~4 passes,
# greatly increasing the chance of at least one cloud-free scene.
MIN_OPTICAL_WINDOW_DAYS = 20

CLEAR_SCL = frozenset({2, 4, 5, 6, 7})

# -- Option A: multi-temporal compositing -------------------------------------
# Sub-interval length for the Stats API: P5D yields one aggregate per Sentinel-2
# revisit cycle. Multiple bins are then merged, keeping only cloud-light ones.
MULTI_COMPOSITE_INTERVAL_DAYS = 5

# A 5-day bin must have at least this fraction of valid (non-masked) pixels to
# be included in the weighted aggregate; below this it is discarded as too cloudy.
MIN_USABLE_PCT_PER_BIN = 0.20

# -- Option B: adaptive window retry ------------------------------------------
# If the best usable_pct after the initial fetch (and all bins) is still below
# this threshold, the stage is retried with progressively wider date windows.
CLOUD_RETRY_THRESHOLD = 0.15

# Successive minimum window spans (days) tried on each retry pass.
CLOUD_RETRY_WINDOWS = [40, 60]

# -----------------------------------------------------------------------------

FIXED_COLS = [
    "field_id", "crop_label", "planting_date",
    "area_hectares", "latitude", "longitude", "stages_covered",
    "interpolated_stages",   # Option D: comma-separated list of interpolated stage names
]
TEXT_COLS = {"field_id", "crop_label", "planting_date", "interpolated_stages"}

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

# Indices added after the original NDVI/NDWI/EVI set. The 'backfill-optical' mode
# populates only these columns for existing rows — it does not re-fetch SAR or
# rewrite the original optical indices.
BACKFILL_OPTICAL_INDICES = ["NDRE", "CIRE", "MTCI", "PSRI", "NDMI"]
BACKFILL_OPTICAL_FEATURE_COLS = [
    f"{idx}_{stat}_{stage}"
    for idx in BACKFILL_OPTICAL_INDICES
    for stage in STAGES
    for stat in STATS
]

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

# colheita part accepts either a date (dd-mm-yy) or the literal "nan" —
# harvest_date is not used for stage windows so colheita_nan is recoverable.
FILENAME_REGEX = re.compile(
    r"(.+?)_(.+)_plantio_([\w-]+)_colheita_([\w-]+)\.kml",
    re.IGNORECASE,
)

CROP_NAME_NORMALIZE = {
    "SOJA": "SOJA", "MILHO": "MILHO", "ARROZ": "ARROZ",
    "TRIGO": "TRIGO", "AVEIA": "AVEIA",
    "CAFE": "CAFE", "CAFÉ": "CAFE",
    "FEIJAO": "FEIJAO", "FEIJÃO": "FEIJAO",
}

# -- Evalscripts --------------------------------------------------------------

EVALSCRIPT_PROCESS = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "SCL", "dataMask"],
        output: [
            { id: "ndvi_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "evi_tiff",  bands: 1, sampleType: "FLOAT32" },
            { id: "ndwi_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "ndre_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "cire_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "mtci_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "psri_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "ndmi_tiff", bands: 1, sampleType: "FLOAT32" },
            { id: "scl_tiff",  bands: 1, sampleType: "UINT8" }
        ]
    };
}

function safeDiv(a, b) { return (b === 0.0) ? 0.0 : a / b; }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

function evaluatePixel(sample) {
    if (sample.dataMask === 0) {
        return {
            ndvi_tiff: [NaN], evi_tiff: [NaN], ndwi_tiff: [NaN],
            ndre_tiff: [NaN], cire_tiff: [NaN], mtci_tiff: [NaN],
            psri_tiff: [NaN], ndmi_tiff: [NaN], scl_tiff: [0]
        };
    }

    let ndvi = index(sample.B08, sample.B04);
    let ndwi = index(sample.B03, sample.B08);
    let ndre = index(sample.B8A, sample.B05);   // red-edge chlorophyll
    let ndmi = index(sample.B08, sample.B11);   // canopy water / drydown

    let denom = sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0;
    let evi = (denom === 0.0) ? 0.0 : 2.5 * (sample.B08 - sample.B04) / denom;

    let cire = clamp(safeDiv(sample.B07, sample.B05) - 1.0, -1.0, 20.0);             // chlorophyll index red-edge
    let mtci = clamp(safeDiv(sample.B06 - sample.B05, sample.B05 - sample.B04), -10.0, 20.0);  // MERIS terrestrial chlorophyll
    let psri = clamp(safeDiv(sample.B04 - sample.B02, sample.B06), -2.0, 2.0);       // plant senescence reflectance

    return {
        ndvi_tiff: [ndvi], evi_tiff: [evi], ndwi_tiff: [ndwi],
        ndre_tiff: [ndre], cire_tiff: [cire], mtci_tiff: [mtci],
        psri_tiff: [psri], ndmi_tiff: [ndmi],
        scl_tiff: [Math.round(sample.SCL)]
    };
}
""".strip()

EVALSCRIPT_OPTICAL_STATS = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "SCL", "dataMask"],
        output: [
            { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
            { id: "evi",  bands: 1, sampleType: "FLOAT32" },
            { id: "ndwi", bands: 1, sampleType: "FLOAT32" },
            { id: "ndre", bands: 1, sampleType: "FLOAT32" },
            { id: "cire", bands: 1, sampleType: "FLOAT32" },
            { id: "mtci", bands: 1, sampleType: "FLOAT32" },
            { id: "psri", bands: 1, sampleType: "FLOAT32" },
            { id: "ndmi", bands: 1, sampleType: "FLOAT32" },
            { id: "dataMask", bands: 1 }
        ]
    };
}

function safeDiv(a, b) { return (b === 0.0) ? 0.0 : a / b; }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

function evaluatePixel(sample) {
    var scl = sample.SCL;
    var isClear = (scl == 2 || scl == 4 || scl == 5 || scl == 6 || scl == 7);
    var mask = (sample.dataMask && isClear) ? 1 : 0;

    var ndvi = index(sample.B08, sample.B04);
    var ndwi = index(sample.B03, sample.B08);
    var ndre = index(sample.B8A, sample.B05);   // red-edge chlorophyll
    var ndmi = index(sample.B08, sample.B11);   // canopy water / drydown

    var denom = sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0;
    var evi = (denom === 0.0) ? 0.0 : 2.5 * (sample.B08 - sample.B04) / denom;

    var cire = clamp(safeDiv(sample.B07, sample.B05) - 1.0, -1.0, 20.0);             // chlorophyll index red-edge
    var mtci = clamp(safeDiv(sample.B06 - sample.B05, sample.B05 - sample.B04), -10.0, 20.0);  // MERIS terrestrial chlorophyll
    var psri = clamp(safeDiv(sample.B04 - sample.B02, sample.B06), -2.0, 2.0);       // plant senescence reflectance

    return {
        ndvi: [ndvi], evi: [evi], ndwi: [ndwi],
        ndre: [ndre], cire: [cire], mtci: [mtci],
        psri: [psri], ndmi: [ndmi],
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

    # plantio_nan → no planting date → stage windows can't be computed → skip
    if plantio_str.lower() == "nan":
        logger.debug("Skipping %s — plantio_nan (no planting date)", filename)
        return None

    try:
        dt_plantio = datetime.strptime(plantio_str, "%d-%m-%y")
    except ValueError:
        logger.debug("Skipping %s — unparseable planting date: %s", filename, plantio_str)
        return None

    # colheita_nan is recoverable — harvest_date is never used for stage windows
    dt_colheita: datetime | None = None
    if colheita_str.lower() != "nan":
        try:
            dt_colheita = datetime.strptime(colheita_str, "%d-%m-%y")
        except ValueError:
            logger.debug("Skipping %s — unparseable harvest date: %s", filename, colheita_str)
            return None

    crop_key = _normalize_crop_name(crop_raw)
    if crop_key is None:
        logger.debug("Skipping %s — unrecognised crop name: %s", filename, crop_raw)
        return None

    return {
        "crop_label": crop_key,
        "field_id": f"{crop_key}_{raw_id}",
        "planting_date": dt_plantio,
        "harvest_date": dt_colheita,   # None when colheita_nan
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
                    {"identifier": "ndre_tiff", "format": {"type": "image/tiff"}},
                    {"identifier": "cire_tiff", "format": {"type": "image/tiff"}},
                    {"identifier": "mtci_tiff", "format": {"type": "image/tiff"}},
                    {"identifier": "psri_tiff", "format": {"type": "image/tiff"}},
                    {"identifier": "ndmi_tiff", "format": {"type": "image/tiff"}},
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
        for idx_name, rk in OPTICAL_INDEX_KEYS.items():
            arr_src = rasters.get(rk)
            if arr_src is None:
                for stat_name in STATS:
                    features[f"{idx_name}_{stat_name}"] = np.nan
                continue
            arr = arr_src.astype(np.float32)
            if idx_name == "EVI":
                arr = np.clip(arr, -1.0, 1.0)
            valid = arr[clear_mask & ~np.isnan(arr)]
            for stat_name, val in compute_zonal_stats(valid).items():
                features[f"{idx_name}_{stat_name}"] = val

        return features, usable_pct

    # -- Optical: Statistical API ---------------------------------------------

    def _build_optical_stats_payload(self, coords, date_start, date_end):
        # Option A: use P5D sub-intervals instead of one aggregate over the full
        # window. The API returns one stats block per 5-day bin; _parse_optical_stats
        # selects and merges cloud-light bins.
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
                        # mosaickingOrder is a Process API concept; omitting it here
                        # lets the Stats API aggregate all clear pixels across all
                        # scenes in the window via the evalscript dataMask.
                    },
                }],
            },
            "aggregation": {
                "timeRange": {
                    "from": f"{date_start}T00:00:00Z",
                    "to": f"{date_end}T23:59:59Z",
                },
                "aggregationInterval": {"of": f"P{MULTI_COMPOSITE_INTERVAL_DAYS}D"},
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
                # 400/403/404 are permanent client errors — retrying never helps.
                # Log the API error body so we can diagnose the root cause.
                if resp.status_code in (400, 403, 404):
                    try:
                        detail = resp.json().get("error", {}).get("message", "")
                    except Exception:
                        detail = resp.text[:200]
                    logger.warning(
                        "Statistical API HTTP %d [%s to %s]: %s",
                        resp.status_code, date_start, date_end, detail,
                    )
                    return None
                if resp.status_code in (429, 500, 502, 503) and attempt < _retries - 1:
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
        """
        Option A: parse all P5D bins from the Stats API response, select the
        cloud-light ones, and return a weighted aggregate of their statistics.

        Selection rule:
          - Any bin with usable_pct >= MIN_USABLE_PCT_PER_BIN is accepted.
          - If no bin passes the threshold, take the single best bin so that
            the stage is never left null solely due to threshold strictness.

        Aggregation:
          - mean : weighted mean (weight = sampleCount per bin)
          - std  : pooled standard deviation (accounts for between-bin variance)
          - p10 / median / p90 : weighted average across bins (approximation;
            exact combination requires raw data which the API does not return)

        Returns (features_dict, best_usable_pct) or None if no data at all.
        """
        def _sf(v, default: float = np.nan) -> float:
            """Safe-cast any API value to float; maps None/non-numeric to default."""
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        data = body.get("data", [])
        if not data:
            return None

        # --- parse each bin independently ---
        bin_stats = []  # (features_dict, sample_count, usable_pct)
        for bin_entry in data:
            outputs = bin_entry.get("outputs", {})
            bin_feats: dict[str, float] = {}
            bin_total_samples = 0
            bin_total_nodata = 0

            for idx_name, key in OPTICAL_INDEX_KEYS.items():
                stats = (
                    outputs.get(key, {})
                    .get("bands", {})
                    .get("B0", {})
                    .get("stats", {})
                )
                sample_count = int(_sf(stats.get("sampleCount"), 0))
                nodata_count = int(_sf(stats.get("noDataCount"), 0))
                bin_total_samples += sample_count
                bin_total_nodata += nodata_count

                if sample_count == 0:
                    for s in STATS:
                        bin_feats[f"{idx_name}_{s}"] = np.nan
                    continue

                pcts = stats.get("percentiles", {})
                bin_feats[f"{idx_name}_mean"]   = _sf(stats.get("mean"))
                bin_feats[f"{idx_name}_median"] = _sf(pcts.get("50.0"))
                bin_feats[f"{idx_name}_std"]    = _sf(stats.get("stDev"))
                bin_feats[f"{idx_name}_p10"]    = _sf(pcts.get("10.0"))
                bin_feats[f"{idx_name}_p90"]    = _sf(pcts.get("90.0"))

            grand_total = bin_total_samples + bin_total_nodata
            usable_pct = bin_total_samples / grand_total if grand_total > 0 else 0.0
            bin_stats.append((bin_feats, bin_total_samples, usable_pct))

        if not bin_stats:
            return None

        # --- select bins ---
        good_bins = [(f, n, u) for f, n, u in bin_stats if u >= MIN_USABLE_PCT_PER_BIN]
        if not good_bins:
            # nothing passes the threshold; take the least-cloudy bin as a fallback
            best = max(bin_stats, key=lambda x: x[2])
            if best[2] == 0.0:
                return None  # truly no data in any bin
            good_bins = [best]

        # --- weighted aggregate ---
        features: dict[str, float] = {}
        for idx_name in OPTICAL_INDICES:
            means, stds, p10s, medians, p90s, weights = [], [], [], [], [], []

            for feat_dict, n, _ in good_bins:
                m = feat_dict.get(f"{idx_name}_mean", np.nan)
                # m is already a float from _sf above; math.isnan is safe here
                if math.isnan(m) or n == 0:
                    continue
                # std: NaN means the API returned no variance; treat as 0
                s = feat_dict.get(f"{idx_name}_std", 0.0)
                s = 0.0 if math.isnan(s) else s
                # percentiles: fall back to mean if the API omitted them
                p10 = feat_dict.get(f"{idx_name}_p10", m)
                p10 = m if math.isnan(p10) else p10
                med = feat_dict.get(f"{idx_name}_median", m)
                med = m if math.isnan(med) else med
                p90 = feat_dict.get(f"{idx_name}_p90", m)
                p90 = m if math.isnan(p90) else p90

                means.append(m)
                stds.append(s)
                p10s.append(p10)
                medians.append(med)
                p90s.append(p90)
                weights.append(n)

            if not weights:
                for s in STATS:
                    features[f"{idx_name}_{s}"] = np.nan
                continue

            total_w = sum(weights)
            w_norm = [w / total_w for w in weights]

            grand_mean = sum(wi * mi for wi, mi in zip(w_norm, means))

            # pooled variance = sum(wi * (si^2 + (mi - grand_mean)^2))
            pooled_var = sum(
                wi * (si ** 2 + (mi - grand_mean) ** 2)
                for wi, si, mi in zip(w_norm, stds, means)
            )

            features[f"{idx_name}_mean"]   = grand_mean
            features[f"{idx_name}_std"]    = math.sqrt(max(pooled_var, 0.0))
            features[f"{idx_name}_p10"]    = sum(wi * v for wi, v in zip(w_norm, p10s))
            features[f"{idx_name}_median"] = sum(wi * v for wi, v in zip(w_norm, medians))
            features[f"{idx_name}_p90"]    = sum(wi * v for wi, v in zip(w_norm, p90s))

        # Clip EVI to valid physical range; atmospheric noise can push it well outside [-1, 1]
        for stat in STATS:
            k = f"EVI_{stat}"
            if k in features and not math.isnan(features[k]):
                features[k] = max(-1.0, min(1.0, features[k]))

        best_usable_pct = max(u for _, _, u in good_bins)
        return features, best_usable_pct

    # -- SAR: Statistical API (Sentinel-1 GRD) --------------------------------

    def _build_sar_stats_payload(self, coords, date_start, date_end,
                                 orbit_direction: str | None = None):
        """
        Build a Sentinel-1 GRD Stats API payload.

        orbit_direction: "ASCENDING", "DESCENDING", or None (both orbits).
        Specifying a direction is used by Option E as a fallback when the
        combined call returns no data.
        """
        dt_start = datetime.strptime(date_start, "%Y-%m-%d")
        dt_end = datetime.strptime(date_end, "%Y-%m-%d")
        interval_days = max(1, (dt_end - dt_start).days)

        data_filter: dict = {
            "timeRange": {
                "from": f"{date_start}T00:00:00Z",
                "to": f"{date_end}T23:59:59Z",
            },
        }
        if orbit_direction is not None:
            data_filter["orbitDirection"] = orbit_direction

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
                    "dataFilter": data_filter,
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

    def _fetch_sar_stats_for_orbit(self, coords, date_start, date_end,
                                   orbit_direction: str | None = None):
        """Single Stats API call for one orbit direction (or both when None)."""
        payload = self._build_sar_stats_payload(
            coords, date_start, date_end, orbit_direction,
        )
        body = self._fetch_stats_api(payload, date_start, date_end)
        if body is None:
            return None
        try:
            return self._parse_sar_stats(body)
        except Exception as e:
            logger.warning("SAR stats parse error [%s]: %s",
                           orbit_direction or "combined", e)
            return None

    def _fetch_sar_stats(self, coords, date_start, date_end):
        """
        Option E: dual-orbit SAR fetch.

        Strategy (zero extra API calls on the happy path):
          1. Combined call — no orbit filter; aggregates all available passes.
             S1 revisits Brazil every ~6 days when both orbits are counted.
          2. If combined returns null, retry ASCENDING and DESCENDING separately
             and take whichever orbit has data.  This covers regions where one
             orbit direction has a systematic coverage gap.
        """
        result = self._fetch_sar_stats_for_orbit(coords, date_start, date_end, None)
        if result is not None:
            return result

        logger.info("    SAR combined null — retrying per orbit direction")
        for direction in ("ASCENDING", "DESCENDING"):
            result = self._fetch_sar_stats_for_orbit(
                coords, date_start, date_end, direction,
            )
            if result is not None:
                logger.info("    SAR recovered via %s orbit", direction)
                return result

        return None

    @staticmethod
    def _parse_sar_stats(body):
        def _sf(v, default: float = np.nan) -> float:
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

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
            sample_count = int(_sf(stats.get("sampleCount"), 0))
            nodata_count = int(_sf(stats.get("noDataCount"), 0))
            total_samples += sample_count
            total_nodata += nodata_count

            if sample_count == 0:
                for s in STATS:
                    features[f"{idx_name}_{s}"] = np.nan
                continue

            pcts = stats.get("percentiles", {})
            features[f"{idx_name}_mean"]   = _sf(stats.get("mean"))
            features[f"{idx_name}_median"] = _sf(pcts.get("50.0"))
            features[f"{idx_name}_std"]    = _sf(stats.get("stDev"))
            features[f"{idx_name}_p10"]    = _sf(pcts.get("10.0"))
            features[f"{idx_name}_p90"]    = _sf(pcts.get("90.0"))

        grand_total = total_samples + total_nodata
        usable_pct = total_samples / grand_total if grand_total > 0 else 0.0

        # VH at -100 dB means every pixel hit the 1e-10 clamp — no real signal
        if features.get("VH_mean", 0.0) <= -50.0:
            for s in STATS:
                features[f"VH_{s}"] = np.nan

        return features, usable_pct

    # -- Optical fetch with retry (Option B) ----------------------------------

    def _fetch_optical_with_retry(
        self,
        coords,
        day_start: int,
        day_end: int,
        base_date: datetime,
        width: int,
        height: int,
        save_tiffs: bool,
        stage_name: str,
        field_meta: dict,
    ) -> tuple[dict | None, float]:
        """
        Fetch optical stats for one stage, retrying with progressively wider
        windows (Option B) if the initial usable_pct is below CLOUD_RETRY_THRESHOLD.

        Returns (features_dict, usable_pct) — features_dict is None if all
        attempts fail to produce any data.
        """
        def _attempt(win_days: int, label: str) -> tuple[dict | None, float]:
            opt_start, opt_end = expand_window(day_start, day_end, win_days)
            ds = (base_date + timedelta(days=opt_start)).strftime("%Y-%m-%d")
            de = (base_date + timedelta(days=opt_end)).strftime("%Y-%m-%d")

            result = None

            if self.use_stats_api and not save_tiffs:
                result = self._fetch_optical_stats(coords, ds, de)

            if result is None:
                rasters = self._fetch_stage_rasters(coords, ds, de, width, height)
                if rasters is not None:
                    if save_tiffs and label == "initial":
                        tiff_dir = os.path.join(
                            self.output_dir, "processed",
                            str(base_date.year), field_meta["crop_label"].lower(),
                            field_meta["field_id"], stage_name,
                        )
                        os.makedirs(tiff_dir, exist_ok=True)
                        for k in OPTICAL_INDEX_KEYS.values():
                            arr = rasters.get(k)
                            if arr is not None:
                                tifffile.imwrite(
                                    os.path.join(tiff_dir, f"{k}.tif"),
                                    arr.astype(np.float32),
                                )
                    result = self._compute_features_from_rasters(rasters)

            if result is None:
                return None, 0.0
            features, usable_pct = result
            return features, usable_pct

        # Initial fetch with MIN_OPTICAL_WINDOW_DAYS
        features, usable_pct = _attempt(MIN_OPTICAL_WINDOW_DAYS, "initial")

        if usable_pct >= CLOUD_RETRY_THRESHOLD:
            return features, usable_pct

        # Option B: retry with wider windows
        best_features = features
        best_usable = usable_pct

        for retry_days in CLOUD_RETRY_WINDOWS:
            if best_usable >= CLOUD_RETRY_THRESHOLD:
                break
            logger.info(
                "    %s: usable %.0f%% < %.0f%% threshold — retrying with %d-day window",
                stage_name, best_usable * 100, CLOUD_RETRY_THRESHOLD * 100, retry_days,
            )
            time.sleep(0.5)
            retry_features, retry_usable = _attempt(retry_days, f"retry_{retry_days}d")
            if retry_features is not None and retry_usable > best_usable:
                best_features = retry_features
                best_usable = retry_usable

        return best_features, best_usable

    # -- Stage dispatch -------------------------------------------------------

    def _process_stage(self, coords, stage_name, base_date, stage_windows,
                       width, height, save_tiffs, field_meta, stagger=0.0):
        if stagger > 0:
            time.sleep(stagger)

        day_start, day_end = stage_windows[stage_name]

        # -- Optical (with Option B adaptive retry) --------------------------
        optical_features, usable_pct = self._fetch_optical_with_retry(
            coords, day_start, day_end, base_date,
            width, height, save_tiffs, stage_name, field_meta,
        )

        # -- SAR (expanded window to guarantee S1 coverage) ------------------
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

        # -- Merge -----------------------------------------------------------
        merged_features = {}

        if optical_features is not None:
            merged_features.update(optical_features)

        if sar_result is not None:
            sar_features, sar_pct = sar_result
            merged_features.update(sar_features)

        if not merged_features:
            return stage_name, None

        return stage_name, (merged_features, usable_pct)

    # -- Stage neighbor interpolation (Option D) ------------------------------

    @staticmethod
    def _interpolate_null_stages(row: dict) -> list[str]:
        """
        Option D: for each optical index×stat combination, linearly interpolate
        null stage values from the nearest valid stage anchors on both sides.

        - Only interior gaps with at least one valid anchor before AND after are
          filled — edge stages (baseline, maturity) are never extrapolated.
        - Multi-gap runs are handled naturally: anchors are always the closest
          non-null stages, regardless of how many consecutive nulls lie between.
        - SAR features are not touched (they have their own backfill path).

        Modifies `row` in-place.  Returns a sorted list of stage names that
        received at least one interpolated value.
        """
        interpolated: set[str] = set()

        for idx_name in OPTICAL_INDICES:
            for stat in STATS:
                # Build index of valid (stage_position, value) pairs
                anchors: list[tuple[int, float]] = []
                for i, stage in enumerate(STAGES):
                    col = f"{idx_name}_{stat}_{stage}"
                    v = row.get(col)
                    if v is not None:
                        try:
                            fv = float(v)
                            if not np.isnan(fv):
                                anchors.append((i, fv))
                        except (TypeError, ValueError):
                            pass

                if len(anchors) < 2:
                    continue  # not enough anchors to interpolate from

                for i, stage in enumerate(STAGES):
                    col = f"{idx_name}_{stat}_{stage}"
                    v = row.get(col)
                    is_null = v is None or (isinstance(v, float) and np.isnan(v))
                    if not is_null:
                        continue

                    prev = [(si, sv) for si, sv in anchors if si < i]
                    nxt  = [(si, sv) for si, sv in anchors if si > i]

                    if not prev or not nxt:
                        continue  # no anchor on one side — skip to avoid extrapolation

                    si_prev, sv_prev = prev[-1]
                    si_next, sv_next = nxt[0]

                    t = (i - si_prev) / (si_next - si_prev)
                    row[col] = sv_prev + t * (sv_next - sv_prev)
                    interpolated.add(stage)

        return sorted(interpolated, key=lambda s: STAGES.index(s))

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

        # -- Option D: interpolate null optical stages from valid neighbors ---
        interpolated = self._interpolate_null_stages(row)
        row["interpolated_stages"] = ",".join(interpolated) if interpolated else ""
        if interpolated:
            logger.info(
                "    interpolated null stages from neighbors: %s",
                ", ".join(interpolated),
            )

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
            cols_def.append('"optical_backfill_done" INTEGER DEFAULT 0')
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

        if "optical_backfill_done" not in existing_cols:
            conn.execute('ALTER TABLE phenology_features ADD COLUMN "optical_backfill_done" INTEGER DEFAULT 0')
            logger.info("Migrated: added column 'optical_backfill_done'")

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
                values.append(v if v is not None else "")
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

            interp_tag = (
                f" | interpolated: {row['interpolated_stages']}"
                if row.get("interpolated_stages") else ""
            )
            logger.info(
                "  -> %d/%d stages covered%s | written to DB",
                row["stages_covered"], len(STAGES), interp_tag,
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

    # -- Optical Backfill -----------------------------------------------------

    def backfill_optical(self, kml_root: str):
        """
        Populate the red-edge / SWIR optical indices (BACKFILL_OPTICAL_INDICES)
        for existing rows.

        Unlike a full re-run this does NOT re-fetch SAR and does NOT overwrite
        the original NDVI/NDWI/EVI values — it issues one Sentinel-2 request per
        field x stage and writes only the new index columns. Rows are matched on
        a null first new-index column and a not-yet-attempted flag, mirroring the
        SAR backfill.
        """
        db_path = os.path.join(self.output_dir, "features.db")
        conn = self._init_db(db_path)

        first_col = f"{BACKFILL_OPTICAL_INDICES[0]}_mean_{STAGES[0]}"
        rows = conn.execute(
            f'SELECT rowid, field_id, crop_label, planting_date '
            f'FROM phenology_features '
            f'WHERE "{first_col}" IS NULL '
            f'AND (optical_backfill_done IS NULL OR optical_backfill_done = 0)'
        ).fetchall()

        if not rows:
            logger.info("No rows need optical backfill -- all populated or already attempted")
            conn.close()
            return

        logger.info("Optical backfill: %d rows to process", len(rows))

        kml_index = {}
        for dirpath, _, filenames in os.walk(kml_root):
            for fn in filenames:
                if fn.lower().endswith(".kml"):
                    meta = parse_metadata_from_filename(fn)
                    if meta:
                        kml_index[meta["field_id"]] = os.path.join(dirpath, fn)

        logger.info("KML index: %d files", len(kml_index))

        update_sql = (
            "UPDATE phenology_features SET "
            + ", ".join([f'"{c}" = ?' for c in BACKFILL_OPTICAL_FEATURE_COLS])
            + ', "optical_backfill_done" = 1'
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
            width, height = self._compute_resolution(coords)
            field_meta = {"field_id": field_id, "crop_label": crop_label}

            logger.info(
                "[%d/%d] optical backfill: %s | %s",
                i, len(rows), field_id, crop_label,
            )

            new_feats = {}
            any_data = False

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        self._fetch_optical_with_retry,
                        coords, stage_windows[sn][0], stage_windows[sn][1],
                        planting_date, width, height, False, sn, field_meta,
                    ): sn
                    for sn in STAGES
                }

                for future in as_completed(futures):
                    stage_name = futures[future]
                    try:
                        feats, usable_pct = future.result()
                    except Exception as e:
                        logger.warning("    %s: exception -- %s", stage_name, e)
                        feats = None

                    stage_has = False
                    if feats is not None:
                        for idx in BACKFILL_OPTICAL_INDICES:
                            v = feats.get(f"{idx}_mean")
                            try:
                                if v is not None and not math.isnan(float(v)):
                                    stage_has = True
                                    break
                            except (TypeError, ValueError):
                                pass

                    if stage_has:
                        for idx in BACKFILL_OPTICAL_INDICES:
                            for stat in STATS:
                                new_feats[f"{idx}_{stat}_{stage_name}"] = feats.get(f"{idx}_{stat}")
                        any_data = True
                        logger.info("    %s: optical OK (%.0f%%)", stage_name, usable_pct * 100)
                    else:
                        for idx in BACKFILL_OPTICAL_INDICES:
                            for stat in STATS:
                                new_feats[f"{idx}_{stat}_{stage_name}"] = np.nan
                        logger.info("    %s: optical no data", stage_name)

            values = []
            for col in BACKFILL_OPTICAL_FEATURE_COLS:
                v = new_feats.get(col)
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

            logger.info("  -> optical %s", "written" if any_data else "no data (marked done, won't retry)")
            time.sleep(self.request_delay)

        conn.close()
        logger.info("Optical backfill complete -- %d updated, %d skipped (no KML)", updated, skipped)


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phenology Feature Pipeline v5 — multi-temporal optical compositing (A), "
            "adaptive cloud-retry windows (B), and null-stage interpolation (C)."
        )
    )
    parser.add_argument("--mode", choices=["full", "backfill-sar", "backfill-optical"], default="full",
                        help="'full' = process new KMLs (optical+SAR), "
                             "'backfill-sar' = add SAR to existing rows with expanded windows, "
                             "'backfill-optical' = add red-edge/SWIR indices (NDRE/CIRE/MTCI/PSRI/NDMI) "
                             "to existing rows without re-fetching SAR")
    parser.add_argument("--max-per-crop", type=int, default=500)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--year", type=int, default=None,
                        help="Only process fields with this planting year (e.g. 2025)")
    parser.add_argument("--no-sar", action="store_true",
                        help="Skip SAR fetching in full mode")
    parser.add_argument("--output-dir", default=os.path.join("src", "data", "features_v5"),
                        help="Directory where features.db will be written (default: src/data/features_v5)")
    parser.add_argument("--kml-root", default=os.path.join("src", "data", "dataset_split", "train"),
                        help="Root directory containing KML files")
    args = parser.parse_args()

    service = SentinelHubService()
    kml_root = args.kml_root
    output_dir = args.output_dir

    pipeline = PhenologyFeaturePipeline(
        service,
        output_dir=output_dir,
        max_workers=args.max_workers,
        use_stats_api=True,
        fetch_sar=not args.no_sar,
    )

    if args.mode == "backfill-sar":
        pipeline.backfill_sar(kml_root)
    elif args.mode == "backfill-optical":
        pipeline.backfill_optical(kml_root)
    else:
        df = pipeline.process_directory(
            root_dir=kml_root,
            max_per_crop=args.max_per_crop,
            save_tiffs=False,
            planting_year=args.year,
        )

        if not df.empty:
            print(f"\nShape: {df.shape}")
            print(f"\nstages_covered distribution:")
            print(df["stages_covered"].value_counts().sort_index().to_string())

            interp_any = df["interpolated_stages"].fillna("").str.len() > 0
            print(f"\nFields with >= 1 interpolated stage: {interp_any.sum()} / {len(df)}")

            print(f"\nNull rate per stage (optical, after interpolation):")
            optical_prefixes = tuple(f"{idx}_" for idx in OPTICAL_INDICES)
            for stage in STAGES:
                stage_cols = [c for c in df.columns
                              if c.endswith(f"_{stage}") and c.startswith(optical_prefixes)]
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
