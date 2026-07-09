"""Phenology feature pipeline v6 — dense temporal series (Path B) + efficient I/O.

Why v6 exists
-------------
v5 issued ONE Statistical-API request per phenological stage (6 optical + 6 SAR per
field, plus cloud-retry calls), and processed fields serially with a 2-thread stage
pool. That is ~12-18 HTTP round-trips/field at concurrency 2 -> ~9.6 s/field.

v6 collapses the time loop into the Statistical API's own `aggregationInterval`: a
single request aggregates the whole crop calendar into time bins server-side. We use a
HYBRID grid:

  * DEKADAL (P10D) over the full season  [planting-15d .. +155d]  -> columns  *_d{k}
  * FINE   (P5D)  over flowering->maturity                        -> columns  *_f{k}

so resolution is spent where AVEIA vs TRIGO actually differ (heading / senescence),
while the early season stays cheap. SAR gets the dekadal grid only (S1 revisit is too
sparse for P5D to be meaningful).

Net: ~3 optical + 1 SAR = ~4 calls/field (was ~12), and fields run concurrently behind a
shared rate limiter. The win is wall-clock (round-trips + concurrency); Processing-Unit
cost is roughly flat since the same pixels x scenes are aggregated either way.

The model side (dense green-up/senescence slopes, dekadal-resolution onset timing,
curvature) is built on the *_d / *_f columns in features_v7 — that is where the 0.857
AUC ceiling either breaks or it doesn't.

Schema is intentionally separate (features_v6 DB); v5 stays untouched and shippable.
"""
import os
import sys
import math
import time
import json
import sqlite3
import logging
import argparse
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_ingestion.request_sentinel_v1 import SentinelHubService
from src.pipelines.phenology_feature_pipeline_v5 import (
    PhenologyFeaturePipeline,
    parse_kml_polygon,
    parse_metadata_from_filename,
    polygon_centroid,
    polygon_area_hectares,
    EVALSCRIPT_OPTICAL_STATS,
    EVALSCRIPT_SAR_STATS,
    OPTICAL_INDEX_KEYS,
    OPTICAL_INDICES,
    SAR_INDICES,
    CROP_STAGE_WINDOWS,
    DEFAULT_STAGE_WINDOWS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# -- Dense temporal grid ------------------------------------------------------
DEKAD_DAYS = 10
DEKAD_START = -15                       # days relative to planting
N_DEKADS = 17                           # covers planting-15d .. +155d
FINE_DAYS = 5
N_FINE = 18                             # covers flowering_start .. +90d at P5D

# Per-bin stats kept for the dense series. Lean on mean + spread (p10/p90); median/std
# are dropped to keep the schema < SQLite's column ceiling (~3 stats x ~50 bins x 12 idx).
DENSE_STATS = ["mean", "p10", "p90"]
MIN_USABLE_PCT_PER_BIN = 0.20           # clear-pixel fraction to accept a bin

SAR_INDEX_KEYS = {"VV": "vv", "VH": "vh", "CR": "cr", "RVI": "rvi"}

# Resolution constraints. The request `bounds` are in EPSG:4326, so `resx`/`resy` MUST be
# expressed in DEGREES to match. (The prior code passed metres here, which the API read as
# ~10 deg/px, collapsing every field to ~1 pixel: large fields breached S2L2A's 1500 m/px
# cap and 400'd, and all fields lost their within-field p10/p90 spread. See
# LARGE_FIELD_FAILURE_INVESTIGATION.md.) Two limits apply: never coarser than the S2L2A
# 1500 m/px cap (the degree conversion below keeps us at ~10 m, far under it) and never a
# grid over ~2500 px/side (coarsen only huge polygons to stay in range).
STATS_MAX_DIM = 2500
TARGET_RES_M = 10.0                     # native S2 resolution we aim for
_M_PER_DEG_LAT = 110_540
_M_PER_DEG_LON_EQ = 111_320


def _adaptive_res(coords) -> tuple[float, float]:
    """(resx, resy) in DEGREES for an EPSG:4326 Statistical-API request.

    Targets ~TARGET_RES_M metres/pixel, but coarsens per-axis so the output grid never
    exceeds STATS_MAX_DIM px on either side (only very large polygons are downsampled)."""
    minx, miny, maxx, maxy = ShapelyPolygon(coords).bounds
    mid_lat = (miny + maxy) / 2.0
    cos_lat = max(math.cos(math.radians(mid_lat)), 1e-6)
    base_x = TARGET_RES_M / (_M_PER_DEG_LON_EQ * cos_lat)   # ~10 m in degrees of longitude
    base_y = TARGET_RES_M / _M_PER_DEG_LAT                  # ~10 m in degrees of latitude
    resx = max(base_x, (maxx - minx) / STATS_MAX_DIM)
    resy = max(base_y, (maxy - miny) / STATS_MAX_DIM)
    return resx, resy

# Map a stat name to its location in the Stats-API band object.
_PCTL = {"p10": "10.0", "median": "50.0", "p90": "90.0"}

FIXED_COLS = [
    "field_id", "crop_label", "planting_date",
    "area_hectares", "latitude", "longitude",
    "dekads_covered", "fine_covered", "interpolated",
]
TEXT_COLS = {"field_id", "crop_label", "planting_date", "interpolated"}


def _optical_cols():
    cols = []
    for idx in OPTICAL_INDICES:
        for k in range(N_DEKADS):
            cols += [f"{idx}_{s}_d{k}" for s in DENSE_STATS]
        for k in range(N_FINE):
            cols += [f"{idx}_{s}_f{k}" for s in DENSE_STATS]
    return cols


def _sar_cols():
    return [f"{idx}_{s}_d{k}" for idx in SAR_INDICES
            for k in range(N_DEKADS) for s in DENSE_STATS]


ALL_COLS = FIXED_COLS + _optical_cols() + _sar_cols()


class RateLimiter:
    """Global token-bucket: paces calls to ~rate/sec across all worker threads
    without holding a lock during the wait (so threads still overlap)."""

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / rate_per_sec if rate_per_sec and rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            slot = max(now, self._next)
            self._next = slot + self.min_interval
        wait = slot - time.time()
        if wait > 0:
            time.sleep(wait)


class PhenologyFeaturePipelineV6(PhenologyFeaturePipeline):
    def __init__(self, sentinel_service, output_dir="output_v6",
                 field_workers: int = 6, rate_per_sec: float = 8.0,
                 fetch_sar: bool = True):
        super().__init__(sentinel_service, output_dir=output_dir,
                         use_stats_api=True, fetch_sar=fetch_sar)
        self.field_workers = field_workers
        self.limiter = RateLimiter(rate_per_sec)
        self._db_lock = threading.Lock()

    # -- payloads -------------------------------------------------------------
    def _optical_payload(self, coords, ds, de, interval_days, res):
        return {
            "input": {
                "bounds": {
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {"timeRange": {"from": f"{ds}T00:00:00Z", "to": f"{de}T23:59:59Z"}},
                }],
            },
            "aggregation": {
                "timeRange": {"from": f"{ds}T00:00:00Z", "to": f"{de}T23:59:59Z"},
                "aggregationInterval": {"of": f"P{interval_days}D"},
                "resx": res[0], "resy": res[1],
                "evalscript": EVALSCRIPT_OPTICAL_STATS,
            },
            "calculations": {"default": {"statistics": {"default": {"percentiles": {"k": [10, 50, 90]}}}}},
        }

    def _sar_payload(self, coords, ds, de, orbit_direction, res):
        data_filter = {"timeRange": {"from": f"{ds}T00:00:00Z", "to": f"{de}T23:59:59Z"}}
        if orbit_direction is not None:
            data_filter["orbitDirection"] = orbit_direction
        return {
            "input": {
                "bounds": {
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [{
                    "type": "sentinel-1-grd",
                    "dataFilter": data_filter,
                    "processing": {"backCoeff": "GAMMA0_TERRAIN", "orthorectify": True},
                }],
            },
            "aggregation": {
                "timeRange": {"from": f"{ds}T00:00:00Z", "to": f"{de}T23:59:59Z"},
                "aggregationInterval": {"of": f"P{DEKAD_DAYS}D"},
                "resx": res[0], "resy": res[1],
                "evalscript": EVALSCRIPT_SAR_STATS,
            },
            "calculations": {"default": {"statistics": {"default": {"percentiles": {"k": [10, 50, 90]}}}}},
        }

    def _rl_fetch(self, payload, ds, de):
        self.limiter.acquire()
        return self._fetch_stats_api(payload, ds, de)

    # -- parsing --------------------------------------------------------------
    @staticmethod
    def _parse_binned(body, index_keys, range_from, interval_days, n_bins, suffix):
        """Map each returned time bin to a grid slot k and emit {idx}_{stat}_{suffix}{k}.

        Bins below MIN_USABLE_PCT_PER_BIN (too cloudy) are left missing -> later
        interpolated from neighbours. Returns (features, covered_slots:set)."""
        out: dict[str, float] = {}
        covered: set[int] = set()
        if not body:
            return out, covered
        for entry in body.get("data", []):
            ifrom = entry.get("interval", {}).get("from")
            if not ifrom:
                continue
            try:
                d = (datetime.strptime(ifrom[:10], "%Y-%m-%d").date() - range_from).days
            except ValueError:
                continue
            k = round(d / interval_days)
            if k < 0 or k >= n_bins:
                continue
            outputs = entry.get("outputs", {})
            slot_has_data = False
            for idx_name, key in index_keys.items():
                stats = outputs.get(key, {}).get("bands", {}).get("B0", {}).get("stats", {})
                sc = stats.get("sampleCount") or 0
                nd = stats.get("noDataCount") or 0
                tot = sc + nd
                if not sc or (tot and sc / tot < MIN_USABLE_PCT_PER_BIN):
                    continue
                pcts = stats.get("percentiles", {})
                for stat in DENSE_STATS:
                    v = stats.get("mean") if stat == "mean" else pcts.get(_PCTL[stat])
                    if v is not None:
                        try:
                            out[f"{idx_name}_{stat}_{suffix}{k}"] = float(v)
                            slot_has_data = True
                        except (TypeError, ValueError):
                            pass
            if slot_has_data:
                covered.add(k)
        return out, covered

    def _sar_binned(self, coords, ds, de, range_from, res):
        """Combined-orbit SAR; fall back to single orbits if combined is empty."""
        body = self._rl_fetch(self._sar_payload(coords, ds, de, None, res), ds, de)
        out, cov = self._parse_binned(body, SAR_INDEX_KEYS, range_from, DEKAD_DAYS, N_DEKADS, "d")
        if out:
            return out
        for direction in ("ASCENDING", "DESCENDING"):
            body = self._rl_fetch(self._sar_payload(coords, ds, de, direction, res), ds, de)
            out, cov = self._parse_binned(body, SAR_INDEX_KEYS, range_from, DEKAD_DAYS, N_DEKADS, "d")
            if out:
                return out
        return {}

    # -- dense-grid interpolation --------------------------------------------
    @staticmethod
    def _interp_grid(row, indices, suffix, n_bins) -> int:
        """Linearly fill interior NaN gaps in each {idx}_{stat}_{suffix}k series.
        Edges are never extrapolated. Returns count of filled cells."""
        filled = 0
        for idx in indices:
            for stat in DENSE_STATS:
                cols = [f"{idx}_{stat}_{suffix}{k}" for k in range(n_bins)]
                vals = [row.get(c) for c in cols]
                anchors = [(i, v) for i, v in enumerate(vals)
                           if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if len(anchors) < 2:
                    continue
                ai = 0
                for i in range(anchors[0][0] + 1, anchors[-1][0]):
                    if vals[i] is not None and not (isinstance(vals[i], float) and math.isnan(vals[i])):
                        continue
                    while anchors[ai + 1][0] < i:
                        ai += 1
                    (li, lv), (ri, rv) = anchors[ai], anchors[ai + 1]
                    row[cols[i]] = lv + (rv - lv) * (i - li) / (ri - li)
                    filled += 1
        return filled

    # -- per field ------------------------------------------------------------
    def _process_field(self, meta, coords):
        base = meta["planting_date"]
        win = CROP_STAGE_WINDOWS.get(meta["crop_label"], DEFAULT_STAGE_WINDOWS)
        flo_start, mat_end = win["flowering"][0], win["maturity"][1]
        lat, lon = polygon_centroid(coords)
        res = _adaptive_res(coords)

        row = {
            "field_id": meta["field_id"],
            "crop_label": meta["crop_label"],
            "planting_date": base.strftime("%Y-%m-%d"),
            "area_hectares": round(polygon_area_hectares(coords), 2),
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
        }

        # 1. optical dekadal, full season
        d_from = base + timedelta(days=DEKAD_START)
        d_to = base + timedelta(days=DEKAD_START + N_DEKADS * DEKAD_DAYS)
        ds, de = d_from.strftime("%Y-%m-%d"), d_to.strftime("%Y-%m-%d")
        body = self._rl_fetch(self._optical_payload(coords, ds, de, DEKAD_DAYS, res), ds, de)
        feats, dek_cov = self._parse_binned(body, OPTICAL_INDEX_KEYS, d_from.date(), DEKAD_DAYS, N_DEKADS, "d")
        row.update(feats)

        # 2. optical fine, flowering -> maturity
        f_from = base + timedelta(days=flo_start)
        f_to = base + timedelta(days=min(flo_start + N_FINE * FINE_DAYS, mat_end + FINE_DAYS))
        fds, fde = f_from.strftime("%Y-%m-%d"), f_to.strftime("%Y-%m-%d")
        fbody = self._rl_fetch(self._optical_payload(coords, fds, fde, FINE_DAYS, res), fds, fde)
        ffeats, fine_cov = self._parse_binned(fbody, OPTICAL_INDEX_KEYS, f_from.date(), FINE_DAYS, N_FINE, "f")
        row.update(ffeats)

        # 3. SAR dekadal, full season
        if self.fetch_sar:
            row.update(self._sar_binned(coords, ds, de, d_from.date(), res))

        # interpolate interior cloud gaps in each dense series
        filled = self._interp_grid(row, OPTICAL_INDICES, "d", N_DEKADS)
        filled += self._interp_grid(row, OPTICAL_INDICES, "f", N_FINE)
        if self.fetch_sar:
            filled += self._interp_grid(row, SAR_INDICES, "d", N_DEKADS)

        row["dekads_covered"] = len(dek_cov)
        row["fine_covered"] = len(fine_cov)
        row["interpolated"] = str(filled)
        return row

    # -- DB -------------------------------------------------------------------
    def _init_db(self, db_path):
        conn = sqlite3.connect(db_path)
        existing = {r[1] for r in conn.execute("PRAGMA table_info(phenology_features)")}
        if not existing:
            defs = [f'"{c}" {"TEXT" if c in TEXT_COLS else "REAL"}' for c in ALL_COLS]
            conn.execute(f"CREATE TABLE phenology_features ({', '.join(defs)})")
            conn.commit()
        return conn

    @staticmethod
    def _insert_row(conn, row):
        cols = [c for c in ALL_COLS if c in row]
        placeholders = ", ".join("?" for _ in cols)
        vals = [row.get(c) for c in cols]
        conn.execute(
            f'INSERT INTO phenology_features ({", ".join(chr(34)+c+chr(34) for c in cols)}) '
            f"VALUES ({placeholders})", vals,
        )
        conn.commit()

    # -- driver ---------------------------------------------------------------
    def process_directory(self, root_dir, max_per_crop=None, planting_year=None,
                          exclude_crops=None, match_ids=None):
        db_path = os.path.join(self.output_dir, "features.db")
        existing_ids = self._load_existing_ids(db_path)
        conn = self._init_db(db_path)
        exclude = {c.strip().upper() for c in exclude_crops} if exclude_crops else set()

        # Build the work list up front (apply all filters deterministically).
        kmls = []
        for dp, _, fns in os.walk(root_dir):
            for fn in fns:
                if fn.lower().endswith(".kml"):
                    kmls.append((dp, fn))

        work = []
        # Seed per-crop counts from rows already in the DB so --max-per-crop is a TOTAL
        # target across resumed runs (not "this run only").
        crop_counts: dict[str, int] = {}
        if existing_ids:
            with sqlite3.connect(db_path) as _c:
                for crop, n in _c.execute(
                        "SELECT crop_label, COUNT(*) FROM phenology_features GROUP BY crop_label"):
                    crop_counts[crop] = n
        for dp, fn in sorted(kmls):
            meta = parse_metadata_from_filename(fn)
            if meta is None or meta["field_id"] in existing_ids:
                continue
            if match_ids is not None and meta["field_id"] not in match_ids:
                continue
            crop = meta["crop_label"]
            if crop.upper() in exclude:
                continue
            if planting_year is not None and meta["planting_date"].year != planting_year:
                continue
            if max_per_crop is not None and crop_counts.get(crop, 0) >= max_per_crop:
                continue
            coords = parse_kml_polygon(os.path.join(dp, fn))
            if coords is None:
                continue
            work.append((meta, coords))
            crop_counts[crop] = crop_counts.get(crop, 0) + 1

        logger.info("v6 extraction: %d fields, %d workers, hybrid P%dD/P%dD",
                    len(work), self.field_workers, DEKAD_DAYS, FINE_DAYS)

        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.field_workers) as pool:
            futures = {pool.submit(self._process_field, m, c): m for m, c in work}
            for fut in as_completed(futures):
                meta = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    logger.warning("  %s failed: %s", meta["field_id"], e)
                    continue
                with self._db_lock:
                    self._insert_row(conn, row)
                done += 1
                logger.info("  [%d/%d] %s | %s | dekads=%s fine=%s interp=%s",
                            done, len(work), row["field_id"], row["crop_label"],
                            row["dekads_covered"], row["fine_covered"], row["interpolated"])

        conn.close()
        dt = time.time() - t0
        logger.info("Done: %d fields in %.1f min (%.1f s/field)",
                    done, dt / 60, dt / done if done else 0.0)
        with sqlite3.connect(db_path) as rc:
            return pd.read_sql("SELECT * FROM phenology_features", rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phenology feature pipeline v6 (dense temporal)")
    ap.add_argument("--kml-root", required=True)
    ap.add_argument("--output-dir", default=os.path.join("src", "data", "features_v6"))
    ap.add_argument("--max-per-crop", type=int, default=None)
    ap.add_argument("--planting-year", type=int, default=None)
    ap.add_argument("--exclude-crops", nargs="*", default=None)
    ap.add_argument("--field-workers", type=int, default=6)
    ap.add_argument("--rate", type=float, default=8.0, help="max Stats-API calls/sec (shared)")
    ap.add_argument("--no-sar", dest="fetch_sar", action="store_false", default=True)
    ap.add_argument("--match-db", default=None,
                    help="Only process field_ids present in this features DB (matched comparison)")
    args = ap.parse_args()

    match_ids = None
    if args.match_db:
        with sqlite3.connect(args.match_db) as mc:
            match_ids = {r[0] for r in mc.execute("SELECT field_id FROM phenology_features")}
        logger.info("Restricting to %d field_ids from %s", len(match_ids), args.match_db)

    os.makedirs(args.output_dir, exist_ok=True)
    svc = SentinelHubService()
    pipe = PhenologyFeaturePipelineV6(
        svc, output_dir=args.output_dir,
        field_workers=args.field_workers, rate_per_sec=args.rate, fetch_sar=args.fetch_sar,
    )
    pipe.process_directory(
        args.kml_root, max_per_crop=args.max_per_crop,
        planting_year=args.planting_year, exclude_crops=args.exclude_crops,
        match_ids=match_ids,
    )
