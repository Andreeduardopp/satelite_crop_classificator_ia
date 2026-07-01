"""Phenology feature pipeline v7 — extended dense grid (CAFE coverage).

Why v7 exists
-------------
v6's dense dekadal grid spans only [planting-15d .. +155d] (N_DEKADS=17). That fits every
annual in the crop universe, but CAFE (perennial, ~270-day plantio->colheita cycle:
flowering 90-150, grain_fill 150-210, maturity 210-270) runs clean off the end — the whole
back half of coffee's season is uncovered. That is the single blocker for the 7th crop.

v7 keeps v6's hybrid schema and every grid-agnostic mechanism (Statistical-API payloads,
server-side time aggregation, bin parsing, interior-gap interpolation, shared rate limiter,
field-level concurrency, resume-safe driver) and changes exactly one thing:

  * DEKADAL (P10D) over the full season  [planting-15d .. +275d]  -> N_DEKADS 17 -> 29

so the dekadal series now covers CAFE's full cycle. The fine (P5D) grid is unchanged
(N_FINE=18): it is per-crop anchored at flowering_start and already covers each annual's
senescence (the AVEIA/TRIGO lever); CAFE separates on the coarse evergreen signal the
extended dekadal grid now captures, so it does not need fine-grid extension.

Because a single multi-crop classifier cannot know the crop at inference time, the grid must
be this CAFE-length grid for EVERY field — annuals included. So the 7-crop training set is
re-extracted uniformly over this window (annuals genuinely show post-harvest bare soil /
cloud in the late dekads rather than artificial NULLs, avoiding a train/serve skew and a
"late bins present => CAFE" shortcut).

Schema is a separate DB (features_v6_ext); v5 and v6 stay untouched and shippable.
New column count ~1485 < SQLite's ~2000 ceiling.

Usage:
    python src/pipelines/phenology_feature_pipeline_v7.py \
        --kml-root src/data/dataset_split/train \
        --match-db src/data/features_v6/features.db \
        --output-dir src/data/features_v6_ext
"""
import os
import sys
import sqlite3
import logging
import argparse
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data_ingestion.request_sentinel_v1 import SentinelHubService
from src.pipelines.phenology_feature_pipeline_v5 import (
    parse_kml_polygon,
    parse_metadata_from_filename,
    polygon_centroid,
    polygon_area_hectares,
    OPTICAL_INDEX_KEYS,
    OPTICAL_INDICES,
    SAR_INDICES,
    CROP_STAGE_WINDOWS,
    DEFAULT_STAGE_WINDOWS,
)
from src.pipelines.phenology_feature_pipeline_v6 import (
    PhenologyFeaturePipelineV6,
    DEKAD_DAYS,
    DEKAD_START,
    FINE_DAYS,
    DENSE_STATS,
    SAR_INDEX_KEYS,
    FIXED_COLS,
    TEXT_COLS,
    _adaptive_res,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# -- Extended dense temporal grid (the only change from v6) -------------------
N_DEKADS = 29                           # covers planting-15d .. +275d (CAFE's ~270d cycle)
N_FINE = 18                             # unchanged: flowering_start .. +90d at P5D


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


class PhenologyFeaturePipelineV7(PhenologyFeaturePipelineV6):
    """v6 pipeline with the dekadal grid extended to N_DEKADS=29 (CAFE coverage).

    Only the grid-dependent methods are overridden; payloads, parsing, interpolation,
    the rate limiter and the resume-safe driver are inherited unchanged from v6.
    """

    # -- SAR: same logic as v6 but over the extended N_DEKADS grid -----------
    def _sar_binned(self, coords, ds, de, range_from, res):
        body = self._rl_fetch(self._sar_payload(coords, ds, de, None, res), ds, de)
        out, _ = self._parse_binned(body, SAR_INDEX_KEYS, range_from, DEKAD_DAYS, N_DEKADS, "d")
        if out:
            return out
        for direction in ("ASCENDING", "DESCENDING"):
            body = self._rl_fetch(self._sar_payload(coords, ds, de, direction, res), ds, de)
            out, _ = self._parse_binned(body, SAR_INDEX_KEYS, range_from, DEKAD_DAYS, N_DEKADS, "d")
            if out:
                return out
        return {}

    # -- per field: same flow as v6, using v7's N_DEKADS/N_FINE --------------
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

        # 1. optical dekadal, full season (extended to +275d)
        d_from = base + timedelta(days=DEKAD_START)
        d_to = base + timedelta(days=DEKAD_START + N_DEKADS * DEKAD_DAYS)
        ds, de = d_from.strftime("%Y-%m-%d"), d_to.strftime("%Y-%m-%d")
        body = self._rl_fetch(self._optical_payload(coords, ds, de, DEKAD_DAYS, res), ds, de)
        feats, dek_cov = self._parse_binned(body, OPTICAL_INDEX_KEYS, d_from.date(), DEKAD_DAYS, N_DEKADS, "d")
        row.update(feats)

        # 2. optical fine, flowering -> maturity (unchanged P5D window)
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

    # -- DB: same as v6 but with v7's wider ALL_COLS ------------------------
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phenology feature pipeline v7 (extended dense grid)")
    ap.add_argument("--kml-root", required=True)
    ap.add_argument("--output-dir", default=os.path.join("src", "data", "features_v6_ext"))
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
    pipe = PhenologyFeaturePipelineV7(
        svc, output_dir=args.output_dir,
        field_workers=args.field_workers, rate_per_sec=args.rate, fetch_sar=args.fetch_sar,
    )
    pipe.process_directory(
        args.kml_root, max_per_crop=args.max_per_crop,
        planting_year=args.planting_year, exclude_crops=args.exclude_crops,
        match_ids=match_ids,
    )
