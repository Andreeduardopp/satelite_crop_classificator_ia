"""
Migration: add latitude/longitude to existing phenology_features rows.

Walks the KML directory, matches each file to a field_id in the DB,
parses the polygon centroid, and writes lat/lon into two new columns.

No API calls — purely local.
"""

import os
import sys
import re
import sqlite3
import logging
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

_KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}

FILENAME_REGEX = re.compile(
    r"(.+?)_(.+)_plantio_(\d{2}-\d{2}-\d{2})_colheita_(\d{2}-\d{2}-\d{2})\.kml",
    re.IGNORECASE,
)

CROP_NAME_NORMALIZE = {
    "SOJA": "SOJA", "MILHO": "MILHO", "ARROZ": "ARROZ",
    "TRIGO": "TRIGO", "AVEIA": "AVEIA",
    "CAFE": "CAFE", "CAFÉ": "CAFE",
    "FEIJAO": "FEIJAO", "FEIJÃO": "FEIJAO",
}


def _normalize_crop_name(raw: str) -> str | None:
    clean = raw.upper().strip()
    if clean in CROP_NAME_NORMALIZE:
        return CROP_NAME_NORMALIZE[clean]
    ascii_only = "".join(ch for ch in clean if ord(ch) < 128)
    for canonical in CROP_NAME_NORMALIZE.values():
        if ascii_only.startswith(canonical[:3]) and len(ascii_only) >= 3:
            return canonical
    return None


def field_id_from_filename(filename: str) -> str | None:
    match = FILENAME_REGEX.match(filename)
    if not match:
        return None
    crop_raw, raw_id, _, _ = match.groups()
    crop_key = _normalize_crop_name(crop_raw)
    if crop_key is None:
        return None
    return f"{crop_key}_{raw_id}"


def parse_kml_centroid(kml_path: str) -> tuple[float, float] | None:
    try:
        tree = ET.parse(kml_path)
        coord_node = tree.getroot().find(".//kml:coordinates", _KML_NS)
        if coord_node is None:
            return None
        raw = coord_node.text.strip()
        coords = [
            (float(p.split(",")[0]), float(p.split(",")[1]))
            for p in raw.split()
            if len(p.split(",")) >= 2
        ]
        if len(coords) < 3:
            return None
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)
        return lat, lon
    except Exception:
        return None


def build_kml_index(root_dir: str) -> dict[str, str]:
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith(".kml"):
                continue
            fid = field_id_from_filename(fn)
            if fid:
                index[fid] = os.path.join(dirpath, fn)
    return index


def migrate(db_path: str, kml_root: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info("Building KML index from %s ...", kml_root)
    kml_index = build_kml_index(kml_root)
    logger.info("Indexed %d KML files", len(kml_index))

    conn = sqlite3.connect(db_path)

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(phenology_features)")}
    for col in ("latitude", "longitude"):
        if col not in existing_cols:
            conn.execute(f'ALTER TABLE phenology_features ADD COLUMN "{col}" REAL')
            logger.info("Added column: %s", col)
    conn.commit()

    rows = conn.execute("SELECT rowid, field_id FROM phenology_features").fetchall()
    logger.info("Processing %d rows...", len(rows))

    updated = 0
    missing = 0

    for rowid, field_id in rows:
        kml_path = kml_index.get(field_id)
        if kml_path is None:
            missing += 1
            continue

        centroid = parse_kml_centroid(kml_path)
        if centroid is None:
            missing += 1
            continue

        lat, lon = centroid
        conn.execute(
            "UPDATE phenology_features SET latitude = ?, longitude = ? WHERE rowid = ?",
            (lat, lon, rowid),
        )
        updated += 1

    conn.commit()
    conn.close()

    logger.info("Done — %d updated, %d missing KML match", updated, missing)


if __name__ == "__main__":
    db_path = os.path.join("src", "data", "features", "features.db")
    kml_root = os.path.join("src", "data", "dataset_split", "train")
    migrate(db_path, kml_root)
