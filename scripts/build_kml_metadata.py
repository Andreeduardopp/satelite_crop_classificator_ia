"""
build_kml_metadata.py
─────────────────────
Parses KML agricultural field files, computes geodesic geometry, reverse-
geocodes each field centroid to a Brazilian municipality using the official
IBGE boundary polygons, and stores everything in a SQLite database.

Strategy
────────
1. On first run: download IBGE municipality metadata (JSON, ~1 MB) and
   boundary polygons (GeoJSON, simplified) – both are cached to disk.
2. Build a shapely STRtree from the ~5 570 municipality polygons.
3. For each KML file: parse → compute geometry → point-in-polygon lookup.
4. Insert into SQLite (UNIQUE constraint prevents duplicate rows, so the
   script is fully resumable with --resume).

Dependencies
────────────
    pip install shapely pyproj requests tqdm

Usage
─────
    python scripts/build_kml_metadata.py
    python scripts/build_kml_metadata.py --source my_sample --db metadata.db
    python scripts/build_kml_metadata.py --resume   # skip already-done files
    python scripts/build_kml_metadata.py --no-geo   # skip geocoding (geometry only)
"""

from __future__ import annotations

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import sqlite3
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from pyproj import Geod
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─── Constants ────────────────────────────────────────────────────────────────

KML_NS = "http://www.opengis.net/kml/2.2"
GEOD = Geod(ellps="WGS84")

# IBGE API endpoints
IBGE_MUNICIPIOS_URL = (
    "https://servicodados.ibge.gov.br/api/v1/localidades/municipios?view=nivelado"
)
# IBGE Malhas per state - use intrarregiao=municipio (NOT resolucao=municipio)
# resolucao is silently ignored at qualidade=minima and returns state outlines.
# intrarregiao=municipio correctly subdivides the state into municipalities.
IBGE_MALHAS_UF_URL = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf_id}"
    "?intrarregiao=municipio&formato=application/vnd.geo+json&qualidade=minima"
)

CACHE_DIR = Path(".ibge_cache")
CACHE_META = CACHE_DIR / "municipios_nivelado.json"
CACHE_GEO  = CACHE_DIR / "municipios_malha_merged.geojson"

# ─── SQLite DDL ───────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS kml_metadata (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filename         TEXT    NOT NULL,
    crop_type        TEXT    NOT NULL,
    document_name    TEXT,
    placemark_name   TEXT,
    -- Geometry (WGS-84)
    centroid_lon     REAL,
    centroid_lat     REAL,
    area_ha          REAL,
    bbox_min_lon     REAL,
    bbox_max_lon     REAL,
    bbox_min_lat     REAL,
    bbox_max_lat     REAL,
    vertex_count     INTEGER,
    -- Location (IBGE)
    country          TEXT    DEFAULT 'Brasil',
    state_name       TEXT,
    state_uf         TEXT,
    state_ibge_id    INTEGER,
    municipality     TEXT,
    ibge_mun_code    INTEGER,
    mesoregion       TEXT,
    mesoregion_id    INTEGER,
    microregion      TEXT,
    microregion_id   INTEGER,
    region_name      TEXT,
    region_sigla     TEXT,
    -- Housekeeping
    processed_at     TEXT,
    UNIQUE(filename, crop_type) ON CONFLICT IGNORE
);
CREATE INDEX IF NOT EXISTS idx_crop     ON kml_metadata(crop_type);
CREATE INDEX IF NOT EXISTS idx_uf       ON kml_metadata(state_uf);
CREATE INDEX IF NOT EXISTS idx_ibge_mun ON kml_metadata(ibge_mun_code);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO kml_metadata (
    filename, crop_type, document_name, placemark_name,
    centroid_lon, centroid_lat, area_ha,
    bbox_min_lon, bbox_max_lon, bbox_min_lat, bbox_max_lat,
    vertex_count,
    country, state_name, state_uf, state_ibge_id,
    municipality, ibge_mun_code,
    mesoregion, mesoregion_id, microregion, microregion_id,
    region_name, region_sigla,
    processed_at
) VALUES (
    :filename, :crop_type, :document_name, :placemark_name,
    :centroid_lon, :centroid_lat, :area_ha,
    :bbox_min_lon, :bbox_max_lon, :bbox_min_lat, :bbox_max_lat,
    :vertex_count,
    :country, :state_name, :state_uf, :state_ibge_id,
    :municipality, :ibge_mun_code,
    :mesoregion, :mesoregion_id, :microregion, :microregion_id,
    :region_name, :region_sigla,
    :processed_at
)
"""

# ─── Download & cache helpers ─────────────────────────────────────────────────

def _download(url: str, dest: Path, label: str) -> None:
    """Download url to dest with a progress indicator."""
    print(f"  Downloading {label} …", end=" ", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            downloaded += len(chunk)
    mb = downloaded / 1_048_576
    print(f"done ({mb:.1f} MB)")


def load_ibge_metadata(cache_dir: Path) -> dict[int, dict]:
    """
    Returns a dict keyed by municipio-id with all metadata fields.
    Fields available: municipio-id, municipio-nome, microrregiao-*, mesorregiao-*,
    UF-id, UF-sigla, UF-nome, regiao-id, regiao-sigla, regiao-nome.
    """
    dest = cache_dir / "municipios_nivelado.json"
    if not dest.exists():
        _download(IBGE_MUNICIPIOS_URL, dest, "IBGE municipality metadata")

    raw: list[dict] = json.loads(dest.read_text(encoding="utf-8"))
    return {int(m["municipio-id"]): m for m in raw}


def _load_features_from_geojson(text: str) -> list:
    """Parse a GeoJSON FeatureCollection and return its features list."""
    fc = json.loads(text)
    if fc.get("type") == "FeatureCollection":
        return fc.get("features", [])
    return []


def load_ibge_spatial(
    cache_dir: Path,
    meta: dict[int, dict],
) -> tuple[STRtree, list, list[int]]:
    """
    Downloads (once per UF) IBGE municipality boundary polygons and merges them.
    Uses per-state endpoints because the BR country endpoint ignores
    resolucao=municipio and only returns the national outline.

    Returns (STRtree, list_of_shapely_geoms, list_of_ibge_codes).
    The tree and lists are aligned: tree.query result index i → geoms[i] → codes[i].
    """
    merged_path = cache_dir / "municipios_malha_merged.geojson"
    all_features: list = []

    # Phase 1: load from merged cache if it exists
    if merged_path.exists():
        print("  Loading cached boundary polygons ...", end=" ", flush=True)
        all_features = _load_features_from_geojson(
            merged_path.read_text(encoding="utf-8")
        )
        if len(all_features) <= 27:
            # Old bad state-level cache — purge it
            print(f"stale ({len(all_features)} features) - purging ...")
            merged_path.unlink()
            for f in cache_dir.glob("malha_uf_*.geojson"):
                f.unlink()
            all_features = []

    # Phase 2: download per-state if still empty
    if not all_features:
        uf_ids: list[int] = sorted({int(m["UF-id"]) for m in meta.values()})
        print(f"  Downloading boundaries for {len(uf_ids)} states ...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for uf_id in uf_ids:
            uf_cache = cache_dir / f"malha_uf_{uf_id}.geojson"
            if not uf_cache.exists():
                url = IBGE_MALHAS_UF_URL.format(uf_id=uf_id)
                print(f"    UF {uf_id:02d} ... ", end="", flush=True)
                try:
                    resp = requests.get(url, timeout=90)
                    resp.raise_for_status()
                    uf_cache.write_bytes(resp.content)
                    kb = len(resp.content) / 1024
                    print(f"OK ({kb:.0f} KB)")
                except Exception as exc:
                    print(f"FAILED ({exc})")
                    continue
            feats = _load_features_from_geojson(uf_cache.read_text(encoding="utf-8"))
            all_features.extend(feats)

        # Save merged cache
        merged = {"type": "FeatureCollection", "features": all_features}
        merged_path.write_text(json.dumps(merged), encoding="utf-8")
        print(f"  Merged {len(all_features)} features -> {merged_path.name}")

    print("  Building spatial index ...", end=" ", flush=True)
    geoms: list = []
    codes: list[int] = []
    skipped = 0
    for feat in all_features:
        try:
            g = shape(feat["geometry"])
            props = feat.get("properties") or {}
            # v3 API returns "codarea" for municipality code
            code_raw = (
                props.get("codarea")
                or props.get("GEOCODIGO")
                or props.get("CD_GEOCMU")
            )
            if code_raw is None or not g.is_valid:
                skipped += 1
                continue
            code_str = str(code_raw).strip()
            if len(code_str) < 7:
                skipped += 1
                continue
            geoms.append(g)
            codes.append(int(code_str[:7]))
        except Exception:
            skipped += 1

    tree = STRtree(geoms)
    print(f"done ({len(geoms)} polygons, {skipped} skipped)")
    return tree, geoms, codes


# ─── KML parsing ─────────────────────────────────────────────────────────────

def _tag(local: str) -> str:
    return f"{{{KML_NS}}}{local}"


def parse_kml(path: Path) -> Optional[dict]:
    """
    Parse a single KML file. Returns a dict with keys:
        document_name, placemark_name, coords  (list of (lon, lat) tuples)
    Returns None on failure.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        doc = root.find(_tag("Document"))
        if doc is None:
            doc = root   # some KML files omit Document wrapper

        # Document name
        doc_name_el = doc.find(_tag("name"))
        document_name = doc_name_el.text.strip() if doc_name_el is not None and doc_name_el.text else None

        # Placemark
        pm = doc.find(_tag("Placemark"))
        if pm is None:
            return None
        pm_name_el = pm.find(_tag("name"))
        placemark_name = pm_name_el.text.strip() if pm_name_el is not None and pm_name_el.text else None

        # Coordinates — try outerBoundaryIs first, else first LinearRing
        coord_text: Optional[str] = None
        for poly in pm.iter(_tag("Polygon")):
            outer = poly.find(f"{_tag('outerBoundaryIs')}/{_tag('LinearRing')}/{_tag('coordinates')}")
            if outer is not None and outer.text:
                coord_text = outer.text.strip()
                break
        if coord_text is None:
            for lr in pm.iter(_tag("LinearRing")):
                c = lr.find(_tag("coordinates"))
                if c is not None and c.text:
                    coord_text = c.text.strip()
                    break

        if not coord_text:
            return None

        coords: list[tuple[float, float]] = []
        for triplet in coord_text.split():
            parts = triplet.split(",")
            if len(parts) >= 2:
                coords.append((float(parts[0]), float(parts[1])))

        if len(coords) < 3:
            return None

        return {
            "document_name": document_name,
            "placemark_name": placemark_name,
            "coords": coords,
        }
    except Exception as exc:
        print(f"\n    [WARN] Failed to parse {path.name}: {exc}", file=sys.stderr)
        return None


# ─── Geometry computation ─────────────────────────────────────────────────────

def compute_geometry(coords: list[tuple[float, float]]) -> dict:
    """
    Given a list of (lon, lat) pairs (outer ring), return:
        centroid_lon, centroid_lat, area_ha, bbox_*, vertex_count
    """
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    # Geodesic area in m² → hectares
    poly_lons = lons + [lons[0]]
    poly_lats = lats + [lats[0]]
    area_m2, _ = GEOD.polygon_area_perimeter(poly_lons, poly_lats)
    area_ha = abs(area_m2) / 10_000.0

    # Simple arithmetic centroid (good enough for small polygons)
    centroid_lon = sum(lons) / len(lons)
    centroid_lat = sum(lats) / len(lats)

    return {
        "centroid_lon": centroid_lon,
        "centroid_lat": centroid_lat,
        "area_ha": area_ha,
        "bbox_min_lon": min(lons),
        "bbox_max_lon": max(lons),
        "bbox_min_lat": min(lats),
        "bbox_max_lat": max(lats),
        "vertex_count": len(coords),
    }


# ─── Geocoding via IBGE spatial ───────────────────────────────────────────────

def geocode(
    lon: float,
    lat: float,
    tree: STRtree,
    geoms: list,
    codes: list[int],
    meta: dict[int, dict],
) -> dict:
    """
    Point-in-polygon lookup against IBGE municipality boundaries.
    Returns a dict with location fields (empty strings on miss).
    """
    pt = Point(lon, lat)
    result = {
        "state_name": None, "state_uf": None, "state_ibge_id": None,
        "municipality": None, "ibge_mun_code": None,
        "mesoregion": None, "mesoregion_id": None,
        "microregion": None, "microregion_id": None,
        "region_name": None, "region_sigla": None,
    }

    # STRtree.query with predicate='contains' returns indices of containing geoms
    try:
        hits = tree.query(pt, predicate="contains")
    except Exception:
        hits = []

    if len(hits) == 0:
        # Fallback: nearest geometry (handles centroid on boundary edge)
        try:
            nearest_idx = tree.nearest(pt)
            if nearest_idx is None:
                return result
            hits = [nearest_idx]
        except Exception:
            return result

    idx = int(hits[0])
    code = codes[idx]
    m = meta.get(code)
    if m is None:
        # IBGE code from boundaries may be 7-digit; metadata uses 7-digit too.
        # Try 6-digit truncation as a fallback.
        m = meta.get(code // 10)

    if m:
        result.update({
            "state_name":    m.get("UF-nome"),
            "state_uf":      m.get("UF-sigla"),
            "state_ibge_id": m.get("UF-id"),
            "municipality":  m.get("municipio-nome"),
            "ibge_mun_code": m.get("municipio-id"),
            "mesoregion":    m.get("mesorregiao-nome"),
            "mesoregion_id": m.get("mesorregiao-id"),
            "microregion":   m.get("microrregiao-nome"),
            "microregion_id":m.get("microrregiao-id"),
            "region_name":   m.get("regiao-nome"),
            "region_sigla":  m.get("regiao-sigla"),
        })

    return result


# ─── Database ─────────────────────────────────────────────────────────────────

def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    conn.commit()
    return conn


def already_done(conn: sqlite3.Connection, filename: str, crop_type: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM kml_metadata WHERE filename=? AND crop_type=? LIMIT 1",
        (filename, crop_type),
    )
    return cur.fetchone() is not None


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build KML metadata SQLite database.")
    p.add_argument(
        "--source", type=Path, default=Path("my_sample_large"),
        help="Root folder with crop sub-folders (default: my_sample/).",
    )
    p.add_argument(
        "--db", type=Path, default=Path("metadata_large.db"),
        help="Output SQLite file (default: metadata.db).",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=Path(".ibge_cache"),
        help="Directory for IBGE data cache (default: .ibge_cache/).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip files already present in the database.",
    )
    p.add_argument(
        "--no-geo", action="store_true",
        help="Skip geocoding (store only geometry metadata).",
    )
    p.add_argument(
        "--commit-every", type=int, default=100,
        help="Commit to SQLite every N rows (default: 100).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    source = (project_root / args.source).resolve()
    db_path = (project_root / args.db).resolve()
    cache_dir = (project_root / args.cache_dir).resolve()

    if not source.exists():
        sys.exit(f"ERROR: Source directory not found: {source}")

    crop_folders = sorted([d for d in source.iterdir() if d.is_dir()])
    if not crop_folders:
        sys.exit(f"ERROR: No sub-folders found in: {source}")

    # Collect all KML files
    all_files: list[tuple[Path, str]] = []
    for folder in crop_folders:
        crop_type = folder.name.replace("arquivos_kml_", "")
        for kml in sorted(folder.glob("*.kml")):
            all_files.append((kml, crop_type))

    total = len(all_files)
    print(f"\nSource    : {source}")
    print(f"Database  : {db_path}")
    print(f"KML files : {total}")
    print(f"Resume    : {args.resume}")
    print(f"Geocoding : {'disabled' if args.no_geo else 'IBGE boundary polygons'}\n")

    # Load IBGE data (unless --no-geo)
    tree = geoms = codes = meta = None  # type: ignore[assignment]
    if not args.no_geo:
        print("Loading IBGE data …")
        meta  = load_ibge_metadata(cache_dir)
        tree, geoms, codes = load_ibge_spatial(cache_dir, meta)
        print(f"  Municipality records: {len(meta)}\n")

    # Init DB
    conn = init_db(db_path)

    # Process files
    iterator = all_files
    if HAS_TQDM:
        iterator = tqdm(all_files, unit="file", desc="Processing")

    inserted = skipped = errors = 0
    batch: list[dict] = []

    for kml_path, crop_type in iterator:
        filename = kml_path.name

        # Resume: skip already-processed
        if args.resume and already_done(conn, filename, crop_type):
            skipped += 1
            continue

        parsed = parse_kml(kml_path)
        if parsed is None:
            errors += 1
            continue

        geom = compute_geometry(parsed["coords"])

        loc: dict = {
            "state_name": None, "state_uf": None, "state_ibge_id": None,
            "municipality": None, "ibge_mun_code": None,
            "mesoregion": None, "mesoregion_id": None,
            "microregion": None, "microregion_id": None,
            "region_name": None, "region_sigla": None,
        }
        if not args.no_geo and tree is not None:
            loc = geocode(geom["centroid_lon"], geom["centroid_lat"], tree, geoms, codes, meta)

        row = {
            "filename":       filename,
            "crop_type":      crop_type,
            "document_name":  parsed["document_name"],
            "placemark_name": parsed["placemark_name"],
            "country":        "Brasil",
            "processed_at":   datetime.now(timezone.utc).isoformat(),
            **geom,
            **loc,
        }
        batch.append(row)
        inserted += 1

        if len(batch) >= args.commit_every:
            conn.executemany(INSERT_SQL, batch)
            conn.commit()
            batch.clear()

    # Flush remaining
    if batch:
        conn.executemany(INSERT_SQL, batch)
        conn.commit()

    conn.close()

    print(f"\n{'-'*50}")
    print(f"Done!")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped  : {skipped}  (already in DB)")
    print(f"  Errors   : {errors}   (parse failures)")
    print(f"  Database : {db_path}")


if __name__ == "__main__":
    main()
