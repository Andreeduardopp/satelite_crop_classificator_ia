from __future__ import annotations

import sys
# Force UTF-8 output on Windows terminals to avoid cp1252 errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
analyze_db.py
─────────────
Pretty-printed analysis of the KML metadata SQLite database.

Sections:
  1. Overview          - record counts, geocoding coverage, null rates
  2. Crop distribution - files and area per crop type
  3. Geographic spread - records per state (UF) and top municipalities
  4. Area statistics   - min / max / mean / median per crop (ha)
  5. Bounding boxes    - spatial extent per crop
  6. Data quality      - parse-fail proxy, zero-area fields, unresolved geocodes

Usage:
    python scripts/analyze_db.py
    python scripts/analyze_db.py --db path/to/metadata.db
    python scripts/analyze_db.py --top 15       # show more rows in tables
"""

import argparse
import sqlite3
import statistics
from pathlib import Path

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze KML metadata SQLite DB.")
    p.add_argument(
        "--db", type=Path, default=Path("metadata_large.db"),
        help="Path to the SQLite database (default: metadata.db).",
    )
    p.add_argument(
        "--top", type=int, default=10,
        help="Number of rows to show in ranked tables (default: 10).",
    )
    return p.parse_args()


# ─── Formatting helpers ───────────────────────────────────────────────────────

W = 72  # total table width

def section(title: str) -> None:
    print(f"\n{'=' * W}")
    pad = (W - len(title) - 2) // 2
    print(f"{'=' * pad} {title} {'=' * (W - pad - len(title) - 2)}")
    print(f"{'=' * W}")

def table(headers: list[str], rows: list[tuple], col_widths: list[int]) -> None:
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        # Truncate each cell to its column width
        cells = [str(v)[:w] if v is not None else "—" for v, w in zip(row, col_widths)]
        print(fmt.format(*cells))

def pct(part: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{100 * part / total:.1f}%"

def fmt_ha(v) -> str:
    if v is None:
        return "—"
    v = float(v)
    if v >= 1000:
        return f"{v:,.1f} ha"
    return f"{v:.2f} ha"

def fmt_n(v) -> str:
    if v is None:
        return "—"
    return f"{int(v):,}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    db_path = (project_root / args.db).resolve()

    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ── 1. Overview ──────────────────────────────────────────────────────────
    section("1 · OVERVIEW")

    total       = cur.execute("SELECT COUNT(*) FROM kml_metadata").fetchone()[0]
    geocoded    = cur.execute("SELECT COUNT(*) FROM kml_metadata WHERE municipality IS NOT NULL").fetchone()[0]
    no_geo      = total - geocoded
    crops_n     = cur.execute("SELECT COUNT(DISTINCT crop_type) FROM kml_metadata").fetchone()[0]
    states_n    = cur.execute("SELECT COUNT(DISTINCT state_uf)  FROM kml_metadata WHERE state_uf IS NOT NULL").fetchone()[0]
    munis_n     = cur.execute("SELECT COUNT(DISTINCT ibge_mun_code) FROM kml_metadata WHERE ibge_mun_code IS NOT NULL").fetchone()[0]
    total_ha    = cur.execute("SELECT SUM(area_ha) FROM kml_metadata").fetchone()[0]
    avg_ha      = cur.execute("SELECT AVG(area_ha) FROM kml_metadata").fetchone()[0]

    print(f"  Database          : {db_path.name}")
    print(f"  Total records     : {fmt_n(total)}")
    print(f"  Crop types        : {crops_n}")
    print(f"  States (UF)       : {states_n}")
    print(f"  Municipalities    : {fmt_n(munis_n)}")
    print(f"  Geocoded          : {fmt_n(geocoded)}  ({pct(geocoded, total)})")
    print(f"  Not geocoded      : {fmt_n(no_geo)}  ({pct(no_geo, total)})")
    print(f"  Total area        : {fmt_ha(total_ha)}")
    print(f"  Average field     : {fmt_ha(avg_ha)}")

    # ── 2. Crop distribution ─────────────────────────────────────────────────
    section("2 · CROP DISTRIBUTION")

    rows = cur.execute("""
        SELECT crop_type,
               COUNT(*)          AS n,
               SUM(area_ha)      AS total_ha,
               AVG(area_ha)      AS avg_ha,
               MIN(area_ha)      AS min_ha,
               MAX(area_ha)      AS max_ha
        FROM kml_metadata
        GROUP BY crop_type
        ORDER BY n DESC
    """).fetchall()

    table(
        ["Crop", "Fields", "Total (ha)", "Avg (ha)", "Min (ha)", "Max (ha)"],
        [(r["crop_type"], fmt_n(r["n"]), fmt_ha(r["total_ha"]),
          fmt_ha(r["avg_ha"]), fmt_ha(r["min_ha"]), fmt_ha(r["max_ha"]))
         for r in rows],
        [10, 8, 16, 12, 12, 14],
    )

    # ── 3. Geographic spread ─────────────────────────────────────────────────
    section("3 · RECORDS PER STATE (UF)")

    rows_uf = cur.execute("""
        SELECT state_uf, state_name, COUNT(*) AS n, SUM(area_ha) AS total_ha
        FROM kml_metadata
        WHERE state_uf IS NOT NULL
        GROUP BY state_uf
        ORDER BY n DESC
        LIMIT ?
    """, (args.top,)).fetchall()

    table(
        ["UF", "State name", "Fields", "Total area (ha)"],
        [(r["state_uf"], r["state_name"], fmt_n(r["n"]), fmt_ha(r["total_ha"]))
         for r in rows_uf],
        [4, 26, 8, 18],
    )

    # ── 3b. Top municipalities ───────────────────────────────────────────────
    print(f"\n  Top {args.top} municipalities by number of fields:\n")
    rows_mun = cur.execute("""
        SELECT municipality, state_uf, COUNT(*) AS n, SUM(area_ha) AS total_ha
        FROM kml_metadata
        WHERE municipality IS NOT NULL
        GROUP BY ibge_mun_code
        ORDER BY n DESC
        LIMIT ?
    """, (args.top,)).fetchall()

    table(
        ["Municipality", "UF", "Fields", "Total area (ha)"],
        [(r["municipality"], r["state_uf"], fmt_n(r["n"]), fmt_ha(r["total_ha"]))
         for r in rows_mun],
        [30, 4, 8, 18],
    )

    # ── 4. Area statistics per crop ──────────────────────────────────────────
    section("4 · AREA STATISTICS PER CROP  (hectares)")

    # Median via Python (SQLite has no built-in MEDIAN)
    all_areas = cur.execute(
        "SELECT crop_type, area_ha FROM kml_metadata WHERE area_ha IS NOT NULL ORDER BY crop_type, area_ha"
    ).fetchall()

    from collections import defaultdict
    areas_by_crop: dict[str, list[float]] = defaultdict(list)
    for r in all_areas:
        areas_by_crop[r[0]].append(float(r[1]))

    stat_rows = []
    for crop, vals in sorted(areas_by_crop.items()):
        stat_rows.append((
            crop,
            fmt_n(len(vals)),
            fmt_ha(min(vals)),
            fmt_ha(statistics.median(vals)),
            fmt_ha(statistics.mean(vals)),
            fmt_ha(max(vals)),
            fmt_ha(statistics.stdev(vals)) if len(vals) > 1 else "—",
        ))

    table(
        ["Crop", "N", "Min", "Median", "Mean", "Max", "StdDev"],
        stat_rows,
        [10, 7, 10, 12, 12, 14, 12],
    )

    # ── 5. Bounding boxes ────────────────────────────────────────────────────
    section("5 · GEOGRAPHIC EXTENT PER CROP")

    bb_rows = cur.execute("""
        SELECT crop_type,
               MIN(bbox_min_lon) AS w, MAX(bbox_max_lon) AS e,
               MIN(bbox_min_lat) AS s, MAX(bbox_max_lat) AS n
        FROM kml_metadata
        GROUP BY crop_type
        ORDER BY crop_type
    """).fetchall()

    table(
        ["Crop", "West (lon)", "East (lon)", "South (lat)", "North (lat)"],
        [(r["crop_type"],
          f"{r['w']:.4f}", f"{r['e']:.4f}",
          f"{r['s']:.4f}", f"{r['n']:.4f}")
         for r in bb_rows],
        [10, 12, 12, 13, 13],
    )

    # ── 6. Data quality ──────────────────────────────────────────────────────
    section("6 · DATA QUALITY")

    zero_area   = cur.execute("SELECT COUNT(*) FROM kml_metadata WHERE area_ha <= 0 OR area_ha IS NULL").fetchone()[0]
    tiny_fields = cur.execute("SELECT COUNT(*) FROM kml_metadata WHERE area_ha < 0.1").fetchone()[0]
    large_fields= cur.execute("SELECT COUNT(*) FROM kml_metadata WHERE area_ha > 10000").fetchone()[0]
    no_state    = cur.execute("SELECT COUNT(*) FROM kml_metadata WHERE state_uf IS NULL").fetchone()[0]
    no_muni     = cur.execute("SELECT COUNT(*) FROM kml_metadata WHERE municipality IS NULL").fetchone()[0]
    dup_check   = cur.execute("SELECT COUNT(*) FROM (SELECT filename, crop_type, COUNT(*) AS c FROM kml_metadata GROUP BY filename, crop_type HAVING c > 1)").fetchone()[0]

    print(f"  Zero / null area fields    : {fmt_n(zero_area)}  ({pct(zero_area, total)})")
    print(f"  Tiny fields  (< 0.1 ha)   : {fmt_n(tiny_fields)}  ({pct(tiny_fields, total)})")
    print(f"  Large fields (> 10 000 ha) : {fmt_n(large_fields)}  ({pct(large_fields, total)})")
    print(f"  Missing state (UF)         : {fmt_n(no_state)}  ({pct(no_state, total)})")
    print(f"  Missing municipality       : {fmt_n(no_muni)}  ({pct(no_muni, total)})")
    print(f"  Duplicate (filename+crop)  : {fmt_n(dup_check)}")

    # ── 6b. Vertex count distribution ────────────────────────────────────────
    print("\n  Vertex count distribution:\n")
    vc_rows = cur.execute("""
        SELECT
            CASE
                WHEN vertex_count < 5   THEN '< 5'
                WHEN vertex_count < 10  THEN '5–9'
                WHEN vertex_count < 20  THEN '10–19'
                WHEN vertex_count < 50  THEN '20–49'
                WHEN vertex_count < 100 THEN '50–99'
                ELSE '≥ 100'
            END AS bucket,
            COUNT(*) AS n
        FROM kml_metadata
        GROUP BY bucket
        ORDER BY MIN(vertex_count)
    """).fetchall()
    table(
        ["Vertex range", "Fields"],
        [(r[0], fmt_n(r[1])) for r in vc_rows],
        [14, 8],
    )

    # ── 7. Crop × State cross-tab ────────────────────────────────────────────
    section("7 · CROP × STATE CROSS-TABLE  (field count)")

    pivot_raw = cur.execute("""
        SELECT crop_type, state_uf, COUNT(*) AS n
        FROM kml_metadata
        WHERE state_uf IS NOT NULL
        GROUP BY crop_type, state_uf
        ORDER BY crop_type, n DESC
    """).fetchall()

    # Build in-memory pivot
    crops_order = sorted(areas_by_crop.keys())
    states_set:  list[str] = []
    seen: set[str] = set()
    for r in pivot_raw:
        if r[1] not in seen:
            states_set.append(r[1])
            seen.add(r[1])
    # sort states by total count
    state_totals = cur.execute("""
        SELECT state_uf, COUNT(*) AS n FROM kml_metadata
        WHERE state_uf IS NOT NULL GROUP BY state_uf ORDER BY n DESC
    """).fetchall()
    states_ordered = [r[0] for r in state_totals[:12]]  # top 12 states

    pivot: dict[tuple[str, str], int] = {}
    for r in pivot_raw:
        pivot[(r[0], r[1])] = r[2]

    col_w = 6
    header = f"  {'Crop':<10}" + "".join(f"{s:>{col_w}}" for s in states_ordered) + f"{'Other':>{col_w}}"
    print(header)
    print("  " + "-" * (10 + col_w * (len(states_ordered) + 1)))
    for crop in crops_order:
        row_str = f"  {crop:<10}"
        other = 0
        shown_keys = set()
        for uf in states_ordered:
            v = pivot.get((crop, uf), 0)
            row_str += f"{v:>{col_w}}"
            shown_keys.add(uf)
        for (c, u), v in pivot.items():
            if c == crop and u not in shown_keys:
                other += v
        row_str += f"{other:>{col_w}}"
        print(row_str)

    print(f"\n{'=' * W}\n")
    conn.close()


if __name__ == "__main__":
    main()
