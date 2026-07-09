"""Sample SICOR KMLs into the features_v9 training staging folder.

Implements the "Standard ~16.5k" composition from src/model/SICOR_DATA_PLAN.md:
per-crop cohorts defined by planting-date window, drawn from kml_sicor_2025 +
kml_sicor_2026 (disjoint id spaces), with hard exclusions:

  * field_ids already in features_v8 (train)      — already extracted, will be merged
  * field_ids in features_v8_test                 — held-out test set
  * field_ids in the 250-KML production benchmark — promotion gate, must stay unseen

Cohort windows (today = 2026-07-07):
  * "complete"    : planted <= 2025-10-05  -> full 29-dekad grid observable
  * "spring_2025" : planted 2025-10-06 .. 2025-11-30 -> the 2025/26 main summer season
                    (cycle complete, tail dekads truncated)
  * "summer_2026" : planted 2025-12-01 .. 2026-03-31 -> crop cycle complete/harvested,
                    post-harvest tail dekads truncated (deliberate: year diversity +
                    realistic truncation for serving)
  * 2026 winter cereals (TRIGO/AVEIA planted Apr+ 2026) are NOT sampled — mid-season;
    extract them in a December 2026 batch.

Selected files are COPIED to src/data/kml_train_v9/<CROP>/ and a manifest with
per-cohort counts is written alongside. Deterministic (seed 42), idempotent.

Usage:  python scripts/sample_kml_v9.py [--dry-run]
"""
import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import zipfile
from datetime import date

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SICOR_ROOTS = [
    os.path.join(BASE, "src", "model", "best_models", "datasets", "kml_sicor_2025"),
    os.path.join(BASE, "src", "model", "best_models", "datasets", "kml_sicor_2026"),
]
TRAIN_DB = os.path.join(BASE, "src", "model", "best_models", "datasets", "features_v8.db")
TEST_DB = os.path.join(BASE, "src", "model", "best_models", "datasets", "features_v8_test.db")
BENCH_ZIP = os.path.join(BASE, "src", "model", "best_models", "kml_test_sample_250.zip")
OUT_DIR = os.path.join(BASE, "src", "data", "kml_train_v9")
SEED = 42

FN_PAT = re.compile(r"^(.+?)_(\d+-\d+)_plantio_(\d{2})-(\d{2})-(\d{2,4})_colheita", re.I)

# (crop, cohort name, plantio window [from, to], target n). None = take all available.
COHORTS = [
    ("SOJA",   "safrinha_2025", date(2025, 1, 1),  date(2025, 3, 31),  None),
    ("SOJA",   "main_complete", date(2024, 6, 1),  date(2025, 10, 5),  1700),
    ("SOJA",   "spring_2025",   date(2025, 10, 6), date(2025, 11, 30), 800),
    ("SOJA",   "summer_2026",   date(2025, 12, 1), date(2026, 3, 31),  1000),
    ("MILHO",  "safrinha_2025", date(2025, 1, 1),  date(2025, 3, 31),  1500),
    ("MILHO",  "main_complete", date(2024, 6, 1),  date(2025, 10, 5),  500),
    ("MILHO",  "spring_2025",   date(2025, 10, 6), date(2025, 11, 30), 300),
    ("MILHO",  "summer_2026",   date(2025, 12, 1), date(2026, 3, 31),  1000),
    ("ARROZ",  "complete",      date(2024, 6, 1),  date(2025, 10, 5),  2500),
    ("ARROZ",  "spring_2025",   date(2025, 10, 6), date(2025, 11, 30), 1200),
    ("ARROZ",  "summer_2026",   date(2025, 12, 1), date(2026, 3, 31),  100),
    ("FEIJAO", "complete",      date(2024, 1, 1),  date(2025, 10, 5),  2300),
    ("FEIJAO", "spring_2025",   date(2025, 10, 6), date(2025, 11, 30), 300),
    ("FEIJAO", "summer_2026",   date(2025, 12, 1), date(2026, 3, 31),  500),
    ("TRIGO",  "complete",      date(2025, 4, 1),  date(2025, 9, 30),  1500),
    ("AVEIA",  "complete",      date(2025, 1, 1),  date(2025, 10, 5),  None),
    ("CAFE",   "planted_2025",  date(2025, 1, 1),  date(2025, 10, 5),  1200),
    ("CAFE",   "planted_2024",  date(2024, 1, 1),  date(2024, 12, 31), 800),
]
# NOTE: the SOJA main_complete / MILHO main_complete windows overlap the safrinha
# window; safrinha cohorts are filled first and their ids removed from later pools.


def parse_fn(fn):
    m = FN_PAT.match(fn)
    if not m:
        return None
    crop = (m.group(1).upper().replace("Ã", "A").replace("É", "E").replace("Á", "A"))
    dd, mm, yy = int(m.group(3)), int(m.group(4)), int(m.group(5))
    yy += 2000 if yy < 100 else 0
    try:
        plantio = date(yy, mm, dd)
    except ValueError:
        return None
    return f"{crop}_{m.group(2)}", crop, plantio


def build_exclusions():
    excl = set()
    for db in (TRAIN_DB, TEST_DB):
        con = sqlite3.connect(db)
        excl |= {r[0] for r in con.execute("SELECT field_id FROM phenology_features")}
        con.close()
    with zipfile.ZipFile(BENCH_ZIP) as z:
        for name in z.namelist():
            p = parse_fn(os.path.basename(name))
            if p:
                excl.add(p[0])
    return excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    excl = build_exclusions()
    print(f"exclusion list: {len(excl)} field_ids (train + test + benchmark)")

    # index every SICOR kml once: field_id -> (crop, plantio, path)
    index = {}
    dupes = 0
    for root in SICOR_ROOTS:
        for d in sorted(os.listdir(root)):
            cdir = os.path.join(root, d)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                p = parse_fn(fn)
                if p is None:
                    continue
                fid, crop, plantio = p
                if fid in index:
                    dupes += 1
                    continue
                index[fid] = (crop, plantio, os.path.join(cdir, fn))
    print(f"indexed {len(index)} unique fields ({dupes} duplicate ids skipped)")

    rng = random.Random(SEED)
    taken = set()
    manifest = {"seed": SEED, "generated": date.today().isoformat(), "cohorts": []}
    plan = []
    for crop, cohort, d_from, d_to, target in COHORTS:
        pool = [fid for fid, (c, pl, _) in index.items()
                if c == crop and d_from <= pl <= d_to
                and fid not in excl and fid not in taken]
        pool.sort()
        n = len(pool) if target is None else min(target, len(pool))
        chosen = pool if target is None or len(pool) <= target else rng.sample(pool, n)
        taken |= set(chosen)
        plan.extend(chosen)
        manifest["cohorts"].append({
            "crop": crop, "cohort": cohort,
            "window": [d_from.isoformat(), d_to.isoformat()],
            "target": target, "available": len(pool), "sampled": len(chosen),
        })
        print(f"{crop:7} {cohort:14} window {d_from}..{d_to}  "
              f"available={len(pool):>6}  sampled={len(chosen):>5}")

    print(f"\nTOTAL sampled: {len(plan)} fields (~{len(plan)*4} Stats-API calls)")
    if args.dry_run:
        return

    for fid in plan:
        crop, _, src = index[fid]
        dst_dir = os.path.join(OUT_DIR, crop)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    with open(os.path.join(OUT_DIR, "sampling_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"staged under {OUT_DIR}")


if __name__ == "__main__":
    main()
