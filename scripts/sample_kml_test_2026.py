"""Sample the 350-KML (50/crop) 2026 test set — the forward-looking held-out set for v10.

Design (see src/model/PLAN_V10_2025_2026.md): 50 fields per crop from the most recent
COMPLETED season available per crop, drawn from kml_sicor_2025 + kml_sicor_2026 with hard
exclusions (features_v8 train, features_v8_test, features_v9, kml_train_v9 staging, and the
250-KML production benchmark). Windows as of 2026-07-09:

  * SOJA / MILHO / FEIJAO : planted 2026-01-01 .. 2026-03-31  (2026 summer/safrinha, harvested)
  * ARROZ                 : planted 2025-10-01 .. 2026-02-28  (2025/26 rice season — zero free
                            plantio-2026 rice with a complete cycle exists in the SICOR pools)
  * CAFE                  : planted 2025-08-01 .. 2025-10-05  (270-day cycle completes by now;
                            these are safra-2026 registry files)
  * TRIGO / AVEIA         : planted 2026-03-01 .. 2026-07-31  (2026 winter season) — sampled and
                            FROZEN now so no future training batch can take them, but flagged
                            `extract_after: 2026-12-01`; the cycle is mid-season today.

Selected files are COPIED to src/data/kml_test_2026/<CROP>/ with a manifest. Deterministic
(seed 42), idempotent. Every future training sampler MUST exclude the field_ids staged here.

Usage:  python scripts/sample_kml_test_2026.py [--dry-run]
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
V9_DB = os.path.join(BASE, "src", "data", "features_v9", "features.db")
BENCH_ZIP = os.path.join(BASE, "src", "model", "best_models", "kml_test_sample_250.zip")
STAGED_V9 = os.path.join(BASE, "src", "data", "kml_train_v9")
OUT_DIR = os.path.join(BASE, "src", "data", "kml_test_2026")
SEED = 42
N_PER_CROP = 50

FN_PAT = re.compile(r"^(.+?)_(\d+-\d+)_plantio_(\d{2})-(\d{2})-(\d{2,4})_colheita", re.I)

# crop -> (window from, window to, extract_after or None)
WINDOWS = {
    "SOJA":   (date(2026, 1, 1),  date(2026, 3, 31), None),
    "MILHO":  (date(2026, 1, 1),  date(2026, 3, 31), None),
    "FEIJAO": (date(2026, 1, 1),  date(2026, 3, 31), None),
    "ARROZ":  (date(2025, 10, 1), date(2026, 2, 28), None),
    "CAFE":   (date(2025, 8, 1),  date(2025, 10, 5), None),
    "TRIGO":  (date(2026, 3, 1),  date(2026, 7, 31), "2026-12-01"),
    "AVEIA":  (date(2026, 3, 1),  date(2026, 7, 31), "2026-12-01"),
}


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
    for db in (TRAIN_DB, TEST_DB, V9_DB):
        if os.path.exists(db):
            con = sqlite3.connect(db)
            excl |= {r[0] for r in con.execute("SELECT field_id FROM phenology_features")}
            con.close()
    with zipfile.ZipFile(BENCH_ZIP) as z:
        for name in z.namelist():
            p = parse_fn(os.path.basename(name))
            if p:
                excl.add(p[0])
    if os.path.isdir(STAGED_V9):
        for d in os.listdir(STAGED_V9):
            cdir = os.path.join(STAGED_V9, d)
            if os.path.isdir(cdir):
                for fn in os.listdir(cdir):
                    p = parse_fn(fn)
                    if p:
                        excl.add(p[0])
    return excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    excl = build_exclusions()
    print(f"exclusion list: {len(excl)} field_ids (v8 train + v8 test + v9 + v9 staging + benchmark)")

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
    manifest = {"seed": SEED, "generated": date.today().isoformat(),
                "n_per_crop": N_PER_CROP, "crops": []}
    plan = []
    for crop in sorted(WINDOWS):
        d_from, d_to, extract_after = WINDOWS[crop]
        pool = [fid for fid, (c, pl, _) in index.items()
                if c == crop and d_from <= pl <= d_to and fid not in excl]
        pool.sort()
        if len(pool) < N_PER_CROP:
            raise SystemExit(f"FATAL: only {len(pool)} free {crop} fields in "
                             f"{d_from}..{d_to} (need {N_PER_CROP})")
        chosen = rng.sample(pool, N_PER_CROP)
        plan.extend(chosen)
        manifest["crops"].append({
            "crop": crop, "window": [d_from.isoformat(), d_to.isoformat()],
            "extract_after": extract_after, "available": len(pool),
            "sampled": len(chosen), "field_ids": sorted(chosen),
        })
        note = f"  (FROZEN, extract after {extract_after})" if extract_after else ""
        print(f"{crop:7} window {d_from}..{d_to}  available={len(pool):>6}  sampled={len(chosen)}{note}")

    print(f"\nTOTAL sampled: {len(plan)} fields")
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
