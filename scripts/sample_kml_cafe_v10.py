"""Stage the v10 CAFE training top-up (~970 fields).

PLAN_V10_2025_2026.md §3: CAFE is the only crop that needs new training extraction to reach
the 2,500/crop balanced target (1,533 usable 2025-planted rows exist; 2024 rows are dropped
by the recent-years decision). Samples plantio 2025-08-01..2025-10-05 (270-day cycle complete
as of 2026-07) from the SICOR pools, excluding everything already used ANYWHERE:
v8 train / v8 test / features_v9 / kml_train_v9 staging / bench250 / kml_test_2026.

Copies to src/data/kml_train_v10_cafe/CAFE/ + manifest. Deterministic (seed 42), idempotent.

Usage:  python scripts/sample_kml_cafe_v10.py [--dry-run] [--target 970]
"""
import argparse
import json
import os
import random
import shutil
from datetime import date

import sys
sys.path.insert(0, os.path.dirname(__file__))
from sample_kml_test_2026 import parse_fn, build_exclusions, SICOR_ROOTS, BASE

TEST_2026 = os.path.join(BASE, "src", "data", "kml_test_2026")
OUT_DIR = os.path.join(BASE, "src", "data", "kml_train_v10_cafe")
SEED = 42
WINDOW = (date(2025, 8, 1), date(2025, 10, 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", type=int, default=970)
    args = ap.parse_args()

    excl = build_exclusions()
    for d in os.listdir(TEST_2026):
        cdir = os.path.join(TEST_2026, d)
        if os.path.isdir(cdir):
            for fn in os.listdir(cdir):
                p = parse_fn(fn)
                if p:
                    excl.add(p[0])
    print(f"exclusion list: {len(excl)} field_ids (incl. kml_test_2026)")

    pool = {}
    for root in SICOR_ROOTS:
        cdir = os.path.join(root, "arquivos_kml_CAFE")
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            p = parse_fn(fn)
            if p is None:
                continue
            fid, crop, plantio = p
            if crop != "CAFE" or fid in excl or fid in pool:
                continue
            if WINDOW[0] <= plantio <= WINDOW[1]:
                pool[fid] = os.path.join(cdir, fn)

    print(f"free CAFE in {WINDOW[0]}..{WINDOW[1]}: {len(pool)}")
    rng = random.Random(SEED)
    ids = sorted(pool)
    chosen = ids if len(ids) <= args.target else rng.sample(ids, args.target)
    print(f"sampled: {len(chosen)}")
    if args.dry_run:
        return

    dst_dir = os.path.join(OUT_DIR, "CAFE")
    os.makedirs(dst_dir, exist_ok=True)
    for fid in chosen:
        src = pool[fid]
        dst = os.path.join(dst_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    manifest = {"seed": SEED, "generated": date.today().isoformat(),
                "window": [WINDOW[0].isoformat(), WINDOW[1].isoformat()],
                "available": len(pool), "sampled": len(chosen),
                "field_ids": sorted(chosen)}
    with open(os.path.join(OUT_DIR, "sampling_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"staged under {OUT_DIR}")


if __name__ == "__main__":
    main()
