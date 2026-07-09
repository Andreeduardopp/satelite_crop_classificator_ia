"""Build the balanced v10 training DB (PLAN_V10_2025_2026.md §3, step 4).

Sources:
  * src/data/features_v8v9_train/features.db  (merged v8+v9 pool, 22,727 rows)
  * src/data/features_v10_cafe/features.db    (the 970-field CAFE top-up)

Rules:
  1. Recent years only: drop every row with planting_date < 2025-01-01 (the 2024 rows).
  2. Safety exclusions: any field_id in features_v8_test, the 250-KML benchmark, or
     kml_test_2026 (zero overlap expected by construction — enforced anyway).
  3. Balance to a 2,500/crop cap with priority-keep + month-stratified fill (seed 42):
       - always keep rows planted in 2026 (truncated summer cohorts = year diversity);
       - keep SOJA planted Jan-Mar in full, and MILHO planted Jan-Mar up to a TOTAL
         safrinha-window share of 1,500 (2026 rows count toward it) — an uncapped rule
         lets safrinha crowd out main-season MILHO entirely;
       - fill the remaining quota proportionally across planting months.
     TRIGO (2,501) and AVEIA (1,623) are under/at cap -> kept whole.

Output: src/data/features_v10_train/features.db + build_manifest.json (counts + kept ids).

Usage:  python scripts/build_v10_train.py [--cap 2500] [--dry-run]
"""
import argparse
import json
import os
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))
from sample_kml_test_2026 import parse_fn, BASE, TEST_DB, BENCH_ZIP

MERGED_DB = os.path.join(BASE, "src", "data", "features_v8v9_train", "features.db")
CAFE_DB = os.path.join(BASE, "src", "data", "features_v10_cafe", "features.db")
TEST_2026_MANIFEST = os.path.join(BASE, "src", "data", "kml_test_2026", "sampling_manifest.json")
OUT_DIR = os.path.join(BASE, "src", "data", "features_v10_train")
SEED = 42
SAFRINHA_MONTHS = {1, 2, 3}
# total safrinha-window share allowed in the kept set (None = keep all)
SAFRINHA_SHARE_CAP = {"SOJA": None, "MILHO": 1500}


def build_exclusions():
    import zipfile
    excl = set()
    con = sqlite3.connect(TEST_DB)
    excl |= {r[0] for r in con.execute("SELECT field_id FROM phenology_features")}
    con.close()
    with zipfile.ZipFile(BENCH_ZIP) as z:
        for name in z.namelist():
            p = parse_fn(os.path.basename(name))
            if p:
                excl.add(p[0])
    with open(TEST_2026_MANIFEST) as f:
        for c in json.load(f)["crops"]:
            excl |= set(c["field_ids"])
    return excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    excl = build_exclusions()
    print(f"exclusion list: {len(excl)} field_ids (v8 test + benchmark + kml_test_2026)")

    # metadata of every candidate row: field_id -> (source_db, crop, planting_date)
    rows = {}
    n_2024 = n_excluded = n_dupe = 0
    for src in (MERGED_DB, CAFE_DB):
        con = sqlite3.connect(src)
        for fid, crop, pdate in con.execute(
                "SELECT field_id, crop_label, planting_date FROM phenology_features"):
            if fid in rows:
                n_dupe += 1
                continue
            if fid in excl:
                n_excluded += 1
                continue
            if (pdate or "") < "2025-01-01":
                n_2024 += 1
                continue
            rows[fid] = (src, crop.upper(), pdate)
        con.close()
    print(f"candidates: {len(rows)}  (dropped: {n_2024} pre-2025, {n_excluded} excluded, "
          f"{n_dupe} duplicate ids)")

    rng = random.Random(SEED)
    by_crop = defaultdict(list)
    for fid, (_, crop, pdate) in rows.items():
        by_crop[crop].append(fid)

    kept = []
    manifest = {"seed": SEED, "generated": date.today().isoformat(), "cap": args.cap,
                "sources": [MERGED_DB, CAFE_DB],
                "dropped_pre_2025": n_2024, "crops": {}}
    for crop in sorted(by_crop):
        fids = sorted(by_crop[crop])
        year = {f: int(rows[f][2][:4]) for f in fids}
        month = {f: int(rows[f][2][5:7]) for f in fids}
        tier_a = [f for f in fids if year[f] >= 2026]
        tier_b = []
        if crop in SAFRINHA_SHARE_CAP:
            safrinha_2025 = sorted(f for f in fids
                                   if year[f] < 2026 and month[f] in SAFRINHA_MONTHS)
            share_cap = SAFRINHA_SHARE_CAP[crop]
            if share_cap is None:
                tier_b = safrinha_2025
            else:
                n_a_safrinha = sum(1 for f in tier_a if month[f] in SAFRINHA_MONTHS)
                room = max(0, share_cap - n_a_safrinha)
                tier_b = (safrinha_2025 if len(safrinha_2025) <= room
                          else rng.sample(safrinha_2025, room))
        priority = tier_a + tier_b
        # non-priority pool: excludes ALL safrinha-window rows for share-capped crops
        # (they lost the tier-b draw; letting the month fill re-add them would defeat the cap)
        rest = [f for f in fids if f not in set(priority)
                and not (crop in SAFRINHA_SHARE_CAP
                         and SAFRINHA_SHARE_CAP[crop] is not None
                         and year[f] < 2026 and month[f] in SAFRINHA_MONTHS)]
        if len(fids) <= args.cap:
            chosen = fids
        else:
            quota = args.cap - len(priority)
            if quota <= 0:
                chosen = rng.sample(priority, args.cap)
            else:
                # month-stratified proportional fill of the remaining quota
                by_month = defaultdict(list)
                for f in rest:
                    by_month[month[f]].append(f)
                chosen = list(priority)
                take = {}
                total_rest = len(rest)
                for m, pool in sorted(by_month.items()):
                    take[m] = round(quota * len(pool) / total_rest)
                # fix rounding drift
                drift = quota - sum(take.values())
                for m in sorted(by_month, key=lambda m: -len(by_month[m])):
                    if drift == 0:
                        break
                    step = 1 if drift > 0 else -1
                    take[m] += step
                    drift -= step
                for m, pool in sorted(by_month.items()):
                    n = min(take[m], len(pool))
                    chosen += pool if n >= len(pool) else rng.sample(sorted(pool), n)
        kept += chosen
        yc = Counter(year[f] for f in chosen)
        manifest["crops"][crop] = {
            "available": len(fids), "kept": len(chosen),
            "priority_kept": len([f for f in chosen if f in set(priority)]),
            "by_year": {str(k): v for k, v in sorted(yc.items())}}
        print(f"{crop:7} available={len(fids):>5}  kept={len(chosen):>5}  "
              f"(priority={manifest['crops'][crop]['priority_kept']}, years={dict(sorted(yc.items()))})")

    print(f"\nTOTAL kept: {len(kept)}")
    if args.dry_run:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    out_db = os.path.join(OUT_DIR, "features.db")
    if os.path.exists(out_db):
        os.remove(out_db)
    con = sqlite3.connect(out_db)
    kept_by_src = defaultdict(list)
    for fid in kept:
        kept_by_src[rows[fid][0]].append(fid)
    first = True
    for src, fids in kept_by_src.items():
        con.execute("ATTACH DATABASE ? AS src", (src,))
        if first:
            con.execute("CREATE TABLE phenology_features AS "
                        "SELECT * FROM src.phenology_features LIMIT 0")
            first = False
        for i in range(0, len(fids), 500):
            chunk = fids[i:i + 500]
            q = ",".join("?" for _ in chunk)
            con.execute(f"INSERT INTO phenology_features "
                        f"SELECT * FROM src.phenology_features WHERE field_id IN ({q})", chunk)
        con.commit()
        con.execute("DETACH DATABASE src")
    n = con.execute("SELECT COUNT(*) FROM phenology_features").fetchone()[0]
    con.close()
    manifest["total_kept"] = len(kept)
    manifest["kept_field_ids"] = sorted(kept)
    with open(os.path.join(OUT_DIR, "build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {n} rows -> {out_db}")


if __name__ == "__main__":
    main()
