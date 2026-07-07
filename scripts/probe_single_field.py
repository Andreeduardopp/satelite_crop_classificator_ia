"""Instrumented single-field extraction probe — large-field failure investigation.

Runs the v6 pipeline's *optical dekadal* Statistical-API request for one (or a small
set of) field(s) and reports the raw HTTP status, timing, and returned-bin count — the
detail the batch pipeline swallows (`_fetch_stats_api` returns None on 400/timeout, then
`_process_field` writes a silent `dekads_covered=0` row).

It also re-issues any 60 s failure at a longer timeout to separate a pure request-timeout
(succeeds when given more time) from a hard server-side rejection (still 400/errors).

Usage:
    python scripts/probe_single_field.py                # runs the built-in seed set
    python scripts/probe_single_field.py FIELD_ID [...]  # probe specific field_ids

See src/pipelines/LARGE_FIELD_FAILURE_INVESTIGATION.md.
"""
import os, sys, re, json, time, math
import requests as http
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datetime import timedelta

from src.data_ingestion.request_sentinel_v1 import SentinelHubService
from src.pipelines.phenology_feature_pipeline_v6 import (
    PhenologyFeaturePipelineV6, _adaptive_res, DEKAD_START, N_DEKADS, DEKAD_DAYS,
)
from src.pipelines.phenology_feature_pipeline_v5 import (
    parse_kml_polygon, parse_metadata_from_filename, polygon_area_hectares,
)

KML_ROOT = os.path.join("culturas", "culturas", "culturas")

# 12 confirmed failures (dcov=0, >=50 ha) spanning size/region + 4 large controls that extracted OK.
SEED_FAIL = [
    "ARROZ_520317024-1", "SOJA_520263424-1", "ARROZ_520267078-1", "TRIGO_519484517-1",
    "ARROZ_520373711-1", "SOJA_519806356-1", "SOJA_520137178-1", "MILHO_518860265-1",
    "SOJA_519943336-1", "SOJA_520179231-1",
]
SEED_CTRL = ["ARROZ_517267258-1"]  # replaced at runtime if not resolvable; controls optional


def find_kml(field_id):
    crop = field_id.split("_")[0]
    num = field_id.split("_", 1)[1]
    d = os.path.join(KML_ROOT, f"arquivos_kml_{crop}")
    if not os.path.isdir(d):
        return None
    for f in os.listdir(d):
        if f.lower().endswith(".kml") and re.search(rf"_{re.escape(num)}_plantio", f):
            return os.path.join(d, f)
    return None


def probe(pipe, token, field_id, timeouts=(60, 180)):
    path = find_kml(field_id)
    if not path:
        print(f"{field_id:22s} KML NOT FOUND"); return
    meta = parse_metadata_from_filename(os.path.basename(path))
    coords = parse_kml_polygon(path)
    if meta is None or coords is None:
        print(f"{field_id:22s} PARSE FAILED"); return
    base = meta["planting_date"]
    res = _adaptive_res(coords)
    area = polygon_area_hectares(coords)
    d_from = base + timedelta(days=DEKAD_START)
    d_to = base + timedelta(days=DEKAD_START + N_DEKADS * DEKAD_DAYS)
    ds, de = d_from.strftime("%Y-%m-%d"), d_to.strftime("%Y-%m-%d")
    payload = pipe._optical_payload(coords, ds, de, DEKAD_DAYS, res)
    url = f"{pipe.service.url_base}/api/v1/statistics"
    hdr = {"Authorization": token, "Content-Type": "application/json"}

    for to in timeouts:
        t0 = time.time()
        try:
            r = http.post(url, timeout=to, headers=hdr, data=json.dumps(payload))
            dt = time.time() - t0
            if r.status_code == 200:
                nb = len(r.json().get("data", []))
                print(f"{field_id:22s} area={area:7.0f}ha res={res:.0f}m  "
                      f"HTTP 200 in {dt:5.1f}s  bins={nb}  [timeout={to}s]")
                return
            detail = ""
            try:
                detail = r.json().get("error", {}).get("message", "")[:160]
            except Exception:
                detail = r.text[:160]
            print(f"{field_id:22s} area={area:7.0f}ha res={res:.0f}m  "
                  f"HTTP {r.status_code} in {dt:5.1f}s  [timeout={to}s]  {detail}")
            if r.status_code in (400, 403, 404):
                return  # permanent — longer timeout won't help
        except http.exceptions.Timeout:
            print(f"{field_id:22s} area={area:7.0f}ha res={res:.0f}m  "
                  f"TIMEOUT after {to}s  -> retrying longer" if to != timeouts[-1]
                  else f"{field_id:22s} area={area:7.0f}ha  TIMEOUT even at {to}s")
        except Exception as e:
            print(f"{field_id:22s} ERROR: {e!r}"); return


def main():
    svc = SentinelHubService()
    pipe = PhenologyFeaturePipelineV6(svc, output_dir="scratch", fetch_sar=False)
    token = pipe._get_token()
    ids = sys.argv[1:] or SEED_FAIL
    print(f"Probing {len(ids)} field(s) — optical dekadal request, timeouts 60s then 180s\n")
    for fid in ids:
        probe(pipe, token, fid)


if __name__ == "__main__":
    main()
