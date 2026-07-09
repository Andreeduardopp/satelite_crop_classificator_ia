# -*- coding: utf-8 -*-
"""
Extract KML field boundaries from SICOR open data, in the same layout as the
existing `culturas/` training dataset.

Usage:  python extract_kml_sicor.py <YEAR>        e.g. python extract_kml_sicor.py 2025
        MAX_FILES=300 python extract_kml_sicor.py 2025   # test batch

Join key: (REF_BACEN, NU_ORDEM)
  - SICOR_OPERACAO_BASICA_ESTADO_<YEAR>.gz -> crop (CD_EMPREENDIMENTO) + dates
  - sicor_glebas_wkt_<YEAR>.gz             -> field polygon (WKT)

Rules:
  - Crops: the 7 crops of the existing dataset, decoded via the 6-digit prefix
    of CD_EMPREENDIMENTO (reconstructed from the existing KMLs, verified 1:1).
  - Filename dates: DT_INIC_PLANTIO + DT_INIC_COLHEITA.
  - SKIP any operation with no planting date (DT_INIC_PLANTIO empty).
  - A missing harvest date is allowed -> written as "NA".

Output: <OUT>/kml_sicor_<YEAR>/arquivos_kml_{CROP}/
          {CROP}_{REF}-{ORDEM}_plantio_{DD-MM-YY}_colheita_{DD-MM-YY}.kml
"""
import gzip
import os
import re
import sys

SICOR_ROOT = r"C:\Users\eduar\Downloads\sicor_dados"

# 6-digit CD_EMPREENDIMENTO prefix -> crop (verified against existing dataset)
PREFIX_CROP = {
    "120109": "ARROZ",
    "120110": "AVEIA",
    "120115": "CAFE",
    "131215": "CAFE",
    "120135": "FEIJAO",
    "120150": "MILHO",
    "120167": "SOJA",
    "120171": "TRIGO",
}

# OPERACAO column indices (verified for 2025 & 2026: 47 fields, identical layout)
O_REF, O_ORD, O_EMP = 0, 1, 12
O_DT_INIC_COLHEITA = 37
O_DT_INIC_PLANTIO = 38

RING_RE = re.compile(r"\(([^()]*)\)")


def to_ddmmyy(d):
    """'15/09/2026' -> '15-09-26'; '' -> None."""
    d = (d or "").strip()
    if not d:
        return None
    p = d.split("/")
    if len(p) != 3:
        return None
    return f"{p[0]}-{p[1]}-{p[2][-2:]}"


def wkt_to_kml_rings(wkt):
    """POLYGON ((x y, ...), (inner...)) -> list of KML coord strings (outer first)."""
    rings = []
    for m in RING_RE.finditer(wkt):
        pts = []
        for pair in m.group(1).split(","):
            pair = pair.strip()
            if not pair:
                continue
            xy = pair.split()
            if len(xy) < 2:
                continue
            pts.append(f"{xy[0]},{xy[1]},0.0")
        if pts:
            rings.append(" ".join(pts))
    return rings


def build_placemark(ref, ordem, emp, rings):
    inner = ""
    for r in rings[1:]:
        inner += (
            "\n                <innerBoundaryIs>\n"
            "                    <LinearRing>\n"
            f"                        <coordinates>{r}</coordinates>\n"
            "                    </LinearRing>\n"
            "                </innerBoundaryIs>"
        )
    return (
        "        <Placemark>\n"
        f"            <name>Area_{ref}-{ordem}-{emp}</name>\n"
        "            <styleUrl>#polystyle</styleUrl>\n"
        "            <Polygon>\n"
        "                <outerBoundaryIs>\n"
        "                    <LinearRing>\n"
        f"                        <coordinates>{rings[0]}</coordinates>\n"
        "                    </LinearRing>\n"
        "                </outerBoundaryIs>"
        f"{inner}\n"
        "            </Polygon>\n"
        "        </Placemark>\n"
    )


def write_kml(path, ref, ordem, placemarks):
    body = "".join(placemarks)
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">\n'
        "    <Document>\n"
        '        <Style id="polystyle">\n'
        "            <LineStyle>\n"
        "                <color>ff000000</color>\n"
        "                <colorMode>normal</colorMode>\n"
        "                <width>2</width>\n"
        "            </LineStyle>\n"
        "            <PolyStyle>\n"
        "                <color>7800ffff</color>\n"
        "                <colorMode>normal</colorMode>\n"
        "                <fill>1</fill>\n"
        "                <outline>1</outline>\n"
        "            </PolyStyle>\n"
        "        </Style>\n"
        f"        <name>Operacao {ref}-{ordem}</name>\n"
        f"{body}"
        "    </Document>\n"
        "</kml>\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(kml)


def main(year):
    op_file = os.path.join(SICOR_ROOT, year, f"SICOR_OPERACAO_BASICA_ESTADO_{year}.gz")
    gl_file = os.path.join(SICOR_ROOT, year, f"sicor_glebas_wkt_{year}.gz")
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"kml_sicor_{year}")
    for f in (op_file, gl_file):
        if not os.path.exists(f):
            sys.exit(f"ERROR: missing input file: {f}")

    max_files = int(os.environ.get("MAX_FILES", "0"))  # 0 = no limit

    # Pass 1: index target operations that HAVE a planting date.
    print(f"[{year}] Pass 1: indexing OPERACAO (target crops with planting date)...", flush=True)
    ops = {}  # (ref,ord) -> (crop, emp, plantio, colheita)
    n = 0
    with gzip.open(op_file, "rt", encoding="latin-1") as f:
        f.readline()
        for line in f:
            n += 1
            p = line.rstrip("\n").split(";")
            if len(p) < 47:
                continue
            emp = p[O_EMP].strip()
            crop = PREFIX_CROP.get(emp[:6])
            if not crop:
                continue
            plantio = to_ddmmyy(p[O_DT_INIC_PLANTIO])
            if not plantio:
                continue
            colheita = to_ddmmyy(p[O_DT_INIC_COLHEITA]) or "NA"
            ops[(p[O_REF], p[O_ORD])] = (crop, emp, plantio, colheita)
    print(f"  scanned {n:,} operations -> {len(ops):,} target ops with planting date", flush=True)

    # Pass 2: stream GLEBAS, accumulate placemarks per operation.
    print(f"[{year}] Pass 2: streaming GLEBAS and building placemarks...", flush=True)
    pm = {}
    g = matched = 0
    with gzip.open(gl_file, "rt", encoding="latin-1") as f:
        f.readline()
        for line in f:
            g += 1
            p = line.rstrip("\n").split(";", 3)
            if len(p) < 4:
                continue
            key = (p[0], p[1])
            meta = ops.get(key)
            if not meta:
                continue
            rings = wkt_to_kml_rings(p[3])
            if not rings:
                continue
            pm.setdefault(key, []).append(build_placemark(p[0], p[1], meta[1], rings))
            matched += 1
    print(f"  scanned {g:,} glebas -> {matched:,} matched polygons over {len(pm):,} operations", flush=True)

    # Pass 3: write one KML per operation.
    print(f"[{year}] Pass 3: writing KML files...", flush=True)
    per_crop = {}
    written = 0
    for key, placemarks in pm.items():
        crop, emp, plantio, colheita = ops[key]
        ref, ordem = key
        crop_dir = os.path.join(out_root, f"arquivos_kml_{crop}")
        os.makedirs(crop_dir, exist_ok=True)
        fname = f"{crop}_{ref}-{ordem}_plantio_{plantio}_colheita_{colheita}.kml"
        write_kml(os.path.join(crop_dir, fname), ref, ordem, placemarks)
        per_crop[crop] = per_crop.get(crop, 0) + 1
        written += 1
        if max_files and written >= max_files:
            print(f"  [TEST] stopped at MAX_FILES={max_files}", flush=True)
            break

    print(f"\n[{year}] Done. Wrote {written:,} KML files to {out_root}")
    for crop in sorted(per_crop):
        print(f"  {crop:8}: {per_crop[crop]:,}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract_kml_sicor.py <YEAR>   (e.g. 2025)")
    main(sys.argv[1])
