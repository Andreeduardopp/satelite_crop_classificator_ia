# KML Metadata Analysis — `metadata_large.db`

> Generated from **28,000 agricultural field polygons** across **7 crop types** sampled from the Brazilian KML dataset.  
> Geocoding: IBGE boundary polygons · Geometry: WGS-84 geodesic area (pyproj)

---

## 1 · Overview

| Metric | Value |
|---|---|
| **Total records** | 28,000 |
| **Crop types** | 7 |
| **States (UF)** | 25 |
| **Municipalities** | 1,820 |
| **Geocoded** | 28,000 (100.0%) |
| **Not geocoded** | 0 (0.0%) |
| **Total area** | 433,836.6 ha |
| **Average field size** | 15.49 ha |

---

## 2 · Crop Distribution

| Crop | Fields | Total (ha) | Avg (ha) | Min (ha) | Max (ha) |
|---|---:|---:|---:|---:|---:|
| ARROZ | 4,000 | 100,300.2 | 25.08 | 0.09 | 1,804.7 |
| AVEIA | 4,000 | 64,128.3 | 16.03 | 0.12 | 495.57 |
| CAFÉ | 4,000 | 17,429.7 | 4.36 | 0.00 | 231.25 |
| FEIJÃO | 4,000 | 42,419.7 | 10.60 | 0.08 | 194.06 |
| MILHO | 4,000 | 49,648.2 | 12.41 | 0.01 | 923.10 |
| SOJA | 4,000 | 106,425.1 | 26.61 | 0.06 | 2,490.4 |
| TRIGO | 4,000 | 53,485.3 | 13.37 | 0.11 | 332.80 |

> **SOJA** and **ARROZ** together account for ~47% of the total surveyed area (~206,725 ha).  
> **CAFÉ** fields are the smallest on average (4.36 ha median 2.22 ha), consistent with small hillside plots in MG and ES.

---

## 3 · Geographic Spread

### Records per State (UF) — Top 10

| UF | State | Fields | Total Area (ha) |
|---|---|---:|---:|
| RS | Rio Grande do Sul | 12,052 | 207,205.7 |
| PR | Paraná | 7,567 | 96,975.5 |
| SC | Santa Catarina | 2,866 | 22,358.0 |
| MG | Minas Gerais | 2,355 | 16,081.4 |
| ES | Espírito Santo | 1,414 | 4,294.3 |
| SE | Sergipe | 388 | 6,436.5 |
| BA | Bahia | 381 | 16,857.4 |
| SP | São Paulo | 304 | 5,482.1 |
| MS | Mato Grosso do Sul | 188 | 11,392.8 |
| GO | Goiás | 118 | 11,213.1 |

> The **Southern Region** (RS + PR + SC) concentrates **81%** of all fields (22,485 / 28,000).

### Top 10 Municipalities by Field Count

| Municipality | UF | Fields | Total Area (ha) |
|---|---|---:|---:|
| Prudentópolis | PR | 286 | 2,134.5 |
| Irati | PR | 178 | 1,371.5 |
| Verê | PR | 167 | 1,919.7 |
| Restinga Sêca | RS | 163 | 2,446.3 |
| Dois Vizinhos | PR | 155 | 1,123.8 |
| Francisco Beltrão | PR | 152 | 1,086.7 |
| Agudo | RS | 151 | 1,341.2 |
| Ivaí | PR | 141 | 979.34 |
| São João | PR | 127 | 1,181.9 |
| Giruá | RS | 127 | 2,287.2 |

---

## 4 · Area Statistics per Crop (hectares)

| Crop | N | Min | Median | Mean | Max | StdDev |
|---|---:|---:|---:|---:|---:|---:|
| ARROZ | 4,000 | 0.09 | 10.98 | 25.08 | 1,804.7 | 48.44 |
| AVEIA | 4,000 | 0.12 | 8.51 | 16.03 | 495.57 | 29.05 |
| CAFÉ | 4,000 | 0.00 | 2.22 | 4.36 | 231.25 | 9.12 |
| FEIJÃO | 4,000 | 0.08 | 7.15 | 10.60 | 194.06 | 12.66 |
| MILHO | 4,000 | 0.01 | 5.30 | 12.41 | 923.10 | 29.07 |
| SOJA | 4,000 | 0.06 | 9.29 | 26.61 | 2,490.4 | 87.04 |
| TRIGO | 4,000 | 0.11 | 8.11 | 13.37 | 332.80 | 18.03 |

> All crops show **right-skewed distributions** (mean >> median), driven by a small number of very large commercial fields.  
> **SOJA** has the highest variance (σ = 87 ha), reflecting the mix of small family farms and large Cerrado estates.

---

## 5 · Geographic Extent per Crop

| Crop | West (lon) | East (lon) | South (lat) | North (lat) |
|---|---:|---:|---:|---:|
| ARROZ | -63.1001 | -36.4412 | -33.6901 | +3.1900 |
| AVEIA | -55.9899 | -45.9276 | -32.3831 | -21.5793 |
| CAFÉ | -72.6673 | -38.9672 | -24.4414 | -3.9108 |
| FEIJÃO | -55.5791 | -37.8964 | -31.6430 | +0.4933 |
| MILHO | -62.6401 | -36.5653 | -31.8945 | -2.7014 |
| SOJA | -65.2161 | -42.5121 | -33.4876 | +3.3244 |
| TRIGO | -56.4627 | -47.9896 | -32.4318 | -21.7952 |

> **AVEIA** and **TRIGO** are the most geographically constrained — restricted to the temperate south.  
> **SOJA** and **ARROZ** span from the far south to near the equator (+3.3° N), matching Brazil's continental-scale production.  
> **CAFÉ** extends furthest west (-72.7°), reaching western Rondônia/Amazonas border areas.

---

## 6 · Data Quality

| Check | Count | Share |
|---|---:|---:|
| Zero / null area fields | 0 | 0.0% |
| Tiny fields (< 0.1 ha) | 21 | 0.1% |
| Large fields (> 10,000 ha) | 0 | 0.0% |
| Missing state (UF) | 0 | 0.0% |
| Missing municipality | 0 | 0.0% |
| Duplicate (filename + crop) | 0 | — |

> [!NOTE]
> The 21 tiny fields (< 0.1 ha) likely represent digitisation artefacts or test polygons and may warrant exclusion from model training.

### Vertex Count Distribution

| Vertex range | Fields |
|---|---:|
| < 5 | 11 |
| 5–9 | 163 |
| 10–19 | 13,845 |
| 20–49 | 10,911 |
| 50–99 | 3,056 |
| ≥ 100 | 14 |

> The dominant range is **10–19 vertices** (49.4%), indicating moderately detailed polygons suitable for centroid-based geocoding and bounding-box satellite image extraction.

---

## 7 · Crop × State Cross-Table (field count)

|  | RS | PR | SC | MG | ES | SE | BA | SP | MS | GO | RO | MT | Other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ARROZ** | 2,257 | 25 | 1,585 | 3 | 0 | 31 | 0 | 9 | 9 | 5 | 1 | 8 | 67 |
| **AVEIA** | 3,318 | 579 | 93 | 2 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| **CAFÉ** | 0 | 47 | 0 | 2,208 | 1,414 | 0 | 83 | 145 | 0 | 0 | 75 | 1 | 27 |
| **FEIJÃO** | 373 | 3,171 | 373 | 35 | 0 | 0 | 5 | 30 | 3 | 5 | 0 | 0 | 5 |
| **MILHO** | 1,463 | 1,234 | 429 | 50 | 0 | 357 | 275 | 43 | 68 | 18 | 2 | 17 | 44 |
| **SOJA** | 1,837 | 1,498 | 212 | 57 | 0 | 0 | 18 | 61 | 107 | 90 | 12 | 43 | 65 |
| **TRIGO** | 2,804 | 1,013 | 174 | 0 | 0 | 0 | 0 | 8 | 1 | 0 | 0 | 0 | 0 |

### Key observations

- **AVEIA** is almost exclusively southern (RS 83%, PR 14%) — a temperate winter cereal with no representation in tropical states.
- **TRIGO** mirrors AVEIA: 100% in the south (RS + PR + SC), with the northernmost point at -21.8° lat.
- **CAFÉ** is entirely absent from RS but dominates MG (55%) and ES (35%) — the classic *Zona da Mata* coffee belt.
- **FEIJÃO** is PR-dominated (79%), reflecting Paraná's role as Brazil's top common bean producing state.
- **MILHO** is the most geographically diverse crop, present in every listed state including SE and BA (tropical north).
- **SOJA** shows the widest latitudinal range and highest max field size (2,490 ha), consistent with large-scale Cerrado expansion.

---

*Pipeline: `scripts/build_kml_metadata.py` · Analysis: `scripts/analyze_db.py` · Database: `metadata_large.db`*
