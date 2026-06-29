# Memory Index

- [AVEIA↔TRIGO bottleneck](aveia-trigo-bottleneck.md) — the one confusion pair that dominates classifier accuracy; explains v4 vs v5 score gaps
- [5-crop experiment](5crops-experiment.md) — excluding ARROZ+CAFE didn't help; only a denominator effect
- [Red-edge indices added](red-edge-indices-added.md) — NDRE/CIRE/MTCI/PSRI/NDMI wired into the pipeline; backfill DONE, all 6226 rows populated
- [Second-season SOJA fix](second-season-soja-fix.md) — safrinha SOJA→MILHO; train_xgboost_v6 (B+C+E); date-ablation does NOT help; root cause is data scarcity (Feb–Mar data now collected, 0→103)
- [SAR coverage gap (safrinha)](sar-coverage-gap-safrinha.md) — SAR 64% missing for new Feb–Mar SOJA; missingness is a spurious season proxy that can fake a recall gain — validate optical-only
- [ARROZ → 6-crop model](arroz-6crop-added.md) — ARROZ added to dense v7 (0.941 macro-F1, ARROZ F1 0.99); how it was extracted + the .venv gotcha
