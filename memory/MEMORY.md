# Memory Index

> Current status/architecture SSOT: `src/model/STATUS_AND_ROADMAP.md`. Entries below are historical
> project notes from earlier pipeline generations (v4–v6) — still useful for the *why*, but check
> STATUS_AND_ROADMAP.md for current numbers.

- [AVEIA↔TRIGO bottleneck](aveia-trigo-bottleneck.md) — the one confusion pair that dominates classifier accuracy; pre-v8 numbers, core insight still holds (see STATUS_AND_ROADMAP for current figures)
- [Red-edge indices added](red-edge-indices-added.md) — NDRE/CIRE/MTCI/PSRI/NDMI wired into the pipeline; still accurate, now a permanent part of the schema
- [Second-season SOJA fix](second-season-soja-fix.md) — safrinha SOJA→MILHO fix that worked on v5/v6 data; **regressed on features_v8** (recall back down to ~0.63) — recovery is open work (Obstacle 3)
- [SAR coverage gap (safrinha)](sar-coverage-gap-safrinha.md) — SAR mostly missing for a specific Feb–Mar SOJA subset; don't confuse with the retired "SAR ~90% null" global claim (current global figure is ~16%)
