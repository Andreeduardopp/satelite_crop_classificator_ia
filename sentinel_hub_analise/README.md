# PU-aware `select` stage for Planet pipeline

A budget controller that reads `x-processunits` / completed-order PU consumption
from Planet, tracks burn rate against a monthly quota, and dynamically tightens
the over-order multiplier in the `select` stage as quota approaches exhaustion.

Built for the Enterprise S plan (400,000 PU/month) but parameterized.

## Files

- `pu_budget.py` — estimator, tracker, and `select_with_budget()` core logic. No Planet SDK dep.
- `planet_integration.py` — Orders API wiring: build → submit → reconcile loop.
- `test_pu_budget.py` — unit tests for the controller.
- `simulate_month.py` — month-long simulation to validate controller behavior.

## Rate card (verified against Planet docs, May 2026)

| Item | PU |
|---|---|
| Asset activation | 20 |
| `clip` tool | 1 |
| `harmonize` tool | 2 |
| Egress (off-platform) | 200 / GB |
| Egress to Planet image collection | 0 |

Tools that aren't on the published rate card (`reproject`, `bandmath`, `composite`)
are estimated at 1 PU. The reconciliation loop calibrates the estimator against
realized PU, so initial inaccuracies wash out within a few orders.

## Usage

```python
from pathlib import Path
from pu_budget import PlanConfig
from planet_integration import run_select_stage

plan = PlanConfig(
    monthly_pu_quota=400_000,         # Enterprise S
    safety_headroom_frac=0.10,        # reserve 10% for end-of-month
    target_keep_rate=0.70,            # how many ordered scenes you expect to keep
    max_over_order_multiplier=3.0,
    deliver_to_collection=False,      # set True if delivering to PIP image collection
)

result = run_select_stage(
    candidate_groups=[
        {"name": "aoi_lisbon_2026-05-18", "item_ids": top_ranked_ids, "scenes_demanded": 4},
        ...
    ],
    plan=plan,
    state_path=Path("/var/state/planet_pu_budget.json"),
    aoi_geojson=your_aoi,
    item_type="PSScene",
    product_bundle="analytic_8b_sr_udm2",
    harmonize=True,
)
```

`run_select_stage` is idempotent across pipeline restarts: state is persisted
to JSON, and the month-anchor check resets the budget on the 1st of each month.

## Important: PU field reconciliation

`planet_integration.py` extracts realized PU from the completed-order details
payload by probing a list of candidate field names (`EXTRACT_PU_KEYS`). The
exact field name has shifted across API versions, so:

1. After your first successful order, dump the full `pl.orders.get_order(order_id)`
   payload and confirm which key holds the PU count.
2. Pin that key at the top of `EXTRACT_PU_KEYS`.

Until reconciled, the calibration coefficient stays at 1.0 — meaning estimates
are used as-is. Worth checking on day 1.

## Controller behavior

The over-order multiplier `k` starts from `1 / target_keep_rate` and is
dampened by the lesser of:

- **pace ratio**: `(budget_frac_remaining) / (time_frac_remaining)`
- **envelope ratio**: `usable_pu_remaining / (ideal_k * scenes_demanded * pu_per_scene)`

If we're ahead of pace (lots of headroom, plenty of month left), `k` can
exceed `1/keep_rate` up to `max_over_order_multiplier`. If we're behind,
`k` collapses toward `1.0`.

The simulation in `simulate_month.py` shows three regimes:
- **Off-platform, normal load**: ends at ~29% burn, `k` ramps 1.4 → 3.0 as confidence grows.
- **In-collection, normal load**: ends at ~5% burn (no egress); same `k` ramp.
- **Off-platform, overcommitted**: ends at ~75% burn (would have been 200%+ uncontrolled).

## Things still worth confirming with the AM

- Whether your contract counts clipped-area or full-scene footprint toward PU consumption (affects `DEFAULT_SCENE_SIZE_GB`).
- Whether `reproject`, `bandmath`, `composite` PU costs match the assumed 1 PU each.
- Whether delivery to your specific cloud bucket counts as "off-platform" egress (it should).
- Top-up pricing if you blow through quota — you want to know the unit cost before the controller's safety headroom kicks in.

## Tests

```bash
python test_pu_budget.py
python simulate_month.py
```
