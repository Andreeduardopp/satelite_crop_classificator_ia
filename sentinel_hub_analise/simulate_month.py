"""
Simulate a month of select-stage decisions to validate the budget controller.

Generates daily demand for scenes across N AOIs, runs the controller, and
plots PU burn vs. ideal pace. Useful to confirm the multiplier converges
sensibly across an entire billing cycle.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from pu_budget import PlanConfig, PUEstimator, PUBudgetTracker, select_with_budget

import random
random.seed(42)


def simulate_month(
    quota: int = 400_000,
    n_aois: int = 12,
    scenes_per_aoi_per_day: int = 1,
    deliver_to_collection: bool = False,
    target_keep_rate: float = 0.70,
):
    """Step through 30 days; each day, each AOI demands some scenes."""
    plan = PlanConfig(
        monthly_pu_quota=quota,
        target_keep_rate=target_keep_rate,
        deliver_to_collection=deliver_to_collection,
    )

    daily_burn = []
    daily_multiplier = []
    daily_remaining = []

    # We mock time-of-month by patching the days_remaining helpers.
    for day in range(1, 31):
        days_remaining = 31 - day  # rough

        def fake_remaining():
            return float(days_remaining if days_remaining > 0 else 0.1)
        def fake_total():
            return 30.0

        with patch.object(PUBudgetTracker, "_days_remaining_in_month",
                          staticmethod(fake_remaining)), \
             patch.object(PUBudgetTracker, "_days_in_current_month",
                          staticmethod(fake_total)):

            # Use a fresh tracker per simulation but seed it with prior orders
            # to preserve burn state across days.
            if day == 1:
                estimator = PUEstimator(plan)
                tracker = PUBudgetTracker(plan, state_path=None)

            day_k_samples = []
            for aoi in range(n_aois):
                # Candidates: 6 per AOI per day, demand = 1
                candidates = [f"d{day}_a{aoi}_s{i}" for i in range(6)]
                decision = select_with_budget(
                    candidate_item_ids=candidates,
                    scenes_demanded=scenes_per_aoi_per_day,
                    item_type="PSScene",
                    asset_bundle="ortho_analytic_8b_sr",
                    tools=["clip", "harmonize"],
                    estimator=estimator,
                    tracker=tracker,
                )
                day_k_samples.append(decision.over_order_multiplier)

                # Submit each picked scene as a single-asset order.
                for sid in decision.keep_ids:
                    order_id = f"ord_{day}_{aoi}_{sid}"
                    est_pu = decision.estimated_pu / max(len(decision.keep_ids), 1)
                    tracker.reserve(order_id, [sid], estimated_pu=est_pu)
                    # Simulate completion: actual PU is est ± 15% noise.
                    actual = est_pu * random.uniform(0.85, 1.15)
                    tracker.finalize(order_id, actual_pu=actual)
                    estimator.reconcile(est_pu, actual)

            daily_burn.append(tracker.realized_pu())
            daily_multiplier.append(
                sum(day_k_samples) / len(day_k_samples) if day_k_samples else 0
            )
            daily_remaining.append(tracker.pu_remaining())

    return {
        "daily_burn": daily_burn,
        "daily_multiplier": daily_multiplier,
        "daily_remaining": daily_remaining,
        "final_calibration": estimator.calibration,
        "final_status": tracker.status(),
    }


def print_report(title, result, quota):
    print(f"\n=== {title} ===")
    print(f"Final calibration: {result['final_calibration']:.3f}")
    print(f"Final status: {result['final_status']}")
    print(f"\n{'day':>4} {'cum_burn':>10} {'pct':>6} {'k':>6} {'remaining':>10} {'ideal_pace':>10}")
    for day in range(1, 31):
        i = day - 1
        burn = result['daily_burn'][i]
        k = result['daily_multiplier'][i]
        rem = result['daily_remaining'][i]
        ideal = quota * (day / 30)
        marker = "  <-- over" if burn > ideal * 1.1 else ("  <-- under" if burn < ideal * 0.5 else "")
        print(f"{day:>4} {burn:>10.0f} {100*burn/quota:>5.1f}% {k:>6.2f} {rem:>10.0f} {ideal:>10.0f}{marker}")


if __name__ == "__main__":
    # Scenario A: off-platform delivery (egress dominates)
    quota = 400_000
    res_a = simulate_month(quota=quota, n_aois=12, deliver_to_collection=False)
    print_report("Off-platform, 12 AOIs, 1 scene/AOI/day", res_a, quota)

    # Scenario B: in-collection delivery (no egress cost)
    res_b = simulate_month(quota=quota, n_aois=12, deliver_to_collection=True)
    print_report("In-collection, 12 AOIs, 1 scene/AOI/day", res_b, quota)

    # Scenario C: aggressive demand that should trigger throttling
    res_c = simulate_month(quota=quota, n_aois=40, deliver_to_collection=False)
    print_report("Off-platform, 40 AOIs (overcommitted)", res_c, quota)
