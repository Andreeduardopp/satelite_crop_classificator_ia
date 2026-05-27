"""Tests for the PU budget controller."""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from pu_budget import (
    PlanConfig,
    PUEstimator,
    PUBudgetTracker,
    select_with_budget,
    RATE_CARD,
)


def test_estimator_offplatform_8band():
    plan = PlanConfig(deliver_to_collection=False)
    est = PUEstimator(plan)
    e = est.estimate("id1", "PSScene", "ortho_analytic_8b_sr", ["clip", "harmonize"])
    # 20 (activate) + 1 (clip) + 2 (harmonize) + 0.5 GB * 200 (egress) = 123
    assert abs(e.estimated_pu - 123.0) < 0.01, e.estimated_pu


def test_estimator_collection_delivery():
    plan = PlanConfig(deliver_to_collection=True)
    est = PUEstimator(plan)
    e = est.estimate("id1", "PSScene", "ortho_analytic_8b_sr", ["clip", "harmonize"])
    # 20 + 1 + 2 + 0 = 23
    assert abs(e.estimated_pu - 23.0) < 0.01, e.estimated_pu


def test_estimator_calibration_reconciles():
    plan = PlanConfig()
    est = PUEstimator(plan)
    initial = est.calibration
    # Real cost is 2x what we estimated: calibration should drift up.
    for _ in range(20):
        est.reconcile(estimated_pu=100, actual_pu=200)
    assert est.calibration > initial * 1.5, est.calibration
    assert est.calibration <= 4.0


def test_tracker_persistence(tmp_path):
    state = tmp_path / "state.json"
    plan = PlanConfig(monthly_pu_quota=100_000)
    t1 = PUBudgetTracker(plan, state)
    t1.reserve("order1", ["item1", "item2"], estimated_pu=246.0)
    t1.finalize("order1", actual_pu=240.0, actual_size_gb=1.0)

    t2 = PUBudgetTracker(plan, state)
    assert "order1" in t2.orders
    assert t2.realized_pu() == 240.0


def test_tracker_failed_order_zero_pu(tmp_path):
    plan = PlanConfig(monthly_pu_quota=100_000)
    t = PUBudgetTracker(plan, tmp_path / "s.json")
    t.reserve("o1", ["i1"], estimated_pu=123.0)
    t.mark_failed("o1")
    assert t.realized_pu() == 0.0
    assert t.orders["o1"].state == "failed"


def test_multiplier_clamps_below_quota(tmp_path):
    plan = PlanConfig(monthly_pu_quota=400_000, target_keep_rate=0.7)
    t = PUBudgetTracker(plan, tmp_path / "s.json")
    # Plenty of headroom early in a month.
    k = t.over_order_multiplier(pu_per_scene_estimate=123.0, scenes_demanded=10)
    assert 1.0 <= k <= 3.0
    assert k >= 1.0


def test_multiplier_throttles_when_near_quota(tmp_path):
    plan = PlanConfig(monthly_pu_quota=400_000, target_keep_rate=0.7,
                      safety_headroom_frac=0.10)
    t = PUBudgetTracker(plan, tmp_path / "s.json")
    # Burn 95% of quota.
    t.reserve("big", ["x"], estimated_pu=380_000)
    t.finalize("big", actual_pu=380_000)
    k = t.over_order_multiplier(pu_per_scene_estimate=123.0, scenes_demanded=100)
    # Should be at or near the floor since we're past the safety headroom.
    assert k <= 1.1, k


def test_select_shrinks_when_unaffordable(tmp_path):
    plan = PlanConfig(monthly_pu_quota=10_000, target_keep_rate=0.7)
    est = PUEstimator(plan)
    t = PUBudgetTracker(plan, tmp_path / "s.json")

    candidates = [f"s{i}" for i in range(100)]
    decision = select_with_budget(
        candidate_item_ids=candidates,
        scenes_demanded=80,           # asks for many more than budget allows
        item_type="PSScene",
        asset_bundle="ortho_analytic_8b_sr",
        tools=["clip", "harmonize"],
        estimator=est,
        tracker=t,
    )
    # 10,000 PU budget @ ~123 PU/scene => ~73 scenes affordable post-headroom.
    assert len(decision.keep_ids) < 80
    assert decision.estimated_pu <= t.plan.monthly_pu_quota


def test_select_picks_best_candidates_first(tmp_path):
    plan = PlanConfig(monthly_pu_quota=400_000)
    est = PUEstimator(plan)
    t = PUBudgetTracker(plan, tmp_path / "s.json")
    candidates = ["best", "good", "ok", "worst"]
    decision = select_with_budget(
        candidate_item_ids=candidates,
        scenes_demanded=2,
        item_type="PSScene",
        asset_bundle="ortho_analytic_8b_sr",
        tools=["clip", "harmonize"],
        estimator=est,
        tracker=t,
    )
    # Order preserved — "best" must be first.
    assert decision.keep_ids[0] == "best"


def test_month_rollover_resets_budget(tmp_path):
    state = tmp_path / "s.json"
    plan = PlanConfig(monthly_pu_quota=400_000)
    t1 = PUBudgetTracker(plan, state)
    t1.reserve("o1", ["i1"], estimated_pu=200_000)
    t1.finalize("o1", actual_pu=200_000)
    t1.persist()

    # Simulate next month.
    raw = json.loads(state.read_text())
    raw["month_anchor"] = "1999-01"  # stale anchor
    state.write_text(json.dumps(raw))

    t2 = PUBudgetTracker(plan, state)
    assert t2.realized_pu() == 0.0
    assert len(t2.orders) == 0


if __name__ == "__main__":
    import tempfile

    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
