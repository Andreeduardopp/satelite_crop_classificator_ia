"""
PU budget tracker and adaptive over-order controller for the `select` stage.

Design
------
At order-submit time we don't know the true PU cost of an order. We can estimate it
from the rate card (activation + tools + egress) but real consumption depends on
final clipped file sizes and which sub-tools fire. So we run a two-loop controller:

  1. Pre-submit: estimate PU cost per candidate scene using the rate card.
     Reject or scale the over-order multiplier so the projected month-to-date
     burn stays inside the budget envelope.

  2. Post-completion: after an order reaches `success`, reconcile the estimate
     against actual PU consumed (parsed from the order details response, or
     from x-processunits headers on Subscriptions/Stats responses). Update a
     calibration coefficient `c` that scales future estimates: c <- c * (actual/est),
     EMA-smoothed.

The over-order multiplier `k` is computed from:
  - days remaining in the billing month
  - PU remaining
  - target keep rate (fraction of ordered scenes that pass post-hoc QC)
  - a safety headroom (e.g. 10% of monthly quota reserved for end-of-month)

The controller is intentionally conservative near quota exhaustion because
Planet returns 403 on order creation and watermarked tiles when PU hits zero.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Rate card (PU). Source: Planet docs, May 2026.
# All numbers per asset unless otherwise noted.
# ---------------------------------------------------------------------------
RATE_CARD = {
    "activation_per_asset": 20,   # PU
    "clip_tool": 1,               # PU per asset processed
    "harmonize_tool": 2,          # PU per asset processed
    "reproject_tool": 1,          # PU per asset (assumed; verify with x-processunits)
    "bandmath_tool": 1,           # PU per asset (assumed; verify)
    "composite_tool": 1,          # PU per asset (assumed; verify)
    "egress_per_gb": 200,         # PU per GB delivered off-platform
    "egress_to_collection": 0,    # PU for delivery to Planet image collection
}

# Empirical mean clipped scene sizes (GB). Tune from your own AOIs.
DEFAULT_SCENE_SIZE_GB = {
    ("PSScene", "ortho_analytic_4b_sr"): 0.25,
    ("PSScene", "ortho_analytic_8b_sr"): 0.50,
    ("PSScene", "ortho_visual"):         0.08,
}


@dataclass
class PlanConfig:
    """Subscription plan configuration."""
    monthly_pu_quota: int = 400_000          # Enterprise S
    safety_headroom_frac: float = 0.10       # reserve 10% for end-of-month
    target_keep_rate: float = 0.70           # fraction of ordered scenes you expect to keep
    max_over_order_multiplier: float = 3.0
    min_over_order_multiplier: float = 1.0
    deliver_to_collection: bool = False      # True => egress = 0


@dataclass
class OrderEstimate:
    """Pre-submit estimate for a single scene-order."""
    item_id: str
    item_type: str
    asset_bundle: str
    tools: list[str]
    estimated_size_gb: float
    estimated_pu: float


@dataclass
class OrderRecord:
    """Realized record for a completed order."""
    order_id: str
    item_ids: list[str]
    estimated_pu: float
    actual_pu: Optional[float] = None
    actual_size_gb: Optional[float] = None
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    state: str = "submitted"  # submitted | success | failed | partial


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------
class PUEstimator:
    """
    Predicts PU cost per scene-order given bundle, tools, and delivery mode.
    Maintains a calibration coefficient `c` updated by reconciliation.
    """

    def __init__(self, plan: PlanConfig, scene_sizes: Optional[dict] = None):
        self.plan = plan
        self.scene_sizes = scene_sizes or dict(DEFAULT_SCENE_SIZE_GB)
        self.calibration = 1.0           # multiplier applied to estimates
        self._cal_ema_alpha = 0.2        # smoothing for calibration updates

    def estimate(
        self,
        item_id: str,
        item_type: str,
        asset_bundle: str,
        tools: list[str],
    ) -> OrderEstimate:
        size_gb = self.scene_sizes.get((item_type, asset_bundle), 0.5)

        pu = RATE_CARD["activation_per_asset"]
        for t in tools:
            key = f"{t}_tool"
            if key in RATE_CARD:
                pu += RATE_CARD[key]

        if self.plan.deliver_to_collection:
            pu += RATE_CARD["egress_to_collection"]
        else:
            pu += size_gb * RATE_CARD["egress_per_gb"]

        pu *= self.calibration

        return OrderEstimate(
            item_id=item_id,
            item_type=item_type,
            asset_bundle=asset_bundle,
            tools=tools,
            estimated_size_gb=size_gb,
            estimated_pu=pu,
        )

    def reconcile(self, estimated_pu: float, actual_pu: float) -> None:
        """Update calibration coefficient using EMA of actual/estimated ratio."""
        if estimated_pu <= 0 or actual_pu <= 0:
            return
        ratio = actual_pu / estimated_pu
        self.calibration = (
            (1 - self._cal_ema_alpha) * self.calibration
            + self._cal_ema_alpha * (self.calibration * ratio)
        )
        # Clamp to sane range so a single bad reconciliation can't blow up.
        self.calibration = max(0.25, min(4.0, self.calibration))


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------
class PUBudgetTracker:
    """
    Tracks month-to-date PU spend and computes the over-order multiplier
    the `select` stage should apply.

    Thread-safe. Persistable to a JSON file so it survives pipeline restarts.
    """

    def __init__(
        self,
        plan: PlanConfig,
        state_path: Optional[Path] = None,
    ):
        self.plan = plan
        self.state_path = state_path
        self._lock = threading.Lock()
        self.orders: dict[str, OrderRecord] = {}
        self.month_anchor: str = self._current_month_key()
        if state_path and state_path.exists():
            self._load()

    # ---- Persistence -------------------------------------------------------
    def _load(self) -> None:
        data = json.loads(self.state_path.read_text())
        if data.get("month_anchor") != self._current_month_key():
            # New billing month: PUs reset, drop prior records.
            self.month_anchor = self._current_month_key()
            return
        self.month_anchor = data["month_anchor"]
        for rid, rec in data["orders"].items():
            self.orders[rid] = OrderRecord(**rec)

    def persist(self) -> None:
        if not self.state_path:
            return
        payload = {
            "month_anchor": self.month_anchor,
            "orders": {rid: asdict(r) for rid, r in self.orders.items()},
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    # ---- Time helpers ------------------------------------------------------
    @staticmethod
    def _current_month_key() -> str:
        now = datetime.now(timezone.utc)
        return f"{now.year:04d}-{now.month:02d}"

    @staticmethod
    def _days_remaining_in_month() -> float:
        now = datetime.now(timezone.utc)
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            next_month = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
        delta = next_month - now
        # +epsilon avoids div/0 at month boundary
        return max(delta.total_seconds() / 86400.0, 1e-3)

    @staticmethod
    def _days_in_current_month() -> float:
        now = datetime.now(timezone.utc)
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return (next_month.replace(hour=0, minute=0, second=0, microsecond=0) - first).total_seconds() / 86400.0

    # ---- Bookkeeping -------------------------------------------------------
    def reserve(self, order_id: str, item_ids: list[str], estimated_pu: float) -> None:
        with self._lock:
            self.orders[order_id] = OrderRecord(
                order_id=order_id,
                item_ids=item_ids,
                estimated_pu=estimated_pu,
            )
            self.persist()

    def finalize(
        self,
        order_id: str,
        actual_pu: float,
        actual_size_gb: Optional[float] = None,
        state: str = "success",
    ) -> None:
        with self._lock:
            rec = self.orders.get(order_id)
            if not rec:
                return
            rec.actual_pu = actual_pu
            rec.actual_size_gb = actual_size_gb
            rec.state = state
            rec.completed_at = datetime.now(timezone.utc).isoformat()
            self.persist()

    def mark_failed(self, order_id: str) -> None:
        # Failed orders do NOT consume PU per Planet docs:
        # "PUs are deducted only when a request successfully executes."
        with self._lock:
            rec = self.orders.get(order_id)
            if rec:
                rec.state = "failed"
                rec.actual_pu = 0.0
                rec.completed_at = datetime.now(timezone.utc).isoformat()
                self.persist()

    # ---- Reporting ---------------------------------------------------------
    def realized_pu(self) -> float:
        return sum((r.actual_pu or 0.0) for r in self.orders.values())

    def in_flight_pu(self) -> float:
        return sum(
            r.estimated_pu for r in self.orders.values()
            if r.state == "submitted"
        )

    def projected_pu(self) -> float:
        return self.realized_pu() + self.in_flight_pu()

    def pu_remaining(self) -> float:
        return max(self.plan.monthly_pu_quota - self.projected_pu(), 0.0)

    def usable_pu_remaining(self) -> float:
        """PU available after reserving safety headroom."""
        reserve = self.plan.monthly_pu_quota * self.plan.safety_headroom_frac
        return max(self.pu_remaining() - reserve, 0.0)

    # ---- The thing `select` actually calls ---------------------------------
    def over_order_multiplier(
        self,
        pu_per_scene_estimate: float,
        scenes_demanded: int,
    ) -> float:
        """
        Compute the over-order multiplier k for the `select` stage.

        Starts from the "ideal" k = 1 / target_keep_rate (e.g. 1.43 at 70% keep)
        then dampens it based on remaining budget vs remaining time.
        """
        if scenes_demanded <= 0 or pu_per_scene_estimate <= 0:
            return self.plan.min_over_order_multiplier

        ideal_k = 1.0 / self.plan.target_keep_rate  # e.g. 1.43

        # Budget pacing: fraction of month remaining vs fraction of budget remaining.
        days_left = self._days_remaining_in_month()
        days_total = self._days_in_current_month()
        time_frac = days_left / days_total
        budget_frac = self.usable_pu_remaining() / max(
            self.plan.monthly_pu_quota * (1 - self.plan.safety_headroom_frac), 1.0
        )

        # If budget_frac > time_frac, we're ahead of pace; allow higher k.
        # If budget_frac < time_frac, we're behind; throttle k toward 1.0.
        pace_ratio = budget_frac / max(time_frac, 1e-3)

        # PU envelope check for this specific request.
        pu_needed_at_ideal = ideal_k * scenes_demanded * pu_per_scene_estimate
        envelope_ratio = self.usable_pu_remaining() / max(pu_needed_at_ideal, 1.0)

        # Combine: the binding constraint wins.
        k = ideal_k * min(pace_ratio, envelope_ratio, 1.0)

        # If we're early in the month and well-funded, allow over-shooting ideal_k
        # up to max_over_order_multiplier.
        if pace_ratio > 1.2 and envelope_ratio > 1.5:
            k = min(ideal_k * pace_ratio, self.plan.max_over_order_multiplier)

        return max(self.plan.min_over_order_multiplier,
                   min(k, self.plan.max_over_order_multiplier))

    # ---- Diagnostics -------------------------------------------------------
    def status(self) -> dict:
        return {
            "month": self.month_anchor,
            "quota": self.plan.monthly_pu_quota,
            "realized_pu": round(self.realized_pu(), 1),
            "in_flight_pu": round(self.in_flight_pu(), 1),
            "projected_pu": round(self.projected_pu(), 1),
            "pu_remaining": round(self.pu_remaining(), 1),
            "usable_pu_remaining": round(self.usable_pu_remaining(), 1),
            "days_remaining": round(self._days_remaining_in_month(), 2),
            "burn_pct": round(100 * self.projected_pu() / self.plan.monthly_pu_quota, 1),
        }


# ---------------------------------------------------------------------------
# The `select` stage entrypoint
# ---------------------------------------------------------------------------
@dataclass
class SelectDecision:
    keep_ids: list[str]
    over_order_multiplier: float
    estimated_pu: float
    rejected_reason: Optional[str] = None


def select_with_budget(
    candidate_item_ids: list[str],
    scenes_demanded: int,
    item_type: str,
    asset_bundle: str,
    tools: list[str],
    estimator: PUEstimator,
    tracker: PUBudgetTracker,
) -> SelectDecision:
    """
    Picks how many of `candidate_item_ids` to actually order, given the budget.

    Assumes candidates are pre-sorted by your quality score (cloud cover,
    clear_percent, view_angle, sun_elevation, etc.) — best first.
    """
    if not candidate_item_ids:
        return SelectDecision([], 1.0, 0.0, "no candidates")

    # Estimate per-scene cost (use first candidate as representative;
    # tighten by averaging over a sample if AOI varies significantly).
    sample_est = estimator.estimate(
        candidate_item_ids[0], item_type, asset_bundle, tools
    )
    pu_per_scene = sample_est.estimated_pu

    k = tracker.over_order_multiplier(pu_per_scene, scenes_demanded)
    n_order = math.ceil(scenes_demanded * k)
    n_order = min(n_order, len(candidate_item_ids))

    total_est = n_order * pu_per_scene

    if total_est > tracker.usable_pu_remaining():
        # Shrink to what we can afford.
        affordable = math.floor(tracker.usable_pu_remaining() / pu_per_scene)
        n_order = max(min(affordable, n_order), 0)
        total_est = n_order * pu_per_scene
        if n_order < scenes_demanded:
            return SelectDecision(
                keep_ids=candidate_item_ids[:n_order],
                over_order_multiplier=k,
                estimated_pu=total_est,
                rejected_reason=f"budget exhausted: only {n_order}/{scenes_demanded} affordable",
            )

    return SelectDecision(
        keep_ids=candidate_item_ids[:n_order],
        over_order_multiplier=k,
        estimated_pu=total_est,
    )
