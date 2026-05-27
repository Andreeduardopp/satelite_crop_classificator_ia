"""
Planet Orders API integration for the PU-aware `select` stage.

This wires the budget controller into the actual order-submit and
order-monitor loops. Two integration points:

  1. submit_budgeted_order: estimate -> reserve PU -> submit -> store order_id
  2. reconcile_completed_orders: poll for completed orders, extract realized
     PU consumption, feed it back to the estimator's calibration.

PU reconciliation notes
-----------------------
- Orders API: PU usage is reported in the completed order details. Field name
  has shifted across API versions; the parser below checks the documented
  fields and falls back to scanning the response for a `*processing_units*`
  or similar key.
- Subscriptions API & Statistical API: realized PU appears in the
  `x-processunits` response header per request.
- Failed orders consume 0 PU (Planet: "PUs are deducted only when a request
  successfully executes").

You'll want to verify the exact field name once against your own account by
inspecting a completed order JSON, then pin EXTRACT_PU_KEYS below.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable, Optional

from planet import Planet, Session, order_request

from pu_budget import (
    PlanConfig,
    PUBudgetTracker,
    PUEstimator,
    select_with_budget,
)

log = logging.getLogger("pu_pipeline")

# Candidate fields where Planet may report realized PU on a completed order.
# Probe these in order; first hit wins. Verify against your own response shape.
EXTRACT_PU_KEYS = (
    "processing_units",
    "pu_consumed",
    "compute_units",
    "billing", # nested object with .processing_units
)


# ---------------------------------------------------------------------------
# Order construction
# ---------------------------------------------------------------------------
def build_order_request(
    name: str,
    item_ids: list[str],
    item_type: str,
    product_bundle: str,
    aoi_geojson: dict,
    *,
    harmonize: bool = True,
    deliver_to_collection: bool = False,
    collection_id: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    gcs_credentials: Optional[str] = None,
) -> dict:
    """Build an Orders API request payload with clip + optional harmonize."""
    tools = [order_request.clip_tool(aoi=aoi_geojson)]
    if harmonize:
        tools.append(order_request.harmonize_tool(target_sensor="Sentinel-2"))

    delivery = None
    if deliver_to_collection and collection_id:
        # Replace with the actual delivery helper for your platform target.
        delivery = {"image_collection": {"id": collection_id}}
    elif gcs_bucket and gcs_credentials:
        delivery = order_request.google_cloud_storage(
            bucket=gcs_bucket,
            credentials=gcs_credentials,
        )

    request = order_request.build_request(
        name=name,
        products=[
            order_request.product(
                item_ids=item_ids,
                product_bundle=product_bundle,
                item_type=item_type,
            )
        ],
        tools=tools,
        delivery=delivery,
    )
    return request


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
def submit_budgeted_order(
    pl: Planet,
    *,
    candidate_item_ids: list[str],
    scenes_demanded: int,
    item_type: str,
    product_bundle: str,
    aoi_geojson: dict,
    tools_used: list[str],
    estimator: PUEstimator,
    tracker: PUBudgetTracker,
    order_name: str,
    harmonize: bool = True,
    deliver_to_collection: bool = False,
    collection_id: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    gcs_credentials: Optional[str] = None,
) -> Optional[str]:
    """
    Run the `select` stage with budget awareness and submit the resulting order.
    Returns the Planet order_id or None if budget rejected the request.
    """
    decision = select_with_budget(
        candidate_item_ids=candidate_item_ids,
        scenes_demanded=scenes_demanded,
        item_type=item_type,
        asset_bundle=product_bundle,
        tools=tools_used,
        estimator=estimator,
        tracker=tracker,
    )

    log.info(
        "select decision: k=%.2f, picked=%d/%d candidates, est_pu=%.1f%s",
        decision.over_order_multiplier,
        len(decision.keep_ids),
        len(candidate_item_ids),
        decision.estimated_pu,
        f" (rejected: {decision.rejected_reason})" if decision.rejected_reason else "",
    )

    if not decision.keep_ids:
        log.warning("Budget rejected: %s", decision.rejected_reason)
        return None

    request = build_order_request(
        name=order_name,
        item_ids=decision.keep_ids,
        item_type=item_type,
        product_bundle=product_bundle,
        aoi_geojson=aoi_geojson,
        harmonize=harmonize,
        deliver_to_collection=deliver_to_collection,
        collection_id=collection_id,
        gcs_bucket=gcs_bucket,
        gcs_credentials=gcs_credentials,
    )

    order = pl.orders.create_order(request)
    order_id = order["id"]

    tracker.reserve(
        order_id=order_id,
        item_ids=decision.keep_ids,
        estimated_pu=decision.estimated_pu,
    )

    log.info("submitted order %s with %d scenes", order_id, len(decision.keep_ids))
    return order_id


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------
def _extract_actual_pu(order_details: dict) -> Optional[float]:
    """Best-effort extraction of realized PU from an order details payload."""
    for key in EXTRACT_PU_KEYS:
        if key in order_details:
            val = order_details[key]
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                for sub in ("processing_units", "pu", "amount", "value"):
                    if sub in val and isinstance(val[sub], (int, float)):
                        return float(val[sub])

    # Last resort: scan top-level keys for anything that looks like PU.
    for k, v in order_details.items():
        if "process" in k.lower() and "unit" in k.lower() and isinstance(v, (int, float)):
            return float(v)
    return None


def _extract_actual_size_gb(order_details: dict) -> Optional[float]:
    """Best-effort extraction of total delivered bytes -> GB."""
    # Common shapes: _links has results, or metadata.delivery_size_bytes
    bytes_total = 0
    for key in ("total_bytes", "delivery_size_bytes", "size_bytes"):
        if key in order_details and isinstance(order_details[key], (int, float)):
            return float(order_details[key]) / (1024 ** 3)
    results = order_details.get("_links", {}).get("results") or []
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("size"), (int, float)):
            bytes_total += r["size"]
    if bytes_total > 0:
        return bytes_total / (1024 ** 3)
    return None


def reconcile_completed_orders(
    pl: Planet,
    estimator: PUEstimator,
    tracker: PUBudgetTracker,
) -> None:
    """
    Poll Planet for the state of every order we've reserved budget for,
    and feed realized PU back into the estimator.
    """
    in_flight = [oid for oid, rec in tracker.orders.items()
                 if rec.state == "submitted"]

    for order_id in in_flight:
        try:
            details = pl.orders.get_order(order_id)
        except Exception as e:
            log.warning("could not fetch order %s: %s", order_id, e)
            continue

        state = details.get("state", "unknown")
        if state in ("queued", "running"):
            continue

        if state == "failed":
            tracker.mark_failed(order_id)
            log.info("order %s failed: 0 PU charged", order_id)
            continue

        actual_pu = _extract_actual_pu(details)
        actual_size = _extract_actual_size_gb(details)

        if actual_pu is None:
            log.warning(
                "order %s reached %s but PU field not found; "
                "leaving as in_flight (verify EXTRACT_PU_KEYS)",
                order_id, state,
            )
            continue

        reserved = tracker.orders[order_id].estimated_pu
        tracker.finalize(order_id, actual_pu=actual_pu,
                         actual_size_gb=actual_size, state=state)
        estimator.reconcile(estimated_pu=reserved, actual_pu=actual_pu)

        log.info(
            "reconciled order %s: est=%.1f actual=%.1f ratio=%.2f cal=%.3f",
            order_id, reserved, actual_pu,
            actual_pu / reserved if reserved else float("nan"),
            estimator.calibration,
        )


# ---------------------------------------------------------------------------
# Convenience: full loop
# ---------------------------------------------------------------------------
def run_select_stage(
    candidate_groups: Iterable[dict],
    *,
    plan: PlanConfig,
    state_path: Path,
    aoi_geojson: dict,
    item_type: str = "PSScene",
    product_bundle: str = "analytic_8b_sr_udm2",
    harmonize: bool = True,
    deliver_to_collection: bool = False,
    collection_id: Optional[str] = None,
) -> dict:
    """
    Entry point a pipeline DAG (Airflow / Prefect / Argo) can call once
    per scheduling tick.

    `candidate_groups` is an iterable of dicts of the form:
        {
            "name": "aoi_A_2026-05-15",
            "item_ids": [...],         # pre-sorted, best first
            "scenes_demanded": 4,      # how many usable scenes you actually need
        }
    """
    pl = Planet()
    estimator = PUEstimator(plan=plan)
    tracker = PUBudgetTracker(plan=plan, state_path=state_path)

    # Reconcile any previously-submitted orders before deciding new spend.
    reconcile_completed_orders(pl, estimator, tracker)

    tools_used = ["clip"] + (["harmonize"] if harmonize else [])
    submitted = []
    rejected = []

    for group in candidate_groups:
        order_id = submit_budgeted_order(
            pl,
            candidate_item_ids=group["item_ids"],
            scenes_demanded=group["scenes_demanded"],
            item_type=item_type,
            product_bundle=product_bundle,
            aoi_geojson=aoi_geojson,
            tools_used=tools_used,
            estimator=estimator,
            tracker=tracker,
            order_name=group["name"],
            harmonize=harmonize,
            deliver_to_collection=deliver_to_collection,
            collection_id=collection_id,
        )
        if order_id:
            submitted.append({"name": group["name"], "order_id": order_id})
        else:
            rejected.append(group["name"])

    return {
        "submitted": submitted,
        "rejected": rejected,
        "budget_status": tracker.status(),
        "calibration": estimator.calibration,
    }
