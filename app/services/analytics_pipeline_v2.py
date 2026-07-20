"""
Weevo Analytics — API v2 data pipeline.

Supersedes the earlier scheduler_shipment_data-based pipeline. This
version talks to the real Weevo API directly, using the exact field
names confirmed from a live raw JSON response (not from any exported
file, which turned out to have a broken formula in one column) shared
2026-07-10:

  - Financial logic confirmed against real records:
      weevo_revenue   = agreed_shipping_cost + transfer_fee
      transfer_fee    = round(amount * 0.01, 2) when payment_method == "cod",
                         else 0.00 (confirmed: online orders always show
                         transfer_fee == "0.00")
      merchant_payout = amount - agreed_shipping_cost - transfer_fee
                         (only meaningful when amount > 0, i.e. cod;
                         for online orders amount is always "0.00"
                         because the customer paid the merchant directly)

  - Actual pickup time is NOT date_to_receive_shipment (that's the
    *target* time). The real pickup timestamp is inside the shipment's
    logs[] array, on the entry where log_flag == "picked_up_by_captain".
    A shipment can have more than one such entry if it was reassigned;
    the LAST one before delivery/return is used.

  - delivered_date_at / returned_date_at are direct top-level fields —
    no need to dig through logs for those.

  - The three handover codes map to three different legs of the trip:
      handover_code_merchant_to_courier -> Pickup code (merchant -> captain)
      handover_code_courier_to_customer -> Delivered code (captain -> client)
      handover_code_courier_to_merchant -> Returned code (captain -> merchant)

SHIPMENT RETRIEVAL (updated 2026-07-15): shipment fetching no longer
happens in this file. It now shares analytics_pipeline.fetch_admin_shipments()
with the main (v1) pipeline — one combined, date-filtered pull against the
official Admin Dashboard backend, walked to its actual last page (bounded
only by the same wall-clock time budget as before, no artificial page
cap) — so the main section and this Financial/Risk section are always
built from the exact same underlying records. See load_shipments_v2() /
build_v2_dataframe() below.

REFERENCE DATA (merchants/couriers) RETRIEVAL (updated 2026-07-15):
fetch_captains()/fetch_merchants() now also hit the same Admin Dashboard
backend (admin-5678vna9k6/couriers and admin-5678vna9k6/merchants —
confirmed field-for-field against real captured responses, see
Merchant.docx) using the exact same Bearer-token auth as shipments —
_get() below shares analytics_pipeline._get_admin_bearer_token(), so the
login-using-token call and its in-process token cache are the single
source of truth for every Admin Dashboard request this whole app makes,
shipments included. The older ai-agent endpoint and its X-Integration-Key
header are no longer used anywhere in this file.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import httpx

from app.services.analytics_pipeline import (
    detect_area,
    enrich_areas_with_cache,
    fetch_admin_shipments,
    _get_admin_bearer_token,
)

WEEVO_API_BASE_URL = os.getenv("WEEVO_API_BASE_URL", "https://eg.api.weevoapp.com")

PRIMARY_STATUSES = ["delivered", "returned"]

INTER_REQUEST_DELAY = 0.35

TERMINAL_STATUSES = {"delivered", "returned", "cancelled", "bulk-shipment-closed", "bulk-shipment-cancelled"}

GATEWAY_ERROR_CODES = {502, 503, 504}
REQUEST_TIMEOUT = 20
MAX_RETRIES = 1

def _get(url: str, api_key: str, params: dict, base_url: str = WEEVO_API_BASE_URL,
          timeout: int = REQUEST_TIMEOUT) -> httpx.Response:
    """Same Bearer-token auth as every other Admin Dashboard call (2026-07-15
    migration): uses the shared, in-process-cached access token from
    analytics_pipeline._get_admin_bearer_token() (obtained via
    login-using-token), sent as `Authorization: Bearer <token>`. On a 401
    the token is refreshed exactly once and the request retried — same
    discipline as analytics_pipeline._fetch_admin_shipments_page(). Same
    gateway-error/timeout retry pattern as before otherwise."""
    attempt = 0
    reauthed = False
    while True:
        token = _get_admin_bearer_token(api_key, base_url=base_url, force_refresh=False)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        try:
            response = httpx.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code == 401 and not reauthed:
                reauthed = True
                _get_admin_bearer_token(api_key, base_url=base_url, force_refresh=True)
                continue
            if response.status_code in GATEWAY_ERROR_CODES and attempt < MAX_RETRIES:
                attempt += 1
                time.sleep(1.5)
                continue
            response.raise_for_status()
            return response
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES:
                attempt += 1
                time.sleep(1.5)
                continue
            raise

def fetch_reference_list(endpoint: str, api_key: str, base_url: str = WEEVO_API_BASE_URL,
                          limit: int = 100, max_pages: int = 5) -> list:
    """Generic paginated fetch for /merchants and /couriers on the Admin
    Dashboard backend (2026-07-15 migration — confirmed field-for-field
    against real captured responses for both endpoints, see Merchant.docx):
    same `paginate`/`page` query params and top-level `current_page`/`data`/
    `last_page` envelope as every other admin-5678vna9k6 endpoint, same
    Bearer-token auth as shipments. Much smaller datasets than shipments,
    but fetched with the same conservative pacing (sequential, short pause
    between pages) as before."""
    url = f"{base_url}/api/v1/admin-5678vna9k6/{endpoint}"
    all_records = []
    page = 1
    while page <= max_pages:
        response = _get(url, api_key, params={"paginate": limit, "page": page}, base_url=base_url)
        payload = response.json() or {}
        records = payload.get("data") or []
        all_records.extend(records)
        last_page = payload.get("last_page") or 1
        if page >= last_page:
            break
        page += 1
        time.sleep(INTER_REQUEST_DELAY)
    return all_records

def fetch_captains(api_key: str, base_url: str = WEEVO_API_BASE_URL) -> pd.DataFrame:
    """Couriers roster (2026-07-15 migration: real endpoint is `/couriers`
    on the Admin Dashboard backend — kept the name fetch_captains() since
    the rest of the dashboard/UI still calls couriers "captains")."""
    records = fetch_reference_list("couriers", api_key, base_url)
    return pd.DataFrame(records)

def fetch_merchants(api_key: str, base_url: str = WEEVO_API_BASE_URL) -> pd.DataFrame:
    records = fetch_reference_list("merchants", api_key, base_url)
    return pd.DataFrame(records)

def _parse_dt(value: Optional[str]):
    """Parses any of the date formats seen in real shipment records.

    Real data mixes formats across records — most fields come back as plain
    'YYYY-MM-DD HH:MM:SS' (timezone-naive), but some records (observed in
    live data on 577 real orders) come back with a timezone offset or a
    'Z' UTC suffix instead. Left as-is, that produces a mix of naive and
    timezone-aware Timestamps across the same column, and pandas refuses to
    subtract one from the other ('Cannot subtract tz-naive and tz-aware
    datetime-like objects') — which is exactly the error seen in the
    Financial Summary section. Every value gets its timezone info stripped
    (not converted — the wall-clock value is kept as-is) right here, so
    everything downstream is guaranteed consistently naive."""
    if not value:
        return None
    try:
        ts = pd.to_datetime(value)
        if ts is not None and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except (ValueError, TypeError):
        return None

def _extract_actual_pickup_at(logs: list):
    """Last 'picked_up_by_captain' log entry (a shipment can have more
    than one if it was reassigned mid-flight; the last one before
    delivery/return is the one that matters)."""
    pickup_logs = [l for l in (logs or []) if l.get("log_flag") == "picked_up_by_captain"]
    if not pickup_logs:
        return None
    latest = max(pickup_logs, key=lambda l: l.get("created_at") or "")
    return _parse_dt(latest.get("created_at"))

def parse_shipment(raw: dict) -> dict:
    status = raw.get("status", "unknown")
    payment_method = (raw.get("payment_method") or "").lower()
    amount = float(raw.get("amount") or 0)
    agreed_shipping_cost = float(raw.get("agreed_shipping_cost") or 0)
    transfer_fee = float(raw.get("transfer_fee") or 0)

    merchant = raw.get("merchant") or {}
    courier = raw.get("courier") or {}

    pickup_actual_at = _extract_actual_pickup_at(raw.get("logs"))
    delivered_at = _parse_dt(raw.get("delivered_date_at"))
    returned_at = _parse_dt(raw.get("returned_date_at"))
    completed_at = delivered_at or returned_at

    delivery_hours = None
    if pickup_actual_at is not None and completed_at is not None:
        delta_hours = (completed_at - pickup_actual_at).total_seconds() / 3600
        if delta_hours >= 0:
            delivery_hours = round(delta_hours, 2)

    target_deliver_at = _parse_dt(raw.get("date_to_deliver_shipment"))
    now = pd.Timestamp.now()
    is_overdue = (
        status not in TERMINAL_STATUSES
        and target_deliver_at is not None
        and target_deliver_at < now
    )

    weevo_revenue = round(agreed_shipping_cost + transfer_fee, 2)
    merchant_payout = round(amount - agreed_shipping_cost - transfer_fee, 2) if amount > 0 else None

    return {
        "shipment_id": raw.get("id"),
        "reference": raw.get("reference"),
        "status": status,
        "merchant_id": merchant.get("id"),
        "merchant_name": merchant.get("name") or "Unknown",
        "courier_id": courier.get("id"),
        "courier_name": courier.get("name") or "Unassigned",
        "client_name": (raw.get("client_name") or "").strip() or None,
        "client_phone": raw.get("client_phone"),
        "payment_method": payment_method or "unknown",
        "amount": amount,
        "agreed_shipping_cost": agreed_shipping_cost,
        "transfer_fee": transfer_fee,
        "weevo_revenue": weevo_revenue,
        "merchant_payout": merchant_payout,
        "delivering_street": raw.get("delivering_street"),
        "area": detect_area(raw.get("delivering_street")),
        "created_at": _parse_dt(raw.get("created_at")),
        "target_pickup_at": _parse_dt(raw.get("date_to_receive_shipment")),
        "pickup_actual_at": pickup_actual_at,
        "target_deliver_at": target_deliver_at,
        "delivered_at": delivered_at,
        "returned_at": returned_at,
        "delivery_hours": delivery_hours,
        "is_overdue": is_overdue,
        "pickup_code": raw.get("handover_code_merchant_to_courier"),
        "delivered_code": raw.get("handover_code_courier_to_customer"),
        "returned_code": raw.get("handover_code_courier_to_merchant"),
    }

def _v2_fetch_diagnostics_from_meta(fetch_meta: Optional[dict]) -> dict:
    """Adapts analytics_pipeline.fetch_admin_shipments()'s generic fetch_meta
    into the exact dict shape this page's Financial/Risk section already
    reads for its own partial-fetch-failure banner: total_statuses (int) /
    succeeded_statuses (list) / failed_statuses (DICT of {label: reason},
    unlike v1's list — this section iterates `.items()`) — unchanged UI
    code, single pseudo-entry instead of one per status now that the fetch
    is combined."""
    if not fetch_meta:
        return {"total_statuses": 0, "succeeded_statuses": [], "failed_statuses": {},
                "fetched_at": datetime.now().isoformat()}
    label = fetch_meta["label"]
    if fetch_meta.get("error") or fetch_meta.get("truncated_by_budget"):
        if fetch_meta.get("error"):
            reason = fetch_meta["error"]
        else:
            reason = (
                f"Stopped after the fetch time budget "
                f"(page {fetch_meta['pages_fetched']} of {fetch_meta['last_page']}) "
                f"— remaining pages skipped."
            )
        return {
            "total_statuses": 1,
            "succeeded_statuses": [label] if fetch_meta["pages_fetched"] > 0 else [],
            "failed_statuses": {label: reason},
            "fetched_at": fetch_meta["fetched_at"],
        }
    return {
        "total_statuses": 1,
        "succeeded_statuses": [label],
        "failed_statuses": {},
        "fetched_at": fetch_meta["fetched_at"],
    }

def build_v2_dataframe(raw_records: list, fetch_meta: Optional[dict] = None) -> pd.DataFrame:
    """Pure mapping: raw admin-API shipment dicts -> the parse_shipment()
    shape this pipeline has always produced. No HTTP in here — split out
    from the old load_shipments_v2() specifically so
    streamlit_ui/analytics_page.py can fetch ONCE (via
    analytics_pipeline.fetch_admin_shipments) and hand the exact same raw
    records to both this function and analytics_pipeline.build_shipments_dataframe,
    guaranteeing the main section and the Financial/Risk section are always
    built from identical underlying shipments."""
    parsed = [parse_shipment(r) for r in raw_records]
    df = pd.DataFrame(parsed)
    for _col in _V2_DATETIME_COLS:
        if _col in df.columns:
            df[_col] = pd.to_datetime(df[_col], errors="coerce")
    if not df.empty:
        df = enrich_areas_with_cache(df)
    df.attrs["fetch_diagnostics"] = _v2_fetch_diagnostics_from_meta(fetch_meta)
    return df

def load_shipments_v2(api_key: str, base_url: str = WEEVO_API_BASE_URL,
                       start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Standalone convenience wrapper: fetch + parse everything into one
    flat DataFrame, same contract this function always had. The Streamlit
    page itself doesn't call this directly for its own load (it shares one
    fetch_admin_shipments() pull with the v1/main section — see
    _cached_admin_shipments in analytics_page.py), but this keeps
    load_shipments_v2() working standalone for any other caller.

    start_date/end_date ("YYYY-MM-DD" or None) are new (2026-07-15 admin-API
    migration) — sent straight to the server, same as v1's load_shipments().
    None/None ("no range given") resolves to the last 30 days here — never
    an unbounded all-history fetch (2026-07-17 default-date fix).
    """
    if start_date is None and end_date is None:
        _end = pd.Timestamp.now().normalize()
        _start = _end - pd.Timedelta(days=30)
        start_date, end_date = _start.strftime("%Y-%m-%d"), _end.strftime("%Y-%m-%d")
    raw_records, _status_counts, fetch_meta = fetch_admin_shipments(
        api_key=api_key, start_date=start_date, end_date=end_date, base_url=base_url,
    )
    return build_v2_dataframe(raw_records, fetch_meta=fetch_meta)

def revenue_summary(df: pd.DataFrame) -> dict:
    """Weevo's actual take, not gross order value — delivered + returned
    only (in-flight orders haven't generated final revenue yet)."""
    completed = df[df["status"].isin(PRIMARY_STATUSES)]
    if completed.empty:
        return {"total_weevo_revenue": 0.0, "delivered_count": 0, "returned_count": 0,
                "delivered_revenue": 0.0, "returned_revenue": 0.0,
                "return_rate_pct": 0.0, "delivered_rate_pct": 0.0}
    delivered = completed[completed["status"] == "delivered"]
    returned = completed[completed["status"] == "returned"]
    return {
        "total_weevo_revenue": round(completed["weevo_revenue"].sum(), 2),
        "delivered_count": int(len(delivered)),
        "returned_count": int(len(returned)),
        "delivered_revenue": round(delivered["weevo_revenue"].sum(), 2),
        "returned_revenue": round(returned["weevo_revenue"].sum(), 2),
        "return_rate_pct": round(len(returned) / len(completed) * 100, 1) if len(completed) else 0.0,
        "delivered_rate_pct": round(len(delivered) / len(completed) * 100, 1) if len(completed) else 0.0,
    }

def delivery_time_summary(df: pd.DataFrame) -> dict:
    """Actual pickup-to-completion time, delivered + returned combined —
    this replaces the old 'two hours from order creation' framing per
    Ahmed's correction: it's pickup-to-completion, not creation-to-completion."""
    valid = df[df["delivery_hours"].notna()]
    if valid.empty:
        return {"avg_hours": None, "median_hours": None, "sample_size": 0}
    return {
        "avg_hours": round(valid["delivery_hours"].mean(), 2),
        "median_hours": round(valid["delivery_hours"].median(), 2),
        "sample_size": int(len(valid)),
    }

def _lifecycle_hours(df: pd.DataFrame) -> pd.Series:
    """created_at -> delivered_at/returned_at, in hours. Only rows with a
    valid non-negative duration are kept (guards against bad/out-of-order
    data, same convention as delivery_hours above)."""
    completed = df[df["status"].isin(PRIMARY_STATUSES)]
    completed_at = completed["delivered_at"].fillna(completed["returned_at"])
    hours = (completed_at - completed["created_at"]).dt.total_seconds() / 3600
    return hours[hours.notna() & (hours >= 0)]

def creation_to_completion_summary(df: pd.DataFrame) -> dict:
    """Avg/median time from order creation to actual delivered/returned
    completion — the real-timestamp-only replacement for delivery_time_summary."""
    valid = _lifecycle_hours(df)
    if valid.empty:
        return {"avg_hours": None, "median_hours": None, "sample_size": 0}
    return {
        "avg_hours": round(valid.mean(), 2),
        "median_hours": round(valid.median(), 2),
        "sample_size": int(len(valid)),
    }

def creation_to_completion_summary_by_status(df: pd.DataFrame) -> dict:
    """Same measure as creation_to_completion_summary(), but delivered and
    returned kept separate instead of blended (ADDED 2026-07-19) — a
    returned shipment went out AND came back, so it's a naturally different
    (usually longer) duration than a straightforward delivery, and blending
    them into one average hid that difference."""
    def _for(status: str, completion_col: str) -> dict:
        seg = df[df["status"] == status]
        hours = (seg[completion_col] - seg["created_at"]).dt.total_seconds() / 3600
        valid = hours[hours.notna() & (hours >= 0)]
        if valid.empty:
            return {"avg_hours": None, "median_hours": None, "sample_size": 0}
        return {
            "avg_hours": round(valid.mean(), 2),
            "median_hours": round(valid.median(), 2),
            "sample_size": int(len(valid)),
        }
    return {
        "delivered": _for("delivered", "delivered_at"),
        "returned": _for("returned", "returned_at"),
    }

def creation_to_completion_by_area(df: pd.DataFrame, n: int = 5, ascending: bool = False) -> pd.DataFrame:
    """Avg creation-to-completion time per delivery area (the `area` column,
    detected from delivering_street) — fastest areas first if ascending=True,
    slowest first if ascending=False. 'Unknown' (address didn't match a
    known area) is excluded, same as the existing Top areas chart."""
    completed = df[df["status"].isin(PRIMARY_STATUSES)].copy()
    completed_at = completed["delivered_at"].fillna(completed["returned_at"])
    completed["lifecycle_hours"] = (completed_at - completed["created_at"]).dt.total_seconds() / 3600
    valid = completed[
        completed["lifecycle_hours"].notna() & (completed["lifecycle_hours"] >= 0)
        & (completed["area"] != "Unknown")
    ]
    if valid.empty:
        return pd.DataFrame(columns=["area", "avg_hours", "orders"])
    grouped = valid.groupby("area")["lifecycle_hours"].agg(avg_hours="mean", orders="count").reset_index()
    grouped["avg_hours"] = grouped["avg_hours"].round(2)
    return grouped.sort_values("avg_hours", ascending=ascending).head(n).reset_index(drop=True)

def creation_to_completion_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Avg creation-to-completion time grouped by the hour of day (0-23)
    the order was CREATED (created_at) — shows whether orders placed at
    certain hours consistently take longer to complete, all 24 hours
    present in the data, sorted by hour for a clean time-of-day chart."""
    valid_hours = _lifecycle_hours(df)
    completed = df.loc[valid_hours.index].copy()
    completed["lifecycle_hours"] = valid_hours
    completed["hour_of_day"] = completed["created_at"].dt.hour
    if completed.empty:
        return pd.DataFrame(columns=["hour_of_day", "avg_hours", "orders"])
    grouped = completed.groupby("hour_of_day")["lifecycle_hours"].agg(avg_hours="mean", orders="count").reset_index()
    grouped["avg_hours"] = grouped["avg_hours"].round(2)
    return grouped.sort_values("hour_of_day").reset_index(drop=True)

def at_risk_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """Shipments still in flight whose target delivery time has already
    passed — the 'captain still has orders overdue' alert Ahmed described.
    Sorted by how overdue (most overdue first)."""
    risky = df[df["is_overdue"] == True].copy()
    if risky.empty:
        return pd.DataFrame(columns=["shipment_id", "courier_name", "merchant_name", "status",
                                      "target_deliver_at", "hours_overdue"])
    now = pd.Timestamp.now()
    risky["hours_overdue"] = ((now - risky["target_deliver_at"]).dt.total_seconds() / 3600).round(1)
    return (
        risky[["shipment_id", "courier_name", "merchant_name", "status", "target_deliver_at", "hours_overdue"]]
        .sort_values("hours_overdue", ascending=False)
    )

def risk_by_courier(df: pd.DataFrame) -> pd.DataFrame:
    """How many overdue shipments each courier currently has — this is
    the 'which captain took an order and hasn't acted' view."""
    risky = at_risk_shipments(df)
    if risky.empty:
        return pd.DataFrame(columns=["courier_name", "overdue_orders", "max_hours_overdue"])
    return (
        risky.groupby("courier_name")
        .agg(overdue_orders=("shipment_id", "count"), max_hours_overdue=("hours_overdue", "max"))
        .reset_index()
        .sort_values("overdue_orders", ascending=False)
    )

SHIFT_START_HOUR = 17
SHIFT_END_HOUR = 23

RISK_LEVEL_ORDER = {"🔴 Overdue": 0, "🔴 At risk": 1, "🟡 Watch": 2, "🟢 Healthy": 3}

def get_shift_progress(now: "pd.Timestamp | None" = None) -> dict:
    """Where 'right now' sits inside today's fixed 5 PM–11 PM shift.
    `active` is False outside that window — the page uses this to decide
    whether to show the courier-risk section at all, rather than showing
    a misleading 0%/100% reading for a shift that isn't running."""
    if now is None:
        now = pd.Timestamp.now()
    shift_start = now.normalize() + pd.Timedelta(hours=SHIFT_START_HOUR)
    shift_end = now.normalize() + pd.Timedelta(hours=SHIFT_END_HOUR)
    shift_length_seconds = (shift_end - shift_start).total_seconds()
    elapsed_seconds = max(0.0, min((now - shift_start).total_seconds(), shift_length_seconds))
    elapsed_pct = round(elapsed_seconds / shift_length_seconds * 100, 1) if shift_length_seconds > 0 else 0.0
    return {
        "active": shift_start <= now <= shift_end,
        "now": now,
        "shift_start": shift_start,
        "shift_end": shift_end,
        "elapsed_pct": elapsed_pct,
        "elapsed_hours": round(elapsed_seconds / 3600, 2),
        "shift_length_hours": round(shift_length_seconds / 3600, 2),
    }

def courier_shift_risk(df: pd.DataFrame, now: "pd.Timestamp | None" = None) -> pd.DataFrame:
    """One row per courier who has any activity today (something closed
    out today, or something still open right now) — couriers with
    neither are left out entirely, there's nothing to show for them.

    completed_today = delivered_at.date() == today OR returned_at.date()
    == today, for that courier (both count as "closed out", whichever
    way the shipment resolved).
    remaining = currently assigned to that courier, status not in
    TERMINAL_STATUSES, AND created_at is today — same today-scoping as
    completed_today (2026-07-19 fix: previously counted ANY open shipment
    regardless of when it was created, so a courier with old stuck
    backlog looked artificially at-risk for today's shift specifically;
    old open shipments are now surfaced separately by
    stale_open_shipments() below instead).
    actual_progress_pct = completed_today / (completed_today + remaining)
    expected_progress_pct = % of the shift elapsed (same number for
    every courier, from get_shift_progress()).
    gap_pct = expected - actual. Positive means behind pace.
    risk_level: 🔴 Overdue if the shift has ended and the courier still
    has remaining orders; otherwise banded by gap_pct."""
    progress = get_shift_progress(now)
    empty = pd.DataFrame(columns=[
        "courier_name", "delivered_today", "returned_today", "completed_today",
        "remaining", "total_handled_today", "actual_progress_pct",
        "expected_progress_pct", "gap_pct", "risk_level",
    ])
    if not progress["active"] and progress["now"] < progress["shift_start"]:
        return empty

    assigned = df[df["courier_id"].notna()].copy()
    if assigned.empty:
        return empty

    today = progress["now"].normalize()
    remaining_mask = (
        ~assigned["status"].isin(TERMINAL_STATUSES)
        & assigned["created_at"].notna()
        & (assigned["created_at"].dt.normalize() == today)
    )
    delivered_today_mask = (
        (assigned["status"] == "delivered")
        & assigned["delivered_at"].notna()
        & (assigned["delivered_at"].dt.normalize() == today)
    )
    returned_today_mask = (
        (assigned["status"] == "returned")
        & assigned["returned_at"].notna()
        & (assigned["returned_at"].dt.normalize() == today)
    )

    assigned["_remaining"] = remaining_mask
    assigned["_delivered_today"] = delivered_today_mask
    assigned["_returned_today"] = returned_today_mask

    grouped = (
        assigned.groupby("courier_name")
        .agg(
            delivered_today=("_delivered_today", "sum"),
            returned_today=("_returned_today", "sum"),
            remaining=("_remaining", "sum"),
        )
        .reset_index()
    )
    grouped["completed_today"] = grouped["delivered_today"] + grouped["returned_today"]
    grouped = grouped[(grouped["completed_today"] > 0) | (grouped["remaining"] > 0)].copy()
    if grouped.empty:
        return empty

    grouped["total_handled_today"] = grouped["completed_today"] + grouped["remaining"]
    grouped["actual_progress_pct"] = (
        grouped["completed_today"] / grouped["total_handled_today"] * 100
    ).round(1)
    grouped["expected_progress_pct"] = progress["elapsed_pct"]
    grouped["gap_pct"] = (grouped["expected_progress_pct"] - grouped["actual_progress_pct"]).round(1)

    shift_over = progress["now"] >= progress["shift_end"]

    def _risk(row) -> str:
        if shift_over and row["remaining"] > 0:
            return "🔴 Overdue"
        if row["gap_pct"] <= 10:
            return "🟢 Healthy"
        if row["gap_pct"] <= 25:
            return "🟡 Watch"
        return "🔴 At risk"

    grouped["risk_level"] = grouped.apply(_risk, axis=1)
    grouped["_sort"] = grouped["risk_level"].map(RISK_LEVEL_ORDER)
    grouped = (
        grouped.sort_values(["_sort", "remaining"], ascending=[True, False])
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    return grouped[[
        "courier_name", "delivered_today", "returned_today", "completed_today",
        "remaining", "total_handled_today", "actual_progress_pct",
        "expected_progress_pct", "gap_pct", "risk_level",
    ]]

def courier_shift_risk_summary(risk_df: pd.DataFrame) -> dict:
    """Counts feeding the small KPI row above the courier-risk table."""
    if risk_df.empty:
        return {"healthy": 0, "watch": 0, "at_risk": 0, "overdue": 0, "total_remaining": 0}
    counts = risk_df["risk_level"].value_counts()
    return {
        "healthy": int(counts.get("🟢 Healthy", 0)),
        "watch": int(counts.get("🟡 Watch", 0)),
        "at_risk": int(counts.get("🔴 At risk", 0)),
        "overdue": int(counts.get("🔴 Overdue", 0)),
        "total_remaining": int(risk_df["remaining"].sum()),
    }

STALE_OPEN_SHIPMENT_DAYS = 2

def stale_open_shipments(df: pd.DataFrame, days: int = STALE_OPEN_SHIPMENT_DAYS,
                          now: "pd.Timestamp | None" = None) -> pd.DataFrame:
    """Shipments created `days`+ days ago that are still open (status not
    in TERMINAL_STATUSES) — deliberately separate from Courier shift risk
    (2026-07-19): a shipment sitting open for days isn't necessarily that
    day's shift falling behind, and lumping it in there both skews a
    courier's today-only risk score with a date it has nothing to do with,
    and buries the real issue (an old unresolved shipment) inside a metric
    that isn't about it.

    Important limit, by design: there's no "last status update" timestamp
    in this data, only created_at and the current status. So this can only
    say "created X days ago and still not delivered/returned/cancelled" —
    NOT "hasn't been touched in X days" (the status may well have changed
    recently, e.g. in-transit -> in-transit-for-return two hours ago; we
    just have no record of when). Sorted oldest-created first."""
    if now is None:
        now = pd.Timestamp.now()
    empty = pd.DataFrame(columns=["shipment_id", "courier_name", "merchant_name", "status",
                                   "created_at", "days_open"])
    if df.empty or "created_at" not in df.columns:
        return empty
    open_mask = ~df["status"].isin(TERMINAL_STATUSES)
    cutoff = now - pd.Timedelta(days=days)
    stale = df[open_mask & df["created_at"].notna() & (df["created_at"] <= cutoff)].copy()
    if stale.empty:
        return empty
    stale["days_open"] = ((now - stale["created_at"]).dt.total_seconds() / 86400).round(1)
    return (
        stale[["shipment_id", "courier_name", "merchant_name", "status", "created_at", "days_open"]]
        .sort_values("days_open", ascending=False)
        .reset_index(drop=True)
    )

DAILY_BREAKDOWN_MAX_DAYS = 14

def courier_performance_by_period(df: pd.DataFrame, start_date, end_date,
                                   now: "pd.Timestamp | None" = None) -> dict:
    """Per-courier workload for whatever date range is selected in the
    sidebar (2026-07-20 replacement for the old 'always checks today,
    5 PM-11 PM' shift view) — same trustworthy logic as courier_shift_risk()
    (actual closed-out work vs. elapsed available time), just generalized
    from a single fixed shift to any start_date/end_date span. Deliberately
    built only from created_at/delivered_at/returned_at — never from
    target_pickup_at/target_deliver_at, since those are editable and not a
    trustworthy benchmark (same reasoning as courier_shift_risk()).

    For each calendar day D in [start_date, end_date]:
      assigned         = orders with created_at.date() == D, per courier
      closed_same_day  = of those, delivered_at.date() == D OR
                          returned_at.date() == D
      carried_over     = assigned - closed_same_day (resolved on a later
                          day, or still open) — only meaningful for days
                          that are already over; today is handled
                          separately below since it isn't over yet.
      carried_over is further split (2026-07-20, ADDITIVE) into 3
      mutually-exclusive buckets that always sum to carried_over:
        closed_later = delivered_at/returned_at set, just not same-day
        still_open   = status not terminal, no delivered_at/returned_at
        cancelled    = status == "cancelled" (source column: status)

    Returns a dict:
      "daily"       — [date, courier_name, assigned, closed_same_day,
                       closed_later, still_open, cancelled, carried_over],
                       only populated when the span is short enough for a
                       table (see show_daily_table)
      "show_daily_table" — bool, span_days <= DAILY_BREAKDOWN_MAX_DAYS
      "trend"       — [date, assigned, closed_same_day], for a chart when
                       the span is too long for a daily table
      "by_courier"  — [courier_name, assigned, closed_same_day,
                       closed_later, still_open, cancelled,
                       avg_days_to_close, carried_over, closure_rate_pct],
                       always populated, one row per courier for the
                       whole period (avg_days_to_close is NaN when
                       closed_later == 0 for that courier)
      "by_area"     — [area, assigned, closed_same_day, closure_rate_pct]
      "today"       — dict with today's elapsed-shift pacing/risk (same
                       shape as courier_shift_risk), only present if today
                       falls inside [start_date, end_date]; None otherwise
      "span_days"   — int
    """
    if now is None:
        now = pd.Timestamp.now()
    today = now.normalize().date()

    result = {
        "daily": pd.DataFrame(columns=[
            "date", "courier_name", "assigned", "closed_same_day",
            "closed_later", "still_open", "cancelled", "carried_over",
        ]),
        "show_daily_table": False,
        "trend": pd.DataFrame(columns=["date", "assigned", "closed_same_day"]),
        "by_courier": pd.DataFrame(columns=[
            "courier_name", "assigned", "closed_same_day", "closed_later",
            "still_open", "cancelled", "avg_days_to_close", "carried_over", "closure_rate_pct",
        ]),
        "by_area": pd.DataFrame(columns=["area", "assigned", "closed_same_day", "closure_rate_pct"]),
        "today": None,
        "span_days": 0,
    }
    if df is None or df.empty or start_date is None or end_date is None or "created_at" not in df.columns:
        return result

    assigned = df[df["courier_id"].notna() & df["created_at"].notna()].copy()
    if assigned.empty:
        return result

    assigned["created_date"] = assigned["created_at"].dt.date
    in_range = assigned[
        (assigned["created_date"] >= start_date) & (assigned["created_date"] <= end_date)
    ].copy()
    if in_range.empty:
        return result

    in_range["closed_same_day"] = (
        (in_range["delivered_at"].notna() & (in_range["delivered_at"].dt.date == in_range["created_date"]))
        | (in_range["returned_at"].notna() & (in_range["returned_at"].dt.date == in_range["created_date"]))
    )

    # --- carried_over breakdown (2026-07-20, ADDITIVE) ---------------------
    # carried_over used to be one lump number (assigned - closed_same_day).
    # Split here into the 3 mutually-exclusive outcomes a non-same-day order
    # can actually be in, using only existing/static columns — no new
    # fetch, no new library:
    #   closed_later = has delivered_at or returned_at, just not same day
    #                  as created_at (both are real event timestamps, not
    #                  editable fields)
    #   cancelled    = status == "cancelled" (source column: `status`,
    #                  same raw field used everywhere else in this file,
    #                  e.g. the TERMINAL_STATUSES checks above)
    #   still_open   = neither of the above — genuinely unresolved as of
    #                  `now`, not just "not yet same-day-closed"
    # Priority is cancelled > closed_later > still_open so the three are
    # mutually exclusive and always sum to exactly carried_over.
    _closed_at_all = in_range["delivered_at"].notna() | in_range["returned_at"].notna()
    _cancelled = in_range["status"] == "cancelled"  # source column: status
    _carried = ~in_range["closed_same_day"]
    in_range["_carried_cancelled"] = _carried & _cancelled
    in_range["_carried_closed_later"] = _carried & ~_cancelled & _closed_at_all
    in_range["_carried_still_open"] = _carried & ~_cancelled & ~_closed_at_all
    # days-to-close, only meaningful for the closed_later bucket (uses
    # whichever of delivered_at/returned_at is populated)
    _close_ts = in_range["delivered_at"].where(in_range["delivered_at"].notna(), in_range["returned_at"])
    in_range["_days_to_close"] = (
        (_close_ts - in_range["created_at"]).dt.total_seconds() / 86400
    ).where(in_range["_carried_closed_later"])
    # -------------------------------------------------------------------

    result["span_days"] = (end_date - start_date).days + 1

    daily = (
        in_range.groupby(["created_date", "courier_name"])
        .agg(
            assigned=("shipment_id", "count"),
            closed_same_day=("closed_same_day", "sum"),
            closed_later=("_carried_closed_later", "sum"),
            still_open=("_carried_still_open", "sum"),
            cancelled=("_carried_cancelled", "sum"),
        )
        .reset_index()
        .rename(columns={"created_date": "date"})
    )
    daily["carried_over"] = daily["assigned"] - daily["closed_same_day"]
    result["daily"] = daily.sort_values(["date", "carried_over"], ascending=[True, False]).reset_index(drop=True)
    result["show_daily_table"] = result["span_days"] <= DAILY_BREAKDOWN_MAX_DAYS

    result["trend"] = (
        in_range.groupby("created_date")
        .agg(assigned=("shipment_id", "count"), closed_same_day=("closed_same_day", "sum"))
        .reset_index()
        .rename(columns={"created_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    by_courier = (
        in_range.groupby("courier_name")
        .agg(
            assigned=("shipment_id", "count"),
            closed_same_day=("closed_same_day", "sum"),
            closed_later=("_carried_closed_later", "sum"),
            still_open=("_carried_still_open", "sum"),
            cancelled=("_carried_cancelled", "sum"),
            avg_days_to_close=("_days_to_close", "mean"),
        )
        .reset_index()
    )
    by_courier["carried_over"] = by_courier["assigned"] - by_courier["closed_same_day"]
    by_courier["closure_rate_pct"] = (by_courier["closed_same_day"] / by_courier["assigned"] * 100).round(1)
    by_courier["avg_days_to_close"] = by_courier["avg_days_to_close"].round(1)  # NaN where closed_later == 0
    result["by_courier"] = by_courier.sort_values("carried_over", ascending=False).reset_index(drop=True)

    if "area" in in_range.columns:
        area_src = in_range[in_range["area"] != "Unknown"]
        if not area_src.empty:
            by_area = (
                area_src.groupby("area")
                .agg(assigned=("shipment_id", "count"), closed_same_day=("closed_same_day", "sum"))
                .reset_index()
            )
            by_area["closure_rate_pct"] = (by_area["closed_same_day"] / by_area["assigned"] * 100).round(1)
            result["by_area"] = by_area.sort_values("closure_rate_pct", ascending=False).reset_index(drop=True)

    if start_date <= today <= end_date:
        progress = get_shift_progress(now)
        today_result = {
            "active": progress["active"], "now": progress["now"],
            "shift_start": progress["shift_start"], "shift_end": progress["shift_end"],
            "elapsed_pct": progress["elapsed_pct"], "elapsed_hours": progress["elapsed_hours"],
            "shift_length_hours": progress["shift_length_hours"],
            "risk": pd.DataFrame(columns=[
                "courier_name", "delivered_today", "returned_today", "completed_today",
                "remaining", "actual_progress_pct", "expected_progress_pct", "gap_pct", "risk_level",
            ]),
        }
        today_slice = in_range[in_range["created_date"] == today]
        if progress["active"] and not today_slice.empty:
            t = today_slice.copy()
            t["_delivered"] = t["delivered_at"].notna() & (t["delivered_at"].dt.date == today)
            t["_returned"] = t["returned_at"].notna() & (t["returned_at"].dt.date == today)
            per_courier = (
                t.groupby("courier_name")
                .agg(delivered_today=("_delivered", "sum"), returned_today=("_returned", "sum"),
                     assigned=("shipment_id", "count"))
                .reset_index()
            )
            per_courier["completed_today"] = per_courier["delivered_today"] + per_courier["returned_today"]
            per_courier["remaining"] = per_courier["assigned"] - per_courier["completed_today"]
            per_courier = per_courier[(per_courier["completed_today"] > 0) | (per_courier["remaining"] > 0)].copy()
            if not per_courier.empty:
                per_courier["actual_progress_pct"] = (
                    per_courier["completed_today"] / per_courier["assigned"] * 100
                ).round(1)
                per_courier["expected_progress_pct"] = progress["elapsed_pct"]
                per_courier["gap_pct"] = (per_courier["expected_progress_pct"] - per_courier["actual_progress_pct"]).round(1)
                shift_over = progress["now"] >= progress["shift_end"]

                def _risk(row):
                    if shift_over and row["remaining"] > 0:
                        return "🔴 Overdue"
                    if row["gap_pct"] <= 10:
                        return "🟢 Healthy"
                    if row["gap_pct"] <= 25:
                        return "🟡 Watch"
                    return "🔴 At risk"

                per_courier["risk_level"] = per_courier.apply(_risk, axis=1)
                per_courier["_sort"] = per_courier["risk_level"].map(RISK_LEVEL_ORDER)
                per_courier = (
                    per_courier.sort_values(["_sort", "remaining"], ascending=[True, False])
                    .drop(columns="_sort").reset_index(drop=True)
                )
                today_result["risk"] = per_courier[[
                    "courier_name", "delivered_today", "returned_today", "completed_today",
                    "remaining", "actual_progress_pct", "expected_progress_pct", "gap_pct", "risk_level",
                ]]
        result["today"] = today_result

    return result

def courier_performance_summary(perf: dict) -> dict:
    """Small KPI-row counts for the period view — historical carried-over
    couriers (span already over for that day) plus today's live risk bands
    (if today is inside the selected period), combined into one summary."""
    by_courier = perf.get("by_courier", pd.DataFrame())
    total_carried_over = int(by_courier["carried_over"].sum()) if not by_courier.empty else 0
    couriers_with_carryover = int((by_courier["carried_over"] > 0).sum()) if not by_courier.empty else 0
    # breakdown totals (2026-07-20, ADDITIVE) — same 3 mutually-exclusive
    # buckets as courier_performance_by_period(), summed across couriers
    total_closed_later = int(by_courier["closed_later"].sum()) if not by_courier.empty else 0
    total_still_open = int(by_courier["still_open"].sum()) if not by_courier.empty else 0
    total_cancelled = int(by_courier["cancelled"].sum()) if not by_courier.empty else 0

    today = perf.get("today")
    healthy = watch = at_risk = overdue = 0
    if today is not None and not today["risk"].empty:
        counts = today["risk"]["risk_level"].value_counts()
        healthy = int(counts.get("🟢 Healthy", 0))
        watch = int(counts.get("🟡 Watch", 0))
        at_risk = int(counts.get("🔴 At risk", 0))
        overdue = int(counts.get("🔴 Overdue", 0))

    return {
        "total_carried_over": total_carried_over,
        "couriers_with_carryover": couriers_with_carryover,
        "total_closed_later": total_closed_later,
        "total_still_open": total_still_open,
        "total_cancelled": total_cancelled,
        "healthy": healthy, "watch": watch, "at_risk": at_risk, "overdue": overdue,
    }

def financial_breakdown(df: pd.DataFrame) -> dict:
    """
    Deeper split of revenue_summary() — same underlying weevo_revenue
    formula, broken into its components (shipping cost vs the 1% COD
    transfer fee) and compared delivered vs returned side by side, plus
    the specific question asked: for a RETURNED order, is what Weevo
    earned in shipping large relative to what that order was actually
    worth? A high ratio here means returns are structurally expensive
    relative to their own value, not just 'a shipment that didn't work out'.
    """
    completed = df[df["status"].isin(PRIMARY_STATUSES)]
    empty_segment = {
        "count": 0, "shipping_revenue": 0.0, "transfer_fee_revenue": 0.0,
        "total_weevo_revenue": 0.0, "avg_weevo_revenue_per_order": 0.0,
        "total_order_value": 0.0,
        "cod_count": 0, "online_count": 0,
        "avg_client_order_value_cod": 0.0, "avg_client_order_value_online": 0.0,
        "total_merchant_payout_cod": 0.0,
    }
    if completed.empty:
        return {"delivered": empty_segment, "returned": empty_segment,
                "returned_revenue_vs_order_value_pct": None}

    def _segment_stats(seg: pd.DataFrame) -> dict:
        if seg.empty:
            return dict(empty_segment)
        cod = seg[seg["payment_method"] == "cod"]
        online = seg[seg["payment_method"] == "online"]
        return {
            "count": int(len(seg)),
            "shipping_revenue": round(seg["agreed_shipping_cost"].sum(), 2),
            "transfer_fee_revenue": round(seg["transfer_fee"].sum(), 2),
            "total_weevo_revenue": round(seg["weevo_revenue"].sum(), 2),
            "avg_weevo_revenue_per_order": round(seg["weevo_revenue"].mean(), 2),
            "total_order_value": round(seg["amount"].sum(), 2),
            "cod_count": int(len(cod)),
            "online_count": int(len(online)),
            "avg_client_order_value_cod": round(cod["amount"].mean(), 2) if not cod.empty else 0.0,
            "avg_client_order_value_online": round(online["amount"].mean(), 2) if not online.empty else 0.0,
            "total_merchant_payout_cod": round(cod["merchant_payout"].sum(), 2) if not cod.empty else 0.0,
        }

    delivered = completed[completed["status"] == "delivered"]
    returned = completed[completed["status"] == "returned"]

    returned_cod = returned[(returned["payment_method"] == "cod") & (returned["amount"] > 0)]
    if not returned_cod.empty:
        ratio_pct = round((returned_cod["weevo_revenue"] / returned_cod["amount"]).mean() * 100, 1)
    else:
        ratio_pct = None

    return {
        "delivered": _segment_stats(delivered),
        "returned": _segment_stats(returned),
        "returned_revenue_vs_order_value_pct": ratio_pct,
    }

def overdue_age_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """How overdue, grouped into buckets — '208 overdue' means something
    very different if most are 1 hour late vs most are 3 days late. This
    is exactly that breakdown."""
    risky = at_risk_shipments(df)
    if risky.empty:
        return pd.DataFrame(columns=["bucket", "orders"])
    bins = [0, 6, 24, 72, float("inf")]
    labels = ["0-6h overdue", "6-24h overdue", "1-3 days overdue", "3+ days overdue"]
    risky["bucket"] = pd.cut(risky["hours_overdue"], bins=bins, labels=labels, right=False)
    result = risky.groupby("bucket", observed=True).size().reset_index(name="orders")
    order_map = {label: i for i, label in enumerate(labels)}
    result["_sort"] = result["bucket"].map(order_map)
    return result.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

def overdue_by_area(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Which areas have the most overdue shipments right now — pairs with
    risk_by_courier() to answer 'is this a courier problem or an area
    problem' (e.g. one area consistently overdue regardless of courier
    points at traffic/distance/access issues, not individual performance)."""
    risky = at_risk_shipments(df)
    if risky.empty or "area" not in df.columns:
        return pd.DataFrame(columns=["area", "overdue_orders"])
    risky = risky.merge(df[["shipment_id", "area"]], on="shipment_id", how="left")
    return (
        risky.groupby("area").size().reset_index(name="overdue_orders")
        .sort_values("overdue_orders", ascending=False).head(n)
    )

ARCHIVE_V2_COLUMNS = [
    "shipment_id", "reference", "status", "merchant_id", "merchant_name",
    "courier_id", "courier_name", "client_name", "client_phone",
    "payment_method", "amount", "agreed_shipping_cost", "transfer_fee",
    "weevo_revenue", "merchant_payout", "delivering_street", "area",
    "created_at", "target_pickup_at", "pickup_actual_at", "target_deliver_at",
    "delivered_at", "returned_at", "delivery_hours", "is_overdue",
    "pickup_code", "delivered_code", "returned_code", "stored_at",
]
_V2_DATETIME_COLS = ["created_at", "target_pickup_at", "pickup_actual_at",
                      "target_deliver_at", "delivered_at", "returned_at", "stored_at"]
_V2_NUMERIC_COLS = ["amount", "agreed_shipping_cost", "transfer_fee",
                     "weevo_revenue", "merchant_payout", "delivery_hours"]

DEFAULT_V2_ARCHIVE_PATH = "./data/analytics_archive_v2.csv"

def _empty_v2_archive_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=ARCHIVE_V2_COLUMNS)
    for col in _V2_DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in _V2_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_overdue"] = df["is_overdue"].astype(bool)
    return df

def get_v2_archive_file_info(archive_path: str = DEFAULT_V2_ARCHIVE_PATH) -> dict:
    """Same diagnostic purpose as v1's get_archive_file_info() — lets the
    page show the archive's real on-disk state (path/exists/last
    modified/size) so a 'my save didn't stick' symptom is visible as
    evidence, not a guess."""
    abs_path = os.path.abspath(archive_path)
    exists = os.path.exists(archive_path)
    if not exists:
        return {"abs_path": abs_path, "exists": False, "last_modified": None,
                "size_kb": None, "row_count": 0}
    stat = os.stat(archive_path)
    return {
        "abs_path": abs_path,
        "exists": True,
        "last_modified": datetime.fromtimestamp(stat.st_mtime),
        "size_kb": round(stat.st_size / 1024, 1),
        "row_count": len(load_v2_archive(archive_path)),
    }

def load_v2_archive(archive_path: str = DEFAULT_V2_ARCHIVE_PATH) -> pd.DataFrame:
    """Reads whatever v2 data has been saved so far. Empty (but
    correctly-shaped) DataFrame on first run — same contract as v1's
    load_archive(). is_overdue is recomputed fresh here rather than
    trusted from the saved CSV — it's a live/current judgement
    (target_deliver_at < now for a non-terminal status), so a value saved
    yesterday would be stale today even though target_deliver_at itself
    didn't change."""
    if not os.path.exists(archive_path):
        return _empty_v2_archive_df()

    df = pd.read_csv(archive_path)
    for col in _V2_DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in _V2_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ARCHIVE_V2_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ARCHIVE_V2_COLUMNS].copy()
    now = pd.Timestamp.now()
    if "status" in df.columns and "target_deliver_at" in df.columns:
        df["is_overdue"] = (
            ~df["status"].isin(TERMINAL_STATUSES)
            & df["target_deliver_at"].notna()
            & (df["target_deliver_at"] < now)
        )
    else:
        df["is_overdue"] = False
    return df

def append_to_v2_archive(df: pd.DataFrame, archive_path: str = DEFAULT_V2_ARCHIVE_PATH) -> dict:
    """Merges `df` (output of load_shipments_v2(), i.e. already
    parse_shipment()-shaped) into the on-disk v2 archive. Same
    dedup-on-shipment_id, newest-write-wins contract as v1's
    append_to_archive() — never raises, always returns a summary dict."""
    if df is None or df.empty:
        return {"added": 0, "updated": 0, "total_in_archive": len(load_v2_archive(archive_path))}

    dir_name = os.path.dirname(archive_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    existing = load_v2_archive(archive_path)
    incoming = df.copy()
    incoming["stored_at"] = pd.Timestamp.now()
    incoming = incoming.reindex(columns=ARCHIVE_V2_COLUMNS)

    existing_ids = set(existing["shipment_id"].dropna()) if not existing.empty else set()
    incoming_ids = set(incoming["shipment_id"].dropna())
    added_count = len(incoming_ids - existing_ids)
    updated_count = len(incoming_ids & existing_ids)

    frames = [f for f in (existing, incoming) if not f.empty and not f.isna().all(axis=None)]
    combined = pd.concat(frames, ignore_index=True) if frames else incoming
    combined = combined.drop_duplicates(subset=["shipment_id"], keep="last")
    if "created_at" in combined.columns:
        combined = combined.sort_values("created_at", na_position="last").reset_index(drop=True)
    combined.drop(columns=["is_overdue"]).to_csv(archive_path, index=False)

    return {"added": added_count, "updated": updated_count, "total_in_archive": len(combined)}

def detect_v2_archive_gap(archive_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
    """Same purpose as v1's detect_archive_gap(): did too much time pass
    between saves such that some shipments scrolled out of the API's
    'most recent N' window before ever being captured? Anchored on
    created_at (a real event timestamp for every status, unlike v1's
    delivery_date) so no plausibility guard is needed here the way the
    main page needs one for in-flight target dates."""
    if archive_df is None or archive_df.empty or "created_at" not in archive_df.columns:
        return {"has_gap": False, "gap_hours": None, "archive_latest": None,
                "live_oldest": None, "reason": "no_archive_yet"}
    if live_df is None or live_df.empty or "created_at" not in live_df.columns:
        return {"has_gap": False, "gap_hours": None, "archive_latest": None,
                "live_oldest": None, "reason": "no_live_data"}

    archive_latest = pd.to_datetime(archive_df["created_at"], errors="coerce").max()
    live_oldest = pd.to_datetime(live_df["created_at"], errors="coerce").min()

    if pd.isna(archive_latest) or pd.isna(live_oldest):
        return {"has_gap": False, "gap_hours": None, "archive_latest": None,
                "live_oldest": None, "reason": "missing_dates"}

    gap_hours = (live_oldest - archive_latest).total_seconds() / 3600
    has_gap = gap_hours > 0.05

    return {
        "has_gap": bool(has_gap),
        "gap_hours": round(gap_hours, 1) if has_gap else 0.0,
        "archive_latest": archive_latest,
        "live_oldest": live_oldest,
        "reason": "gap_detected" if has_gap else "no_gap",
    }
