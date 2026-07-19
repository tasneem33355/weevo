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
import httpx  # matches app/services/external_api.py — requests is not in requirements.txt

from app.services.analytics_pipeline import (
    detect_area,
    enrich_areas_with_cache,
    fetch_admin_shipments,  # single shared admin-API fetch (2026-07-15 migration) — see load_shipments_v2 below
    _get_admin_bearer_token,  # shared Bearer-token cache/login (2026-07-15 migration) — see _get() below
)

WEEVO_API_BASE_URL = os.getenv("WEEVO_API_BASE_URL", "https://eg.api.weevoapp.com")

# Ahmed's stated priority: Delivered and Returned are the two states that
# matter most for the finance/ops analysis ("لو أدانا عشرة سلمنا تمانية
# ورجعنا اتنين"). These get the full page budget below.
PRIMARY_STATUSES = ["delivered", "returned"]

# Concurrency: NONE. Live evidence (2026-07-11): capping concurrent status
# requests at 3 still correlated with every single status failing at once
# (502 Bad Gateway — the backend itself rejecting/erroring, a step worse
# than the 504 timeouts seen before that change). All requests below are
# fired one at a time, with a short pause between each, so this pipeline
# never has more than one request in flight against their backend.
INTER_REQUEST_DELAY = 0.35   # seconds between every individual HTTP call —
                              # still used by fetch_reference_list() below
                              # (captains/merchants, unchanged) and shared
                              # with the admin shipments fetch imported from
                              # analytics_pipeline (its own pacing/time-budget
                              # constants live there now, see fetch_admin_shipments).

# Terminal states — a shipment in one of these is done, one way or another,
# and should never show up in the "at risk / overdue" view.
TERMINAL_STATUSES = {"delivered", "returned", "cancelled", "bulk-shipment-closed", "bulk-shipment-cancelled"}


# ---------------------------------------------------------------------------
# Low-level HTTP — same retry pattern as the v1 pipeline (502/503/504 and
# client-side timeouts get one retry at double the timeout before the
# failure is surfaced rather than masked).
# ---------------------------------------------------------------------------
GATEWAY_ERROR_CODES = {502, 503, 504}
REQUEST_TIMEOUT = 20   # seconds. A 502/503/504 means the gateway on THEIR
                        # side already gave up — a longer client timeout
                        # doesn't fix that, so retries below use a short
                        # pause instead of a bigger timeout.
MAX_RETRIES = 1         # 1 retry (2 attempts total) per request


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


# fetch_shipments_page / fetch_shipments_by_status_multipage / fetch_all_statuses
# (the old per-status AI-agent shipment fetch) are superseded by
# fetch_admin_shipments(), imported from analytics_pipeline — see
# load_shipments_v2() below. fetch_reference_list()/fetch_captains()/
# fetch_merchants() below are ALSO now migrated (2026-07-15) onto the same
# admin-5678vna9k6 Admin Dashboard backend and the same shared Bearer-token
# auth as shipments — confirmed against real captured merchants/couriers
# responses, see Merchant.docx.



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


# ---------------------------------------------------------------------------
# Parsing — raw API record -> flat row with every derived field the
# dashboard needs. This is the one place all the field-mapping knowledge
# from the raw JSON investigation lives.
# ---------------------------------------------------------------------------
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

    # Admin Dashboard API (2026-07-15 migration) has no `logs` array at all,
    # so this always evaluates to None now — confirmed with the business:
    # leave it null rather than approximate with a different timestamp, so
    # delivery_hours below is also always None. No code change needed here;
    # raw.get("logs") -> None -> (logs or []) -> [] handles it already.
    pickup_actual_at = _extract_actual_pickup_at(raw.get("logs"))
    delivered_at = _parse_dt(raw.get("delivered_date_at"))
    returned_at = _parse_dt(raw.get("returned_date_at"))
    completed_at = delivered_at or returned_at

    delivery_hours = None
    if pickup_actual_at is not None and completed_at is not None:
        delta_hours = (completed_at - pickup_actual_at).total_seconds() / 3600
        if delta_hours >= 0:  # guards against bad/out-of-order data
            delivery_hours = round(delta_hours, 2)

    target_deliver_at = _parse_dt(raw.get("date_to_deliver_shipment"))
    now = pd.Timestamp.now()
    is_overdue = (
        status not in TERMINAL_STATUSES
        and target_deliver_at is not None
        and target_deliver_at < now
    )

    # Weevo's own revenue on this shipment (confirmed formula, see module
    # docstring). merchant_payout only makes sense for cod (amount > 0);
    # for online payments the customer already paid the merchant directly,
    # so there's no "amount collected to hand back" — left as None.
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
    if not df.empty:
        df = enrich_areas_with_cache(df)  # free: reuses anything classified via v1 or the AI button
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


# ---------------------------------------------------------------------------
# Aggregations specific to what Ahmed asked for
# ---------------------------------------------------------------------------
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


def at_risk_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """Shipments still in flight whose target delivery time has already
    passed — the 'captain still has orders overdue' alert Ahmed described.
    Sorted by how overdue (most overdue first)."""
    risky = df[df["is_overdue"] == True].copy()  # noqa: E712 (explicit bool compare reads clearer here)
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

    # The specific comparison asked for: on returned COD orders, weevo's
    # shipping+fee revenue as a % of the order's own value. >100% means
    # Weevo earned MORE from the failed delivery than the order was worth.
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
    # at_risk_shipments() doesn't carry 'area' through — re-merge it here
    # rather than changing that function's return columns for everyone else.
    risky = risky.merge(df[["shipment_id", "area"]], on="shipment_id", how="left")
    return (
        risky.groupby("area").size().reset_index(name="overdue_orders")
        .sort_values("overdue_orders", ascending=False).head(n)
    )


# ---------------------------------------------------------------------------
# Archive (ADDED) — same purpose as v1's archive in analytics_pipeline.py:
# the live API only ever returns "most recent N per status", so the only
# way to ever answer a real historical date-range question is to save our
# own daily snapshots over time and read those back later.
#
# This is a SEPARATE file from v1's archive (DEFAULT_ARCHIVE_PATH), not an
# extension of it. Two deliberate reasons, both about avoiding the "ممنوع
# أي إيرور" risk of touching a working file:
#   1. Schema: v1's ARCHIVE_COLUMNS is a fixed 17-column shape used by
#      load_uploaded_csv()'s validation and by every v1 aggregation
#      function via strict `df[ARCHIVE_COLUMNS]` selection. Widening that
#      file to also hold v2's 26 columns (revenue, pickup logs, is_overdue)
#      would touch code that currently works.
#   2. Grain: v1 rows come from ALL statuses with a fixed pagination
#      budget; v2 rows come from PRIMARY_STATUSES with a much larger
#      budget (5,000/status) plus a separate risk-only pull. Merging them
#      into one file would mean most rows have huge gaps in one schema or
#      the other — not an error, just messy and easy to misread.
# Same shipment saved from both places (e.g. a delivered order visible to
# both v1 and v2) simply exists in both archive files under its own
# shipment_id — harmless, each file stays internally consistent.
# ---------------------------------------------------------------------------
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
    # is_overdue is a live judgement, not a fact worth freezing — drop it
    # from what's written to disk and let load_v2_archive() recompute it
    # fresh every read (see that function's docstring).
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
