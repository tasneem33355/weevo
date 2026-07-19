"""
Data Analytics page — plugs into the existing dashboard.py sidebar
selectbox as one more option, alongside Test Tia / System Status / etc.

Zero coupling with the rest of dashboard.py: this file only exposes
render_analytics_page(), which draws everything inside the current
Streamlit script run. It does not call st.set_page_config (dashboard.py
already does that once) and does not touch st.sidebar itself beyond
what dashboard.py's own selectbox already handles.

Dependency note: uses ONLY Streamlit's built-in chart elements
(st.bar_chart) plus plain HTML/CSS via st.markdown for card styling —
no new package needs to be added to requirements.txt or installed in
the deployment environment. Verified to boot with plotly uninstalled.

Integration required in dashboard.py (additive only — see bottom of
this file for the exact 3 copy-paste additions).
"""

import os
import re
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.analytics_pipeline import (
    load_shipments,
    orders_over_time,
    top_areas,
    courier_leaderboard,
    merchant_leaderboard,
    merchant_activity,
    merchants_with_zero_orders,
    recent_orders,
    summary_kpis,
    status_breakdown,
    load_archive,
    append_to_archive,
    detect_archive_gap,
    get_archive_file_info,
    load_uploaded_csv,
    DEFAULT_ARCHIVE_PATH,
    enrich_areas_with_cache,
    classify_unknown_addresses,
    HEAVY_STATUSES_V1,
    # ADDED (2026-07-15 admin-API migration): one shared fetch + a pure
    # per-pipeline dataframe builder, so both this page's main section and
    # its Financial/Risk section can be built from the exact same raw pull.
    fetch_admin_shipments,
    build_shipments_dataframe,
)

# Statuses whose delivery_date is a REAL event timestamp (order actually
# received/delivered) vs. a scheduled/target one (in-flight statuses use
# date_to_receive_shipment as a planned pickup time, which can be today,
# tomorrow, or occasionally a stale target far in the past — see the date
# range filter below for why this distinction matters). Reuses the same
# set the pipeline already uses for its own heavy/light pagination split,
# so the two can't drift apart.
DATED_STATUSES = HEAVY_STATUSES_V1

# v2: separate module, talks to the real Weevo API directly using the exact
# field names/financial formula confirmed from a live raw JSON response
# (2026-07-10). Kept fully separate from the v1 functions above — nothing
# above this import changes, this only adds the new Financial/Risk section
# further down the page.

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.analytics_pipeline_v2 import (
    load_shipments_v2,
    build_v2_dataframe,  # ADDED (2026-07-15) — pure mapping, used on the same shared raw pull as v1
    revenue_summary,
    financial_breakdown,
    delivery_time_summary as v2_delivery_time_summary,
    at_risk_shipments,
    risk_by_courier,
    overdue_age_buckets,
    overdue_by_area,
    fetch_merchants,
    PRIMARY_STATUSES,
    DEFAULT_V2_ARCHIVE_PATH,
    append_to_v2_archive,
    load_v2_archive,
    get_v2_archive_file_info,
    detect_v2_archive_gap,
)

PURPLE = "#5B4CFF"
TEAL = "#00A389"
CORAL = "#E8623D"
AMBER = "#C88A1D"
RED = "#C4453A"
INK = "#1A1A2E"
MUTED = "#6B6B80"
CARD_BG = "#F7F7FB"
BORDER = "#E9E9F2"

STATUS_STYLE = {
    "Healthy":   ("#0F6E56", "#E1F5EE", "🟢"),
    "Growing":   ("#0F6E56", "#E1F5EE", "🟢"),
    "New":       ("#185FA5", "#E6F1FB", "🆕"),
    "Watch":     ("#854F0B", "#FAEEDA", "🟡"),
    "Declining": ("#A32D2D", "#FCEBEB", "🔴"),
}

CACHE_TTL_SECONDS = 300  # 5 minutes


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching latest shipment data from Weevo…")
def _cached_admin_shipments(api_key: str, start_date, end_date):
    """
    ADDED (2026-07-15 admin-API migration) — the ONE shared live fetch.
    Both _cached_api_load (main section) and _cached_v2_load (Financial/
    Risk section) below call this with the same (api_key, start_date,
    end_date), so within a single script run — and across reruns inside
    the cache TTL — they hit the same cache entry and are guaranteed to be
    built from byte-identical raw records. This is what makes "both
    sections always use the exact same dataset" true, not just a
    convention someone has to remember to keep passing the same args.

    start_date/end_date are "YYYY-MM-DD" strings or None (None/None = no
    date filter sent at all, i.e. "All time" — same as this always meant:
    no additional limit invented here, the only ceiling is the existing
    fetch time budget inside fetch_admin_shipments itself).
    """
    return fetch_admin_shipments(api_key=api_key, start_date=start_date, end_date=end_date)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching latest shipment data from Weevo…")
def _cached_api_load(api_key: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Caches the live API pull for CACHE_TTL_SECONDS.

    Without this, Streamlit re-runs this whole page top-to-bottom on
    EVERY interaction — clicking a filter, opening an expander, anything
    — which means the full API fetch (already improved to run in
    parallel, but still bounded by whichever status is slowest that
    moment) was happening again on every single click, not just on first
    load. That's the main reason it felt like it "never stopped loading".
    Now it's fetched once, reused for 5 minutes, with a manual refresh
    button below for anyone who wants the latest data sooner.

    start_date/end_date ADDED 2026-07-15 — forwarded straight to the
    shared fetch above so a changed date-range selection actually
    refetches (previously the cache key was only api_key, so changing
    the date preset never invalidated the cache either).
    """
    raw_records, status_counts, fetch_meta = _cached_admin_shipments(api_key, start_date, end_date)
    return build_shipments_dataframe(raw_records, fetch_meta=fetch_meta, status_counts=status_counts)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Pulling delivered/returned shipments for financial + risk analysis…")
def _cached_v2_load(api_key: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Same caching rationale as _cached_api_load above, separate cache entry
    since it's a differently-shaped DataFrame — but built from the SAME
    shared _cached_admin_shipments() pull (see above), not a second live
    fetch. This is the fix for "the two sections must never fetch
    different data" — previously this called load_shipments_v2(), which
    did its own separate, differently-scoped HTTP pull.
    """
    raw_records, _status_counts, fetch_meta = _cached_admin_shipments(api_key, start_date, end_date)
    return build_v2_dataframe(raw_records, fetch_meta=fetch_meta)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching full merchant roster…")
def _cached_merchant_roster(api_key: str) -> pd.DataFrame:
    """
    The FULL registered merchant list (from the Admin Dashboard
    admin-5678vna9k6/merchants endpoint, 2026-07-15 migration — see
    analytics_pipeline_v2.fetch_merchants),
    not just merchants who happen to appear in the shipments pulled above.
    This is what makes it possible to show merchants with genuinely ZERO
    orders in the period — merchant_leaderboard() alone can never show
    that, since it's built purely from shipment records (a merchant with
    no orders never appears in a shipment at all).
    """
    return fetch_merchants(api_key=api_key)


def _clean(html: str) -> str:
    """
    Streamlit/Markdown gotcha fix: any line that starts with 4+ spaces of
    indentation gets treated as a fenced code block by the Markdown parser,
    even with unsafe_allow_html=True — the HTML tags then show up as raw
    text instead of being rendered (this was the bug in the screenshot).
    This strips leading whitespace from every line before it reaches
    st.markdown, without touching the actual HTML structure/content.
    """
    return re.sub(r"\n[ \t]+", "\n", html).strip()


def _inject_css():
    st.markdown(
        _clean(f"""
        <style>
        #weevo-analytics * {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
        .wa-header {{
            display: flex; justify-content: space-between; align-items: flex-end;
            margin-bottom: 4px;
        }}
        .wa-title {{ font-size: 26px; font-weight: 700; color: {INK}; margin: 0; }}
        .wa-subtitle {{ font-size: 13px; color: {MUTED}; margin: 2px 0 0; }}
        .wa-synced {{ font-size: 12px; color: {MUTED}; text-align: right; }}
        .wa-kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0 22px; }}
        .wa-kpi {{
            background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 14px 16px; position: relative; overflow: hidden;
        }}
        a.wa-kpi-link {{ text-decoration: none; color: inherit; display: block; }}
        a.wa-kpi-link .wa-kpi {{ cursor: pointer; transition: border-color 0.15s ease; }}
        a.wa-kpi-link .wa-kpi:hover {{ border-color: {PURPLE}; }}
        .wa-kpi-breakdown {{ font-size: 11px; color: {MUTED}; margin-top: 6px; line-height: 1.5; }}
        .wa-kpi-icon {{
            width: 30px; height: 30px; border-radius: 9px; display: flex;
            align-items: center; justify-content: center; font-size: 15px; margin-bottom: 10px;
        }}
        .wa-kpi-label {{ font-size: 12px; color: {MUTED}; margin-bottom: 2px; }}
        .wa-kpi-value {{ font-size: 22px; font-weight: 700; color: {INK}; }}
        .wa-kpi-delta {{ font-size: 11px; margin-top: 4px; font-weight: 600; }}
        .wa-section {{
            background: white; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 16px 18px; margin-bottom: 16px;
        }}
        .wa-section-title {{ font-size: 15px; font-weight: 700; color: {INK}; margin: 0 0 2px; }}
        .wa-section-sub {{ font-size: 12px; color: {MUTED}; margin: 0 0 12px; }}
        .wa-badge {{
            display: inline-flex; align-items: center; gap: 4px; font-size: 11px;
            font-weight: 600; padding: 3px 9px; border-radius: 999px;
        }}
        .wa-mock-banner {{
            background: #FFF4E5; border: 1px solid #FFD79A; color: #8A5300;
            padding: 8px 14px; border-radius: 10px; font-size: 12px; margin-bottom: 14px;
        }}
        .wa-alert-banner {{
            background: #FCEBEB; border: 1px solid #F0AFAF; color: #A32D2D;
            padding: 10px 14px; border-radius: 10px; font-size: 13px; font-weight: 600;
            margin-bottom: 16px;
        }}
        </style>
        """),
        unsafe_allow_html=True,
    )


def _kpi_card(icon: str, icon_bg: str, icon_color: str, label: str, value: str, delta: str, delta_color: str,
              breakdown: str = "", anchor: str = "") -> str:
    breakdown_html = f'<div class="wa-kpi-breakdown">{breakdown}</div>' if breakdown else ""
    card = _clean(f"""
    <div class="wa-kpi">
        <div class="wa-kpi-icon" style="background:{icon_bg}; color:{icon_color};">{icon}</div>
        <div class="wa-kpi-label">{label}</div>
        <div class="wa-kpi-value">{value}</div>
        <div class="wa-kpi-delta" style="color:{delta_color};">{delta}</div>
        {breakdown_html}
    </div>
    """)
    if anchor:
        return f'<a class="wa-kpi-link" href="#{anchor}">{card}</a>'
    return card


def render_analytics_page():
    st.markdown('<div id="weevo-analytics">', unsafe_allow_html=True)
    _inject_css()

    # =========================================================================
    # TEMPORARY DEBUG (2026-07-16, 401 investigation) — REMOVE once resolved.
    # Placed here, at the very top of the page body (not the sidebar, not an
    # expander that could go unnoticed), so it's visible immediately without
    # needing Streamlit Cloud log/secrets access. Traces the ONE point where
    # api_key becomes a concrete value in this app: os.environ["WEEVO_API_KEY"]
    # (Streamlit secrets.toml root-level entries are exposed as env vars
    # automatically — entries nested under a [section] are NOT), with the
    # sidebar's session-only text_input as the only fallback if that's empty.
    # Nothing else in the codebase sources api_key from anywhere else — no
    # config file, no direct GitHub Actions secret read at runtime. Shows
    # only existence/length/a 12-char prefix — never the full secret — so it
    # can be compared against the JWT captured from the Weevo Admin Dashboard.
    _debug_api_key = os.environ.get("WEEVO_API_KEY", "")
    st.markdown(
        _clean(f"""
        <div style="background:#FFF4E5; border:1px solid #FFD79A; color:#8A5300;
                     padding:10px 14px; border-radius:10px; font-size:13px; margin-bottom:12px;">
        <b>🔧 TEMPORARY DEBUG — api_key (WEEVO_API_KEY) at runtime</b><br>
        exists: <b>{bool(_debug_api_key)}</b> &nbsp;|&nbsp;
        length: <b>{len(_debug_api_key)}</b> &nbsp;|&nbsp;
        prefix (first 12 chars): <b>{_debug_api_key[:12] if _debug_api_key else "(empty)"}</b>
        </div>
        """),
        unsafe_allow_html=True,
    )
    # --- END TEMPORARY DEBUG -------------------------------------------------

    # ---- Server-side date filter — computed BEFORE the fetch (ADDED) ------
    # The date-range picker widget itself still renders further down the
    # page in its ORIGINAL spot (see "Date range filter" below) — nothing
    # about its position, label, or options changes. This block only reads
    # whatever value is ALREADY in st.session_state for that widget's key,
    # so the live-API fetch below can use it too. Streamlit updates
    # session_state for a widget's key before the script reruns on that
    # widget's own change, so by the time this block runs on the rerun
    # triggered by picking a new preset, the new value is already here.
    #
    # This is the fix for the CEO's top priority: previously "Last 7 days"
    # only sliced an already-loaded, bounded "most recent N" pull
    # client-side — now the exact same preset is sent to the Admin
    # Dashboard backend (start_delivery_date/end_delivery_date) BEFORE any
    # analytics are calculated, so the numbers match the official
    # dashboard for that range. "All time" is unchanged: no date filter is
    # sent at all, same as it always meant — no new artificial limit.
    _preset = st.session_state.get("wa_date_preset", "All time")
    _today_real = datetime.now().date()
    admin_start_date = None
    admin_end_date = None
    if _preset == "Custom range":
        _custom = st.session_state.get("wa_date_range_custom")
        # GUARD (2026-07-17): mid-selection (e.g. only "From" picked, "To"
        # not yet chosen), Streamlit's range date_input stores a bare
        # datetime.date here instead of a tuple/list — len() on a plain
        # date raises TypeError. Only treat it as a complete range when
        # it's actually a 2-item tuple/list.
        if (
            isinstance(_custom, (tuple, list))
            and len(_custom) == 2
            and _custom[0]
            and _custom[1]
        ):
            admin_start_date, admin_end_date = _custom
        else:
            # FIX (2026-07-19): see docstring above this block — reuse the
            # last fully-resolved range instead of falling through to the
            # 30-day default, so a mid-pick (only "From" chosen) never
            # changes start_date/end_date and never triggers a re-fetch.
            _last_resolved = st.session_state.get("wa_admin_resolved_range")
            if _last_resolved:
                admin_start_date, admin_end_date = _last_resolved
    if admin_start_date is None or admin_end_date is None:
        # BUG FIX (2026-07-16): "All time" used to send no date filter at
        # all, so the fetch tried to page through the entire history —
        # hitting the 90s per-status time budget ("Stopped after the 90s
        # time budget (page 72 of 1397)") long before reaching the end.
        # Falling back to the same reasonable default window as "Last 30
        # days" keeps the page fast by requesting fewer pages, without
        # touching the time budget itself. Named presets are unaffected
        # (each still maps to its own day count); "All time" resolves
        # through this same default via dict.get's fallback — and so does
        # "Custom range" before the user has finished picking both dates
        # (2026-07-17: previously left start/end at None in that case,
        # same unbounded-fetch defect as the old "All time" bug).
        _preset_days = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}.get(_preset, 30)
        admin_end_date = _today_real
        admin_start_date = _today_real - timedelta(days=_preset_days)
    admin_start_str = admin_start_date.strftime("%Y-%m-%d") if admin_start_date else None
    admin_end_str = admin_end_date.strftime("%Y-%m-%d") if admin_end_date else None
    st.session_state["wa_admin_resolved_range"] = (admin_start_date, admin_end_date)

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Analytics data source**")
        source_label = st.radio(
            "Source",
            ["Live API", "Uploaded Archive"],
            index=0,
            label_visibility="collapsed",
        )

        api_key = os.environ.get("WEEVO_API_KEY", "")
        if source_label == "Live API":
            if api_key:
                st.caption("✓ Integration key found in environment")
            else:
                api_key = st.text_input(
                    "Integration key",
                    type="password",
                    help="Session-only — not saved anywhere. For repeated use, "
                         "set the WEEVO_API_KEY environment variable instead so "
                         "it never has to be typed or stored in this page.",
                )
            if api_key:
                st.caption(f"Data is cached for {CACHE_TTL_SECONDS // 60} min to keep the page fast — "
                           f"filters/clicks reuse it instead of re-fetching every time.")
                if st.button("🔄 Refresh now", help="Fetch the latest data immediately, bypassing the cache."):
                    _cached_api_load.clear()
                    # No explicit st.rerun() needed: this button click already
                    # triggers a full script rerun on its own (standard
                    # Streamlit behavior), and this cache-clear happens earlier
                    # in that same rerun than the _cached_api_load() call below
                    # — so the next load is already fresh without forcing a
                    # second rerun on top of it.

        archive_path = DEFAULT_ARCHIVE_PATH
        v2_archive_path = DEFAULT_V2_ARCHIVE_PATH
        uploaded_files = None
        if source_label == "Uploaded Archive":
            with st.expander("Archive location", expanded=False):
                archive_path = st.text_input("Archive file path", value=DEFAULT_ARCHIVE_PATH)
                v2_archive_path = st.text_input(
                    "Financial/risk archive file path (v2)",
                    value=DEFAULT_V2_ARCHIVE_PATH,
                    help="Separate file from the one above — holds the richer "
                         "revenue/pickup-time/overdue data the Financial Summary "
                         "section below uses. Built up the same way, via the "
                         "'Save this snapshot' button in Live API mode.",
                )
            uploaded_files = st.file_uploader(
                "Upload snapshot CSV(s) — day / week / month exports",
                type=["csv"], accept_multiple_files=True,
                help="Use CSVs downloaded from this dashboard's 'Download snapshot' "
                     "button (in Live API mode) so the columns line up.",
            )

        with st.expander("Advanced: local DB cache instead"):
            use_db_cache = st.checkbox("Use local weevo_chatbot.db cache", value=False)
            db_path = st.text_input("Database path", value="./weevo_chatbot.db", disabled=not use_db_cache)

    load_error = None
    upload_messages = []

    if use_db_cache:
        source, active_source_label = "db", "Local cache"
    elif source_label == "Live API":
        source, active_source_label = "api", "Live API"
    else:
        source, active_source_label = "archive", "Uploaded Archive"

    def _fallback(reason_kind: str, reason_detail: str):
        """Shared fallback order when the primary source can't be loaded:
        try the archive first (real history, even if the live source is
        down), and only drop to demo data if there's truly nothing else."""
        nonlocal load_error
        load_error = (reason_kind, reason_detail)
        archived = load_archive(archive_path if source_label == "Uploaded Archive" else DEFAULT_ARCHIVE_PATH)
        if not archived.empty:
            return archived, "Archived data (fallback)"
        return load_shipments(source="mock"), "Demo data (fallback)"

    try:
        if source == "api":
            if not api_key:
                st.info("Enter the Integration key in the sidebar to load live data.")
                df_all, active_source_label = _fallback("api", "No Integration key provided.")
            else:
                df_all = _cached_api_load(api_key, start_date=admin_start_str, end_date=admin_end_str)
                # Cheap (local file read only) — makes any address classified
                # via the "Classify unknown areas with AI" button below show
                # up immediately on rerun, without waiting for the 5-minute
                # API cache to expire or re-hitting the slow Weevo API.
                df_all = enrich_areas_with_cache(df_all)
        elif source == "db":
            df_all = load_shipments(source="db", db_path=db_path)
        else:  # archive
            if uploaded_files:
                for f in uploaded_files:
                    try:
                        parsed = load_uploaded_csv(f)
                        summary = append_to_archive(parsed, archive_path=archive_path)
                        upload_messages.append(
                            (True, f"{f.name}: {summary['added']} new, {summary['updated']} updated.")
                        )
                    except ValueError as e:
                        upload_messages.append((False, f"{f.name}: {e}"))
            df_all = load_archive(archive_path)
    except FileNotFoundError:
        df_all, active_source_label = _fallback("db", db_path)
    except (ConnectionError, ValueError) as e:
        df_all, active_source_label = _fallback("api", str(e))

    # ---- Header ---------------------------------------------------------
    badge_color = (
        TEAL if active_source_label == "Live API"
        else "#185FA5" if active_source_label == "Uploaded Archive"
        else AMBER if "fallback" in active_source_label or active_source_label == "Local cache"
        else MUTED
    )
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(
            _clean(f"""
            <p class="wa-title">📈 Weevo analytics</p>
            <p class="wa-subtitle">Operational visibility for orders, merchants and couriers &nbsp;
            <span class="wa-badge" style="color:{badge_color}; background:{badge_color}22;">● {active_source_label}</span>
            </p>
            """),
            unsafe_allow_html=True,
        )
    with hcol2:
        st.markdown(
            f'<div class="wa-synced">Last synced<br>'
            f'<b>{datetime.now().strftime("%d %b %Y, %I:%M %p")}</b></div>',
            unsafe_allow_html=True,
        )

    for ok, msg in upload_messages:
        (st.success if ok else st.error)(msg)

    # ---- Data coverage (ADDED) --------------------------------------------
    # Answers "data for how long back?" concretely instead of leaving it
    # implicit. The pull is a fixed COUNT of most-recent records per status
    # (not a fixed calendar window), so the actual time span it covers
    # shifts with order volume — showing the real min/max date found in
    # what was actually loaded is the only honest way to answer that.
    #
    # BUG FIX (2026-07-12): in-flight delivery_date is a scheduled/target
    # time (date_to_receive_shipment), not something that already
    # happened. A genuinely still-in-flight order realistically can't have
    # a target date from months ago — that's stale/bad data (e.g. an
    # order that got stuck and never closed out), not a real 5-month-long
    # delivery. A single such record was silently dragging this whole
    # banner's "in-flight" range back to January. Fixed by excluding
    # implausible target dates from the range shown here specifically
    # (order counts elsewhere are NOT affected — this only changes what
    # date range gets displayed/anchored on).
    IN_FLIGHT_PLAUSIBLE_PAST_DAYS = 45   # a target pickup date older than this is treated as stale data, not a real long-running order
    IN_FLIGHT_PLAUSIBLE_FUTURE_DAYS = 14  # scheduled further ahead than this is treated the same way

    def _plausible_in_flight_mask(dates: pd.Series) -> pd.Series:
        now = pd.Timestamp.now()
        lower = now - pd.Timedelta(days=IN_FLIGHT_PLAUSIBLE_PAST_DAYS)
        upper = now + pd.Timedelta(days=IN_FLIGHT_PLAUSIBLE_FUTURE_DAYS)
        return dates.between(lower, upper)

    if not df_all.empty and "delivery_date" in df_all.columns and df_all["delivery_date"].notna().any():
        heavy_mask = df_all["status"].isin(["delivered", "returned"]) if "status" in df_all.columns else pd.Series(False, index=df_all.index)
        heavy_df, light_df = df_all[heavy_mask], df_all[~heavy_mask]
        coverage_bits = []
        stale_note = ""
        if not heavy_df.empty:
            coverage_bits.append(
                f"delivered/returned: {heavy_df['delivery_date'].min():%d %b} → "
                f"{heavy_df['delivery_date'].max():%d %b} ({len(heavy_df):,} orders)"
            )
        if not light_df.empty:
            plausible_mask = _plausible_in_flight_mask(light_df["delivery_date"])
            plausible_light = light_df[plausible_mask]
            stale_count = int((~plausible_mask).sum())
            if not plausible_light.empty:
                coverage_bits.append(
                    f"in-flight statuses: {plausible_light['delivery_date'].min():%d %b} → "
                    f"{plausible_light['delivery_date'].max():%d %b} ({len(light_df):,} orders)"
                )
            elif len(light_df) > 0:
                coverage_bits.append(f"in-flight statuses: ({len(light_df):,} orders, no plausible target dates)")
            if stale_count:
                stale_note = (
                    f" ⚠️ {stale_count} in-flight order(s) excluded from this date range specifically "
                    f"for having an implausible target date (more than {IN_FLIGHT_PLAUSIBLE_PAST_DAYS} days "
                    f"in the past or {IN_FLIGHT_PLAUSIBLE_FUTURE_DAYS} days in the future) — likely stale "
                    f"data worth checking with the backend team. These orders still count normally "
                    f"everywhere else on this page, just not in this date-range display."
                )
        if coverage_bits:
            st.caption(
                "📅 Data actually covers — " + " · ".join(coverage_bits)
                + ". This is the most recent N orders per status, not a fixed calendar "
                  "window — the span shifts with order volume." + stale_note
            )

    if load_error:
        kind, detail = load_error
        st.markdown(
            f'<div class="wa-mock-banner">Showing {active_source_label.lower()} — '
            f'primary source unavailable: {detail}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Technical details"):
            st.code(detail)

    # ---- Partial fetch failure (some statuses OK, some not) --------------
    # This is different from load_error above: the page still has real data
    # and renders normally, but a slice of it (e.g. every "delivered" order)
    # may be silently missing — which is exactly what caused confusing swings
    # between refreshes before. Never hide this.
    fetch_diag = getattr(df_all, "attrs", {}).get("fetch_diagnostics")
    if fetch_diag and fetch_diag.get("failed_statuses") and not load_error:
        failed = fetch_diag["failed_statuses"]
        succeeded = fetch_diag["succeeded_statuses"]
        total = fetch_diag["total_statuses"]
        st.markdown(
            f'<div class="wa-mock-banner">⚠️ Partial data — {len(failed)} of {total} '
            f'order statuses failed to load ({", ".join(failed)}). Numbers below may be '
            f'incomplete for those statuses — try refreshing the page.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Which statuses loaded, and why the rest failed"):
            st.write(f"✅ Loaded successfully: {', '.join(succeeded) if succeeded else '(none)'}")
            st.write(f"❌ Failed: {', '.join(failed)}")
            st.code(fetch_diag["errors"])

    # ---- Snapshot tools (Live API mode only) -------------------------
    if source == "api" and api_key and not df_all.empty and not load_error:
        snap_col1, snap_col2, snap_col3 = st.columns([1.6, 1.3, 1.3])
        with snap_col1:
            if st.button("💾 Save this snapshot to archive (v1 + Financial/v2)"):
                summary = append_to_archive(df_all, archive_path=archive_path)
                msg = (
                    f"Main data: {summary['added']} new, {summary['updated']} updated, "
                    f"{summary['total_in_archive']} total."
                )
                # Same click also archives the richer v2 (financial/risk) shape —
                # this is what "the archive should save everything I fetch, not
                # just v1" means in practice: one save action, both archives kept
                # in sync, rather than two separate buttons someone can forget.
                try:
                    df_v2_for_save = _cached_v2_load(api_key, start_date=admin_start_str, end_date=admin_end_str)
                    if not df_v2_for_save.empty:
                        v2_summary = append_to_v2_archive(df_v2_for_save, archive_path=v2_archive_path)
                        msg += (
                            f" | Financial/v2: {v2_summary['added']} new, "
                            f"{v2_summary['updated']} updated, {v2_summary['total_in_archive']} total."
                        )
                    else:
                        msg += " | Financial/v2: nothing to save (empty load)."
                except Exception as e:
                    msg += f" | Financial/v2 save failed: {e}"
                st.success(msg)
        with snap_col2:
            st.download_button(
                "⬇️ Download main snapshot",
                data=df_all.to_csv(index=False).encode("utf-8"),
                file_name=f"weevo_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        with snap_col3:
            try:
                df_v2_for_download = _cached_v2_load(api_key, start_date=admin_start_str, end_date=admin_end_str)
            except Exception:
                df_v2_for_download = pd.DataFrame()
            st.download_button(
                "⬇️ Download financial snapshot",
                data=df_v2_for_download.to_csv(index=False).encode("utf-8"),
                file_name=f"weevo_financial_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                disabled=df_v2_for_download.empty,
            )

        # Always-visible ground truth about the archive file itself. If
        # "778 new, 0 updated" repeats identically on every save, this
        # panel is where that gets diagnosed: either the resolved path
        # changes between loads, or last-modified doesn't reflect the save
        # that was JUST clicked — both point to the file not surviving
        # between requests (hosting/storage config), not a code bug.
        with st.expander("🔍 Archive file diagnostics"):
            info = get_archive_file_info(archive_path)
            st.write(f"**Resolved path:** `{info['abs_path']}`")
            if info["exists"]:
                st.write(f"**Last modified:** {info['last_modified']:%Y-%m-%d %H:%M:%S}")
                st.write(f"**Size:** {info['size_kb']} KB — **{info['row_count']:,} rows**")
                st.caption(
                    "If 'Last modified' does NOT update to just now right after you click "
                    "'Save this snapshot', or this path looks different on a page reload, "
                    "the server isn't persisting this file between requests — that's a "
                    "hosting/storage question for the backend team, not something fixable "
                    "from this page."
                )
            else:
                st.write("**File does not exist yet** — click 'Save this snapshot' once to create it.")

        # Warns if too much time passed since the last save such that some
        # shipments already scrolled out of the API's "most recent N"
        # window before ever being archived — those are gone for good, so
        # this needs to be visible *before* it happens again, not after.
        gap_info = detect_archive_gap(load_archive(archive_path), df_all)
        if gap_info["reason"] == "no_archive_yet":
            st.caption(
                "No archive saved yet — click 'Save this snapshot' to start "
                "building history. Save regularly (daily is safest) so no "
                "shipments scroll out of the live window uncaptured."
            )
        elif gap_info["has_gap"]:
            st.warning(
                f"⚠️ Archive gap detected: the last saved snapshot ends at "
                f"{gap_info['archive_latest']:%Y-%m-%d %H:%M}, but the oldest "
                f"shipment visible live right now starts at "
                f"{gap_info['live_oldest']:%Y-%m-%d %H:%M} — roughly "
                f"{gap_info['gap_hours']:.0f}h with no coverage in between. "
                "Shipments from that window weren't captured and can't be "
                "recovered now. Save more frequently to avoid this going forward."
            )
        elif gap_info["reason"] == "no_gap":
            st.caption("✓ Archive is up to date — no gap since the last saved snapshot.")

        with st.expander("🔍 Financial (v2) archive diagnostics"):
            v2_info = get_v2_archive_file_info(v2_archive_path)
            st.write(f"**Resolved path:** `{v2_info['abs_path']}`")
            if v2_info["exists"]:
                st.write(f"**Last modified:** {v2_info['last_modified']:%Y-%m-%d %H:%M:%S}")
                st.write(f"**Size:** {v2_info['size_kb']} KB — **{v2_info['row_count']:,} rows**")
            else:
                st.write("**File does not exist yet** — click 'Save this snapshot' once to create it.")

        try:
            v2_gap_info = detect_v2_archive_gap(
                load_v2_archive(v2_archive_path),
                _cached_v2_load(api_key, start_date=admin_start_str, end_date=admin_end_str),
            )
        except Exception:
            v2_gap_info = {"reason": "no_live_data", "has_gap": False}
        if v2_gap_info["reason"] == "no_archive_yet":
            st.caption(
                "No Financial/v2 archive saved yet — same 'Save this snapshot' button "
                "above covers it too."
            )
        elif v2_gap_info["has_gap"]:
            st.warning(
                f"⚠️ Financial (v2) archive gap detected: last saved snapshot ends at "
                f"{v2_gap_info['archive_latest']:%Y-%m-%d %H:%M}, but the oldest shipment "
                f"visible live right now starts at {v2_gap_info['live_oldest']:%Y-%m-%d %H:%M} "
                f"— roughly {v2_gap_info['gap_hours']:.0f}h with no coverage in between."
            )
        elif v2_gap_info["reason"] == "no_gap":
            st.caption("✓ Financial (v2) archive is up to date — no gap since the last saved snapshot.")

    if df_all.empty:
        if source == "archive" and not load_error:
            st.info(
                "No archived snapshots yet. Switch to 'Live API' and click "
                "'Save this snapshot to archive', or upload a CSV above."
            )
        else:
            st.warning("No shipment data found yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ---- Filters ----------------------------------------------------------
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        area_options = sorted(df_all["area"].dropna().unique().tolist())
        selected_areas = st.multiselect("Area", area_options, default=[], placeholder="All areas")
    with fcol2:
        merchant_options = sorted(df_all["merchant_name"].dropna().unique().tolist())
        selected_merchants = st.multiselect("Merchant", merchant_options, default=[], placeholder="All merchants")
    with fcol3:
        granularity_label = st.selectbox("Group orders by", ["Daily", "Weekly", "Monthly"], index=1)
        granularity_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}

    # ---- Date range filter (ADDED) -----------------------------------------
    # Filters by delivery_date (= date_to_receive_shipment). Only ever
    # narrows down whatever was actually loaded above — it can't pull in
    # data from further back than what's already in df_all. On "Live API"
    # that's usually a few days at most (see the "Data actually covers"
    # caption above); on "Uploaded Archive" it can be as wide as whatever
    # has been saved over time. Picking "Last 30 days" doesn't guarantee a
    # full month of DATA — only that anything older than 30 days is excluded.
    #
    # BUG FIX (2026-07-12): min/max used to come from ALL statuses,
    # in-flight included. In-flight delivery_date is a scheduled/target
    # time, not an "occurred" one, so a single stale target (e.g. an old
    # in-flight shipment that was never marked delivered) silently dragged
    # min_date back by months. That corrupted this block two ways at once:
    # it suppressed the "data only goes back to X" clamp caption below
    # (min_date looked artificially old, so the clamp condition never
    # triggered), and it made preset windows for in-flight orders
    # meaningless. Fixed by anchoring min/max to DATED_STATUSES
    # (delivered/returned) only — see the filtering step further down for
    # the matching fix on the actual row-filtering side.
    date_col = df_all.loc[
        df_all["status"].isin(DATED_STATUSES), "delivery_date"
    ].dropna() if "status" in df_all.columns else df_all["delivery_date"].dropna()
    dcol1, dcol2 = st.columns([1, 2])
    date_range = None
    if date_col.empty and not df_all["delivery_date"].dropna().empty:
        st.caption(
            "Date range filter isn't available — no delivered/returned orders are "
            "loaded yet to anchor it to (only in-flight orders, whose dates are "
            "scheduled targets rather than real event times)."
        )
    if not date_col.empty:
        min_date, max_date = date_col.min().date(), date_col.max().date()
        # BUG FIX (2026-07-11): presets used to anchor "Last 24 hours" etc.
        # to max_date (the latest value actually present in the data).
        # For in-flight orders, delivery_date comes from
        # date_to_receive_shipment — a PLANNED pickup time that can be
        # scheduled for later today or tomorrow, not something that already
        # happened. If even one in-flight order has a near-future target
        # time, max_date gets pulled forward past "today", and "Last 24
        # hours" ends up anchored to that future point instead of the real
        # current moment — silently including a wide net of scheduled
        # in-flight orders that a real "last 24 hours" shouldn't. This is
        # why a manually-picked recent custom range could show FEWER
        # orders than the "Last 24 hours" preset: the preset's anchor was
        # inflated by future-scheduled dates, the custom pick wasn't.
        # Fix: presets now anchor to the real wall-clock "now", same as
        # anyone would expect "last 24 hours" to mean. Custom range is
        # unaffected — it already used real picked calendar dates.
        today_real = pd.Timestamp.now().date()
        with dcol1:
            preset = st.selectbox(
                "Date range",
                ["All time", "Last 24 hours", "Last 7 days", "Last 30 days", "Custom range"],
                index=0,
                key="wa_date_preset",
            )
        if preset == "Custom range":
            with dcol2:
                # BUG FIX (2026-07-16): passing value=(min_date, max_date)
                # on every rerun — even with `key` set — resets the widget
                # back to the full default range the instant a single date
                # is clicked in the calendar popup (Streamlit sees an
                # in-progress 1-tuple selection as incomplete and reseeds
                # from `value` on the next rerun before the second date can
                # be picked). Typing both dates into the text box commits a
                # full 2-tuple in one action, so it never hit this. Seeding
                # the default into session_state ONCE instead, and no longer
                # passing `value=` on every call, lets calendar clicks
                # persist across reruns like any other keyed widget.
                # WIDENED BOUND (2026-07-17): st.date_input can resolve an
                # internal default of today's real date before any date is
                # picked. If max_date (latest delivery_date actually in the
                # data) is behind today, that internal default lands outside
                # max_value and raises. Widening ONLY the bound passed to
                # this widget to include today keeps that default in-range —
                # max_date itself is untouched everywhere else on the page
                # (captions, presets, "data actually covers" display), so a
                # picked range that reaches today simply returns whatever
                # data exists (i.e. up through the last real shipment) with
                # no separate fallback logic needed.
                _widget_max_date = max(max_date, today_real)
                # WIDENED LOWER BOUND (2026-07-19): min_date reflects only
                # whatever is currently loaded — after picking a narrow
                # custom range (e.g. 07/15-07/16), the next fetch only
                # covers that span, so min_date collapsed to it too and the
                # calendar got stuck unable to navigate to any earlier date
                # (only a full page refresh, resetting to "All time",
                # widened it back). Only future dates should ever be
                # blocked; a fixed 2-year floor keeps navigation open
                # regardless of how narrow the currently-loaded data is.
                _widget_min_date = min(min_date, today_real - pd.Timedelta(days=730))
                st.session_state.setdefault("wa_date_range_custom", (min_date, _widget_max_date))
                _stored_custom = st.session_state["wa_date_range_custom"]
                if _stored_custom is None:
                    _clamped_custom = (min_date, _widget_max_date)
                elif isinstance(_stored_custom, (tuple, list)):
                    _clamped_custom = tuple(min(max(d, _widget_min_date), _widget_max_date) for d in _stored_custom)
                else:
                    _clamped_custom = min(max(_stored_custom, _widget_min_date), _widget_max_date)
                if _clamped_custom != _stored_custom:
                    st.session_state["wa_date_range_custom"] = _clamped_custom
                try:
                    date_range = st.date_input(
                        "Pick dates", min_value=_widget_min_date, max_value=_widget_max_date,
                        key="wa_date_range_custom",
                    )
                except st.errors.StreamlitAPIException:
                    # SAFETY NET (2026-07-17): if session_state still holds a
                    # value the widget rejects for any reason, reset it to a
                    # known-valid range instead of crashing the whole page.
                    st.session_state["wa_date_range_custom"] = (min_date, _widget_max_date)
                    date_range = (min_date, _widget_max_date)
        elif preset != "All time":
            days = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}[preset]
            window_end = min(max_date, today_real)  # never anchor past real "now"
            naive_start = window_end - pd.Timedelta(days=days)
            window_start = max(min_date, naive_start)
            date_range = (window_start, window_end)
            if max_date > today_real:
                st.caption(
                    f"ℹ️ Some loaded orders have a scheduled/target date after today "
                    f"({max_date.strftime('%d %b')}) — likely in-flight orders with a "
                    f"future pickup time. '{preset}' is anchored to today, not to those "
                    f"future dates, so it won't include them."
                )
            # BUG FIX (2026-07-12): this warning used to only check
            # preset == "Last 30 days", so picking "Last 7 days" on data
            # that only actually covers ~7 days showed the exact same
            # numbers as "Last 30 days" with NO explanation why they
            # matched — looked like the filter wasn't doing anything.
            # Now checks whatever preset is selected, and covers two
            # related cases:
            #   1. Genuinely clamped: the preset asked for more history
            #      than exists (e.g. "Last 30 days" but only 7 available).
            #   2. Exact coincidental match: the preset's window happens
            #      to line up exactly with all available data (e.g. "Last
            #      7 days" when there's exactly 7 days total) — still
            #      worth surfacing, because from the screen alone there's
            #      no way to tell "this is deliberately 7 days of a much
            #      longer history" from "this is literally all we have".
            if naive_start <= min_date:
                actual_span = (window_end - min_date).days + 1
                if naive_start < min_date:
                    st.caption(
                        f"⚠️ Data only goes back to {min_date.strftime('%d %b')} ({actual_span} day(s) "
                        f"available) — '{preset}' would need data back to {naive_start.strftime('%d %b')}. "
                        f"Showing everything available instead of a full {days}-day window."
                    )
                else:
                    st.caption(
                        f"ℹ️ This is exactly all the data currently available ({actual_span} day(s), "
                        f"since {min_date.strftime('%d %b')}) — a wider date range would show the same result."
                    )

    df = df_all.copy()
    if selected_areas:
        df = df[df["area"].isin(selected_areas)]
    if selected_merchants:
        df = df[df["merchant_name"].isin(selected_merchants)]

    # ---- Date range fix (2026-07-12) ---------------------------------------
    # Previously this sliced EVERY status (in-flight included) by
    # delivery_date. In-flight delivery_date is a scheduled/target time,
    # not when the order actually happened — so "Last 24 hours" could show
    # MORE orders than a specific past custom range simply because a batch
    # of in-flight orders happened to be scheduled for today, while "Last
    # 7 days" and "Last 30 days" could show identical totals because the
    # date filter was silently a no-op for that whole slice of the data.
    #
    # Fix (superseded 2026-07-17, see FINDING 6 FIX below): the date range
    # only ever filtered DATED_STATUSES (delivered/returned — real event
    # timestamps), leaving in-flight orders always included in full.

    date_filter_active = bool(date_range and len(date_range) == 2)
    if date_filter_active:
        start, end = date_range
        # FINDING 6 FIX (2026-07-17): the DATED_STATUSES carve-out above
        # meant Cancelled — and every other non-delivered/returned status —
        # never respected the selected date range at all, always showing in
        # full regardless of range. That carve-out made sense for
        # delivery_date (a SCHEDULED/target time that hasn't happened yet
        # for in-flight orders) but not for "created_at" (the field
        # filtered on since the Create Date default-filter fix above),
        # which is always a real past event for every shipment regardless
        # of its current status. The official dashboard's own
        # start_date/end_date filtering isn't status-specific either
        # (confirmed: no `status` param is sent). Filter now applies to
        # every row, all statuses alike, matching that behavior.
        df = df[(df["created_at"].dt.date >= start) & (df["created_at"].dt.date <= end)]

    if df.empty:
        st.info("No orders match the selected filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ---- KPI row ------------------------------------------------------
    kpis = summary_kpis(df)
    status_counts = status_breakdown(df)
    ma_for_breakdown = merchant_activity(df)
    watch_count = int((ma_for_breakdown["status"] == "Watch").sum()) if not ma_for_breakdown.empty else 0

    # "577 = 200 delivered + 200 returned + 177 in-flight" — the actual
    # composition behind the single Total orders number, built from
    # whatever statuses are really present (never hardcoded), so this
    # can't drift out of sync with the data.
    if not status_counts.empty:
        top_statuses = status_counts.head(4)
        orders_breakdown_txt = " + ".join(f"{int(r.orders):,} {r.status}" for r in top_statuses.itertuples())
        if len(status_counts) > 4:
            orders_breakdown_txt += f" + {len(status_counts) - 4} more"
    else:
        orders_breakdown_txt = ""

    # Same idea for the money: gross order value split by delivered vs
    # returned vs everything still in flight.
    if not df.empty and "status" in df.columns:
        delivered_val = df.loc[df["status"] == "delivered", "amount"].sum()
        returned_val = df.loc[df["status"] == "returned", "amount"].sum()
        other_val = df.loc[~df["status"].isin(["delivered", "returned"]), "amount"].sum()
        value_breakdown_txt = (
            f"{delivered_val:,.0f} delivered + {returned_val:,.0f} returned + {other_val:,.0f} in-flight (EGP)"
        )
    else:
        value_breakdown_txt = ""

    cards = [
        ("📦", "#EEEDFE", PURPLE, "Total orders", f"{kpis['total_orders']:,}", orders_breakdown_txt, "orders-breakdown"),
        ("🚴", "#E6F1FB", "#185FA5", "Active couriers", f"{kpis['active_couriers']}", "Click for full leaderboard + least-active", "courier-leaderboard"),
        ("🏬", "#FAEEDA", AMBER, "Active merchants", f"{kpis['active_merchants']}", "Click for top/least-active merchants", "merchant-tables"),
        ("⚠️", "#FCEBEB", RED, "Merchants at risk", f"{kpis['merchants_at_risk']}", f"{kpis['merchants_at_risk']} declining + {watch_count} watch", "merchant-health"),
    ]
    kpi_html = '<div class="wa-kpi-row">' + "".join(
        _kpi_card(icon, bg, color, label, value, "", "", breakdown, anchor)
        for icon, bg, color, label, value, breakdown, anchor in cards
    ) + "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    if kpis["merchants_at_risk"] > 0:
        st.markdown(
            f'<div class="wa-alert-banner">⚠️ {kpis["merchants_at_risk"]} merchant(s) showing a '
            f'significant drop in order volume this week — see Merchant Health below.</div>',
            unsafe_allow_html=True,
        )

    # ---- Orders breakdown by status (ADDED) --------------------------------
    st.markdown('<div class="wa-section" id="orders-breakdown">', unsafe_allow_html=True)
    st.markdown(
        '<p class="wa-section-title">📦 Orders breakdown by status</p>'
        '<p class="wa-section-sub">What the Total orders / Gross order value numbers above are actually made of</p>',
        unsafe_allow_html=True,
    )
    if not status_counts.empty:
        sb_col1, sb_col2 = st.columns([1, 1.4])
        with sb_col1:
            st.dataframe(
                status_counts.rename(columns={"status": "Status", "orders": "Orders", "pct": "% of total"}),
                use_container_width=True, hide_index=True, height=min(300, 45 + 35 * len(status_counts)),
                column_config={"% of total": st.column_config.NumberColumn(format="%.1f%%")},
            )
        with sb_col2:
            st.bar_chart(status_counts.set_index("status")[["orders"]], y="orders", color=PURPLE, height=280)
        st.caption(
            f"💰 Value split — delivered: {delivered_val:,.0f} EGP · returned: {returned_val:,.0f} EGP · "
            f"still in flight: {other_val:,.0f} EGP (all client order value, not Weevo's revenue — "
            f"see Financial Summary below for actual revenue)."
        )
    else:
        st.info("No status data available for the current filter.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Orders over time -------------------------------------------------
    st.markdown('<div class="wa-section">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="wa-section-title">Orders over time</p>'
        f'<p class="wa-section-sub">{granularity_label} order volume</p>',
        unsafe_allow_html=True,
    )
    ot = orders_over_time(df, granularity=granularity_map[granularity_label])
    st.bar_chart(ot.set_index("period")[["orders"]], y="orders", color=PURPLE, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Top areas + Courier leaderboard -----------------------------
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="wa-section">', unsafe_allow_html=True)
        st.markdown('<p class="wa-section-title">Top areas by order volume</p>', unsafe_allow_html=True)
        areas = top_areas(df, n=8)  # already sorted descending by the pipeline
        st.bar_chart(areas.set_index("area")[["orders"]], y="orders", color=TEAL, height=280)
        excluded = areas.attrs.get("excluded_unknown_count", 0)
        if excluded:
            st.caption(f"{excluded:,} order(s) excluded — address didn't match a known area.")
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if not openai_key:
                st.caption(
                    "Set OPENAI_API_KEY on the server to enable AI classification "
                    "for these addresses."
                )
            elif st.button("🤖 Classify unknown areas with AI", key="classify_areas_btn"):
                with st.spinner(f"Classifying up to {excluded:,} unmatched address(es) — "
                                 "uses OpenAI (gpt-4o-mini), each address is only ever sent once "
                                 "and remembered permanently after that…"):
                    _, class_stats = classify_unknown_addresses(df_all, openai_key)
                st.session_state["_last_area_classification"] = class_stats
                st.rerun()
            last_run = st.session_state.get("_last_area_classification")
            if last_run:
                st.caption(
                    f"Last AI classification run: {last_run['unique_addresses_sent']} unique "
                    f"address(es) sent, {last_run['newly_classified']} newly classified, "
                    f"{last_run['unknown_after']} still unmatched."
                )
                if last_run["errors"]:
                    with st.expander("AI classification errors"):
                        for err in last_run["errors"]:
                            st.write(f"❌ {err}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="wa-section" id="courier-leaderboard">', unsafe_allow_html=True)
        st.markdown('<p class="wa-section-title">Courier leaderboard</p>', unsafe_allow_html=True)
        couriers = courier_leaderboard(df, n=8)  # already sorted descending by the pipeline
        st.bar_chart(couriers.set_index("courier_name")[["orders"]], y="orders", color=PURPLE, height=280)
        excluded = couriers.attrs.get("excluded_unassigned_count", 0)
        if excluded:
            st.caption(f"{excluded:,} order(s) excluded — no courier assigned yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Least-busy areas + Least-active couriers (ADDED) -----------------
    least_col1, least_col2 = st.columns(2)
    with least_col1:
        least_areas = top_areas(df, n=8, ascending=True)
        if not least_areas.empty:
            st.markdown('<div class="wa-section">', unsafe_allow_html=True)
            st.markdown(
                '<p class="wa-section-title">Least-busy areas</p>'
                '<p class="wa-section-sub">Real areas only (excludes the "didn\'t match a known area" count above)</p>',
                unsafe_allow_html=True,
            )
            st.bar_chart(least_areas.set_index("area")[["orders"]], y="orders", color=CORAL, height=280)
            st.markdown("</div>", unsafe_allow_html=True)

    with least_col2:
        least_couriers = courier_leaderboard(df, n=8, ascending=True)
        if not least_couriers.empty:
            st.markdown('<div class="wa-section">', unsafe_allow_html=True)
            st.markdown(
                '<p class="wa-section-title">Least-active couriers</p>'
                '<p class="wa-section-sub">Candidates for more orders routed to them, or worth checking in on. '
                'For couriers sitting on overdue orders specifically, see "Overdue orders by courier" in '
                'Financial Summary & Risk below.</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                least_couriers.rename(columns={"courier_name": "Courier", "orders": "Orders", "total_value": "Order value (EGP) (amount)"}),
                use_container_width=True, hide_index=True, height=280,
                column_config={"Order value (EGP) (amount)": st.column_config.NumberColumn(format="%.0f")},
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Courier volume above is order count and total handled value only. "
        "Real delivery-time and overdue tracking is in the Financial Summary & "
        "Risk section below, which uses actual pickup/completion timestamps."
    )

    # ---- All couriers, full ranking (ADDED 2026-07-19) ---------------------
    all_couriers = courier_leaderboard(df, n=df["courier_name"].nunique() if "courier_name" in df.columns else 0)
    if not all_couriers.empty:
        all_couriers_col, all_couriers_chart_col = st.columns(2)
        with all_couriers_col:
            st.markdown('<div class="wa-section">', unsafe_allow_html=True)
            st.markdown(
                '<p class="wa-section-title">All couriers</p>'
                '<p class="wa-section-sub">Every courier with at least one order in this window, most to least active</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                all_couriers.rename(columns={"courier_name": "Courier", "orders": "Orders", "total_value": "Order value (EGP) (amount)"}),
                use_container_width=True, hide_index=True, height=280,
                column_config={"Order value (EGP) (amount)": st.column_config.NumberColumn(format="%.0f")},
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with all_couriers_chart_col:
            st.markdown('<div class="wa-section">', unsafe_allow_html=True)
            st.markdown(
                '<p class="wa-section-title">Orders by courier</p>'
                '<p class="wa-section-sub">Same ranking as the table, visualized</p>',
                unsafe_allow_html=True,
            )
            st.bar_chart(all_couriers.set_index("courier_name")[["orders"]], y="orders", color=PURPLE, height=280)
            st.markdown("</div>", unsafe_allow_html=True)

    # NOTE: the old "2-hour promise tracking" section that used to live here
    # has been removed (2026-07-10, per Ahmed's direction). It measured the
    # gap between date_to_receive_shipment and date_to_deliver_shipment —
    # both are TARGET/planned times from the live API, never the actual
    # pickup or delivery moment, so the numbers it showed (e.g. "166 hours
    # average") were not measuring anything real. The Risk section below
    # (part of Financial Summary & Risk) replaces this correctly: it uses
    # the actual pickup timestamp from each shipment's logs plus the real
    # delivered_at / returned_at fields, and flags shipments that are
    # overdue against their target — which is the metric Ahmed actually
    # wants tracked.

    # ---- Top merchants + Recent orders -----------------------------------
    col_a, col_b = st.columns([1, 1.3])
    with col_a:
        st.markdown('<div class="wa-section" id="merchant-tables">', unsafe_allow_html=True)
        st.markdown('<p class="wa-section-title">Top merchants by orders</p>', unsafe_allow_html=True)
        top_merch = merchant_leaderboard(df, n=8)
        st.dataframe(
            top_merch.rename(columns={"merchant_name": "Merchant", "orders": "Orders", "total_value": "Order value (EGP) (amount)"}),
            use_container_width=True, hide_index=True, height=300,
            column_config={"Order value (EGP) (amount)": st.column_config.NumberColumn(format="%.0f")},
        )
        excluded = top_merch.attrs.get("excluded_unknown_count", 0)
        if excluded:
            st.caption(f"{excluded:,} order(s) excluded — no merchant name on record.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="wa-section">', unsafe_allow_html=True)
        st.markdown('<p class="wa-section-title">Recent orders</p>', unsafe_allow_html=True)
        ro = recent_orders(df, n=10)
        st.dataframe(
            ro.rename(columns={
                "shipment_id": "Order ID", "merchant_name": "Merchant", "courier_name": "Courier",
                "area": "Area", "amount": "Amount (EGP)", "delivery_date": "Date",
            }),
            use_container_width=True, hide_index=True, height=300,
            column_config={
                "Amount (EGP)": st.column_config.NumberColumn(format="%.0f"),
                "Date": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Least-active merchants + true zero-order merchants (ADDED) -------
    col_c, col_d = st.columns(2)
    with col_c:
        least_merch = merchant_leaderboard(df, n=8, ascending=True)
        if not least_merch.empty:
            st.markdown('<div class="wa-section">', unsafe_allow_html=True)
            st.markdown(
                '<p class="wa-section-title">Least-active merchants</p>'
                '<p class="wa-section-sub">Among merchants who placed at least one order in this window</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                least_merch.rename(columns={"merchant_name": "Merchant", "orders": "Orders", "total_value": "Order value (EGP) (amount)"}),
                use_container_width=True, hide_index=True, height=280,
                column_config={"Order value (EGP) (amount)": st.column_config.NumberColumn(format="%.0f")},
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        all_merch = merchant_leaderboard(df, n=df["merchant_name"].nunique() if "merchant_name" in df.columns else 0)
        if not all_merch.empty:
            st.markdown('<div class="wa-section">', unsafe_allow_html=True)
            st.markdown(
                '<p class="wa-section-title">All active merchants</p>'
                '<p class="wa-section-sub">Every merchant with at least one order in this window, most to least active '
                '— registered merchants who never placed an order are not included</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                all_merch.rename(columns={"merchant_name": "Merchant", "orders": "Orders", "total_value": "Order value (EGP) (amount)"}),
                use_container_width=True, hide_index=True, height=280,
                column_config={"Order value (EGP) (amount)": st.column_config.NumberColumn(format="%.0f")},
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- Merchant health --------------------------------------------------
    st.markdown('<div class="wa-section" id="merchant-health">', unsafe_allow_html=True)
    st.markdown(
        '<p class="wa-section-title">Merchant health — early warning</p>'
        '<p class="wa-section-sub">Compares last 7 days of orders vs. the 7 days before that</p>',
        unsafe_allow_html=True,
    )
    ma = merchant_activity(df, recent_days=7, compare_days=7)

    rows_html = ""
    for _, row in ma.iterrows():
        text_color, bg_color, dot = STATUS_STYLE.get(row["status"], ("#444", "#eee", "⚪"))
        change_color = TEAL if row["change_pct"] >= 0 else RED
        rows_html += _clean(f"""
        <tr style="border-bottom:1px solid {BORDER};">
            <td style="padding:8px 6px; font-size:13px;">{row['merchant_name']}</td>
            <td style="padding:8px 6px; font-size:13px; text-align:right;">{row['recent_orders']}</td>
            <td style="padding:8px 6px; font-size:13px; text-align:right; color:{MUTED};">{row['previous_orders']}</td>
            <td style="padding:8px 6px; font-size:13px; text-align:right; color:{change_color}; font-weight:600;">{row['change_pct']:+.1f}%</td>
            <td style="padding:8px 6px; text-align:right;">
                <span class="wa-badge" style="color:{text_color}; background:{bg_color};">{dot} {row['status']}</span>
            </td>
        </tr>""")

    table_html = _clean(f"""
    <table style="width:100%; border-collapse:collapse;">
        <thead>
            <tr style="border-bottom:2px solid {BORDER};">
                <th style="text-align:left; padding:6px; font-size:12px; color:{MUTED};">Merchant</th>
                <th style="text-align:right; padding:6px; font-size:12px; color:{MUTED};">Last 7 days</th>
                <th style="text-align:right; padding:6px; font-size:12px; color:{MUTED};">Previous 7 days</th>
                <th style="text-align:right; padding:6px; font-size:12px; color:{MUTED};">Change</th>
                <th style="text-align:right; padding:6px; font-size:12px; color:{MUTED};">Status</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """)
    st.markdown(table_html, unsafe_allow_html=True)
    ma_excluded = ma.attrs.get("excluded_unknown_count", 0)
    if ma_excluded:
        st.caption(f"{ma_excluded:,} order(s) excluded from this comparison — no merchant name on record.")
    st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # NEW SECTION — Financial Summary, real pickup-to-completion delivery
    # time, and Risk/overdue alerts. Fully separate from everything above:
    # sourced from analytics_pipeline_v2 (direct API + pagination + the
    # confirmed financial formula), not from the v1 tables. If this section
    # has a problem, it does not affect anything above it.
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="wa-section" id="financial-detail">', unsafe_allow_html=True)
    st.markdown(
        '<p class="wa-section-title">💰 Financial Summary & Risk</p>'
        '<p class="wa-section-sub">Real revenue, actual pickup-to-completion time, and overdue '
        'shipments — delivered + returned focus, in-flight statuses checked separately for risk. '
        'Follows the same Live API / Uploaded Archive source picked in the sidebar, same as the '
        'main section above — this used to always pull live regardless of that choice; fixed so '
        'both sections behave consistently.</p>',
        unsafe_allow_html=True,
    )

    v2_from_archive = source_label == "Uploaded Archive"
    if not v2_from_archive and not api_key:
        st.info(
            "Enter the Integration key in the sidebar (Live API source) to see financial and "
            "risk data, or switch the source above to 'Uploaded Archive' if a Financial "
            "snapshot was saved before."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if v2_from_archive:
            try:
                df_v2 = load_v2_archive(v2_archive_path)
                v2_source_note = f"Uploaded Archive — `{v2_archive_path}`"
            except Exception as e:
                st.error(f"Couldn't load the Financial/v2 archive: {e}")
                df_v2 = pd.DataFrame()
                v2_source_note = "Uploaded Archive (load failed)"
        else:
            try:
                df_v2 = _cached_v2_load(api_key, start_date=admin_start_str, end_date=admin_end_str)
                if not df_v2.empty:
                    df_v2 = enrich_areas_with_cache(df_v2)
                v2_source_note = "Live API"
            except Exception as e:
                st.error(f"Couldn't load financial/risk data: {e}")
                df_v2 = pd.DataFrame()
                v2_source_note = "Live API (load failed)"
        st.caption(f"📡 Financial/Risk data source: {v2_source_note}")

        # Apply the SAME Area / Merchant filters selected above (fcol1/fcol2).
        # Previously this section always showed the full, unfiltered dataset
        # regardless of what was picked there — this is the fix for that.
        # df_v2 uses the same "area" / "merchant_name" columns as df (both
        # come from the same detect_area()/merchant.name logic), so this is
        # a direct, safe filter — not a re-implementation.
        if not df_v2.empty:
            if selected_areas:
                df_v2 = df_v2[df_v2["area"].isin(selected_areas)]
            if selected_merchants:
                df_v2 = df_v2[df_v2["merchant_name"].isin(selected_merchants)]

            # BUG FIX (2026-07-12): same category of issue as the main date
            # filter above. This used to slice ALL of df_v2 by created_at —
            # including in-flight/overdue rows feeding the Risk section
            # further down (at_risk_shipments, risk_by_courier,
            # overdue_age_buckets, overdue_by_area). "How many shipments
            # are overdue RIGHT NOW" is a live/current question — an order
            # created 5 days ago that's still overdue today doesn't stop
            # being overdue just because "Last 24 hours" is selected up
            # top. Filtering it out there made the whole Risk section
            # silently under-report (or empty out) whenever any date
            # preset other than "All time" was active.
            # Fix: date range now only ever slices PRIMARY_STATUSES
            # (delivered/returned — real revenue events with a genuine
            # created_at). Risk/overdue rows are always included in full,
            # same as the in-flight fix above.
            v2_is_primary = df_v2["status"].isin(PRIMARY_STATUSES) if "status" in df_v2.columns else pd.Series(True, index=df_v2.index)
            df_v2_dated = df_v2[v2_is_primary]
            df_v2_risk = df_v2[~v2_is_primary]

            v2_full_dated_dates = df_v2_dated["created_at"].dropna() if "created_at" in df_v2_dated.columns else pd.Series(dtype="datetime64[ns]")
            v2_min_date = v2_full_dated_dates.min().date() if not v2_full_dated_dates.empty else None
            v2_max_date = v2_full_dated_dates.max().date() if not v2_full_dated_dates.empty else None

            # FINDING 3 FIX (2026-07-17): removed the created_at re-filter that
            # used to run here. df_v2_dated already reflects the server-side
            # date filter from the shared fetch (same as the main section —
            # see _cached_admin_shipments), so re-filtering by created_at was
            # applying a second, different date field on top of that, which is
            # exactly why this section's counts didn't match the main section
            # for the identical picked range. v2_date_filter_active is still
            # computed — used below only for informational captions.
            v2_date_filter_active = bool(date_range and len(date_range) == 2 and "created_at" in df_v2_dated.columns)

            df_v2 = pd.concat([df_v2_dated, df_v2_risk], ignore_index=True) if not df_v2_risk.empty else df_v2_dated.copy()

        v2_diag = getattr(df_v2, "attrs", {}).get("fetch_diagnostics")
        if v2_diag and v2_diag.get("failed_statuses"):
            failed = v2_diag["failed_statuses"]
            succeeded = v2_diag["succeeded_statuses"]
            st.markdown(
                f'<div class="wa-mock-banner">⚠️ Partial data — {len(failed)} of '
                f'{v2_diag["total_statuses"]} statuses failed to load '
                f'({", ".join(failed.keys())}). Financial/risk numbers below may be '
                f'incomplete for those statuses.</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Which statuses loaded, and why the rest failed"):
                st.write(f"✅ Loaded: {', '.join(succeeded) if succeeded else '(none)'}")
                for status, err in failed.items():
                    st.write(f"❌ **{status}**: {err}")

        if df_v2.empty:
            if selected_areas or selected_merchants:
                st.info("No delivered/returned shipments match the selected Area/Merchant filter.")
            elif v2_from_archive:
                st.info(
                    "No Financial/v2 archive saved yet. Switch to 'Live API' above and click "
                    "'Save this snapshot to archive (v1 + Financial/v2)' to start building it."
                )
            else:
                st.warning("No shipment data returned yet.")
        else:
            if "created_at" in df_v2_dated.columns and df_v2_dated["created_at"].notna().any():
                pulled_from = "the API" if not v2_from_archive else "the saved archive"
                # FIX (2026-07-19): this used to describe an older fetch that
                # pulled a fixed 200 most-recent orders per status regardless
                # of date (pre-2026-07-15 admin-API migration). It now shares
                # the same date-filtered fetch as the main section above, so
                # it DOES respect the selected calendar range — the wording
                # was leftover from that old behavior and no longer matched
                # reality.
                st.caption(
                    f"📅 Covers orders created {df_v2_dated['created_at'].min():%d %b} → "
                    f"{df_v2_dated['created_at'].max():%d %b} — every delivered + returned order "
                    f"in the selected date range, pulled from {pulled_from}."
                )
            # Clamp warning — same idea as the main section above: tell the
            # user explicitly when the selected preset asked for more
            # history than what's actually loaded, instead of silently
            # showing a smaller range with no explanation (which is what
            # made "Last 7 days" == "Last 30 days" look like a bug earlier).
            if v2_date_filter_active and v2_min_date and v2_max_date:
                naive_start = date_range[0]
                if naive_start < v2_min_date:
                    actual_span = (v2_max_date - v2_min_date).days + 1
                    st.warning(
                        f"⚠️ Financial data only goes back to {v2_min_date.strftime('%d %b')} "
                        f"({actual_span} day(s) available) — the selected range would need data "
                        f"back to {naive_start.strftime('%d %b')}. Showing everything available instead."
                    )
                elif naive_start == v2_min_date:
                    actual_span = (v2_max_date - v2_min_date).days + 1
                    st.caption(
                        f"ℹ️ This is exactly all the delivered/returned data currently loaded "
                        f"({actual_span} day(s)) — a wider range would show the same numbers."
                    )
            if v2_date_filter_active and not df_v2_risk.empty:
                st.caption(
                    f"ℹ️ {len(df_v2_risk):,} in-flight/overdue shipment(s) below are always included "
                    "regardless of the date range — risk is a live snapshot, not a historical count."
                )

            # --- Revenue ---------------------------------------------------
            rev = revenue_summary(df_v2)
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            rc1.metric("Weevo revenue (delivered + returned)", f"{rev['total_weevo_revenue']:,.0f} EGP")
            rc2.metric("Delivered orders", f"{rev['delivered_count']:,}")
            rc3.metric("Returned orders", f"{rev['returned_count']:,}")
            rc4.metric("Delivered rate", f"{rev['delivered_rate_pct']}%")
            rc5.metric("Return rate", f"{rev['return_rate_pct']}%")
            st.caption(
                "Revenue = agreed shipping cost + transfer fee (1% of COD amount, 0 for online "
                "payments) — Weevo's actual take per shipment, not the customer's order value."
            )

            # --- Deeper financial breakdown (ADDED) -------------------------
            fin = financial_breakdown(df_v2)
            d, r = fin["delivered"], fin["returned"]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="wa-section-title" style="font-size:14px;">Revenue breakdown — delivered vs. returned</p>', unsafe_allow_html=True)
            fin_col1, fin_col2 = st.columns(2)
            with fin_col1:
                st.markdown(f"**✅ Delivered ({d['count']:,} orders)**")
                st.write(f"- Shipping revenue *(agreed_shipping_cost)*: **{d['shipping_revenue']:,.0f} EGP**")
                st.write(f"- Transfer fee revenue (1% COD) *(transfer_fee)*: **{d['transfer_fee_revenue']:,.0f} EGP**")
                st.write(f"- Total Weevo revenue *(agreed_shipping_cost + transfer_fee)*: **{d['total_weevo_revenue']:,.0f} EGP**")
                st.write(f"- Avg. revenue per order *(weevo_revenue)*: **{d['avg_weevo_revenue_per_order']:,.2f} EGP**")
                st.write(f"- Total order value *(amount)*: **{d['total_order_value']:,.0f} EGP**")
                st.write(f"- Payment split *(payment_method)*: {d['cod_count']:,} COD · {d['online_count']:,} online")
                if d["cod_count"]:
                    st.write(f"- Avg. COD order value *(amount)*: **{d['avg_client_order_value_cod']:,.0f} EGP**")
                    st.write(f"- Total handed back to merchants (COD) *(merchant_payout)*: **{d['total_merchant_payout_cod']:,.0f} EGP**")
                if d["online_count"]:
                    st.write(f"- Avg. online order value *(amount)*: **{d['avg_client_order_value_online']:,.0f} EGP**")
            with fin_col2:
                st.markdown(f"**↩️ Returned ({r['count']:,} orders)**")
                st.write(f"- Shipping revenue *(agreed_shipping_cost)*: **{r['shipping_revenue']:,.0f} EGP**")
                st.write(f"- Transfer fee revenue (1% COD) *(transfer_fee)*: **{r['transfer_fee_revenue']:,.0f} EGP**")
                st.write(f"- Total Weevo revenue *(agreed_shipping_cost + transfer_fee)*: **{r['total_weevo_revenue']:,.0f} EGP**")
                st.write(f"- Avg. revenue per order *(weevo_revenue)*: **{r['avg_weevo_revenue_per_order']:,.2f} EGP**")
                st.write(f"- Total order value *(amount)*: **{r['total_order_value']:,.0f} EGP**")
                st.write(f"- Payment split *(payment_method)*: {r['cod_count']:,} COD · {r['online_count']:,} online")
                if r["cod_count"]:
                    st.write(f"- Avg. COD order value *(amount)*: **{r['avg_client_order_value_cod']:,.0f} EGP**")
                if r["online_count"]:
                    st.write(f"- Avg. online order value *(amount)*: **{r['avg_client_order_value_online']:,.0f} EGP**")

            if fin["returned_revenue_vs_order_value_pct"] is not None:
                pct = fin["returned_revenue_vs_order_value_pct"]
                if pct >= 100:
                    insight = (
                        f"On average, a returned COD order earns Weevo **{pct:.0f}%** of what the "
                        f"order itself was worth in shipping alone — meaning Weevo often makes MORE "
                        f"from a failed delivery's shipping fee than the order's own value. Worth "
                        f"flagging to merchants with high return rates."
                    )
                else:
                    insight = (
                        f"On average, a returned COD order's shipping revenue is **{pct:.0f}%** of "
                        f"the order's own value — shipping cost is recovered on returns, but it's "
                        f"not pure profit relative to what the order was worth."
                    )
                st.info(f"💡 {insight}")

            # --- Delivery time (pickup -> completion) -----------------------
            st.markdown("<br>", unsafe_allow_html=True)
            dts = v2_delivery_time_summary(df_v2)
            if dts["sample_size"] > 0:
                dt1, dt2, dt3 = st.columns(3)
                dt1.metric("Avg. pickup → completion", f"{dts['avg_hours']} hrs")
                dt2.metric("Median pickup → completion", f"{dts['median_hours']} hrs")
                dt3.metric("Based on", f"{dts['sample_size']:,} shipments")
                st.caption(
                    "Measured from the actual pickup timestamp (captain scan) to delivered/returned "
                    "— not from order creation time."
                )
            else:
                st.caption("No shipments with both a recorded pickup and completion time yet.")

            # --- Risk / overdue ---------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            risky = at_risk_shipments(df_v2)
            if not risky.empty:
                st.markdown(
                    f'<div class="wa-alert-banner">⚠️ {len(risky)} shipment(s) are past their '
                    f'target delivery time and still not delivered or returned.</div>',
                    unsafe_allow_html=True,
                )
                rbc = risk_by_courier(df_v2)
                risk_col1, risk_col2 = st.columns([1, 1.4])
                with risk_col1:
                    st.markdown('<p class="wa-section-title" style="font-size:13px;">Overdue orders by courier</p>', unsafe_allow_html=True)
                    st.dataframe(
                        rbc.rename(columns={"courier_name": "Courier", "overdue_orders": "Overdue orders",
                                             "max_hours_overdue": "Most overdue (hrs)"}),
                        use_container_width=True, hide_index=True, height=250,
                    )
                with risk_col2:
                    st.markdown('<p class="wa-section-title" style="font-size:13px;">Overdue shipments</p>', unsafe_allow_html=True)
                    st.dataframe(
                        risky.rename(columns={
                            "shipment_id": "Shipment", "courier_name": "Courier", "merchant_name": "Merchant",
                            "status": "Status", "target_deliver_at": "Was due", "hours_overdue": "Hours overdue",
                        }),
                        use_container_width=True, hide_index=True, height=250,
                    )

                # --- How overdue, and where (ADDED) --------------------------
                st.markdown("<br>", unsafe_allow_html=True)
                age_col, area_col = st.columns(2)
                with age_col:
                    st.markdown(
                        '<p class="wa-section-title" style="font-size:13px;">How overdue — age breakdown</p>'
                        '<p class="wa-section-sub">A count of "overdue" means something very different if most '
                        'are 1 hour late vs. 3 days late</p>',
                        unsafe_allow_html=True,
                    )
                    buckets = overdue_age_buckets(df_v2)
                    if not buckets.empty:
                        st.bar_chart(buckets.set_index("bucket")[["orders"]], y="orders", color=RED, height=220)
                with area_col:
                    st.markdown('<p class="wa-section-title" style="font-size:13px;">Overdue by area</p>', unsafe_allow_html=True)
                    by_area = overdue_by_area(df_v2, n=8)
                    if not by_area.empty:
                        st.dataframe(
                            by_area.rename(columns={"area": "Area", "overdue_orders": "Overdue orders"}),
                            use_container_width=True, hide_index=True, height=220,
                        )
                        st.caption("A single area overdue across many different couriers usually points to a "
                                   "traffic/distance/access issue there rather than individual courier performance.")
                    else:
                        st.caption("Area not available on these overdue shipments.")
            else:
                st.success("✅ No overdue shipments right now — everything in flight is still within its target window.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f'<p style="font-size:11px; color:{MUTED}; text-align:center; margin-top:8px;">'
        f'Metrics are based on shipment data from the selected source above. '
        f'Delivery-time is included when using the Live API. Rating and live GPS-tracking '
        f'metrics will be added once that data is available.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# INTEGRATION NOTES — copy-paste additions for dashboard.py (do not edit
# any existing line, only add):
#
# 1. Near the top, with the other imports:
#      from streamlit_ui.analytics_page import render_analytics_page
#
# 2. In the sidebar selectbox options list, add one more string:
#      [..., "🔧 Admin Tools", "📈 Data Analytics"]
#
# 3. After the final existing elif block (the one for "🔧 Admin Tools"),
#    add:
#      elif page == "📈 Data Analytics":
#          render_analytics_page()
#
# No new pip packages required — this file only uses st.bar_chart /
# st.dataframe / st.markdown, all bundled with streamlit==1.29.0
# already pinned in requirements.txt.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Analytics", layout="wide")
    render_analytics_page()
