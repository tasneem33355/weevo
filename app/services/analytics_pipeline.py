
"""
Analytics data pipeline for the Weevo Interactive Analytics Dashboard.

Data source strategy
---------------------
Three modes, controlled by `source`:

1. "api" (REAL, PRIMARY) — calls the official Weevo ADMIN Dashboard
   backend directly (the same backend that powers the PHP Admin
   Dashboard — the single source of truth, migrated 2026-07 away from
   the older AI-agent endpoint):
   GET https://eg.api.weevoapp.com/api/v1/admin-5678vna9k6/shipments
   Auth (2026-07-16 correction): login-using-token turned out to be a
   REFRESH endpoint for an already-authenticated browser session, not a
   bootstrap login — confirmed 401 when tried with the old WEEVO_API_KEY
   integration key, since it expects an existing valid JWT instead. The
   real bootstrap is POST .../admin-5678vna9k6/login with an email+password
   body,
   confirmed against a real captured browser login, returning a
   short-lived `access_token` + `expires_at`, cached in-process and sent
   as `Authorization: Bearer <access_token>` on every subsequent Admin
   Dashboard request. The token is refreshed only when it has actually
   expired (checked against `expires_at`, with a small safety skew) or
   the backend itself returns HTTP 401 — never on every request.
   Query params: country_id (fixed at 1), start_delivery_date,
   end_delivery_date, page, paginate, in_batch (fixed at 0). No `status`
   filter is sent — verified against a real captured response that
   omitting `status` returns every status combined in one paginated
   walk (pagination.total matches statusCounts.Total exactly), which is
   both correct and the minimum number of requests. Date filtering is
   done server-side by the backend itself, so the numbers shown match
   the official Admin Dashboard for the same date range — this is the
   whole point of the migration.

   Confirmed real fields per shipment (from a captured live response):
   id, reference, status, client_name, client_phone, payment_method,
   amount, agreed_shipping_cost, transfer_fee, attempts,
   date_to_receive_shipment, date_to_deliver_shipment, created_at,
   delivered_date_at, returned_date_at, delivering_street,
   delivering_building_number, notes, handover_code_merchant_to_courier,
   handover_code_courier_to_customer, handover_code_courier_to_merchant,
   and nested merchant/courier objects (id, name, brand_name, phone).
   There is NO rating field, NO product_name field, and NO `logs` array
   (so no real actual-pickup-scan timestamp) anywhere in this payload —
   any metric that would need those is intentionally left out or left
   null below rather than approximated (confirmed with the business
   2026-07-15 — do not invent a substitute for a missing field).

2. "db" (REAL, SECONDARY / fallback) — reads the local
   `scheduler_shipment_data` table inside weevo_chatbot.db. This is only
   a rolling cache the scheduler has touched since Tia went live, not
   the full order history — kept here as a fallback path only, not the
   default, since it won't have data until the deployed scheduler has
   been running a while.

3. "mock" (DEMO) — generates a structurally identical DataFrame so the
   dashboard can be built/demoed before real credentials are wired in.

Everything downstream (aggregation functions) works identically across
all three, because they all produce the same DataFrame shape.

SECURITY NOTE: real credentials are never hardcoded here. WEEVO_ADMIN_EMAIL
and WEEVO_ADMIN_PASSWORD are read from the environment and used ONLY to
obtain a Bearer access token from the bootstrap login — they are not sent
on shipment/merchant/courier requests themselves, and never appear in any
exception message this module raises. Do not commit real credentials into
this file. The old WEEVO_API_KEY integration key is no longer used for
authentication at all (see the 2026-07-16 correction above). The
resulting Bearer access tokens are cached in memory only (never written
to disk) and are discarded when the process exits.
"""

from __future__ import annotations

import os
import sqlite3
import random
import threading
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import httpx  # matches app/services/external_api.py — requests is not in requirements.txt

WEEVO_API_BASE_URL = os.getenv("WEEVO_API_BASE_URL", "https://eg.api.weevoapp.com")

# Admin Dashboard endpoint (2026-07 migration — see module docstring).
# One combined pull covers every status in a date range, so there is no
# more "WEEVO_STATUS_OPTIONS" per-status loop to configure.
ADMIN_SHIPMENTS_PATH = "/api/v1/admin-5678vna9k6/shipments"
ADMIN_PAGE_SIZE = 100          # matches `paginate` in the confirmed example request
ADMIN_DEFAULT_COUNTRY_ID = 1   # confirmed always 1
ADMIN_DEFAULT_IN_BATCH = 0     # confirmed default, same as the official dashboard

# Bearer-token auth (2026-07-16 correction): login-using-token turned out to
# be a REFRESH endpoint for an already-authenticated browser session, not a
# bootstrap login — it expects an existing valid JWT, not the old 26-char
# WEEVO_API_KEY integration key, which is why that path 401'd. The real
# bootstrap is a standard email+password login, confirmed against a real
# captured browser request. WEEVO_ADMIN_EMAIL / WEEVO_ADMIN_PASSWORD (never
# WEEVO_API_KEY) now log in once via ADMIN_EMAIL_LOGIN_PATH to obtain a
# short-lived access token. That token is what goes on every Admin
# Dashboard request afterwards, as `Authorization: Bearer <access_token>`.
# Path confirmed 2026-07-16 against the real captured Request URL:
# https://eg.api.weevoapp.com/api/v1/admin-5678vna9k6/login — same
# admin-5678vna9k6 scope as every other endpoint in this module.
ADMIN_EMAIL_LOGIN_PATH = "/api/v1/admin-5678vna9k6/login"
TOKEN_EXPIRY_SKEW_V1 = 30      # seconds of safety margin subtracted from the
                               # server's expires_at, so a token never gets
                               # used right up to the wire and rejected
                               # mid-request by clock drift.
_ADMIN_TOKEN_CACHE: dict = {}       # keyed by (base_url, api_key) -> {"access_token", "expires_at"}
_ADMIN_TOKEN_CACHE_LOCK = threading.Lock()

# `HEAVY_STATUSES_V1` is still used by streamlit_ui/analytics_page.py (the
# "data coverage" banner) to distinguish statuses with a real dated event
# from in-flight ones — kept exactly as-is, it's a calculation detail, not
# a fetch detail.
HEAVY_STATUSES_V1 = {"delivered", "returned"}
GATEWAY_ERROR_CODES_V1 = {502, 503, 504}
REQUEST_TIMEOUT_V1 = 20          # seconds — a 502/504 is the gateway giving
                                  # up on ITS side, so a longer client-side
                                  # timeout doesn't help; kept modest instead.
MAX_RETRIES_V1 = 1               # 1 retry (2 attempts total) per request
RETRY_PAUSE_V1 = 1.5             # seconds to wait before a retry
INTER_REQUEST_DELAY_V1 = 0.35    # seconds between every individual request,
                                  # sequential only — no concurrency — so
                                  # requests never pile up on their backend
FETCH_TIME_BUDGET_V1 = 90        # seconds — hard ceiling on total wall-clock
                                  # time for a full fetch_admin_shipments()
                                  # call. Once exceeded, remaining pages are
                                  # marked as skipped rather than attempted, so
                                  # a fully-down backend fails predictably in
                                  # ~90s instead of stalling the page for minutes.

# Canonical column set shared by every source (api / db / mock) — used by
# the archive functions below so a saved/uploaded snapshot round-trips
# perfectly regardless of which source it originally came from.
ARCHIVE_COLUMNS = [
    "phone_number", "shipment_id", "product_name", "merchant_name",
    "courier_name", "courier_phone", "delivery_address", "delivery_city",
    "delivery_state", "amount", "delivery_date", "stored_at", "status",
    "attempts", "area", "received_at", "delivered_at", "delivery_hours",
]
DEFAULT_ARCHIVE_PATH = "./data/analytics_archive.csv"
AREA_CACHE_PATH = "./data/area_classification_cache.csv"

# ---------------------------------------------------------------------------
# Area detection — this is the SAME full keyword list as
# streamlit_ui/dashboard.py's AREA_KEYWORDS (Dispatch Planner), copied here
# directly instead of imported.
#
# WHY NOT IMPORT IT: `from streamlit_ui.dashboard import AREA_KEYWORDS` looks
# harmless but actually executes the ENTIRE dashboard.py module (Python has
# no way to import just one name without running the whole file first) —
# including Tia's LangChain/OpenAI initialization and every other module-
# level side effect in that file. If ANY of that fails for any reason
# (a missing env var at that moment, a slow DB connection, etc.), the import
# silently fails and this used to fall back to a 5-area stub — which is
# almost certainly why areas were missing even though the real 20+ area
# list already exists in dashboard.py. Copying the dict directly here
# removes that entire fragile dependency chain.
#
# If Weevo's real coverage areas expand or the keywords in dashboard.py's
# AREA_KEYWORDS get updated, copy the updated dict here too so the two
# stay in sync.
# ---------------------------------------------------------------------------
AREA_KEYWORDS = {
    "New Cairo": [
        "التجمع", "التجمع الاول", "التجمع الأول", "التجمع التالت", "التجمع الثالث",
        "التجمع الخامس", "القاهرة الجديدة", "new cairo",
        "tagamo3", "tagamoa", "tagamo3 el khames","tagmo3", "tagmo3 el awal", "elbanfseg", "el banafseg", "el tagamo3 el khames","uptown cairo", "stone residence", "stone residence compound", "moon valley", "arabella", "غرب ارابيلا", "غرب أرابيلا", "concord gardens", "مدينة ابو غزالة", "مدينة أبو غزالة", "مدينة الفرسان", "akoya",
        "fifth settlement", "5th settlement", "first settlement", "third settlement",
        "التسعين", "شارع التسعين", "north 90", "south 90", "90 avenue",
        "اللوتس", "lotus", "النرجس", "narges", "البنفسج", "banafseg",
        "الياسمين", "yasmeen", "القرنفل", "kornfol", "جنوب الاكاديمية", "south academy",
        "بيت الوطن", "beit el watan", "الدبلوماسيين", "الشويفات", "choueifat",
        "cairo festival city", "festival city", "cfc",
        "الرحاب", "rehab", "el rehab", "al rehab", "rehab city",
        "مدينتي", "madinaty",
        "الشروق", "shorouk", "sherouk", "el shorouk",
        "المستقبل", "future city", "mostakbal",
        "hydepark", "hyde park", "compound hydepark",
        "mountain view", "ماونتن فيو",
        "park view", "jayd compound", "taj city",
        "one kattamya", "katameya", "katameya heights", "kattamya", "القطامية",
        "gharb el golf", "غرب الجولف", "west golf",
        "gardenia", "جاردينيا", "district5", "district 5",
        "mivida", "lake view", "lakeview",
        "galleria moon valley", "fountain park", "patio oro",
        "marasem", "fifth square", "the villa",
        "mirage city", "mirage gardens", "east town", "eastown",
        "les rois", "nakheel park", "nakhel park",
        "la rosa", "oreana", "sarai", "saraya", "sarai compound"
    ],

    "Nasr City": [
        "مدينة نصر", "مدينه نصر", "م نصر", "nasr city",
        "عباس العقاد", "abbas el akkad",
        "مكرم عبيد", "makram ebeid", "makram ebaid", "maram ebed",
        "مصطفى النحاس", "mostafa el nahaas", "mustafa el nahas",
        "حسن المأمون", "hassan el maamoun",
        "الطيار", "el tayaran", "الطيران",
        "يوسف عباس", "youssef abbas",
        "النادي الاهلي", "النادى الاهلى",
        "الحي السابع", "الحي الثامن", "الحي العاشر",
        "المنطقة الاولى", "المنطقة الأولى", "المنطقة السادسة",
        "زهراء مدينة نصر", "zahraa nasr city",
        "samir abdel rouf", "samir a. #bdel rouf"
    ],

    "Heliopolis": [
        "مصر الجديدة", "heliopolis", "هليوبوليس", "كلية البنات", "ahmed taiser", "احمد تيسير", "النزهه الجديده", "النزهة الجديدة", "masr el gedida",
        "شيراتون", "sheraton",
        "النزهة", "nozha", "new nozha",
        "روكسي", "roxy", "الكوربة", "korba",
        "الميرغني", "هارون", "haroun", "baghdad street",
        "جسر السويس", "gesr el suez",
        "الف مسكن", "ألف مسكن",
        "عين شمس", "ain shams",
        "المطرية", "matariya",
        "الحجاز", "hegaz",
        "حلمية الزيتون", "الزيتون", "zayton", "el zayton",
        "حدائق الزيتون",
        "حدائق القبة", "حدائق القبه",
        "سرايا القبة", "سرايا القبه",
        "العباسية", "abbasia",
        "ارض الجولف", "ard el golf",
        "joseph tito",
        "هليوبوليس الجديدة"
    ],

    "Maadi": [
        "المعادي", "maadi",
        "دجلة", "degla",
        "زهراء المعادي", "zahraa el maadi",
        "ثكنات المعادي",
        "كورنيش المعادي",
        "حدائق المعادي",
        "المعادي الجديدة", "new maadi",
        "autostrad", "الاوتوستراد"
    ],

    "Mokattam": [
    "mokktam",
    "المقطم",
    "mokattam",
    "هضبة وسطى",
    "هضبة العليا",
    "asmarat",
    "الاسمرات"
],
    "Downtown": [
       "وسط البلد", "downtown",
        "التحرير", "باب الشعريه", "باب الشعرية", "السيدة زينب", "السيده زينب", "الفجالة", "الفجاله", "fagalaa", "fagala",# "tahrir",
        "طلعت حرب", "talaat harb",
        "رمسيس", "ramses",
        "العتبة", "ataba",
        "الموسكي", "mosky",
        "باب اللوق",
        "قصر النيل",
        "قصر العيني", "kasr el aini",
        "المنيل", "manial",
        "garden city",
        "غمرة", "ghamra",
        "الفسطاط", "مدينة الفسطاط", "الفسطاط الجديدة"
    ],

    "Zamalek": [
        "الزمالك", "zamalek"
    ],

    "Mohandessin": [
        "المهندسين", "el mohandsen", "mohandsen", "mohandessin",
        "جامعة الدول", "gamaa el dowal",
        "سوريا", "شارع سوريا",
        "لبنان", "شارع لبنان"
    ],

    "Dokki": [
        "الدقي", "dokki",
        "مصدق", "mesaha", "المساحة",
        "ميشيل باخوم"
    ],

    "Agouza": [
        "العجوزة", "agouza",
        "ميت عقبة", "mit okba"
    ],

    "Imbaba": [
        "امبابة", "إمبابة", "imbaba",
        "الوراق", "warraq",
        "بولاق الدكرور", "boulaq ad dakrour", "boulaq al dakrour"
    ],

    "Haram": [
        "الهرم", "haram",
        "المريوطية", "maryoteya", "marioteya",
        "مشعل", "meshaal",
        "الطالبية", "talbeya",
        "ترسا", "tersa"
    ],

    "Faisal": [
        "فيصل", "faisal"
    ],

    "Hadayek Al Ahram": [
        "حدائق الاهرام", "حدائق الأهرام",
        "hadayek al ahram","البوابه التانيه", "البوابة الثانية", "البوابه الرابعه", "البوابة الرابعة", "hadayek el ahram",
        "pyramids gardens", "pyramids garden",
        "بوابة خوفو", "بوابة حورس", "بوابة منقرع", "بوابة مينا"
    ],

    "Giza": [
        "الجيزة", "giza",
        "ميدان الجيزة",
        "الجلاء",
        "كرداسة", "kerdasa",
        "البدرشين",
        "صفط اللبن", "saft el laban",
        "بين السرايات",
        "ساقية مكي",
        "العمرانية", "omraneya",
        "المنيب", "moneeb"
    ],

    "October": [
        "اكتوبر", "أكتوبر",
        "6 october", "6th of october", "6 of october", "october", "palm hills", "palmhills", "bamboo extension", "السياحية الاولى", "السياحية الأولى", "جنه اكتوبر", "جنة اكتوبر", "جنة أكتوبر",
        "sixth october",
        "الحصري", "hosary",
        "دريم لاند", "dreamland",
        "حدائق اكتوبر", "حدائق أكتوبر", "october gardens",
        "غرب سوميد", "west somid",
        "degla palms",
        "wahat road", "طريق الواحات",
        "mena garden city",
        "palm parks",
        "gardenia park",
        "lakefront",
        "ابنى بيتك", "ibny beitak",
        "jannah october", "جنة اكتوبر"
    ],

    "Sheikh Zayed": [
        "الشيخ زايد", "زايد",
        "sheikh zayed", "zayed","بفرلي هيلز", "beverly hills", "الربوه", "الربوة",
        "بيفرلي", "beverly",
        "joulz",
        "الربوة",
        "زايد ديونز", "zayed dunes",
        "داندي مول", "dandy mall",
        "هايبر وان", "hyper one",
        "الخمايل", "khamayel",
        "جنة زايد", "jannah zayed"
    ],

    "Helwan": [
        "حلوان", "helwan",
        "عين حلوان",
        "وادي حوف",
        "حدائق حلوان",
        "دار السلام",
        "البساتين",
        "طرة", "tora",
        "المعصرة", "maasara",
        "15 مايو", "15 may"
    ],

    "Shubra": [
        "شبرا", "شبرا مصر", "shobra", "shoubra",
        "روض الفرج", "rod el farag",
        "الساحل", "elsahel"
    ],

    "North Cairo": [
        "العبور","الزاويه الحمراء", "الزاوية الحمراء", "obour",
        "السلام", "salam city",
        "المرج", "marg",
        "الخصوص",
        "الخانكة",
        "شبرا الخيمة", "shobra el kheima",
        "بهتيم", "bahtim",
        "مسطرد", "mostorod"
    ],

    "Alexandria": [
        "الاسكندرية", "الإسكندرية", "alexandria",
        "سموحة", "smouha",
        "فوزي معاذ",
        "sidi gaber",
        "لوران", "laurent",
        "ميامي", "miami",
        "محرم بك", "moharam bek",
        "العجمي", "agamy"
    ],

    "OOZ": [
        "العاشر من رمضان","فارسكور", "كفر العرب", "10th of ramadan",
        "العاصمة الادارية", "العاصمة الإدارية", "new capital",
        "الاسماعيلية", "الإسماعيلية", "ismailia",
        "السويس", "suez",
        "الفيوم", "fayoum",
        "السادات", "sadat",
        "بني سويف", "bani suef",
        "galala", "ain sokhna", "sokhna", "shokna",
        "aswan", "أسوان",
        "qalyubia", "قليوبية",
        "abu zabal",
        "al burouj", "el burouj",
        "القنطرة", "qantara",
        "بلقاس", "belqas",
        "جمصة", "gamasa",
        "دمياط", "damietta",
        "test"
    ]
}


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = (
        text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        .replace("ة", "ه").replace("ى", "ي")
    )
    return text.lower().strip()


_AREA_KEYWORDS_NORMALIZED = [
    (area, _normalize_text(kw))
    for area, kws in AREA_KEYWORDS.items()
    for kw in kws
]


def detect_area(address: Optional[str]) -> str:
    """Same keyword-matching logic as the existing Dispatch Planner."""
    if not address:
        return "Unknown"
    text = _normalize_text(address)
    for area, norm_kw in _AREA_KEYWORDS_NORMALIZED:
        if norm_kw in text:
            return area
    return "Unknown"


# ---------------------------------------------------------------------------
# Area classification cache + optional AI fallback for addresses that don't
# match any AREA_KEYWORDS entry (real addresses will always have spelling
# variants a fixed keyword list can't fully cover).
#
# Two-layer design:
#   1. Cache lookup (enrich_areas_with_cache) — FREE, runs automatically on
#      every load. Once an address has been classified once (by cache or by
#      AI), it's remembered on disk forever and never costs anything again.
#   2. AI classification (classify_unknown_addresses) — PAID (calls OpenAI),
#      only runs when explicitly triggered by the person (a button in the
#      UI), reuses dashboard.py's classify_unknown_areas_gpt logic exactly
#      but without any Streamlit coupling so it works as a plain pipeline
#      function. Every result it produces is written to the cache, so the
#      same address is never sent to GPT twice.
# ---------------------------------------------------------------------------
def _load_area_cache(cache_path: str = AREA_CACHE_PATH) -> dict:
    """normalized_address -> area. Missing/corrupt cache = empty dict, never an error."""
    if not os.path.exists(cache_path):
        return {}
    try:
        cache_df = pd.read_csv(cache_path, encoding="utf-8-sig")
        return dict(zip(cache_df["normalized_address"], cache_df["area"]))
    except Exception:
        return {}


def _save_area_cache(cache: dict, cache_path: str = AREA_CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    pd.DataFrame(
        [{"normalized_address": k, "area": v} for k, v in cache.items()]
    ).to_csv(cache_path, index=False, encoding="utf-8-sig")


def enrich_areas_with_cache(df: pd.DataFrame, cache_path: str = AREA_CACHE_PATH) -> pd.DataFrame:
    """FREE pass: fills in df['area'] for any 'Unknown' row whose address was
    already classified before (by a previous AI run or manually curated
    into the cache file). Safe to call unconditionally on every load — pure
    local file read, no network call, degrades to a no-op if there's no
    cache yet or no 'delivering_street'/'area' columns."""
    if df.empty or "area" not in df.columns:
        return df
    cache = _load_area_cache(cache_path)
    if not cache:
        return df
    addr_col = "delivering_street" if "delivering_street" in df.columns else None
    if addr_col is None:
        return df
    unknown_mask = df["area"] == "Unknown"
    if not unknown_mask.any():
        return df
    normalized = df.loc[unknown_mask, addr_col].fillna("").map(_normalize_text)
    resolved = normalized.map(cache)  # NaN where not in cache
    df.loc[unknown_mask, "area"] = resolved.fillna(df.loc[unknown_mask, "area"])
    return df


def classify_unknown_areas_gpt(addresses: list, known_areas: list, openai_api_key: str) -> dict:
    """Same approach as streamlit_ui/dashboard.py's classify_unknown_areas_gpt
    (gpt-4o-mini, batches of 80, JSON response) — reimplemented here without
    any Streamlit dependency so it's a plain, testable pipeline function.
    Returns {index: area_name}; on any batch failure that batch is simply
    skipped (returns fewer entries than requested), never raises, so a
    partial AI outage doesn't take down the whole classification run."""
    import json
    from openai import OpenAI

    if not addresses or not openai_api_key:
        return {}

    client = OpenAI(api_key=openai_api_key)
    areas_str = ", ".join(known_areas)
    results = {}
    batch_size = 80

    for batch_start in range(0, len(addresses), batch_size):
        batch = addresses[batch_start:batch_start + batch_size]
        addr_lines = "\n".join(f"{i}: {addr}" for i, addr in enumerate(batch))
        prompt = (
            "You are classifying Egyptian delivery addresses into delivery zones.\n\n"
            f"Known zones: {areas_str}\n\n"
            "For each numbered address below, pick the best matching zone from the list.\n"
            'Use "Unknown" only if truly none match.\n'
            'Reply ONLY with a JSON object mapping index (as string) to zone name:\n'
            '{"0": "New Cairo", "1": "Giza", ...}\n\n'
            f"Addresses:\n{addr_lines}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0,
                response_format={"type": "json_object"},
            )
            batch_result = json.loads(response.choices[0].message.content)
            for k, v in batch_result.items():
                results[batch_start + int(k)] = v
        except Exception as e:
            results[f"_error_batch_{batch_start // batch_size + 1}"] = str(e)

    return results


def classify_unknown_addresses(df: pd.DataFrame, openai_api_key: str,
                                cache_path: str = AREA_CACHE_PATH) -> tuple:
    """
    Main entry point for the 'Classify unknown areas with AI' button.

    1. Runs the free cache pass first (in case anything's already known).
    2. Collects the UNIQUE remaining 'Unknown' addresses (not one call per
       shipment — many shipments share the same or similar address, so this
       is usually a much smaller list than the row count suggests).
    3. Sends only those to GPT, saves every result to the cache immediately
       so a crash/interruption mid-run doesn't lose already-paid-for work.
    4. Applies the newly learned areas back onto the full dataframe.

    Returns (updated_df, stats) where stats = {
        "unknown_before": int, "unique_addresses_sent": int,
        "newly_classified": int, "unknown_after": int, "errors": list[str],
    }
    """
    df = enrich_areas_with_cache(df, cache_path)
    if df.empty or "area" not in df.columns:
        return df, {"unknown_before": 0, "unique_addresses_sent": 0,
                     "newly_classified": 0, "unknown_after": 0, "errors": []}

    addr_col = "delivering_street" if "delivering_street" in df.columns else None
    unknown_before = int((df["area"] == "Unknown").sum())
    if unknown_before == 0 or addr_col is None:
        return df, {"unknown_before": unknown_before, "unique_addresses_sent": 0,
                     "newly_classified": 0, "unknown_after": unknown_before, "errors": []}

    unknown_rows = df[df["area"] == "Unknown"]
    # Map normalized_address -> one representative original address string
    # (GPT sees the original text; the cache key is the normalized version
    # so minor whitespace/diacritic differences still hit the same cache
    # entry next time).
    unique_addrs = {}
    for raw_addr in unknown_rows[addr_col].fillna(""):
        norm = _normalize_text(raw_addr)
        if norm and norm not in unique_addrs:
            unique_addrs[norm] = raw_addr

    norm_keys = list(unique_addrs.keys())
    original_addrs = [unique_addrs[k] for k in norm_keys]
    known_areas = [a for a in AREA_KEYWORDS.keys()]

    raw_results = classify_unknown_areas_gpt(original_addrs, known_areas, openai_api_key)
    errors = [v for k, v in raw_results.items() if isinstance(k, str) and k.startswith("_error_")]

    cache = _load_area_cache(cache_path)
    newly_classified = 0
    for idx, area in raw_results.items():
        if isinstance(idx, str):  # skip the "_error_batch_N" entries
            continue
        norm_key = norm_keys[idx]
        if area and area != "Unknown":
            cache[norm_key] = area
            newly_classified += 1
    _save_area_cache(cache, cache_path)

    df = enrich_areas_with_cache(df, cache_path)
    unknown_after = int((df["area"] == "Unknown").sum())

    return df, {
        "unknown_before": unknown_before,
        "unique_addresses_sent": len(norm_keys),
        "newly_classified": newly_classified,
        "unknown_after": unknown_after,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# REAL data loader — Live API (primary, authoritative source)
#
# 2026-07-15 migration: this now talks to the official Weevo ADMIN
# Dashboard backend instead of the older AI-agent endpoint. Two things
# changed on purpose, both requested by the business:
#   1. Date filtering is sent to the server (start_delivery_date /
#      end_delivery_date) instead of being applied after the fact to a
#      bounded "most recent N" pull — so numbers now match the official
#      Admin Dashboard for the same date range.
#   2. One combined pull per date range instead of one request per
#      status — confirmed against a real captured response that omitting
#      `status` returns every status combined, with pagination.total
#      matching the response's own statusCounts.Total exactly.
# Every row-mapping choice below (which field feeds which column) is
# UNCHANGED from the previous implementation — same columns, same
# fallbacks, same semantics (e.g. "delivered_at" is still the target
# date_to_deliver_shipment, not the real delivered_date_at, to keep
# delivery_hours meaning exactly what it meant before) — confirmed with
# the business rather than "upgraded" silently, since the goal of this
# migration is a different data source, not different analytics.
# ---------------------------------------------------------------------------
def _parse_token_expiry(payload: dict) -> datetime:
    """Normalizes whatever the login response gives us for expiry into a
    timezone-aware UTC datetime, minus TOKEN_EXPIRY_SKEW_V1 seconds of
    safety margin. Handles the two shapes an auth endpoint like this
    commonly returns:
      - `expires_at`: an ISO-8601 timestamp, or a unix epoch (int/float/
        numeric string)
      - `expires_in`: seconds-from-now, used only if `expires_at` is absent
    Falls back to a conservative 5-minute lifetime if neither is present,
    rather than guessing something longer and risking stale-token 401s."""
    now = datetime.now(timezone.utc)
    raw_expires_at = payload.get("expires_at")

    if raw_expires_at is not None:
        if isinstance(raw_expires_at, (int, float)):
            expires_at = datetime.fromtimestamp(float(raw_expires_at), tz=timezone.utc)
        else:
            text = str(raw_expires_at).strip()
            expires_at = None
            # Confirmed real shape (captured on the old login-using-token
            # response; assumed — not yet independently confirmed — to be
            # the same shape on the real /login bootstrap response):
            # "expires_at": "2126-07-15 04:05:11" — a naive "YYYY-MM-DD
            # HH:MM:SS" string, no timezone offset. Treated as UTC, same
            # as every other timestamp compared against in this module.
            try:
                expires_at = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
            if expires_at is None:
                try:
                    expires_at = datetime.fromisoformat(text.replace("Z", "+00:00"))
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            if expires_at is None:
                try:
                    expires_at = datetime.fromtimestamp(float(text), tz=timezone.utc)
                except ValueError:
                    expires_at = now + timedelta(minutes=5)
    elif payload.get("expires_in") is not None:
        expires_at = now + timedelta(seconds=float(payload["expires_in"]))
    else:
        expires_at = now + timedelta(minutes=5)

    return expires_at - timedelta(seconds=TOKEN_EXPIRY_SKEW_V1)


def _login_admin(base_url: str = WEEVO_API_BASE_URL,
                  timeout: int = REQUEST_TIMEOUT_V1) -> dict:
    """POSTs to the real bootstrap login (email+password) and returns the
    parsed {"access_token", "expires_at"} pair. Confirmed 2026-07-16
    against a real captured browser login: login-using-token (the prior
    attempt) turned out to require an already-valid JWT, not the old
    WEEVO_API_KEY integration key — this is the actual bootstrap step.
    Credentials come ONLY from WEEVO_ADMIN_EMAIL / WEEVO_ADMIN_PASSWORD in
    the environment — never hardcoded, never logged, never included in any
    error message this function raises. Same gateway-error retry
    discipline as every other Admin Dashboard call in this module. Raises
    httpx.HTTPStatusError / httpx.RequestError / ValueError on failure —
    the caller decides how to surface that (auth failure is not "no
    data")."""
    email = os.environ.get("WEEVO_ADMIN_EMAIL", "")
    password = os.environ.get("WEEVO_ADMIN_PASSWORD", "")
    if not email or not password:
        raise ValueError(
            "WEEVO_ADMIN_EMAIL / WEEVO_ADMIN_PASSWORD are not set in the environment"
        )

    url = f"{base_url}{ADMIN_EMAIL_LOGIN_PATH}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    body = {"email": email, "password": password}

    attempt = 0
    while True:
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=timeout)
            if response.status_code in GATEWAY_ERROR_CODES_V1 and attempt < MAX_RETRIES_V1:
                attempt += 1
                time.sleep(RETRY_PAUSE_V1)
                continue
            response.raise_for_status()
            payload = response.json() or {}
            # Some backends nest the token under a "data" envelope — accept
            # either shape rather than hard-failing on a wrapper difference.
            data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            access_token = data.get("access_token")
            if not access_token:
                raise ValueError("login response did not include an access_token")
            return {"access_token": access_token, "expires_at": _parse_token_expiry(data)}
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES_V1:
                attempt += 1
                time.sleep(RETRY_PAUSE_V1)
                continue
            raise


def _get_admin_bearer_token(api_key: str, base_url: str = WEEVO_API_BASE_URL,
                             force_refresh: bool = False) -> str:
    """Returns a valid Bearer access token for base_url, logging in only
    when there is no cached token yet, the cached one has expired, or the
    caller explicitly asks for a refresh (used after an HTTP 401). Cached
    in-process only — never persisted to disk.

    `api_key` is kept in the signature UNCHANGED (2026-07-16) purely so
    every existing caller in this module and in analytics_pipeline_v2.py /
    streamlit_ui/analytics_page.py keeps working without modification —
    real auth no longer depends on it at all (see _login_admin() above);
    the email+password credentials come from the environment directly."""
    cache_key = base_url
    with _ADMIN_TOKEN_CACHE_LOCK:
        cached = _ADMIN_TOKEN_CACHE.get(cache_key)
        if (not force_refresh and cached is not None
                and datetime.now(timezone.utc) < cached["expires_at"]):
            return cached["access_token"]

    fresh = _login_admin(base_url=base_url)
    with _ADMIN_TOKEN_CACHE_LOCK:
        _ADMIN_TOKEN_CACHE[cache_key] = fresh
    return fresh["access_token"]


def _fetch_admin_shipments_page(api_key: str, page: int, limit: int = ADMIN_PAGE_SIZE,
                                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                                 country_id: int = ADMIN_DEFAULT_COUNTRY_ID,
                                 in_batch: int = ADMIN_DEFAULT_IN_BATCH,
                                 base_url: str = WEEVO_API_BASE_URL,
                                 timeout: int = REQUEST_TIMEOUT_V1) -> tuple[list, dict, dict]:
    """One page of the combined (no status filter) admin shipments pull.
    Same retry-on-gateway-error / timeout discipline as before. Returns
    (records, pagination, status_counts) — status_counts is the response's
    own `statusCounts` block, which (confirmed) reflects the ENTIRE
    filtered result set for the date range, not just this page.

    Auth (2026-07-16): sends the cached Bearer access token, obtained via
    the real email+password bootstrap login (_login_admin()). On a 401
    the token is refreshed exactly once and the request retried with the
    new token — a 401 after that refresh is treated as a real auth
    failure, not retried further."""
    url = f"{base_url}{ADMIN_SHIPMENTS_PATH}"
    params = {"country_id": country_id, "page": page, "paginate": limit, "in_batch": in_batch}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    attempt = 0
    reauthed = False
    while True:
        token = _get_admin_bearer_token(api_key, base_url=base_url, force_refresh=False)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        try:
            response = httpx.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code == 401 and not reauthed:
                # Token expired/invalidated server-side ahead of our cached
                # expiry — refresh once and retry, exactly one time.
                reauthed = True
                _get_admin_bearer_token(api_key, base_url=base_url, force_refresh=True)
                continue
            if response.status_code in GATEWAY_ERROR_CODES_V1 and attempt < MAX_RETRIES_V1:
                attempt += 1
                time.sleep(RETRY_PAUSE_V1)
                continue
            response.raise_for_status()
            payload = response.json() or {}
            shipments_block = payload.get("shipments") or {}
            records = shipments_block.get("data") or []
            pagination = {
                "current_page": shipments_block.get("current_page"),
                "last_page": shipments_block.get("last_page"),
                "per_page": shipments_block.get("per_page"),
                "total": shipments_block.get("total"),
            }
            status_counts = payload.get("statusCounts") or {}
            return records, pagination, status_counts
        except httpx.TimeoutException:
            if attempt < MAX_RETRIES_V1:
                attempt += 1
                time.sleep(RETRY_PAUSE_V1)
                continue
            raise


def fetch_admin_shipments(api_key: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                           country_id: int = ADMIN_DEFAULT_COUNTRY_ID, in_batch: int = ADMIN_DEFAULT_IN_BATCH,
                           base_url: str = WEEVO_API_BASE_URL, limit: int = ADMIN_PAGE_SIZE) -> tuple[list, dict, dict]:
    """
    Walks every page (sequential only, same short pause between requests,
    same hard wall-clock time budget) for the given date range and returns
    every shipment combined across all statuses. This is the ONE fetch
    both the main analytics section and the Financial/Risk (v2) section
    are built from (see streamlit_ui/analytics_page.py), so both always
    reflect the exact same underlying records.

    If `start_date`/`end_date` are both None ("All time"), no date filter
    is sent at all — same "preserve existing behavior, no new artificial
    limit" as before; the only ceiling is the existing FETCH_TIME_BUDGET_V1
    safety valve, exactly as it worked previously (remaining pages are
    marked skipped rather than attempted, so a fully-down backend still
    fails predictably instead of stalling the page).

    Returns (records, status_counts, fetch_meta) where fetch_meta carries
    enough detail for each pipeline (v1/v2) to build its own
    fetch_diagnostics dict in the exact shape its downstream UI code
    already expects.
    """
    all_records: list = []
    status_counts: dict = {}
    start_time = time.monotonic()
    page = 1
    last_page = 1
    error = None
    truncated_by_budget = False
    pages_fetched = 0

    label = "shipments"
    if start_date or end_date:
        label = f"shipments ({start_date or '…'} → {end_date or '…'})"

    while page <= last_page:
        if time.monotonic() - start_time > FETCH_TIME_BUDGET_V1:
            truncated_by_budget = True
            break
        try:
            records, pagination, sc = _fetch_admin_shipments_page(
                api_key, page=page, limit=limit, start_date=start_date, end_date=end_date,
                country_id=country_id, in_batch=in_batch, base_url=base_url,
            )
        except httpx.HTTPStatusError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            break
        except httpx.RequestError as e:
            error = str(e)
            break

        all_records.extend(records)
        pages_fetched += 1
        if page == 1:
            status_counts = sc
        last_page = pagination.get("last_page") or 1
        if page >= last_page:
            break
        page += 1
        time.sleep(INTER_REQUEST_DELAY_V1)

    if error and not all_records:
        # Total failure — almost certainly auth/connectivity, not "no
        # orders". Same non-silent behavior as the old per-status loader.
        raise ConnectionError(
            f"Could not reach the Weevo admin API. Error: {error}"
        )

    fetch_meta = {
        "label": label,
        "error": error,
        "truncated_by_budget": truncated_by_budget,
        "pages_fetched": pages_fetched,
        "last_page": last_page,
        "fetched_at": datetime.now().isoformat(),
    }
    return all_records, status_counts, fetch_meta


def _v1_fetch_diagnostics_from_meta(fetch_meta: Optional[dict]) -> Optional[dict]:
    """Adapts fetch_admin_shipments()'s generic fetch_meta into the exact
    dict shape streamlit_ui/analytics_page.py already reads for the "partial
    fetch failure" banner (total_statuses / succeeded_statuses (list) /
    failed_statuses (list) / errors (dict)) — unchanged UI code, just a
    single pseudo-entry instead of one entry per status now that fetching
    is combined."""
    if not fetch_meta:
        return None
    label = fetch_meta["label"]
    if fetch_meta.get("error") or fetch_meta.get("truncated_by_budget"):
        if fetch_meta.get("error"):
            reason = fetch_meta["error"]
        else:
            reason = (
                f"Stopped after the {FETCH_TIME_BUDGET_V1}s time budget "
                f"(page {fetch_meta['pages_fetched']} of {fetch_meta['last_page']}) "
                f"— remaining pages skipped."
            )
        return {
            "total_statuses": 1,
            "succeeded_statuses": [label] if fetch_meta["pages_fetched"] > 0 else [],
            "failed_statuses": [label],
            "errors": {label: reason},
            "fetched_at": fetch_meta["fetched_at"],
        }
    return {
        "total_statuses": 1,
        "succeeded_statuses": [label],
        "failed_statuses": [],
        "errors": {},
        "fetched_at": fetch_meta["fetched_at"],
    }


def build_shipments_dataframe(raw_records: list, fetch_meta: Optional[dict] = None,
                               status_counts: Optional[dict] = None) -> pd.DataFrame:
    """Pure mapping: raw admin-API shipment dicts -> the same ARCHIVE_COLUMNS
    shape/semantics this pipeline has always produced. No HTTP in here —
    split out from the old load_real_shipments_from_api() specifically so
    streamlit_ui/analytics_page.py can fetch ONCE and hand the same raw
    records to both this function and analytics_pipeline_v2's equivalent
    (build_v2_dataframe), guaranteeing both dashboard sections are built
    from the exact same underlying shipments."""
    rows = []
    for s in raw_records:
        merchant_obj = s.get("merchant") or {}
        merchant_name = merchant_obj.get("name") or merchant_obj.get("brand_name") or "Unknown"

        courier_obj = s.get("courier") or {}
        courier_name = courier_obj.get("name") or "Unassigned"
        courier_phone = courier_obj.get("phone")

        # Unchanged semantics, confirmed with the business (2026-07-15):
        # "received_at"/"delivered_at" stay the TARGET dates
        # (date_to_receive_shipment / date_to_deliver_shipment), not the
        # real delivered_date_at now available — switching would change
        # what delivery_hours measures, which is out of scope for this
        # migration (data source only, not calculations).
        received_raw = s.get("date_to_receive_shipment")
        delivered_raw = s.get("date_to_deliver_shipment")
        order_date = received_raw or delivered_raw

        street = s.get("delivering_street") or ""
        building = s.get("delivering_building_number") or ""
        address = f"{street} {building}".strip()

        rows.append({
            "phone_number": s.get("client_phone"),
            "shipment_id": s.get("id") or s.get("barcode"),
            "product_name": None,  # not present in the payload
            "merchant_name": merchant_name,
            "courier_name": courier_name,
            "courier_phone": courier_phone,
            "delivery_address": address,
            "delivery_city": detect_area(address),
            "delivery_state": None,
            "amount": s.get("total_amount") if s.get("total_amount") is not None else s.get("amount"),
            "delivery_date": order_date,
            "stored_at": received_raw,
            "status": s.get("status"),
            "attempts": s.get("attempts") or 1,
            "area": detect_area(address),
            "received_at": received_raw,
            "delivered_at": delivered_raw,
            "created_at": s.get("created_at"),
        })

    df = pd.DataFrame(rows)
    diagnostics = _v1_fetch_diagnostics_from_meta(fetch_meta)
    if df.empty:
        if diagnostics is not None:
            df.attrs["fetch_diagnostics"] = diagnostics
        if status_counts is not None:
            df.attrs["status_counts"] = status_counts
        return df

    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    # FINDING 2 FIX (2026-07-17): used to unconditionally drop any shipment
    # missing BOTH date_to_receive_shipment and date_to_deliver_shipment via
    # df.dropna(subset=["delivery_date"]) — applied to every status, even
    # with no date filter active, silently shrinking Total Orders/status
    # breakdown below what the server actually returned. Removed: rows with
    # a NaT delivery_date are kept (they just don't participate in the
    # date-based views below, which already skip NaT safely via pandas'
    # default skipna behavior in resample/max/min/comparisons/sort_values).

    df["received_at"] = pd.to_datetime(df["received_at"], errors="coerce")
    df["delivered_at"] = pd.to_datetime(df["delivered_at"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    both_present = df["received_at"].notna() & df["delivered_at"].notna()
    df["delivery_hours"] = pd.NA
    df.loc[both_present, "delivery_hours"] = (
        (df.loc[both_present, "delivered_at"] - df.loc[both_present, "received_at"])
        .dt.total_seconds() / 3600
    )
    df["delivery_hours"] = pd.to_numeric(df["delivery_hours"], errors="coerce")
    df = enrich_areas_with_cache(df)
    if diagnostics is not None:
        df.attrs["fetch_diagnostics"] = diagnostics
    if status_counts is not None:
        df.attrs["status_counts"] = status_counts
    return df


def load_real_shipments_from_admin_api(api_key: str, base_url: str = WEEVO_API_BASE_URL,
                                        start_date: Optional[str] = None,
                                        end_date: Optional[str] = None) -> pd.DataFrame:
    """Standalone convenience wrapper: fetch + map in one call, same
    contract load_real_shipments_from_api() used to have. The Streamlit
    page itself doesn't call this directly (it shares one fetch_admin_shipments()
    pull across v1 and v2 — see _cached_admin_shipments in analytics_page.py),
    but this keeps load_shipments(source="api") working standalone for any
    other caller.

    start_date/end_date default to None/None ("no range given"), which
    resolves to the last 30 days here — never an unbounded all-history
    fetch — matching the Streamlit page's own "All time" default."""
    if start_date is None and end_date is None:
        _end = datetime.now().date()
        _start = _end - timedelta(days=30)
        start_date, end_date = _start.strftime("%Y-%m-%d"), _end.strftime("%Y-%m-%d")
    records, status_counts, fetch_meta = fetch_admin_shipments(
        api_key=api_key, start_date=start_date, end_date=end_date, base_url=base_url,
    )
    return build_shipments_dataframe(records, fetch_meta=fetch_meta, status_counts=status_counts)


# ---------------------------------------------------------------------------
# Real data loader — local DB cache (secondary / fallback source)
# ---------------------------------------------------------------------------
def load_real_shipments(db_path: str = "./weevo_chatbot.db") -> pd.DataFrame:
    """
    Reads scheduler_shipment_data as-is. Synchronous sqlite3 (read-only) is
    used instead of aiosqlite here on purpose — this pipeline only ever
    reads, and Streamlit's execution model makes sync code simpler and
    safer than juggling an event loop per page render.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run from the project root, "
            f"or pass the correct db_path."
        )

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT phone_number, shipment_id, product_name, merchant_name,
                   courier_name, courier_phone, delivery_address,
                   delivery_city, delivery_state, amount, delivery_date,
                   stored_at
            FROM scheduler_shipment_data
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["area"] = df["delivery_address"].apply(detect_area)

    # This source has no receive/deliver timestamp pair (only a single
    # delivery_date) — keep the columns present but empty so downstream
    # delivery-time functions can detect "not available" instead of
    # erroring on a missing column.
    df["received_at"] = pd.NaT
    df["delivered_at"] = pd.NaT
    df["delivery_hours"] = pd.NA
    df["delivery_hours"] = pd.to_numeric(df["delivery_hours"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Mock data generator — clearly labeled, structurally identical to real data
# ---------------------------------------------------------------------------
_MOCK_MERCHANTS = [
    "Cairo Kicks Store", "Bella Home Decor", "TechZone Electronics",
    "Nour Beauty Supplies", "Fresh Cart Grocery", "Al Rayan Fashion",
    "Baby World Egypt", "Sport Land", "The Book Nook", "Glow Cosmetics",
]
_MOCK_COURIERS = [
    "Ahmed Mahmoud", "Mohamed Ali", "Youssef Kamal", "Omar Hassan",
    "Karim Fathy", "Mahmoud Saeed", "Hassan Nabil",
]
_MOCK_AREAS = list(AREA_KEYWORDS.keys())[:8] or ["New Cairo", "Nasr City", "Maadi"]
# A few couriers deliberately slower in the demo, purely so the new
# delivery-time section has visible signal to show — has zero effect on
# any of the original mock fields/columns above.
_MOCK_SLOW_COURIERS = {"Karim Fathy", "Hassan Nabil"}


def generate_mock_shipments(n: int = 1800, days_back: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generates a mock dataset with the same columns load_real_shipments()
    returns, so every function below works unchanged on either source.
    Not randomized beyond a fixed seed, so the demo dashboard looks the
    same every time it's shown.
    """
    rng = random.Random(seed)
    rows = []
    today = datetime.now()

    # give merchants and couriers uneven activity so "top" lists look real
    merchant_weights = [rng.uniform(0.3, 3.0) for _ in _MOCK_MERCHANTS]
    courier_weights = [rng.uniform(0.5, 2.5) for _ in _MOCK_COURIERS]

    for i in range(n):
        days_ago = rng.betavariate(1.6, 3.2) * days_back  # skew toward recent
        delivery_date = today - timedelta(days=days_ago)
        merchant = rng.choices(_MOCK_MERCHANTS, weights=merchant_weights)[0]
        courier = rng.choices(_MOCK_COURIERS, weights=courier_weights)[0]
        area = rng.choice(_MOCK_AREAS)
        amount = round(rng.uniform(120, 2400), 2)

        # Additive-only: received_at/delivered_at/delivery_hours simulate
        # what the real API's date_to_receive_shipment / date_to_deliver_shipment
        # pair looks like once a shipment is delivered. ~85% of mock rows are
        # "delivered" (have both timestamps); the rest simulate in-progress
        # shipments with only a receive time, matching the real data shape.
        received_at = delivery_date - timedelta(hours=rng.uniform(0.5, 3))
        is_delivered = rng.random() < 0.85
        base_hours = rng.uniform(1.0, 2.6)
        if courier in _MOCK_SLOW_COURIERS:
            base_hours += rng.uniform(1.5, 3.5)
        delivered_at = received_at + timedelta(hours=base_hours) if is_delivered else None
        if is_delivered:
            status = "delivered"
        else:
            status = rng.choices(
                ["returned", "available", "on-delivery", "in-transit", "in-transit-for-return", "courier-applied-to-shipment"],
                weights=[0.15, 0.25, 0.25, 0.15, 0.1, 0.1],
            )[0]

        rows.append({
            "phone_number": f"2010{rng.randint(10000000, 99999999)}",
            "shipment_id": f"SHIP-{100000 + i}",
            "product_name": rng.choice([
                "Running Shoes", "Table Lamp", "Wireless Earbuds", "Face Cream",
                "Grocery Bundle", "Winter Jacket", "Baby Stroller", "Notebook Set",
            ]),
            "merchant_name": merchant,
            "courier_name": courier,
            "courier_phone": f"2011{rng.randint(10000000, 99999999)}",
            "delivery_address": f"{area} area, street {rng.randint(1, 50)}",
            "delivery_city": area,
            "delivery_state": "Cairo" if "Alex" not in area else "Alexandria",
            "amount": amount,
            "delivery_date": delivery_date,
            "stored_at": delivery_date.isoformat(),
            "status": status,
            "area": area,
            "received_at": received_at,
            "delivered_at": delivered_at,
        })

    df = pd.DataFrame(rows)
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])
    df["received_at"] = pd.to_datetime(df["received_at"])
    df["delivered_at"] = pd.to_datetime(df["delivered_at"])
    both_present = df["received_at"].notna() & df["delivered_at"].notna()
    df["delivery_hours"] = pd.NA
    df.loc[both_present, "delivery_hours"] = (
        (df.loc[both_present, "delivered_at"] - df.loc[both_present, "received_at"])
        .dt.total_seconds() / 3600
    )
    df["delivery_hours"] = pd.to_numeric(df["delivery_hours"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_shipments(
    source: str = "mock",
    api_key: Optional[str] = None,
    base_url: str = WEEVO_API_BASE_URL,
    db_path: str = "./weevo_chatbot.db",
    # kept for backward compatibility with earlier calls:
    use_mock_data: Optional[bool] = None,
    # ADDED (2026-07-15 admin-API migration): only used when source="api".
    # "YYYY-MM-DD" strings or None. None/None ("no range given") resolves
    # to the last 30 days inside load_real_shipments_from_admin_api below
    # (2026-07-17 default-date fix) — never an unbounded all-history fetch.
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    source: "api" (live Weevo Admin Dashboard API, needs api_key), "db"
    (local sqlite cache), or "mock" (demo data). `use_mock_data` is
    accepted as a legacy alias — if passed, it overrides `source`.
    `db`/`mock` behavior is completely unchanged by the 2026-07 API
    migration; only the "api" branch talks to a different endpoint.
    """
    if use_mock_data is not None:
        source = "mock" if use_mock_data else "db"

    if source == "api":
        if not api_key:
            raise ValueError("source='api' requires api_key (set WEEVO_API_KEY or pass it explicitly).")
        return load_real_shipments_from_admin_api(
            api_key=api_key, base_url=base_url, start_date=start_date, end_date=end_date,
        )
    elif source == "db":
        return load_real_shipments(db_path=db_path)
    else:
        return generate_mock_shipments()


# ---------------------------------------------------------------------------
# Aggregations — every one of these only uses fields confirmed to exist.
# No "rating" or "delivery duration" metric is computed anywhere, because
# neither field exists in scheduler_shipment_data today (see module docstring).
# ---------------------------------------------------------------------------
def orders_over_time(df: pd.DataFrame, granularity: str = "D") -> pd.DataFrame:
    """granularity: 'D' daily, 'W' weekly, 'M' monthly."""
    if df.empty:
        return pd.DataFrame(columns=["period", "orders", "revenue"])
    grouped = (
        df.set_index("delivery_date")
        .resample(granularity)
        .agg(orders=("shipment_id", "count"), revenue=("amount", "sum"))
        .reset_index()
        .rename(columns={"delivery_date": "period"})
    )
    return grouped


def status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Every status present in the loaded window, with count and share of
    total — the actual composition behind the single 'Total orders' KPI
    number (e.g. '577 = 200 delivered + 200 returned + 177 in-flight')."""
    if df.empty or "status" not in df.columns:
        return pd.DataFrame(columns=["status", "orders", "pct"])
    counts = df["status"].value_counts(dropna=True).reset_index()
    counts.columns = ["status", "orders"]
    counts["pct"] = round(counts["orders"] / counts["orders"].sum() * 100, 1)
    return counts.sort_values("orders", ascending=False).reset_index(drop=True)


def top_areas(df: pd.DataFrame, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    """
    Ranks real areas only. 'Unknown' (address didn't match any known
    area keyword) is a data-quality placeholder, not a real area — if
    it were left in, it would appear in the ranking as if it were a
    place couriers could actually be sent to. Excluded here; the count
    of excluded orders is attached via .attrs so the page can still show
    it transparently instead of just hiding it.

    ascending=True gives the LEAST busy areas instead of the most —
    useful for spotting areas that may be under-served or where demand
    is genuinely low (as opposed to areas with no orders because of an
    address-matching gap, which is a different problem tracked separately
    via excluded_unknown_count).
    """
    if df.empty:
        result = pd.DataFrame(columns=["area", "orders"])
        result.attrs["excluded_unknown_count"] = 0
        return result
    excluded_count = int((df["area"] == "Unknown").sum())
    known = df[df["area"] != "Unknown"]
    result = (
        known.groupby("area")["shipment_id"]
        .count()
        .reset_index(name="orders")
        .sort_values("orders", ascending=ascending)
        .head(n)
    )
    result.attrs["excluded_unknown_count"] = excluded_count
    return result


def courier_leaderboard(df: pd.DataFrame, n: int = 15, ascending: bool = False) -> pd.DataFrame:
    """
    Order count + total handled value per courier. Does NOT include
    delivery time or rating — those fields don't exist yet (see docstring).

    'Unassigned' (no captain on record yet — normal for e.g. 'available'
    status orders that haven't been picked up) is excluded from the
    ranking so it doesn't show up as if it were a real courier with a
    huge order count. Excluded count attached via .attrs.

    ascending=True gives the LEAST active couriers instead of the most —
    same underlying numbers, just sorted the other way, for spotting
    couriers who may need attention or more orders routed to them.
    """
    if df.empty:
        result = pd.DataFrame(columns=["courier_name", "orders", "total_value"])
        result.attrs["excluded_unassigned_count"] = 0
        return result
    excluded_count = int((df["courier_name"] == "Unassigned").sum())
    assigned = df[df["courier_name"] != "Unassigned"]
    result = (
        assigned.groupby("courier_name")
        .agg(orders=("shipment_id", "count"), total_value=("amount", "sum"))
        .reset_index()
        .sort_values("orders", ascending=ascending)
        .head(n)
    )
    result.attrs["excluded_unassigned_count"] = excluded_count
    return result


def merchant_activity(df: pd.DataFrame, recent_days: int = 7, compare_days: int = 7) -> pd.DataFrame:
    """
    Early-warning view: for each merchant, compares order count in the most
    recent `recent_days` window vs. the `compare_days` window before it.
    Status buckets:
      - New:       had 0 orders before, has orders now
      - Declining: had >=3 orders before, dropped by 50%+
      - Watch:     had >=3 orders before, dropped by 20-50%
      - Growing:   order count grew by 15%+
      - Healthy:   everything else (roughly stable)

    'Unknown' (no merchant name on record for that shipment) is excluded —
    it's not a real merchant that could be contacted about a "Declining"
    alert, so leaving it in would produce a meaningless alert with no
    actionable owner. Excluded count (within the combined comparison
    window) attached via .attrs.
    """
    if df.empty:
        result = pd.DataFrame(columns=["merchant_name", "recent_orders", "previous_orders", "change_pct", "status"])
        result.attrs["excluded_unknown_count"] = 0
        return result

    latest = df["delivery_date"].max()
    recent_start = latest - pd.Timedelta(days=recent_days)
    prev_start = recent_start - pd.Timedelta(days=compare_days)

    window = df[df["delivery_date"] >= prev_start]
    excluded_count = int((window["merchant_name"] == "Unknown").sum())
    known = df[df["merchant_name"] != "Unknown"]

    recent = known[known["delivery_date"] >= recent_start].groupby("merchant_name")["shipment_id"].count()
    previous = known[(known["delivery_date"] >= prev_start) & (known["delivery_date"] < recent_start)].groupby("merchant_name")["shipment_id"].count()

    merchants = sorted(set(recent.index) | set(previous.index))
    rows = []
    for m in merchants:
        r = int(recent.get(m, 0))
        p = int(previous.get(m, 0))
        change_pct = ((r - p) / p * 100) if p > 0 else (0.0 if r == 0 else 100.0)

        if p == 0 and r > 0:
            status = "New"
        elif p >= 3 and r < p * 0.5:
            status = "Declining"
        elif p >= 3 and r < p * 0.8:
            status = "Watch"
        elif change_pct >= 15:
            status = "Growing"
        else:
            status = "Healthy"

        rows.append({
            "merchant_name": m,
            "recent_orders": r,
            "previous_orders": p,
            "change_pct": round(change_pct, 1),
            "status": status,
        })

    result = pd.DataFrame(rows).sort_values("recent_orders", ascending=False)
    result.attrs["excluded_unknown_count"] = excluded_count
    return result


def merchant_leaderboard(df: pd.DataFrame, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    """
    Total orders + total revenue per merchant, all-time within the loaded
    window. 'Unknown' excluded for the same reason as merchant_activity
    above — it's a data-quality placeholder, not a merchant that can
    appear on a "top merchants" leaderboard. Excluded count via .attrs.

    ascending=True gives the LEAST active merchants (within the ones that
    placed at least one order in the loaded window). Note this is NOT the
    same as "merchants with zero orders" — a merchant who placed zero
    orders in this window never appears here at all, since this table is
    built purely from shipment data. See merchants_with_zero_orders() for
    that (needs the full merchant roster, a separate API call).
    """
    if df.empty:
        result = pd.DataFrame(columns=["merchant_name", "orders", "total_value"])
        result.attrs["excluded_unknown_count"] = 0
        return result
    excluded_count = int((df["merchant_name"] == "Unknown").sum())
    known = df[df["merchant_name"] != "Unknown"]
    result = (
        known.groupby("merchant_name")
        .agg(orders=("shipment_id", "count"), total_value=("amount", "sum"))
        .reset_index()
        .sort_values("orders", ascending=ascending)
        .head(n)
    )
    result.attrs["excluded_unknown_count"] = excluded_count
    return result


def merchants_with_zero_orders(full_roster_df: pd.DataFrame, shipments_df: pd.DataFrame) -> pd.DataFrame:
    """
    True 'inactive merchants' — registered with Weevo but with ZERO orders
    in the currently loaded window. This is different from
    merchant_leaderboard(ascending=True), which can only ever show
    merchants who placed at least one order (it's built from shipments,
    so a merchant with no orders never appears in it at all). Needs
    full_roster_df from fetch_merchants() (a separate API call, the full
    registered merchant list) — without it, "zero orders" merchants are
    invisible to every other function here.
    """
    if full_roster_df is None or full_roster_df.empty:
        return pd.DataFrame(columns=["merchant_name", "merchant_id"])
    name_col = "name" if "name" in full_roster_df.columns else None
    id_col = "id" if "id" in full_roster_df.columns else None
    if name_col is None or id_col is None:
        return pd.DataFrame(columns=["merchant_name", "merchant_id"])

    active_ids = set()
    if not shipments_df.empty and "merchant_id" in shipments_df.columns:
        active_ids = set(shipments_df["merchant_id"].dropna().unique())
    elif not shipments_df.empty and "merchant_name" in shipments_df.columns:
        # v1's DataFrame has no merchant_id — fall back to matching by name
        active_names = set(shipments_df["merchant_name"].dropna().unique())
        roster = full_roster_df.rename(columns={name_col: "merchant_name", id_col: "merchant_id"})
        return roster[~roster["merchant_name"].isin(active_names)][["merchant_name", "merchant_id"]].reset_index(drop=True)

    roster = full_roster_df.rename(columns={name_col: "merchant_name", id_col: "merchant_id"})
    return roster[~roster["merchant_id"].isin(active_ids)][["merchant_name", "merchant_id"]].reset_index(drop=True)


def recent_orders(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Most recent shipments, for a live-feed-style table."""
    if df.empty:
        return pd.DataFrame(columns=["shipment_id", "merchant_name", "courier_name", "area", "amount", "delivery_date"])
    return (
        df.sort_values("delivery_date", ascending=False)
        [["shipment_id", "merchant_name", "courier_name", "area", "amount", "delivery_date"]]
        .head(n)
        .reset_index(drop=True)
    )


def _authoritative_total_orders(df: pd.DataFrame) -> Optional[int]:
    """FINDING 5 FIX (2026-07-17): the admin API's own `statusCounts.Total`
    (confirmed in fetch_admin_shipments' docstring to match pagination.total
    exactly) reflects the ENTIRE filtered result set for the date range —
    unlike a local count/nunique over `df`, which only reflects however many
    rows actually made it into `df` (e.g. undercounts silently when a fetch
    was truncated by the time budget, see Known Bug 1). Returns None if not
    available, so callers can fall back to the previous local-count
    behaviour unchanged."""
    status_counts = df.attrs.get("status_counts") if hasattr(df, "attrs") else None
    if not status_counts:
        return None
    total = status_counts.get("Total")
    if total is None:
        return None
    try:
        return int(total)
    except (TypeError, ValueError):
        return None


def summary_kpis(df: pd.DataFrame) -> dict:
    api_total_orders = _authoritative_total_orders(df)
    if df.empty:
        return {"total_orders": api_total_orders if api_total_orders is not None else 0,
                "total_revenue": 0.0, "active_merchants": 0,
                "active_couriers": 0, "merchants_at_risk": 0}
    ma = merchant_activity(df)
    at_risk = int((ma["status"] == "Declining").sum()) if not ma.empty else 0
    # "Unassigned" / "Unknown" are placeholders for shipments with no real
    # courier/merchant on record yet — not counted as if they were one,
    # otherwise every KPI here is inflated by +1 the moment a single
    # shipment lacks that field (which is normal for e.g. "available" status).
    real_couriers = df.loc[df["courier_name"] != "Unassigned", "courier_name"]
    real_merchants = df.loc[df["merchant_name"] != "Unknown", "merchant_name"]
    return {
        "total_orders": api_total_orders if api_total_orders is not None else int(df["shipment_id"].nunique()),
        "total_revenue": float(df["amount"].sum()),
        "active_merchants": int(real_merchants.nunique()),
        "active_couriers": int(real_couriers.nunique()),
        "merchants_at_risk": at_risk,
    }


# ---------------------------------------------------------------------------
# Delivery-time metrics (ADDED — uses date_to_receive_shipment /
# date_to_deliver_shipment from the live API). Every function here degrades
# gracefully to an empty result when delivery_hours isn't available (e.g.
# db-cache source, or a demo run with no delivered orders yet) rather than
# raising, so nothing above breaks regardless of data source.
# ---------------------------------------------------------------------------
def _has_delivery_time_data(df: pd.DataFrame) -> bool:
    return (
        not df.empty
        and "delivery_hours" in df.columns
        and df["delivery_hours"].notna().any()
    )


def delivery_time_overview(df: pd.DataFrame, target_hours: float = 2.0) -> dict:
    """Headline delivery-time numbers, incl. how many deliveries met the
    2-hour promise — directly ties this dashboard back to Weevo's USP."""
    if not _has_delivery_time_data(df):
        return {"available": False, "avg_hours": None, "median_hours": None,
                "within_target_pct": None, "based_on_orders": 0}

    valid = df["delivery_hours"].dropna()
    within_target = (valid <= target_hours).sum()
    return {
        "available": True,
        "avg_hours": round(float(valid.mean()), 2),
        "median_hours": round(float(valid.median()), 2),
        "within_target_pct": round(float(within_target) / len(valid) * 100, 1),
        "based_on_orders": int(len(valid)),
    }


def delivery_time_by_courier(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Avg/median delivery time per courier, delivered orders only."""
    if not _has_delivery_time_data(df):
        return pd.DataFrame(columns=["courier_name", "avg_hours", "median_hours", "delivered_orders"])
    valid = df.dropna(subset=["delivery_hours"])
    result = (
        valid.groupby("courier_name")["delivery_hours"]
        .agg(avg_hours="mean", median_hours="median", delivered_orders="count")
        .reset_index()
    )
    result["avg_hours"] = result["avg_hours"].round(2)
    result["median_hours"] = result["median_hours"].round(2)
    return result.sort_values("avg_hours").head(n)


def delivery_time_by_area(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Avg delivery time per area — flags which areas are structurally slow
    regardless of which courier is assigned there. 'Unknown' excluded —
    same reasoning as top_areas: it's not a real place to act on.
    Excluded count (among delivered orders with timing data) via .attrs.
    """
    if not _has_delivery_time_data(df):
        result = pd.DataFrame(columns=["area", "avg_hours", "delivered_orders"])
        result.attrs["excluded_unknown_count"] = 0
        return result
    valid = df.dropna(subset=["delivery_hours"])
    excluded_count = int((valid["area"] == "Unknown").sum())
    known = valid[valid["area"] != "Unknown"]
    result = (
        known.groupby("area")["delivery_hours"]
        .agg(avg_hours="mean", delivered_orders="count")
        .reset_index()
    )
    result["avg_hours"] = result["avg_hours"].round(2)
    result = result.sort_values("avg_hours", ascending=False).head(n)
    result.attrs["excluded_unknown_count"] = excluded_count
    return result


# ---------------------------------------------------------------------------
# Snapshot archive (ADDED) — lets end-of-day/week/month snapshots accumulate
# into real historical data on disk, instead of only ever seeing whatever
# window the Live API currently returns. Pure pandas + stdlib csv/os only —
# no new dependency.
# ---------------------------------------------------------------------------
def _empty_archive_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=ARCHIVE_COLUMNS)
    for col in ("delivery_date", "received_at", "delivered_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("amount", "attempts", "delivery_hours"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_archive_file_info(archive_path: str = DEFAULT_ARCHIVE_PATH) -> dict:
    """Reports the archive file's actual state on disk right now — the
    resolved absolute path, whether it exists, when it was last modified,
    and its size. Built specifically to diagnose 'the archive keeps
    resetting' symptoms: if this info changes unexpectedly between two
    page loads (different absolute path, or last-modified time doesn't
    reflect a recent save), that's direct on-screen evidence the file
    isn't persisting on the server between requests — a hosting/storage
    issue, not a bug in the save logic itself (verified separately: saving
    the same data twice in a row correctly reports '0 added, N updated'
    when the file DOES persist)."""
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
        "row_count": len(load_archive(archive_path)),
    }


def load_archive(archive_path: str = DEFAULT_ARCHIVE_PATH) -> pd.DataFrame:
    """Reads whatever has been saved so far. Returns an empty (but
    correctly-shaped) DataFrame if nothing has been archived yet — this is
    the normal first-run state, not an error."""
    if not os.path.exists(archive_path):
        return _empty_archive_df()

    df = pd.read_csv(archive_path)
    for col in ("delivery_date", "received_at", "delivered_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("amount", "attempts", "delivery_hours"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Any archive column missing from an older file version is added back
    # empty, so downstream aggregation functions never hit a KeyError.
    for col in ARCHIVE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[ARCHIVE_COLUMNS]


def append_to_archive(df: pd.DataFrame, archive_path: str = DEFAULT_ARCHIVE_PATH) -> dict:
    """Merges `df` into the on-disk archive. Shipments already in the
    archive get refreshed (e.g. status changed since last snapshot) rather
    than duplicated — matched on shipment_id, newest write wins. Returns a
    small summary dict instead of raising, so a save action always gives
    the caller something clear to show."""
    if df is None or df.empty:
        return {"added": 0, "updated": 0, "total_in_archive": len(load_archive(archive_path))}

    dir_name = os.path.dirname(archive_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    existing = load_archive(archive_path)
    incoming = df.reindex(columns=ARCHIVE_COLUMNS)

    existing_ids = set(existing["shipment_id"].dropna()) if not existing.empty else set()
    incoming_ids = set(incoming["shipment_id"].dropna())
    added_count = len(incoming_ids - existing_ids)
    updated_count = len(incoming_ids & existing_ids)

    frames = [f for f in (existing, incoming) if not f.empty and not f.isna().all(axis=None)]
    combined = pd.concat(frames, ignore_index=True) if frames else incoming
    combined = combined.drop_duplicates(subset=["shipment_id"], keep="last")
    # Sorted chronologically purely for readability if someone opens the
    # raw CSV directly — has zero effect on any dashboard calculation,
    # since every aggregation here groups/resamples rather than relying
    # on row order.
    if "delivery_date" in combined.columns:
        combined = combined.sort_values("delivery_date", na_position="last").reset_index(drop=True)
    combined.to_csv(archive_path, index=False)

    return {"added": added_count, "updated": updated_count, "total_in_archive": len(combined)}


def detect_archive_gap(archive_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
    """
    Answers: "did too much time pass between snapshot saves such that some
    shipments scrolled out of the Live API's 'most recent N' window before
    ever being captured?" — those shipments are permanently lost to the
    archive once that happens (there's no way to re-fetch a past window
    from this API, as established earlier).

    Compares the newest delivery_date already in the archive against the
    oldest delivery_date currently visible in a fresh live pull:
      - archive_latest >= live_oldest  -> full coverage (normal, even with
        some re-covered overlap — dedup on shipment_id handles that safely)
      - archive_latest <  live_oldest  -> a real gap: nothing was ever
        saved for the window between those two points in time

    Returns a plain dict, never raises — always safe to call even with an
    empty archive (first-ever save) or empty live data (fetch failed).
    """
    if archive_df is None or archive_df.empty or "delivery_date" not in archive_df.columns:
        return {"has_gap": False, "gap_hours": None, "archive_latest": None,
                "live_oldest": None, "reason": "no_archive_yet"}
    if live_df is None or live_df.empty or "delivery_date" not in live_df.columns:
        return {"has_gap": False, "gap_hours": None, "archive_latest": None,
                "live_oldest": None, "reason": "no_live_data"}

    archive_latest = pd.to_datetime(archive_df["delivery_date"], errors="coerce").max()
    live_oldest = pd.to_datetime(live_df["delivery_date"], errors="coerce").min()

    if pd.isna(archive_latest) or pd.isna(live_oldest):
        return {"has_gap": False, "gap_hours": None, "archive_latest": None,
                "live_oldest": None, "reason": "missing_dates"}

    gap_hours = (live_oldest - archive_latest).total_seconds() / 3600
    has_gap = gap_hours > 0.05  # ignore sub-3-minute float/measurement noise around zero

    return {
        "has_gap": bool(has_gap),
        "gap_hours": round(gap_hours, 1) if has_gap else 0.0,
        "archive_latest": archive_latest,
        "live_oldest": live_oldest,
        "reason": "gap_detected" if has_gap else "no_gap",
    }


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Parses a snapshot CSV a user uploads (normally one previously
    produced by this same dashboard's 'Download snapshot' button, so the
    shape is guaranteed to match). Raises ValueError with a clear message
    on a genuinely incompatible file, rather than a confusing pandas
    traceback — the page catches this and shows it nicely."""
    required = {"shipment_id", "delivery_date", "amount", "merchant_name", "courier_name", "area"}
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Couldn't read this file as CSV: {e}")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "This file doesn't look like a Weevo analytics snapshot — missing columns: "
            + ", ".join(sorted(missing))
            + ". Use the 'Download snapshot' button on this dashboard to export a "
              "compatible file, then re-upload that."
        )

    for col in ARCHIVE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[ARCHIVE_COLUMNS]
    for col in ("delivery_date", "received_at", "delivered_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("amount", "attempts", "delivery_hours"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
