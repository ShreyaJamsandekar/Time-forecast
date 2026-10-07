#!/usr/bin/env python3
"""
customer_material_reconciled_forecast_v3.py  (v3)
=================================================

Customer/material daily forecaster with holiday-aware seasonality and a
backlog-consistent reconciliation against an internally computed
company-wide forecast. Standalone: no Streamlit, no UI.

WHAT v3 ADDS ON TOP OF v2
-------------------------
 A. WEEKDAY SHAPING. AutoETS/AutoARIMA on zero-filled daily series often return
    a nearly FLAT daily forecast (e.g. 16,115 lb every day, weekends included),
    while the company-wide model has a real weekly rhythm. Reconciling daily
    then "fixes" the mismatch by pushing never-ship days to 0 and weekdays up
    by 100-200%. v3 keeps each combo's model WEEKLY TOTAL but spreads it over
    the days using that combo's own weekday averages (last 26 weeks).
 B. WEEKLY RECONCILIATION. The gap is now computed and allocated per
    Sunday-Saturday week (overall weekly estimate vs. sum of combo weekly
    estimates), then each combo's weekly adjustment is spread over its days in
    proportion to its own daily model shape. Daily shape comes from the combos;
    only weekly totals are forced to agree with the overall model.
 C. New sheet "Reconciliation (Weekly)". Disable A with --no-weekday-shaping.

WHAT CHANGED IN v2 VS. v1 (and why)
-----------------------------------
 1. RECONCILIATION NOW HAPPENS ON THE MODEL LAYER, NOT ON backlog-floored TOTALS.
    v1 compared  sum_i max(model_i, backlog_i)  (per-combo floors)  against
    max(overall_model, overall_backlog)  (one floor). Because max() is convex,
    the first number is always >= the second wherever backlog is dense, which
    produced large downward "reconciliation" in the first weeks that faded to
    zero as backlog ran out. Now:
        a) every combo's model estimate m_i (base model x holiday factor) is
           reconciled so that  sum_i m_i  ==  overall model estimate M;
        b) THEN each combo is floored at its confirmed backlog:
           final_i = max(m_i', backlog_i)  =  backlog_i + additional_i.
    Whatever excess the floors create (confirmed orders that beat a combo's
    model) is real demand. It is REPORTED ("Backlog-Driven Excess") and never
    pushed onto other combos.
 2. HOLIDAY / CALENDAR SEASONALITY replaces the month-of-year index. Effects
    are learned as Sunday-Saturday week offsets around each holiday
    (Thanksgiving, Christmas, New Year, Super Bowl, Easter, Memorial Day,
    July 4th, Labor Day), e.g. "Thanksgiving wk-2". They are estimated on
    detrended history (iteratively, against a centered 91-day baseline),
    learned hierarchically (company -> material -> customer/material combo,
    each shrunk toward its parent by how much evidence it has), and applied by
    fitting the base model on holiday-adjusted history and re-applying the
    factor to the forecast.
 3. RECONCILIATION ALLOCATION is weighted by forecast uncertainty (variance of
    the same weekday over the last 26 weeks, i.e. the minimum-variance split)
    and bounded by each combo's own recent range (p95 of the same weekday,
    widened by the holiday factor), instead of recent-average weights bounded
    by the all-time maximum.
 4. DATE ALIGNMENT BUG FIXED: statsforecast always forecasts starting at
    as_of+1. If the forecast window started later (--align-to-sunday,
    --forecast-start-date) v1 mislabeled the values. Forecasts are now indexed
    by their true dates.
 5. DIAGNOSTICS: "Reconciliation Summary" sheet (per-date gap before/after,
    backlog share, backlog-driven excess), "Holiday Effects" sheet (learned
    factors), and an optional holdout backtest (--backtest) that compares the
    model with and without holiday effects.
 6. Identifier cleaning collapses repeated whitespace
    ("WA  171833" -> "WA 171833").

CLASSIFICATION -> MODEL (unchanged)
-----------------------------------
    Regular      -> AutoETS         dense, long history
    Intermittent -> AutoARIMA       sparse but still active
    New          -> Seasonal Naive  too little history
    Dormant      -> no model        forecast = confirmed backlog only
A missing order date is ZERO demand (zero-filled, never interpolated).

OUTPUT (one Excel workbook)
---------------------------
    Forecast                   final Net Pounds per combo per date
    Overall Forecast (Weekly)  company-wide, Sunday-Saturday weeks
    Reconciliation Summary     per-date diagnostics
    Reasoning                  how each number was built, per combo per date
    Holiday Effects            learned factors (company + each material)
    Reconciliation (Weekly)    where reconciliation actually happens (per week)
    Backtest                   only with --backtest

INSTALL
-------
    pip install pandas numpy openpyxl xlsxwriter statsforecast

RUN
---
    python customer_material_reconciled_forecast_v3.py --workbook "Sales Order 2024-26.xlsx"
    (see --help; useful: --as-of, --horizon, --limit, --backtest, --no-reasoning)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dateutil.easter import easter as _easter

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("customer_material_reconciled_forecast_v3")

# =============================================================================
# CONFIG
# =============================================================================

RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)

DATE_COLUMN_NAME: str = "Mat.avail.dt"
POUNDS_COLUMN_NAME: str = "Net Pounds"
CUSTOMER_COLUMN_ALIASES: List[str] = ["Customer Name"]
MATERIAL_NAME_COLUMN_ALIASES: List[str] = ["ReMat Name"]
MATERIAL_NO_COLUMN_ALIASES: List[str] = ["ReMat No"]
PLANT_COLUMN_ALIASES: List[str] = ["Plant", "Ship Pt"]

DEFAULT_HORIZON_DAYS: int = 21
SEASONAL_PERIOD: int = 7

# --- classification thresholds (CLI-overridable) ---------------------------
MIN_FIT_DAYS: int = 14
DORMANT_INACTIVITY_DAYS: int = 60
NEW_MIN_HISTORY_DAYS: int = 90
REGULAR_DENSITY_THRESHOLD: float = 0.4
DENSITY_WINDOW_DAYS: int = 90
RECENT_AVG_WINDOW_DAYS: int = 28

MODEL_REGULAR = "AutoETS"
MODEL_INTERMITTENT = "AutoARIMA"
MODEL_NEW = "Seasonal Naive"
CLASS_REGULAR = "Regular"
CLASS_INTERMITTENT = "Intermittent"
CLASS_NEW = "New"
CLASS_DORMANT = "Dormant"


# --- holiday calendar -------------------------------------------------------
# Each holiday is a Sunday-Saturday "week 0" (the week containing the holiday)
# plus N weeks before and M weeks after. Orders are keyed on material
# availability date, so ramps start well before the holiday itself.
# To add a holiday: add a date function below and a row to HOLIDAYS.

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(weekday - first.dayofweek) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    last = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.dayofweek - weekday) % 7)


HOLIDAYS: Dict[str, Tuple[Callable[[int], pd.Timestamp], int, int]] = {
    #  name            date function(year)                          wks before, wks after
    "Thanksgiving": (lambda y: _nth_weekday(y, 11, 3, 4),                    4, 1),
    "Christmas":    (lambda y: pd.Timestamp(y, 12, 25),                      2, 0),
    "New Year":     (lambda y: pd.Timestamp(y, 1, 1),                        0, 1),
    "Super Bowl":   (lambda y: _nth_weekday(y, 2, 6, 2),                     3, 1),  # 2nd Sunday of Feb (2022+)
    "Easter":       (lambda y: pd.Timestamp(_easter(y)),                     3, 1),
    "Memorial Day": (lambda y: _last_weekday(y, 5, 0),                       2, 1),
    "July 4th":     (lambda y: pd.Timestamp(y, 7, 4),                        2, 1),
    "Labor Day":    (lambda y: _nth_weekday(y, 9, 0, 1),                     2, 1),
}

HOLIDAY_BASELINE_WINDOW: int = 91        # centered-mean baseline used to detrend
HOLIDAY_BASELINE_MIN_DAYS: int = 28
HOLIDAY_PASSES: int = 3                  # baseline <-> factor alternation passes
HOLIDAY_FULL_EVIDENCE_NZ_DAYS: int = 3   # a block with >= this many order days counts as 1 full observation
HOLIDAY_FACTOR_CLIP: Tuple[float, float] = (0.3, 3.0)
# Credibility = n_obs / (n_obs + K): how fast each level trusts its own data over its parent's.
HOLIDAY_K_OVERALL: float = 0.5
HOLIDAY_K_MATERIAL: float = 1.5
HOLIDAY_K_COMBO: float = 4.0

# --- reconciliation ---------------------------------------------------------
RECONCILE_TOLERANCE_LBS: float = 1.0
RECONCILE_MAX_ITER: int = 25
RECON_STATS_WINDOW_DAYS: int = 182       # same-weekday stats (std, p95) window
RECON_CAP_PCTL: float = 95.0
RECON_MAX_CUT_FRAC: float = 0.5          # a combo's model estimate can be cut by at most this fraction
RECON_WEIGHT_POWER: float = 2.0          # weight = std ** power (2 = minimum-variance split)


@dataclass
class Config:
    fast_arima: bool = True
    n_jobs: int = -1
    dormant_inactivity_days: int = DORMANT_INACTIVITY_DAYS
    new_min_history_days: int = NEW_MIN_HISTORY_DAYS
    regular_density_threshold: float = REGULAR_DENSITY_THRESHOLD
    holiday_k_combo: float = HOLIDAY_K_COMBO
    max_cut_frac: float = RECON_MAX_CUT_FRAC
    chunk_size: int = 5000
    weekday_shaping: bool = True


# =============================================================================
# COLUMN / SHEET NAME NORMALIZATION + LOADING
# =============================================================================

def _normalize_token(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


_NORM_DATE_COL = _normalize_token(DATE_COLUMN_NAME)
_NORM_POUNDS_COL = _normalize_token(POUNDS_COLUMN_NAME)
_NORM_CUSTOMER_ALIASES = {_normalize_token(a) for a in CUSTOMER_COLUMN_ALIASES}
_NORM_MATERIAL_NAME_ALIASES = {_normalize_token(a) for a in MATERIAL_NAME_COLUMN_ALIASES}
_NORM_MATERIAL_NO_ALIASES = {_normalize_token(a) for a in MATERIAL_NO_COLUMN_ALIASES}
_NORM_PLANT_ALIASES = {_normalize_token(a) for a in PLANT_COLUMN_ALIASES}
_HALF_YEAR_SHEET_RE = re.compile(r"^(\d{2})h(?:y)?([12])$")


def is_half_year_sheet(sheet_name: str) -> bool:
    return _HALF_YEAR_SHEET_RE.match(_normalize_token(sheet_name)) is not None


def half_year_sort_key(sheet_name: str) -> Tuple[int, int]:
    m = _HALF_YEAR_SHEET_RE.match(_normalize_token(sheet_name))
    return (int(m.group(1)), int(m.group(2))) if m else (99, 9)


def find_target_columns(columns: List[object]) -> Tuple[Optional[object], Optional[object]]:
    date_col, pounds_col = None, None
    for c in columns:
        norm = _normalize_token(c)
        if norm == _NORM_DATE_COL and date_col is None:
            date_col = c
        elif norm == _NORM_POUNDS_COL and pounds_col is None:
            pounds_col = c
    return date_col, pounds_col


def find_segment_columns(columns: List[object]):
    """Returns (customer_col, material_name_col, material_no_col, plant_col)."""
    customer_col, material_name_col, material_no_col, plant_col = None, None, None, None
    for c in columns:
        norm = _normalize_token(c)
        if norm in _NORM_CUSTOMER_ALIASES and customer_col is None:
            customer_col = c
        elif norm in _NORM_MATERIAL_NAME_ALIASES and material_name_col is None:
            material_name_col = c
        elif norm in _NORM_MATERIAL_NO_ALIASES and material_no_col is None:
            material_no_col = c
        elif norm in _NORM_PLANT_ALIASES and plant_col is None:
            plant_col = c
    return customer_col, material_name_col, material_no_col, plant_col


_NUMERIC_STRIP_RE = re.compile(r"[^0-9.\-]")


def clean_net_pounds_value(raw: object) -> Optional[float]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        val = float(raw)
        return None if np.isnan(val) else val
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "n/a", "na", "-"}:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    s_clean = _NUMERIC_STRIP_RE.sub("", s.replace(",", "").replace(" ", ""))
    if s_clean in {"", "-", "."} or s_clean.count(".") > 1 or s_clean.count("-") > 1:
        return None
    try:
        val = float(s_clean)
    except ValueError:
        return None
    return -val if negative else val


def clean_id_text(raw: object) -> Optional[str]:
    """Normalize an identifier cell (Customer Name, Material No, Plant) to a
    stable string. Guards against the pandas/Excel gotcha where an integer-
    like code becomes float64 ("38" vs "38.0") when any cell in the column is
    blank, which would silently split one combination into two keys. Also
    collapses repeated whitespace ("WA  171833" -> "WA 171833")."""
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    if isinstance(raw, (int, np.integer)):
        return str(int(raw))
    if isinstance(raw, (float, np.floating)):
        return str(int(raw)) if float(raw).is_integer() else str(raw)
    s = re.sub(r"\s+", " ", str(raw)).strip()
    return s if s else None


def load_and_clean_workbook(path: Union[str, Path]) -> pd.DataFrame:
    """Read every half-year sheet, clean it, and return one combined frame
    with standardized columns: date, net_pounds, customer, material_no,
    material_name (+ plant if present). Duplicate detection compares the FULL
    original row so two distinct orders sharing date + pounds are kept."""
    raw_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    matched = {name: df for name, df in raw_sheets.items() if is_half_year_sheet(name)}
    matched = dict(sorted(matched.items(), key=lambda kv: half_year_sort_key(kv[0])))
    if not matched:
        raise ValueError(
            f"No sheets matching the half-year naming convention (e.g. '24HY1', '24H2') found. "
            f"Sheets present: {list(raw_sheets.keys())}"
        )
    log.info("Matched %d half-year sheet(s): %s", len(matched), list(matched.keys()))

    cleaned_frames = []
    total_loaded = total_invalid_date = total_invalid_pounds = 0
    for sheet_name, df in matched.items():
        date_col, pounds_col = find_target_columns(list(df.columns))
        if date_col is None or pounds_col is None:
            log.warning("Sheet '%s': could not find date/Net Pounds columns -- skipped.", sheet_name)
            continue
        working = df.copy()
        working["__date__"] = pd.to_datetime(working[date_col], errors="coerce")
        working["__net_pounds__"] = working[pounds_col].apply(clean_net_pounds_value)
        invalid_date = working["__date__"].isna()
        invalid_pounds = working["__net_pounds__"].isna()
        total_loaded += len(working)
        total_invalid_date += int(invalid_date.sum())
        total_invalid_pounds += int(invalid_pounds.sum())

        clean = working.loc[~(invalid_date | invalid_pounds)].copy()
        clean["date"] = clean["__date__"].dt.normalize()
        clean["net_pounds"] = clean["__net_pounds__"]
        clean = clean.drop(columns=["__date__", "__net_pounds__"])

        customer_col, material_name_col, material_no_col, plant_col = find_segment_columns(list(df.columns))
        if customer_col is not None:
            clean["customer"] = clean[customer_col].apply(clean_id_text)
        if material_name_col is not None:
            clean["material_name"] = clean[material_name_col].apply(clean_id_text)
        if material_no_col is not None:
            clean["material_no"] = clean[material_no_col].apply(clean_id_text)
        if plant_col is not None:
            clean["plant"] = clean[plant_col].apply(clean_id_text)
        clean["__sheet__"] = sheet_name
        cleaned_frames.append(clean)

    combined = pd.concat(cleaned_frames, ignore_index=True, sort=False) if cleaned_frames else pd.DataFrame()
    if combined.empty:
        raise ValueError("No valid rows survived cleaning across all sheets.")

    dedup_cols = [c for c in combined.columns if c != "__sheet__"]
    dup_mask = combined.duplicated(subset=dedup_cols, keep="first")
    dup_count = int(dup_mask.sum())
    combined = combined.loc[~dup_mask].copy()

    for col in ("customer", "material_no", "material_name", "plant"):
        if col in combined.columns:
            combined.loc[combined[col].astype(str).str.len() == 0, col] = np.nan
    if "plant" in combined.columns and combined["plant"].isna().all():
        combined = combined.drop(columns=["plant"])

    log.info(
        "Cleaning summary: %d rows loaded, %d invalid dates, %d invalid Net Pounds, "
        "%d exact-duplicate rows removed, %d valid rows remain.",
        total_loaded, total_invalid_date, total_invalid_pounds, dup_count, len(combined),
    )
    if "customer" not in combined.columns or "material_no" not in combined.columns:
        raise ValueError(
            "No customer / material number columns were recognized in this workbook. "
            "Add your exact header text to CUSTOMER_COLUMN_ALIASES / MATERIAL_NO_COLUMN_ALIASES "
            "near the top of this file."
        )
    if "material_name" not in combined.columns:
        combined["material_name"] = combined["material_no"]
    if "plant" in combined.columns:
        log.info("Plant/Ship Pt column found -- combination key is Plant + Customer + Material No.")
    else:
        log.info("No Plant/Ship Pt column found -- combination key is Customer + Material No.")
    return combined


# =============================================================================
# DATE HELPERS
# =============================================================================

def next_sunday_on_or_after(d: pd.Timestamp) -> pd.Timestamp:
    return d + pd.Timedelta(days=(6 - d.dayofweek) % 7)


def resolve_forecast_start(as_of: pd.Timestamp, align_to_sunday: bool) -> pd.Timestamp:
    tomorrow = as_of + pd.Timedelta(days=1)
    return next_sunday_on_or_after(tomorrow) if align_to_sunday else tomorrow


def sunday_start_dow(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.dayofweek + 1) % 7  # 0=Sunday ... 6=Saturday


# =============================================================================
# DENSE SERIES (zero-filled history + future backlog)
# =============================================================================

@dataclass
class SegmentSeries:
    customer: str
    material_no: str
    material_name: str
    dense_hist: pd.Series            # daily, zero-filled, first_order_date .. as_of
    backlog: pd.Series               # daily, zero-filled, as_of+1 .. last known date
    first_order_date: pd.Timestamp
    last_nonzero_date: Optional[pd.Timestamp]
    plant: str = ""


def build_segment_series(raw: pd.Series, as_of: pd.Timestamp) -> Optional[SegmentSeries]:
    """raw: sparse (date -> summed net_pounds) series for one combination."""
    if raw.empty:
        return None
    raw = raw.sort_index()
    first_date = raw.index.min()
    last_date = raw.index.max()
    full_index = pd.date_range(first_date, max(last_date, as_of), freq="D")
    dense = raw.reindex(full_index, fill_value=0.0)
    hist = dense.loc[dense.index <= as_of]
    if hist.empty:
        return None
    backlog = dense.loc[dense.index > as_of]
    nonzero_hist = hist.loc[hist > 0]
    last_nonzero = nonzero_hist.index.max() if not nonzero_hist.empty else None
    return SegmentSeries(
        customer="", material_no="", material_name="",
        dense_hist=hist, backlog=backlog,
        first_order_date=first_date, last_nonzero_date=last_nonzero,
    )


# =============================================================================
# HOLIDAY CALENDAR + HIERARCHICAL HOLIDAY-EFFECT ESTIMATION
# =============================================================================
#
# A "slot" is (holiday, week offset), e.g. "Thanksgiving wk-2". A "block" is
# one slot in one specific year. Every calendar day belongs to at most one
# block: when windows overlap (e.g. Thanksgiving wk+1 and Christmas wk-3) the
# day goes to the holiday it is closest to.
#
# Estimation for one series y (daily, zero-filled):
#   repeat HOLIDAY_PASSES times:
#       z        = y / factor                       (holiday-adjusted)
#       baseline = centered 91-day mean of z        (detrended level)
#       ratio    = sum(y in slot's blocks) / sum(baseline in those blocks)
#       factor   = credibility * ratio + (1 - credibility) * parent_factor
# credibility = n_obs / (n_obs + K); n_obs counts observed blocks, each
# weighted by how many order days it contained (a block with no orders adds
# no evidence). Only fully observed blocks are used. Parent = material level
# (for a combo), company level (for a material), 1.0 (for the company).

@dataclass
class HolidayCalendar:
    start: pd.Timestamp
    n_days: int
    slot_of: np.ndarray
    block_of: np.ndarray
    slot_names: List[str]
    block_slot: np.ndarray
    block_first: np.ndarray
    block_last: np.ndarray
    block_len: np.ndarray

    @property
    def n_slots(self) -> int:
        return len(self.slot_names)

    @property
    def n_blocks(self) -> int:
        return len(self.block_slot)

    def pos(self, ts: pd.Timestamp) -> int:
        return int((ts - self.start).days)


def build_holiday_calendar(start: pd.Timestamp, end: pd.Timestamp) -> HolidayCalendar:
    n_days = int((end - start).days) + 1
    slot_names: List[str] = []
    slot_index: Dict[Tuple[str, int], int] = {}
    for name, (_, before, after) in HOLIDAYS.items():
        for w in range(-before, after + 1):
            slot_index[(name, w)] = len(slot_names)
            slot_names.append(f"{name} wk0" if w == 0 else f"{name} wk{w:+d}")

    best: Dict[int, Tuple[int, int, Tuple[str, int, int]]] = {}
    for year in range(start.year - 1, end.year + 2):
        for name, (fn, before, after) in HOLIDAYS.items():
            hd = fn(year)
            week0 = hd - pd.Timedelta(days=int((hd.dayofweek + 1) % 7))  # Sunday of the holiday's week
            for w in range(-before, after + 1):
                wstart = week0 + pd.Timedelta(days=7 * w)
                for k in range(7):
                    day = wstart + pd.Timedelta(days=k)
                    p = int((day - start).days)
                    if p < 0 or p >= n_days:
                        continue
                    dist = abs(int((day - hd).days))
                    cur = best.get(p)
                    if cur is None or dist < cur[0]:
                        best[p] = (dist, slot_index[(name, w)], (name, year, w))

    slot_of = np.full(n_days, -1, dtype=np.int64)
    block_of = np.full(n_days, -1, dtype=np.int64)
    block_ids: Dict[Tuple[str, int, int], int] = {}
    block_slot_l: List[int] = []
    for p in sorted(best):
        _, slot, bkey = best[p]
        if bkey not in block_ids:
            block_ids[bkey] = len(block_slot_l)
            block_slot_l.append(slot)
        slot_of[p] = slot
        block_of[p] = block_ids[bkey]

    n_blocks = len(block_slot_l)
    block_first = np.full(n_blocks, 10 ** 9, dtype=np.int64)
    block_last = np.full(n_blocks, -1, dtype=np.int64)
    block_len = np.zeros(n_blocks, dtype=np.int64)
    for p in range(n_days):
        b = block_of[p]
        if b >= 0:
            block_first[b] = min(block_first[b], p)
            block_last[b] = max(block_last[b], p)
            block_len[b] += 1
    return HolidayCalendar(start, n_days, slot_of, block_of, slot_names,
                           np.array(block_slot_l, dtype=np.int64), block_first, block_last, block_len)


def centered_mean(x: np.ndarray, window: int, min_count: int) -> np.ndarray:
    n = len(x)
    half = window // 2
    c = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half + 1, n)
    cnt = hi - lo
    s = c[hi] - c[lo]
    return np.where(cnt >= min_count, s / np.maximum(cnt, 1), np.nan)


def factor_series(profile: np.ndarray, slots: np.ndarray) -> np.ndarray:
    return np.where(slots >= 0, profile[np.maximum(slots, 0)], 1.0)


def estimate_holiday_profile(
    y: np.ndarray, p0: int, cal: HolidayCalendar, parent: np.ndarray, k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (profile[S], n_obs[S]). With no usable evidence the profile is
    simply the parent's."""
    n = len(y)
    S, B = cal.n_slots, cal.n_blocks
    blk = cal.block_of[p0:p0 + n]
    slot = cal.slot_of[p0:p0 + n]
    zeros_s = np.zeros(S)
    in_blk = blk >= 0
    if not in_blk.any():
        return parent.copy(), zeros_s
    safe_blk = np.where(in_blk, blk, 0)
    usable_block = (cal.block_first >= p0) & (cal.block_last <= p0 + n - 1)
    usable_day = in_blk & usable_block[safe_blk]
    if not usable_day.any():
        return parent.copy(), zeros_s

    lo, hi = HOLIDAY_FACTOR_CLIP
    profile = parent.copy()
    n_obs = zeros_s
    f = factor_series(profile, slot)
    for _ in range(HOLIDAY_PASSES):
        z = y / f
        base = np.nan_to_num(centered_mean(z, HOLIDAY_BASELINE_WINDOW, HOLIDAY_BASELINE_MIN_DAYS), nan=0.0)
        ok = usable_day & (base > 0)
        bb = blk[ok]
        sum_y = np.bincount(bb, weights=y[ok], minlength=B)
        sum_b = np.bincount(bb, weights=base[ok], minlength=B)
        nz = np.bincount(bb, weights=(y[ok] > 0).astype(float), minlength=B)
        cnt = np.bincount(bb, minlength=B)
        block_ok = (cnt == cal.block_len) & (sum_b > 0)
        if not block_ok.any():
            return parent.copy(), zeros_s
        bs = cal.block_slot[block_ok]
        slot_y = np.bincount(bs, weights=sum_y[block_ok], minlength=S)
        slot_b = np.bincount(bs, weights=sum_b[block_ok], minlength=S)
        n_obs = np.bincount(bs, weights=np.minimum(1.0, nz[block_ok] / HOLIDAY_FULL_EVIDENCE_NZ_DAYS), minlength=S)
        has = slot_b > 0
        ratio = np.where(has, slot_y / np.where(has, slot_b, 1.0), 1.0)
        cred = n_obs / (n_obs + k)
        profile = np.where(has, cred * ratio + (1.0 - cred) * parent, parent)
        profile = np.clip(profile, lo, hi)
        f = factor_series(profile, slot)
    return profile, n_obs


# =============================================================================
# CLASSIFICATION -> MODEL ASSIGNMENT
# =============================================================================

@dataclass
class SegmentProfile:
    classification: str
    model: Optional[str]
    days_since_last_nonzero: Optional[int]
    history_span_days: int
    recent_avg: float
    hist_min: float
    hist_max: float
    dow_std: np.ndarray        # (7,) std of same-weekday values, last RECON_STATS_WINDOW_DAYS (0=Sunday)
    dow_cap: np.ndarray        # (7,) p95 of same-weekday values, same window
    dow_mean: np.ndarray       # (7,) mean of same-weekday values (zeros included), same window
    holiday_profile: Optional[np.ndarray] = None
    holiday_n_obs: Optional[np.ndarray] = None


def classify_segment(seg: SegmentSeries, as_of: pd.Timestamp, cfg: Config) -> SegmentProfile:
    hist = seg.dense_hist
    history_span_days = int((as_of - seg.first_order_date).days) + 1
    days_since = (int((as_of - seg.last_nonzero_date).days) if seg.last_nonzero_date is not None
                  else history_span_days)
    recent = hist.tail(RECENT_AVG_WINDOW_DAYS)
    recent_avg = float(recent.mean()) if not recent.empty else 0.0

    tail = hist.tail(RECON_STATS_WINDOW_DAYS)
    tail_dow = sunday_start_dow(tail.index)
    tail_vals = tail.to_numpy()
    dow_std = np.zeros(7)
    dow_cap = np.zeros(7)
    dow_mean = np.zeros(7)
    for d in range(7):
        v = tail_vals[tail_dow == d]
        if len(v) >= 1:
            dow_mean[d] = float(v.mean())
        if len(v) >= 2:
            dow_std[d] = float(v.std(ddof=1))
            dow_cap[d] = float(np.percentile(v, RECON_CAP_PCTL))
        elif len(v) == 1:
            dow_cap[d] = float(v[0])

    def make(cls: str, model: Optional[str]) -> SegmentProfile:
        return SegmentProfile(cls, model, days_since, history_span_days, recent_avg,
                              float(hist.min()), float(hist.max()), dow_std, dow_cap, dow_mean)

    if history_span_days < MIN_FIT_DAYS or days_since > cfg.dormant_inactivity_days:
        return make(CLASS_DORMANT, None)
    if history_span_days < cfg.new_min_history_days:
        return make(CLASS_NEW, MODEL_NEW)
    window = hist.tail(DENSITY_WINDOW_DAYS)
    density = float((window > 0).mean()) if not window.empty else 0.0
    if density >= cfg.regular_density_threshold:
        return make(CLASS_REGULAR, MODEL_REGULAR)
    return make(CLASS_INTERMITTENT, MODEL_INTERMITTENT)


# =============================================================================
# MODEL FITTING (vectorized per model group, chunked)
# =============================================================================

def _build_statsforecast_model(model_name: str, fast_arima: bool):
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
    if model_name == MODEL_REGULAR:
        return AutoETS(season_length=SEASONAL_PERIOD)
    if model_name == MODEL_INTERMITTENT:
        kwargs = dict(season_length=SEASONAL_PERIOD)
        if fast_arima:
            kwargs.update(approximation=True, max_order=3, max_p=2, max_q=2, max_P=1, max_Q=1)
        return AutoARIMA(**kwargs)
    if model_name == MODEL_NEW:
        return SeasonalNaive(season_length=SEASONAL_PERIOD)
    raise ValueError(f"Unknown model: {model_name}")


_MODEL_OUTPUT_COL = {MODEL_REGULAR: "AutoETS", MODEL_INTERMITTENT: "AutoARIMA", MODEL_NEW: "SeasonalNaive"}


def vectorized_forecast_group(
    series: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]], model_name: str, n_steps: int, cfg: Config,
) -> Dict[str, np.ndarray]:
    """One statsforecast panel call per chunk of combinations assigned to
    `model_name`. Returns {key: array(n_steps)} starting at as_of + 1,
    clipped >= 0. Series that fail to fit are simply absent."""
    from statsforecast import StatsForecast

    keys = [k for k, (_, v) in series.items() if len(v) >= MIN_FIT_DAYS]
    out: Dict[str, np.ndarray] = {}
    col = _MODEL_OUTPUT_COL[model_name]
    for c0 in range(0, len(keys), cfg.chunk_size):
        chunk = keys[c0:c0 + cfg.chunk_size]
        ids = np.concatenate([np.full(len(series[k][1]), j, dtype=np.int64) for j, k in enumerate(chunk)])
        ds = np.concatenate([series[k][0].values for k in chunk])
        y = np.concatenate([series[k][1] for k in chunk])
        panel = pd.DataFrame({"unique_id": ids, "ds": ds, "y": y})
        sf = StatsForecast(models=[_build_statsforecast_model(model_name, cfg.fast_arima)],
                           freq="D", n_jobs=cfg.n_jobs)
        try:
            fc = sf.forecast(df=panel, h=n_steps)
        except Exception as exc:  # noqa: BLE001
            log.warning("Forecast for model %s (chunk of %d) failed entirely: %s", model_name, len(chunk), exc)
            continue
        if "unique_id" not in fc.columns:
            fc = fc.reset_index()
        fc = fc.sort_values(["unique_id", "ds"])
        for uid, g in fc.groupby("unique_id", sort=False):
            arr = g[col].to_numpy()
            if len(arr) >= n_steps and np.isfinite(arr[:n_steps]).all():
                out[chunk[int(uid)]] = np.maximum(arr[:n_steps], 0.0)
    return out


def week_blocks(dates: List[pd.Timestamp]) -> List[Tuple[int, int]]:
    """Column ranges [i0, i1) of `dates` grouped into Sunday-Saturday weeks
    (the first/last block may be partial)."""
    dow = sunday_start_dow(pd.DatetimeIndex(dates))
    starts = [0] + [i for i in range(1, len(dates)) if dow[i] == 0]
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(dates)])]


def apply_weekday_shaping(model: np.ndarray, dates: List[pd.Timestamp], dow_mean: np.ndarray,
                          has_model: np.ndarray) -> np.ndarray:
    """Keep each combo's model total per Sunday-Saturday block, but spread it
    over the block's days in proportion to the combo's own weekday averages."""
    out = model.copy()
    dow = sunday_start_dow(pd.DatetimeIndex(dates))
    for i0, i1 in week_blocks(dates):
        w = dow_mean[:, dow[i0:i1]]
        total = model[:, i0:i1].sum(axis=1)
        sw = w.sum(axis=1)
        ok = has_model & (sw > 0) & (total > 0)
        if ok.any():
            out[ok, i0:i1] = total[ok, None] * w[ok] / sw[ok, None]
    return out


# =============================================================================
# FORECAST PIPELINE (model layer only -- no reconciliation here)
# =============================================================================

@dataclass
class Forecasts:
    as_of: pd.Timestamp
    future_dates: List[pd.Timestamp]
    key_cols: List[str]
    has_plant: bool
    keys: List[str]
    segments: Dict[str, SegmentSeries]
    profiles: Dict[str, SegmentProfile]
    base: np.ndarray            # (n,H) base model output (on holiday-adjusted history)
    hf: np.ndarray              # (n,H) holiday factor applied
    model: np.ndarray           # (n,H) model-layer estimate m = base * hf (0 where no model)
    backlog: np.ndarray         # (n,H) confirmed backlog
    has_model: np.ndarray       # (n,) bool
    overall: pd.DataFrame       # index = overall_dates; base, hf, model, backlog, additional, initial
    cal: HolidayCalendar
    overall_profile: np.ndarray
    overall_n_obs: np.ndarray
    material_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]]
    slot_f: np.ndarray          # (H,) holiday slot of each forecast date (-1 = none)


def build_forecasts(
    combined: pd.DataFrame, as_of: pd.Timestamp, future_dates: List[pd.Timestamp],
    overall_dates: List[pd.Timestamp], cfg: Config,
    include_backlog: bool = True, use_holidays: bool = True, limit: Optional[int] = None,
) -> Forecasts:
    if future_dates[0] <= as_of or overall_dates[0] <= as_of:
        raise ValueError("Forecast dates must all be after the as-of date.")
    H = len(future_dates)
    last_needed = max(future_dates[-1], overall_dates[-1])
    n_steps = int((last_needed - as_of).days)
    steps_f = np.array([(d - as_of).days - 1 for d in future_dates])
    steps_o = np.array([(d - as_of).days - 1 for d in overall_dates])

    cal = build_holiday_calendar(combined["date"].min() - pd.Timedelta(days=60),
                                 last_needed + pd.Timedelta(days=60))
    S = cal.n_slots
    ones = np.ones(S)
    zeros_s = np.zeros(S)
    slot_f = np.array([cal.slot_of[cal.pos(d)] for d in future_dates])
    slot_o = np.array([cal.slot_of[cal.pos(d)] for d in overall_dates])

    # ---------------- company-wide series: holiday profile + model ----------------
    daily = combined.groupby("date")["net_pounds"].sum().sort_index()
    daily.index = pd.DatetimeIndex(daily.index)
    seg_all = build_segment_series(daily, as_of)
    if seg_all is None:
        raise RuntimeError("Could not build the company-wide daily series.")
    prof_all = classify_segment(seg_all, as_of, cfg)
    y_all = seg_all.dense_hist.to_numpy()
    p0_all = cal.pos(seg_all.dense_hist.index[0])
    if use_holidays:
        overall_profile, overall_n_obs = estimate_holiday_profile(y_all, p0_all, cal, ones, HOLIDAY_K_OVERALL)
    else:
        overall_profile, overall_n_obs = ones.copy(), zeros_s.copy()
    log.info("Overall series classified as %s -> model: %s", prof_all.classification, prof_all.model or "None")

    # ---------------- material-level holiday profiles ----------------
    material_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if use_holidays:
        mat_raw = combined.groupby(["material_no", "date"])["net_pounds"].sum()
        for mat, s in mat_raw.groupby(level=0):
            s = s.droplevel(0)
            seg_m = build_segment_series(s, as_of)
            if seg_m is None:
                continue
            material_profiles[str(mat)] = estimate_holiday_profile(
                seg_m.dense_hist.to_numpy(), cal.pos(seg_m.dense_hist.index[0]), cal,
                overall_profile, HOLIDAY_K_MATERIAL)
        log.info("Learned holiday profiles for %d material(s) + company-wide.", len(material_profiles))

    # ---------------- company-wide model (thorough search; it is one series) ----------------
    n_over = len(overall_dates)
    base_o = np.zeros(n_over)
    if prof_all.model is not None:
        z_all = y_all / factor_series(overall_profile, cal.slot_of[p0_all:p0_all + len(y_all)])
        res = vectorized_forecast_group({"__overall__": (seg_all.dense_hist.index, z_all)}, prof_all.model,
                                        n_steps, dataclasses.replace(cfg, fast_arima=False, n_jobs=1))
        if "__overall__" in res:
            base_o = res["__overall__"][steps_o]
        else:
            log.warning("Overall series model fit failed -- overall forecast is backlog only.")
    else:
        log.warning("Overall series has no usable model (classification=%s) -- overall forecast is backlog only.",
                    prof_all.classification)
    hf_o = factor_series(overall_profile, slot_o)
    M = base_o * hf_o
    B = (seg_all.backlog.reindex(overall_dates, fill_value=0.0).to_numpy() if include_backlog else np.zeros(n_over))
    overall = pd.DataFrame({
        "base": base_o, "hf": hf_o, "model": M, "backlog": B,
        "additional": np.maximum(M - B, 0.0), "initial": np.maximum(M, B),
    }, index=pd.DatetimeIndex(overall_dates))

    # ---------------- every customer/material combination ----------------
    has_plant = "plant" in combined.columns
    key_cols = (["plant", "customer", "material_no"] if has_plant else ["customer", "material_no"])
    material_name_by_no = combined.groupby("material_no")["material_name"].first()
    grouped_raw = combined.groupby(key_cols + ["date"])["net_pounds"].sum()

    segments: Dict[str, SegmentSeries] = {}
    profiles: Dict[str, SegmentProfile] = {}
    n_levels = len(key_cols)
    for lookup_key, raw in grouped_raw.groupby(level=list(range(n_levels))):
        if limit is not None and len(segments) >= limit:
            break
        lookup_key = tuple(lookup_key) if isinstance(lookup_key, tuple) else (lookup_key,)
        raw = raw.droplevel(list(range(n_levels)))
        seg = build_segment_series(raw, as_of)
        if seg is None:
            continue
        if has_plant:
            plant, cust, mat_no = lookup_key
        else:
            plant, (cust, mat_no) = "", lookup_key
        seg.customer, seg.material_no, seg.plant = cust, mat_no, plant
        seg.material_name = material_name_by_no.get(mat_no, mat_no)
        key = "||".join(lookup_key)
        segments[key] = seg
        profiles[key] = classify_segment(seg, as_of, cfg)
    if not segments:
        raise RuntimeError("No customer/material combinations could be built.")
    log.info("Enumerated %d combination(s) (key: %s).", len(segments), " + ".join(key_cols))
    log.info("Classification counts: %s",
             pd.Series([p.classification for p in profiles.values()]).value_counts().to_dict())

    # ---- per-combo holiday profile, then fit on holiday-adjusted history ----
    group_series: Dict[str, Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]]] = {
        MODEL_REGULAR: {}, MODEL_INTERMITTENT: {}, MODEL_NEW: {}}
    t0 = time.time()
    for key, seg in segments.items():
        prof = profiles[key]
        if prof.model is None:
            continue
        y = seg.dense_hist.to_numpy()
        p0 = cal.pos(seg.dense_hist.index[0])
        parent = material_profiles.get(seg.material_no, (overall_profile, None))[0]
        if use_holidays:
            hp, nobs = estimate_holiday_profile(y, p0, cal, parent, cfg.holiday_k_combo)
        else:
            hp, nobs = ones.copy(), zeros_s.copy()
        prof.holiday_profile, prof.holiday_n_obs = hp, nobs
        z = y / factor_series(hp, cal.slot_of[p0:p0 + len(y)])
        group_series[prof.model][key] = (seg.dense_hist.index, z)
    log.info("Holiday profiles estimated for modeled combinations in %.1fs.", time.time() - t0)

    raw_model: Dict[str, np.ndarray] = {}
    for model_name, grp in group_series.items():
        if not grp:
            continue
        log.info("Fitting %s for %d combination(s)...", model_name, len(grp))
        t1 = time.time()
        res = vectorized_forecast_group(grp, model_name, n_steps, cfg)
        raw_model.update(res)
        log.info("  ...done in %.1fs (%d/%d fit successfully).", time.time() - t1, len(res), len(grp))

    keys = list(segments.keys())
    n = len(keys)
    base = np.zeros((n, H))
    hf = np.ones((n, H))
    backlog = np.zeros((n, H))
    has_model = np.zeros(n, dtype=bool)
    for r, key in enumerate(keys):
        seg, prof = segments[key], profiles[key]
        if include_backlog:
            backlog[r] = seg.backlog.reindex(future_dates, fill_value=0.0).to_numpy()
        arr = raw_model.get(key)
        if arr is not None:
            has_model[r] = True
            base[r] = arr[steps_f]
            hf[r] = factor_series(prof.holiday_profile, slot_f)
    model = base * hf
    if cfg.weekday_shaping:
        dow_mean = np.vstack([profiles[k].dow_mean for k in keys])
        model = apply_weekday_shaping(model, future_dates, dow_mean, has_model)

    return Forecasts(
        as_of=as_of, future_dates=list(future_dates), key_cols=key_cols, has_plant=has_plant,
        keys=keys, segments=segments, profiles=profiles, base=base, hf=hf, model=model,
        backlog=backlog, has_model=has_model, overall=overall, cal=cal,
        overall_profile=overall_profile, overall_n_obs=overall_n_obs,
        material_profiles=material_profiles, slot_f=slot_f,
    )


# =============================================================================
# RECONCILIATION (model layer)
# =============================================================================
#
# For each Sunday-Saturday week:  gap = sum_week(M) - sum_i sum_week(m_i)
# (M = overall model estimate). The gap is allocated across combinations that
# can defensibly carry it, then each combo's weekly adjustment is spread over
# its days in proportion to its own daily model shape. Weights =
# sqrt(sum of same-weekday variances over the week, last 26 wks) ** 2 (noisier combos absorb more:
# the minimum-variance split), bounded per combo by:
#     increase: p95 of the same weekday (x holiday factor if > 1) - m_i
#     decrease: at most max_cut_frac * m_i
# Never touched: Dormant combos, combos without a fitted model, and combos
# with zero historical variance on every day of the week.
# Increases: Regular + Intermittent only. Decreases: Regular + Intermittent + New.
# What can't be absorbed is left alone and reported as "unresolved" -- never forced.
# Backlog is applied AFTER this step as a floor; it is never reduced.

def allocate_reconciliation(gap: float, weights: np.ndarray, headroom: np.ndarray) -> Tuple[np.ndarray, float]:
    """Water-filling. Returns (signed allocation, signed unresolved remainder)."""
    sign = 1.0 if gap > 0 else -1.0
    remaining = abs(gap)
    alloc = np.zeros(len(weights))
    room = headroom.astype(float).copy()
    w = weights.astype(float)
    active = (w > 0) & (room > 1e-9)
    for _ in range(RECONCILE_MAX_ITER):
        if remaining <= RECONCILE_TOLERANCE_LBS or not active.any():
            break
        wa = np.where(active, w, 0.0)
        share = remaining * wa / wa.sum()
        take = np.minimum(share, room)
        alloc += take
        room -= take
        remaining -= take.sum()
        active = active & (room > 1e-9)
    return sign * alloc, sign * remaining


@dataclass
class Reconciled:
    adj: np.ndarray            # (n,H) signed adjustment applied to the model estimate
    model_final: np.ndarray    # (n,H) m' = m + adj
    final: np.ndarray          # (n,H) max(m', backlog)
    summary: pd.DataFrame      # per-date diagnostics (informational: shape comes from combos)
    blocks: pd.DataFrame       # per Sunday-Saturday week: where reconciliation actually happens
    unresolved: Dict[str, float]   # week-start date -> unresolved lb (signed: negative = detail above target)


def reconcile_forecasts(fc: Forecasts, cfg: Config) -> Reconciled:
    """Weekly reconciliation on the MODEL layer (see section header)."""
    n, H = fc.model.shape
    dates = fc.future_dates
    m, b = fc.model, fc.backlog
    target = fc.overall.loc[dates, "model"].to_numpy()
    overall_initial = fc.overall.loc[dates, "initial"].to_numpy()

    classification = np.array([fc.profiles[k].classification for k in fc.keys])
    dow_std = np.vstack([fc.profiles[k].dow_std for k in fc.keys])
    dow_cap = np.vstack([fc.profiles[k].dow_cap for k in fc.keys])
    dow_of_date = sunday_start_dow(pd.DatetimeIndex(dates))
    can_up = fc.has_model & np.isin(classification, [CLASS_REGULAR, CLASS_INTERMITTENT])
    can_down = fc.has_model & np.isin(classification, [CLASS_REGULAR, CLASS_INTERMITTENT, CLASS_NEW])

    adj = np.zeros((n, H))
    unresolved: Dict[str, float] = {}
    block_rows = []
    for i0, i1 in week_blocks(dates):
        mb = m[:, i0:i1]
        tb = mb.sum(axis=1)
        m_target = float(target[i0:i1].sum())
        sum_pre = float(tb.sum())
        gap = m_target - sum_pre
        applied, remainder, n_elig, n_adj = 0.0, 0.0, 0, 0
        if abs(gap) > RECONCILE_TOLERANCE_LBS:
            days_dow = dow_of_date[i0:i1]
            sig_b = np.sqrt((dow_std[:, days_dow] ** 2).sum(axis=1))
            if gap > 0:
                cap_b = (dow_cap[:, days_dow] * np.maximum(fc.hf[:, i0:i1], 1.0)).sum(axis=1)
                elig = can_up & (sig_b > 0) & (tb > 0)
                headroom = np.where(elig, np.clip(cap_b - tb, 0.0, None), 0.0)
            else:
                elig = can_down & (sig_b > 0) & (tb > 0)
                headroom = np.where(elig, tb * cfg.max_cut_frac, 0.0)
            weights = np.where(elig, sig_b ** RECON_WEIGHT_POWER, 0.0)
            alloc, remainder = allocate_reconciliation(gap, weights, headroom)
            adj[:, i0:i1] = alloc[:, None] * mb / np.where(tb > 0, tb, 1.0)[:, None]   # spread by own daily shape
            applied = float(alloc.sum())
            n_elig, n_adj = int(elig.sum()), int((np.abs(alloc) > 1e-9).sum())
            if abs(remainder) > RECONCILE_TOLERANCE_LBS:
                unresolved[dates[i0].strftime("%Y-%m-%d")] = round(float(remainder), 1)
        block_rows.append((i0, i1, m_target, sum_pre, gap, applied, remainder, n_elig, n_adj))

    model_final = np.maximum(m + adj, 0.0)
    final = np.maximum(model_final, b)
    sum_pre_d = m.sum(axis=0)
    sum_post = model_final.sum(axis=0)
    excess = np.maximum(b - model_final, 0.0).sum(axis=0)
    detail_total = final.sum(axis=0)

    summary = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Overall Model Estimate (M)": target,
        "Sum of Combo Models (before)": sum_pre_d,
        "Gap Before (lb)": target - sum_pre_d,
        "Gap Before (% of M)": np.where(target > 1e-6, (target - sum_pre_d) / np.where(target > 1e-6, target, 1.0) * 100.0, np.nan),
        "Adjustment Applied (lb)": adj.sum(axis=0),
        "Sum of Combo Models (after)": sum_post,
        "Daily Residual vs M (lb)": target - sum_post,
        "Confirmed Backlog (all combos)": b.sum(axis=0),
        "Backlog-Driven Excess (lb)": excess,
        "Overall Total = max(M, backlog)": overall_initial,
        "Detail Total (Forecast sheet)": detail_total,
        "Detail - Overall (lb)": detail_total - overall_initial,
    })
    blocks = pd.DataFrame({
        "Week Start": [dates[r[0]].strftime("%Y-%m-%d") for r in block_rows],
        "Week End": [dates[r[1] - 1].strftime("%Y-%m-%d") for r in block_rows],
        "Days In Window": [r[1] - r[0] for r in block_rows],
        "Overall Model Estimate (M)": [r[2] for r in block_rows],
        "Sum of Combo Models (before)": [r[3] for r in block_rows],
        "Gap Before (lb)": [r[4] for r in block_rows],
        "Gap Before (% of M)": [(r[4] / r[2] * 100.0) if r[2] > 1e-6 else np.nan for r in block_rows],
        "Adjustment Applied (lb)": [r[5] for r in block_rows],
        "Unresolved (lb)": [r[6] for r in block_rows],
        "Combos Eligible": [r[7] for r in block_rows],
        "Combos Adjusted": [r[8] for r in block_rows],
        "Confirmed Backlog (all combos)": [float(b[:, r[0]:r[1]].sum()) for r in block_rows],
        "Backlog-Driven Excess (lb)": [float(excess[r[0]:r[1]].sum()) for r in block_rows],
        "Overall Total = max(M, backlog)": [float(overall_initial[r[0]:r[1]].sum()) for r in block_rows],
        "Detail Total (Forecast sheet)": [float(detail_total[r[0]:r[1]].sum()) for r in block_rows],
        "Detail - Overall (lb)": [float((detail_total - overall_initial)[r[0]:r[1]].sum()) for r in block_rows],
    })
    return Reconciled(adj=adj, model_final=model_final, final=final, summary=summary,
                      blocks=blocks, unresolved=unresolved)


# =============================================================================
# BACKTEST (holdout, no backlog, with vs. without holiday effects)
# =============================================================================

def _wape(actual: np.ndarray, fcst: np.ndarray) -> float:
    den = float(np.abs(actual).sum())
    return float(np.abs(fcst - actual).sum() / den) if den > 0 else float("nan")


def _bias(actual: np.ndarray, fcst: np.ndarray) -> float:
    den = float(np.abs(actual).sum())
    return float((fcst - actual).sum() / den) if den > 0 else float("nan")


def run_backtest(
    combined: pd.DataFrame, as_of: pd.Timestamp, horizon: int, cfg: Config,
    limit: Optional[int], cutoff: Optional[pd.Timestamp],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pretend the history ends at `cutoff`, forecast the next `horizon` days
    with NO backlog information (the workbook's future-dated rows would
    otherwise leak the answer), and score against what actually happened.
    Runs the full pipeline twice: with and without holiday effects."""
    cutoff = cutoff if cutoff is not None else as_of - pd.Timedelta(days=horizon)
    dates = list(pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D"))
    log.info("BACKTEST: cutoff %s, scoring %s to %s.", cutoff.date(), dates[0].date(), dates[-1].date())
    actual_company = combined.groupby("date")["net_pounds"].sum().reindex(dates, fill_value=0.0)
    key_cols = ["plant", "customer", "material_no"] if "plant" in combined.columns else ["customer", "material_no"]
    act = combined.groupby(key_cols + ["date"])["net_pounds"].sum()
    act = act[act.index.get_level_values("date").isin(dates)]

    variants = {}
    for label, flag in (("With holiday effects", True), ("Without holiday effects", False)):
        log.info("BACKTEST variant: %s", label)
        fc = build_forecasts(combined, cutoff, dates, dates, cfg, include_backlog=False,
                             use_holidays=flag, limit=limit)
        rec = reconcile_forecasts(fc, cfg)
        idx = pd.MultiIndex.from_tuples(
            [((fc.segments[k].plant,) if fc.has_plant else ()) + (fc.segments[k].customer, fc.segments[k].material_no)
             for k in fc.keys], names=key_cols)
        if len(act):
            A = act.unstack("date").reindex(index=idx, columns=dates).fillna(0.0).to_numpy()
        else:
            A = np.zeros((len(fc.keys), horizon))
        variants[label] = dict(M=fc.overall["model"].to_numpy(), bottom_up=fc.model.sum(axis=0),
                               recon=rec.model_final, pre=fc.model, A=A)

    def weekly(x: np.ndarray) -> pd.Series:
        return pd.Series(x, index=pd.DatetimeIndex(dates)).resample("W-SAT").sum()

    a_co = actual_company.to_numpy()
    rows = []
    for label, v in variants.items():
        A = v["A"]
        a_cov = A.sum(axis=0)
        rows.append({"Metric": "Company daily WAPE -- overall model", "Variant": label, "Value": _wape(a_co, v["M"])})
        rows.append({"Metric": "Company weekly WAPE -- overall model", "Variant": label,
                     "Value": _wape(weekly(a_co).to_numpy(), weekly(v["M"]).to_numpy())})
        rows.append({"Metric": "Company bias -- overall model", "Variant": label, "Value": _bias(a_co, v["M"])})
        rows.append({"Metric": "Covered combos daily WAPE -- bottom-up total, before reconciliation", "Variant": label,
                     "Value": _wape(a_cov, v["bottom_up"])})
        rows.append({"Metric": "Covered combos weekly WAPE -- bottom-up total, before reconciliation", "Variant": label,
                     "Value": _wape(weekly(a_cov).to_numpy(), weekly(v["bottom_up"]).to_numpy())})
        rows.append({"Metric": "Combo-level daily WAPE -- reconciled", "Variant": label, "Value": _wape(A, v["recon"])})
        rows.append({"Metric": "Combo-level horizon-total WAPE -- reconciled", "Variant": label,
                     "Value": _wape(A.sum(axis=1), v["recon"].sum(axis=1))})
        rows.append({"Metric": "Combo-level horizon-total WAPE -- before reconciliation", "Variant": label,
                     "Value": _wape(A.sum(axis=1), v["pre"].sum(axis=1))})
    summary = pd.DataFrame(rows).pivot(index="Metric", columns="Variant", values="Value").reset_index()
    wk = pd.DataFrame({"Week End (Sat)": weekly(a_co).index.strftime("%Y-%m-%d"),
                       "Days In Window": pd.Series(1, index=pd.DatetimeIndex(dates)).resample("W-SAT").sum().to_numpy(),
                       "Actual (lbs)": weekly(a_co).to_numpy()})
    for label, v in variants.items():
        wk[f"Overall Model -- {label}"] = weekly(v["M"]).to_numpy()
        wk[f"Bottom-up -- {label}"] = weekly(v["bottom_up"]).to_numpy()
    for _, r in summary.iterrows():
        log.info("BACKTEST  %-72s  with: %.3f   without: %.3f", r["Metric"],
                 r.get("With holiday effects", float("nan")), r.get("Without holiday effects", float("nan")))
    return summary, wk


# =============================================================================
# OUTPUT TABLES
# =============================================================================

def holiday_summary_text(prof: SegmentProfile, cal: HolidayCalendar, top: int = 6) -> str:
    hp, nobs = prof.holiday_profile, prof.holiday_n_obs
    if hp is None:
        return "n/a (no model)"
    dev = np.abs(hp - 1.0)
    order = [i for i in np.argsort(-dev) if dev[i] >= 0.10][:top]
    if not order:
        return "No material holiday effects (all within +/-10%)"
    return "; ".join(f"{cal.slot_names[i]} x{hp[i]:.2f} ({'own' if nobs[i] >= 1.0 else 'borrowed'})" for i in order)


def build_weekly_overall_sheet(fc: Forecasts, rec: Reconciled, weekly_start: pd.Timestamp, weeks: int) -> pd.DataFrame:
    ov = fc.overall
    idx = pd.DatetimeIndex(fc.future_dates)
    detail_total = pd.Series(rec.final.sum(axis=0), index=idx)
    excess_s = pd.Series(rec.summary["Backlog-Driven Excess (lb)"].to_numpy(), index=idx)
    rows = []
    for ws in pd.date_range(weekly_start, periods=weeks, freq="7D"):
        w = ov.loc[(ov.index >= ws) & (ov.index <= ws + pd.Timedelta(days=6))]
        if w.empty:
            continue
        we = ws + pd.Timedelta(days=6)
        dt = detail_total.reindex(w.index)
        detail = round(float(dt.sum()), 1) if (len(w) == 7 and dt.notna().all()) else None
        overall_total = float(w["initial"].sum())
        labels = []
        for d in w.index:
            s = fc.cal.slot_of[fc.cal.pos(d)]
            if s >= 0:
                txt = f"{fc.cal.slot_names[s]} (x{fc.overall_profile[s]:.2f})"
                if txt not in labels:
                    labels.append(txt)
        rows.append({
            "Week Start": ws.strftime("%Y-%m-%d"), "Week End": we.strftime("%Y-%m-%d"),
            "Confirmed Backlog (lbs)": round(float(w["backlog"].sum()), 1),
            "Additional Model Demand (lbs)": round(float(w["additional"].sum()), 1),
            "Overall Model Total (lbs)": round(overall_total, 1),
            "Detail Total (lbs)": detail,
            "Detail - Overall (lbs)": (round(detail - overall_total, 1) if detail is not None else None),
            "of which Backlog-Driven Excess (lbs)": (round(float(excess_s.reindex(w.index).sum()), 1) if detail is not None else None),
            "of which Unresolved Reconciliation (lbs)": (round(-float(rec.unresolved.get(ws.strftime("%Y-%m-%d"), 0.0)), 1) if detail is not None else None),
            "Holiday Context": "; ".join(labels) if labels else "",
        })
    return pd.DataFrame(rows)


def build_holiday_effects_sheet(fc: Forecasts) -> pd.DataFrame:
    cols = fc.cal.slot_names
    rows = [["ALL (company) -- factor", ""] + [round(float(x), 3) for x in fc.overall_profile],
            ["ALL (company) -- blocks of evidence", ""] + [round(float(x), 2) for x in fc.overall_n_obs]]
    for mat, (hp, _) in sorted(fc.material_profiles.items()):
        rows.append(["Material -- factor", mat] + [round(float(x), 3) for x in hp])
    return pd.DataFrame(rows, columns=["Level", "Material No"] + cols)


def build_output_tables(fc: Forecasts, rec: Reconciled, cfg: Config, include_reasoning: bool):
    n, H = fc.model.shape
    dates = fc.future_dates
    date_cols = [d.strftime("%Y-%m-%d") for d in dates]
    id_names = (["Plant"] if fc.has_plant else []) + ["Customer", "Material No", "Material Name"]
    combo_fields = id_names + ["Classification", "Model Used", "Days Since Last Nonzero", "History Span (days)",
                               "Recent 28-Day Avg", "Historical Min/Max", "Learned Holiday Effects (top)"]
    date_fields = ["Base x Holiday x Weekday Shape = Model", "Backlog + Additional = Final",
                   "Reconciliation Adj (lb)", "Final Forecast", "Reason"]
    forecast_id_cols = (["plant"] if fc.has_plant else []) + ["customer", "material_no", "material_name"]

    forecast_rows, reasoning_rows = [], []
    for r, key in enumerate(fc.keys):
        seg, prof = fc.segments[key], fc.profiles[key]
        frow = {"customer": seg.customer, "material_no": seg.material_no, "material_name": seg.material_name}
        if fc.has_plant:
            frow["plant"] = seg.plant
        frow.update({date_cols[i]: round(float(rec.final[r, i]), 1) for i in range(H)})
        forecast_rows.append(frow)
        if not include_reasoning:
            continue

        row: List[object] = ([seg.plant] if fc.has_plant else []) + [
            seg.customer, seg.material_no, seg.material_name, prof.classification,
            prof.model or "None (backlog only)", prof.days_since_last_nonzero, prof.history_span_days,
            round(prof.recent_avg, 1), f"{prof.hist_min:.0f} / {prof.hist_max:.0f}",
            holiday_summary_text(prof, fc.cal),
        ]
        for i in range(H):
            b = float(fc.backlog[r, i])
            base_v, hf_v = float(fc.base[r, i]), float(fc.hf[r, i])
            m0 = float(fc.model[r, i])
            adj = float(rec.adj[r, i])
            m1 = float(rec.model_final[r, i])
            final = float(rec.final[r, i])
            reasons = []
            if prof.classification == CLASS_DORMANT:
                reasons.append(f"Dormant ({prof.days_since_last_nonzero}d since last order) -- backlog only")
            elif not fc.has_model[r]:
                reasons.append(f"{prof.classification}: {prof.model} fit failed -- backlog only")
            else:
                reasons.append(f"{prof.classification} demand -> {prof.model}")
                s = fc.slot_f[i]
                if s >= 0 and abs(hf_v - 1.0) > 0.05:
                    src = "own history" if prof.holiday_n_obs[s] >= 1.0 else "borrowed from material/company"
                    reasons.append(f"{fc.cal.slot_names[s]} x{hf_v:.2f} ({src})")
            if b > 0:
                reasons.append(f"Confirmed backlog {b:.0f} lb")
            if abs(adj) > RECONCILE_TOLERANCE_LBS:
                reasons.append(f"{'Increased' if adj > 0 else 'Decreased'} model {abs(adj):.0f} lb to align "
                               f"with overall model")
            if b > m1 + RECONCILE_TOLERANCE_LBS and fc.has_model[r]:
                reasons.append(f"Backlog exceeds model by {b - m1:.0f} lb -- floor applied (reported, not redistributed)")
            if abs(adj) <= RECONCILE_TOLERANCE_LBS:
                reasons.append("No reconciliation needed")
            row.extend([
                f"{base_v:.0f} x {hf_v:.2f} x {(m0 / (base_v * hf_v) if base_v * hf_v > 1e-9 else 1.0):.2f} = {m0:.0f}",
                f"{b:.0f} + {final - b:.0f} = {final:.0f}",
                round(adj, 1), round(final, 1), "; ".join(reasons),
            ])
        reasoning_rows.append(row)

    forecast_df = pd.DataFrame(forecast_rows, columns=forecast_id_cols + date_cols)
    return forecast_df, reasoning_rows, combo_fields, date_fields, date_cols


def write_excel(
    out_path: Union[str, Path], forecast_df: pd.DataFrame, reasoning_rows: List[List[object]],
    combo_fields: List[str], date_fields: List[str], date_cols: List[str],
    weekly_df: pd.DataFrame, rec: Reconciled, holiday_df: pd.DataFrame,
    backtest: Optional[Tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    with pd.ExcelWriter(out_path, engine="xlsxwriter", engine_kwargs={"options": {"nan_inf_to_errors": True}}) as writer:
        wb = writer.book
        bold = wb.add_format({"bold": True})
        num0 = wb.add_format({"num_format": "#,##0"})
        num1 = wb.add_format({"num_format": "#,##0.0"})
        pct = wb.add_format({"num_format": "0.0"})
        dec3 = wb.add_format({"num_format": "0.000"})

        forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
        ws = writer.sheets["Forecast"]
        ws.set_row(0, None, bold)
        n_id = len(forecast_df.columns) - len(date_cols)
        ws.set_column(0, n_id - 1, 24)
        ws.set_column(n_id, n_id + len(date_cols) - 1, 12, num0)
        ws.freeze_panes(1, n_id)

        weekly_df.to_excel(writer, sheet_name="Overall Forecast (Weekly)", index=False)
        ws3 = writer.sheets["Overall Forecast (Weekly)"]
        ws3.set_row(0, None, bold)
        ws3.set_column(0, 1, 12)
        ws3.set_column(2, 8, 24, num0)
        ws3.set_column(9, 9, 60)

        rec.summary.to_excel(writer, sheet_name="Reconciliation Summary", index=False)
        ws4 = writer.sheets["Reconciliation Summary"]
        ws4.set_row(0, None, bold)
        ws4.set_column(0, 0, 12)
        ws4.set_column(1, len(rec.summary.columns) - 1, 20, num0)
        ws4.set_column(4, 4, 18, pct)
        ws4.freeze_panes(1, 1)

        rec.blocks.to_excel(writer, sheet_name="Reconciliation (Weekly)", index=False)
        ws4b = writer.sheets["Reconciliation (Weekly)"]
        ws4b.set_row(0, None, bold)
        ws4b.set_column(0, 1, 12)
        ws4b.set_column(2, len(rec.blocks.columns) - 1, 20, num0)
        ws4b.set_column(6, 6, 18, pct)
        ws4b.freeze_panes(1, 2)

        holiday_df.to_excel(writer, sheet_name="Holiday Effects", index=False)
        ws5 = writer.sheets["Holiday Effects"]
        ws5.set_row(0, None, bold)
        ws5.set_column(0, 1, 30)
        ws5.set_column(2, len(holiday_df.columns) - 1, 16, dec3)
        ws5.freeze_panes(1, 2)

        if backtest is not None:
            summ, wk = backtest
            summ.to_excel(writer, sheet_name="Backtest", index=False, startrow=0)
            wk.to_excel(writer, sheet_name="Backtest", index=False, startrow=len(summ) + 3)
            ws6 = writer.sheets["Backtest"]
            ws6.set_column(0, 0, 75)
            ws6.set_column(1, 8, 26, num1)

        if reasoning_rows:
            ws2 = wb.add_worksheet("Reasoning")
            hdr = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
            n_combo = len(combo_fields)
            n_df = len(date_fields)
            ws2.merge_range(0, 0, 0, n_combo - 1, "Combination", hdr)
            for d_idx, dcol in enumerate(date_cols):
                s = n_combo + d_idx * n_df
                ws2.merge_range(0, s, 0, s + n_df - 1, dcol, hdr)
            for c_idx, name in enumerate(combo_fields):
                ws2.write(1, c_idx, name, hdr)
            for d_idx in range(len(date_cols)):
                for f_idx, name in enumerate(date_fields):
                    ws2.write(1, n_combo + d_idx * n_df + f_idx, name, hdr)
            for r_idx, values in enumerate(reasoning_rows, start=2):
                ws2.write_row(r_idx, 0, values)
            ws2.set_column(0, n_combo - 1, 16)
            ws2.set_column(n_combo - 1, n_combo - 1, 50)
            ws2.set_column(n_combo, n_combo + n_df * len(date_cols) - 1, 16)
            ws2.freeze_panes(2, n_combo)


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run(args: argparse.Namespace) -> None:
    t_start = time.time()
    cfg = Config(fast_arima=args.fast_arima, n_jobs=args.n_jobs,
                 dormant_inactivity_days=args.dormant_inactivity_days,
                 new_min_history_days=args.new_min_history_days,
                 regular_density_threshold=args.regular_density_threshold,
                 holiday_k_combo=args.holiday_k, max_cut_frac=args.max_cut_frac, chunk_size=args.chunk_size,
                 weekday_shaping=not args.no_weekday_shaping)
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(date.today())
    horizon = args.horizon

    log.info("Loading workbook: %s", args.workbook)
    combined = load_and_clean_workbook(args.workbook)

    forecast_start = (pd.Timestamp(args.forecast_start_date) if args.forecast_start_date
                      else resolve_forecast_start(as_of, args.align_to_sunday))
    if forecast_start <= as_of:
        raise SystemExit(f"Forecast start ({forecast_start.date()}) must be after the as-of date ({as_of.date()}).")
    future_dates = list(pd.date_range(forecast_start, periods=horizon, freq="D"))
    log.info("Forecasting %s to %s (%d days), as-of %s.",
             future_dates[0].date(), future_dates[-1].date(), horizon, as_of.date())

    weekly_start = pd.Timestamp(args.weekly_start_date) if args.weekly_start_date else next_sunday_on_or_after(forecast_start)
    if weekly_start.dayofweek != 6:
        raise SystemExit(f"--weekly-start-date must be a Sunday (got {weekly_start.date()}).")
    if weekly_start <= as_of:
        raise SystemExit("--weekly-start-date must be after the as-of date.")
    weeks = args.weekly_horizon_weeks or max(1, -(-horizon // 7))
    weekly_end = weekly_start + pd.Timedelta(days=weeks * 7 - 1)
    log.info("Overall weekly sheet: %s to %s (%d week(s)).", weekly_start.date(), weekly_end.date(), weeks)

    overall_dates = list(pd.date_range(min(forecast_start, weekly_start), max(future_dates[-1], weekly_end), freq="D"))

    fc = build_forecasts(combined, as_of, future_dates, overall_dates, cfg,
                         include_backlog=True, use_holidays=not args.no_holidays, limit=args.limit)
    rec = reconcile_forecasts(fc, cfg)

    s = rec.summary
    log.info("Model-layer gap before reconciliation (first 5 dates, lb): %s",
             {s["Date"][i]: round(float(s["Gap Before (lb)"][i]), 1) for i in range(min(5, len(s)))})
    wb = rec.blocks
    log.info("Weekly model-layer gap before reconciliation (lb, %% of overall): %s",
             {wb["Week Start"][i]: (round(float(wb["Gap Before (lb)"][i]), 1), round(float(wb["Gap Before (% of M)"][i]), 2))
              for i in range(len(wb))})
    if args.limit is not None:
        log.warning("--limit is set: the detail covers only %d combination(s) but the overall target covers the "
                    "whole workbook, so large unresolved gaps below are expected. Use --limit for timing only.",
                    len(fc.keys))
    if rec.unresolved:
        log.warning("Reconciliation could not fully close the gap on %d date(s) -- no eligible combination had "
                    "headroom left. Unresolved (lb): %s", len(rec.unresolved), rec.unresolved)
    else:
        log.info("Reconciliation closed the model-layer gap on every date (within %.1f lb).", RECONCILE_TOLERANCE_LBS)
    log.info("Backlog-driven excess (detail above overall because confirmed orders beat a combo's model), "
             "first 5 dates (lb): %s",
             {s["Date"][i]: round(float(s["Backlog-Driven Excess (lb)"][i]), 1) for i in range(min(5, len(s)))})

    forecast_df, reasoning_rows, combo_fields, date_fields, date_cols = build_output_tables(
        fc, rec, cfg, include_reasoning=not args.no_reasoning)
    weekly_df = build_weekly_overall_sheet(fc, rec, weekly_start, weeks)
    holiday_df = build_holiday_effects_sheet(fc)

    backtest = None
    if args.backtest:
        cutoff = pd.Timestamp(args.backtest_cutoff) if args.backtest_cutoff else None
        backtest = run_backtest(combined, as_of, horizon, cfg, args.limit, cutoff)

    write_excel(args.excel_out, forecast_df, reasoning_rows, combo_fields, date_fields, date_cols,
                weekly_df, rec, holiday_df, backtest)
    log.info("Done in %.1fs total. %d combination(s) written. Wrote %s.",
             time.time() - t_start, len(fc.keys), args.excel_out)


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", default="Sales Order 2024-26 - One Material.xlsx")
    p.add_argument("--excel-out", default="customer_material_reconciled_forecast_one_mat.xlsx")
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON_DAYS, help="Daily/detail horizon in days.")
    p.add_argument("--as-of", default=None, help="Historical cutoff (YYYY-MM-DD). Default: today.")
    p.add_argument("--align-to-sunday", action="store_true",
                   help="Start the daily forecast on the next Sunday. Ignored if --forecast-start-date is given.")
    p.add_argument("--forecast-start-date", default=None, help="Explicit daily forecast start (YYYY-MM-DD).")
    p.add_argument("--weekly-start-date", default=None,
                   help="Start (a Sunday) of the 'Overall Forecast (Weekly)' sheet. Default: next Sunday on/after "
                        "the daily forecast start.")
    p.add_argument("--weekly-horizon-weeks", type=int, default=None,
                   help="Weeks on the weekly sheet. Default: enough to cover --horizon.")
    arima = p.add_mutually_exclusive_group()
    arima.add_argument("--fast-arima", action="store_true", default=True,
                       help="Reduced AutoARIMA search for the Intermittent bucket (default).")
    arima.add_argument("--thorough-arima", dest="fast_arima", action="store_false",
                       help="Full AutoARIMA search for the Intermittent bucket (slower).")
    p.add_argument("--n-jobs", type=int, default=-1, help="CPU cores per model call (-1 = all).")
    p.add_argument("--chunk-size", type=int, default=5000, help="Combinations per statsforecast call (memory control).")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N combinations (timing test).")
    p.add_argument("--dormant-inactivity-days", type=int, default=DORMANT_INACTIVITY_DAYS)
    p.add_argument("--new-min-history-days", type=int, default=NEW_MIN_HISTORY_DAYS)
    p.add_argument("--regular-density-threshold", type=float, default=REGULAR_DENSITY_THRESHOLD)
    p.add_argument("--holiday-k", type=float, default=HOLIDAY_K_COMBO,
                   help="Combo-level shrinkage strength toward the material pattern (higher = trust own history "
                        "less). Credibility = n_obs / (n_obs + K).")
    p.add_argument("--max-cut-frac", type=float, default=RECON_MAX_CUT_FRAC,
                   help="Max fraction of a combo's model estimate reconciliation may remove on one date.")
    p.add_argument("--no-weekday-shaping", action="store_true",
                   help="Keep each combo's raw daily model shape instead of re-spreading its weekly total by weekday history.")
    p.add_argument("--no-holidays", action="store_true", help="Disable holiday effects (ablation / comparison).")
    p.add_argument("--no-reasoning", action="store_true",
                   help="Skip the Reasoning sheet (much faster/smaller output at tens of thousands of combos).")
    p.add_argument("--backtest", action="store_true",
                   help="Also run a holdout backtest (no backlog) with vs. without holiday effects. Roughly "
                        "doubles runtime. Writes a 'Backtest' sheet.")
    p.add_argument("--backtest-cutoff", default=None,
                   help="History cutoff (YYYY-MM-DD) for the backtest. Default: as-of minus horizon. "
                        "E.g. 2025-11-09 to test the 2025 Thanksgiving ramp.")
    return p


def main() -> None:
    run(_build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
