#!/usr/bin/env python3
# =============================================================================
# production_forecasting_mvp.py
# GLC Sales Order Daily Net Pounds Forecasting Dashboard
# =============================================================================
#
# INSTALLATION (core, required)
# ------------------------------------------------------------------------
#   pip install pandas numpy openpyxl streamlit plotly scikit-learn statsforecast scipy
#
#   ...or, using the pinned file shipped alongside this script:
#   pip install -r requirements.txt
#
# INSTALLATION (optional, Amazon Chronos-2 model only)
# ------------------------------------------------------------------------
#   See OPTIONAL_MODELS_SETUP.txt for the exact commands to install torch +
#   chronos-forecasting and to pre-download the amazon/chronos-2 weights.
#   The app runs perfectly well without this -- Chronos-2 will simply show
#   as SKIPPED on the dashboard and every other model still runs.
#
# RUN
# ------------------------------------------------------------------------
#   streamlit run production_forecasting_mvp.py
#
# DATA
# ------------------------------------------------------------------------
#   By default the app looks for "Sales Order 2024-26.xlsx" in the same
#   folder as this script (DEFAULT_WORKBOOK_PATH below). Because that file
#   is large, the app is built to load it automatically on launch so you
#   land straight on results. A sidebar file-uploader is also provided so
#   the same tool can be pointed at a different/updated workbook without
#   touching the code.
#
# =============================================================================
# ASSUMPTIONS & BUSINESS LOGIC (read this before trusting the numbers)
# =============================================================================
#
# 1. HISTORICAL ACTUALS vs. KNOWN BACKLOG
#    The workbook contains open/scheduled sales orders, so some rows carry a
#    Mat.avail.dt that is still in the future relative to today. Those rows
#    are NOT treated as historical observations (that would leak future
#    information into model training). Instead:
#       - Historical Actuals = daily Net Pounds totals for dates <= today
#       - Known Backlog      = daily Net Pounds totals for dates >  today
#         (i.e. orders already booked/committed with a known ship date)
#    Every model is trained ONLY on Historical Actuals.
#    The final projected number shown for a future date combines the two:
#         Combined Projection(date) = max(Statistical Forecast(date),
#                                          Known Backlog(date))
#    Backlog acts as a floor, since it is already committed and will not be
#    "un-booked". If the statistical model expects more organic demand than
#    what's already on the books, the higher number is shown. This
#    combination logic is a judgment call, clearly labeled in the UI/report,
#    and easy to change in `combine_backlog_and_forecast()` below.
#
# 2. HIGH-PRODUCTION-DATE INPUT -- TWO ROLES
#    The same date list/CSV (comma text or upload) is used for two purposes
#    depending on whether a date falls in the past or the future:
#      - PAST dates (on/before the as-of date) are treated as real labeled
#        high-production days: they become a training feature
#        (is_high_production_date) for HistGradientBoostingRegressor, and
#        they're used to compute an EMPIRICAL uplift estimate (actual value
#        vs. the same day-of-week average over the trailing weeks). This is
#        what "should be used for actuals as well" means in practice.
#      - FUTURE dates (within the forecast horizon) drive the
#        business-adjustment uplift ramp described below.
#      - Dates that are neither (before any historical data, or beyond the
#        forecast horizon) have no effect and are flagged to the user rather
#        than silently ignored.
#    Note: if a labeled historical date falls on a fixed recurring calendar
#    holiday (July 4th, Christmas), the model's day-of-year/seasonal
#    features may already capture that pattern on their own, so the extra
#    flag adds little there. It matters most for one-off, non-recurring
#    high-production days that the calendar features can't already predict.
#    Changing which DATES are entered requires clicking "Update Forecast"
#    (it changes a training feature, so the backtest + model fit re-run).
#    Changing only a CSV uplift_percent does not require a re-run.
#
# 3. UPLIFT RAMP ("business-adjusted forecast")
#    A future high-production date is never applied as a single-day spike.
#    Production/shipping ramps up and down around real events, so a
#    triangular ramp is applied across the date +/- 2 days:
#         Day -2: 25% of the uplift   Day -1: 50%   Day 0: 100% (peak)
#         Day +1: 50%                 Day +2: 25%
#    Default uplift = the empirical estimate from labeled historical dates
#    if one can be computed, else a flat +20% if the user supplies only
#    dates with no percent and no usable historical baseline exists.
#    If several high-production dates are close together, their ramps are
#    summed (stacked), then applied multiplicatively to the Combined
#    Projection for that day. See `apply_uplift_ramp()`.
#
# 4. TimesFM 3.0 was named in the original brief as a possible research
#    model. As of this writing, Google distributes the TimesFM 3.0
#    *pretrained weights* under "timesfm-non-commercial-license-v1.0",
#    which explicitly prohibits commercial/production use of the weights
#    (the code itself is Apache-2.0, but the weights are not). Because this
#    tool is used for real GLC production planning, TimesFM was excluded
#    on licensing grounds rather than included "for research only" inside
#    a production tool. The five models actually implemented are Seasonal
#    Naive, AutoARIMA, AutoETS, HistGradientBoostingRegressor, and
#    Amazon Chronos-2 (Apache-2.0, open weights, runs 100% locally).
#
# 5. Net Pounds cleaning: commas are treated as thousands separators,
#    parentheses as an accounting negative, and anything else that can't be
#    coerced to a float is REJECTED (never silently coerced to 0) and
#    counted in the data-quality report.
#
# 6. Duplicate detection compares every non-forecasting column (Order#,
#    MaterialNo, Material, Doc Dt, Mat.avail.dt, Net Pounds) across all six
#    sheets combined. Exact duplicate rows are reported AND excluded from
#    the daily aggregation by default (a duplicated order line is a data
#    error, not real additional demand) -- this can be turned off in the
#    sidebar to see the raw, un-deduplicated totals instead.
#
# =============================================================================

from __future__ import annotations

import io
import re
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Streamlit / Plotly are required for the UI. Import them at module scope so
# that a clear error is raised immediately if they are missing, but keep all
# *pure* data/model logic below independent of streamlit so it can be unit
# tested by importing this file without launching the app.
# --------------------------------------------------------------------------
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED: int = 42
np.random.seed(RANDOM_SEED)

DEFAULT_WORKBOOK_PATH: Path = Path(__file__).resolve().parent / "Sales Order 2024-26.xlsx"
EXPECTED_SHEETS: List[str] = ["24HY1", "24H2", "25H1", "25H2", "26HY1", "26HY2"]

# Forecasting columns (business names). Everything else in the workbook is
# loaded for duplicate-detection purposes only and ignored for modeling.
DATE_COLUMN_NAME: str = "Mat.avail.dt"
POUNDS_COLUMN_NAME: str = "Net Pounds"

DEFAULT_HORIZON_DAYS: int = 14
MAX_HORIZON_DAYS: int = 60
SEASONAL_PERIOD: int = 7
ANNUAL_PERIOD: float = 365.25

N_BACKTEST_WINDOWS_PREFERRED: int = 8
MIN_TRAIN_DAYS: int = 60  # minimum history required before the first backtest origin

DEFAULT_UPLIFT_PCT: float = 20.0
RAMP_WEIGHTS: Dict[int, float] = {-2: 0.25, -1: 0.50, 0: 1.00, 1: 0.50, 2: 0.25}

# A "meaningful" improvement over Seasonal Naive, in absolute WAPE percentage
# points, required before a more complex model is recommended over it.
MEANINGFUL_WAPE_IMPROVEMENT_PP: float = 2.0
MAX_WINDOW_FAILURE_RATE: float = 0.20  # exclude a model if >20% of windows fail

OUTPUT_DIR: Path = Path(__file__).resolve().parent / "forecast_outputs"

CHRONOS_MODEL_ID: str = "amazon/chronos-2"

# ---- GLC brand palette (Great Lakes Cheese: Regal Blue / Saffron / Flesh) --
COLOR_PRIMARY: str = "#003E72"      # Regal Blue
COLOR_ACCENT: str = "#F5B334"       # Saffron
COLOR_ACCENT_SOFT: str = "#FECDA5"  # Flesh
COLOR_NEUTRAL_DARK: str = "#2B2B2B"
COLOR_NEUTRAL_MED: str = "#6E6E6E"
COLOR_NEUTRAL_LIGHT: str = "#F4F6F8"
COLOR_GRID: str = "#E3E7EC"
COLOR_SUCCESS: str = "#2E7D32"
COLOR_WARN: str = "#B45309"
COLOR_DANGER: str = "#B3261E"

MODEL_COLOR_SEQUENCE: List[str] = [
    COLOR_PRIMARY,
    COLOR_ACCENT,
    "#5C8A8A",
    "#8E6C8A",
    "#4F6D3A",
]

PLOTLY_FONT = dict(family="Segoe UI, Helvetica, Arial, sans-serif", color=COLOR_NEUTRAL_DARK)


def _base_layout(title: str, y_title: str = "Net Pounds") -> dict:
    """Shared, professional Plotly layout kwargs (no emoji, GLC palette)."""
    return dict(
        title=dict(text=title, font=dict(size=18, color=COLOR_PRIMARY, family=PLOTLY_FONT["family"])),
        font=PLOTLY_FONT,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Date", showgrid=True, gridcolor=COLOR_GRID, tickformat="%b %d, %Y"),
        yaxis=dict(title=y_title, showgrid=True, gridcolor=COLOR_GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=70, b=50),
        hovermode="x unified",
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SheetLoadReport:
    """Data-quality outcome for a single worksheet."""
    sheet_name: str
    rows_loaded: int = 0
    rows_missing_required_columns: int = 0
    invalid_date_rows: int = 0
    invalid_pounds_rows: int = 0
    valid_rows: int = 0
    error: Optional[str] = None


@dataclass
class DataQualityReport:
    """Aggregate data-quality summary across the whole workbook."""
    sheet_reports: List[SheetLoadReport] = field(default_factory=list)
    total_rows_loaded: int = 0
    total_invalid_dates: int = 0
    total_invalid_pounds: int = 0
    total_valid_rows: int = 0
    duplicate_row_count: int = 0
    duplicates_excluded: bool = True
    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None
    missing_calendar_dates: List[pd.Timestamp] = field(default_factory=list)
    missing_date_treatment: str = "Keep missing dates as missing"
    as_of_date: Optional[pd.Timestamp] = None
    historical_days: int = 0
    backlog_days: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.sheet_reports:
            rows.append(
                {
                    "sheet": r.sheet_name,
                    "rows_loaded": r.rows_loaded,
                    "invalid_dates": r.invalid_date_rows,
                    "invalid_net_pounds": r.invalid_pounds_rows,
                    "valid_rows": r.valid_rows,
                    "error": r.error or "",
                }
            )
        return pd.DataFrame(rows)


@dataclass
class ModelForecastResult:
    """Standard output of any model's .predict() call."""
    model_name: str
    dates: List[pd.Timestamp]
    forecast: np.ndarray
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    skipped: bool = False
    skip_reason: str = ""
    device: str = "n/a"


@dataclass
class BacktestWindowResult:
    model_name: str
    origin_date: pd.Timestamp
    success: bool
    runtime_seconds: float = 0.0
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    error: str = ""


@dataclass
class ModelMetrics:
    model_name: str
    wape: float = np.nan
    mae: float = np.nan
    rmse: float = np.nan
    smape: float = np.nan
    mase: float = np.nan
    bias: float = np.nan
    abs_pct_bias: float = np.nan
    accuracy_score: float = np.nan
    runtime_seconds: float = 0.0
    n_success_windows: int = 0
    n_failed_windows: int = 0
    skipped: bool = False
    skip_reason: str = ""


# =============================================================================
# COLUMN / SHEET NAME NORMALIZATION
# =============================================================================

def _normalize_token(name: object) -> str:
    """Lower-case and strip everything that isn't a letter or digit, so that
    'Mat.avail.dt', ' mat avail dt ', 'MAT_AVAIL_DT' all normalize the same."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


_NORM_DATE_COL = _normalize_token(DATE_COLUMN_NAME)
_NORM_POUNDS_COL = _normalize_token(POUNDS_COLUMN_NAME)

# The six sheets follow a "two-digit year + half-year" naming convention, but
# the exact spelling varies by workbook -- "24HY1", "24H1", "24 HY 1" all mean
# the same thing. Match on the underlying pattern rather than a fixed list of
# exact strings, so real-world spelling differences never cause a sheet to be
# silently skipped.
_HALF_YEAR_SHEET_RE = re.compile(r"^(\d{2})h(?:y)?([12])$")


def find_target_columns(columns: List[object]) -> Tuple[Optional[object], Optional[object]]:
    """Return (date_column, pounds_column) actual-header matches from a list
    of raw column headers, tolerant of spacing/case/punctuation differences.
    """
    date_col, pounds_col = None, None
    for c in columns:
        norm = _normalize_token(c)
        if norm == _NORM_DATE_COL and date_col is None:
            date_col = c
        elif norm == _NORM_POUNDS_COL and pounds_col is None:
            pounds_col = c
    return date_col, pounds_col


def is_half_year_sheet(sheet_name: str) -> bool:
    """True if a sheet name matches the 'YYHY#'/'YYH#' half-year naming
    convention, regardless of spacing, case, or 'H' vs 'HY' spelling."""
    return _HALF_YEAR_SHEET_RE.match(_normalize_token(sheet_name)) is not None


def half_year_sort_key(sheet_name: str) -> Tuple[int, int]:
    """Chronological sort key (year, half) for a matched half-year sheet."""
    m = _HALF_YEAR_SHEET_RE.match(_normalize_token(sheet_name))
    return (int(m.group(1)), int(m.group(2))) if m else (99, 9)


# =============================================================================
# NET POUNDS CLEANING
# =============================================================================

_NUMERIC_STRIP_RE = re.compile(r"[^0-9.\-]")


def clean_net_pounds_value(raw: object) -> Tuple[Optional[float], Optional[str]]:
    """Parse a single Net Pounds cell.

    Returns (value, rejection_reason). value is None if the cell could not
    be safely parsed -- it is NEVER coerced to 0.0. Handles thousands commas,
    stray whitespace, and accounting-style negative parentheses.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, "missing"
    if isinstance(raw, (int, float, np.integer, np.floating)):
        val = float(raw)
        if np.isnan(val):
            return None, "missing"
        return val, None

    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "n/a", "na", "-"}:
        return None, "blank"

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    s_clean = s.replace(",", "").replace(" ", "")
    s_clean = _NUMERIC_STRIP_RE.sub("", s_clean)

    if s_clean in {"", "-", "."}:
        return None, "unparseable"

    # Reject strings with more than one decimal point or malformed sign usage
    if s_clean.count(".") > 1 or s_clean.count("-") > 1:
        return None, "unparseable"

    try:
        val = float(s_clean)
    except ValueError:
        return None, "unparseable"

    if negative:
        val = -val
    return val, None


def clean_net_pounds_series(raw: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Vectorized wrapper around clean_net_pounds_value.

    Returns (cleaned_values, rejection_reason) as two aligned Series.
    """
    parsed = raw.apply(clean_net_pounds_value)
    values = parsed.apply(lambda t: t[0])
    reasons = parsed.apply(lambda t: t[1])
    return values.astype(float), reasons


# =============================================================================
# WORKBOOK LOADING + CLEANING
# =============================================================================

def load_workbook_sheets(source: Union[str, Path, io.BytesIO]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Read every sheet from the workbook and keep the ones that match the
    half-year naming convention (tolerant of 'H2' vs 'HY2' style spelling
    differences), keyed by their ORIGINAL sheet name so reports show exactly
    what's in the actual file. Returns (matched_sheets, ignored_sheet_names)
    sorted chronologically."""
    raw_sheets = pd.read_excel(source, sheet_name=None, engine="openpyxl")
    matched: Dict[str, pd.DataFrame] = {}
    ignored: List[str] = []
    for actual_name, df in raw_sheets.items():
        if is_half_year_sheet(actual_name):
            matched[actual_name] = df
        else:
            ignored.append(actual_name)
    matched = dict(sorted(matched.items(), key=lambda kv: half_year_sort_key(kv[0])))
    return matched, ignored


def clean_single_sheet(
    df: pd.DataFrame, sheet_name: str
) -> Tuple[pd.DataFrame, SheetLoadReport]:
    """Clean one sheet's rows. Returns (clean_df, report).

    clean_df has columns: sheet, date, net_pounds, plus every original
    (non-forecasting) column, preserved for duplicate-row detection.
    Rows with an invalid date OR invalid Net Pounds are dropped from
    clean_df but counted in the report -- never silently zero-filled.
    """
    report = SheetLoadReport(sheet_name=sheet_name, rows_loaded=len(df))

    date_col, pounds_col = find_target_columns(list(df.columns))
    if date_col is None or pounds_col is None:
        report.error = (
            f"Could not find required columns on sheet '{sheet_name}' "
            f"(need something matching '{DATE_COLUMN_NAME}' and '{POUNDS_COLUMN_NAME}')."
        )
        report.rows_missing_required_columns = len(df)
        return pd.DataFrame(columns=["sheet", "date", "net_pounds"]), report

    working = df.copy()
    working["__date__"] = pd.to_datetime(working[date_col], errors="coerce")
    working["__net_pounds__"], reject_reason = clean_net_pounds_series(working[pounds_col])

    invalid_date_mask = working["__date__"].isna()
    invalid_pounds_mask = working["__net_pounds__"].isna()

    report.invalid_date_rows = int(invalid_date_mask.sum())
    report.invalid_pounds_rows = int(invalid_pounds_mask.sum())

    valid_mask = ~(invalid_date_mask | invalid_pounds_mask)
    clean = working.loc[valid_mask].copy()
    report.valid_rows = len(clean)

    clean["sheet"] = sheet_name
    clean["date"] = clean["__date__"].dt.normalize()
    clean["net_pounds"] = clean["__net_pounds__"]
    clean = clean.drop(columns=["__date__", "__net_pounds__"])

    return clean, report


def combine_and_deduplicate(
    cleaned_sheets: Dict[str, pd.DataFrame],
    exclude_duplicates: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """Concatenate all cleaned sheets and flag/optionally drop exact
    duplicate rows (same Order#, Material, dates, and Net Pounds appearing
    more than once anywhere in the workbook).

    Returns (combined_df, duplicate_row_count).
    """
    if not cleaned_sheets:
        return pd.DataFrame(columns=["sheet", "date", "net_pounds"]), 0

    combined = pd.concat(cleaned_sheets.values(), ignore_index=True, sort=False)
    if combined.empty:
        return combined, 0

    dedup_cols = [c for c in combined.columns if c != "sheet"]
    dup_mask = combined.duplicated(subset=dedup_cols, keep="first")
    duplicate_count = int(dup_mask.sum())

    if exclude_duplicates:
        combined = combined.loc[~dup_mask].copy()

    return combined, duplicate_count


def aggregate_daily_series(combined: pd.DataFrame) -> pd.Series:
    """Sum Net Pounds by calendar date across the (already cleaned,
    deduplicated) combined dataframe. Returns a float Series indexed by
    normalized daily Timestamps, sorted chronologically."""
    if combined.empty:
        return pd.Series(dtype=float)
    s = combined.groupby("date")["net_pounds"].sum().sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


def build_dense_calendar(series: pd.Series) -> pd.Series:
    """Reindex a daily series onto a complete calendar (min..max date),
    leaving true gaps as NaN. This is the canonical 'raw truth' series used
    for data-quality reporting and for the 'keep missing as missing'
    treatment option."""
    if series.empty:
        return series
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_index)


def apply_missing_date_treatment(dense_series: pd.Series, treatment: str) -> pd.Series:
    """Produce the series actually fed to the forecasting models, per the
    sidebar's missing-date treatment selection. The dense_series (with true
    NaN gaps) is always preserved separately for reporting/download."""
    if dense_series.empty:
        return dense_series
    if treatment == "Keep missing dates as missing":
        return dense_series.dropna()
    if treatment == "Treat missing dates as zero":
        return dense_series.fillna(0.0)
    if treatment == "Interpolate missing dates for modeling only":
        return dense_series.interpolate(method="linear", limit_direction="both")
    raise ValueError(f"Unknown missing-date treatment: {treatment}")


# =============================================================================
# HISTORICAL / KNOWN-BACKLOG SPLIT  (see ASSUMPTIONS section #1 at top)
# =============================================================================

def split_historical_and_backlog(
    dense_series: pd.Series, as_of_date: pd.Timestamp
) -> Tuple[pd.Series, pd.Series]:
    """Split the dense daily series at as_of_date.

    historical = dates <= as_of_date (used to train every model)
    backlog    = dates >  as_of_date (already-booked future orders, used
                 only as a floor when combining with the statistical
                 forecast -- never used to train a model)
    """
    if dense_series.empty:
        return dense_series, dense_series
    historical = dense_series.loc[dense_series.index <= as_of_date]
    backlog = dense_series.loc[dense_series.index > as_of_date]
    return historical, backlog


def combine_backlog_and_forecast(
    forecast_dates: List[pd.Timestamp],
    forecast_values: np.ndarray,
    backlog: pd.Series,
) -> np.ndarray:
    """Combined Projection(date) = max(statistical forecast, known backlog).
    Backlog acts as a floor because it is already committed. See
    ASSUMPTIONS #1 at the top of this file."""
    backlog_on_dates = np.array(
        [float(backlog.get(d, 0.0)) if not pd.isna(backlog.get(d, 0.0)) else 0.0 for d in forecast_dates]
    )
    return np.maximum(forecast_values, backlog_on_dates)


# =============================================================================
# HIGH-PRODUCTION-DATE UPLIFT  (see ASSUMPTIONS section #2 at top)
# =============================================================================

@dataclass
class HighProductionEvent:
    event_date: pd.Timestamp
    uplift_pct: Optional[float]  # None until resolved against a default/empirical value


def parse_high_production_input(
    date_text: str,
    uploaded_csv: Optional[pd.DataFrame],
) -> List[HighProductionEvent]:
    """Parse the high-production-date input into a list of events, covering
    BOTH past dates (used to label real historical high-production days for
    model training) and future dates (used for the business-adjustment
    ramp). Dates entered without an explicit percent get uplift_pct=None,
    resolved later by resolve_uplift_defaults() once we know whether an
    empirical, data-derived uplift can be computed.
    """
    events: List[HighProductionEvent] = []

    if date_text and date_text.strip():
        for raw in date_text.split(","):
            raw = raw.strip()
            if not raw:
                continue
            parsed = pd.to_datetime(raw, errors="coerce")
            if pd.isna(parsed):
                continue
            events.append(HighProductionEvent(parsed.normalize(), None))

    if uploaded_csv is not None and not uploaded_csv.empty:
        cols = list(uploaded_csv.columns)
        date_col = None
        uplift_col = None
        for c in cols:
            norm = _normalize_token(c)
            if norm in {"date", "highproductiondate"}:
                date_col = c
            elif norm in {"upliftpercent", "uplift", "upliftpct"}:
                uplift_col = c
        if date_col is not None:
            for _, row in uploaded_csv.iterrows():
                parsed = pd.to_datetime(row[date_col], errors="coerce")
                if pd.isna(parsed):
                    continue
                pct = None
                if uplift_col is not None:
                    try:
                        pct = float(row[uplift_col])
                    except (ValueError, TypeError):
                        pct = None
                events.append(HighProductionEvent(parsed.normalize(), pct))

    # de-duplicate on date, keeping an explicit percent over a missing one,
    # and the larger of two explicit percents
    by_date: Dict[pd.Timestamp, Optional[float]] = {}
    for e in events:
        existing = by_date.get(e.event_date, "unset")
        if existing == "unset" or e.uplift_pct is None:
            by_date.setdefault(e.event_date, e.uplift_pct)
        if e.uplift_pct is not None:
            current = by_date.get(e.event_date)
            by_date[e.event_date] = e.uplift_pct if current is None else max(current, e.uplift_pct)
    return [HighProductionEvent(d, p) for d, p in sorted(by_date.items())]


@dataclass
class ClassifiedEvents:
    historical: List[HighProductionEvent]  # date within available historical data, <= as_of
    future_in_horizon: List[HighProductionEvent]  # date within the forecast horizon
    out_of_range: List[HighProductionEvent]  # neither -- has no effect anywhere


def classify_high_production_events(
    events: List[HighProductionEvent],
    historical_series: pd.Series,
    future_dates: List[pd.Timestamp],
) -> ClassifiedEvents:
    """Split parsed events by what they actually do:
      - historical: falls within the range of actual historical data -> used
        to label real high-production days for model training (the
        is_high_production_date feature, and the empirical uplift estimate).
      - future_in_horizon: falls on one of the forecast dates -> drives the
        business-adjustment uplift ramp.
      - out_of_range: neither (e.g. a date before any historical data, or a
        future date beyond the current forecast horizon) -> has no effect,
        and is flagged to the user rather than silently dropped.
    """
    hist_dates = historical_series.dropna().index
    hist_min, hist_max = (hist_dates.min(), hist_dates.max()) if len(hist_dates) else (None, None)
    horizon_set = set(future_dates)

    historical, future_in_horizon, out_of_range = [], [], []
    for e in events:
        if hist_min is not None and hist_min <= e.event_date <= hist_max:
            historical.append(e)
        elif e.event_date in horizon_set:
            future_in_horizon.append(e)
        else:
            out_of_range.append(e)
    return ClassifiedEvents(historical, future_in_horizon, out_of_range)


def compute_empirical_uplift_pct(
    historical_series: pd.Series,
    historical_event_dates: List[pd.Timestamp],
    lookback_weeks: int = 8,
) -> Optional[float]:
    """Estimate a data-driven uplift percentage from real historical
    high-production days: for each labeled date, compare the actual value
    to the average of the same day-of-week over the preceding
    `lookback_weeks` (excluding other labeled days from the baseline).
    Returns None if there isn't enough data to compute a stable estimate.
    """
    s = historical_series.dropna()
    if not historical_event_dates or s.empty:
        return None

    event_set = set(historical_event_dates)
    ratios = []
    for d in historical_event_dates:
        if d not in s.index:
            continue
        actual = s.loc[d]
        baseline_vals = []
        for w in range(1, lookback_weeks + 1):
            ref_date = d - pd.Timedelta(weeks=w)
            if ref_date in s.index and ref_date not in event_set:
                baseline_vals.append(s.loc[ref_date])
        if len(baseline_vals) >= 3 and np.mean(baseline_vals) > 0:
            ratios.append((actual - np.mean(baseline_vals)) / np.mean(baseline_vals))

    if len(ratios) < 2:
        return None
    return float(np.mean(ratios) * 100.0)


def resolve_uplift_defaults(
    events: List[HighProductionEvent], default_pct: float
) -> List[HighProductionEvent]:
    """Fill in uplift_pct=None entries with default_pct (either the flat
    +20% fallback or an empirically-computed value)."""
    return [HighProductionEvent(e.event_date, e.uplift_pct if e.uplift_pct is not None else default_pct)
            for e in events]


def apply_uplift_ramp(
    forecast_dates: List[pd.Timestamp],
    base_values: np.ndarray,
    events: List[HighProductionEvent],
) -> Tuple[np.ndarray, Dict[pd.Timestamp, float]]:
    """Apply the triangular +/-2-day uplift ramp for every event that falls
    (fully or partially) inside forecast_dates. Overlapping ramps stack
    additively before being applied multiplicatively to the base value.
    Events outside forecast_dates simply contribute nothing (safe to pass
    the full historical+future event list here).

    Returns (adjusted_values, per_date_total_uplift_fraction_applied).
    """
    date_index = {d: i for i, d in enumerate(forecast_dates)}
    total_uplift_frac = np.zeros(len(forecast_dates))

    for event in events:
        pct = event.uplift_pct if event.uplift_pct is not None else 0.0
        for offset, weight in RAMP_WEIGHTS.items():
            target_date = event.event_date + pd.Timedelta(days=offset)
            idx = date_index.get(target_date)
            if idx is not None:
                total_uplift_frac[idx] += (pct / 100.0) * weight

    adjusted = base_values * (1.0 + total_uplift_frac)
    per_date_map = {d: float(total_uplift_frac[i]) for i, d in enumerate(forecast_dates)}
    return adjusted, per_date_map


# =============================================================================
# FEATURE ENGINEERING (HistGradientBoostingRegressor only)
# =============================================================================

LAG_DAYS: List[int] = [1, 2, 3, 7, 14, 21, 28]
ROLLING_WINDOWS: List[int] = [7, 14, 28]


def build_hgb_features(
    series: pd.Series, high_production_dates: Optional[set] = None
) -> pd.DataFrame:
    """Build a leakage-safe feature matrix for every date in `series`.

    Every feature derived from the target uses shift(1) or later, i.e. the
    value for date t never uses Net Pounds observed on date t itself. This
    function is used both to build the training matrix and, one row at a
    time, during recursive forecasting.
    """
    high_production_dates = high_production_dates or set()
    df = pd.DataFrame({"y": series.astype(float)})
    df.index = pd.DatetimeIndex(df.index)

    shifted = df["y"].shift(1)  # never allow same-day target to leak in

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"roll_mean_{window}"] = shifted.rolling(window, min_periods=max(2, window // 2)).mean()
    df["roll_std_7"] = shifted.rolling(7, min_periods=3).std()

    dow = df.index.dayofweek
    doy = df.index.dayofyear
    df["day_of_week"] = dow
    df["is_weekend"] = (dow >= 5).astype(int)
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["day_of_year"] = doy
    df["weekly_sin"] = np.sin(2 * np.pi * dow / SEASONAL_PERIOD)
    df["weekly_cos"] = np.cos(2 * np.pi * dow / SEASONAL_PERIOD)
    df["annual_sin"] = np.sin(2 * np.pi * doy / ANNUAL_PERIOD)
    df["annual_cos"] = np.cos(2 * np.pi * doy / ANNUAL_PERIOD)
    df["is_high_production_date"] = df.index.isin(high_production_dates).astype(int)

    return df


HGB_FEATURE_COLUMNS: List[str] = (
    [f"lag_{l}" for l in LAG_DAYS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + ["roll_std_7", "day_of_week", "is_weekend", "month", "quarter",
       "day_of_year", "weekly_sin", "weekly_cos", "annual_sin", "annual_cos",
       "is_high_production_date"]
)


# =============================================================================
# MODELS
# =============================================================================

class BaseForecastModel(ABC):
    name: str = "Base"

    def __init__(self) -> None:
        self._history: Optional[pd.Series] = None

    @abstractmethod
    def fit(self, history: pd.Series, high_production_dates: Optional[set] = None) -> None:
        ...

    @abstractmethod
    def predict(self, horizon: int, future_dates: List[pd.Timestamp],
                high_production_dates: Optional[set] = None) -> ModelForecastResult:
        ...


class SeasonalNaiveModel(BaseForecastModel):
    """Benchmark model: forecast(t) = value observed 7 days earlier in the
    same weekly cycle, wrapping around the end of history as needed."""
    name = "Seasonal Naive"

    def __init__(self, season_length: int = SEASONAL_PERIOD) -> None:
        super().__init__()
        self.season_length = season_length

    def fit(self, history: pd.Series, high_production_dates: Optional[set] = None) -> None:
        self._history = history.dropna()

    def predict(self, horizon: int, future_dates: List[pd.Timestamp],
                high_production_dates: Optional[set] = None) -> ModelForecastResult:
        hist = self._history
        if hist is None or len(hist) < self.season_length:
            return ModelForecastResult(self.name, future_dates, np.full(horizon, np.nan),
                                        skipped=True, skip_reason="Not enough history for seasonal pattern.")
        last_season = hist.values[-self.season_length:]
        reps = int(np.ceil(horizon / self.season_length))
        forecast = np.tile(last_season, reps)[:horizon]
        # simple empirical prediction interval from in-sample seasonal residuals
        resid = hist.values[self.season_length:] - hist.values[:-self.season_length]
        std = np.nanstd(resid) if len(resid) > 1 else 0.0
        lower = np.maximum(forecast - 1.28 * std, 0)
        upper = forecast + 1.28 * std
        return ModelForecastResult(self.name, future_dates, forecast, lower, upper)


class _StatsForecastModelWrapper(BaseForecastModel):
    """Shared wrapper around a single statsforecast model (AutoARIMA/AutoETS)."""

    def __init__(self, name: str, sf_model_factory):
        super().__init__()
        self.name = name
        self._sf_model_factory = sf_model_factory
        self._sf = None

    def fit(self, history: pd.Series, high_production_dates: Optional[set] = None) -> None:
        from statsforecast import StatsForecast
        self._history = history.dropna()
        df = pd.DataFrame({
            "unique_id": "series",
            "ds": self._history.index,
            "y": self._history.values,
        })
        self._sf = StatsForecast(models=[self._sf_model_factory()], freq="D", n_jobs=1)
        self._sf.fit(df)

    def predict(self, horizon: int, future_dates: List[pd.Timestamp],
                high_production_dates: Optional[set] = None) -> ModelForecastResult:
        if self._sf is None or self._history is None or len(self._history) < SEASONAL_PERIOD * 2:
            return ModelForecastResult(self.name, future_dates, np.full(horizon, np.nan),
                                        skipped=True, skip_reason="Not enough history to fit model.")
        fc = self._sf.predict(h=horizon, level=[80])
        model_col = [c for c in fc.columns if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c][0]
        forecast = fc[model_col].to_numpy()
        lo_col, hi_col = f"{model_col}-lo-80", f"{model_col}-hi-80"
        lower = fc[lo_col].to_numpy() if lo_col in fc.columns else None
        upper = fc[hi_col].to_numpy() if hi_col in fc.columns else None
        return ModelForecastResult(self.name, future_dates, forecast, lower, upper)


def make_auto_arima_model() -> _StatsForecastModelWrapper:
    from statsforecast.models import AutoARIMA
    return _StatsForecastModelWrapper("AutoARIMA", lambda: AutoARIMA(season_length=SEASONAL_PERIOD))


def make_auto_ets_model() -> _StatsForecastModelWrapper:
    from statsforecast.models import AutoETS
    return _StatsForecastModelWrapper("AutoETS", lambda: AutoETS(season_length=SEASONAL_PERIOD))


class HistGradientBoostingModel(BaseForecastModel):
    """scikit-learn HistGradientBoostingRegressor with leakage-safe lag /
    rolling / calendar features and recursive multi-step forecasting."""
    name = "HistGradientBoostingRegressor"

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def fit(self, history: pd.Series, high_production_dates: Optional[set] = None) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._history = history.dropna()
        features = build_hgb_features(self._history, high_production_dates or set())
        train = features.dropna(subset=HGB_FEATURE_COLUMNS)
        if len(train) < 30:
            self._model = None
            return
        X = train[HGB_FEATURE_COLUMNS].to_numpy()
        y = train["y"].to_numpy()
        # min_samples_leaf lowered from sklearn's default (20) to 5: a
        # handful of labeled high-production days (sometimes fewer than 20
        # in the whole history) would otherwise never be large enough to
        # form their own leaf, so the is_high_production_date feature would
        # be structurally unusable no matter how informative it is.
        self._model = HistGradientBoostingRegressor(random_state=RANDOM_SEED, min_samples_leaf=5)
        self._model.fit(X, y)

    def predict(self, horizon: int, future_dates: List[pd.Timestamp],
                high_production_dates: Optional[set] = None) -> ModelForecastResult:
        if self._model is None or self._history is None:
            return ModelForecastResult(self.name, future_dates, np.full(horizon, np.nan),
                                        skipped=True, skip_reason="Not enough history to train features (need 28+ days).")

        high_production_dates = high_production_dates or set()
        working_series = self._history.copy()
        preds = []
        for target_date in future_dates:
            # extend the working series with a placeholder index up to target_date
            working_series = working_series.reindex(
                pd.date_range(working_series.index.min(), target_date, freq="D")
            )
            feats_all = build_hgb_features(working_series, high_production_dates)
            row = feats_all.loc[[target_date], HGB_FEATURE_COLUMNS]
            row = row.fillna(0.0)  # sklearn HGB tolerates NaN, but 0 keeps early-horizon rows stable
            pred = float(self._model.predict(row.to_numpy())[0])
            pred = max(0.0, pred)  # never negative
            preds.append(pred)
            working_series.loc[target_date] = pred  # feed back in for the next step (recursive)

        forecast = np.array(preds)
        return ModelForecastResult(self.name, future_dates, forecast, None, None)


def _detect_device() -> str:
    """Auto-detect the best available torch device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def _chronos_weights_cached(model_id: str = CHRONOS_MODEL_ID) -> bool:
    """Best-effort check for whether the Chronos-2 weights are already in
    the local Hugging Face cache, so we never trigger a surprise download."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        return any(repo.repo_id == model_id for repo in cache_info.repos)
    except Exception:
        return False


class Chronos2Model(BaseForecastModel):
    """Amazon Chronos-2 (amazon/chronos-2), run 100% locally. No workbook
    data is ever sent anywhere -- inference happens entirely in-process.

    allow_download controls whether from_pretrained() is permitted to pull
    weights from Hugging Face if they are not already cached locally. If
    they are already cached, loading proceeds regardless (no new network
    activity is triggered either way).
    """
    name = "Amazon Chronos-2"

    def __init__(self, allow_download: bool = False) -> None:
        super().__init__()
        self.allow_download = allow_download
        self.device = "n/a"
        self._pipeline = None
        self._unavailable_reason: Optional[str] = None

        try:
            import chronos  # noqa: F401
        except ImportError:
            self._unavailable_reason = (
                "chronos-forecasting is not installed. See OPTIONAL_MODELS_SETUP.txt."
            )
            return

        cached = _chronos_weights_cached()
        if not cached and not self.allow_download:
            self._unavailable_reason = (
                "Chronos-2 weights are not cached locally and download is disabled "
                "in the sidebar. Enable download, or pre-download using "
                "OPTIONAL_MODELS_SETUP.txt, then re-run."
            )
            return

        self.device = _detect_device()

    def fit(self, history: pd.Series) -> None:
        self._history = history.dropna()
        if self._unavailable_reason:
            return
        try:
            from chronos import Chronos2Pipeline
            self._pipeline = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL_ID, device_map=self.device)
        except Exception as exc:  # model init failures must never crash the app
            self._unavailable_reason = f"Chronos-2 failed to initialize: {exc}"
            self._pipeline = None

    def predict(self, horizon: int, future_dates: List[pd.Timestamp],
                high_production_dates: Optional[set] = None) -> ModelForecastResult:
        if self._unavailable_reason or self._pipeline is None or self._history is None:
            return ModelForecastResult(
                self.name, future_dates, np.full(horizon, np.nan),
                skipped=True, skip_reason=self._unavailable_reason or "Chronos-2 unavailable.",
                device=self.device,
            )
        try:
            context_df = pd.DataFrame({
                "id": "series",
                "timestamp": self._history.index,
                "target": self._history.values,
            })
            pred_df = self._pipeline.predict_df(
                context_df,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
            forecast = np.maximum(pred_df["0.5"].to_numpy(), 0.0)
            lower = np.maximum(pred_df["0.1"].to_numpy(), 0.0)
            upper = np.maximum(pred_df["0.9"].to_numpy(), 0.0)
            return ModelForecastResult(self.name, future_dates, forecast, lower, upper, device=self.device)
        except Exception as exc:
            return ModelForecastResult(
                self.name, future_dates, np.full(horizon, np.nan),
                skipped=True, skip_reason=f"Chronos-2 inference failed: {exc}", device=self.device,
            )


def build_model_registry(chronos_allow_download: bool = False) -> Dict[str, BaseForecastModel]:
    """Instantiate a fresh set of all five models. Called once per full
    run/backtest so no state leaks between windows."""
    return {
        "Seasonal Naive": SeasonalNaiveModel(),
        "AutoARIMA": _StatsForecastModelWrapper("AutoARIMA", lambda: __import__(
            "statsforecast.models", fromlist=["AutoARIMA"]).AutoARIMA(season_length=SEASONAL_PERIOD)),
        "AutoETS": _StatsForecastModelWrapper("AutoETS", lambda: __import__(
            "statsforecast.models", fromlist=["AutoETS"]).AutoETS(season_length=SEASONAL_PERIOD)),
        "HistGradientBoostingRegressor": HistGradientBoostingModel(),
        "Amazon Chronos-2": Chronos2Model(allow_download=chronos_allow_download),
    }


MODEL_DISPLAY_ORDER: List[str] = [
    "Seasonal Naive", "AutoARIMA", "AutoETS", "HistGradientBoostingRegressor", "Amazon Chronos-2",
]


# =============================================================================
# METRICS
# =============================================================================

def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / denom


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)
    return 100.0 * float(np.mean(ratio))


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, in_sample_history: np.ndarray,
                  seasonal_period: int = SEASONAL_PERIOD) -> float:
    if len(in_sample_history) <= seasonal_period:
        return np.nan
    naive_errors = np.abs(in_sample_history[seasonal_period:] - in_sample_history[:-seasonal_period])
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def compute_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean forecast error / bias: positive means the model over-forecasts."""
    return float(np.mean(y_pred - y_true))


def compute_abs_pct_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total_true = np.sum(y_true)
    if total_true == 0:
        return np.nan
    return 100.0 * abs(np.sum(y_pred) - total_true) / abs(total_true)


def wape_based_accuracy(wape: float) -> float:
    if np.isnan(wape):
        return np.nan
    return max(0.0, 100.0 - wape)


# =============================================================================
# ROLLING-ORIGIN BACKTESTING
# =============================================================================

def compute_backtest_origins(n_obs: int, horizon: int,
                              preferred_windows: int = N_BACKTEST_WINDOWS_PREFERRED,
                              min_train: int = MIN_TRAIN_DAYS) -> List[int]:
    """Return a list of train-set lengths (one per backtest origin), oldest
    first. Non-overlapping horizon-sized test windows are used, walking
    backwards from the end of history. Uses fewer windows if history is
    short; returns [] if there isn't even enough for one window."""
    if n_obs < min_train + horizon:
        return []
    max_possible = (n_obs - min_train) // horizon
    n_windows = max(1, min(preferred_windows, max_possible))
    origins = [n_obs - horizon * k for k in range(n_windows, 0, -1)]
    return [o for o in origins if o >= min_train]


def run_backtest(
    model_factory,
    model_name: str,
    modeling_series: pd.Series,
    horizon: int,
    high_production_dates: Optional[set] = None,
) -> Tuple[List[BacktestWindowResult], float]:
    """Rolling-origin backtest for a single model. Returns
    (window_results, total_runtime_seconds). A failure in one window is
    caught and recorded -- it never stops the rest of the backtest or the
    other models.

    high_production_dates is passed to both fit() and predict() at every
    window. This is safe leakage-wise: it's a calendar-derived indicator
    (is this date a known/labeled high-production day?), never derived from
    the target values themselves, so knowing it at any forecast origin
    doesn't leak future information about Net Pounds.
    """
    high_production_dates = high_production_dates or set()
    origins = compute_backtest_origins(len(modeling_series), horizon)
    results: List[BacktestWindowResult] = []
    total_runtime = 0.0

    for train_len in origins:
        train = modeling_series.iloc[:train_len]
        test = modeling_series.iloc[train_len:train_len + horizon]
        if len(test) < horizon:
            continue
        origin_date = train.index[-1]
        future_dates = list(test.index)

        t0 = time.time()
        try:
            model = model_factory()
            model.fit(train, high_production_dates=high_production_dates)
            res = model.predict(horizon, future_dates, high_production_dates=high_production_dates)
            dt = time.time() - t0
            total_runtime += dt
            if res.skipped or np.any(np.isnan(res.forecast)):
                results.append(BacktestWindowResult(model_name, origin_date, False, dt,
                                                      error=res.skip_reason or "NaN forecast produced"))
            else:
                results.append(BacktestWindowResult(
                    model_name, origin_date, True, dt,
                    y_true=test.to_numpy(), y_pred=res.forecast,
                ))
        except Exception as exc:
            dt = time.time() - t0
            total_runtime += dt
            results.append(BacktestWindowResult(
                model_name, origin_date, False, dt,
                error=f"{type(exc).__name__}: {exc}",
            ))
    return results, total_runtime



def summarize_backtest(
    model_name: str, windows: List[BacktestWindowResult], full_history_for_mase: np.ndarray,
) -> ModelMetrics:
    """Aggregate per-window backtest results into a single ModelMetrics row."""
    successes = [w for w in windows if w.success]
    failures = [w for w in windows if not w.success]
    total_runtime = sum(w.runtime_seconds for w in windows)

    if not windows:
        return ModelMetrics(model_name, skipped=True, skip_reason="No backtest windows could be run "
                             "(insufficient history for the chosen horizon).")

    failure_rate = len(failures) / len(windows)
    if not successes:
        return ModelMetrics(model_name, n_success_windows=0, n_failed_windows=len(failures),
                             runtime_seconds=total_runtime, skipped=True,
                             skip_reason="Every backtest window failed: " +
                             (failures[0].error if failures else "unknown error"))

    y_true_all = np.concatenate([w.y_true for w in successes])
    y_pred_all = np.concatenate([w.y_pred for w in successes])

    wape = compute_wape(y_true_all, y_pred_all)
    metrics = ModelMetrics(
        model_name=model_name,
        wape=wape,
        mae=compute_mae(y_true_all, y_pred_all),
        rmse=compute_rmse(y_true_all, y_pred_all),
        smape=compute_smape(y_true_all, y_pred_all),
        mase=compute_mase(y_true_all, y_pred_all, full_history_for_mase),
        bias=compute_bias(y_true_all, y_pred_all),
        abs_pct_bias=compute_abs_pct_bias(y_true_all, y_pred_all),
        accuracy_score=wape_based_accuracy(wape),
        runtime_seconds=total_runtime,
        n_success_windows=len(successes),
        n_failed_windows=len(failures),
        skipped=False,
    )
    if failure_rate > MAX_WINDOW_FAILURE_RATE:
        metrics.skipped = True
        metrics.skip_reason = (
            f"{len(failures)}/{len(windows)} backtest windows failed "
            f"({failure_rate:.0%} > {MAX_WINDOW_FAILURE_RATE:.0%} threshold)."
        )
    return metrics


# =============================================================================
# MODEL SELECTION
# =============================================================================

@dataclass
class ModelSelectionResult:
    leaderboard: pd.DataFrame
    recommended_model: str
    beat_seasonal_naive: bool
    reason: str


def select_recommended_model(all_metrics: List[ModelMetrics]) -> ModelSelectionResult:
    """Apply the business's model-selection rules:
      1. Exclude skipped models.
      2. Exclude models failing more than 20% of backtest windows
         (already reflected in `.skipped` by summarize_backtest).
      3. Rank eligible models by lowest average WAPE.
      4. Tie-break: lowest MAE, then lowest absolute bias.
      5. A complex model must beat Seasonal Naive's WAPE by at least
         MEANINGFUL_WAPE_IMPROVEMENT_PP points to be recommended over it.
    """
    board = pd.DataFrame([{
        "model": m.model_name, "wape": m.wape, "mae": m.mae, "rmse": m.rmse,
        "smape": m.smape, "mase": m.mase, "bias": m.bias, "abs_pct_bias": m.abs_pct_bias,
        "accuracy_score": m.accuracy_score, "runtime_seconds": m.runtime_seconds,
        "success_windows": m.n_success_windows, "failed_windows": m.n_failed_windows,
        "skipped": m.skipped, "skip_reason": m.skip_reason,
    } for m in all_metrics])

    naive_row = board.loc[board["model"] == "Seasonal Naive"]
    naive_wape = float(naive_row["wape"].iloc[0]) if not naive_row.empty and not naive_row["skipped"].iloc[0] else np.nan

    eligible = board.loc[~board["skipped"]].copy()
    eligible = eligible.sort_values(by=["wape", "mae", "abs_pct_bias"],
                                     ascending=[True, True, True], na_position="last")

    board = board.sort_values(by=["skipped", "wape"], ascending=[True, True], na_position="last").reset_index(drop=True)

    if eligible.empty:
        return ModelSelectionResult(board, "None", False,
                                     "No model produced usable backtest results; check the data-quality report.")

    winner = eligible.iloc[0]
    winner_name = winner["model"]

    if winner_name == "Seasonal Naive" or np.isnan(naive_wape):
        return ModelSelectionResult(
            board, winner_name, winner_name != "Seasonal Naive",
            "Seasonal Naive was the best-performing eligible model on backtested WAPE."
            if winner_name == "Seasonal Naive" else
            "Seasonal Naive could not be evaluated for comparison; recommending the best eligible model by WAPE.",
        )

    improvement = naive_wape - float(winner["wape"])
    if improvement >= MEANINGFUL_WAPE_IMPROVEMENT_PP:
        reason = (
            f"{winner_name} beat Seasonal Naive by {improvement:.1f} WAPE points "
            f"({winner['wape']:.1f} vs {naive_wape:.1f}), a meaningful improvement "
            f"(threshold: {MEANINGFUL_WAPE_IMPROVEMENT_PP:.1f} points)."
        )
        return ModelSelectionResult(board, winner_name, True, reason)
    else:
        reason = (
            f"{winner_name} had a lower WAPE than Seasonal Naive ({winner['wape']:.1f} vs {naive_wape:.1f}), "
            f"but the {improvement:.1f}-point gap doesn't clear the "
            f"{MEANINGFUL_WAPE_IMPROVEMENT_PP:.1f}-point bar for a 'meaningful' improvement, "
            f"so Seasonal Naive is recommended as the safer, simpler choice."
        )
        return ModelSelectionResult(board, "Seasonal Naive", False, reason)


# =============================================================================
# FINAL FORECAST TABLE + EXPORTS
# =============================================================================

def build_final_forecast_table(
    model_name: str,
    forecast_dates: List[pd.Timestamp],
    original_forecast: np.ndarray,
    adjusted_forecast: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    backlog: pd.Series,
    uplift_by_date: Dict[pd.Timestamp, float],
) -> pd.DataFrame:
    """Build the standard final-forecast output schema."""
    now = pd.Timestamp.now()
    rows = []
    for i, d in enumerate(forecast_dates):
        backlog_val = backlog.get(d, np.nan)
        rows.append({
            "forecast_date": d.date(),
            "model": model_name,
            "original_forecast_net_pounds": round(float(original_forecast[i]), 2),
            "adjusted_forecast_net_pounds": round(float(adjusted_forecast[i]), 2),
            "lower_interval": round(float(lower[i]), 2) if lower is not None else np.nan,
            "upper_interval": round(float(upper[i]), 2) if upper is not None else np.nan,
            "is_high_production_date": uplift_by_date.get(d, 0.0) > 0,
            "uplift_percent": round(uplift_by_date.get(d, 0.0) * 100, 2),
            "known_backlog_net_pounds": round(float(backlog_val), 2) if not pd.isna(backlog_val) else 0.0,
            "generated_at": now.isoformat(timespec="seconds"),
        })
    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_combined_excel_workbook(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write every named dataframe to its own sheet in one xlsx workbook,
    returned as raw bytes suitable for a Streamlit download button."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31] if name else "Sheet"
            (df if not df.empty else pd.DataFrame({"note": ["no data"]})).to_excel(
                writer, sheet_name=safe_name, index=False
            )
    return buffer.getvalue()


# =============================================================================
# FULL PIPELINE ORCHESTRATION
# =============================================================================

@dataclass
class DataPrepResult:
    """Output of the cheap data-loading step. Cache this on (file, horizon,
    missing-date treatment, as-of date, dedup toggle) -- it never depends on
    high-production dates and is fast enough to always recompute."""
    dq_report: DataQualityReport
    dense_series: pd.Series          # full calendar, true NaN gaps, historical only
    historical_series: pd.Series     # same as dense_series (alias kept for clarity elsewhere)
    modeling_series: pd.Series       # after missing-date treatment, used to fit models
    backlog_series: pd.Series        # known future backlog beyond as_of_date
    as_of_date: pd.Timestamp
    horizon: int
    future_dates: List[pd.Timestamp]
    combined_raw_rows: pd.DataFrame


def prepare_data(
    source: Union[str, Path, io.BytesIO],
    horizon: int,
    missing_date_treatment: str,
    as_of_date: pd.Timestamp,
    exclude_duplicates: bool,
    progress_callback=None,
) -> DataPrepResult:
    """Read, clean, aggregate, and split the workbook. Cheap (no model
    fitting/backtesting) -- safe to re-run on every Streamlit rerun."""

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    _progress("Reading workbook...")
    raw_sheets, ignored_sheet_names = load_workbook_sheets(source)
    if not raw_sheets:
        raise ValueError(
            "No sheets matching the half-year naming convention (e.g. '24HY1', '24H2') "
            "were found in this workbook. Found sheets: "
            + (", ".join(ignored_sheet_names) if ignored_sheet_names else "(workbook appears empty)")
        )

    notes: List[str] = []
    if len(raw_sheets) != 6:
        notes.append(
            f"Found {len(raw_sheets)} half-year sheet(s) ({', '.join(raw_sheets.keys())}) -- "
            f"expected 6. Double-check sheet names if this is unexpected."
        )
    if ignored_sheet_names:
        notes.append(
            f"Ignored {len(ignored_sheet_names)} sheet(s) that didn't match the half-year "
            f"naming pattern: {', '.join(ignored_sheet_names)}."
        )

    _progress("Cleaning sheets...")
    cleaned: Dict[str, pd.DataFrame] = {}
    sheet_reports: List[SheetLoadReport] = []
    for sheet_name, sheet_df in raw_sheets.items():
        clean, rep = clean_single_sheet(sheet_df, sheet_name)
        cleaned[sheet_name] = clean
        sheet_reports.append(rep)

    _progress("Combining sheets and checking for duplicates...")
    combined, duplicate_count = combine_and_deduplicate(cleaned, exclude_duplicates=exclude_duplicates)

    dq = DataQualityReport(
        sheet_reports=sheet_reports,
        notes=notes,
        total_rows_loaded=sum(r.rows_loaded for r in sheet_reports),
        total_invalid_dates=sum(r.invalid_date_rows for r in sheet_reports),
        total_invalid_pounds=sum(r.invalid_pounds_rows for r in sheet_reports),
        total_valid_rows=sum(r.valid_rows for r in sheet_reports),
        duplicate_row_count=duplicate_count,
        duplicates_excluded=exclude_duplicates,
        missing_date_treatment=missing_date_treatment,
        as_of_date=as_of_date,
    )

    if combined.empty:
        raise ValueError("No valid rows survived data cleaning -- check the data-quality report below.")

    _progress("Aggregating daily Net Pounds...")
    daily = aggregate_daily_series(combined)
    dense = build_dense_calendar(daily)
    dq.min_date, dq.max_date = dense.index.min(), dense.index.max()
    dq.missing_calendar_dates = list(dense[dense.isna()].index)

    historical, backlog = split_historical_and_backlog(dense, as_of_date)
    if historical.dropna().empty:
        raise ValueError(
            f"No historical data on or before {as_of_date.date()}. "
            "Check the as-of date setting in the sidebar."
        )
    dq.historical_days = len(historical)
    dq.backlog_days = int((~backlog.isna()).sum())

    modeling_series = apply_missing_date_treatment(historical, missing_date_treatment)
    last_hist_date = historical.dropna().index.max()
    future_dates = list(pd.date_range(last_hist_date + pd.Timedelta(days=1), periods=horizon, freq="D"))

    return DataPrepResult(
        dq_report=dq, dense_series=dense, historical_series=historical,
        modeling_series=modeling_series, backlog_series=backlog, as_of_date=as_of_date,
        horizon=horizon, future_dates=future_dates, combined_raw_rows=combined,
    )


@dataclass
class ModelFitResult:
    """Output of the expensive step: backtesting + final model fit/predict.
    Depends on the *committed* high-production date labels (they become a
    training feature), so this should only be recomputed when the data
    changes or the user explicitly commits new high-production dates."""
    fitted_models: Dict[str, BaseForecastModel]
    base_forecasts: Dict[str, ModelForecastResult]     # already reflects committed high-production labels
    backtest_metrics: List[ModelMetrics]
    selection: ModelSelectionResult


def fit_and_backtest_models(
    data: DataPrepResult,
    high_production_dates: set,
    chronos_allow_download: bool,
    progress_callback=None,
) -> ModelFitResult:
    """Backtest and final-fit every model, using high_production_dates (the
    full committed set -- historical labels AND future business dates) as a
    calendar feature everywhere. This is the slow step."""

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    modeling_series, horizon, future_dates = data.modeling_series, data.horizon, data.future_dates

    all_metrics: List[ModelMetrics] = []
    fitted_models: Dict[str, BaseForecastModel] = {}
    base_forecasts: Dict[str, ModelForecastResult] = {}

    model_factories = {
        "Seasonal Naive": lambda: SeasonalNaiveModel(),
        "AutoARIMA": make_auto_arima_model,
        "AutoETS": make_auto_ets_model,
        "HistGradientBoostingRegressor": lambda: HistGradientBoostingModel(),
        "Amazon Chronos-2": lambda: Chronos2Model(allow_download=chronos_allow_download),
    }

    for name in MODEL_DISPLAY_ORDER:
        factory = model_factories[name]
        _progress(f"Backtesting {name}...")
        windows, _ = run_backtest(factory, name, modeling_series, horizon, high_production_dates)
        metrics = summarize_backtest(name, windows, modeling_series.values)
        all_metrics.append(metrics)

        _progress(f"Fitting final {name} model on full history...")
        try:
            model = factory()
            model.fit(modeling_series, high_production_dates=high_production_dates)
            fc = model.predict(horizon, future_dates, high_production_dates=high_production_dates)
            if not fc.skipped:
                # Final displayed/exported forecasts are never negative, regardless
                # of model (backtest-time metrics deliberately stay unclipped so
                # they still reveal an unstable model).
                fc.forecast = np.maximum(fc.forecast, 0.0)
                if fc.lower is not None:
                    fc.lower = np.maximum(fc.lower, 0.0)
                if fc.upper is not None:
                    fc.upper = np.maximum(fc.upper, 0.0)
        except Exception as exc:
            fc = ModelForecastResult(name, future_dates, np.full(horizon, np.nan),
                                      skipped=True, skip_reason=f"{type(exc).__name__}: {exc}")
            model = None
        fitted_models[name] = model
        base_forecasts[name] = fc

    selection = select_recommended_model(all_metrics)
    return ModelFitResult(fitted_models, base_forecasts, all_metrics, selection)


def compute_business_adjustment(
    data: DataPrepResult,
    fit_result: ModelFitResult,
    future_events: List[HighProductionEvent],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Cheap, always-fresh step: for every successful model, combine the
    (already-computed, already high-production-aware) forecast with the
    known backlog, then apply the uplift ramp for in-horizon events. No
    model re-fitting or re-predicting happens here."""
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for name, fc in fit_result.base_forecasts.items():
        if fc.skipped:
            continue
        combined = combine_backlog_and_forecast(data.future_dates, fc.forecast, data.backlog_series)
        adjusted, uplift_map = apply_uplift_ramp(data.future_dates, combined, future_events)
        results[name] = {
            "original_forecast": fc.forecast,
            "combined_projection": combined,
            "adjusted_forecast": adjusted,
            "lower": fc.lower,
            "upper": fc.upper,
            "uplift_map": uplift_map,
        }
    return results


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def chart_model_forecast(
    model_name: str,
    historical_tail: pd.Series,
    future_dates: List[pd.Timestamp],
    original: np.ndarray,
    adjusted: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    high_production_dates: List[pd.Timestamp],
    highlight: bool = False,
) -> go.Figure:
    fig = go.Figure()
    line_color = COLOR_PRIMARY if highlight else COLOR_NEUTRAL_MED

    fig.add_trace(go.Scatter(
        x=historical_tail.index, y=historical_tail.values, mode="lines",
        name="Historical Actuals", line=dict(color=COLOR_NEUTRAL_DARK, width=1.5),
    ))

    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself", fillcolor="rgba(0,62,114,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Prediction Interval", showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=future_dates, y=original, mode="lines+markers", name="Original Forecast",
        line=dict(color=line_color, width=2, dash="dot"), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=adjusted, mode="lines+markers", name="Business-Adjusted Forecast",
        line=dict(color=COLOR_ACCENT, width=3), marker=dict(size=6),
    ))

    for hp_date in high_production_dates:
        fig.add_vline(x=hp_date.timestamp() * 1000, line_dash="dash",
                       line_color=COLOR_WARN, opacity=0.6)

    title = f"{model_name}" + ("  (Recommended)" if highlight else "")
    fig.update_layout(**_base_layout(title))
    return fig


def chart_historical_line(dense_series: pd.Series, backlog: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dense_series.index, y=dense_series.values, mode="lines",
                              name="Historical Actuals (daily)", line=dict(color=COLOR_PRIMARY, width=1.5)))
    if not backlog.dropna().empty:
        fig.add_trace(go.Scatter(x=backlog.index, y=backlog.values, mode="lines",
                                  name="Known Future Backlog", line=dict(color=COLOR_ACCENT, width=1.5, dash="dot")))
    fig.update_layout(**_base_layout("Historical Daily Net Pounds"))
    return fig


def chart_backtest_actual_vs_predicted(model_name: str, windows: List[BacktestWindowResult]) -> go.Figure:
    fig = go.Figure()
    successes = [w for w in windows if w.success]
    if not successes:
        fig.update_layout(**_base_layout(f"{model_name}: Backtest Actual vs. Predicted (no successful windows)"))
        return fig
    all_true, all_pred, all_x = [], [], []
    cursor = 0
    for w in successes:
        n = len(w.y_true)
        all_true.extend(w.y_true)
        all_pred.extend(w.y_pred)
        all_x.extend(range(cursor, cursor + n))
        cursor += n
    fig.add_trace(go.Scatter(x=all_x, y=all_true, mode="lines", name="Actual",
                              line=dict(color=COLOR_NEUTRAL_DARK, width=2)))
    fig.add_trace(go.Scatter(x=all_x, y=all_pred, mode="lines", name="Predicted",
                              line=dict(color=COLOR_ACCENT, width=2, dash="dot")))
    layout = _base_layout(f"{model_name}: Backtest Actual vs. Predicted (all windows, concatenated)")
    layout["xaxis"] = dict(title="Backtest step (concatenated across windows)", showgrid=True, gridcolor=COLOR_GRID)
    fig.update_layout(**layout)
    return fig


def chart_combined_all_models(
    historical_tail: pd.Series,
    future_dates: List[pd.Timestamp],
    adjusted_by_model: Dict[str, np.ndarray],
    recommended: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_tail.index, y=historical_tail.values, mode="lines",
                              name="Historical Actuals", line=dict(color=COLOR_NEUTRAL_DARK, width=1.5)))
    for i, (name, values) in enumerate(adjusted_by_model.items()):
        is_rec = name == recommended
        fig.add_trace(go.Scatter(
            x=future_dates, y=values, mode="lines+markers", name=name,
            line=dict(color=MODEL_COLOR_SEQUENCE[i % len(MODEL_COLOR_SEQUENCE)],
                       width=3 if is_rec else 1.5, dash=None if is_rec else "dot"),
            marker=dict(size=6 if is_rec else 4),
        ))
    fig.update_layout(**_base_layout("Combined Forecast Comparison -- All Successful Models"))
    return fig


def chart_rolling_windows(series: pd.Series, model_name: str) -> go.Figure:
    """Page-3 diagnostic: pure line graphs of the rolling techniques feeding
    a model, with no numeric tables."""
    s = series.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="Daily Actual",
                              line=dict(color=COLOR_NEUTRAL_MED, width=1)))
    for window, color in zip([7, 14, 28], [COLOR_PRIMARY, COLOR_ACCENT, "#5C8A8A"]):
        roll = s.rolling(window, min_periods=max(2, window // 2)).mean()
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines",
                                  name=f"{window}-day rolling mean", line=dict(color=color, width=2)))
    fig.update_layout(**_base_layout(f"{model_name}: Rolling-Window Patterns"))
    return fig


def chart_seasonal_profile(series: pd.Series, model_name: str) -> go.Figure:
    """Average net pounds by day-of-week, shown as a line (no numeric table)."""
    s = series.dropna()
    dow_avg = s.groupby(s.index.dayofweek).mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=[dow_avg.get(i, np.nan) for i in range(7)],
                              mode="lines+markers", line=dict(color=COLOR_PRIMARY, width=3),
                              marker=dict(size=8, color=COLOR_ACCENT)))
    layout = _base_layout(f"{model_name}: Average Weekly Seasonal Profile", y_title="Avg Net Pounds")
    layout["xaxis"] = dict(title="Day of Week", showgrid=True, gridcolor=COLOR_GRID)
    fig.update_layout(**layout)
    return fig


def chart_quantile_fan(model_name: str, future_dates: List[pd.Timestamp],
                        median: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(future_dates) + list(future_dates)[::-1],
                              y=list(upper) + list(lower)[::-1], fill="toself",
                              fillcolor="rgba(245,179,52,0.25)", line=dict(color="rgba(0,0,0,0)"),
                              name="10th-90th percentile"))
    fig.add_trace(go.Scatter(x=future_dates, y=median, mode="lines+markers",
                              line=dict(color=COLOR_PRIMARY, width=3), name="Median forecast"))
    fig.update_layout(**_base_layout(f"{model_name}: Probabilistic Forecast Fan"))
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

_CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {COLOR_NEUTRAL_LIGHT}; }}
    h1, h2, h3 {{ color: {COLOR_PRIMARY}; font-family: 'Segoe UI', Helvetica, Arial, sans-serif; }}
    [data-testid="stSidebar"] {{ background-color: white; border-right: 1px solid {COLOR_GRID}; }}
    div.stButton > button:first-child {{
        background-color: {COLOR_PRIMARY}; color: white; border: none; border-radius: 4px;
        font-weight: 600; padding: 0.5rem 1.25rem;
    }}
    div.stButton > button:first-child:hover {{ background-color: #002B4D; color: white; }}
    .glc-badge {{
        display: inline-block; background-color: {COLOR_ACCENT}; color: {COLOR_NEUTRAL_DARK};
        padding: 2px 10px; border-radius: 3px; font-weight: 600; font-size: 0.8rem;
    }}
    .glc-card {{
        background-color: white; border: 1px solid {COLOR_GRID}; border-radius: 6px;
        padding: 1rem 1.25rem; margin-bottom: 0.75rem;
    }}
    [data-testid="stMetricValue"] {{ color: {COLOR_PRIMARY}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: white; border-radius: 4px 4px 0 0; padding: 8px 18px;
        border: 1px solid {COLOR_GRID}; font-weight: 600; color: {COLOR_NEUTRAL_MED};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {COLOR_PRIMARY} !important; color: white !important; }}
</style>
"""


def _settings_signature(file_token: str, horizon: int, treatment: str, as_of: pd.Timestamp,
                         dedup: bool, chronos_dl: bool) -> tuple:
    return (file_token, horizon, treatment, as_of.isoformat(), dedup, chronos_dl)


def _resolve_data_source():
    """Sidebar data-source controls. Returns (source, file_token, description)."""
    st.sidebar.markdown("### 1. Data Source")
    uploaded = st.sidebar.file_uploader(
        "Upload a Sales Order workbook (optional)", type=["xlsx"],
        help="Leave empty to use the default workbook shipped next to this script.",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        return io.BytesIO(data), f"upload:{uploaded.name}:{len(data)}", f"Uploaded file: {uploaded.name}"

    if DEFAULT_WORKBOOK_PATH.exists():
        stat = DEFAULT_WORKBOOK_PATH.stat()
        token = f"default:{DEFAULT_WORKBOOK_PATH}:{stat.st_mtime}:{stat.st_size}"
        return DEFAULT_WORKBOOK_PATH, token, f"This is the data being used: {DEFAULT_WORKBOOK_PATH.name}"

    return None, None, None


def _sidebar_controls():
    st.sidebar.markdown("### 2. Forecast Settings")
    horizon = st.sidebar.number_input(
        "Forecast horizon (calendar days)", min_value=7, max_value=MAX_HORIZON_DAYS,
        value=DEFAULT_HORIZON_DAYS, step=1,
    )
    as_of = st.sidebar.date_input("As-of date (historical cutoff)", value=date.today())
    treatment = st.sidebar.radio(
        "Missing calendar dates",
        ["Keep missing dates as missing", "Treat missing dates as zero",
         "Interpolate missing dates for modeling only"],
        index=2,
        help="A missing date is not automatically a zero-production date -- choose how to treat gaps.",
    )
    dedup = st.sidebar.checkbox("Exclude exact duplicate order rows before aggregating", value=True)

    st.sidebar.markdown("### 3. Amazon Chronos-2 (optional)")
    chronos_dl = st.sidebar.checkbox(
        "Allow downloading Chronos-2 weights from Hugging Face if not cached",
        value=False,
        help="Leave unchecked to keep the app fully offline. See OPTIONAL_MODELS_SETUP.txt "
             "to pre-download the weights yourself instead.",
    )

    return horizon, pd.Timestamp(as_of), treatment, dedup, chronos_dl


def _high_production_panel(container):
    """Renders in the Overview tab's right-hand column (not the sidebar) so
    it sits right next to the model forecasts, as requested."""
    container.markdown("#### Expected High-Production Dates")
    container.caption(
        "Enter dates inside the forecast horizon. Default uplift is +20%, applied as a "
        "ramp across the date +/- 2 days -- see Assumptions below."
    )
    date_text = container.text_area(
        "Comma-separated dates (e.g. 2026-09-20, 2026-09-21)", value="", height=80, key="hp_date_text",
    )
    csv_file = container.file_uploader(
        "...or upload a CSV with columns: date, uplift_percent", type=["csv"], key="hp_csv",
    )
    uploaded_csv_df = None
    if csv_file is not None:
        try:
            uploaded_csv_df = pd.read_csv(csv_file)
        except Exception as exc:
            container.error(f"Could not read CSV: {exc}")
    update_clicked = container.button("Update Forecast", key="update_forecast_btn")

    with container.expander("Assumptions used in this forecast"):
        container.markdown(
            "- **Combined Projection** = the higher of the statistical forecast and any "
            "already-booked (known backlog) quantity for that date.\n"
            "- **Business-Adjusted Forecast** applies your high-production dates as a "
            "triangular ramp (25% / 50% / 100% / 50% / 25% of the uplift, centered on the date).\n"
            "- Models are trained only on historical actuals (dates on or before the as-of date); "
            "future-dated open orders are never used as training data."
        )
    return date_text, uploaded_csv_df, update_clicked


def _events_key(events: List[HighProductionEvent]) -> tuple:
    """Hashable signature of an event list's DATES only (not percents) --
    used to decide whether the expensive fit/backtest step must re-run.
    Percent-only edits never require a re-run (they only affect the cheap
    uplift-ramp layer)."""
    return tuple(sorted(e.event_date.isoformat() for e in events))


def main() -> None:
    st.set_page_config(
        page_title="GLC Sales Order Forecasting",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    st.title("Sales Order Net Pounds Forecasting")
    st.caption("Great Lakes Cheese -- daily Net Pounds forecast, powered by five independently "
               "benchmarked models. Historical data, an already-booked backlog, and your "
               "expected high-production dates are combined into one business-adjusted forecast.")

    source, file_token, description = _resolve_data_source()
    horizon, as_of_date, treatment, dedup, chronos_dl = _sidebar_controls()

    if source is None:
        st.warning(
            f"No workbook found. Place **{DEFAULT_WORKBOOK_PATH.name}** next to this script, "
            "or upload one using the sidebar."
        )
        st.stop()

    st.markdown(f'<span class="glc-badge">Data source</span> &nbsp; {description}', unsafe_allow_html=True)

    # ---- Step 1: cheap data load/clean/aggregate (always safe to re-run) ----
    data_signature = _settings_signature(file_token, horizon, treatment, as_of_date, dedup, False)
    needs_data_reload = (
        "data_prep" not in st.session_state
        or st.session_state.get("data_signature") != data_signature
    )
    if needs_data_reload:
        try:
            with st.spinner("Loading and cleaning the workbook..."):
                data = prepare_data(
                    source=source, horizon=int(horizon), missing_date_treatment=treatment,
                    as_of_date=as_of_date, exclude_duplicates=dedup,
                )
            st.session_state["data_prep"] = data
            st.session_state["data_signature"] = data_signature
        except Exception as exc:
            st.error(f"Could not load the workbook: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            st.stop()
    data: DataPrepResult = st.session_state["data_prep"]

    # ---- Tabs + high-production panel drawn now so we have this run's raw
    # input values before deciding whether to re-fit/backtest models. ----
    tab1, tab2, tab3 = st.tabs(["Overview", "Metrics & Backtesting", "Model Diagnostics"])

    with tab1:
        col_main, col_side = st.columns([0.7, 0.3])
        with col_side:
            date_text, uploaded_csv_df, update_clicked = _high_production_panel(col_side)

    current_events = resolve_uplift_defaults(
        parse_high_production_input(date_text, uploaded_csv_df), DEFAULT_UPLIFT_PCT
    )
    classified_now = classify_high_production_events(current_events, data.historical_series, data.future_dates)
    empirical_pct = compute_empirical_uplift_pct(
        data.historical_series, [e.event_date for e in classified_now.historical]
    )
    if empirical_pct is not None:
        current_events = resolve_uplift_defaults(
            parse_high_production_input(date_text, uploaded_csv_df), round(empirical_pct, 1)
        )
        classified_now = classify_high_production_events(current_events, data.historical_series, data.future_dates)

    first_run = "committed_events" not in st.session_state
    if first_run or update_clicked:
        st.session_state["committed_events"] = current_events
        st.session_state["committed_events_key"] = _events_key(current_events)
    committed_events: List[HighProductionEvent] = st.session_state["committed_events"]
    committed_key = st.session_state["committed_events_key"]

    unapplied = _events_key(current_events) != committed_key
    classified_committed = classify_high_production_events(committed_events, data.historical_series, data.future_dates)

    # ---- Step 2: expensive backtest + final fit, keyed on data + the
    # committed events' DATES (percent-only edits never trigger this). ----
    fit_signature = (data_signature, committed_key, chronos_dl)
    needs_refit = (
        "fit_result" not in st.session_state
        or st.session_state.get("fit_signature") != fit_signature
    )
    if needs_refit:
        progress_box = st.empty()

        def _report(msg: str) -> None:
            progress_box.info(msg)

        try:
            with st.spinner("Backtesting and training models (this can take a minute)..."):
                fit_result = fit_and_backtest_models(
                    data=data,
                    high_production_dates={e.event_date for e in committed_events},
                    chronos_allow_download=chronos_dl,
                    progress_callback=_report,
                )
            progress_box.empty()
            st.session_state["fit_result"] = fit_result
            st.session_state["fit_signature"] = fit_signature
        except Exception as exc:
            progress_box.empty()
            st.error(f"Could not build the forecast: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
            st.stop()
    fit_result: ModelFitResult = st.session_state["fit_result"]

    business = compute_business_adjustment(data, fit_result, classified_committed.future_in_horizon)

    with tab1:
        with col_side:
            if classified_committed.historical:
                st.success(
                    f"{len(classified_committed.historical)} historical high-production date(s) "
                    f"used to train the model" +
                    (f" (empirical uplift: {empirical_pct:+.1f}%)." if empirical_pct is not None else ".")
                )
            if classified_committed.future_in_horizon:
                st.success(f"{len(classified_committed.future_in_horizon)} date(s) applied to this forecast.")
            if classified_committed.out_of_range:
                st.warning(
                    "Outside available history and the forecast horizon (no effect): "
                    + ", ".join(e.event_date.strftime("%Y-%m-%d") for e in classified_committed.out_of_range)
                )
            if unapplied:
                st.info("You have unapplied date changes -- click **Update Forecast** to retrain with them.")

        with col_main:
            hist_tail = data.historical_series.dropna().tail(90)
            leaderboard = fit_result.selection.leaderboard
            ranked_names = [n for n in leaderboard["model"] if n in business]
            skipped_names = [m.model_name for m in fit_result.backtest_metrics if m.skipped]

            st.markdown(
                f"**Historical range:** {data.dq_report.min_date.date()} to "
                f"{data.historical_series.dropna().index.max().date()} &nbsp;&nbsp; "
                f"**As-of date:** {as_of_date.date()} &nbsp;&nbsp; "
                f"**Known backlog days ahead:** {data.dq_report.backlog_days}"
            )

            for name in ranked_names:
                d = business[name]
                is_top = name == fit_result.selection.recommended_model
                fig = chart_model_forecast(
                    name, hist_tail, data.future_dates, d["combined_projection"],
                    d["adjusted_forecast"], d["lower"], d["upper"],
                    [e.event_date for e in classified_committed.future_in_horizon], highlight=is_top,
                )
                st.plotly_chart(fig)

            if skipped_names:
                st.info("Skipped models (see Metrics tab for details): " + ", ".join(skipped_names))

    with tab2:
        st.subheader("Data-Quality Summary")
        dq = data.dq_report
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows loaded", f"{dq.total_rows_loaded:,}")
        c2.metric("Valid rows", f"{dq.total_valid_rows:,}")
        c3.metric("Invalid dates", f"{dq.total_invalid_dates:,}")
        c4.metric("Invalid Net Pounds", f"{dq.total_invalid_pounds:,}")
        st.dataframe(dq.as_dataframe(), width="stretch", hide_index=True)
        for note in dq.notes:
            st.warning(note)
        st.markdown(
            f"- Duplicate rows detected: **{dq.duplicate_row_count}** "
            f"({'excluded' if dq.duplicates_excluded else 'kept'} from aggregation)\n"
            f"- Date range: **{dq.min_date.date()} to {dq.max_date.date()}**\n"
            f"- Missing calendar dates: **{len(dq.missing_calendar_dates)}** "
            f"(treatment selected: *{dq.missing_date_treatment}*)\n"
            f"- Historical days used for modeling: **{dq.historical_days}**  |  "
            f"Known backlog days: **{dq.backlog_days}**"
        )

        st.subheader("Historical Daily Net Pounds")
        st.plotly_chart(chart_historical_line(data.dense_series, data.backlog_series))

        st.subheader("Model Leaderboard")
        board = fit_result.selection.leaderboard.copy()
        display_cols = ["model", "wape", "mae", "rmse", "smape", "mase", "bias", "abs_pct_bias",
                         "accuracy_score", "runtime_seconds", "success_windows", "failed_windows",
                         "skipped", "skip_reason"]
        st.dataframe(board[display_cols].round(2), width="stretch", hide_index=True)
        st.markdown(
            f"**Recommended model: {fit_result.selection.recommended_model}** "
            f"({'beats' if fit_result.selection.beat_seasonal_naive else 'does not beat'} Seasonal Naive)\n\n"
            f"{fit_result.selection.reason}"
        )

        st.subheader("Backtesting -- Actual vs. Predicted")
        factory_map = {
            "Seasonal Naive": lambda: SeasonalNaiveModel(),
            "AutoARIMA": make_auto_arima_model,
            "AutoETS": make_auto_ets_model,
            "HistGradientBoostingRegressor": lambda: HistGradientBoostingModel(),
            "Amazon Chronos-2": lambda: Chronos2Model(allow_download=chronos_dl),
        }
        committed_set = {e.event_date for e in committed_events}
        for name in MODEL_DISPLAY_ORDER:
            metrics = next((m for m in fit_result.backtest_metrics if m.model_name == name), None)
            if metrics is None or metrics.skipped:
                continue
            windows, _ = run_backtest(factory_map[name], name, data.modeling_series, data.horizon, committed_set)
            st.plotly_chart(chart_backtest_actual_vs_predicted(name, windows))

        st.subheader("Combined Forecast Comparison")
        adjusted_by_model = {n: business[n]["adjusted_forecast"] for n in ranked_names}
        st.plotly_chart(
            chart_combined_all_models(hist_tail, data.future_dates, adjusted_by_model,
                                       fit_result.selection.recommended_model),
        )

        st.subheader("Downloadable Results")
        rec_name = fit_result.selection.recommended_model
        rec = business.get(rec_name)

        cleaned_hist_csv = dataframe_to_csv_bytes(data.dense_series.rename("net_pounds").reset_index().rename(columns={"index": "date"}))
        dq_csv = dataframe_to_csv_bytes(dq.as_dataframe())
        leaderboard_csv = dataframe_to_csv_bytes(board[display_cols])

        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button("Cleaned Daily Historical Data", cleaned_hist_csv, "cleaned_daily_history.csv", "text/csv")
        dl2.download_button("Data-Quality Report", dq_csv, "data_quality_report.csv", "text/csv")
        dl3.download_button("Model Metrics Leaderboard", leaderboard_csv, "model_leaderboard.csv", "text/csv")

        per_model_tables = {}
        for name in ranked_names:
            d = business[name]
            per_model_tables[name] = build_final_forecast_table(
                name, data.future_dates, d["combined_projection"], d["adjusted_forecast"],
                d["lower"], d["upper"], data.backlog_series, d["uplift_map"],
            )
        combined_forecast_df = pd.concat(per_model_tables.values(), ignore_index=True) if per_model_tables else pd.DataFrame()

        dl4, dl5, dl6 = st.columns(3)
        if rec is not None:
            rec_table = per_model_tables[rec_name]
            dl4.download_button("Recommended Model Forecast", dataframe_to_csv_bytes(rec_table),
                                 f"forecast_{rec_name.replace(' ', '_')}.csv", "text/csv")
        dl5.download_button("Combined Model Forecasts", dataframe_to_csv_bytes(combined_forecast_df),
                             "combined_forecasts.csv", "text/csv")

        excel_sheets = {
            "Data Quality": dq.as_dataframe(),
            "Historical Daily": data.dense_series.rename("net_pounds").reset_index().rename(columns={"index": "date"}),
            "Leaderboard": board[display_cols],
            "Combined Forecasts": combined_forecast_df,
        }
        for name, tbl in per_model_tables.items():
            excel_sheets[f"FC_{name}"[:31]] = tbl
        workbook_bytes = build_combined_excel_workbook(excel_sheets)
        dl6.download_button("Full Excel Workbook (all sheets)", workbook_bytes,
                             "sales_order_forecast_results.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab3:
        st.caption("Pure visual diagnostics of each model's underlying technique -- no numeric tables here by design.")
        hist = data.historical_series.dropna()
        for name in ranked_names:
            st.markdown(f"#### {name}")
            st.plotly_chart(chart_rolling_windows(hist, name))
            st.plotly_chart(chart_seasonal_profile(hist, name))
            d = business.get(name)
            if d is not None and d["lower"] is not None and d["upper"] is not None:
                st.plotly_chart(
                    chart_quantile_fan(name, data.future_dates, d["combined_projection"],
                                        d["lower"], d["upper"]),
                )


if __name__ == "__main__":
    main()
