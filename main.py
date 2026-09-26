"""
Macro FX Forecasting Engine
============================

Pipeline: fetch macro data (FRED) + market data (yfinance) -> compute
real yield spreads -> rolling Z-score normalization -> serialize signal.

Run manually:
    pip install -r requirements.txt
    python main.py

Run in CI:
    See the accompanying GitHub Actions workflow, which runs this on a
    daily schedule and commits the resulting latest_signal.json back
    into the repo.

Output:
    latest_signal.json — timestamped snapshot of raw inputs, the
    computed real yield spread, its rolling Z-score, VIX level, and a
    structural signal classification (Bullish / Bearish / Neutral).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

try:
    import pandas_datareader.data as web
except ImportError:  # pragma: no cover
    web = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LOG = logging.getLogger("macro_fx_engine")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

OUTPUT_FILE = "latest_signal.json"

# FRED series identifiers
FRED_US_NOMINAL_YIELD = "GS10"                # US 10-Year Treasury Yield (monthly)
FRED_US_CPI = "CPILFESL"                      # US Core CPI (monthly index)
FRED_AU_NOMINAL_YIELD = "INTGSTAU193N"        # AU 10-Year Government Bond Yield (monthly)
FRED_AU_CPI = "CPALTT01AUM657N"               # AU Consumer Price Index (monthly index)

# yfinance ticker
YFINANCE_VIX = "^VIX"

# Rolling window for Z-score normalization, expressed in trading/monthly
# observations. FRED series here are monthly, so 3 years = 36 observations.
ROLLING_WINDOW_PERIODS = 36
MIN_PERIODS_FOR_ROLLING = 12  # allow a signal once at least 1 year of history exists

# Z-score thresholds for classification
BULLISH_THRESHOLD = 1.0
BEARISH_THRESHOLD = -1.0

# History depth to request from FRED (extra years give safe padding
# beyond the 3-year rolling window plus 12 months lost to YoY diffing).
FRED_START_DATE = "2000-01-01"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class MacroSeriesBundle:
    """Raw macro series pulled from FRED, aligned to a common daily index."""
    us_nominal_yield: pd.Series
    us_cpi_index: pd.Series
    au_nominal_yield: pd.Series
    au_cpi_index: pd.Series


@dataclass
class SignalOutput:
    generated_at_utc: str
    us_10y_nominal_yield: Optional[float]
    us_core_cpi_yoy_pct: Optional[float]
    us_real_yield: Optional[float]
    au_10y_nominal_yield: Optional[float]
    au_cpi_yoy_pct: Optional[float]
    au_real_yield: Optional[float]
    real_yield_spread_au_minus_us: Optional[float]
    spread_rolling_mean: Optional[float]
    spread_rolling_std: Optional[float]
    spread_z_score: Optional[float]
    vix_close: Optional[float]
    signal: str
    warnings: list[str]


# --------------------------------------------------------------------------- #
# 1. Data Ingestion Engine
# --------------------------------------------------------------------------- #

def fetch_fred_series(series_id: str, start: str = FRED_START_DATE) -> Optional[pd.Series]:
    """Fetch a single FRED series via pandas_datareader. Returns None on failure."""
    if web is None:
        LOG.error("pandas_datareader is not installed; cannot fetch FRED series %s", series_id)
        return None

    try:
        LOG.info("Fetching FRED series: %s", series_id)
        df = web.DataReader(series_id, "fred", start)
        series = df[series_id].dropna()
        if series.empty:
            LOG.warning("FRED series %s returned no data", series_id)
            return None
        LOG.info("Fetched %d observations for %s (latest: %s)", len(series), series_id, series.index[-1].date())
        return series
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a resilience boundary
        LOG.error("Failed to fetch FRED series %s: %s", series_id, exc)
        return None


def fetch_yfinance_close(ticker: str, period: str = "5d") -> Optional[float]:
    """Fetch the most recent close for a yfinance ticker. Returns None on failure."""
    if yf is None:
        LOG.error("yfinance is not installed; cannot fetch %s", ticker)
        return None

    try:
        LOG.info("Fetching yfinance ticker: %s", ticker)
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            LOG.warning("yfinance returned no data for %s", ticker)
            return None
        latest_close = float(data["Close"].dropna().iloc[-1])
        LOG.info("Latest close for %s: %.4f", ticker, latest_close)
        return latest_close
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to fetch yfinance ticker %s: %s", ticker, exc)
        return None


def ingest_macro_data() -> tuple[Optional[MacroSeriesBundle], list[str]]:
    """Pull all required macro series. Returns (bundle_or_None, warnings)."""
    warnings: list[str] = []

    us_nominal = fetch_fred_series(FRED_US_NOMINAL_YIELD)
    us_cpi = fetch_fred_series(FRED_US_CPI)
    au_nominal = fetch_fred_series(FRED_AU_NOMINAL_YIELD)
    au_cpi = fetch_fred_series(FRED_AU_CPI)

    missing = [
        name for name, series in [
            ("US nominal yield", us_nominal),
            ("US CPI", us_cpi),
            ("AU nominal yield", au_nominal),
            ("AU CPI", au_cpi),
        ] if series is None
    ]

    if missing:
        msg = f"Missing required macro series: {', '.join(missing)}"
        LOG.error(msg)
        warnings.append(msg)
        return None, warnings

    bundle = MacroSeriesBundle(
        us_nominal_yield=us_nominal,
        us_cpi_index=us_cpi,
        au_nominal_yield=au_nominal,
        au_cpi_index=au_cpi,
    )
    return bundle, warnings


# --------------------------------------------------------------------------- #
# 2. Quant Data Matrices
# --------------------------------------------------------------------------- #

def build_aligned_frame(bundle: MacroSeriesBundle) -> pd.DataFrame:
    """
    Re-index disparate series onto a common monthly index and forward-fill
    gaps (e.g. CPI released on a lag relative to yield data).
    """
    frame = pd.DataFrame({
        "us_nominal_yield": bundle.us_nominal_yield,
        "us_cpi_index": bundle.us_cpi_index,
        "au_nominal_yield": bundle.au_nominal_yield,
        "au_cpi_index": bundle.au_cpi_index,
    })

    # Resample to month-end to normalize any mixed reporting frequencies,
    # then forward-fill so every column has a value at every timestamp.
    frame = frame.resample("ME").last()
    frame = frame.ffill()

    LOG.info("Aligned macro frame built: %d rows, spanning %s to %s",
              len(frame), frame.index[0].date() if not frame.empty else "n/a",
              frame.index[-1].date() if not frame.empty else "n/a")
    return frame


# --------------------------------------------------------------------------- #
# 3. Macro Engine Calculus
# --------------------------------------------------------------------------- #

def compute_real_yields(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Derive YoY inflation from each CPI index, then compute real yield
    (nominal yield - YoY inflation) for both countries plus the spread.
    """
    frame = frame.copy()

    frame["us_cpi_yoy_pct"] = frame["us_cpi_index"].pct_change(periods=12) * 100
    frame["au_cpi_yoy_pct"] = frame["au_cpi_index"].pct_change(periods=12) * 100

    frame["us_real_yield"] = frame["us_nominal_yield"] - frame["us_cpi_yoy_pct"]
    frame["au_real_yield"] = frame["au_nominal_yield"] - frame["au_cpi_yoy_pct"]

    # Target asset pair spread: AU real yield minus US real yield.
    # Positive = AU real yields relatively richer -> AUD-supportive tilt.
    frame["real_yield_spread"] = frame["au_real_yield"] - frame["us_real_yield"]

    return frame


# --------------------------------------------------------------------------- #
# 4. Statistical Signaling
# --------------------------------------------------------------------------- #

def compute_rolling_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize the real yield spread to a rolling Z-score."""
    frame = frame.copy()

    rolling = frame["real_yield_spread"].rolling(
        window=ROLLING_WINDOW_PERIODS,
        min_periods=MIN_PERIODS_FOR_ROLLING,
    )
    frame["spread_rolling_mean"] = rolling.mean()
    frame["spread_rolling_std"] = rolling.std()
    frame["spread_z_score"] = (
        (frame["real_yield_spread"] - frame["spread_rolling_mean"])
        / frame["spread_rolling_std"]
    )

    return frame


def classify_signal(z_score: Optional[float]) -> str:
    """Map a Z-score to a structural signal classification."""
    if z_score is None or pd.isna(z_score):
        return "Neutral"
    if z_score >= BULLISH_THRESHOLD:
        return "Bullish"
    if z_score <= BEARISH_THRESHOLD:
        return "Bearish"
    return "Neutral"


# --------------------------------------------------------------------------- #
# 5. Data Output Layer
# --------------------------------------------------------------------------- #

def safe_float(value) -> Optional[float]:
    """Convert to a plain Python float for JSON serialization, or None if NaN/missing."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def build_signal_output(frame: pd.DataFrame, vix_close: Optional[float], warnings: list[str]) -> SignalOutput:
    if frame.empty:
        warnings.append("Aligned macro frame is empty; no signal computed.")
        return SignalOutput(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            us_10y_nominal_yield=None,
            us_core_cpi_yoy_pct=None,
            us_real_yield=None,
            au_10y_nominal_yield=None,
            au_cpi_yoy_pct=None,
            au_real_yield=None,
            real_yield_spread_au_minus_us=None,
            spread_rolling_mean=None,
            spread_rolling_std=None,
            spread_z_score=None,
            vix_close=safe_float(vix_close),
            signal="Neutral",
            warnings=warnings,
        )

    latest = frame.iloc[-1]
    z_score = safe_float(latest.get("spread_z_score"))

    return SignalOutput(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        us_10y_nominal_yield=safe_float(latest.get("us_nominal_yield")),
        us_core_cpi_yoy_pct=safe_float(latest.get("us_cpi_yoy_pct")),
        us_real_yield=safe_float(latest.get("us_real_yield")),
        au_10y_nominal_yield=safe_float(latest.get("au_nominal_yield")),
        au_cpi_yoy_pct=safe_float(latest.get("au_cpi_yoy_pct")),
        au_real_yield=safe_float(latest.get("au_real_yield")),
        real_yield_spread_au_minus_us=safe_float(latest.get("real_yield_spread")),
        spread_rolling_mean=safe_float(latest.get("spread_rolling_mean")),
        spread_rolling_std=safe_float(latest.get("spread_rolling_std")),
        spread_z_score=z_score,
        vix_close=safe_float(vix_close),
        signal=classify_signal(z_score),
        warnings=warnings,
    )


def write_signal_output(signal_output: SignalOutput, path: str = OUTPUT_FILE) -> None:
    """Serialize the signal output to a JSON file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(signal_output), f, indent=2)
        LOG.info("Wrote signal output to %s", path)
    except OSError as exc:
        LOG.error("Failed to write output file %s: %s", path, exc)
        raise


# --------------------------------------------------------------------------- #
# Pipeline entrypoint
# --------------------------------------------------------------------------- #

def run_pipeline() -> SignalOutput:
    LOG.info("=== Macro FX Forecasting Engine: pipeline start ===")

    # 1. Ingestion
    bundle, warnings = ingest_macro_data()
    vix_close = fetch_yfinance_close(YFINANCE_VIX)
    if vix_close is None:
        warnings.append("VIX data unavailable from yfinance.")

    if bundle is None:
        LOG.error("Aborting: required macro data could not be fetched.")
        signal_output = build_signal_output(pd.DataFrame(), vix_close, warnings)
        write_signal_output(signal_output)
        return signal_output

    # 2. Alignment
    frame = build_aligned_frame(bundle)

    # 3. Real yield calculus
    frame = compute_real_yields(frame)

    # 4. Statistical signaling
    frame = compute_rolling_zscore(frame)

    # 5. Output
    signal_output = build_signal_output(frame, vix_close, warnings)
    write_signal_output(signal_output)

    LOG.info("=== Pipeline complete. Signal: %s (Z=%s) ===",
              signal_output.signal, signal_output.spread_z_score)
    return signal_output


if __name__ == "__main__":
    run_pipeline()
