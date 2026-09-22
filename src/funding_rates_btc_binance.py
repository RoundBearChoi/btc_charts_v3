#!/usr/bin/env python3
"""BTCUSDT funding rates + cached BTC daily price.

Price is cache-only:
    src/cg_data/BTC_data_daily.csv
Seed that with analyst_get_daily_data.py first.

Funding is fetched from Binance public futures (no API key) and cached at:
    src/cg_data/BTC_funding_binance.csv
Empty or stale cache -> increment from the last row (or seed LOOKBACK_YEARS).

Three panels: BTC close + SMA50/SMA111, daily avg funding, funding z-score.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import requests

from analyst_get_daily_data import daily_cache_path, load_daily

# ====================== CONFIG ======================
SYMBOL = "BTCUSDT"
LOOKBACK_YEARS = 2
STALE_AFTER_DAYS = 2
REQUEST_PAUSE_SEC = 0.15

SMA_FAST = 50
SMA_SLOW = 111
ZSCORE_WINDOW = 180

SHOW_ABSOLUTE_REFS = True
ABS_MODERATE = 0.025
ABS_HIGH = 0.04

BLOCK_WINDOW = True
SHOW_GRID = True
FIGURE_SIZE = (14, 11)
HEIGHT_RATIOS = (2.5, 1.8, 1.2)

PRICE_COLOR = "#1f77b4"
SMA50_COLOR = "#2ca02c"
SMA111_COLOR = "#ff7f0e"
FUNDING_COLOR = "#1f77b4"
ZSCORE_COLOR = "#d62728"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cg_data"
CACHE_FILE = DATA_DIR / "BTC_funding_binance.csv"
BINANCE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
# ====================================================


def _backend_is_interactive() -> bool:
    backend = matplotlib.get_backend().lower()
    return backend not in {"agg", "svg", "pdf", "ps", "cairo", "template"}


def format_month_year(x, _p=None) -> str:
    """Tick label like 'Sep 2026'."""
    return mdates.num2date(x).strftime("%b %Y")


def add_quarter_date_formatters(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_month_year))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _naive_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def lookback_start(years: int = LOOKBACK_YEARS) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=years * 365 + 30)


def fetch_binance_funding(
    symbol: str = SYMBOL,
    start: datetime | None = None,
    years: int = LOOKBACK_YEARS,
) -> pd.DataFrame:
    start = _as_utc(start or lookback_start(years))
    current_start = int(start.timestamp() * 1000)
    rows: list[dict] = []
    print(f"Fetching {symbol} funding from {start.strftime('%Y-%m-%d %H:%M')} UTC...")
    while True:
        params = {"symbol": symbol, "startTime": current_start, "limit": 1000}
        try:
            resp = requests.get(BINANCE_URL, params=params, timeout=20)
            if resp.status_code == 429:
                print("Binance 429 — sleeping 2s")
                time.sleep(2)
                continue
            resp.raise_for_status()
            batch = resp.json()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Binance funding request failed: {exc}") from exc
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1]["fundingTime"])
        current_start = last_ts + 1
        if len(batch) < 1000:
            break
        if REQUEST_PAUSE_SEC > 0:
            time.sleep(REQUEST_PAUSE_SEC)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame = frame[["timestamp", "funding_rate"]].dropna()
    frame = frame.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return frame.reset_index(drop=True)


def load_funding_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.read_csv(CACHE_FILE, parse_dates=["timestamp"])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df = df.dropna(subset=["timestamp", "funding_rate"])
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def save_funding_cache(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    out[["timestamp", "funding_rate"]].to_csv(CACHE_FILE, index=False)
    return CACHE_FILE


def load_or_update_funding(years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    cutoff = lookback_start(years)
    cached = load_funding_cache()
    if not cached.empty:
        cached = cached[cached["timestamp"] >= cutoff].reset_index(drop=True)

    now = datetime.now(timezone.utc)
    if cached.empty:
        fresh = fetch_binance_funding(SYMBOL, start=cutoff, years=years)
        if fresh.empty:
            raise RuntimeError("Binance returned no funding rates.")
        path = save_funding_cache(fresh)
        print(f"Saved {len(fresh)} funding rows → {path}")
        return fresh

    latest = _as_utc(cached["timestamp"].max().to_pydatetime())
    gap_days = (now - latest).total_seconds() / 86400
    if gap_days <= STALE_AFTER_DAYS:
        print(
            f"Funding cache current through {latest.strftime('%Y-%m-%d %H:%M')} UTC "
            f"({len(cached)} rows)."
        )
        return cached

    print(f"Funding cache is {gap_days:.1f} days behind; fetching tail...")
    fresh = fetch_binance_funding(SYMBOL, start=latest, years=years)
    if fresh.empty:
        print("No new funding rows; using existing cache.")
        return cached
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    combined = combined[combined["timestamp"] >= cutoff].reset_index(drop=True)
    path = save_funding_cache(combined)
    print(f"Saved {len(combined)} funding rows → {path}")
    return combined


def add_funding_zscore(daily_series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    min_periods = max(30, window // 3)
    mean = daily_series.rolling(window=window, min_periods=min_periods).mean()
    std = daily_series.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    return (daily_series - mean) / std


def _require_btc_daily() -> pd.DataFrame:
    path = daily_cache_path("BTC")
    raw = load_daily("BTC")
    if raw.empty or "close" not in raw.columns:
        raise FileNotFoundError(
            f"No BTC daily cache at {path}.\n"
            "Run analyst_get_daily_data.py first."
        )
    raw = raw.sort_index()
    raw = raw[raw["close"].notna()]
    if raw.empty:
        raise ValueError(f"{path} has no usable close prices.")
    print(
        f"Using cached BTC daily: {raw.index.min().date()} → "
        f"{raw.index.max().date()} ({len(raw)} rows)\n  {path}"
    )
    return raw


def print_stats(df: pd.DataFrame, years: int, zscore_window: int) -> None:
    if df.empty:
        return
    rate_pct = df["funding_rate"] * 100
    print("=" * 55)
    print(f"BTCUSDT Funding Rates + Price Context (Last {years} years)")
    print("=" * 55)
    print(f"Period:           {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"# Funding prints: {len(df):,}")
    print(f"Mean / Median:    {rate_pct.mean():.5f}% / {rate_pct.median():.5f}%")
    print(f"Std / Max / Min:  {rate_pct.std():.5f}% / {rate_pct.max():.5f}% / {rate_pct.min():.5f}%")
    print(f"% Positive:       {(rate_pct > 0).mean() * 100:.1f}%")
    print(f"Z-Score window:   {zscore_window} days")
    print("=" * 55)
    print()


def draw(
    lookback_years: int = LOOKBACK_YEARS,
    zscore_window: int = ZSCORE_WINDOW,
    block_window: bool = BLOCK_WINDOW,
) -> None:
    funding_df = load_or_update_funding(lookback_years)
    if funding_df.empty:
        raise RuntimeError("No funding rate data available.")
    print_stats(funding_df, lookback_years, zscore_window)

    funding = funding_df.set_index("timestamp").sort_index()
    funding.index = pd.to_datetime(funding.index, utc=True)
    daily_funding = funding["funding_rate"].resample("D").mean() * 100
    daily_funding.index = _naive_utc_index(daily_funding.index)
    daily_z = add_funding_zscore(daily_funding, window=zscore_window)

    price = _require_btc_daily()
    price = price.copy()
    price.index = _naive_utc_index(price.index)
    price[f"SMA{SMA_FAST}"] = price["close"].rolling(window=SMA_FAST).mean()
    price[f"SMA{SMA_SLOW}"] = price["close"].rolling(window=SMA_SLOW).mean()
    start = daily_funding.index.min() - pd.Timedelta(days=5)
    end = daily_funding.index.max() + pd.Timedelta(days=5)
    price = price.loc[start:end]
    if price.empty:
        raise ValueError(
            "BTC daily cache does not overlap the funding window. "
            "Run analyst_get_daily_data.py first."
        )

    fig = plt.figure(figsize=FIGURE_SIZE)
    plt.style.use("fast")
    gs = fig.add_gridspec(3, 1, height_ratios=list(HEIGHT_RATIOS), hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    ax1.plot(price.index, price["close"], color=PRICE_COLOR, linewidth=1.3, label="BTC Close")
    ax1.plot(
        price.index, price[f"SMA{SMA_FAST}"],
        color=SMA50_COLOR, linewidth=1.15, label=f"{SMA_FAST}-Day SMA",
    )
    ax1.plot(
        price.index, price[f"SMA{SMA_SLOW}"],
        color=SMA111_COLOR, linewidth=1.15, label=f"{SMA_SLOW}-Day SMA",
    )
    ax1.set_ylabel("BTC Price (USD)")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: f"${x:,.0f}"))
    ax1.set_title(
        f"BTC Price + Funding Rates + Z-Score ({zscore_window}d) | Last {lookback_years} Years",
        fontsize=14,
        pad=12,
    )
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
    if SHOW_GRID:
        ax1.grid(True, alpha=0.3)

    ax2.plot(daily_funding.index, daily_funding, color=FUNDING_COLOR, linewidth=1.4, label="Daily Avg Funding")
    ax2.fill_between(
        daily_funding.index, daily_funding, 0,
        where=(daily_funding >= 0), color="#2ca02c", alpha=0.30, interpolate=True,
    )
    ax2.fill_between(
        daily_funding.index, daily_funding, 0,
        where=(daily_funding < 0), color="#d62728", alpha=0.30, interpolate=True,
    )
    ax2.axhline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    if SHOW_ABSOLUTE_REFS:
        ax2.axhline(ABS_MODERATE, color="#ff7f0e", linestyle=":", linewidth=1.0, alpha=0.6, label=f"+{ABS_MODERATE:.3f}%")
        ax2.axhline(-ABS_MODERATE, color="#ff7f0e", linestyle=":", linewidth=1.0, alpha=0.6)
        ax2.axhline(ABS_HIGH, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.5, label=f"+{ABS_HIGH:.3f}%")
        ax2.axhline(-ABS_HIGH, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.5)
    ax2.set_ylabel("Funding Rate (%)")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: f"{x:.3f}%"))
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.92)
    if SHOW_GRID:
        ax2.grid(True, alpha=0.3)

    ax3.plot(daily_funding.index, daily_z, color=ZSCORE_COLOR, linewidth=1.5, label=f"Funding Z-Score ({zscore_window}d)")
    ax3.axhline(0, color="black", linewidth=0.9, alpha=0.8)
    ax3.axhline(2, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.85, label="+2σ")
    ax3.axhline(-2, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.85, label="-2σ")
    ax3.axhline(3, color="#8B0000", linestyle=":", linewidth=1.1, alpha=0.7, label="+3σ")
    ax3.axhline(-3, color="#006400", linestyle=":", linewidth=1.1, alpha=0.7, label="-3σ")
    ax3.set_ylabel("Z-Score")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.92)
    if SHOW_GRID:
        ax3.grid(True, alpha=0.3)

    add_quarter_date_formatters(ax1)
    add_quarter_date_formatters(ax2)
    add_quarter_date_formatters(ax3)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="x", labelbottom=False)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.08)

    if not _backend_is_interactive():
        out = "btc_funding_rates.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to: {out}")
        plt.close(fig)
        return

    print(f"Using interactive backend: {matplotlib.get_backend()}")
    plt.show(block=block_window)
    if block_window:
        plt.close(fig)


if __name__ == "__main__":
    draw()
