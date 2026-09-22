#!/usr/bin/env python3
"""Month-end BTC close vs US M2SL.

Price is cache-only:
    src/cg_data/BTC_data_daily.csv
Seed that with analyst_get_daily_data.py first.

US M2 comes from the public FRED CSV (no API key) and is cached at:
    src/cg_data/US_M2SL.csv
Empty or stale cache -> re-download the full sheet.

Two panels: BTC monthly close on top, M2 (billions) on the bottom.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import requests
from pandas.tseries.offsets import MonthEnd

from analyst_get_daily_data import daily_cache_path, load_daily

# ====================== CONFIG ======================
STALE_AFTER_DAYS = 14
BLOCK_WINDOW = True
SHOW_GRID = False
FIGURE_SIZE = (12, 8)
HEIGHT_RATIOS = (3, 1)

BTC_COLOR = "green"
M2_COLOR = "blue"

M2_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cg_data"
M2_CACHE = DATA_DIR / "US_M2SL.csv"
# ====================================================


def _backend_is_interactive() -> bool:
    backend = matplotlib.get_backend().lower()
    return backend not in {"agg", "svg", "pdf", "ps", "cairo", "template"}


def add_year_quarter_date_formatters(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", which="major", labelsize=9)
    ax.tick_params(axis="x", which="minor", labelsize=7)


def _require_btc_monthly() -> pd.Series:
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
    monthly = raw.resample("ME").agg({"close": "last"})["close"]
    return monthly.dropna()


def fetch_us_m2() -> pd.Series:
    print("Downloading US M2SL from FRED...")
    response = requests.get(M2_URL, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download M2 data (status {response.status_code})")
    m2 = pd.read_csv(
        StringIO(response.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )["M2SL"]
    m2 = pd.to_numeric(m2, errors="coerce").dropna()
    m2.index = m2.index + MonthEnd(0)
    m2 = m2[~m2.index.duplicated(keep="last")].sort_index()
    if m2.empty:
        raise RuntimeError("FRED returned no usable M2SL rows.")
    return m2


def load_or_update_m2() -> pd.Series:
    now = datetime.now(timezone.utc)
    if M2_CACHE.exists():
        age_days = (now.timestamp() - M2_CACHE.stat().st_mtime) / 86400
        if age_days <= STALE_AFTER_DAYS:
            m2 = pd.read_csv(M2_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
            m2 = pd.to_numeric(m2, errors="coerce").dropna().sort_index()
            if not m2.empty:
                print(
                    f"M2 cache current ({age_days:.1f}d old): "
                    f"{m2.index.min().date()} → {m2.index.max().date()} ({len(m2)} months)\n"
                    f"  {M2_CACHE}"
                )
                return m2
            print("M2 cache empty; re-downloading.")

    m2 = fetch_us_m2()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    m2.rename("M2SL").to_csv(M2_CACHE, header=True)
    print(f"Saved {len(m2)} M2 months → {M2_CACHE}")
    print(f"M2 range: {m2.index.min().date()} → {m2.index.max().date()}")
    return m2


def draw(block_window: bool = BLOCK_WINDOW) -> None:
    monthly_btc = _require_btc_monthly()
    us_m2 = load_or_update_m2()

    df = pd.DataFrame({"BTC": monthly_btc, "US_M2": us_m2}).dropna()
    if df.empty:
        raise ValueError("No overlapping BTC / M2 months. Seed BTC daily first.")

    print(
        f"Overlap: {df.index.min().date()} → {df.index.max().date()} ({len(df)} months)"
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=FIGURE_SIZE,
        gridspec_kw={"height_ratios": list(HEIGHT_RATIOS)},
        sharex=True,
    )
    plt.style.use("fast")

    ax1.plot(df.index, df["BTC"], label="BTC Monthly Close", color=BTC_COLOR, linewidth=0.9)
    ax1.set_title("BTC Price vs US M2 (FRED M2SL)")
    ax1.set_ylabel("Price (USD)")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
    if SHOW_GRID:
        ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df["US_M2"], label="US M2 (Billions USD)", color=M2_COLOR, linewidth=1.0)
    ax2.set_ylabel("M2 (Billions USD)")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: f"{x:,.0f}"))
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.92)
    if SHOW_GRID:
        ax2.grid(True, alpha=0.3)

    add_year_quarter_date_formatters(ax1)
    add_year_quarter_date_formatters(ax2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()

    if not _backend_is_interactive():
        out = "usd_m2_vs_btc.png"
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
