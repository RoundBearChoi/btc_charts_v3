#!/usr/bin/env python3
"""KOSPI moving-average chart.

Fetches daily KOSPI (KS11) via FinanceDataReader, computes SMAs on a
slightly longer history so the longest window is seeded, then draws
the last DAYS_BACK trading days.

Config lives at the top. No coin menu — this is KOSPI-only.
"""

from __future__ import annotations

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
from datetime import datetime, timedelta

import FinanceDataReader as fdr

# ====================== CONFIG ======================
DAYS_BACK = 360 * 2
SMA_WINDOWS = [20, 50, 111, 200]
BLOCK_WINDOW = True
SHOW_GRID = True
FIGURE_SIZE = (14, 8)
CLOSE_COLOR = "#9EB3DB"
CLOSE_WIDTH = 1.1
SMA_COLORS = ["#E15FC3", "#00D118", "#FF9900", "#C80C01"]
# ====================================================


def _backend_is_interactive() -> bool:
    backend = matplotlib.get_backend().lower()
    return backend not in {"agg", "svg", "pdf", "ps", "cairo", "template"}


def format_price(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    abs_v = abs(value)
    if abs_v >= 100:
        return f"{value:,.0f}"
    if abs_v >= 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def format_month_year(x, _p=None) -> str:
    """Tick label like 'Sep 2026'."""
    return mdates.num2date(x).strftime("%b %Y")


def add_month_date_formatters(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_month_year))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def add_sma(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.DataFrame:
    col = f"SMA{window}"
    df[col] = df[price_col].rolling(window=window, min_periods=window).mean()
    return df


def fetch_kospi(days_back: int, max_sma: int) -> pd.DataFrame:
    """Fetch enough history so the longest SMA is fully seeded."""
    # Extra calendar days to cover weekends/holidays + SMA warmup
    extra = max_sma * 2
    start = (datetime.now() - timedelta(days=days_back + extra)).strftime("%Y-%m-%d")
    raw = fdr.DataReader("KS11", start)
    if raw.empty:
        raise ValueError("FinanceDataReader returned no KOSPI data.")
    # Normalize column names
    raw = raw.rename(columns=str.lower)
    if "close" not in raw.columns:
        raise ValueError(f"Expected 'Close' column, got: {list(raw.columns)}")
    raw = raw.sort_index()
    raw = raw[raw["close"].notna()]
    print(
        f"Fetched KOSPI: {raw.index.min().date()} → {raw.index.max().date()} "
        f"({len(raw)} rows)"
    )
    return raw


def print_snapshot(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    price = float(last["close"])
    print("=" * 56)
    print(f"KOSPI snapshot  {df.index[-1].date()}")
    print("=" * 56)
    print(f"Close:           {format_price(price)}")
    print(f"Range:           {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} days)")
    print()
    print(f"{'MA':<10}{'Value':>12}{'vs close':>12}")
    print("-" * 34)
    for w in SMA_WINDOWS:
        col = f"SMA{w}"
        if col not in df.columns or pd.isna(last[col]):
            continue
        ma = float(last[col])
        pct = (price - ma) / ma * 100 if ma else float("nan")
        print(f"SMA{w:<7}{format_price(ma):>12}{pct:+10.1f}%")
    print("=" * 56)
    print()


def draw(block_window: bool = BLOCK_WINDOW) -> None:
    max_sma = max(SMA_WINDOWS) if SMA_WINDOWS else 0
    df = fetch_kospi(DAYS_BACK, max_sma).copy()

    for w in SMA_WINDOWS:
        df = add_sma(df, w)

    if DAYS_BACK is not None:
        df = df.iloc[-DAYS_BACK:]
    if df.empty:
        raise ValueError("No rows left after slicing to DAYS_BACK.")

    print_snapshot(df)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    plt.style.use("fast")

    ax.plot(
        df.index,
        df["close"],
        label="KOSPI Close",
        color=CLOSE_COLOR,
        linewidth=CLOSE_WIDTH,
        zorder=3,
    )

    for i, w in enumerate(SMA_WINDOWS):
        col = f"SMA{w}"
        color = SMA_COLORS[i % len(SMA_COLORS)]
        ax.plot(
            df.index,
            df[col],
            label=f"{w}-Day SMA",
            color=color,
            linewidth=1.2,
            zorder=4,
        )

    title = "KOSPI • Moving Averages"
    if DAYS_BACK:
        title += f" — Last {DAYS_BACK} days"
    ax.set_title(title, fontsize=14, pad=16)
    ax.set_ylabel("Index")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format_price(x)))
    if SHOW_GRID:
        ax.grid(True, alpha=0.3)

    add_month_date_formatters(ax)
    fig.tight_layout()

    if not _backend_is_interactive():
        out = "kospi_ma.png"
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
