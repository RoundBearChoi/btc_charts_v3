#!/usr/bin/env python3
"""KOSPI moving-average chart with RSI.

Fetches daily KOSPI (KS11) via FinanceDataReader, computes SMAs and RSI
on a slightly longer history so the longest window is seeded, then draws
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
FIGURE_SIZE = (14, 9)
HEIGHT_RATIOS = (3, 1)

RSI_WINDOW = 14
RSI_MA_DAYS = 9
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

CLOSE_COLOR = "#9EB3DB"
CLOSE_WIDTH = 1.1
SMA_COLORS = ["#E15FC3", "#00D118", "#FF9900", "#C80C01"]
RSI_COLOR = "#FF9900"
RSI_MA_COLOR = "#9467bd"
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


def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "close", out_col: str = "RSI") -> pd.DataFrame:
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    for i in range(window, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) + loss.iloc[i]) / window
    avg_loss_safe = avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + avg_gain / avg_loss_safe))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    df[out_col] = rsi
    return df


def add_rsi_ema(rsi: pd.Series, span: int = RSI_MA_DAYS) -> pd.Series:
    """Recursive EMA of RSI. Does not recompute RSI on smoothed price."""
    return rsi.ewm(
        span=span,
        min_periods=max(5, span // 2),
        adjust=False,
    ).mean()


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
    rsi = float(last["RSI"]) if "RSI" in df.columns and pd.notna(last["RSI"]) else float("nan")
    rsi_ema = float(last["RSI_EMA"]) if "RSI_EMA" in df.columns and pd.notna(last["RSI_EMA"]) else float("nan")
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
    print()
    rsi_txt = f"{rsi:.1f}" if pd.notna(rsi) else "n/a"
    rsi_ema_txt = f"{rsi_ema:.1f}" if pd.notna(rsi_ema) else "n/a"
    print(f"RSI({RSI_WINDOW}):       {rsi_txt}")
    print(f"RSI {RSI_MA_DAYS}d EMA:     {rsi_ema_txt}")
    print("=" * 56)
    print()


def draw(block_window: bool = BLOCK_WINDOW) -> None:
    max_sma = max(SMA_WINDOWS) if SMA_WINDOWS else 0
    df = fetch_kospi(DAYS_BACK, max_sma).copy()

    for w in SMA_WINDOWS:
        df = add_sma(df, w)
    df = add_rsi(df, window=RSI_WINDOW)
    df["RSI_EMA"] = add_rsi_ema(df["RSI"], span=RSI_MA_DAYS)

    if DAYS_BACK is not None:
        df = df.iloc[-DAYS_BACK:]
    if df.empty:
        raise ValueError("No rows left after slicing to DAYS_BACK.")

    print_snapshot(df)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=FIGURE_SIZE,
        gridspec_kw={"height_ratios": list(HEIGHT_RATIOS)},
        sharex=True,
    )
    plt.style.use("fast")

    ax1.plot(
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
        ax1.plot(
            df.index,
            df[col],
            label=f"{w}-Day SMA",
            color=color,
            linewidth=1.2,
            zorder=4,
        )

    title = f"KOSPI • Moving Averages + RSI({RSI_WINDOW})"
    if DAYS_BACK:
        title += f" — Last {DAYS_BACK} days"
    ax1.set_title(title, fontsize=14, pad=16)
    ax1.set_ylabel("Index")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format_price(x)))
    if SHOW_GRID:
        ax1.grid(True, alpha=0.3)

    ax2.plot(
        df.index, df["RSI"],
        color=RSI_COLOR, linewidth=1.2, alpha=0.8,
        label=f"RSI({RSI_WINDOW})",
    )
    ax2.plot(
        df.index, df["RSI_EMA"],
        color=RSI_MA_COLOR, linewidth=1.6,
        label=f"RSI {RSI_MA_DAYS}d EMA",
    )
    ax2.axhline(RSI_OVERBOUGHT, color="#E15FC3", linestyle="--", alpha=0.6, label=f"Overbought ({RSI_OVERBOUGHT})")
    ax2.axhline(RSI_OVERSOLD, color="#00D118", linestyle="--", alpha=0.6, label=f"Oversold ({RSI_OVERSOLD})")
    ax2.axhline(50, color="gray", linestyle=":", alpha=0.5, label="Midline (50)")
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.92)
    if SHOW_GRID:
        ax2.grid(True, alpha=0.3)

    add_month_date_formatters(ax1)
    add_month_date_formatters(ax2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(axis="x", labelbottom=False)
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
