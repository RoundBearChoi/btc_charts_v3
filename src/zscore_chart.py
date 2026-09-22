#!/usr/bin/env python3
"""Price + SMA200 + rolling z-score from the existing daily cache.

Does not download. Reads:
    src/cg_data/{SYMBOL}_data_daily.csv
Seed that with analyst_get_daily_data.py first.

Z-score and SMA200 are computed on the full cache, then DAYS_BACK
rows are drawn (None = entire file). A 365-day z-score needs Analyst
history; Demo daily will barely fill the window.

Startup prompt is 1..N from coins.csv, plus ALL.
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

from analyst_get_daily_data import daily_cache_path, load_daily
from coin_menu import get_coin_choice

# ====================== CONFIG ======================
DAYS_BACK = None
BLOCK_WINDOW = True
SHOW_GRID = True
LOG_SCALE = False
FIGURE_SIZE = (14, 8)
HEIGHT_RATIOS = (3, 1)

ZSCORE_WINDOW = 365
SMA_SLOW = 200

CLOSE_COLOR = "#1f77b4"
CLOSE_WIDTH = 1.2
SMA200_COLOR = "#ff7f0e"
SMA200_WIDTH = 1.5
ZSCORE_COLOR = "#d62728"

PRICE_GRID_COLOR = "#b0b0b0"
PRICE_GRID_LINEWIDTH = 1.0
PRICE_GRID_ALPHA = 0.6
ZSCORE_GRID_COLOR = "#b0b0b0"
ZSCORE_GRID_LINEWIDTH = 1.0
ZSCORE_GRID_ALPHA = 0.6
# ====================================================


def _backend_is_interactive() -> bool:
    backend = matplotlib.get_backend().lower()
    return backend not in {"agg", "svg", "pdf", "ps", "cairo", "template"}


def format_price(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    abs_v = abs(value)
    if abs_v >= 100:
        return f"${value:,.0f}"
    if abs_v >= 1:
        return f"${value:,.2f}"
    if abs_v >= 0.01:
        return f"${value:,.4f}"
    return f"${value:.6f}"


def add_year_quarter_date_formatters(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", which="major", labelsize=9)
    ax.tick_params(axis="x", which="minor", labelsize=7)


def add_zscore(
    df: pd.DataFrame,
    window: int = ZSCORE_WINDOW,
    price_col: str = "close",
    out_col: str | None = None,
) -> pd.DataFrame:
    if out_col is None:
        out_col = f"ZScore_{window}d"
    min_periods = max(30, window // 2)
    mean = df[price_col].rolling(window=window, min_periods=min_periods).mean()
    std = df[price_col].rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    df[out_col] = (df[price_col] - mean) / std
    return df


def _require_daily(symbol: str) -> pd.DataFrame:
    path = daily_cache_path(symbol)
    raw = load_daily(symbol)
    if raw.empty or "close" not in raw.columns:
        raise FileNotFoundError(
            f"No {symbol} daily cache at {path}.\n"
            "Run analyst_get_daily_data.py first."
        )
    raw = raw.sort_index()
    raw = raw[raw["close"].notna()]
    if raw.empty:
        raise ValueError(f"{path} has no usable close prices.")
    print(
        f"Using cached {symbol} daily: {raw.index.min().date()} → "
        f"{raw.index.max().date()} ({len(raw)} rows)\n  {path}"
    )
    return raw


def draw_one_chart(
    coin_name: str,
    coin_ticker: str,
    *,
    block_window: bool = BLOCK_WINDOW,
    close_after: bool = True,
) -> None:
    df = _require_daily(coin_ticker).copy()
    z_col = f"ZScore_{ZSCORE_WINDOW}d"
    df = add_zscore(df, window=ZSCORE_WINDOW, out_col=z_col)
    df[f"SMA{SMA_SLOW}"] = df["close"].rolling(window=SMA_SLOW).mean()

    if DAYS_BACK is not None:
        df = df.iloc[-DAYS_BACK:]
    if df.empty:
        raise ValueError(f"{coin_ticker} has no rows in the visible window.")

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=FIGURE_SIZE,
        gridspec_kw={"height_ratios": list(HEIGHT_RATIOS)},
        sharex=True,
    )
    plt.style.use("fast")

    ax1.plot(df.index, df["close"], label=f"{coin_name} Close", color=CLOSE_COLOR, linewidth=CLOSE_WIDTH)
    ax1.plot(
        df.index, df[f"SMA{SMA_SLOW}"],
        label=f"{SMA_SLOW} SMA",
        color=SMA200_COLOR,
        linewidth=SMA200_WIDTH,
        linestyle="--",
    )
    title = f"{coin_name} Price + Rolling Z-Score ({ZSCORE_WINDOW}d window)"
    if LOG_SCALE:
        ax1.set_yscale("log")
        title += " (LOG SCALE)"
    if DAYS_BACK:
        title += f" — Last {DAYS_BACK} days"
    ax1.set_title(title, fontsize=14, pad=16)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format_price(x)))
    if SHOW_GRID:
        ax1.grid(True, color=PRICE_GRID_COLOR, linewidth=PRICE_GRID_LINEWIDTH, alpha=PRICE_GRID_ALPHA)

    ax2.plot(df.index, df[z_col], label=f"Z-Score ({ZSCORE_WINDOW}d)", color=ZSCORE_COLOR, linewidth=1.5)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.9, alpha=0.8, label="Mean (0)")
    ax2.axhline(2, color="red", linestyle="--", alpha=0.7, label="+2σ")
    ax2.axhline(-2, color="green", linestyle="--", alpha=0.7, label="-2σ")
    ax2.axhline(3, color="darkred", linestyle=":", alpha=0.6)
    ax2.axhline(-3, color="darkgreen", linestyle=":", alpha=0.6)
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.92)
    if SHOW_GRID:
        ax2.grid(True, color=ZSCORE_GRID_COLOR, linewidth=ZSCORE_GRID_LINEWIDTH, alpha=ZSCORE_GRID_ALPHA)

    add_year_quarter_date_formatters(ax1)
    add_year_quarter_date_formatters(ax2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()

    print(f"Drawing {coin_name} price + z-score (window={ZSCORE_WINDOW}d)...")

    if not _backend_is_interactive():
        safe = coin_ticker.lower().replace(" ", "_")
        out = f"{safe}_zscore.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to: {out}")
        plt.close(fig)
        return

    print(f"Using interactive backend: {matplotlib.get_backend()}")
    plt.show(block=block_window)
    if close_after and block_window:
        plt.close(fig)


def draw(block_window: bool = BLOCK_WINDOW) -> None:
    choices = get_coin_choice("Price + Z-Score Chart - Coin Selection")
    total = len(choices)
    for i, (coin_name, coin_ticker) in enumerate(choices, start=1):
        is_last = i == total
        if total > 1:
            print(f"\n[{i}/{total}] {coin_name}")
        per_coin_block = block_window if total == 1 else True
        try:
            draw_one_chart(
                coin_name,
                coin_ticker,
                block_window=per_coin_block,
                close_after=True,
            )
        except Exception as exc:
            print(f"✘ {coin_name} ({coin_ticker}) failed: {exc}")
        if is_last and total > 1:
            print(f"\nDone. Stopped after last coin ({coin_name}).")


if __name__ == "__main__":
    draw()
