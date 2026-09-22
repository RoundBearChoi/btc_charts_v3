#!/usr/bin/env python3
"""SMA111 vs SMA50 + RSI from the existing daily cache.

Does not download. Reads:
    src/cg_data/{SYMBOL}_data_daily.csv
Seed that with analyst_get_daily_data.py first.

Indicators are computed on the full cache, then the last DAYS_BACK
rows are drawn so SMA111 is seeded.

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
DAYS_BACK = 180
BLOCK_WINDOW = True
SHOW_GRID = True
FIGURE_SIZE = (14, 8)
HEIGHT_RATIOS = (3, 1)

RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

SMA_FAST = 50
SMA_SLOW = 111

CLOSE_COLOR = "#9EB3DB"
CLOSE_WIDTH = 1.1
SMA111_COLOR = "#E15FC3"
SMA50_COLOR = "#00D118"
RSI_COLOR = "#FF9900"
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


def add_sma(df: pd.DataFrame, window: int, price_col: str = "close", out_col: str | None = None) -> pd.DataFrame:
    if out_col is None:
        out_col = f"SMA{window}"
    df[out_col] = df[price_col].rolling(window=window).mean()
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
    fast_col = f"SMA{SMA_FAST}"
    slow_col = f"SMA{SMA_SLOW}"
    df = add_sma(df, window=SMA_SLOW, out_col=slow_col)
    df = add_sma(df, window=SMA_FAST, out_col=fast_col)
    df = add_rsi(df, window=RSI_WINDOW)

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

    ax1.plot(df.index, df["close"], label=f"{coin_name} Close Price", linewidth=CLOSE_WIDTH, color=CLOSE_COLOR)
    ax1.plot(df.index, df[slow_col], label=f"{SMA_SLOW}-Day SMA", linewidth=0.95, color=SMA111_COLOR)
    ax1.plot(df.index, df[fast_col], label=f"{SMA_FAST}-Day SMA", linewidth=0.95, color=SMA50_COLOR)

    title = f"{coin_name} • {SMA_SLOW}-Day SMA vs {SMA_FAST}-Day SMA + RSI({RSI_WINDOW})"
    if DAYS_BACK:
        title += f" — Last {DAYS_BACK} days"
    ax1.set_title(title, fontsize=14, pad=16)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format_price(x)))
    if SHOW_GRID:
        ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df["RSI"], color=RSI_COLOR, linewidth=1.5, label=f"RSI({RSI_WINDOW})")
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

    print(f"Drawing {coin_name} chart with {SMA_SLOW}/{SMA_FAST} SMAs + RSI({RSI_WINDOW})...")

    if not _backend_is_interactive():
        safe = coin_ticker.lower().replace(" ", "_")
        out = f"{safe}_sma_vs_sma.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to: {out}")
        plt.close(fig)
        return

    print(f"Using interactive backend: {matplotlib.get_backend()}")
    plt.show(block=block_window)
    if close_after and block_window:
        plt.close(fig)


def draw(block_window: bool = BLOCK_WINDOW) -> None:
    choices = get_coin_choice("111/50 SMA + RSI Chart - Coin Selection")
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
