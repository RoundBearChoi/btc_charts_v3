#!/usr/bin/env python3
"""Coin ratio from existing daily caches.

Does not download. Reads:
    src/cg_data/{SYMBOL}_data_daily.csv

Seed those files with analyst_get_daily_data.py first.
Demo daily is only a short increment.

Prompts twice via coin_menu (base, then quote). ALL on either
side expands to every ordered pair with two different symbols.
Ratio is base close / quote close. Swap the prompts to invert.
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
DAYS_BACK = None           # None = full overlapping daily history
BLOCK_WINDOW = True
SHOW_GRID = True
MIN_OVERLAP_DAYS = 30

MA_WINDOWS = (7, 30)
ZSCORE_WINDOW = 90
ZSCORE_OVER = 2.0
ZSCORE_UNDER = -2.0

FIGURE_SIZE = (14, 8.4)
HEIGHT_RATIOS = (3.0, 1.15)

RATIO_COLOR = "#2E86AB"
MA_COLORS = ("#E8871E", "#C73E1D")
ZSCORE_COLOR = "#C0392B"
# ====================================================


def _backend_is_interactive() -> bool:
    backend = matplotlib.get_backend().lower()
    return backend not in {"agg", "svg", "pdf", "ps", "cairo", "template"}


def _require_cached_daily(name: str, symbol: str) -> pd.DataFrame:
    path = daily_cache_path(symbol)
    raw = load_daily(symbol)
    if raw.empty or "close" not in raw.columns:
        raise FileNotFoundError(
            f"No daily cache for {name} ({symbol}) at {path}.\n"
            "Run analyst_get_daily_data.py first."
        )
    raw = raw.sort_index()
    raw = raw[raw["close"].notna() & (raw["close"] != 0)]
    if raw.empty:
        raise ValueError(f"{path} has no usable close prices.")
    print(
        f"Using cached {symbol} daily: {raw.index.min().date()} → "
        f"{raw.index.max().date()} ({len(raw)} rows)\n  {path}"
    )
    return raw


def build_pairs(
    bases: list[tuple[str, str]],
    quotes: list[tuple[str, str]],
) -> list[tuple[str, str, str, str]]:
    pairs = []
    seen: set[tuple[str, str]] = set()
    for base_name, base_symbol in bases:
        for quote_name, quote_symbol in quotes:
            if base_symbol == quote_symbol:
                continue
            key = (base_symbol, quote_symbol)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((base_name, base_symbol, quote_name, quote_symbol))
    return pairs


def align_ratio(
    base: pd.DataFrame,
    quote: pd.DataFrame,
    base_name: str,
    quote_name: str,
    days_back: int | None = DAYS_BACK,
) -> pd.DataFrame:
    common = base.index.intersection(quote.index)
    if len(common) < MIN_OVERLAP_DAYS:
        raise ValueError(
            f"Only {len(common)} overlapping days for {base_name}/{quote_name} "
            f"(need {MIN_OVERLAP_DAYS}). Seed both with analyst_get_daily_data.py."
        )
    ratio = (base.loc[common, "close"] / quote.loc[common, "close"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df = pd.DataFrame({"close": ratio}).dropna()
    df = df.sort_index()
    if days_back is not None and days_back > 0:
        df = df.iloc[-min(days_back, len(df)):]
    if df.empty:
        raise ValueError(f"No usable {base_name}/{quote_name} ratio rows.")
    print(
        f"  Overlap: {df.index.min().date()} → {df.index.max().date()} "
        f"({len(df)} days)  current={df['close'].iloc[-1]:.4f}"
    )
    return df


def add_sma(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.DataFrame:
    df[f"SMA{window}"] = df[price_col].rolling(window=window).mean()
    return df


def add_zscore(df: pd.DataFrame, window: int, price_col: str = "close") -> pd.DataFrame:
    min_periods = max(30, window // 2)
    mean = df[price_col].rolling(window=window, min_periods=min_periods).mean()
    std = df[price_col].rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    df["ZScore"] = (df[price_col] - mean) / std
    return df


def format_ratio(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    abs_v = abs(value)
    if abs_v >= 1000:
        return f"{value:,.0f}"
    if abs_v >= 1:
        return f"{value:,.4f}"
    return f"{value:.6f}"


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


def print_snapshot(
    df: pd.DataFrame,
    base_name: str,
    base_symbol: str,
    quote_name: str,
    quote_symbol: str,
) -> None:
    last = df.iloc[-1]
    print("=" * 64)
    print(f"{base_name}/{quote_name}  ({base_symbol}/{quote_symbol})")
    print("=" * 64)
    print(f"Latest ratio:    {format_ratio(float(last['close']))}   ({df.index[-1].date()})")
    print(f"Range:           {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} days)")
    print(f"Window min/max:  {format_ratio(float(df['close'].min()))} → {format_ratio(float(df['close'].max()))}")
    z = last["ZScore"] if "ZScore" in df.columns else float("nan")
    print(f"Z-Score({ZSCORE_WINDOW}):   {float(z):.2f}" if pd.notna(z) else f"Z-Score({ZSCORE_WINDOW}):   n/a")
    print()


def draw_one_chart(
    base_name: str,
    base_symbol: str,
    quote_name: str,
    quote_symbol: str,
    *,
    days_back: int | None = DAYS_BACK,
    block_window: bool = BLOCK_WINDOW,
    close_after: bool = True,
) -> None:
    base = _require_cached_daily(base_name, base_symbol)
    quote = _require_cached_daily(quote_name, quote_symbol)
    df = align_ratio(base, quote, base_name, quote_name, days_back=days_back)
    for window in MA_WINDOWS:
        df = add_sma(df, window)
    df = add_zscore(df, ZSCORE_WINDOW)
    print_snapshot(df, base_name, base_symbol, quote_name, quote_symbol)

    label = f"{base_name} / {quote_name}"
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        sharex=True,
        gridspec_kw={"height_ratios": list(HEIGHT_RATIOS)},
    )
    plt.style.use("fast")

    ax_top.plot(df.index, df["close"], color=RATIO_COLOR, linewidth=1.15, label=label)
    for i, window in enumerate(MA_WINDOWS):
        col = f"SMA{window}"
        ax_top.plot(
            df.index,
            df[col],
            color=MA_COLORS[i % len(MA_COLORS)],
            linewidth=1.4,
            label=col,
        )
    ax_top.set_title(
        f"{label}  •  SMA{'+'.join(str(w) for w in MA_WINDOWS)}  •  "
        f"Z-Score {ZSCORE_WINDOW}d  —  {df.index[0].date()} → {df.index[-1].date()}",
        fontsize=12,
        pad=10,
    )
    ax_top.set_ylabel(f"Ratio ({base_symbol}/{quote_symbol})")
    ax_top.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _p: format_ratio(x))
    )
    current = float(df["close"].iloc[-1])
    ax_top.annotate(
        f"Current: {format_ratio(current)}",
        xy=(df.index[-1], current),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
    )
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.92)
    if SHOW_GRID:
        ax_top.grid(True, alpha=0.3)

    z = df["ZScore"]
    ax_bot.plot(df.index, z, color=ZSCORE_COLOR, linewidth=1.2, label=f"Z-Score({ZSCORE_WINDOW})")
    ax_bot.axhline(0, color="#707070", linewidth=1.0)
    ax_bot.axhline(ZSCORE_OVER, color="#E74C3C", linestyle="--", linewidth=1.1, label=f"+{ZSCORE_OVER}σ")
    ax_bot.axhline(ZSCORE_UNDER, color="#27AE60", linestyle="--", linewidth=1.1, label=f"{ZSCORE_UNDER}σ")
    ax_bot.axhline(1.0, color="#B0AFAB", linestyle=":", linewidth=0.9, alpha=0.7)
    ax_bot.axhline(-1.0, color="#B0AFAB", linestyle=":", linewidth=0.9, alpha=0.7)
    finite = z.dropna()
    if not finite.empty:
        top = max(float(finite.max()) + 0.3, ZSCORE_OVER + 0.8, 3.0)
        bot = min(float(finite.min()) - 0.3, ZSCORE_UNDER - 0.8, -3.0)
        ax_bot.axhspan(ZSCORE_OVER, top, color="red", alpha=0.06)
        ax_bot.axhspan(bot, ZSCORE_UNDER, color="green", alpha=0.06)
        ax_bot.set_ylim(bot, top)
    ax_bot.set_ylabel(f"Z-Score\n({ZSCORE_WINDOW}d)")
    ax_bot.set_xlabel("Date")
    ax_bot.legend(loc="upper left", fontsize=8, framealpha=0.92)
    if SHOW_GRID:
        ax_bot.grid(True, alpha=0.3)

    add_quarter_date_formatters(ax_top)
    add_quarter_date_formatters(ax_bot)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()

    if not _backend_is_interactive():
        out = f"{base_symbol.lower()}_{quote_symbol.lower()}_ratio.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to: {out}")
        plt.close(fig)
        return

    print(f"Using interactive backend: {matplotlib.get_backend()}")
    plt.show(block=block_window)
    if close_after and block_window:
        plt.close(fig)


def draw(days_back: int | None = DAYS_BACK, block_window: bool = BLOCK_WINDOW) -> None:
    bases = get_coin_choice("Ratio base - Coin Selection")
    quotes = get_coin_choice("Ratio quote - Coin Selection")
    pairs = build_pairs(bases, quotes)
    if not pairs:
        raise SystemExit("Need two different coins. Same coin cannot be both base and quote.")

    total = len(pairs)
    for i, (base_name, base_symbol, quote_name, quote_symbol) in enumerate(pairs, start=1):
        is_last = i == total
        if total > 1:
            print(f"\n[{i}/{total}] {base_name}/{quote_name}")
        try:
            draw_one_chart(
                base_name,
                base_symbol,
                quote_name,
                quote_symbol,
                days_back=days_back,
                block_window=block_window if is_last else True,
                close_after=not is_last,
            )
        except Exception as exc:
            print(f"✘ {base_name}/{quote_name} failed: {exc}")
            continue


if __name__ == "__main__":
    draw()
