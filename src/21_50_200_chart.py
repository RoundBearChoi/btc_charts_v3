#!/usr/bin/env python3
"""EMA21 / SMA50 / SMA200 + volume + RSI from the existing daily cache.

Does not download. Reads:
    src/cg_data/{SYMBOL}_data_daily.csv
Seed that with analyst_get_daily_data.py first.

Indicators are computed on the full cache, then the visible window is
the last DAYS_BACK rows so SMA200 is seeded when history exists.

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

# ==================================================
# CONFIGURATION
# ==================================================
LOG_SCALE = False
DAYS_BACK = 360 * 6
BLOCK_WINDOW = True
SHOW_GRID = True

EMA_FAST = 21
SMA_MID = 50
SMA_SLOW = 200

GRID_COLOR = "gray"
GRID_LINEWIDTH = 1.0
GRID_ALPHA = 0.7
GRID_LINESTYLE = ":"

FIGURE_SIZE = (14, 10)
HEIGHT_RATIOS = (3, 1, 1)

RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

SHOW_TREND_CLOUD = True
SHOW_CROSSES = True
COLOR_VOLUME_BY_DIRECTION = True
SHOW_LAST_LABELS = True
SHOW_RSI_ZONES = True

CLOSE_COLOR = "#9EB3DB"
CLOSE_WIDTH = 0.9
EMA21_COLOR = "#E15FC3"
SMA50_COLOR = "#00D118"
SMA200_COLOR = "#C80C01"
VOLUME_COLOR = "#8F8C57"
VOLUME_UP_COLOR = "#2ca02c"
VOLUME_DOWN_COLOR = "#d62728"
VOLUME_SMA_DAYS = 15
VOLUME_SMA_COLOR = "#263549"
CLOUD_UP_COLOR = "#00D118"
CLOUD_DOWN_COLOR = "#C80C01"

CROSS_MARKER_SIZE = 36
FAST_GOLDEN_CROSS_COLOR = "#00D118"
FAST_DEATH_CROSS_COLOR = "#C80C01"
SLOW_GOLDEN_CROSS_COLOR = "#F5C518"
SLOW_DEATH_CROSS_COLOR = "#5B6CFF"
# ==================================================


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


def format_volume(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.0f}M"
    return f"${value:,.0f}"


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


def apply_grid(ax) -> None:
    if not SHOW_GRID:
        ax.grid(False)
        return
    ax.grid(
        True,
        color=GRID_COLOR,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        linestyle=GRID_LINESTYLE,
    )


def add_ema(df: pd.DataFrame, span: int, price_col: str = "close", out_col: str | None = None) -> pd.DataFrame:
    if out_col is None:
        out_col = f"EMA{span}"
    df[out_col] = df[price_col].ewm(span=span, adjust=False).mean()
    return df


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


def last_crossover(fast: pd.Series, slow: pd.Series) -> tuple[pd.Timestamp | None, str | None]:
    aligned = pd.concat({"fast": fast, "slow": slow}, axis=1).dropna()
    if len(aligned) < 2:
        return None, None
    prev_fast = aligned["fast"].shift(1)
    prev_slow = aligned["slow"].shift(1)
    crossed_above = (aligned["fast"] > aligned["slow"]) & (prev_fast <= prev_slow)
    crossed_below = (aligned["fast"] < aligned["slow"]) & (prev_fast >= prev_slow)
    last_above = aligned.index[crossed_above].max() if crossed_above.any() else None
    last_below = aligned.index[crossed_below].max() if crossed_below.any() else None
    if last_above is None and last_below is None:
        return None, None
    if last_below is None or (last_above is not None and last_above >= last_below):
        return last_above, "crossed above"
    return last_below, "crossed below"


def _pct_from(price: float, level: float) -> str:
    if pd.isna(level) or level == 0:
        return "n/a"
    return f"{(price - level) / level * 100:+.1f}%"


def _regime_label(price: float, ema_fast: float, sma_mid: float, sma_slow: float) -> str:
    if any(pd.isna(v) for v in (price, ema_fast, sma_mid, sma_slow)):
        return "n/a"
    above_slow = price >= sma_slow
    fast_above_mid = ema_fast >= sma_mid
    if above_slow and fast_above_mid:
        return f"risk-on (above {SMA_SLOW} · {EMA_FAST}>{SMA_MID})"
    if above_slow:
        return f"above {SMA_SLOW} · {EMA_FAST}<{SMA_MID}"
    if fast_above_mid:
        return f"below {SMA_SLOW} · {EMA_FAST}>{SMA_MID}"
    return f"defensive (below {SMA_SLOW} · {EMA_FAST}<{SMA_MID})"


def _format_cross(ts, direction: str | None) -> str:
    if ts is None or direction is None:
        return "none in view"
    when = ts.date() if hasattr(ts, "date") else ts
    return f"{when} {direction}"


def print_snapshot(df: pd.DataFrame, coin_name: str, coin_ticker: str, rsi_window: int) -> None:
    last = df.iloc[-1]
    price = float(last["close"])
    ema_col = f"EMA{EMA_FAST}"
    mid_col = f"SMA{SMA_MID}"
    slow_col = f"SMA{SMA_SLOW}"
    ema = float(last[ema_col]) if pd.notna(last[ema_col]) else float("nan")
    mid = float(last[mid_col]) if pd.notna(last[mid_col]) else float("nan")
    slow = float(last[slow_col]) if pd.notna(last[slow_col]) else float("nan")
    rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else float("nan")

    print("=" * 64)
    print(f"{coin_name} ({coin_ticker})  {EMA_FAST}/{SMA_MID}/{SMA_SLOW} snapshot")
    print("=" * 64)
    print(f"Latest close:    {format_price(price)}   ({df.index[-1].date()})")
    print(f"Range:           {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} days)")
    print(f"Regime:          {_regime_label(price, ema, mid, slow)}")
    print()
    print(f"{'MA':<10}{'Value':>14}{'vs close':>12}")
    print("-" * 36)
    print(f"{'EMA' + str(EMA_FAST):<10}{format_price(ema):>14}{_pct_from(price, ema):>12}")
    print(f"{'SMA' + str(SMA_MID):<10}{format_price(mid):>14}{_pct_from(price, mid):>12}")
    print(f"{'SMA' + str(SMA_SLOW):<10}{format_price(slow):>14}{_pct_from(price, slow):>12}")
    print()
    rsi_txt = f"{rsi:.1f}" if pd.notna(rsi) else "n/a"
    print(f"RSI({rsi_window}):       {rsi_txt}")
    print(f"Last {EMA_FAST}/{SMA_MID} cross:  {_format_cross(*last_crossover(df[ema_col], df[mid_col]))}")
    print(f"Last {SMA_MID}/{SMA_SLOW} cross: {_format_cross(*last_crossover(df[mid_col], df[slow_col]))}")
    print("=" * 64)
    print()


def _mark_crosses(
    ax,
    df: pd.DataFrame,
    fast_col: str,
    slow_col: str,
    *,
    size: int,
    golden_color: str,
    death_color: str,
    label_prefix: str,
) -> None:
    prev_fast = df[fast_col].shift(1)
    prev_slow = df[slow_col].shift(1)
    golden = (df[fast_col] > df[slow_col]) & (prev_fast <= prev_slow)
    death = (df[fast_col] < df[slow_col]) & (prev_fast >= prev_slow)
    g = df.loc[golden]
    d = df.loc[death]
    if not g.empty:
        ax.scatter(
            g.index, g[fast_col],
            color=golden_color, s=size, marker="^", zorder=6,
            label=f"{label_prefix} golden",
        )
    if not d.empty:
        ax.scatter(
            d.index, d[fast_col],
            color=death_color, s=size, marker="v", zorder=6,
            label=f"{label_prefix} death",
        )


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
    ema_col = f"EMA{EMA_FAST}"
    mid_col = f"SMA{SMA_MID}"
    slow_col = f"SMA{SMA_SLOW}"

    df = add_ema(df, EMA_FAST, out_col=ema_col)
    df = add_sma(df, SMA_MID, out_col=mid_col)
    df = add_sma(df, SMA_SLOW, out_col=slow_col)
    if VOLUME_SMA_DAYS > 0 and "volumeto" in df.columns:
        df["VOLUME_SMA"] = df["volumeto"].rolling(window=VOLUME_SMA_DAYS).mean()
    df = add_rsi(df, window=RSI_WINDOW)

    if DAYS_BACK is not None:
        df = df.iloc[-DAYS_BACK:]
    if df.empty:
        raise ValueError(f"{coin_ticker} has no rows in the visible window.")

    print_snapshot(df, coin_name, coin_ticker, RSI_WINDOW)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=FIGURE_SIZE,
        gridspec_kw={"height_ratios": list(HEIGHT_RATIOS)},
        sharex=True,
    )
    plt.style.use("fast")

    if SHOW_TREND_CLOUD:
        ax1.fill_between(
            df.index, df[ema_col], df[mid_col],
            where=df[ema_col] >= df[mid_col],
            color=CLOUD_UP_COLOR, alpha=0.12, interpolate=True, zorder=0,
        )
        ax1.fill_between(
            df.index, df[ema_col], df[mid_col],
            where=df[ema_col] < df[mid_col],
            color=CLOUD_DOWN_COLOR, alpha=0.12, interpolate=True, zorder=0,
        )

    ax1.plot(df.index, df["close"], label=f"{coin_name} Close", linewidth=CLOSE_WIDTH, color=CLOSE_COLOR, zorder=3)
    ax1.plot(df.index, df[ema_col], label=f"{EMA_FAST} EMA", color=EMA21_COLOR, linewidth=1.3, zorder=4)
    ax1.plot(df.index, df[mid_col], label=f"{SMA_MID} SMA", color=SMA50_COLOR, linewidth=1.3, zorder=4)
    ax1.plot(
        df.index, df[slow_col],
        label=f"{SMA_SLOW} SMA (Long-term)",
        color=SMA200_COLOR, linewidth=1.6, linestyle="--", zorder=4,
    )

    if SHOW_CROSSES:
        _mark_crosses(
            ax1, df, mid_col, slow_col,
            size=CROSS_MARKER_SIZE,
            golden_color=SLOW_GOLDEN_CROSS_COLOR,
            death_color=SLOW_DEATH_CROSS_COLOR,
            label_prefix=f"{SMA_MID}/{SMA_SLOW}",
        )
        _mark_crosses(
            ax1, df, ema_col, mid_col,
            size=CROSS_MARKER_SIZE,
            golden_color=FAST_GOLDEN_CROSS_COLOR,
            death_color=FAST_DEATH_CROSS_COLOR,
            label_prefix=f"{EMA_FAST}/{SMA_MID}",
        )

    last = df.iloc[-1]
    price = float(last["close"])
    ema = float(last[ema_col]) if pd.notna(last[ema_col]) else float("nan")
    mid = float(last[mid_col]) if pd.notna(last[mid_col]) else float("nan")
    slow = float(last[slow_col]) if pd.notna(last[slow_col]) else float("nan")
    regime = _regime_label(price, ema, mid, slow)

    title = (
        f"{coin_name} • {EMA_FAST} EMA vs {SMA_MID} SMA + {SMA_SLOW} SMA "
        f"+ Volume + RSI({RSI_WINDOW})"
    )
    if LOG_SCALE:
        ax1.set_yscale("log")
        title += " (LOG)"
    if DAYS_BACK:
        title += f" — Last {DAYS_BACK} days"
    title += f"\n{regime}"
    ax1.set_title(title, fontsize=13, pad=16)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.92)
    apply_grid(ax1)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format_price(x)))

    if SHOW_LAST_LABELS:
        rsi_val = float(last["RSI"]) if pd.notna(last["RSI"]) else float("nan")
        rsi_line = f"RSI({RSI_WINDOW}) {rsi_val:.1f}" if pd.notna(rsi_val) else f"RSI({RSI_WINDOW}) n/a"
        box = (
            f"Close  {format_price(price)}\n"
            f"EMA{EMA_FAST}  {format_price(ema)}  ({_pct_from(price, ema)})\n"
            f"SMA{SMA_MID}  {format_price(mid)}  ({_pct_from(price, mid)})\n"
            f"SMA{SMA_SLOW} {format_price(slow)}  ({_pct_from(price, slow)})\n"
            f"{rsi_line}"
        )
        ax1.text(
            0.99, 0.02, box,
            transform=ax1.transAxes,
            ha="right", va="bottom",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )

    if "volumeto" not in df.columns:
        ax2.text(0.5, 0.5, "No volume column in cache", ha="center", va="center", transform=ax2.transAxes)
    else:
        if COLOR_VOLUME_BY_DIRECTION and "open" in df.columns:
            vol_colors = [
                VOLUME_UP_COLOR if up else VOLUME_DOWN_COLOR
                for up in df["close"] >= df["open"]
            ]
        elif COLOR_VOLUME_BY_DIRECTION:
            vol_colors = [
                VOLUME_UP_COLOR if up else VOLUME_DOWN_COLOR
                for up in df["close"] >= df["close"].shift(1)
            ]
        else:
            vol_colors = VOLUME_COLOR
        ax2.bar(df.index, df["volumeto"], color=vol_colors, alpha=0.75, width=0.9)
        if "VOLUME_SMA" in df.columns:
            ax2.plot(
                df.index, df["VOLUME_SMA"],
                color=VOLUME_SMA_COLOR, linewidth=1.5, label=f"{VOLUME_SMA_DAYS}d Vol SMA",
            )
            ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylabel("Volume (USD)")
    apply_grid(ax2)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format_volume(x)))

    if SHOW_RSI_ZONES:
        ax3.axhspan(RSI_OVERBOUGHT, 100, color="#E15FC3", alpha=0.08, zorder=0)
        ax3.axhspan(0, RSI_OVERSOLD, color="#00D118", alpha=0.08, zorder=0)
    ax3.plot(df.index, df["RSI"], color="#FF9900", linewidth=1.5, label=f"RSI({RSI_WINDOW})")
    ax3.axhline(RSI_OVERBOUGHT, color="#E15FC3", linestyle="--", alpha=0.6, label="Overbought")
    ax3.axhline(RSI_OVERSOLD, color="#00D118", linestyle="--", alpha=0.6, label="Oversold")
    ax3.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax3.set_ylabel("RSI")
    ax3.set_ylim(0, 100)
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=8)
    apply_grid(ax3)

    add_quarter_date_formatters(ax1)
    add_quarter_date_formatters(ax2)
    add_quarter_date_formatters(ax3)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()

    if not _backend_is_interactive():
        safe = coin_ticker.lower().replace(" ", "_")
        out = f"{safe}_21_50_200.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to: {out}")
        plt.close(fig)
        return

    print(f"Using interactive backend: {matplotlib.get_backend()}")
    plt.show(block=block_window)
    if close_after and block_window:
        plt.close(fig)


def draw(block_window: bool = BLOCK_WINDOW) -> None:
    choices = get_coin_choice("21/50/200 + Volume + RSI Chart - Coin Selection")
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
