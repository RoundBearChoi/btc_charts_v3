#!/usr/bin/env python3
"""Last 72 hours of price + derived 1h volume from the existing hourly cache.

Does not download. Reads:
    src/cg_data/{SYMBOL}_data_hourly.csv
Seed that with analyst_get_hourly_data.py (or demo_get_hourly_data.py) first.

CoinGecko's `volume` column is a sliding 24h USD sum, not that hour's
session volume. Bar height is max(volume.diff(), 0) so the pane reads
like hourly volume. Color is green when that hour's price is >= the
previous hour, red otherwise.

The visible window is the last HOURS_BACK hours of the cache (from the
latest cached timestamp), not wall-clock now. Cache timestamps stay
naive UTC; the chart and snapshot print in DISPLAY_TZ (Asia/Seoul).

Startup prompt is 1..N from coins.csv, plus ALL.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from analyst_get_hourly_data import hourly_cache_path, load_hourly
from coin_menu import get_coin_choice

# ====================== CONFIG ======================
HOURS_BACK = 72
BLOCK_WINDOW = True
SHOW_GRID = True
FIGURE_SIZE = (14, 8)
HEIGHT_RATIOS = (3, 1)

# Cache stays naive UTC. Convert only for the chart + snapshot.
DISPLAY_TZ = ZoneInfo("Asia/Seoul")
DISPLAY_TZ_LABEL = "KST UTC+9"

CLOSE_COLOR = "#9EB3DB"
CLOSE_WIDTH = 1.4
VOLUME_UP_COLOR = "#2ca02c"
VOLUME_DOWN_COLOR = "#d62728"
VOLUME_ALPHA = 0.85

GRID_COLOR = "gray"
GRID_LINEWIDTH = 1.0
GRID_ALPHA = 0.55
GRID_LINESTYLE = ":"
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


def format_volume(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


def to_display_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Naive cache index is UTC. Return a copy labeled in DISPLAY_TZ."""
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx.tz_convert(DISPLAY_TZ)
    out.index.name = df.index.name
    return out


def format_ts(ts) -> str:
    """Format a cache or plot timestamp in DISPLAY_TZ."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC").tz_convert(DISPLAY_TZ)
    else:
        t = t.tz_convert(DISPLAY_TZ)
    return t.strftime("%Y-%m-%d %H:%M %Z")


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


def add_hour_date_formatters(ax) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6, tz=DISPLAY_TZ))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1, tz=DISPLAY_TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=DISPLAY_TZ))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _require_hourly(symbol: str) -> pd.DataFrame:
    path = hourly_cache_path(symbol)
    raw = load_hourly(symbol)
    if raw.empty or "price" not in raw.columns:
        raise FileNotFoundError(
            f"No {symbol} hourly cache at {path}.\n"
            "Run analyst_get_hourly_data.py (or demo_get_hourly_data.py) first."
        )
    raw = raw.sort_index()
    raw = raw[raw["price"].notna()]
    if raw.empty:
        raise ValueError(f"{path} has no usable prices.")
    print(
        f"Using cached {symbol} hourly: {raw.index.min()} → "
        f"{raw.index.max()} UTC ({len(raw)} rows)\n  {path}"
    )
    return raw


def _window_from_cache_end(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Keep rows in the last `hours` hours of the cache (not wall-clock now)."""
    latest = df.index.max()
    start = latest - pd.Timedelta(hours=hours)
    window = df.loc[df.index >= start]
    if window.empty:
        window = df.iloc[-hours:]
    return window


def print_snapshot(df: pd.DataFrame, coin_name: str, coin_ticker: str) -> None:
    last = df.iloc[-1]
    first = df.iloc[0]
    price = float(last["price"])
    prev = float(first["price"])
    change = price - prev
    pct = (change / prev * 100.0) if prev else float("nan")
    last_vol = float(last["volume_1h"]) if pd.notna(last["volume_1h"]) else float("nan")
    peak_vol = float(df["volume_1h"].max()) if df["volume_1h"].notna().any() else float("nan")
    up_hours = int((df["price_change"] >= 0).sum())
    down_hours = int((df["price_change"] < 0).sum())

    print("=" * 64)
    print(f"{coin_name} ({coin_ticker})  last {HOURS_BACK}h volume snapshot")
    print("=" * 64)
    print(f"Latest price:    {format_price(price)}   ({format_ts(df.index[-1])})")
    print(
        f"Window:          {format_ts(df.index[0])} → "
        f"{format_ts(df.index[-1])}  ({len(df)} hours)"
    )
    print(f"Window change:   {format_price(change)}  ({pct:+.2f}%)")
    print(f"Last 1h vol:     {format_volume(last_vol)}")
    print(f"Peak 1h vol:     {format_volume(peak_vol)}")
    print(f"Up / down hours: {up_hours} / {down_hours}")
    print("=" * 64)
    print()


def draw_one_chart(
    coin_name: str,
    coin_ticker: str,
    *,
    block_window: bool = BLOCK_WINDOW,
    close_after: bool = True,
) -> None:
    full = _require_hourly(coin_ticker).copy()

    # Color and derived volume need one extra row before the window.
    latest = full.index.max()
    lookback_start = latest - pd.Timedelta(hours=HOURS_BACK)
    padded = full.loc[full.index >= (lookback_start - pd.Timedelta(hours=1))].copy()
    if padded.empty:
        padded = full.copy()

    padded["price_change"] = padded["price"].diff()
    if "volume" not in padded.columns:
        padded["volume"] = float("nan")
    padded["volume_1h"] = padded["volume"].diff().clip(lower=0)

    df = _window_from_cache_end(padded, HOURS_BACK)
    if df.empty:
        raise ValueError(f"{coin_ticker} has no rows in the last {HOURS_BACK} hours of cache.")

    df = to_display_tz(df)
    print_snapshot(df, coin_name, coin_ticker)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        gridspec_kw={"height_ratios": list(HEIGHT_RATIOS)},
        sharex=True,
    )
    plt.style.use("fast")

    ax1.plot(
        df.index,
        df["price"],
        label=f"{coin_name} Price",
        linewidth=CLOSE_WIDTH,
        color=CLOSE_COLOR,
        zorder=3,
    )
    ax1.set_title(f"{coin_name} • last {HOURS_BACK} hours", fontsize=14, pad=16)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format_price(x)))
    apply_grid(ax1)

    if df["volume_1h"].notna().any():
        vol_colors = [
            VOLUME_UP_COLOR if (pd.notna(chg) and chg >= 0) else VOLUME_DOWN_COLOR
            for chg in df["price_change"]
        ]
        # ~40 minutes in day-units so 72 hourly bars don't melt together.
        bar_width = 1.0 / 24.0 * 0.7
        ax2.bar(
            df.index,
            df["volume_1h"],
            color=vol_colors,
            alpha=VOLUME_ALPHA,
            width=bar_width,
            align="center",
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No volume column in cache",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    ax2.set_ylabel("Approx. 1h Volume (USD)")
    ax2.set_xlabel(f"{DISPLAY_TZ_LABEL}")
    apply_grid(ax2)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: format_volume(x)))

    add_hour_date_formatters(ax1)
    add_hour_date_formatters(ax2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.tick_params(axis="x", labelbottom=False)
    fig.tight_layout()

    print(f"Drawing {coin_name} last-{HOURS_BACK}h price + volume...")

    if not _backend_is_interactive():
        safe = coin_ticker.lower().replace(" ", "_")
        out = f"{safe}_latest_volume.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to: {out}")
        plt.close(fig)
        return

    print(f"Using interactive backend: {matplotlib.get_backend()}")
    plt.show(block=block_window)
    if close_after and block_window:
        plt.close(fig)


def draw(block_window: bool = BLOCK_WINDOW) -> None:
    choices = get_coin_choice("Latest 72h Volume Chart - Coin Selection")
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
