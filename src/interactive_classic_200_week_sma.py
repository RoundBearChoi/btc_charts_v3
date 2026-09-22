#!/usr/bin/env python3
"""BTC daily close with an adjustable Sunday-week SMA.

Price is cache-only:
    src/cg_data/BTC_data_daily.csv
Seed that with analyst_get_daily_data.py first.

Weekly closes are W-SUN. The slider retunes the SMA window
(default 195, classic 200-week pulled in a little).
A 195-week SMA needs ~3.8 years of daily history.
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
from matplotlib.widgets import Slider
import pandas as pd

from analyst_get_daily_data import daily_cache_path, load_daily

# ==================== CONFIG ====================
MIN_WEEKS = 3
MAX_WEEKS = 250
DEFAULT_WEEKS = 195

FIGURE_SIZE = (12, 8)
BLOCK_WINDOW = True
SHOW_GRID = False

PRICE_COLOR = "#1f77b4"
SMA_COLOR = "orange"
# ================================================


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
    if not isinstance(raw.index, pd.DatetimeIndex):
        raw.index = pd.to_datetime(raw.index)
    print(
        f"Using cached BTC daily: {raw.index.min().date()} → "
        f"{raw.index.max().date()} ({len(raw)} rows)\n  {path}"
    )
    return raw


def sunday_weekly_close(daily: pd.Series) -> pd.Series:
    return daily.resample("W-SUN").last().dropna()


def print_2018_diagnostic(weekly: pd.Series, sma: pd.Series) -> None:
    bottom = weekly.loc["2018-12-01":"2019-02-28"]
    if bottom.empty:
        print("2018-2019 bottom not in cache; skip diagnostic.")
        return
    min_idx = bottom.idxmin()
    min_price = float(bottom.min())
    sma_at_min = float(sma.loc[min_idx]) if min_idx in sma.index and pd.notna(sma.loc[min_idx]) else None
    print()
    print("=== 2018-2019 Bottom Diagnostic ===")
    print(f"Lowest weekly close : {min_price:,.2f} on {min_idx.date()}")
    if sma_at_min is None:
        print("SMA at that date    : n/a (window not filled yet)")
    else:
        dipped = min_price < sma_at_min
        print(f"SMA at that date    : {sma_at_min:,.2f}")
        print(f"Price dipped below SMA? → {'YES' if dipped else 'NO (very close)'}")


def draw(
    initial_weeks: int = DEFAULT_WEEKS,
    min_weeks: int = MIN_WEEKS,
    max_weeks: int = MAX_WEEKS,
    block_window: bool = BLOCK_WINDOW,
) -> None:
    daily = _require_btc_daily()
    weekly = sunday_weekly_close(daily["close"])
    filled_weeks = weekly.notna().sum()
    print(f"Sunday weekly closes: {weekly.index.min().date()} → {weekly.index.max().date()} ({filled_weeks} weeks)")
    if filled_weeks < initial_weeks:
        print(
            f"Cache has {filled_weeks} weeks; {initial_weeks}-week SMA needs {initial_weeks}. "
            "Run analyst_get_daily_data.py for a full history."
        )

    sma = weekly.rolling(window=initial_weeks).mean()
    print_2018_diagnostic(weekly, sma)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    plt.style.use("fast")
    fig.subplots_adjust(bottom=0.16, left=0.10, right=0.96, top=0.92)

    ax.plot(daily.index, daily["close"], label="Bitcoin Price", linewidth=1.2, color=PRICE_COLOR)
    ma_line, = ax.plot(
        sma.index, sma,
        label=f"{initial_weeks}-Week Moving Average",
        linewidth=1.5,
        color=SMA_COLOR,
    )
    ax.set_title("Bitcoin Price with Real-Time Adjustable Weekly Moving Average")
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Time")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _p: f"${x:,.0f}"))
    ax.legend(loc="upper left")
    if SHOW_GRID:
        ax.grid(True, alpha=0.3)
    add_year_quarter_date_formatters(ax)

    print(f"Drawing weekly SMA (range: {min_weeks}-{max_weeks} weeks, start {initial_weeks})")
    print("  → Sunday weekly closes")

    if not _backend_is_interactive():
        out = "btc_200_week_sma.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to: {out} (slider needs an interactive backend)")
        plt.close(fig)
        return

    slider_ax = fig.add_axes([0.20, 0.04, 0.60, 0.03], facecolor="lightgray")
    weeks_slider = Slider(
        ax=slider_ax,
        label="Moving Average (weeks)",
        valmin=min_weeks,
        valmax=max_weeks,
        valinit=initial_weeks,
        valstep=1,
        valfmt="%d weeks",
    )

    def update(_val) -> None:
        weeks = int(weeks_slider.val)
        series = weekly.rolling(window=weeks).mean()
        ma_line.set_data(series.index, series.to_numpy())
        ma_line.set_label(f"{weeks}-Week Moving Average")
        ax.legend(loc="upper left")
        fig.canvas.draw_idle()

    weeks_slider.on_changed(update)
    print(f"Using interactive backend: {matplotlib.get_backend()}")
    plt.show(block=block_window)
    if block_window:
        plt.close(fig)


if __name__ == "__main__":
    draw()
