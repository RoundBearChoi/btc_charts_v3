"""Monthly RSI vs next Bitcoin halving.

Reads the existing CoinGecko daily cache written by the v3 fetchers.
Does not download, increment, or write any price data.

Expected cache (created by analyst_get_daily_data.py / demo_get_daily_data.py):
    src/cg_data/BTC_data_daily.csv

X-axis is pinned from 2013-04-28 through the last cached daily row.
"""

from __future__ import annotations

import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dateutil.relativedelta as rel
import numpy
import pandas

from analyst_get_daily_data import daily_cache_path, load_daily


# ========================== CONFIG ==========================
AXIS_START = pandas.Timestamp("2013-04-28")
# Major ticks land on the 28th of Apr / Jul / Oct / Jan
AXIS_TICK_MONTHS = (4, 7, 10, 1)
AXIS_TICK_DAY = 28

# Configuration for the thin vertical cycle-progress lines
CYCLE_PROGRESS_LINES = {
    "enabled": True,                    # Set to False to hide all % markers
    "interval_percent": 10,             # ← Change this! (5, 10, 20, 25, 50, etc.)
    "color": "black",
    "linestyle": "--",                  # "--" dashed, ":" dotted, "-." dash-dot, "-" solid
    "linewidth": 0.8,
    "alpha": 0.5,
    "zorder": 0,
}

# Configuration for the halving-date marker lines
HALVING_MARKERS = {
    "enabled": True,                    # Set to False to hide halving lines
    "color": "blue",
    "linestyle": "--",
    "linewidth": 1.7,
    "alpha": 0.5,
    "zorder": 2,                        # drawn on top of progress lines
}

# Configuration for horizontal grid lines on the RSI y-axis
HORIZONTAL_GRID_LINES = {
    "enabled": True,                    # Set to False to hide horizontal reference lines
    "levels": [20, 30, 40, 50, 60, 70, 80],
    "color": "gray",
    "linestyle": ":",
    "linewidth": 1.3,
    "alpha": 0.4,
    "zorder": 1,
}

# Bitcoin halving dates (update future ones as needed)
HALVING_DATES = [
    "2012-11-28",
    "2016-07-09",
    "2020-05-11",
    "2024-04-20",
    "2028-04-11",
]
# ===========================================================


def _month_year_label(x, _pos=None) -> str:
    """Tick text like 'apr 2013'."""
    return mdates.num2date(x).strftime("%b %Y").lower()


def _require_cached_btc_daily() -> pandas.DataFrame:
    """Load BTC daily rows from disk. Never hits CoinGecko."""
    path = daily_cache_path("BTC")
    data_frame = load_daily("BTC")

    if data_frame.empty or "close" not in data_frame.columns:
        raise FileNotFoundError(
            f"No BTC daily cache at {path}.\n"
            "Run analyst_get_daily_data.py first (Demo daily is only a short increment)."
        )

    data_frame = data_frame.sort_index()
    data_frame = data_frame[data_frame["close"].notna()]
    if data_frame.empty:
        raise ValueError(f"{path} has no usable close prices.")

    print(
        f"Using cached BTC daily: {data_frame.index.min().date()} → "
        f"{data_frame.index.max().date()} ({len(data_frame)} rows)\n"
        f"  {path}"
    )
    return data_frame


def draw(block_window):
    data_frame = _require_cached_btc_daily()

    plt.figure(figsize=(14, 6))

    plt.style.use("fast")
    plt.grid(False)

    # Full cached history, month-end last close
    data_frame = data_frame.resample("ME").last().asfreq("ME")

    # Calculate RSI on monthly closes (same method as v2)
    delta = data_frame["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down

    data_frame["RSI"] = 100 - (100 / (1 + rs))

    __plot(data_frame, block_window)


def __plot(data_frame, block_window):
    halving_dates = pandas.to_datetime(HALVING_DATES)
    halving_dates = pandas.DatetimeIndex(halving_dates)

    axis_start = AXIS_START
    axis_end = data_frame.index.max()
    if pandas.isna(axis_end) or axis_end < axis_start:
        raise ValueError(
            f"Cached monthly series ends at {axis_end}, which is before {axis_start.date()}."
        )

    # === 1. Cycle progress lines (every X%) ===
    if CYCLE_PROGRESS_LINES["enabled"]:
        interval = CYCLE_PROGRESS_LINES["interval_percent"]
        for i in range(len(halving_dates) - 1):
            start = halving_dates[i]
            end = halving_dates[i + 1]
            duration = end - start
            for k in range(1, 100 // interval):
                progress_date = start + (k * interval / 100.0) * duration
                plt.axvline(
                    x=progress_date,
                    color=CYCLE_PROGRESS_LINES["color"],
                    linestyle=CYCLE_PROGRESS_LINES["linestyle"],
                    linewidth=CYCLE_PROGRESS_LINES["linewidth"],
                    alpha=CYCLE_PROGRESS_LINES["alpha"],
                    zorder=CYCLE_PROGRESS_LINES["zorder"],
                )

    # === 2. Halving date markers ===
    if HALVING_MARKERS["enabled"]:
        for hd in halving_dates:
            plt.axvline(
                x=hd,
                color=HALVING_MARKERS["color"],
                linestyle=HALVING_MARKERS["linestyle"],
                linewidth=HALVING_MARKERS["linewidth"],
                alpha=HALVING_MARKERS["alpha"],
                zorder=HALVING_MARKERS["zorder"],
            )

    # === 3. Horizontal RSI grid / reference lines ===
    if HORIZONTAL_GRID_LINES["enabled"]:
        for y_level in HORIZONTAL_GRID_LINES["levels"]:
            plt.axhline(
                y=y_level,
                color=HORIZONTAL_GRID_LINES["color"],
                linestyle=HORIZONTAL_GRID_LINES["linestyle"],
                linewidth=HORIZONTAL_GRID_LINES["linewidth"],
                alpha=HORIZONTAL_GRID_LINES["alpha"],
                zorder=HORIZONTAL_GRID_LINES["zorder"],
            )

    plt.ylim(0, 100)
    plt.ylabel("RSI")
    plt.xlim(axis_start, axis_end)

    ax = plt.gca()
    ax.xaxis.set_major_locator(
        mdates.MonthLocator(bymonth=AXIS_TICK_MONTHS, bymonthday=AXIS_TICK_DAY)
    )
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_month_year_label))
    ax.tick_params(axis="x", which="major", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.xlabel("Date")

    # plot RSI (original color-by-months-to-halving logic)
    norm = plt.Normalize(0, 50)

    for i in numpy.arange(1, len(data_frame)):
        x_values = data_frame.index[i - 1:i + 1]
        y_values = data_frame["RSI"][i - 1:i + 1]

        months_left = None

        for halving_date in halving_dates:
            if x_values[1] < halving_date:
                diff = rel.relativedelta(halving_date, x_values[0])
                months_left = diff.years * 12 + diff.months
                break

        cmap = colors.LinearSegmentedColormap.from_list("my_cmap", ["lightgreen", "red"])
        color_value = cmap(norm(months_left))

        plt.plot(x_values, y_values, color=color_value)

    plt.title(
        f"Monthly RSI vs Next Halving "
        f"({axis_start.date()} → {axis_end.date()}; "
        f"{CYCLE_PROGRESS_LINES['interval_percent']}% markers + "
        f"halving lines + horizontal grids)"
    )

    print(
        f"\nDrawing Monthly RSI vs Next Halving "
        f"({axis_start.date()} → {axis_end.date()}, "
        f"{CYCLE_PROGRESS_LINES['interval_percent']}% cycle markers + "
        f"halving lines + horizontal grids).."
    )

    plt.tight_layout()
    plt.show(block=block_window)


if __name__ == "__main__":
    draw(True)
