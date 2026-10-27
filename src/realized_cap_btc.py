#!/usr/bin/env python3
"""BTC Realized Cap vs Price chart.

Realized cap (top) and price (bottom).
Uses local cache produced by src/cm_get_realized_cap.py
(stored in src/cm_data/BTC_realized_cap_daily.csv)
and daily price cache from CoinGecko (src/cg_data/).

Config at top controls the lookback window.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

from analyst_get_daily_data import daily_cache_path, load_daily
from cm_get_realized_cap import load_realized_cap

# --- config ---
DAYS_BACK = 8 * 360          # lookback window in days; None = full history
HEIGHT_RATIOS = (2.2, 1)     # realized cap panel : price panel
FIGURE_SIZE = (14, 8)
RCAP_COLOR = "#1f77b4"
PRICE_COLOR = "#ff7f0e"
PRICE_WIDTH = 1.0


def _naive_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force a naive UTC-midnight DatetimeIndex so joins don't fail on tz."""
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx.floor("D")
    out.index.name = "time"
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_price() -> pd.DataFrame:
    path = daily_cache_path("BTC")
    if not path.exists():
        raise FileNotFoundError(
            f"No BTC daily price cache at {path}. "
            "Run analyst_get_daily_data.py or demo_get_daily_data.py first."
        )
    return _naive_day_index(load_daily("BTC"))


def plot(df: pd.DataFrame) -> None:
    fig, (ax_rcap, ax_price) = plt.subplots(
        2, 1,
        figsize=FIGURE_SIZE,
        gridspec_kw={"height_ratios": list(HEIGHT_RATIOS)},
        sharex=True,
    )

    ax_rcap.plot(df.index, df["realized_cap"], color=RCAP_COLOR, linewidth=1.4)
    ax_rcap.set_ylabel("Realized Cap (USD)")
    ax_rcap.set_title("BTC Realized Cap")
    ax_rcap.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: f"${x/1e12:.2f}T")
    )
    ax_rcap.grid(True, alpha=0.3)

    ax_price.plot(df.index, df["close"], color=PRICE_COLOR, linewidth=PRICE_WIDTH)
    ax_price.set_ylabel("Price (USD)")
    ax_price.set_title("BTC Price")
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: f"${x:,.0f}")
    )
    ax_price.grid(True, alpha=0.3)

    ax_price.xaxis.set_major_locator(mdates.YearLocator())
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_price.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def main() -> None:
    rcap = _naive_day_index(load_realized_cap())
    if rcap.empty:
        raise SystemExit(
            "No realized cap cache. Run src/cm_get_realized_cap.py first."
        )

    price = load_price()
    if price.empty:
        raise SystemExit("No BTC price data.")

    df = rcap.join(price[["close"]], how="inner").sort_index()
    if df.empty:
        raise SystemExit("No overlapping dates between realized cap and price.")

    if DAYS_BACK is not None:
        df = df.iloc[-DAYS_BACK:]

    print(f"Plotting {len(df)} days: {df.index.min().date()} → {df.index.max().date()}")
    plot(df)


if __name__ == "__main__":
    main()
