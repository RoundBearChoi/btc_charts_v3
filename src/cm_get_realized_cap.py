#!/usr/bin/env python3
"""Download Coin Metrics community BTC data and cache realized cap.

Source (public, updated daily):
    https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv

Computes:
    realized_cap = CapMrktCurUSD / CapMVRVCur

On disk:
    src/cg_data/BTC_realized_cap_daily.csv
        time, realized_cap, CapMrktCurUSD, CapMVRVCur, PriceUSD

Re-run anytime to refresh. No API key needed.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# --- config ---
CM_BTC_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
REQUEST_TIMEOUT = 60

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cg_data"
RCAP_CACHE = DATA_DIR / "BTC_realized_cap_daily.csv"

KEEP_COLS = ["realized_cap", "CapMrktCurUSD", "CapMVRVCur", "PriceUSD"]


def fetch_coinmetrics_btc() -> pd.DataFrame:
    """Download the full community CSV and compute realized_cap."""
    print(f"Downloading {CM_BTC_URL}")
    resp = requests.get(CM_BTC_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    cm = pd.read_csv(StringIO(resp.text), parse_dates=["time"])
    cm = cm.set_index("time").sort_index()
    cm.index = cm.index.floor("D")
    cm.index.name = "time"

    # realized_cap only where both inputs exist and MVRV is positive
    mask = (
        cm["CapMrktCurUSD"].notna()
        & cm["CapMVRVCur"].notna()
        & (cm["CapMVRVCur"] > 0)
    )
    out = cm.loc[mask].copy()
    out["realized_cap"] = out["CapMrktCurUSD"] / out["CapMVRVCur"]

    # keep only the columns we care about
    available = [c for c in KEEP_COLS if c in out.columns]
    out = out[available]

    return out


def save_realized_cap(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RCAP_CACHE)
    return RCAP_CACHE


def load_realized_cap() -> pd.DataFrame:
    """Load the local cache (empty DataFrame if missing)."""
    if not RCAP_CACHE.exists():
        return pd.DataFrame()
    df = pd.read_csv(RCAP_CACHE, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).floor("D")
    df.index.name = "time"
    return df.sort_index()


def get_realized_cap(force_refresh: bool = True) -> pd.DataFrame:
    """
    Main entry point.

    force_refresh=True  → always re-download (recommended for latest data)
    force_refresh=False → use cache if it exists
    """
    if not force_refresh and RCAP_CACHE.exists():
        df = load_realized_cap()
        print(f"Using existing cache: {df.index.min().date()} → {df.index.max().date()} ({len(df)} rows)")
        return df

    df = fetch_coinmetrics_btc()
    path = save_realized_cap(df)
    print(f"Saved {len(df)} rows → {path}")
    print(f"Range: {df.index.min().date()} → {df.index.max().date()}")
    if not df.empty:
        latest = df.iloc[-1]
        print(f"Latest realized_cap: ${latest['realized_cap']/1e12:.3f}T")
        if "CapMVRVCur" in latest:
            print(f"Latest MVRV: {latest['CapMVRVCur']:.3f}")
    return df


def main() -> None:
    get_realized_cap(force_refresh=True)


if __name__ == "__main__":
    main()
