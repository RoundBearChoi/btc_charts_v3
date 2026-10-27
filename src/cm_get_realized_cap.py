#!/usr/bin/env python3
"""Download Coin Metrics community BTC data and cache realized cap.

Source (free community API, no key):
    https://community-api.coinmetrics.io/v4

Computes:
    realized_cap = CapMrktCurUSD / CapMVRVCur

On disk:
    src/cm_data/BTC_realized_cap_daily.csv
        time, realized_cap, CapMrktCurUSD, CapMVRVCur, PriceUSD

Re-run anytime to refresh. No API key needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

# --- config ---
CM_BASE = "https://community-api.coinmetrics.io/v4"
METRICS = "CapMrktCurUSD,CapMVRVCur,PriceUSD"
PAGE_SIZE = 10000
REQUEST_TIMEOUT = 60

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cm_data"
RCAP_CACHE = DATA_DIR / "BTC_realized_cap_daily.csv"

KEEP_COLS = ["realized_cap", "CapMrktCurUSD", "CapMVRVCur", "PriceUSD"]


def _fetch_page(next_page_token: str | None = None) -> dict:
    params = {
        "assets": "btc",
        "metrics": METRICS,
        "frequency": "1d",
        "page_size": PAGE_SIZE,
    }
    if next_page_token:
        params["next_page_token"] = next_page_token

    url = f"{CM_BASE}/timeseries/asset-metrics"
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_coinmetrics_btc() -> pd.DataFrame:
    """Paginate the community API and compute realized_cap."""
    print(f"Fetching from {CM_BASE}")
    rows = []
    token = None
    page = 0

    while True:
        page += 1
        payload = _fetch_page(token)
        data = payload.get("data") or []
        rows.extend(data)
        print(f"  page {page}: {len(data)} rows (total {len(rows)})")

        token = payload.get("next_page_token")
        if not token:
            break

    if not rows:
        raise RuntimeError("Coin Metrics community API returned no data")

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"]).dt.floor("D")
    df = df.set_index("time").sort_index()
    df.index.name = "time"

    # drop duplicate days if any
    df = df[~df.index.duplicated(keep="last")]

    for col in ["CapMrktCurUSD", "CapMVRVCur", "PriceUSD"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = (
        df["CapMrktCurUSD"].notna()
        & df["CapMVRVCur"].notna()
        & (df["CapMVRVCur"] > 0)
    )
    out = df.loc[mask].copy()
    out["realized_cap"] = out["CapMrktCurUSD"] / out["CapMVRVCur"]

    available = [c for c in KEEP_COLS if c in out.columns]
    return out[available]


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
