"""CoinGecko Analyst hourly fetcher.

File: src/analyst_get_hourly_data.py

Longer-range sibling of src/demo_get_hourly_data.py.
Same hourly schema and cache path; this script never writes daily CSVs.

Coin pick comes from src/coin_menu.py + src/coins.csv.

On disk (hourly only):
    src/cg_data/{SYMBOL}_data_hourly.csv
        hourly snapshots: time, price, volume

volume is CoinGecko's sliding 24h sum, not session volume.

Rules:
    - No cache -> seed latest UTC hour back LOOKBACK_DAYS.
    - Cache gap <= LOOKBACK_DAYS -> fetch the missing tail and merge.
    - Cache gap  > LOOKBACK_DAYS -> error (raise LOOKBACK_DAYS or delete CSV).
    - Existing short Demo caches are not backfilled behind the earliest row.
    - Requests are split into CHUNK_DAYS windows with interval=hourly.
    - Each successful chunk is saved so a mid-seed rerun can resume.
    - Fetch end is the last UTC o'clock (HH:00), not UTC midnight.
      Daily scripts still stop at 00:00 UTC.

Env:
    COINGECKO_ANALYST_API_KEY   Analyst / Pro API key only (export in ~/.bashrc).
    This file never reads COINGECKO_DEMO_API_KEY or COINGECKO_API_KEY.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from coin_menu import get_coin_choice

# --- config ---
LOOKBACK_DAYS = 360 * 4  # empty-cache seed; also max allowed increment gap
CHUNK_DAYS = 90          # must stay <= 100 when interval=hourly
MIN_FETCH_DAYS = 2       # tiny gaps still request at least this much
OVERLAP_HOURS = 2        # overlap into cached tail / between chunks
REQUEST_PAUSE_SEC = 0.25

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cg_data"

ANALYST_HOST = "https://pro-api.coingecko.com/api/v3"
ANALYST_KEY_HEADER = "x-cg-pro-api-key"
ANALYST_KEY_ENV = "COINGECKO_ANALYST_API_KEY"
HOURLY_INTERVAL_MAX_DAYS = 100

DEFAULT_SYMBOL = "BTC"
GECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XMR": "monero",
    "FARTCOIN": "fartcoin",
    "TROLL": "troll-2",
}

OVERLAP = timedelta(hours=OVERLAP_HOURS)


class CacheTooStaleError(RuntimeError):
    """Existing cache is more than LOOKBACK_DAYS behind the last UTC hour."""


def validate_config() -> None:
    if LOOKBACK_DAYS < 1:
        raise ValueError(f"LOOKBACK_DAYS must be >= 1, got {LOOKBACK_DAYS}")
    if CHUNK_DAYS < 1:
        raise ValueError(f"CHUNK_DAYS must be >= 1, got {CHUNK_DAYS}")
    if CHUNK_DAYS > HOURLY_INTERVAL_MAX_DAYS:
        raise ValueError(
            f"CHUNK_DAYS is {CHUNK_DAYS}. CoinGecko interval=hourly "
            f"allows at most {HOURLY_INTERVAL_MAX_DAYS} days per request."
        )


def latest_utc_hour(now: datetime | None = None) -> datetime:
    """Most recent UTC o'clock (HH:00) that has already started."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current.replace(minute=0, second=0, microsecond=0)


def gecko_id_for(symbol: str) -> str:
    ticker = symbol.strip().upper()
    if ticker not in GECKO_IDS:
        known = ", ".join(GECKO_IDS)
        raise ValueError(f"Unsupported symbol {ticker!r}. Known: {known}")
    return GECKO_IDS[ticker]


def hourly_cache_path(symbol: str = DEFAULT_SYMBOL) -> Path:
    return DATA_DIR / f"{symbol.strip().upper()}_data_hourly.csv"


def _legacy_hourly_path(symbol: str = DEFAULT_SYMBOL) -> Path:
    return DATA_DIR / f"{symbol.strip().upper()}_data.csv"


def cache_path(symbol: str = DEFAULT_SYMBOL) -> Path:
    """Hourly cache path. Kept as an alias."""
    return hourly_cache_path(symbol)


def analyst_api_key() -> str:
    """Read the Analyst key only. Does not fall back to Demo or generic keys."""
    value = os.getenv(ANALYST_KEY_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"No Analyst API key found. Export {ANALYST_KEY_ENV} in ~/.bashrc "
            "and open a new shell (or run: source ~/.bashrc)."
        )
    return value


def _naive_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out.index.name = "time"
    return out.sort_index()


def _read_hourly_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        return pd.DataFrame(columns=["price", "volume"])
    df = _naive_utc_index(df)
    for col in ("price", "volume"):
        if col not in df.columns:
            df[col] = float("nan")
    return df[["price", "volume"]]


def load_hourly(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    path = hourly_cache_path(symbol)
    if path.exists():
        return _read_hourly_csv(path)
    legacy = _legacy_hourly_path(symbol)
    if legacy.exists():
        print(f"Found legacy hourly cache {legacy.name}; will save as {path.name}.")
        return _read_hourly_csv(legacy)
    return pd.DataFrame(columns=["price", "volume"])


def save_hourly(df: pd.DataFrame, symbol: str = DEFAULT_SYMBOL) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = hourly_cache_path(symbol)
    out = _naive_utc_index(df)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out[["price", "volume"]].to_csv(path)
    legacy = _legacy_hourly_path(symbol)
    if legacy.exists() and legacy != path:
        legacy.unlink()
        print(f"Removed legacy file {legacy.name}.")
    return path


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _series_from_pairs(pairs: list) -> pd.Series:
    if not pairs:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime([int(ts) for ts, _ in pairs], unit="ms", utc=True)
    values = [float(val) for _, val in pairs]
    series = pd.Series(values, index=idx, dtype="float64")
    return series[~series.index.duplicated(keep="last")].sort_index()


def _request_json(path: str, params: dict, retries: int = 3) -> dict:
    headers = {ANALYST_KEY_HEADER: analyst_api_key()}
    url = f"{ANALYST_HOST}{path}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 429:
                wait = 2 ** attempt
                print(f"CoinGecko 429 — sleeping {wait}s")
                time.sleep(wait)
                continue
            if response.status_code in {401, 403}:
                raise RuntimeError(
                    f"CoinGecko auth failed ({response.status_code}). "
                    f"Check {ANALYST_KEY_ENV} (Analyst key + {ANALYST_KEY_HEADER})."
                )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"CoinGecko error: {data['error']}")
            return data
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(f"Request failed ({attempt + 1}/{retries}): {exc}")
            if attempt == retries - 1:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"CoinGecko request failed after {retries} tries: {path}") from last_error


def fetch_hourly_range(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """One Analyst /market_chart/range call with interval=hourly."""
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return pd.DataFrame(columns=["price", "volume"])

    span_days = (end - start).total_seconds() / 86400
    if span_days > HOURLY_INTERVAL_MAX_DAYS:
        raise ValueError(
            f"Fetch window is {span_days:.1f} days. "
            f"interval=hourly allows at most {HOURLY_INTERVAL_MAX_DAYS} days."
        )

    coin_id = gecko_id_for(symbol)
    payload = _request_json(
        f"/coins/{coin_id}/market_chart/range",
        {
            "vs_currency": "usd",
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
            "interval": "hourly",
        },
    )
    prices = _series_from_pairs(payload.get("prices") or [])
    volumes = _series_from_pairs(payload.get("total_volumes") or [])
    if prices.empty:
        return pd.DataFrame(columns=["price", "volume"])

    frame = pd.DataFrame({"price": prices})
    frame["volume"] = volumes.reindex(frame.index)
    return _naive_utc_index(frame)


def _chunk_windows(start: datetime, end: datetime) -> list:
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return []

    windows = []
    cursor = start
    chunk = timedelta(days=CHUNK_DAYS)
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        windows.append((cursor, chunk_end))
        if chunk_end >= end:
            break
        nxt = chunk_end - OVERLAP
        cursor = nxt if nxt > cursor else chunk_end
    return windows


def fetch_hourly_chunks(symbol: str, start: datetime, end: datetime, cached=None):
    combined = cached.copy() if cached is not None and not cached.empty else pd.DataFrame(columns=["price", "volume"])
    windows = _chunk_windows(start, end)
    total = len(windows)
    for i, (win_start, win_end) in enumerate(windows, start=1):
        print(f"  chunk {i}/{total}  {win_start.strftime('%Y-%m-%d %H:%M')} → {win_end.strftime('%Y-%m-%d %H:%M')} UTC")
        fresh = fetch_hourly_range(symbol, win_start, win_end)
        if not fresh.empty:
            if combined.empty:
                combined = fresh
            else:
                combined = pd.concat([combined, fresh])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            save_hourly(combined, symbol)
        elif combined.empty:
            print("  (no rows in this window)")
        else:
            print("  (no new rows in this window)")
        if i < total and REQUEST_PAUSE_SEC > 0:
            time.sleep(REQUEST_PAUSE_SEC)
    return combined


def _plan_window(cached):
    end = latest_utc_hour()
    seed_start = end - timedelta(days=LOOKBACK_DAYS)
    if cached.empty:
        return "seed", seed_start, end
    latest = cached.index.max()
    if not isinstance(latest, datetime):
        latest = pd.Timestamp(latest).to_pydatetime()
    latest_utc = _as_utc(latest)
    gap_days = (end - latest_utc).total_seconds() / 86400
    if gap_days <= 0:
        return "current", end, end
    if gap_days > LOOKBACK_DAYS:
        raise CacheTooStaleError(
            f"Cache is {gap_days:.1f} days behind "
            f"(latest {latest_utc.strftime('%Y-%m-%d %H:%M UTC')}, need through {end.strftime('%Y-%m-%d %H:%M UTC')}). "
            f"analyst_get_hourly_data.py only fills up to {LOOKBACK_DAYS} days. "
            "Raise LOOKBACK_DAYS or delete the hourly CSV to re-seed."
        )
    start = latest_utc - OVERLAP
    min_start = end - timedelta(days=MIN_FETCH_DAYS)
    if start > min_start:
        start = min_start
    return "increment", start, end


def get_hourly_data(symbol: str = DEFAULT_SYMBOL):
    validate_config()
    analyst_api_key()
    symbol = symbol.strip().upper()
    cached = load_hourly(symbol)
    action, start, end = _plan_window(cached)
    if action == "current":
        print(f"{symbol} hourly cache is current through {end.strftime('%Y-%m-%d %H:%M UTC')} ({len(cached)} hourly rows).")
        combined = cached
    else:
        if action == "seed":
            print(f"No {symbol} hourly cache. Seeding {LOOKBACK_DAYS} days {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} UTC.")
        else:
            print(f"{symbol} hourly fill {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} UTC.")
        combined = fetch_hourly_chunks(symbol, start, end, cached)
        if combined.empty:
            raise RuntimeError(f"CoinGecko returned no hourly prices for {symbol}")
    hourly_path = save_hourly(combined, symbol)
    print(f"Saved {len(combined)} hourly rows → {hourly_path}\nHourly range: {combined.index.min()} → {combined.index.max()}")
    return combined


def get_hourly(symbol: str = DEFAULT_SYMBOL):
    return get_hourly_data(symbol)


def main() -> None:
    validate_config()
    analyst_api_key()
    choices = get_coin_choice("Analyst hourly data - Coin Selection")
    total = len(choices)
    for i, (name, symbol) in enumerate(choices, start=1):
        print(f"\n[{i}/{total}] {name} ({symbol})")
        try:
            hourly = get_hourly_data(symbol)
            print()
            print(hourly.tail())
        except Exception as exc:
            print(f"✘ {name} ({symbol}) failed: {exc}")
            continue


if __name__ == "__main__":
    main()
