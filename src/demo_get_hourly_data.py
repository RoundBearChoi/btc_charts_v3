"""CoinGecko Demo hourly increment fetcher.

File: src/demo_get_hourly_data.py

Hourly-only sibling of src/demo_get_daily_data.py.
Does not read or write daily CSVs.

Coin pick comes from src/coin_menu.py + src/coins.csv.

On disk:
    src/cg_data/{SYMBOL}_data_hourly.csv
        hourly snapshots: time, price, volume

volume is CoinGecko's sliding 24h sum, not session volume.

Rules:
    - No cache -> seed latest UTC midnight back LOOKBACK_DAYS.
    - Cache gap <= LOOKBACK_DAYS -> fetch the missing tail and merge.
    - Cache gap  > LOOKBACK_DAYS -> error (run analyst_get_hourly_data.py).
    - Existing short caches are not backfilled behind the earliest row.
    - Demo auto-granularity is hourly only inside 2-90 days, so
      LOOKBACK_DAYS must stay <= 90.

Env:
    COINGECKO_DEMO_API_KEY   Demo API key only (export in ~/.bashrc).
    This file never reads COINGECKO_API_KEY or a Pro/Analyst key.
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
LOOKBACK_DAYS = 60
MIN_FETCH_DAYS = 2
OVERLAP_HOURS = 2
REQUEST_PAUSE_SEC = 0.35
DEMO_HOURLY_MAX_DAYS = 90

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cg_data"

DEMO_HOST = "https://api.coingecko.com/api/v3"
DEMO_KEY_HEADER = "x-cg-demo-api-key"
DEMO_KEY_ENV = "COINGECKO_DEMO_API_KEY"

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
    """Existing cache is more than LOOKBACK_DAYS behind UTC midnight."""


def validate_config() -> None:
    if LOOKBACK_DAYS < 1:
        raise ValueError(f"LOOKBACK_DAYS must be >= 1, got {LOOKBACK_DAYS}")
    if LOOKBACK_DAYS > DEMO_HOURLY_MAX_DAYS:
        raise ValueError(
            f"LOOKBACK_DAYS is {LOOKBACK_DAYS}. Demo auto-granularity turns "
            f"daily above {DEMO_HOURLY_MAX_DAYS} days. Keep LOOKBACK_DAYS "
            f"<= {DEMO_HOURLY_MAX_DAYS} or use analyst_get_hourly_data.py."
        )
    if MIN_FETCH_DAYS < 1:
        raise ValueError(f"MIN_FETCH_DAYS must be >= 1, got {MIN_FETCH_DAYS}")
    if OVERLAP_HOURS < 0:
        raise ValueError(f"OVERLAP_HOURS must be >= 0, got {OVERLAP_HOURS}")


def latest_utc_midnight(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current.replace(minute=0, second=0, microsecond=0, hour=0)


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


def demo_api_key() -> str:
    value = os.getenv(DEMO_KEY_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"No Demo API key found. Export {DEMO_KEY_ENV} in ~/.bashrc "
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


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


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
    out = out[~out.index.duplicated(keep="last")]
    out[["price", "volume"]].to_csv(path)
    legacy = _legacy_hourly_path(symbol)
    if legacy.exists() and legacy != path:
        legacy.unlink()
        print(f"Removed legacy file {legacy.name}.")
    return path


def _series_from_pairs(pairs: list) -> pd.Series:
    if not pairs:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime([int(ts) for ts, _ in pairs], unit="ms", utc=True)
    values = [float(val) for _, val in pairs]
    series = pd.Series(values, index=idx, dtype="float64")
    return series[~series.index.duplicated(keep="last")].sort_index()


def _request_json(path: str, params: dict, retries: int = 3) -> dict:
    headers = {DEMO_KEY_HEADER: demo_api_key()}
    url = f"{DEMO_HOST}{path}"
    last_error = None
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
                    f"Check {DEMO_KEY_ENV} (Demo key + {DEMO_KEY_HEADER})."
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
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return pd.DataFrame(columns=["price", "volume"])
    span_days = (end - start).total_seconds() / 86400
    if span_days > LOOKBACK_DAYS:
        raise CacheTooStaleError(
            f"Fetch window is {span_days:.1f} days. This script only covers {LOOKBACK_DAYS} days."
        )
    if span_days > DEMO_HOURLY_MAX_DAYS:
        raise CacheTooStaleError(
            f"Fetch window is {span_days:.1f} days. Demo auto-granularity turns daily above {DEMO_HOURLY_MAX_DAYS} days."
        )
    coin_id = gecko_id_for(symbol)
    payload = _request_json(
        f"/coins/{coin_id}/market_chart/range",
        {"vs_currency": "usd", "from": int(start.timestamp()), "to": int(end.timestamp())},
    )
    prices = _series_from_pairs(payload.get("prices") or [])
    volumes = _series_from_pairs(payload.get("total_volumes") or [])
    if prices.empty:
        return pd.DataFrame(columns=["price", "volume"])
    frame = pd.DataFrame({"price": prices})
    frame["volume"] = volumes.reindex(frame.index)
    return _naive_utc_index(frame)


def _plan_window(cached: pd.DataFrame):
    utc0 = latest_utc_midnight()
    seed_start = utc0 - timedelta(days=LOOKBACK_DAYS)
    if cached.empty:
        return "seed", seed_start, utc0
    latest = cached.index.max()
    if not isinstance(latest, datetime):
        latest = pd.Timestamp(latest).to_pydatetime()
    latest_utc = _as_utc(latest)
    gap_days = (utc0 - latest_utc).total_seconds() / 86400
    if gap_days <= 0:
        return "current", utc0, utc0
    if gap_days > LOOKBACK_DAYS:
        raise CacheTooStaleError(
            f"Hourly cache is {gap_days:.1f} days behind "
            f"(latest {latest_utc.strftime('%Y-%m-%d %H:%M UTC')}, need through {utc0.strftime('%Y-%m-%d %H:%M UTC')}). "
            f"demo_get_hourly_data.py only fills up to {LOOKBACK_DAYS} days. "
            "Run analyst_get_hourly_data.py first."
        )
    start = latest_utc - OVERLAP
    min_start = utc0 - timedelta(days=MIN_FETCH_DAYS)
    if start > min_start:
        start = min_start
    return "increment", start, utc0


def get_hourly_data(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    validate_config()
    demo_api_key()
    symbol = symbol.strip().upper()
    cached = load_hourly(symbol)
    action, start, end = _plan_window(cached)
    if action == "current":
        print(f"{symbol} hourly cache is current through {end.strftime('%Y-%m-%d %H:%M UTC')} ({len(cached)} hourly rows).")
        combined = cached
    else:
        if action == "seed":
            print(f"No {symbol} hourly cache. Seeding {LOOKBACK_DAYS} days {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} UTC.")
        else:
            print(f"{symbol} hourly fill {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} UTC.")
        fresh = fetch_hourly_range(symbol, start, end)
        if fresh.empty:
            if cached.empty:
                raise RuntimeError(f"CoinGecko returned no hourly prices for {symbol}")
            print("No new hourly rows; using existing cache.")
            combined = cached
        elif cached.empty:
            combined = fresh
        else:
            combined = pd.concat([cached, fresh])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    hourly_path = save_hourly(combined, symbol)
    print(f"Saved {len(combined)} hourly rows → {hourly_path}\nHourly range: {combined.index.min()} → {combined.index.max()}")
    return combined


def get_hourly(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    return get_hourly_data(symbol)


def main() -> None:
    validate_config()
    demo_api_key()
    choices = get_coin_choice("Demo hourly data - Coin Selection")
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
