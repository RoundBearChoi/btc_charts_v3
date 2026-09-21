"""CoinGecko Analyst daily fetcher.

File: src/analyst_get_daily_data.py

Daily sibling of src/analyst_get_hourly_data.py.
Same daily schema and cache path as src/demo_get_daily_data.py.
This script never writes hourly CSVs.

Coin pick comes from src/coin_menu.py + src/coins.csv.

On disk (daily only):
    src/cg_data/{SYMBOL}_data_daily.csv
        time, open, high, low, close, volumeto

Source is /coins/{id}/market_chart/range?interval=daily
(Analyst / Pro host). That endpoint returns one UTC-midnight
price + sliding 24h volume per day — the same volume meaning
as the hourly scripts.

OHLC from that daily print:
    close    = CoinGecko daily price at 00:00 UTC
    high     = close
    low      = close
    open     = previous day's close (first row uses close)
    volumeto = CoinGecko sliding 24h volume at that midnight

That matches the column layout demo_get_daily_data.py writes.

Rules:
    - No cache -> seed HISTORY_START through latest UTC midnight
      in DAYS_PER_BATCH windows.
    - Cache exists -> fill last cached UTC midnight through latest
      UTC midnight. No backfill behind the earliest row.
    - No stale-gap error: a 5-year-old daily file is just a long
      increment.
    - Each successful chunk is saved so a mid-seed rerun can resume.

Env:
    COINGECKO_ANALYST_API_KEY   Analyst / Pro API key only.
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
DAYS_PER_BATCH = 180          # range window per request; raise to cut call count
HISTORY_START = datetime(2013, 1, 1, tzinfo=timezone.utc)
MIN_FETCH_DAYS = 2            # tiny gaps still request at least this much
OVERLAP_DAYS = 1              # overlap into cached tail / between chunks
REQUEST_PAUSE_SEC = 0.25

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "cg_data"

ANALYST_HOST = "https://pro-api.coingecko.com/api/v3"
ANALYST_KEY_HEADER = "x-cg-pro-api-key"
ANALYST_KEY_ENV = "COINGECKO_ANALYST_API_KEY"

DEFAULT_SYMBOL = "BTC"
GECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XMR": "monero",
    "FARTCOIN": "fartcoin",
    "TROLL": "troll-2",
}

DAILY_COLS = ["open", "high", "low", "close", "volumeto"]
OVERLAP = timedelta(days=OVERLAP_DAYS)


def validate_config() -> None:
    if DAYS_PER_BATCH < 1:
        raise ValueError(f"DAYS_PER_BATCH must be >= 1, got {DAYS_PER_BATCH}")
    if MIN_FETCH_DAYS < 1:
        raise ValueError(f"MIN_FETCH_DAYS must be >= 1, got {MIN_FETCH_DAYS}")
    if OVERLAP_DAYS < 0:
        raise ValueError(f"OVERLAP_DAYS must be >= 0, got {OVERLAP_DAYS}")
    start = HISTORY_START
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if start >= datetime.now(timezone.utc):
        raise ValueError(f"HISTORY_START must be in the past, got {HISTORY_START}")


def latest_utc_midnight(now: datetime | None = None) -> datetime:
    """Most recent 00:00 UTC. Closed daily prints run through this instant."""
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


def daily_cache_path(symbol: str = DEFAULT_SYMBOL) -> Path:
    return DATA_DIR / f"{symbol.strip().upper()}_data_daily.csv"


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


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _floor_utc_days(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to naive UTC midnight so demo/analyst rows line up."""
    out = _naive_utc_index(df)
    if out.empty:
        return out
    out.index = pd.to_datetime(out.index).floor("D")
    out.index.name = "time"
    return out[~out.index.duplicated(keep="last")].sort_index()


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_COLS)


def _read_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        return _empty_daily()
    df = _floor_utc_days(df)
    for col in DAILY_COLS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[DAILY_COLS]


def load_daily(symbol: str = DEFAULT_SYMBOL) -> pd.DataFrame:
    path = daily_cache_path(symbol)
    if path.exists():
        return _read_daily_csv(path)
    return _empty_daily()


def _recompute_open(df: pd.DataFrame) -> pd.DataFrame:
    """Open = previous close across the whole series after a merge."""
    out = df.copy()
    if out.empty or "close" not in out.columns:
        return out
    out["open"] = out["close"].shift(1)
    out["open"] = out["open"].fillna(out["close"])
    return out


def save_daily(df: pd.DataFrame, symbol: str = DEFAULT_SYMBOL) -> Path:
    """Same path + columns as demo_get_daily_data.save_daily."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = daily_cache_path(symbol)
    out = _floor_utc_days(df)
    out = _recompute_open(out)
    out[DAILY_COLS].to_csv(path)
    return path


def _series_from_pairs(pairs: list) -> pd.Series:
    if not pairs:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime([int(ts) for ts, _ in pairs], unit="ms", utc=True)
    values = [float(val) for _, val in pairs]
    series = pd.Series(values, index=idx, dtype="float64")
    return series[~series.index.duplicated(keep="last")].sort_index()


def _request_json(path: str, params: dict, retries: int = 3):
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


def verify_analyst_key() -> None:
    """Fail fast if the env key is missing or rejected by the Pro host."""
    analyst_api_key()
    try:
        _request_json("/key", {})
    except RuntimeError as exc:
        if "auth failed" in str(exc).lower():
            raise
        _request_json(
            "/coins/bitcoin/market_chart/range",
            {
                "vs_currency": "usd",
                "from": int((latest_utc_midnight() - timedelta(days=3)).timestamp()),
                "to": int(latest_utc_midnight().timestamp()),
                "interval": "daily",
            },
        )


def _daily_from_chart(payload: dict) -> pd.DataFrame:
    prices = _series_from_pairs(payload.get("prices") or [])
    volumes = _series_from_pairs(payload.get("total_volumes") or [])
    if prices.empty:
        return _empty_daily()
    frame = pd.DataFrame({"close": prices})
    frame["volumeto"] = volumes.reindex(frame.index)
    frame["high"] = frame["close"]
    frame["low"] = frame["close"]
    frame["open"] = frame["close"]
    frame = _floor_utc_days(frame)
    return _recompute_open(frame)[DAILY_COLS]


def fetch_daily_range(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return _empty_daily()
    coin_id = gecko_id_for(symbol)
    payload = _request_json(
        f"/coins/{coin_id}/market_chart/range",
        {
            "vs_currency": "usd",
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
            "interval": "daily",
        },
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected CoinGecko payload for {symbol}: {type(payload)}")
    return _daily_from_chart(payload)


def _chunk_windows(start: datetime, end: datetime) -> list:
    start = _as_utc(start)
    end = _as_utc(end)
    if end <= start:
        return []
    windows = []
    cursor = start
    chunk = timedelta(days=DAYS_PER_BATCH)
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        windows.append((cursor, chunk_end))
        if chunk_end >= end:
            break
        nxt = chunk_end - OVERLAP
        cursor = nxt if nxt > cursor else chunk_end
    return windows


def fetch_daily_chunks(symbol: str, start: datetime, end: datetime, cached=None):
    combined = cached.copy() if cached is not None and not cached.empty else _empty_daily()
    windows = _chunk_windows(start, end)
    total = len(windows)
    for i, (win_start, win_end) in enumerate(windows, start=1):
        print(f"  chunk {i}/{total}  {win_start.strftime('%Y-%m-%d')} → {win_end.strftime('%Y-%m-%d')} UTC")
        fresh = fetch_daily_range(symbol, win_start, win_end)
        if not fresh.empty:
            if combined.empty:
                combined = fresh
            else:
                combined = pd.concat([combined, fresh])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined = _recompute_open(_floor_utc_days(combined))
            save_daily(combined, symbol)
        elif combined.empty:
            print("  (no rows in this window)")
        else:
            print("  (no new rows in this window)")
        if i < total and REQUEST_PAUSE_SEC > 0:
            time.sleep(REQUEST_PAUSE_SEC)
    return combined


def _history_start() -> datetime:
    start = HISTORY_START
    if start.tzinfo is None:
        return start.replace(tzinfo=timezone.utc)
    return start.astimezone(timezone.utc)


def _plan_window(cached):
    utc0 = latest_utc_midnight()
    if cached.empty:
        return "seed", _history_start(), utc0
    latest = cached.index.max()
    if not isinstance(latest, datetime):
        latest = pd.Timestamp(latest).to_pydatetime()
    latest_utc = _as_utc(latest).replace(minute=0, second=0, microsecond=0, hour=0)
    gap_days = (utc0 - latest_utc).total_seconds() / 86400
    if gap_days <= 0:
        return "current", utc0, utc0
    start = latest_utc - OVERLAP
    min_start = utc0 - timedelta(days=MIN_FETCH_DAYS)
    if start > min_start:
        start = min_start
    return "increment", start, utc0


def get_daily_data(symbol: str = DEFAULT_SYMBOL):
    validate_config()
    analyst_api_key()
    symbol = symbol.strip().upper()
    cached = load_daily(symbol)
    action, start, end = _plan_window(cached)
    if action == "current":
        print(f"{symbol} daily cache is current through {end.strftime('%Y-%m-%d %H:%M UTC')} ({len(cached)} daily rows).")
        combined = cached
    else:
        if action == "seed":
            print(f"No {symbol} daily cache. Seeding full history {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} UTC in {DAYS_PER_BATCH}-day batches.")
        else:
            print(f"{symbol} daily fill {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} UTC.")
        combined = fetch_daily_chunks(symbol, start, end, cached)
        if combined.empty:
            raise RuntimeError(f"CoinGecko returned no daily prices for {symbol}")
    daily_path = save_daily(combined, symbol)
    print(f"Saved {len(combined)} daily rows → {daily_path}\nDaily range: {combined.index.min().date()} → {combined.index.max().date()}")
    return combined


def get_daily(symbol: str = DEFAULT_SYMBOL):
    return get_daily_data(symbol)


def main() -> None:
    validate_config()
    try:
        verify_analyst_key()
    except Exception as exc:
        raise SystemExit(f"✘ Analyst key error: {exc}") from exc
    choices = get_coin_choice("Analyst daily data - Coin Selection")
    total = len(choices)
    for i, (name, symbol) in enumerate(choices, start=1):
        print(f"\n[{i}/{total}] {name} ({symbol})")
        try:
            daily = get_daily_data(symbol)
            print()
            print(daily.tail())
        except Exception as exc:
            print(f"✘ {name} ({symbol}) failed: {exc}")
            continue


if __name__ == "__main__":
    main()
