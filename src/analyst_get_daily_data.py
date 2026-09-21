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
