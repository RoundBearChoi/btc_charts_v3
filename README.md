# btc_charts_v3

## Install

Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

Interactive matplotlib windows need a GUI backend. On Debian / Ubuntu:

```bash
sudo apt install python3-tk
```

`resistance_and_support.py` tries `TkAgg`. If the backend is non-interactive (`Agg`, CI, SSH with no display), it writes `{symbol}_resistance_and_support.png` instead of opening a window.

## API keys

Each family reads only its own env var. There is no fallback to `COINGECKO_API_KEY`.

| Scripts | Env var | Host | Header |
| --- | --- | --- | --- |
| `analyst_get_*` | `COINGECKO_ANALYST_API_KEY` | `pro-api.coingecko.com` | `x-cg-pro-api-key` |
| `demo_get_*` | `COINGECKO_DEMO_API_KEY` | `api.coingecko.com` | `x-cg-demo-api-key` |

Export in `~/.bashrc`, then `source ~/.bashrc` or open a new shell:

```bash
export COINGECKO_ANALYST_API_KEY="..."
export COINGECKO_DEMO_API_KEY="..."
```

Get keys from [CoinGecko API](https://www.coingecko.com/en/api). Demo is enough for short increments. Analyst / Pro is required for a first long hourly seed (~4 years) and for a first full daily history.

## Coins

The numbered prompt comes from `src/coins.csv` (`name,symbol`). Last option is `ALL`.

Fetchers also need a CoinGecko id in the `GECKO_IDS` map inside each fetcher. Current built-in map:

| Symbol | CoinGecko id |
| --- | --- |
| BTC | `bitcoin` |
| ETH | `ethereum` |
| SOL | `solana` |
| XMR | `monero` |
| FARTCOIN | `fartcoin` |
| TROLL | `troll-2` |

Adding a coin means both a `coins.csv` row and a `GECKO_IDS` entry in every fetcher you plan to use.

## Cache files

Written under `src/cg_data/` (gitignored).

| File | Columns | Written by |
| --- | --- | --- |
| `{SYMBOL}_data_hourly.csv` | `time, price, volume` | `*_get_hourly_data.py` |
| `{SYMBOL}_data_daily.csv` | `time, open, high, low, close, volumeto` | `*_get_daily_data.py` |

Notes:

- Hourly `volume` and daily `volumeto` are CoinGecko's sliding 24h sum, not session volume.
- Daily files from the fetchers use the midnight `market_chart` print: `close` is that print, `high`/`low` equal `close`, and `open` is the previous day's close.
- `resistance_and_support.py` does **not** use those daily files. It resamples hourly snapshots to UTC daily OHLC so swing highs/lows have an intra-day range.
- Hourly fetchers still recognize a legacy `{SYMBOL}_data.csv` and rewrite it as `{SYMBOL}_data_hourly.csv`.
- Chunked Analyst (and Demo daily) fetches save after each successful window, so a mid-seed rerun can resume.

## Fetchers

Run from `src/` so `coin_menu` imports resolve:

```bash
cd src
```

### Analyst / Pro — long seed

```bash
python analyst_get_hourly_data.py
python analyst_get_daily_data.py
```

| | Hourly Analyst | Daily Analyst |
| --- | --- | --- |
| Empty cache | seed last **1440 days** (~4 years) to the last UTC hour | seed from **2013-01-01** to latest UTC midnight |
| Fresh cache | increment the missing tail | increment the missing tail |
| Stale gap | error if the hole is **> 1440 days** (raise `LOOKBACK_DAYS` or delete the CSV) | no stale-gap error; a years-old file is just a long increment |
| Request size | `CHUNK_DAYS = 90` (`interval=hourly` max is 100) | `DAYS_PER_BATCH = 180` |
| Does not write | daily CSVs | hourly CSVs |

### Demo — short increment

```bash
python demo_get_hourly_data.py
python demo_get_daily_data.py
```

| | Hourly Demo | Daily Demo |
| --- | --- | --- |
| Empty cache | seed last **60 days** | seed last **60 days** |
| Fresh cache | increment the missing tail | increment the missing tail |
| Stale gap | error if the hole is **> 60 days** — run the Analyst hourly script first | error if the hole is **> 60 days** — run the Analyst daily script first |
| Limit | Demo auto-granularity turns daily above 90 days; keep `LOOKBACK_DAYS <= 90` | same 60-day cap |

Demo never backfills behind the earliest existing row. Use Analyst once to build history, then Demo (or Analyst again) to keep the tail current.

## Charts (cache only)

```bash
python rsi_vs_halving.py
python resistance_and_support.py
```

### `rsi_vs_halving.py`

- Reads only `src/cg_data/BTC_data_daily.csv`.
- Seed that file with `analyst_get_daily_data.py` first. Demo daily is only a 60-day increment and is not enough for the 2013+ axis.
- Resamples daily closes to month-end, computes RSI, colors segments by months until the next date in `HALVING_DATES`.
- Halving dates, cycle-progress %, and RSI grid lines are constants at the top of the file.

### `resistance_and_support.py`

- Reads `src/cg_data/{SYMBOL}_data_hourly.csv` for the coin(s) you pick.
- Seed those files with `analyst_get_hourly_data.py` first (~4 years). Demo hourly is only ~60 days, which is thin for SMA200 and swing structure.
- Resamples hourly snapshots → UTC daily OHLC, then plots close + EMA21 / SMA50 / SMA200, confirmed swing highs/lows, an MA role table, and RSI(14).
- Role rule: price above the MA → support; price below → resistance.
- A swing high is a bar whose high is the max of `SWING_LEFT` bars before and `SWING_RIGHT` bars after (same idea inverted for lows). The last `SWING_RIGHT` bars are shaded as unconfirmed.
- `DAYS_BACK = None` uses the full cache. Set it to an int to clip the window.

## Typical workflow

1. Export both API keys.
2. `cd src`
3. Seed history with Analyst (`hourly` and/or `daily`).
4. Later, refresh the tail with Demo or Analyst again.
5. Run the chart that matches the cache you seeded.

## Related

- [btc_charts](https://github.com/RoundBearChoi/btc_charts)
- [btc_charts_v2](https://github.com/RoundBearChoi/btc_charts_v2)
