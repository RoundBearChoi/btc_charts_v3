# btc_charts_v3

CoinGecko hourly and daily price fetchers, plus cache-only charts.

## Install

```bash
python -m pip install -r requirements.txt
```

## API keys

Export in `~/.bashrc` (then `source ~/.bashrc`):

```bash
export COINGECKO_ANALYST_API_KEY="..."
export COINGECKO_DEMO_API_KEY="..."
```

## Run

```bash
cd src

python analyst_get_hourly_data.py
python analyst_get_daily_data.py
python demo_get_hourly_data.py
python demo_get_daily_data.py

python rsi_vs_halving.py
python resistance_and_support.py
```

`rsi_vs_halving.py` only reads `src/cg_data/BTC_data_daily.csv`.
Seed that file with `analyst_get_daily_data.py` first.

`resistance_and_support.py` only reads `src/cg_data/{SYMBOL}_data_hourly.csv`.
Seed those files with `analyst_get_hourly_data.py` first (about four years).
It resamples hourly snapshots to daily OHLC and does not download.

Caches land in `src/cg_data/` and are gitignored.
