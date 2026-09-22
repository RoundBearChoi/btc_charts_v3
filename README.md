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
```

`rsi_vs_halving.py` only reads `src/cg_data/BTC_data_daily.csv`. It does not download.
Seed that file with `analyst_get_daily_data.py` first.

Caches land in `src/cg_data/` and are gitignored.
