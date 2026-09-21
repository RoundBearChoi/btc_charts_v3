# btc_charts_v3

CoinGecko hourly and daily price fetchers.

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
```

Caches land in `src/cg_data/` and are gitignored.
