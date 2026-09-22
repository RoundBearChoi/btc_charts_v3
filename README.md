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
