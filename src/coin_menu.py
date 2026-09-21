"""Shared coins.csv loader and numbered coin prompt.

Used by Demo increment, Analyst backfill, and later chart scripts.
Row order in src/coins.csv is prompt order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_COINS_CSV = Path(__file__).with_name("coins.csv")


def load_coins(csv_path: Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else DEFAULT_COINS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Coin list not found: {path}")

    coins = pd.read_csv(path)
    coins.columns = coins.columns.str.strip().str.lower()

    required = {"name", "symbol"}
    missing = required - set(coins.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    coins["name"] = coins["name"].astype(str).str.strip()
    coins["symbol"] = coins["symbol"].astype(str).str.strip().str.upper()
    coins = coins.dropna(subset=["name", "symbol"])
    coins = coins[(coins["name"] != "") & (coins["symbol"] != "")]

    if coins.empty:
        raise ValueError(f"{path} has no usable name/symbol rows")

    return coins.reset_index(drop=True)


def coins_as_pairs(coins: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    """[(display_name, symbol), ...] in menu order."""
    if coins is None:
        coins = load_coins()
    return [(row["name"], row["symbol"]) for _, row in coins.iterrows()]


def resolve_coin_arg(coin: str, coins: pd.DataFrame | None = None) -> tuple[str, str]:
    """Match a ticker, display name, or 1-based menu index."""
    if coins is None:
        coins = load_coins()

    raw = coin.strip()
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(coins):
            row = coins.iloc[idx - 1]
            return row["name"], row["symbol"]

    key = raw.upper()
    for _, row in coins.iterrows():
        if row["symbol"] == key or row["name"].upper() == key:
            return row["name"], row["symbol"]

    known = ", ".join(
        f"{row['name']} ({row['symbol']})" for _, row in coins.iterrows()
    )
    raise SystemExit(f"Unknown coin '{coin}'. Known: {known}")


def get_coin_choice(
    prompt_title: str = "Coin Selection",
    csv_path: Path | None = None,
) -> list[tuple[str, str]]:
    """Prompt 1..N from coins.csv, plus ALL as the last option.

    ALL means every coin listed above it, in that same order.
    Returns [(name, symbol), ...].
    """
    coins = load_coins(csv_path)
    n = len(coins)
    all_idx = n + 1

    print()
    print("=" * 60)
    print(prompt_title)
    print("=" * 60)
    for i, row in coins.iterrows():
        print(f"{i + 1}) {row['name']}")
    print(f"{all_idx}) ALL")
    print("=" * 60)

    while True:
        raw = input(f"Enter 1-{all_idx} (or ALL): ").strip()
        if raw.lower() == "all" or (raw.isdigit() and int(raw) == all_idx):
            chosen = coins_as_pairs(coins)
            labels = ", ".join(name for name, _ in chosen)
            print(f"→ ALL ({labels})")
            return chosen
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= n:
                row = coins.iloc[idx - 1]
                name, symbol = row["name"], row["symbol"]
                print(f"→ {name} ({symbol})")
                return [(name, symbol)]
        print(f"✘ Invalid. Enter a number from 1 to {all_idx}, or ALL.")
