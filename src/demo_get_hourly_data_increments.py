"""Deprecated wrapper. Use the split Demo scripts instead.

File: src_v3/demo_get_hourly_data_increments.py

This used to write both hourly and daily CSVs. Rebuilding daily from
hourly would wipe a long Analyst daily tail, so this file no longer
touches daily data.

    python demo_get_hourly_data.py
    python demo_get_daily_data.py
"""

from __future__ import annotations

from demo_get_hourly_data import (
    CacheTooStaleError,
    DEFAULT_SYMBOL,
    LOOKBACK_DAYS,
    demo_api_key,
    get_hourly,
    get_hourly_data,
    hourly_cache_path,
    load_hourly,
    main as _hourly_main,
    save_hourly,
    validate_config,
)


def get_hourly_data_increments(symbol: str = DEFAULT_SYMBOL):
    """Hourly-only. Daily updates live in demo_get_daily_data.py."""
    print(
        "demo_get_hourly_data_increments.py is deprecated.\n"
        "  hourly -> demo_get_hourly_data.py\n"
        "  daily  -> demo_get_daily_data.py\n"
        "This wrapper only updates the hourly cache."
    )
    return get_hourly_data(symbol)


def get_increments(symbol: str = DEFAULT_SYMBOL):
    return get_hourly_data_increments(symbol)


def main() -> None:
    print(
        "demo_get_hourly_data_increments.py is deprecated.\n"
        "  hourly -> demo_get_hourly_data.py\n"
        "  daily  -> demo_get_daily_data.py\n"
        "Running the hourly updater only.\n"
    )
    _hourly_main()


if __name__ == "__main__":
    main()
