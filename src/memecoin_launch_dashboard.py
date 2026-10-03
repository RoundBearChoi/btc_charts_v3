#!/usr/bin/env python3
"""Memecoin launchpad dashboard, one-year view.

Launch counts from MemeFees only exist from 2026-08-15. Requesting 365 days
does not invent earlier rows. Fees and revenue from DefiLlama, via the same
MemeFees pad-daily feed, do cover a trailing year.

No API key. CC BY 4.0 for MemeFees-produced fields. Credit MemeFees and link
https://memefees.com. DefiLlama figures stay under DefiLlama's terms.
"""

import argparse
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import requests

LAUNCHES_API = "https://memefees.com/api/v1/launches-daily"
FEES_API = "https://memefees.com/api/v1/pad-daily"
NAMES = {
    "pump-fun": "Pump.fun",
    "flap-sh": "Flap.sh",
    "pons": "Pons",
    "bags": "Bags",
    "pools": "Pools",
    "pools-trade": "Pools",
    "virtuals": "Virtuals",
    "letsbonk": "LetsBonk",
    "letscash": "LetsCash",
    "four-meme": "Four.meme",
    "clanker": "Clanker",
    "jupiter-studio": "Jup Studio",
    "meteora-dbc": "Meteora DBC",
    "raydium-launchlab": "Raydium LaunchLab",
    "noxa": "NOXA",
    "believe": "Believe",
    "boop": "Boop",
    "moonshot": "Moonshot",
    "fomo": "fomo",
    "zora": "Zora",
}
COLORS = {
    "Pump.fun": "#4d4d4d",
    "Flap.sh": "#e67e22",
    "Pons": "#8e44ad",
    "Bags": "#3498db",
    "Four.meme": "#e74c3c",
    "Raydium LaunchLab": "#27ae60",
    "LetsBonk": "#9b59b6",
    "fomo": "#f1c40f",
    "Meteora DBC": "#34495e",
    "Clanker": "#2ecc71",
    "Believe": "#ff1493",
    "Virtuals": "#1abc9c",
    "Zora": "#e74c3c",
}


def fetch_rows(url: str):
    print(f"Fetching {url}")
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("rows") or []
    if not rows:
        raise SystemExit(f"No rows from {url}")
    return payload, pd.DataFrame(rows)


def name_pad(pad: str) -> str:
    return NAMES.get(pad, pad)


class LaunchDashboard:
    def __init__(self, days: int, outfile: str):
        self.days = days
        self.outfile = outfile
        self.fees = None
        self.launches = None

    def fetch(self) -> None:
        _, fees = fetch_rows(f"{FEES_API}?days={self.days}")
        fees["day"] = pd.to_datetime(fees["day"])
        fees["platform"] = fees["pad"].map(name_pad)
        fees["fees_usd"] = pd.to_numeric(fees["fees_usd"], errors="coerce")
        self.fees = fees.dropna(subset=["fees_usd"])

        _, launches = fetch_rows(f"{LAUNCHES_API}?days={self.days}")
        launches["day"] = pd.to_datetime(launches["day"])
        launches["platform"] = launches["pad"].map(name_pad)
        launches["launches"] = pd.to_numeric(launches["launches"], errors="coerce").fillna(0)
        launches["complete"] = launches["complete"].astype(bool)
        launches["partial_day"] = launches["partial_day"].astype(bool)
        partial = launches.loc[launches["partial_day"], "day"].max()
        if pd.notna(partial):
            print(f"Dropping in-progress launch day {partial.date()}")
            launches = launches.loc[~launches["partial_day"]].copy()
        self.launches = launches

        print(
            f"Fees {self.fees['day'].min().date()} → {self.fees['day'].max().date()} "
            f"({self.fees['day'].nunique()} days)"
        )
        print(
            f"Launches {self.launches['day'].min().date()} → {self.launches['day'].max().date()} "
            f"({self.launches['day'].nunique()} days)"
        )

    def print_rank(self) -> None:
        latest = self.fees["day"].max()
        window = self.fees[self.fees["day"] >= latest - pd.Timedelta(days=6)]
        totals = window.groupby("platform")["fees_usd"].sum().sort_values(ascending=False)
        print()
        print(f"FEES, LAST 7 CLOSED DAYS ending {latest.date()}   ${totals.sum():,.0f}")
        for i, (name, n) in enumerate(totals.head(12).items(), 1):
            print(f"  {i:2d}. {name:<20} ${n:>14,.0f}")

        latest_l = self.launches["day"].max()
        day = (self.launches[self.launches["day"] == latest_l]
               .groupby("platform")["launches"].sum()
               .sort_values(ascending=False))
        incomplete = set(
            self.launches.loc[
                (self.launches["day"] == latest_l) & ~self.launches["complete"],
                "platform",
            ]
        )
        print()
        print(f"LAUNCHES, LAST CLOSED DAY {latest_l.date()}   {int(day.sum()):,}")
        print("Launch counts do not exist before 2026-08-15 in this feed.")
        for i, (name, n) in enumerate(day.head(12).items(), 1):
            mark = "+" if name in incomplete else ""
            print(f"  {i:2d}. {name:<20} {int(n):>8,}{mark}")
        print("+ is a lower bound.")

    def plot(self) -> None:
        fees = (self.fees.pivot_table(index="day", columns="platform",
                                      values="fees_usd", aggfunc="sum")
                .fillna(0)
                .sort_index())
        order = fees.sum().sort_values(ascending=False).head(8).index.tolist()
        fees = fees[order]
        ma = fees.rolling(7, min_periods=1).mean()
        total = fees.sum(axis=1)
        share = fees.div(fees.sum(axis=1).replace(0, pd.NA), axis=0) * 100
        colors = [COLORS.get(name, "#7f8c8d") for name in order]

        launches = (self.launches.pivot_table(index="day", columns="platform",
                                              values="launches", aggfunc="sum")
                    .fillna(0)
                    .sort_index())
        launch_order = [p for p in order if p in launches.columns]
        launch_order += [p for p in launches.sum().sort_values(ascending=False).index
                         if p not in launch_order][:4]
        launches = launches.reindex(columns=launch_order).fillna(0)
        launch_ma = launches.rolling(7, min_periods=1).mean()

        as_of = self.fees["day"].max().strftime("%b %d, %Y")
        fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=120)

        for name in order:
            axs[0, 0].plot(ma.index, ma[name] / 1e6, label=name, linewidth=2.1,
                           color=COLORS.get(name))
        axs[0, 0].set_title("7-day MA — daily fees")
        axs[0, 0].set_ylabel("USD millions")
        axs[0, 0].legend(fontsize=8, loc="upper left", ncol=2)
        axs[0, 0].grid(True, alpha=0.3)

        axs[0, 1].plot(total.index, total.rolling(7, min_periods=1).mean() / 1e6,
                       color="#6c3483", linewidth=2.6, label="7-day MA")
        axs[0, 1].plot(total.index, total / 1e6, color="#6c3483", alpha=0.25,
                       label="Raw daily total")
        axs[0, 1].set_title("Total daily fees, top pads")
        axs[0, 1].set_ylabel("USD millions")
        axs[0, 1].legend(fontsize=9)
        axs[0, 1].grid(True, alpha=0.3)

        share.plot.area(ax=axs[1, 0], stacked=True, linewidth=0, alpha=0.9,
                        color=colors, legend=False)
        axs[1, 0].set_title("Share of fees")
        axs[1, 0].set_ylabel("% of top pads")
        axs[1, 0].set_ylim(0, 100)
        axs[1, 0].legend(order, fontsize=7, loc="upper left", ncol=2)
        axs[1, 0].grid(True, alpha=0.3)

        for name in launch_order:
            axs[1, 1].plot(launch_ma.index, launch_ma[name], label=name, linewidth=2.1,
                           color=COLORS.get(name))
        axs[1, 1].set_title("7-day MA — launches (only days counted)")
        axs[1, 1].set_ylabel("Avg daily creations")
        axs[1, 1].legend(fontsize=7, loc="upper left", ncol=2)
        axs[1, 1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Launchpad year — fees, plus launches where counted\n{as_of}",
            fontsize=15, fontweight="bold",
        )
        fig.text(
            0.01, 0.01,
            "Fees: DefiLlama via MemeFees pad-daily. Launches: MemeFees, from 2026-08-15, "
            "pump.fun is a lower bound. https://memefees.com",
            fontsize=8, color="#555555",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        fig.savefig(self.outfile, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {self.outfile}")

    def run(self) -> None:
        self.fetch()
        self.print_rank()
        self.plot()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="1..730")
    parser.add_argument("--out", default="memecoin_launch_dashboard.png")
    args = parser.parse_args()
    LaunchDashboard(args.days, args.out).run()
    print(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


if __name__ == "__main__":
    main()
