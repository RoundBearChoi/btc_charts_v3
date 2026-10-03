#!/usr/bin/env python3
"""Memecoin launchpad dashboard.

Recreates the Feb 2026 LP_stuff chart (memeData.py / Dune query 4010816)
with current public data. No API key.

Source: MemeFees launches-daily, CC BY 4.0.
https://memefees.com/api/v1/launches-daily
Credit MemeFees and link https://memefees.com when citing figures.

Coverage starts 2026-08-15. pump.fun and Bags are platform-feed lower bounds
(complete=false). Factory-event pads (Flap, Pons, Pools) are exact.
Today's UTC day is partial and is dropped from the ranking by default.
"""

import argparse
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import pandas as pd
import requests

API = "https://memefees.com/api/v1/launches-daily"
NAMES = {
    "pump-fun": "Pump.fun",
    "flap-sh": "Flap.sh",
    "pons": "Pons",
    "bags": "Bags",
    "pools": "Pools",
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
}
COLORS = {
    "Pump.fun": "#4d4d4d",
    "Flap.sh": "#e67e22",
    "Pons": "#8e44ad",
    "Bags": "#3498db",
    "Pools": "#16a085",
    "Virtuals": "#f1c40f",
    "LetsBonk": "#9b59b6",
    "LetsCash": "#1abc9c",
    "Four.meme": "#e74c3c",
    "Clanker": "#2ecc71",
    "Jup Studio": "#e67e22",
    "Meteora DBC": "#34495e",
    "Raydium LaunchLab": "#27ae60",
    "NOXA": "#95a5a6",
}


class LaunchDashboard:
    def __init__(self, days: int, outfile: str, keep_partial: bool):
        self.days = days
        self.outfile = outfile
        self.keep_partial = keep_partial
        self.meta = {}
        self.df = None

    def fetch(self) -> pd.DataFrame:
        url = f"{API}?days={self.days}"
        print(f"Fetching {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        self.meta = payload
        rows = payload.get("rows") or []
        if not rows:
            raise SystemExit("No rows returned.")

        df = pd.DataFrame(rows)
        df["day"] = pd.to_datetime(df["day"])
        df["platform"] = df["pad"].map(lambda p: NAMES.get(p, p))
        df["launches"] = pd.to_numeric(df["launches"], errors="coerce").fillna(0)
        df["complete"] = df["complete"].astype(bool)
        df["partial_day"] = df["partial_day"].astype(bool)

        if not self.keep_partial:
            partial = df.loc[df["partial_day"], "day"].max()
            if pd.notna(partial):
                print(f"Dropping in-progress UTC day {partial.date()}")
                df = df.loc[~df["partial_day"]].copy()

        self.df = df.sort_values(["day", "launches"], ascending=[True, False])
        print(f"asOf {payload.get('asOf')}  rows {len(self.df)}  "
              f"{self.df['day'].min().date()} → {self.df['day'].max().date()}")
        return self.df

    def print_rank(self) -> None:
        latest = self.df["day"].max()
        window = self.df[self.df["day"] >= latest - pd.Timedelta(days=6)]
        totals = (window.groupby("platform")["launches"]
                  .sum()
                  .sort_values(ascending=False))
        share = totals / totals.sum() * 100
        day = (self.df[self.df["day"] == latest]
               .groupby("platform")["launches"].sum()
               .sort_values(ascending=False))
        incomplete = set(
            self.df.loc[(self.df["day"] == latest) & ~self.df["complete"], "platform"]
        )

        print()
        print(f"LAST CLOSED UTC DAY  {latest.date()}   total {int(day.sum()):,}")
        for i, (name, n) in enumerate(day.items(), 1):
            mark = "+" if name in incomplete else ""
            print(f"  {i:2d}. {name:<20} {int(n):>8,}{mark}")

        print()
        print(f"LAST 7 CLOSED DAYS   total {int(totals.sum()):,}")
        for i, (name, n) in enumerate(totals.items(), 1):
            mark = "+" if name in incomplete else ""
            print(f"  {i:2d}. {name:<20} {int(n):>8,}{mark}   {share[name]:5.1f}%")
        print()
        print("+ means a lower bound (platform feed, complete=false).")

    def plot(self) -> None:
        pivot = (self.df.pivot_table(index="day", columns="platform",
                                     values="launches", aggfunc="sum")
                 .fillna(0)
                 .sort_index())
        order = pivot.sum().sort_values(ascending=False).index.tolist()
        pivot = pivot[order]
        ma = pivot.rolling(7, min_periods=1).mean()
        total = pivot.sum(axis=1)
        share = pivot.div(pivot.sum(axis=1).replace(0, pd.NA), axis=0) * 100
        cumulative = pivot.cumsum()

        as_of = self.df["day"].max().strftime("%b %d, %Y")
        title_pads = " • ".join(order[:4])
        if len(order) > 4:
            title_pads += " • etc."

        fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=120)
        colors = [COLORS.get(name, "#7f8c8d") for name in order]

        for name in order:
            axs[0, 0].plot(ma.index, ma[name], label=name, linewidth=2.2,
                           color=COLORS.get(name))
        axs[0, 0].set_title("7-day MA — daily launches by platform")
        axs[0, 0].set_ylabel("Avg daily creations")
        axs[0, 0].legend(fontsize=8, loc="upper left", ncol=2)
        axs[0, 0].grid(True, alpha=0.3)

        axs[0, 1].plot(total.index, total.rolling(7, min_periods=1).mean(),
                       color="#6c3483", linewidth=2.6, label="7-day MA")
        axs[0, 1].plot(total.index, total, color="#6c3483", alpha=0.25,
                       label="Raw daily total")
        axs[0, 1].set_title("Total daily launches, all counted pads")
        axs[0, 1].legend(fontsize=9)
        axs[0, 1].grid(True, alpha=0.3)

        share.plot.area(ax=axs[1, 0], stacked=True, linewidth=0, alpha=0.9,
                        color=colors, legend=False)
        axs[1, 0].set_title("Share of counted launches")
        axs[1, 0].set_ylabel("% of total")
        axs[1, 0].set_ylim(0, 100)
        axs[1, 0].legend(order, fontsize=7, loc="upper left", ncol=2)
        axs[1, 0].grid(True, alpha=0.3)

        cumulative.plot.area(ax=axs[1, 1], stacked=True, linewidth=0, alpha=0.9,
                             color=colors, legend=False)
        axs[1, 1].set_title("Cumulative launches in this window")
        axs[1, 1].legend(order, fontsize=7, loc="upper left", ncol=2)
        axs[1, 1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Memecoin launch dashboard — {as_of}\n{title_pads}",
            fontsize=15, fontweight="bold",
        )
        fig.text(
            0.01, 0.01,
            "Source: MemeFees launches-daily (CC BY 4.0). "
            "Pump.fun/Bags are lower bounds. Partial UTC day excluded. "
            "https://memefees.com",
            fontsize=8, color="#555555",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(self.outfile, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved {self.outfile}")

    def run(self) -> None:
        self.fetch()
        self.print_rank()
        self.plot()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, help="1..730")
    parser.add_argument("--out", default="memecoin_launch_dashboard.png")
    parser.add_argument("--keep-partial", action="store_true")
    args = parser.parse_args()
    LaunchDashboard(args.days, args.out, args.keep_partial).run()
    print(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


if __name__ == "__main__":
    main()
