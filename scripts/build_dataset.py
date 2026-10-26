"""Combine raw scraped CSVs into one clean dataset.

Concatenates every CSV in the input directory and keeps one row per
(team, team_opp, date). This is the step that prevents the duplicate-row
problem: raw scrapes can overlap, and left unchecked the duplicates corrupt
every rolling feature and blow up later joins.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --raw data/raw --out data/processed/games.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GAME_KEY = ["team", "team_opp", "date"]


def build_dataset(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {raw_dir.resolve()}")

    combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=GAME_KEY, keep="last").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Combined {len(csv_files)} files: {before:,} -> {len(combined):,} rows after dedup")
    print(f"Wrote {out_path}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw", help="directory of raw scraped CSVs")
    parser.add_argument("--out", default="data/processed/games.csv", help="output CSV path")
    args = parser.parse_args()
    build_dataset(Path(args.raw), Path(args.out))


if __name__ == "__main__":
    main()
