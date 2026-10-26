"""Scrape every NBA game played on a single day.

Usage:
    python scripts/scrape_day.py 2025-01-15
    python scripts/scrape_day.py 2025-01-15 --out data/raw --show

Writes data/raw/<date>_games.csv. Requires Selenium + ChromeDriver.
Respect Basketball Reference's terms of use and rate limits.
"""

from __future__ import annotations

import argparse
from datetime import date as date_cls
from pathlib import Path

import pandas as pd

from scrape_common import make_driver, parse_boxscore, polite_get, season_for


def boxscore_links(driver, day: date_cls) -> list[str]:
    """All box-score page links for a given day's scoreboard."""
    from selenium.webdriver.common.by import By

    url = (
        "https://www.basketball-reference.com/boxscores/"
        f"?month={day.month}&day={day.day}&year={day.year}"
    )
    polite_get(driver, url)
    anchors = driver.find_elements(By.CSS_SELECTOR, "td.gamelink a")
    return [a.get_attribute("href") for a in anchors]


def scrape_day(day: date_cls, *, headless: bool = True) -> pd.DataFrame:
    driver = make_driver(headless=headless)
    try:
        season = season_for(day.year, day.month)
        date_str = day.isoformat()
        games = []
        for link in boxscore_links(driver, day):
            polite_get(driver, link)
            games.append(parse_boxscore(driver, season=season, date=date_str))
        return pd.concat(games, ignore_index=True) if games else pd.DataFrame()
    finally:
        driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="day to scrape, YYYY-MM-DD")
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--no-headless", action="store_true", help="show the browser")
    parser.add_argument("--show", action="store_true", help="print the scraped frame")
    args = parser.parse_args()

    day = date_cls.fromisoformat(args.date)
    df = scrape_day(day, headless=not args.no_headless)

    if df.empty:
        print(f"No games found for {args.date}.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.date}_games.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({len(df) // 2} games) to {out_path}")
    if args.show:
        print(df[["team", "team_opp", "total", "total_opp", "won"]])


if __name__ == "__main__":
    main()
