"""Scrape one or more full NBA seasons of box scores.

Usage:
    python scripts/scrape_season.py 2024
    python scripts/scrape_season.py 2022 2023 2024 --out data/raw

Writes data/raw/<season>_games.csv for each season. A season is named by its
ending year (the 2023-24 season is "2024"). Requires Selenium + ChromeDriver.

This fetches hundreds of pages per season at ~3.5s each, so a single season
takes well over an hour. Respect Basketball Reference's terms of use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scrape_common import make_driver, parse_boxscore, polite_get


def month_page_links(driver, season: int) -> list[str]:
    """Links to each month's schedule page for a season."""
    from selenium.webdriver.common.by import By

    polite_get(driver, f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html")
    anchors = driver.find_elements(By.CSS_SELECTOR, "div.filter a")
    return [a.get_attribute("href") for a in anchors]


def boxscore_links_for_month(driver, month_url: str) -> list[str]:
    """All box-score links listed on one month's schedule page."""
    from selenium.webdriver.common.by import By

    polite_get(driver, month_url)
    anchors = driver.find_elements(By.CSS_SELECTOR, "td[data-stat='box_score_text'] a")
    return [a.get_attribute("href") for a in anchors]


def scrape_season(season: int, *, headless: bool = True) -> pd.DataFrame:
    driver = make_driver(headless=headless)
    try:
        games = []
        for month_url in month_page_links(driver, season):
            for link in boxscore_links_for_month(driver, month_url):
                polite_get(driver, link)
                date_str = _date_from_boxscore_url(link)
                games.append(parse_boxscore(driver, season=season, date=date_str))
        return pd.concat(games, ignore_index=True) if games else pd.DataFrame()
    finally:
        driver.quit()


def _date_from_boxscore_url(url: str) -> str:
    """Box-score URLs end in .../YYYYMMDD0TTT.html -> 'YYYY-MM-DD'."""
    code = url.rstrip("/").split("/")[-1]  # e.g. 202401150BOS.html
    digits = code[:8]
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seasons", type=int, nargs="+", help="season end-years, e.g. 2023 2024")
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--no-headless", action="store_true", help="show the browser")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for season in args.seasons:
        print(f"Scraping season {season} ...")
        df = scrape_season(season, headless=not args.no_headless)
        if df.empty:
            print(f"  no games found for {season}")
            continue
        out_path = out_dir / f"{season}_games.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {len(df)} rows ({len(df) // 2} games) to {out_path}")


if __name__ == "__main__":
    main()
