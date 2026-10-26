"""Shared helpers for scraping box scores from Basketball Reference.

These functions turn a single box-score page into the two-row-per-game format
the rest of the project expects (team perspective + opponent perspective).

Please scrape responsibly: Basketball Reference asks for no more than ~20
requests per minute, so every page fetch is spaced out by ``REQUEST_DELAY``.
Review their terms of use before running any scraper.
"""

from __future__ import annotations

import time
from io import StringIO

import pandas as pd

BASE_URL = "https://www.basketball-reference.com"
REQUEST_DELAY = 3.5  # seconds between page loads (stay under ~20 req/min)


def make_driver(headless: bool = True):
    """Create a Chrome WebDriver. Selenium + ChromeDriver must be installed."""
    from selenium import webdriver

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def polite_get(driver, url: str) -> None:
    """Load a URL, then wait so we stay within the site's rate limit."""
    driver.get(url)
    time.sleep(REQUEST_DELAY)


def season_for(year: int, month: int) -> int:
    """Basketball Reference labels a season by its ending year (Oct-Dec -> +1)."""
    return year + 1 if month >= 7 else year


def _read_table(driver, div_id: str, *, numeric: bool) -> pd.DataFrame:
    """Read one stats table (by its wrapper div id) into a DataFrame."""
    from selenium.webdriver.common.by import By

    html = driver.find_element(By.ID, div_id).get_attribute("innerHTML")
    df = pd.read_html(StringIO(html), index_col=0 if numeric else None)[0]
    df.columns = df.columns.droplevel()  # drop the grouped header row
    if numeric:
        df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _team_stats(driver, team: str) -> pd.Series:
    """One team's totals (last row) + per-player maxes, basic & advanced combined."""
    basic = _read_table(driver, f"div_box-{team}-game-basic", numeric=True)
    advanced = _read_table(driver, f"div_box-{team}-game-advanced", numeric=True)

    totals = pd.concat([basic.iloc[-1], advanced.iloc[-1]])
    totals.index = totals.index.str.lower()

    maxes = pd.concat([basic.iloc[:-1].max(), advanced.iloc[:-1].max()])
    maxes.index = maxes.index.str.lower() + "_max"

    return pd.concat([totals, maxes])


def parse_boxscore(driver, *, season: int, date: str) -> pd.DataFrame:
    """Parse the currently loaded box-score page into a 2-row game frame."""
    line = _read_table(driver, "div_line_score", numeric=False)
    line = line.rename(columns={"Unnamed: 0_level_1": "team", "T": "total"})
    line = line[["team", "total"]].reset_index(drop=True)

    teams = list(line["team"])
    stats = [_team_stats(driver, t) for t in teams]

    base_cols = [c for c in stats[0].index.drop_duplicates() if "bpm" not in c]
    stat_df = pd.concat([s[base_cols] for s in stats], axis=1).T.reset_index(drop=True)

    game = pd.concat([stat_df, line], axis=1)
    game["home"] = [0, 1]  # visitor listed first on Basketball Reference

    opp = game.iloc[::-1].reset_index(drop=True)
    opp.columns = [f"{c}_opp" for c in opp.columns]

    full = pd.concat([game, opp], axis=1)
    full["season"] = season
    full["date"] = date
    full["won"] = full["total"] > full["total_opp"]
    return full
