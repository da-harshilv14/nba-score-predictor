"""Load and clean the raw NBA games table.

Each row in the raw data is **one team's perspective of one game**: that team's
box-score stats, the same stats suffixed ``_opp`` for its opponent, plus
``team``, ``team_opp``, ``home``, ``season``, ``date`` and ``won``. Every game
therefore appears as two rows (home perspective and away perspective).

The raw file has two known problems that this module fixes:

1. **Duplicate rows.** The season files are cumulative snapshots that overlap,
   so a naive concatenation is ~83% exact duplicates. Left in, they corrupt
   every rolling/streak feature and blow up a later self-join into millions of
   rows (out-of-memory). We keep exactly one row per ``(team, team_opp, date)``.
2. **Junk columns.** A few scraper artifacts (duplicated ``mp`` columns, an
   ``index_opp`` column, empty ``unnamed: ...`` columns) carry no signal.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns the scraper leaves behind that carry no information.
JUNK_COLUMNS = ["mp.1", "mp_opp.1", "index_opp"]

# The identity of a single game, from one team's perspective. Two scrapes of the
# same game produce identical keys, which is how we detect duplicates.
GAME_KEY = ["team", "team_opp", "date"]

# Columns that must be present and non-null for a row to be usable.
REQUIRED_COLUMNS = ["team", "team_opp", "date", "won"]


def load_games(path: str | Path, *, dedup: bool = True) -> pd.DataFrame:
    """Read the raw games CSV and return a cleaned, chronologically sorted frame.

    Parameters
    ----------
    path:
        Path to the games CSV (e.g. ``data/sample/games_sample.csv``).
    dedup:
        If ``True`` (default), keep one row per ``(team, team_opp, date)``.
        Turn it off only if you are certain the input is already unique.

    Returns
    -------
    pandas.DataFrame
        Cleaned games, sorted by ``date`` with a fresh ``RangeIndex``.
    """
    path = Path(path)
    df = pd.read_csv(path)

    df = _drop_junk_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")

    # Parse dates up front so sorting and later time-based features are correct.
    df["date"] = pd.to_datetime(df["date"])

    # A row with no team, opponent, date or result cannot be used for anything.
    df = df.dropna(subset=REQUIRED_COLUMNS)

    if dedup:
        df = df.drop_duplicates(subset=GAME_KEY, keep="last")

    # Chronological order is a hard requirement for every downstream feature
    # (rolling averages, streaks, Elo) and for the season-based backtest.
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove scraper-artifact columns: named junk, empty, and 'unnamed' columns."""
    df = df.drop(columns=JUNK_COLUMNS, errors="ignore")

    # Empty 'unnamed: ...' columns come from blank header cells in the scrape.
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    df = df.drop(columns=unnamed, errors="ignore")

    # Drop any column that is entirely null (no information to learn from).
    df = df.dropna(axis=1, how="all")
    return df
