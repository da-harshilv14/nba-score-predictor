"""Turn cleaned games into model features.

Modelling setup: each row is a game a team already played. The **label**
(``target``) is whether that team wins its *next* game. The **features** on the
row describe the team's state up to and including the current game. Because we
only ever predict the next game, using the current game's stats is not leakage —
those stats are known before the next game is played.

Every feature here is computed from past games only (or the current one), never
from the future, and all of it is vectorised, so it runs in a couple of seconds
instead of the minutes the original row-by-row version took.

Features added:
    target                      label: did the team win its next game (0/1)
    home_next/team_opp_next/... context of the next game (who, where, when)
    elo_pre/opp_elo_pre/elo_diff Elo rating before the game
    *_r10                        rolling mean of each stat over the last 10 games
    win_streak/lose_streak       current streak length
    hot/cold                     streak >= 3 flags
    home_wr/away_wr              team's prior win rate at home / away
    h2h_wr_next                  prior win rate vs the next opponent
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRAL_WR = 0.5  # win rate used when there is no history yet


# --------------------------------------------------------------------------- #
# Label + next-game context
# --------------------------------------------------------------------------- #
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row with whether the team won its *next* game."""
    df = df.copy()
    df["target"] = df.groupby("team", sort=False)["won"].shift(-1)
    return df


def add_next_game_context(df: pd.DataFrame) -> pd.DataFrame:
    """Attach who/where/when of each team's next game."""
    df = df.copy()
    g = df.groupby("team", sort=False)
    df["home_next"] = g["home"].shift(-1)
    df["team_opp_next"] = g["team_opp"].shift(-1)
    df["date_next"] = g["date"].shift(-1)
    return df


# --------------------------------------------------------------------------- #
# Elo ratings (single chronological pass)
# --------------------------------------------------------------------------- #
def add_elo(
    df: pd.DataFrame,
    *,
    k: float = 20.0,
    base: float = 1500.0,
    carry: float = 0.75,
    use_mov: bool = True,
) -> pd.DataFrame:
    """Add each team's Elo rating *before* each game, plus the Elo difference.

    Elo rises when you win and falls when you lose, by more when the result is a
    surprise. At the start of a new season a team's rating is pulled 25% back
    toward the 1500 average (``carry=0.75``) so last year doesn't dominate.

    With ``use_mov=True`` the update is scaled by the **margin of victory**
    (538's NBA formula): blowouts move the rating more than narrow wins, with a
    correction so that heavy favourites don't gain too much from winning big.
    """
    df = df.sort_values("date").reset_index(drop=True)
    teams = pd.unique(pd.concat([df["team"], df["team_opp"]]))
    elo = {t: base for t in teams}
    last_season = {t: int(df["season"].min()) for t in teams}

    # Stable per-game id (same for both perspective rows): sorted teams + date.
    lo = np.minimum(df["team"].values, df["team_opp"].values)
    hi = np.maximum(df["team"].values, df["team_opp"].values)
    game_id = lo + "_" + hi + "_" + df["date"].dt.strftime("%Y%m%d").values

    pre_team = np.full(len(df), np.nan)
    pre_opp = np.full(len(df), np.nan)

    for positions in pd.Series(game_id).groupby(game_id, sort=False).indices.values():
        i0 = positions[0]
        team = df.at[i0, "team"]
        opp = df.at[i0, "team_opp"]
        season = int(df.at[i0, "season"])
        team_won = bool(df.at[i0, "won"])

        # Season regression toward the mean.
        for t in (team, opp):
            if season != last_season[t]:
                elo[t] = elo[t] * carry + base * (1 - carry)
                last_season[t] = season

        # Record the pre-game ratings on both perspective rows.
        for j in positions:
            pre_team[j] = elo[df.at[j, "team"]]
            pre_opp[j] = elo[df.at[j, "team_opp"]]

        # Update once per game.
        r_team, r_opp = elo[team], elo[opp]
        expected_team = 1.0 / (1.0 + 10 ** ((r_opp - r_team) / 400.0))
        score_team = 1.0 if team_won else 0.0

        mult = 1.0
        if use_mov:
            mult = _mov_multiplier(
                margin=df.at[i0, "total"] - df.at[i0, "total_opp"],
                winner_elo=r_team if team_won else r_opp,
                loser_elo=r_opp if team_won else r_team,
            )

        delta = k * mult * (score_team - expected_team)
        elo[team] = r_team + delta
        elo[opp] = r_opp - delta  # zero-sum: opponent loses what the team gains

    df["elo_pre"] = pre_team
    df["opp_elo_pre"] = pre_opp
    df["elo_diff"] = df["elo_pre"] - df["opp_elo_pre"]
    return df


def _mov_multiplier(*, margin: float, winner_elo: float, loser_elo: float) -> float:
    """538's margin-of-victory multiplier for the Elo update.

    Grows with the winning margin but divides by a term in the winner's rating
    edge, so a strong favourite blowing out a weak team gains less (it was
    expected). Falls back to 1.0 if the score is missing.
    """
    if pd.isna(margin):
        return 1.0
    margin = abs(float(margin))
    elo_edge = winner_elo - loser_elo
    return ((margin + 3.0) ** 0.8) / (7.5 + 0.006 * elo_edge)


# --------------------------------------------------------------------------- #
# Rolling form
# --------------------------------------------------------------------------- #
def add_rolling(df: pd.DataFrame, *, window: int = 10) -> pd.DataFrame:
    """Add the rolling mean of every stat over each team's last ``window`` games."""
    df = df.copy()
    exclude = {"season", "home", "target", "home_next"}
    stat_cols = [
        c for c in df.select_dtypes("number").columns
        if c not in exclude and not c.endswith("_r%d" % window)
    ]
    rolled = (
        df.groupby("team", sort=False)[stat_cols]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolled.columns = [f"{c}_r{window}" for c in stat_cols]
    return pd.concat([df, rolled], axis=1)


# --------------------------------------------------------------------------- #
# Streaks
# --------------------------------------------------------------------------- #
def add_streaks(df: pd.DataFrame, *, hot_threshold: int = 3) -> pd.DataFrame:
    """Add current win/lose streak length and hot/cold flags."""
    df = df.copy()
    won = df.groupby("team", sort=False)["won"]

    df["win_streak"] = won.transform(lambda s: _run_length(s.astype(bool)))
    df["lose_streak"] = won.transform(lambda s: _run_length(~s.astype(bool)))
    df["hot"] = (df["win_streak"] >= hot_threshold).astype(int)
    df["cold"] = (df["lose_streak"] >= hot_threshold).astype(int)
    return df


def _run_length(flag: pd.Series) -> pd.Series:
    """Length of the current run of True values, resetting to 0 on False."""
    blocks = (flag != flag.shift()).cumsum()
    run = flag.groupby(blocks).cumcount() + 1
    return run.where(flag, 0)


# --------------------------------------------------------------------------- #
# Win rates (home/away and vs next opponent)
# --------------------------------------------------------------------------- #
def add_home_away_winrate(df: pd.DataFrame) -> pd.DataFrame:
    """Add each team's prior win rate at home and away (forward-filled)."""
    df = df.copy()
    df["home_wr"] = _venue_winrate(df, home_flag=1).fillna(NEUTRAL_WR)
    df["away_wr"] = _venue_winrate(df, home_flag=0).fillna(NEUTRAL_WR)
    return df


def _venue_winrate(df: pd.DataFrame, *, home_flag: int) -> pd.Series:
    """Prior (shifted) expanding win rate in one venue, carried onto every row."""
    venue = df[df["home"] == home_flag]
    wr = venue.groupby("team", sort=False)["won"].transform(
        lambda s: s.shift().expanding().mean()
    )
    out = pd.Series(np.nan, index=df.index)
    out.loc[venue.index] = wr
    return out.groupby(df["team"], sort=False).ffill()


def add_h2h_vs_next_opp(df: pd.DataFrame) -> pd.DataFrame:
    """Add the team's prior win rate against its next opponent."""
    df = df.copy()

    # Prior win rate of team vs the opponent it actually faced (shifted = past only).
    h2h = df.groupby(["team", "team_opp"], sort=False)["won"].transform(
        lambda s: s.shift().expanding().mean()
    )
    lookup = (
        df[["team", "team_opp", "date"]]
        .assign(h2h_wr=h2h)
        .rename(columns={"team_opp": "opp"})
        .sort_values("date")
    )
    query = (
        df[["team", "team_opp_next", "date"]]
        .rename(columns={"team_opp_next": "opp"})
        .sort_values("date")
    )

    # For each row, grab the most recent known head-to-head win rate (date <= now).
    merged = pd.merge_asof(
        query, lookup, on="date", by=["team", "opp"], direction="backward"
    )
    df["h2h_wr_next"] = merged["h2h_wr"].to_numpy()
    df["h2h_wr_next"] = df["h2h_wr_next"].fillna(NEUTRAL_WR)
    return df


# --------------------------------------------------------------------------- #
# Schedule / rest
# --------------------------------------------------------------------------- #
def add_schedule_features(df: pd.DataFrame, *, max_rest: int = 10) -> pd.DataFrame:
    """Add rest-days and schedule-density features.

    Fatigue is a real edge in the NBA: teams on a back-to-back (playing a second
    game in two nights) win less often. These are known ahead of time from the
    schedule, so they are fair features for predicting the next game.

        rest_days_prev  days since the team's previous game (recent fatigue)
        rest_days_next  days of rest before the next game (the one we predict)
        b2b_next        1 if the next game is a back-to-back (<= 1 day rest)
        games_last_7    games played in the trailing 7 days (schedule density)

    Rest is clipped at ``max_rest`` so long off-season gaps don't look like
    meaningful "rest".
    """
    df = df.sort_values(["team", "date"]).copy()
    g = df.groupby("team", sort=False)

    df["rest_days_prev"] = (
        (df["date"] - g["date"].shift(1)).dt.days.clip(upper=max_rest).fillna(max_rest)
    )
    df["rest_days_next"] = (df["date_next"] - df["date"]).dt.days.clip(upper=max_rest)
    df["b2b_next"] = (df["rest_days_next"] <= 1).astype(int)

    # Count of games in the trailing 7 days (needs a datetime index to roll on).
    by_date = df.set_index("date")
    df["games_last_7"] = (
        by_date.groupby("team", sort=False)["won"].rolling("7D").count().to_numpy()
    )

    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, *, rolling_window: int = 10) -> pd.DataFrame:
    """Run the full feature pipeline on a cleaned games frame."""
    df = add_target(df)
    df = add_next_game_context(df)
    df = add_rolling(df, window=rolling_window)
    df = add_schedule_features(df)
    df = add_streaks(df)
    df = add_home_away_winrate(df)
    df = add_h2h_vs_next_opp(df)
    df = add_elo(df)
    return df
