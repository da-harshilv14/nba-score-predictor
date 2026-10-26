"""Predict the winner of upcoming games.

The model was trained on rows that describe a team's state *up to its most
recent game*, with a label for its *next* game. To predict a real upcoming
game we therefore take each team's latest known feature row (its current form,
streaks, rolling stats and rating) and override the "next game" fields with the
actual upcoming matchup — who the opponent is, whether it's a home game, the
head-to-head record, and the Elo gap.

Each game is scored from both sides (home's chance of winning its next game, and
away's chance of winning its next game) and the two are combined into a single
probability that the home team wins.

Note: a team's rating is taken from its most recent row (``elo_pre``), i.e. its
rating entering its last game — a one-game-stale approximation, which is fine
for a forward prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import select_features, win_probability

Game = tuple  # (home, away) or (home, away, date)


def predict_games(
    feat: pd.DataFrame,
    model,
    games: list[Game],
    *,
    features: list[str] | None = None,
    fit: bool = True,
) -> pd.DataFrame:
    """Predict a list of upcoming games.

    Parameters
    ----------
    feat:   feature frame from ``features.build_features``.
    model:  a model from ``models`` (fitted here if ``fit=True``).
    games:  list of ``(home, away)`` or ``(home, away, date)`` tuples.
    fit:    if True, train ``model`` on all labelled rows first.

    Returns
    -------
    DataFrame with columns: date, home, away, p_home_win, predicted_winner.
    """
    features = features or select_features(feat)

    if fit:
        labelled = feat.dropna(subset=["target"] + features).copy()
        model.fit(labelled[features], labelled["target"].astype(int))

    latest = _latest_row_per_team(feat)
    elos = _current_elo(feat)

    rows = []
    for game in games:
        home, away = game[0], game[1]
        date = game[2] if len(game) > 2 else None
        _check_known(latest, home, away)

        home_row = _prediction_row(latest, feat, elos, team=home, opp=away, is_home=True)
        away_row = _prediction_row(latest, feat, elos, team=away, opp=home, is_home=False)

        X = pd.DataFrame(
            [home_row[features].to_numpy(), away_row[features].to_numpy()],
            columns=features,
        ).astype(float)

        p = win_probability(model, X)  # P(team wins its next game), per perspective
        p_home_win = float((p[0] + (1 - p[1])) / 2)  # combine both viewpoints

        rows.append(
            {
                "date": date,
                "home": home,
                "away": away,
                "p_home_win": round(p_home_win, 3),
                "predicted_winner": home if p_home_win >= 0.5 else away,
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _latest_row_per_team(feat: pd.DataFrame) -> pd.DataFrame:
    """Most recent feature row for each team, indexed by team."""
    idx = feat.groupby("team")["date"].idxmax()
    return feat.loc[idx].set_index("team")


def _current_elo(feat: pd.DataFrame) -> dict[str, float]:
    """Each team's latest known Elo rating."""
    latest = _latest_row_per_team(feat)
    return latest["elo_pre"].to_dict()


def _h2h_winrate(feat: pd.DataFrame, team: str, opp: str) -> float:
    """Team's historical win rate against a specific opponent (0.5 if never met)."""
    games = feat[(feat["team"] == team) & (feat["team_opp"] == opp)]
    return float(games["won"].mean()) if len(games) else 0.5


def _prediction_row(
    latest: pd.DataFrame,
    feat: pd.DataFrame,
    elos: dict[str, float],
    *,
    team: str,
    opp: str,
    is_home: bool,
) -> pd.Series:
    """Build one team's feature row for the upcoming game."""
    row = latest.loc[team].copy()
    row["home_next"] = 1 if is_home else 0
    row["team_opp_next"] = opp
    row["h2h_wr_next"] = _h2h_winrate(feat, team, opp)
    row["elo_pre"] = elos.get(team, 1500.0)
    row["opp_elo_pre"] = elos.get(opp, 1500.0)
    row["elo_diff"] = row["elo_pre"] - row["opp_elo_pre"]
    return row


def _check_known(latest: pd.DataFrame, *teams: str) -> None:
    unknown = [t for t in teams if t not in latest.index]
    if unknown:
        raise KeyError(f"No history for team(s): {unknown}. Known: {sorted(latest.index)}")
