"""Streamlit front-end for the NBA game outcome predictor.

Run from the repo root:
    streamlit run app/streamlit_app.py

Pick a home and away team and get a predicted winner with a win probability,
plus each team's current Elo rating and recent form.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from nba_predictor.data import load_games  # noqa: E402
from nba_predictor.features import build_features  # noqa: E402
from nba_predictor.models import build_models, select_features  # noqa: E402
from nba_predictor.predict import predict_games  # noqa: E402


@st.cache_data(show_spinner="Loading data and building features…")
def get_features():
    path = ROOT / "data" / "processed" / "games.csv"
    if not path.exists():
        path = ROOT / "data" / "sample" / "games_sample.csv"
    return build_features(load_games(path)), path.name


@st.cache_resource(show_spinner="Training model…")
def get_model(_feat):
    features = select_features(_feat)
    model = build_models()["xgb"]
    labelled = _feat.dropna(subset=["target"] + features)
    model.fit(labelled[features], labelled["target"].astype(int))
    return model, features


def _team_snapshot(feat, team):
    """Latest Elo and recent win rate (last 10 games) for a team, for display."""
    games = feat[feat["team"] == team].sort_values("date")
    elo = round(float(games["elo_pre"].iloc[-1]))
    form = f"{games['won'].tail(10).mean():.0%}"
    return elo, form


def main():
    st.set_page_config(page_title="NBA Score Predictor", page_icon="🏀")
    st.title("🏀 NBA Game Outcome Predictor")
    st.caption("Predicts a matchup winner from team form, Elo ratings, and schedule.")

    feat, source = get_features()
    model, features = get_model(feat)
    teams = sorted(feat["team"].unique())

    col1, col2 = st.columns(2)
    home = col1.selectbox("Home team", teams, index=teams.index("BOS") if "BOS" in teams else 0)
    away = col2.selectbox("Away team", teams, index=teams.index("LAL") if "LAL" in teams else 1)

    if home == away:
        st.warning("Pick two different teams.")
        st.stop()

    # Context on each team.
    h_elo, h_form = _team_snapshot(feat, home)
    a_elo, a_form = _team_snapshot(feat, away)
    c1, c2 = st.columns(2)
    c1.metric(f"{home} (home)", f"Elo {h_elo}", f"form {h_form}")
    c2.metric(f"{away} (away)", f"Elo {a_elo}", f"form {a_form}")

    if st.button("Predict winner", type="primary"):
        result = predict_games(feat, model, [(home, away)], features=features, fit=False)
        p = float(result.iloc[0]["p_home_win"])
        winner = result.iloc[0]["predicted_winner"]

        st.subheader(f"Predicted winner: {winner}")
        st.progress(p)
        st.write(f"**{home}** win probability: **{p:.0%}**  ·  **{away}**: **{1 - p:.0%}**")

    st.divider()
    st.caption(
        f"Data: {source} · Model: XGBoost · ~60% walk-forward accuracy. "
        "Educational use only."
    )


if __name__ == "__main__":
    main()
