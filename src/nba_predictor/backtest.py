"""Season-based walk-forward evaluation.

You cannot validate a time-series model with a random train/test split — that
would train on future games and test on past ones. Instead we walk forward
season by season: train on every season *before* season S, test on season S,
then step to the next season. This mirrors how the model would really be used.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score

from .models import win_probability


def _trainable(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rows usable for training/testing: known label and no missing features."""
    return df.dropna(subset=["target"] + features).copy()


def backtest(
    df: pd.DataFrame,
    model,
    features: list[str],
    *,
    start_index: int = 1,
) -> pd.DataFrame:
    """Walk forward over seasons, returning one row per test prediction.

    Parameters
    ----------
    df:        feature frame from ``features.build_features``.
    model:     any estimator with ``fit`` / ``predict`` (see ``models``).
    features:  predictor column names.
    start_index: index into the sorted season list of the first test season
                 (``1`` means the earliest season is only ever used for training).
    """
    data = _trainable(df, features)
    data["target"] = data["target"].astype(int)
    seasons = sorted(data["season"].unique())

    out = []
    for i in range(start_index, len(seasons)):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        if train.empty or test.empty:
            continue

        model.fit(train[features], train["target"])
        preds = model.predict(test[features])
        proba = win_probability(model, test[features])

        out.append(
            pd.DataFrame(
                {
                    "season": season,
                    "date": test["date"].to_numpy(),
                    "team": test["team"].to_numpy(),
                    "actual": test["target"].to_numpy(),
                    "prediction": preds,
                    "proba": proba,
                }
            )
        )

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def accuracy(predictions: pd.DataFrame) -> float:
    """Overall accuracy of a backtest result."""
    if predictions.empty:
        return float("nan")
    return accuracy_score(predictions["actual"], predictions["prediction"])


def ensemble_backtest(
    df: pd.DataFrame,
    models: dict[str, object],
    features: list[str],
    *,
    start_index: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest several models and a majority-vote ensemble.

    Returns the ensemble prediction frame and a ``{name: accuracy}`` dict that
    also includes the ``"ensemble"`` entry.
    """
    per_model = {
        name: backtest(df, model, features, start_index=start_index)
        for name, model in models.items()
    }
    scores = {name: accuracy(pred) for name, pred in per_model.items()}

    # All models predict the same aligned test rows, so stack their outputs.
    base = next(iter(per_model.values()))
    votes = pd.DataFrame({name: p["prediction"].to_numpy() for name, p in per_model.items()})
    probas = pd.DataFrame({name: p["proba"].to_numpy() for name, p in per_model.items()})

    ensemble = base.copy()
    ensemble["prediction"] = votes.mode(axis=1)[0].astype(int)  # majority vote
    ensemble["proba"] = probas.mean(axis=1).to_numpy()          # mean probability
    scores["ensemble"] = accuracy(ensemble)
    return ensemble, scores
