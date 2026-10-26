"""Compare ways of combining the base models.

Given the per-model walk-forward predictions, we try four ways to merge them:

    hard      majority vote of the 0/1 predictions (what we started with)
    soft      average of the predicted probabilities, then threshold at 0.5
    weighted  probability average weighted by each model's accuracy
    stack     a logistic meta-model trained (season-aware) on the base
              models' probabilities — it *learns* how to combine them

The honest expectation: when the base models are similar and already ~60%,
these rarely beat the simplest option by much. The value is in *showing* the
comparison with proper metrics, not assuming a winner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .backtest import backtest
from .metrics import summarize


def _base_frames(df, models, features, start_index):
    """Run each base model's walk-forward backtest; return aligned frames."""
    return {
        name: backtest(df, model, features, start_index=start_index)
        for name, model in models.items()
    }


def _meta_table(base_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per test game with each model's probability + the shared label."""
    first = next(iter(base_frames.values()))
    meta = first[["season", "actual"]].reset_index(drop=True).copy()
    for name, frame in base_frames.items():
        meta[f"p_{name}"] = frame["proba"].to_numpy()
    return meta


def _result(meta: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Wrap a probability vector into the frame shape metrics.summarize expects."""
    return pd.DataFrame(
        {
            "actual": meta["actual"].to_numpy(),
            "proba": proba,
            "prediction": (proba >= 0.5).astype(int),
        }
    )


def _stack(meta: pd.DataFrame, proba_cols: list[str]) -> np.ndarray:
    """Season-aware stacking: a logistic meta-model over the base probabilities.

    For each test season we train the meta-model on earlier seasons only, so it
    never sees the season it predicts. The earliest season (no prior data) falls
    back to the plain average.
    """
    seasons = sorted(meta["season"].unique())
    out = np.full(len(meta), np.nan)
    for i, season in enumerate(seasons):
        test_mask = (meta["season"] == season).to_numpy()
        if i == 0:
            out[test_mask] = meta.loc[test_mask, proba_cols].mean(axis=1).to_numpy()
            continue
        train = meta[meta["season"] < season]
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(train[proba_cols], train["actual"])
        out[test_mask] = meta_model.predict_proba(meta.loc[test_mask, proba_cols])[:, 1]
    return out


def compare_ensembles(df, models, features, *, start_index: int = 2) -> pd.DataFrame:
    """Score the base models and four ensembles; return a metrics table."""
    base_frames = _base_frames(df, models, features, start_index)
    meta = _meta_table(base_frames)
    proba_cols = [f"p_{name}" for name in base_frames]

    rows = {}

    # Base models on their own.
    for name, frame in base_frames.items():
        rows[name] = summarize(frame)

    # Ensembles.
    probs = meta[proba_cols].to_numpy()
    rows["soft"] = summarize(_result(meta, probs.mean(axis=1)))

    accs = np.array([rows[name]["accuracy"] for name in base_frames])
    weights = accs / accs.sum()
    rows["weighted"] = summarize(_result(meta, probs @ weights))

    hard = (probs >= 0.5).astype(int)
    hard_vote = (hard.mean(axis=1) >= 0.5).astype(int)
    rows["hard"] = summarize(
        pd.DataFrame({"actual": meta["actual"], "proba": probs.mean(axis=1), "prediction": hard_vote})
    )

    rows["stack"] = summarize(_result(meta, _stack(meta, proba_cols)))

    table = pd.DataFrame(rows).T[["accuracy", "brier", "log_loss"]]
    return table.sort_values("accuracy", ascending=False).round(4)
