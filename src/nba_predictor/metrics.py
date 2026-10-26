"""Evaluation metrics beyond accuracy.

Accuracy only asks "was the winner called right?". For predictions you might act
on, you also want the *probabilities* to be honest — a "70%" should win about 70%
of the time. These metrics measure that:

    accuracy   fraction of correct hard predictions (higher is better)
    brier      mean squared error of the probability (lower is better)
    log_loss   penalises confident wrong predictions heavily (lower is better)

``reliability_curve`` buckets predictions by confidence so you can plot predicted
vs actual win rate — a perfectly calibrated model lies on the diagonal.

All functions take a backtest result frame with ``actual`` and ``proba`` columns
(as produced by ``backtest``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def summarize(predictions: pd.DataFrame) -> dict[str, float]:
    """Return accuracy, Brier score and log-loss for a backtest result."""
    if predictions.empty:
        return {"accuracy": float("nan"), "brier": float("nan"), "log_loss": float("nan")}

    actual = predictions["actual"].to_numpy()
    proba = predictions["proba"].clip(1e-6, 1 - 1e-6).to_numpy()
    return {
        "accuracy": accuracy_score(actual, predictions["prediction"]),
        "brier": brier_score_loss(actual, proba),
        "log_loss": log_loss(actual, proba),
    }


def reliability_curve(predictions: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """Bucket predictions by confidence; compare predicted vs actual win rate.

    Returns one row per bucket with the mean predicted probability, the actual
    fraction of wins, and the count. Plot ``mean_pred`` against ``frac_pos`` —
    points on the diagonal mean the probabilities are well calibrated.
    """
    df = predictions[["actual", "proba"]].copy()
    df["bucket"] = pd.cut(df["proba"], np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = df.groupby("bucket", observed=True)
    out = grouped.agg(
        mean_pred=("proba", "mean"),
        frac_pos=("actual", "mean"),
        count=("actual", "size"),
    ).reset_index(drop=True)
    return out
