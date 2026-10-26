"""The prediction models and the feature list they train on.

Four models, built behind small factory functions so the rest of the code never
hard-codes hyperparameters:

    Ridge        linear baseline (scaled)
    LogReg       logistic regression (scaled)
    XGBoost      gradient-boosted trees (no scaling needed)
    NN           small Keras neural network (scaled, TensorFlow loaded lazily)

Ridge / LogReg / NN are sensitive to feature scale, so they are wrapped so that
a MinMaxScaler is fitted on the *training* fold only — scaling the whole dataset
up front would leak test-set information. XGBoost is scale-invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Engineered scalar features (everything ending in _r<window> is added on top).
BASE_FEATURES = [
    "elo_pre",
    "opp_elo_pre",
    "elo_diff",
    "win_streak",
    "lose_streak",
    "hot",
    "cold",
    "home_wr",
    "away_wr",
    "h2h_wr_next",
    "home_next",
    "rest_days_prev",
    "rest_days_next",
    "b2b_next",
    "games_last_7",
]


def select_features(df: pd.DataFrame) -> list[str]:
    """Return the predictor columns present in ``df`` (rolling + engineered)."""
    rolling = [c for c in df.columns if "_r" in c and df[c].dtype != object]
    cols = [c for c in BASE_FEATURES if c in df.columns] + rolling
    # Keep only numeric, de-duplicate while preserving order.
    seen: dict[str, None] = {}
    for c in cols:
        if c in df.columns and df[c].dtype != object:
            seen.setdefault(c, None)
    return list(seen)


def build_ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline([("scale", MinMaxScaler()), ("model", RidgeClassifier(alpha=alpha))])


def build_logreg(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        [("scale", MinMaxScaler()), ("model", LogisticRegression(C=c, max_iter=max_iter))]
    )


def build_xgb(**params) -> XGBClassifier:
    defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        verbosity=0,
    )
    defaults.update(params)
    return XGBClassifier(**defaults)


class NNClassifier:
    """Minimal scikit-learn-style wrapper around a small Keras network.

    Implements ``fit`` / ``predict`` / ``predict_proba`` so it can flow through
    the same backtest and ensemble as the other models. TensorFlow is imported
    lazily so that using the other models never pays the import cost.
    """

    def __init__(self, epochs: int = 50, batch_size: int = 32, l2: float = 0.02):
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.scaler = MinMaxScaler()
        self.model = None

    def _build(self, input_dim: int):
        from tensorflow.keras.layers import Dense, Dropout, Input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.regularizers import l2 as l2_reg

        model = Sequential(
            [
                Input(shape=(input_dim,)),
                Dense(12, activation="tanh", kernel_regularizer=l2_reg(self.l2)),
                Dropout(0.2),
                Dense(6, activation="tanh", kernel_regularizer=l2_reg(self.l2)),
                Dropout(0.1),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y):
        X = self.scaler.fit_transform(np.asarray(X, dtype="float32"))
        y = np.asarray(y, dtype="float32")
        self.model = self._build(X.shape[1])
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X):
        X = self.scaler.transform(np.asarray(X, dtype="float32"))
        p = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_models(include_nn: bool = False) -> dict[str, object]:
    """Return the standard set of models keyed by name."""
    models = {"ridge": build_ridge(), "xgb": build_xgb(), "logreg": build_logreg()}
    if include_nn:
        models["nn"] = NNClassifier()
    return models


def win_probability(model, X) -> np.ndarray:
    """P(win) for each row, working across all model types.

    Uses ``predict_proba`` when available, a sigmoid of ``decision_function``
    for RidgeClassifier, and the hard 0/1 prediction as a last resort.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return 1.0 / (1.0 + np.exp(-model.decision_function(X)))
    return model.predict(X).astype(float)


def calibrated(estimator, *, method: str = "sigmoid", cv: int = 3):
    """Wrap an estimator so its probabilities are calibrated (Platt/isotonic).

    Calibration makes a predicted "0.7" mean closer to a true 70% chance. It
    does not change the hard predictions much, but improves Brier/log-loss.
    """
    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(estimator, method=method, cv=cv)
