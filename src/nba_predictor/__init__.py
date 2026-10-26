"""nba_predictor: predict NBA game outcomes from historical box-score data.

Pipeline, in the order you'd call it:
    data      -> load and clean the raw games table
    features  -> engineer Elo, rolling averages, win-rates, streaks
    models    -> build the Ridge / XGBoost / LogReg / NN estimators
    backtest  -> evaluate with season-based walk-forward validation
    predict   -> predict the winner of upcoming games
"""

__version__ = "0.1.0"
