# NBA Game Outcome Prediction

Predict the winner of NBA games from historical box-score data, using feature
engineering (Elo ratings, rolling form, streaks, head-to-head records) and a set
of machine-learning models evaluated with honest, season-based backtesting.

## What it does

Given years of NBA box scores, the project:

1. **Cleans** the raw data (de-duplicates games, drops junk columns, sorts by date).
2. **Engineers features** that describe each team's state going into a game —
   Elo rating, 10-game rolling averages, win/lose streaks, home/away win rate,
   and record against the next opponent.
3. **Trains and evaluates** four models (Ridge, Logistic Regression, XGBoost, and
   a small neural network) plus a majority-vote ensemble.
4. **Predicts** the winner of upcoming matchups through a single clean function.

## Results

Evaluated with **season-based walk-forward validation** (train on past seasons,
test on the next — never on future games):

| Model                | Accuracy |
|----------------------|:--------:|
| Ridge                |  60.3%   |
| Logistic Regression  |  60.4%   |
| XGBoost              |  60.4%   |
| **Ensemble**         | **60.5%** |
| Baseline (coin flip) |  50.0%   |

> These numbers are deliberately *honest*. Earlier versions of this kind of
> project report ~65%, but that comes from data leakage — duplicate rows and
> scaling the whole dataset before splitting. With those fixed, ~60% on a strict
> walk-forward split is the trustworthy figure, and roughly in line with how hard
> NBA outcomes are to predict.

Beyond accuracy, the predicted **probabilities are well-calibrated** (a predicted
70% wins ~70% of the time). The full set of experiments — rest/schedule features,
margin-of-victory Elo, calibration, and an ensemble comparison — is written up in
**[RESULTS.md](RESULTS.md)**, including the honest finding that ensembling gave no
lift because the base models weren't diverse enough.

## Project structure

```
nba-score-predictor/
├── data/
│   ├── sample/games_sample.csv   # small committed sample so the repo runs out of the box
│   └── processed/games.csv       # full dataset (built locally; gitignored)
├── src/nba_predictor/            # the library
│   ├── data.py                   # load + clean + de-duplicate
│   ├── features.py               # Elo, rolling averages, streaks, win-rates
│   ├── models.py                 # Ridge / LogReg / XGBoost / neural net
│   ├── backtest.py               # season walk-forward evaluation + ensemble
│   └── predict.py                # predict upcoming games
├── scripts/                      # data pipeline
│   ├── scrape_day.py             # scrape one day of games
│   ├── scrape_season.py          # scrape full seasons
│   ├── scrape_common.py          # shared scraping helpers
│   └── build_dataset.py          # combine raw scrapes -> processed dataset
└── notebooks/
    └── nba_prediction_demo.ipynb # end-to-end narrative demo
```

## Installation

Developed and tested on **Python 3.14**.

```bash
git clone https://github.com/da-harshilv14/nba-score-predictor.git
cd nba-score-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the demo

```bash
jupyter lab notebooks/nba_prediction_demo.ipynb
```

It runs top to bottom on the committed sample (or the full dataset if present)
and shows the backtest, a confusion matrix, and sample predictions.

### Use the library

```python
import sys; sys.path.insert(0, "src")
from nba_predictor.data import load_games
from nba_predictor.features import build_features
from nba_predictor.models import build_models, select_features
from nba_predictor.predict import predict_games

feat = build_features(load_games("data/sample/games_sample.csv"))
model = build_models()["xgb"]

predict_games(feat, model, [("BOS", "LAL"), ("DEN", "GSW")])
```

### Build the dataset yourself

```bash
python scripts/scrape_season.py 2023 2024     # scrape -> data/raw/
python scripts/build_dataset.py                # combine + de-duplicate -> data/processed/games.csv
```

> Scraping hits Basketball Reference one page at a time with a delay to stay
> within its rate limits. Please review and respect their terms of use.

## Methodology

- **Features are leak-free.** Every feature is computed from past games (or the
  current one), never the future. The label for each row is the outcome of the
  team's *next* game.
- **Elo** rises/falls with each result and regresses toward 1500 between seasons.
- **Scaling** (for the linear models and NN) is fitted on the training fold only,
  inside the backtest, so the test set never influences it.
- **Validation** is strictly chronological (walk-forward by season).

## Future improvements

- Player-level stats, injuries, and rest/travel (back-to-backs).
- Margin-of-victory in the Elo update.
- Probability calibration and a high-confidence threshold for selective picks.
- Hyperparameter tuning (an Optuna pass) baked into the pipeline.

## Data

Data is scraped from [Basketball Reference](https://www.basketball-reference.com/)
for personal, educational use. No scraped data is redistributed beyond the small
sample in `data/sample/`; please follow Basketball Reference's terms of use.

**Reproducibility:** the repo ships only a 2-season sample so it runs out of the
box. The full results in [RESULTS.md](RESULTS.md) were computed on the complete
dataset (all seasons, 2017–2026); to reproduce them, rebuild the full dataset
with the scraping scripts (`scripts/scrape_season.py` → `scripts/build_dataset.py`,
which writes `data/processed/games.csv`). Running on the bundled sample alone
gives partial results.

## Acknowledgements

Inspired by, and adapted with help from,
[Pirkn/NBA-Game-Outcome-Prediction](https://github.com/Pirkn/NBA-Game-Outcome-Prediction).
This version was rebuilt from scratch with a modular package layout, a fixed and
de-duplicated data pipeline, leak-free feature engineering, and honest
season-based evaluation.

## Author

[@da-harshilv14](https://github.com/da-harshilv14)
