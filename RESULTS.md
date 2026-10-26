# Results & Findings

All numbers come from **season-based walk-forward validation**: train on past
seasons, test on the next, never on future games. Dataset: 25,348 de-duplicated
team-games (2017–2026 seasons). The label is whether a team wins its *next*
game, so a coin-flip baseline is 50%.

## Headline

| Model / method        | Accuracy | Brier ↓ | Log-loss ↓ |
|-----------------------|:--------:|:-------:|:----------:|
| Baseline (coin flip)  |  50.0%   |    —    |     —      |
| Ridge                 |  60.3%   |  0.237  |   0.667    |
| Logistic Regression   |  60.3%   |  0.234  |   0.661    |
| XGBoost               |  60.2%   |  0.236  |   0.664    |
| Hard-vote ensemble    |  60.4%   |  0.234  |   0.660    |
| Stacked ensemble      |  60.3%   |  0.234  |   0.660    |

**Accuracy lands around 60%** and the **probabilities are well-calibrated**
(a predicted 70% wins ~70% of the time). For context, Vegas sits near 70% and
strong public models reach 65–68%, so ~60% leak-free is a respectable, honest
result — not an inflated one.

## Experiments

### 1. Rest / schedule features
Added rest days, back-to-back flags, and 7-day game density.

- **Real signal:** teams whose next game is a back-to-back win **44.6%** vs
  **51.1%** when rested — a genuine ~6.5-point fatigue effect.
- **But flat on accuracy:** the overall number didn't move. The feature only
  touches ~17% of games and the models already capture some of it. Kept anyway —
  it's principled and real.

### 2. Margin-of-victory Elo (538-style)
Scaled the Elo update by winning margin, with a favourite correction.

- **More realistic ratings:** the spread widened to ~1212–1837 (plain Elo was
  flatter), which makes the rating a more meaningful feature and a better number
  to display.
- **Flat on accuracy** again — the ordering of teams barely changes.

### 3. Calibration & proper metrics
Measured Brier score, log-loss, and a reliability curve.

- **Probabilities are already trustworthy** — predicted vs actual win rate sit
  on the diagonal across every confidence bucket.
- Explicitly calibrating (Platt/isotonic) changed Brier by 0.0001 — i.e. it
  wasn't needed. Good to know rather than assume.

### 4. Ensemble experiments
Compared hard vote, soft vote, weighted vote, and a stacked meta-model.

- **No meaningful lift** — everything sat inside a ~0.1% band.
- **Why:** the base models are too *similar* (all ~60%, correlated errors), so
  combining them cancels almost nothing. Ensembles help when models are diverse;
  these weren't.

## Takeaways

- The honest ceiling for this feature set is ~60% accuracy with trustworthy
  probabilities.
- Model choice and ensembling are **not** the bottleneck — the base models
  already agree. The bottleneck is **feature/data richness**.
- The clearest path to real gains is better inputs: player availability and
  injuries, rest/travel beyond back-to-backs, and (if allowed) market odds —
  not more models.
