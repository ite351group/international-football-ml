# Model Training Supplement

This file documents the additional data and model-training work contributed to the project. It does not replace the existing project documentation.

## Data and validation

The training script uses the repository's existing `data/results.csv` file. It contains 49,945 valid matches from 1872-11-30 to 2026-10-06 in this run. The local input SHA-256 was `0b67e6a268a329892d288e03fa90000fac7566022d7bca9c59b104c3704d5716`.

Prediction samples are restricted to 1990 onwards (32,821 rows), matching the EDA scope. Earlier games initialise histories only. The test partition begins on 2019-11-19 and contains 6,609 matches; 26,212 earlier modern-period matches train the models. Dates are never shared between partitions. Earlier test results update history for later test dates; model weights stay fixed. This simulates sequential match-day prediction.

## Leakage controls and features

Each fixture receives features before its own result updates either team's state. Fixtures occurring on the same calendar date use the state from before that full date, so same-day results cannot leak into another prediction.

Features include prior win rate, points per game, goals for/against and goal difference per game, five- and ten-match form, rest days, directional head-to-head history, tournament, neutral venue, and a sequential Elo rating. Elo starts at 1500 for a new team, uses K=30, and applies a 60-point home advantage only for non-neutral fixtures.

## Models and actual results

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Majority-class reference | 0.477 | 0.333 | 0.215 |
| Logistic Regression | 0.608 | 0.509 | 0.465 |
| Random Forest | 0.599 | 0.511 | 0.480 |
| Extra Trees | 0.592 | **0.522** | **0.509** |

Macro-F1 and balanced accuracy are included because draws are less common and harder to predict than home wins. The full classification reports and confusion matrices are in `model_metrics.json`.

## Reproduce

From the repository root:

```powershell
pip install -r src/model_requirements.txt
python src/train.py
```

The executed `notebooks/02_feature_engineering_and_models.ipynb` includes data inspection, score-leakage assertions, training outputs, class reports and labelled confusion matrices. Run in Python 3.12 using the dedicated model requirements; the team's original root requirements remain unchanged. Actual versions are in run metadata.

Scores include extra time and exclude shootouts, so this is not strictly a 90-minute target. Elo expectancy is an expected result score, not a three-class win probability. This holdout is an exploratory benchmark: full-period EDA has been viewed, so it is not a pristine blind final test. Further tuning should use earlier rolling validation windows and reserve a new future period. No statistical significance claim is made. All hyperparameters are fixed; class weighting differs across models.

EDA review notes: venue statistics and the outcome figure use full history whereas later summaries use 1990+ matches; the goal histogram covers a limited range. These teammate files are preserved.
