# Predicting International Football Match Outcomes Using Machine Learning

## Project Overview

This project investigates the use of machine learning to predict international football match outcomes using historical match data. It explores how historical team performance, recent form, Elo ratings and match context influence the prediction of home wins, draws and away wins.

## Research Question

How accurately can machine learning models predict international football match outcomes using historical team performance and match context?

## Dataset

The project uses the International Football Results dataset, obtained from Kaggle.

- **Coverage:** 1872–2026
- **Total records:** 49,945 international matches
- **Primary dataset:** `data/results.csv`
- **Additional datasets:** `goalscorers.csv`, `shootouts.csv` and `former_names.csv`

The main modelling pipeline uses `results.csv`. The other datasets are available for supplementary analysis.

## Project Objectives

- Explore historical international football match trends.
- Investigate home advantage and differences between friendly and competitive matches.
- Examine the predictive value of recent team form and historical performance.
- Develop machine learning models to predict home wins, draws and away wins.
- Compare classification performance using accuracy, balanced accuracy, macro-F1 and class-specific metrics.
- Investigate the challenges of predicting draws and handling class imbalance.

## Methodology

### 1. Exploratory Data Analysis
Historical match records are analysed to investigate:
- Match outcome distributions.
- Goal-scoring patterns.
- Home advantage.
- Tournament and match-type distributions.
- Historical team performance.

### 2. Feature Engineering

Pre-match predictive features include:
- Historical win rate and points per game.
- Goals scored and conceded per game.
- Recent five-match and ten-match form.
- Rest days between matches.
- Head-to-head historical performance.
- Sequential Elo ratings.
- Tournament and neutral-venue information.

Features are constructed using information available before each match to reduce data leakage.

### 3. Machine Learning Models

The project compares four classification approaches:

- Majority-class baseline.
- Logistic Regression.
- Random Forest.
- Extra Trees.

Models are evaluated using a chronological training and testing split to better reflect real-world match prediction.

### 4. Model Evaluation

Evaluation metrics include:
- Accuracy.
- Balanced accuracy.
- Macro-F1.
- Precision and recall.
- Confusion matrices.
- Log loss, Brier score and Ranked Probability Score.

Additional experiments investigate feature contributions, chronological validation and class weighting to improve draw prediction.

## Main Holdout Results

The following results are from the chronological holdout beginning on 19 November 2019.

| Model | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Majority-class baseline | 47.72% | 33.33% | 21.54% |
| Logistic Regression | **60.84%** | 50.92% | 46.48% |
| Random Forest | 59.90% | 51.05% | 48.03% |
| Extra Trees | 59.22% | **52.25%** | **50.87%** |

Logistic Regression achieves the highest overall accuracy, while Extra Trees achieves the highest balanced accuracy and macro-F1. This highlights the importance of considering multiple evaluation metrics when predicting imbalanced match outcomes.

Draw prediction remains a major challenge across all models.

## Further Experiments

Additional experiments are documented separately from the main holdout results:

- **Feature ablation:** Investigates the contribution of recent form, Elo and other historical features.
- **Chronological validation:** Evaluates models across three earlier historical validation windows.
- **Class weighting:** Examines the trade-off between overall accuracy and draw recall.
- **Probability evaluation:** Compares probabilistic predictions using log loss, Brier score and Ranked Probability Score.

These experiments provide further insight into model behaviour without replacing the published main holdout results.

## Repository Structure

- `proposal/` – Project proposal and research idea.
- `data/` – Dataset files and dataset documentation.
- `notebooks/` – Exploratory data analysis, feature engineering and model experiments.
- `src/` – Model training and supporting source code.
- `results/` – Model metrics, visualisations, probability evaluation and validation experiments.

## Reproducing the Model

Use Python 3.12 and install the dedicated model dependencies from the repository root:

```powershell
pip install -r src/model_requirements.txt
```

Run the training pipeline:

```powershell
python src/train.py
```

For the separate chronological feature-validation experiment:

```powershell
python results/validation/defence_experiments.py --repo . --output results/validation
```

Refer to the individual README files in `data/`, `src/` and `results/` for further details.

## Limitations

- Draws are considerably harder to predict than home wins and away wins.
- The main holdout has already been examined and should be treated as an exploratory benchmark, not a pristine blind test.
- Historical data may contain uncertain dates or conflicting fixture records.
- Match scores include extra time and exclude penalty shootouts; the target therefore does not represent strictly 90-minute outcomes.
- Results should not be interpreted as proof of statistical significance or guaranteed future predictive performance.