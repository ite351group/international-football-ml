# Results and Findings

## Overview

This folder contains the results generated from exploratory data analysis, machine learning experiments, model evaluation and supplementary validation.

The experiments investigate historical international football trends and compare different approaches to predicting home wins, draws and away wins.

## 1. Exploratory Data Analysis

The `figures/` directory contains visualisations of historical match trends:

- `goal_distribution.png` – Distribution of goals scored.
- `home_advantage.png` – Historical home advantage.
- `match_type_distribution.png` – Distribution of match types.
- `outcome_distribution.png` – Distribution of match outcomes.
- `top_teams.png` – Historical performance of leading teams.
- `tournament_distribution.png` – Distribution of tournament categories.

These figures provide context for understanding historical match patterns and potential predictive features.

## 2. Main Model Evaluation

The main model evaluation uses a chronological training and testing split.

The test period begins on 19 November 2019 and contains 6,609 matches.

### Model Comparison

| Model | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Majority-class baseline | 47.72% | 33.33% | 21.54% |
| Logistic Regression | **60.84%** | 50.92% | 46.48% |
| Random Forest | 59.90% | 51.05% | 48.03% |
| Extra Trees | 59.22% | **52.25%** | **50.87%** |

Logistic Regression achieves the highest accuracy, while Extra Trees performs best in balanced accuracy and macro-F1.

The results suggest that evaluating overall accuracy alone is insufficient because the three match outcomes are not equally represented.

### Model Visualisations

The `model_figures/` directory contains:

- `01_model_comparison.png` – Comparison of model performance.
- `02_confusion_matrices.png` – Classification errors across the three outcomes.
- `03_per_class_performance.png` – Precision, recall and F1-score for each outcome.
- `04_draw_errors.png` – Further examination of draw prediction errors.
- `05_chronological_split.png` – Visualisation of the chronological evaluation split.

SVG versions are also available for higher-resolution viewing.

## 3. Probability Evaluation

The `probability/` directory contains supplementary probability-based evaluation.

It evaluates:
- Log loss.
- Multiclass Brier score.
- Ranked Probability Score.
- Draw probability calibration.
- Predicted versus actual draw frequencies.

Logistic Regression achieves the best probability scores among the three machine learning models, while Extra Trees achieves the strongest macro-F1 and draw recall on the main holdout.

The probability experiments do not replace the original classification results.

## 4. Feature Ablation and Chronological Validation

The `validation/` directory contains a separate experiment investigating whether historical team performance and pre-match features improve prediction.

Three expanding chronological validation windows are used.

### Main Findings

- Adding recent five-match and ten-match form improves mean macro-F1 from approximately 0.216 to 0.368.
- Adding pre-match Elo increases mean macro-F1 further to approximately 0.435.
- The full feature set achieves approximately 0.450 mean macro-F1.
- Applying balanced class weights to Logistic Regression increases draw recall to approximately 30%, while reducing accuracy.

These results suggest that recent team form and Elo provide useful predictive information. However, improving draw recall involves a trade-off with overall accuracy.

The validation experiments are separate from the published main holdout evaluation.

## 5. Key Findings and Discussion

The experiments indicate that:

1. Historical team performance, recent form and Elo ratings provide useful information for predicting international football match outcomes.
2. Logistic Regression achieves the highest accuracy among the evaluated models.
3. Extra Trees achieves the highest balanced accuracy and macro-F1 on the main holdout.
4. Draws remain difficult to predict because of class imbalance and the complexity of football match outcomes.
5. Class weighting can improve draw detection, but may reduce overall accuracy.
6. Probability evaluation provides additional information about model confidence beyond classification accuracy.

## 6. Limitations

- The main chronological holdout has already been examined and is not an untouched blind test.
- The three historical validation windows provide descriptive averages rather than formal statistical significance tests.
- Draw prediction remains relatively weak.
- Historical fixture data may contain uncertain dates or conflicting records.
- Match scores include extra time and exclude penalty shootouts.
- The experiments do not establish guaranteed performance on future international matches.

## 7. Reproducibility

Main model training:

```powershell
python src/train.py
```

Probability evaluation and supplementary analysis are documented in `probability/README.md`.

Feature ablation and chronological validation:

```powershell
python results/validation/defence_experiments.py --repo . --output results/validation
```

Refer to `model_methodology.md` for the main model methodology and `model_metrics.json` for detailed classification results.