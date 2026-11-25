# Source Code Documentation

## 1. Overview

This folder contains the Python source code used to prepare historical international football match data, construct predictive features, train classification models and evaluate their performance.

The main objective is to predict three match outcomes:
- Home win
- Draw
- Away win

## 2. Source Files

| Filename | Description |
|---|---|
| `train.py` | Main machine learning pipeline, including data preparation, feature engineering, model training and evaluation |
| `plot_model_results.py` | Generates visualisations of model evaluation results |
| `model_requirements.txt` | Dedicated Python dependencies for the modelling environment |

## 3. Data Processing and Feature Engineering

The training pipeline reads `data/results.csv` and prepares historical match records.

Predictive features include:

- Historical team win rates and points per game.
- Goals scored and conceded per game.
- Recent five-match and ten-match form.
- Rest days between matches.
- Historical head-to-head performance.
- Sequential Elo ratings.
- Tournament and neutral-venue information.

Features are calculated using information available before each match. Matches on the same date use pre-date team histories to avoid same-day result leakage.

## 4. Machine Learning Models

The pipeline trains and compares four classification approaches:

1. Majority-class dummy baseline.
2. Logistic Regression.
3. Random Forest.
4. Extra Trees.

The models are evaluated using a chronological training and testing split.

## 5. Model Evaluation

The training pipeline produces:

- Accuracy.
- Balanced accuracy.
- Macro-F1.
- Classification reports.
- Confusion matrices.
- Model run metadata.

The generated metrics and visualisations are saved under `results/`.

## 6. Dependencies

The modelling environment uses Python 3.12.

Install the dedicated dependencies from the repository root:

```powershell
pip install -r src/model_requirements.txt
```

The requirements file specifies the tested versions of pandas, NumPy, scikit-learn, SciPy, joblib and threadpoolctl.

Additional plotting functionality may require Matplotlib.

## 7. Running the Training Pipeline

From the repository root:

```powershell
python src/train.py
```

To generate model result visualisations:

```powershell
python src/plot_model_results.py
```

Refer to the script and results documentation for the exact inputs and outputs.

## 8. Reproducibility and Limitations

The training pipeline uses fixed model configurations and a chronological split beginning on 19 November 2019 for the main holdout evaluation.

The main holdout is exploratory because the broader historical dataset has already been examined during exploratory analysis.

Further feature validation and class-weight experiments are implemented separately under `results/validation/`.