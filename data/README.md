# Dataset Documentation

## 1. Dataset Source

The project uses the International Football Results dataset, obtained from Kaggle.

- **Dataset:** International Football Results
- **Kaggle identifier:** `martj42/international-football-results-from-1872-to-2017`
- **Historical coverage in the current local dataset:** 30 November 1872 – 6 October 2026
- **Primary modelling dataset:** `results.csv`

The Kaggle identifier reflects the dataset's original naming. The local dataset contains records extending through 2026.

## 2. Dataset Files

| Filename | Description |
|---|---|
| `results.csv` | Historical international football match results |
| `goalscorers.csv` | Individual goal-scoring records |
| `shootouts.csv` | Penalty shootout information |
| `former_names.csv` | Historical team names and name changes |

## 3. Primary Dataset

The main analysis and machine learning pipeline uses `results.csv`.

Important columns include:

- `date` – Match date.
- `home_team` – Home team.
- `away_team` – Away team.
- `home_score` – Home team goals.
- `away_score` – Away team goals.
- `tournament` – Tournament or match competition.
- `city` – Match location.
- `country` – Host country.
- `neutral` – Whether the match was played at a neutral venue.

The target variable is derived from the match score and represents three possible outcomes:

- Home win.
- Draw.
- Away win.

## 4. Data Preparation and Quality Checks

The modelling pipeline:
- Checks required columns and valid match records.
- Uses matches from 1990 onwards for model training and evaluation.
- Uses earlier historical matches to initialise team performance histories.
- Constructs pre-match features before updating team histories with match results.
- Creates same-day pre-match snapshots to prevent information from one match on a date leaking into predictions for other matches on that date.

The source-data audit identified one repeated date/team fixture key with conflicting scores: Tahiti vs New Caledonia on 17 February 1974. Both records are preserved pending verification of the historical dates and original source.

No exact duplicate rows were identified in the audited dataset.

## 5. Dataset Integrity

The local modelling run used:
- 49,945 valid match records.
- Date range: 1872-11-30 to 2026-10-