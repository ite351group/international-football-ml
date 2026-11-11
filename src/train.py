"""Leakage-free chronological football outcome models for the ITE351 project."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import platform

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "results.csv"
OUTPUT = ROOT / "results"
TARGETS = ["home_win", "draw", "away_win"]
NUMERIC = [
    "home_games", "away_games", "home_win_rate", "away_win_rate", "home_ppg", "away_ppg",
    "home_gdpg", "away_gdpg", "home_gfpg", "away_gfpg", "home_gapg", "away_gapg",
    "home_form5", "away_form5", "home_form10", "away_form10", "home_rest_days", "away_rest_days",
    "win_rate_diff", "ppg_diff", "gdpg_diff", "gfpg_diff", "gapg_diff", "form5_diff", "form10_diff", "rest_diff",
    "h2h_games", "h2h_home_ppg", "h2h_home_win_rate",
    "home_elo", "away_elo", "elo_diff", "elo_home_expectancy",
]
CATEGORICAL = ["tournament", "neutral"]


def div(a, b):
    return a / b if b else float("nan")


def result(home, away):
    return "home_win" if home > away else "away_win" if home < away else "draw"


def initial_team():
    return {"games": 0, "wins": 0, "draws": 0, "gf": 0, "ga": 0, "recent": deque(maxlen=10), "last": None}


def team_features(state, match_date):
    recent = list(state["recent"])
    five = recent[-5:]
    g = state["games"]
    return {
        "games": float(g), "win_rate": div(state["wins"], g), "ppg": div(3 * state["wins"] + state["draws"], g),
        "gdpg": div(state["gf"] - state["ga"], g), "gfpg": div(state["gf"], g), "gapg": div(state["ga"], g),
        "form5": div(sum(x["points"] for x in five), len(five)),
        "form10": div(sum(x["points"] for x in recent), len(recent)),
        "rest_days": float((match_date - state["last"]).days) if state["last"] is not None else float("nan"),
    }


def update_team(state, goals_for, goals_against, date):
    state["games"] += 1
    state["gf"] += goals_for
    state["ga"] += goals_against
    points = 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
    if points == 3:
        state["wins"] += 1
    elif points == 1:
        state["draws"] += 1
    state["recent"].append({"points": points})
    state["last"] = date


def elo_expectancy(home, away, neutral):
    adjusted_home = home if neutral else home + 60.0
    return 1 / (1 + 10 ** ((away - adjusted_home) / 400))


def build_features(matches):
    teams = defaultdict(initial_team)
    h2h = defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0})
    elo = defaultdict(lambda: 1500.0)
    rows = []
    for _, games_today in matches.sort_values(["date", "source_row"], kind="stable").groupby("date", sort=False):
        snapshots = []
        # Read all states first: results from this date cannot become features.
        for m in games_today.itertuples(index=False):
            home, away = team_features(teams[m.home_team], m.date), team_features(teams[m.away_team], m.date)
            history = dict(h2h[(m.home_team, m.away_team)])
            helo, aelo = elo[m.home_team], elo[m.away_team]
            snapshots.append((m, home, away, history, helo, aelo, elo_expectancy(helo, aelo, bool(m.neutral))))
        for m, home, away, history, helo, aelo, expected in snapshots:
            row = {"date": m.date, "source_row": m.source_row, "tournament": m.tournament, "neutral": str(m.neutral), "target": result(m.home_score, m.away_score)}
            for key, value in home.items(): row["home_" + key] = value
            for key, value in away.items(): row["away_" + key] = value
            for key in ["win_rate", "ppg", "gdpg", "gfpg", "gapg", "form5", "form10"]: row[key + "_diff"] = home[key] - away[key]
            row["rest_diff"] = home["rest_days"] - away["rest_days"]
            row["h2h_games"] = history["games"]
            row["h2h_home_ppg"] = div(3 * history["wins"] + history["draws"], history["games"])
            row["h2h_home_win_rate"] = div(history["wins"], history["games"])
            row.update({"home_elo": helo, "away_elo": aelo, "elo_diff": helo - aelo, "elo_home_expectancy": expected})
            rows.append(row)
        changes = defaultdict(float)
        for m, _, _, _, helo, aelo, expected in snapshots:
            actual = 1.0 if m.home_score > m.away_score else 0.5 if m.home_score == m.away_score else 0.0
            delta = 30.0 * (actual - expected)
            changes[m.home_team] += delta
            changes[m.away_team] -= delta
        for team, change in changes.items(): elo[team] += change
        for m, _, _, _, _, _, _ in snapshots:
            update_team(teams[m.home_team], int(m.home_score), int(m.away_score), m.date)
            update_team(teams[m.away_team], int(m.away_score), int(m.home_score), m.date)
            for key, gf, ga in [((m.home_team, m.away_team), int(m.home_score), int(m.away_score)), ((m.away_team, m.home_team), int(m.away_score), int(m.home_score))]:
                h2h[key]["games"] += 1
                if gf > ga: h2h[key]["wins"] += 1
                elif gf == ga: h2h[key]["draws"] += 1
    return pd.DataFrame(rows)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def metrics(y_true, y_pred):
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 6), "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6), "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6), "classification_report": classification_report(y_true, y_pred, labels=TARGETS, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(y_true, y_pred, labels=TARGETS).tolist()}


def main():
    raw = pd.read_csv(INPUT)
    required = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    raw["source_row"] = range(len(raw))
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["home_score"] = pd.to_numeric(raw["home_score"], errors="coerce")
    raw["away_score"] = pd.to_numeric(raw["away_score"], errors="coerce")
    raw["neutral"] = raw["neutral"].astype(str).str.lower().map({"true": True, "false": False})
    clean = raw.dropna(subset=required).copy()
    clean[["home_score", "away_score"]] = clean[["home_score", "away_score"]].astype(int)
    features = build_features(clean)
    ordered = features.loc[features.date >= "1990-01-01"].sort_values(["date", "source_row"], kind="stable").reset_index(drop=True)
    cutoff = ordered.loc[int(len(ordered) * 0.8), "date"]
    train, test = ordered[ordered.date < cutoff], ordered[ordered.date >= cutoff]
    preprocessor = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC), ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)])
    x_train, y_train = train[NUMERIC + CATEGORICAL], train.target
    x_test, y_test = test[NUMERIC + CATEGORICAL], test.target
    models = {"majority_class_dummy": DummyClassifier(strategy="most_frequent"), "logistic_regression": LogisticRegression(max_iter=2000, random_state=42), "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42), "extra_trees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)}
    report = {}
    for name, estimator in models.items():
        print("Training:", name, flush=True)
        model = Pipeline([("features", preprocessor), ("model", estimator)])
        model.fit(x_train, y_train)
        report[name] = metrics(y_test, model.predict(x_test))
    metadata = {"source": "Kaggle: martj42/international-football-results-from-1872-to-2017", "input_sha256": digest(INPUT), "run_utc": datetime.now(timezone.utc).isoformat(), "rows_raw": len(raw), "rows_clean": len(clean), "date_min": str(clean.date.min().date()), "date_max": str(clean.date.max().date()), "cutoff_date": str(cutoff.date()), "train_rows": len(train), "test_rows": len(test), "features": "pre-match form, rest, directional head-to-head, Elo, tournament, neutral venue"}
    metadata.update({"model_rows": len(ordered), "model_start": "1990-01-01", "python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__, "evaluation": "Sequential pre-match prediction: earlier test outcomes update histories; model weights stay fixed.", "confusion_matrix_labels": TARGETS})
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT / "model_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": name, "accuracy": value["accuracy"], "balanced_accuracy": value["balanced_accuracy"], "macro_f1": value["macro_f1"]} for name, value in report.items()]).to_csv(OUTPUT / "model_metrics_summary.csv", index=False)
    print(json.dumps({"metadata": metadata, "summary": {name: {metric: value[metric] for metric in ["accuracy", "balanced_accuracy", "macro_f1"]} for name, value in report.items()}}, indent=2))


if __name__ == "__main__":
    main()
