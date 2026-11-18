"""Reproduce the published holdout and add per-match probability diagnostics.

Standalone supplement: it imports the group training module but writes only to
the selected output directory. No model selection or threshold tuning uses the
holdout. Example:

    python probability_eval.py --repo path/to/international-football-ml --output path/to/preview
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ORDER = ["away_win", "draw", "home_win"]  # ordinal order used for RPS
MODELS = {
    "majority_class_dummy": lambda: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    ),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=42,
    ),
}


def load_group_train(repo: Path):
    spec = importlib.util.spec_from_file_location("group_train", repo / "src" / "train.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import group src/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_matches(repo: Path, group):
    source = repo / "data" / "results.csv"
    raw = pd.read_csv(source)
    raw["source_row"] = range(len(raw))
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["home_score"] = pd.to_numeric(raw["home_score"], errors="coerce")
    raw["away_score"] = pd.to_numeric(raw["away_score"], errors="coerce")
    raw["neutral"] = raw["neutral"].astype(str).str.lower().map({"true": True, "false": False})
    required = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]
    clean = raw.dropna(subset=required).copy()
    clean[["home_score", "away_score"]] = clean[["home_score", "away_score"]].astype(int)
    features = group.build_features(clean)
    ordered = features.loc[features.date >= "1990-01-01"].sort_values(
        ["date", "source_row"], kind="stable"
    ).reset_index(drop=True)
    cutoff = ordered.loc[int(len(ordered) * .8), "date"]
    train = ordered[ordered.date < cutoff]
    test = ordered[ordered.date >= cutoff]
    return source, raw, clean, train, test, cutoff


def make_preprocessor(group):
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), group.NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), group.CATEGORICAL),
    ])


def scores(y, proba):
    onehot = (np.asarray(y)[:, None] == np.asarray(ORDER)[None, :]).astype(float)
    brier_each = np.square(proba - onehot).sum(axis=1)  # multiclass sum, 0..2
    rps_each = np.square(
        np.cumsum(proba, axis=1)[:, :2] - np.cumsum(onehot, axis=1)[:, :2]
    ).mean(axis=1)  # mean across K-1=2 thresholds, 0..1
    true_idx = onehot.argmax(axis=1)
    log_each = -np.log(np.clip(proba[np.arange(len(y)), true_idx], 1e-15, 1.0))
    return brier_each, rps_each, log_each


def metric_block(y, pred, proba):
    brier, rps, log_each = scores(y, proba)
    draw = np.asarray(y) == "draw"
    draw_prob = proba[:, ORDER.index("draw")]
    return {
        "matches": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=ORDER, average="macro", zero_division=0)),
        "draw_recall": float(recall_score(y, pred, labels=["draw"], average=None, zero_division=0)[0]),
        "actual_draw_rate": float(draw.mean()),
        "predicted_draw_rate": float((np.asarray(pred) == "draw").mean()),
        "mean_predicted_draw_probability": float(draw_prob.mean()),
        "mean_predicted_draw_probability_on_actual_draws": float(draw_prob[draw].mean()),
        "log_loss": float(log_loss(y, proba, labels=ORDER)),
        "multiclass_brier_sum": float(brier.mean()),
        "rps_normalized": float(rps.mean()),
    }


def draw_bins(frame):
    rows = []
    for name in MODELS:
        prob = frame[f"{name}_p_draw"].to_numpy()
        actual = (frame.actual == "draw").to_numpy()
        for left in np.arange(0, 1, .1):
            right = round(float(left) + .1, 1)
            mask = (prob >= left) & ((prob < right) if right < 1 else (prob <= right))
            rows.append({
                "model": name, "probability_lower": round(float(left), 1),
                "probability_upper": right, "n": int(mask.sum()),
                "mean_predicted_draw_probability": float(prob[mask].mean()) if mask.any() else np.nan,
                "observed_draw_rate": float(actual[mask].mean()) if mask.any() else np.nan,
            })
    return pd.DataFrame(rows)


def macro_f1_from_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(axis=-2) + cm.sum(axis=-1)
    return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0).mean(axis=-1)


def month_block_bootstrap(frame, repeats=5000, seed=42):
    """Paired resampling of calendar-month blocks; descriptive, not a blind test."""
    months = frame.date.dt.to_period("M")
    keys = months.drop_duplicates().tolist()
    y = frame.actual.to_numpy()
    cm = np.zeros((len(keys), 2, 3, 3), dtype=np.int64)
    correct = np.zeros((len(keys), 2), dtype=np.int64)
    n = np.zeros(len(keys), dtype=np.int64)
    for i, month in enumerate(keys):
        mask = (months == month).to_numpy()
        n[i] = mask.sum()
        for j, name in enumerate(["logistic_regression", "extra_trees"]):
            pred = frame.loc[mask, f"{name}_predicted"].to_numpy()
            cm[i, j] = confusion_matrix(y[mask], pred, labels=ORDER)
            correct[i, j] = (y[mask] == pred).sum()
    assert cm.sum(axis=(0, 2, 3)).tolist() == [len(frame), len(frame)]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(repeats, len(keys)))
    boot_cm = cm[picks].sum(axis=1)
    macro_delta = macro_f1_from_cm(boot_cm[:, 1]) - macro_f1_from_cm(boot_cm[:, 0])
    boot_n = n[picks].sum(axis=1)
    acc_delta = (correct[picks, 1].sum(axis=1) - correct[picks, 0].sum(axis=1)) / boot_n
    point_macro = float(macro_f1_from_cm(cm[:, 1].sum(axis=0)) - macro_f1_from_cm(cm[:, 0].sum(axis=0)))
    point_acc = float((correct[:, 1].sum() - correct[:, 0].sum()) / len(frame))
    return {
        "comparison": "extra_trees minus logistic_regression",
        "resampling_unit": "calendar month (paired, with replacement)",
        "month_blocks": len(keys), "replicates": repeats, "seed": seed,
        "macro_f1_difference": {
            "point": point_macro,
            "percentile_95_interval": np.quantile(macro_delta, [.025, .975]).tolist(),
        },
        "accuracy_difference": {
            "point": point_acc,
            "percentile_95_interval": np.quantile(acc_delta, [.025, .975]).tolist(),
        },
        "interpretation_limit": "Exploratory uncertainty for this already inspected holdout, not a confirmatory significance test. Months are not fully independent, and the models have different class weights.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    group = load_group_train(repo)
    source, raw, clean, train, test, cutoff = read_matches(repo, group)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    published_meta = json.loads((repo / "results" / "model_run_metadata.json").read_text(encoding="utf-8"))
    assert source_sha == published_meta["input_sha256"], "Dataset differs from published run"
    assert (len(raw), len(clean), len(train), len(test), str(cutoff.date())) == (
        published_meta["rows_raw"], published_meta["rows_clean"],
        published_meta["train_rows"], published_meta["test_rows"], published_meta["cutoff_date"],
    ), "Split differs from published run"
    x_train, x_test = train[group.NUMERIC + group.CATEGORICAL], test[group.NUMERIC + group.CATEGORICAL]
    y_train, y_test = train.target, test.target
    raw_by_row = raw.set_index("source_row")
    detail = pd.DataFrame({
        "source_row": test.source_row.to_numpy(), "date": test.date.to_numpy(),
        "home_team": raw_by_row.loc[test.source_row, "home_team"].to_numpy(),
        "away_team": raw_by_row.loc[test.source_row, "away_team"].to_numpy(),
        "actual": y_test.to_numpy(),
    })
    published = json.loads((repo / "results" / "model_metrics.json").read_text(encoding="utf-8"))
    metrics = {}
    for name, factory in MODELS.items():
        print("Training", name, flush=True)
        model = Pipeline([("features", make_preprocessor(group)), ("model", factory())])
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        classes = model.named_steps["model"].classes_
        raw_proba = model.predict_proba(x_test)
        proba = raw_proba[:, [list(classes).index(label) for label in ORDER]]
        assert np.allclose(proba.sum(axis=1), 1, atol=1e-10)
        assert np.isfinite(proba).all()
        block = metric_block(y_test, pred, proba)
        assert abs(block["accuracy"] - published[name]["accuracy"]) < 1e-6
        assert abs(block["macro_f1"] - published[name]["macro_f1"]) < 1e-6
        assert confusion_matrix(y_test, pred, labels=group.TARGETS).tolist() == published[name]["confusion_matrix"]
        detail[f"{name}_predicted"] = pred
        for col, label in enumerate(ORDER):
            detail[f"{name}_p_{label}"] = proba[:, col]
        metrics[name] = block
    detail.to_csv(out / "holdout_predictions.csv", index=False, float_format="%.12g", date_format="%Y-%m-%d")
    draw_bins(detail).to_csv(out / "draw_calibration_bins.csv", index=False, float_format="%.8g")
    (out / "probability_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    bootstrap = month_block_bootstrap(detail)
    (out / "paired_block_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    metadata = {
        "source_sha256": source_sha, "published_split_cutoff": str(cutoff.date()),
        "train_rows": len(train), "holdout_rows": len(test),
        "python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "models": "Exact estimators and preprocessing in group src/train.py; group module imports feature builder. Published accuracy, macro-F1 and confusion matrices asserted equal before saving.",
        "probability_class_order": ORDER,
        "brier_definition": "Mean across matches of sum over three classes of (probability - onehot)^2; 0..2.",
        "rps_definition": "Mean across matches of mean squared cumulative-probability error at away_win and draw thresholds; ORDER away_win, draw, home_win; 0..1.",
        "holdout_reuse": "Published 2019-11-19+ holdout has already been inspected. This is descriptive analysis only; no model/threshold selection or calibration fit.",
    }
    (out / "probability_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "bootstrap": bootstrap}, indent=2), flush=True)


if __name__ == "__main__":
    main()
