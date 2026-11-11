"""Render presentation figures from saved, measured model reports (no retraining).

Python 3.12: pip install -r src/model_requirements.txt matplotlib==3.10.8
Run from any directory: python src/plot_model_results.py
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results' / 'model_figures'
LABELS = ['Home win', 'Draw', 'Away win']
KEYS = ['home_win', 'draw', 'away_win']
NAMES = {'majority_class_dummy': 'Majority baseline', 'logistic_regression': 'Logistic regression', 'random_forest': 'Random forest', 'extra_trees': 'Extra Trees'}
COLORS = ['#96A0AF', '#2563EB', '#0F9488', '#E58B27']


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    reports = json.loads((ROOT / 'results/model_metrics.json').read_text())
    meta = json.loads((ROOT / 'results/model_run_metadata.json').read_text())
    models = list(NAMES)
    for name in models:
        matrix = np.asarray(reports[name]['confusion_matrix'])
        assert matrix.shape == (3, 3) and matrix.sum() == meta['test_rows']
        assert abs(matrix.trace() / matrix.sum() - reports[name]['accuracy']) < 1e-6
        for i, key in enumerate(KEYS):
            assert matrix[i].sum() == reports[name]['classification_report'][key]['support']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold', 'figure.facecolor': '#F7F9FC', 'axes.facecolor': '#F7F9FC', 'savefig.facecolor': '#F7F9FC'})
    scope = f"1990+ samples | Test: {meta['cutoff_date']} to {meta['date_max']} | n={meta['test_rows']:,}"

    def save(fig, name, title, note):
        fig.suptitle(title, x=.06, ha='left', fontsize=19, fontweight='bold', y=.98)
        fig.text(.06, .90, scope, fontsize=10, color='#526074')
        fig.text(.06, .025, note, fontsize=10, color='#526074')
        fig.savefig(DEST / (name + '.png'), dpi=190, bbox_inches='tight')
        fig.savefig(DEST / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.subplots_adjust(top=.79, bottom=.20, left=.06, right=.98, wspace=.30)
    for ax, metric, title in zip(axes, ['accuracy','balanced_accuracy','macro_f1'], ['Accuracy', 'Balanced accuracy', 'Macro-F1']):
        values = [reports[m][metric] for m in models]
        ax.barh(range(4), values, color=COLORS, height=.62)
        ax.set_yticks(range(4), [NAMES[m] for m in models] if metric == 'accuracy' else [''] * 4)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(title, loc='left', pad=14)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        for y, value in enumerate(values): ax.text(value+.018, y, f'{value:.3f}', va='center', fontsize=11)
    save(fig, '01_model_comparison', 'A higher accuracy does not mean better class balance', 'Same holdout for every model. Settings are fixed; this is an exploratory benchmark, not a significance test.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    fig.subplots_adjust(top=.77, bottom=.20, left=.07, right=.95, wspace=.36)
    for ax, model in zip(axes, models[1:]):
        cm = np.asarray(reports[model]['confusion_matrix'])
        rates = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(rates, vmin=0, vmax=1, cmap='Blues')
        ax.set_xticks(range(3), LABELS, rotation=25, ha='right')
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel('Predicted outcome')
        ax.set_title(NAMES[model], fontsize=13, pad=12)
        for i in range(3):
            for j in range(3): ax.text(j, i, f'{cm[i,j]:,}\n{rates[i,j]:.1%}', ha='center', va='center', color='white' if rates[i,j] > .55 else '#102944', fontsize=11)
    axes[0].set_ylabel('Actual outcome')
    save(fig, '02_confusion_matrices', 'Most real draws are classified as wins', 'Cell labels: match count and percentage within the actual class. Shared colour scale: 0% to 100%.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.subplots_adjust(top=.77, bottom=.25, left=.08, right=.98, wspace=.30)
    x = np.arange(3)
    for ax, metric, title in zip(axes, ['recall','f1-score'], ['Recall: share of each class recovered', 'F1: precision-recall balance']):
        for j, model in enumerate(models[1:]):
            values = [reports[model]['classification_report'][k][metric] for k in KEYS]
            bars = ax.bar(x+(j-1)*.24, values, width=.23, label=NAMES[model], color=COLORS[j+1])
            ax.bar_label(bars, labels=[f'{v*100:.1f}' if metric == 'recall' else f'{v:.3f}' for v in values], fontsize=8, padding=4)
        ax.set_xticks(x, LABELS)
        ax.set_ylim(0, 1)
        if metric == 'recall': ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title, fontsize=12, loc='left')
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0,-.16), ncol=3, fontsize=9, frameon=False)
    save(fig, '03_per_class_performance', 'Draw prediction remains the largest weakness', 'Class weights differ across models. Extra Trees recovers more draws, but draw recall is still only about 16%.')

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(top=.79, bottom=.24, left=.20, right=.97)
    left = np.zeros(3)
    for j in range(3):
        values = [np.asarray(reports[m]['confusion_matrix'])[1,j] for m in models[1:]]
        ax.barh(range(3), values, left=left, color=[COLORS[1], '#E58B27', COLORS[2]][j], label='Predicted '+LABELS[j].lower(), height=.55)
        for i,v in enumerate(values): ax.text(left[i]+v/2, i, str(v), ha='center', va='center', color='white', weight='bold')
        left += values
    ax.set_yticks(range(3), [NAMES[m] for m in models[1:]])
    ax.invert_yaxis()
    ax.set_xlim(0, int(left[0]))
    ax.set_xlabel('Number of matches that actually ended in a draw')
    ax.legend(loc='upper left', bbox_to_anchor=(0,-.20), ncol=3, frameon=False, fontsize=10)
    save(fig, '04_draw_errors', 'Where do the 1,507 real draws go?', 'These are counts for the same actual-draw subset. Correct draw predictions: LR 51, RF 113, Extra Trees 242.')

    data = ROOT / 'data/results.csv'
    assert hashlib.sha256(data.read_bytes()).hexdigest() == meta['input_sha256'], 'Dataset differs from the measured run.'
    training, testing, warmup = Counter(), Counter(), 0
    with data.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            d = row['date']
            if d < '1990-01-01': warmup += 1
            elif d < meta['cutoff_date']: training[int(d[:4])] += 1
            else: testing[int(d[:4])] += 1
    assert sum(training.values()) == meta['train_rows'] and sum(testing.values()) == meta['test_rows']
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.subplots_adjust(top=.78, bottom=.23, left=.07, right=.98)
    years = np.arange(1990, int(meta['date_max'][:4])+1)
    train_values = [training[y] for y in years]
    ax.bar(years, train_values, color=COLORS[1], label=f"Train: {meta['train_rows']:,}")
    ax.bar(years, [testing[y] for y in years], bottom=train_values, color=COLORS[3], label=f"Test: {meta['test_rows']:,}")
    ax.set_ylabel('Matches')
    ax.set_xlabel('Match year (2019 contains both partitions, separated by exact date)')
    ax.legend(frameon=False, loc='upper left', ncol=2)
    ax.set_xticks(np.arange(1990,2027,4))
    save(fig, '05_chronological_split', 'Train on the past, evaluate on later match dates', f"{warmup:,} pre-1990 matches initialise histories only. Earlier test outcomes update later histories; weights stay fixed.")
    print('Saved five figures as PNG and SVG:', DEST)


if __name__ == '__main__':
    main()
