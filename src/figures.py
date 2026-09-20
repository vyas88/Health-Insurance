"""Figures are derived only from saved frozen holdout predictions."""
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score
from .data import ROOT, CLASSES

COLORS = ['#2878a5', '#d69b26', '#228576']


def main():
    result = json.loads((ROOT / 'outputs/results.json').read_text())
    pred = pd.read_csv(ROOT / 'outputs/evaluation_predictions.csv')
    if set(pred.run_id) != {result['run_id']}:
        raise ValueError('Prediction run mismatch')
    output = ROOT / 'outputs/figures'
    output.mkdir(exist_ok=True)
    files = {}
    def save(fig, name):
        fig.text(.5, .015, f"Held-out test n={len(pred)} | development-only fitting | run {result['run_id']}", ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(output / name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        files[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    matrix = confusion_matrix(pred.tier, pred['k-NN'], labels=CLASSES)
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]}\n{normalized[i,j]:.1%}', ha='center', va='center', color='white' if normalized[i,j] > .5 else '#102a43', fontsize=13)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted group', ylabel='Actual group', title='k-NN held-out confusion matrix\nCounts and percentages within each actual group')
    save(fig, 'knn_confusion.png')
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    for i, name in enumerate(['LDA', 'QDA', 'k-NN']):
        values = recall_score(pred.tier, pred[name], labels=CLASSES, average=None, zero_division=0)
        bars = ax.bar(x + (i-1)*.25, values, .24, label=name, color=COLORS, alpha=[.45,.7,1][i], edgecolor='#102a43', hatch=['//','..',''][i])
        ax.bar_label(bars, labels=[f'{v:.1%}' for v in values], fontsize=9, padding=3)
    counts = pred.tier.value_counts()
    ax.set(xticks=x, xticklabels=[f'{c}\n(n={counts[c]})' for c in CLASSES], ylim=(0,1.17), ylabel='Recall', title='Held-out class recall: fixed LDA, regularized QDA and primary k-NN')
    ax.legend(loc='upper right', ncol=3)
    save(fig, 'recall_comparison.png')
    (output / 'manifest.json').write_text(json.dumps({'schema_version': result['schema_version'], 'run_id': result['run_id'], 'files': files}, indent=2) + '\n')


if __name__ == '__main__':
    main()
