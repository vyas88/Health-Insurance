"""Figures are derived only from saved frozen holdout predictions."""
import hashlib
import json
import matplotlib
# Use a noninteractive renderer before importing pyplot. Figure generation
# then works in a terminal or headless hosting environment without a GUI.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score
from .data import ROOT, CLASSES

# Class colours follow Low, Medium, High. Model identity is additionally
# shown with hatch patterns and opacity so colour is not the only visual cue.
COLORS = ['#2878a5', '#d69b26', '#228576']


# Read saved evaluation predictions and generate two figures. No model is
# trained here, so figure regeneration cannot influence model selection.
def main():
    result = json.loads((ROOT / 'outputs/results.json').read_text())
    pred = pd.read_csv(ROOT / 'outputs/evaluation_predictions.csv')
    # Check that every prediction row belongs to the results run before plotting.
    # A visually plausible figure from another run would still be invalid evidence.
    if set(pred.run_id) != {result['run_id']}:
        raise ValueError('Prediction run mismatch')
    output = ROOT / 'outputs/figures'
    output.mkdir(exist_ok=True)
    files = {}
    # Centralize captions, resolution and checksums so both figures use the same
    # publication settings. tight_layout reserves bottom space for the run caption;
    # closing figures releases memory during repeated command-line generation.
    def save(fig, name):
        fig.text(.5, .015, f"Held-out test n={len(pred)} | development-only fitting | run {result['run_id']}", ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(output / name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        files[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    # Counts have actual classes on rows and predicted classes on columns.
    # Passing labels explicitly fixes the order even if data arrives differently.
    matrix = confusion_matrix(pred.tier, pred['k-NN'], labels=CLASSES)
    # Divide each row by its actual-class support. keepdims retains a (3, 1)
    # array so NumPy broadcasts the correct denominator across each row.
    # The analysis entry point requires every test class to be represented.
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    # Use a fixed 0-to-1 colour scale so intensity represents the same percentage
    # in every cell. Cell text keeps the original counts visible alongside rates.
    ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]}\n{normalized[i,j]:.1%}', ha='center', va='center', color='white' if normalized[i,j] > .5 else '#102a43', fontsize=13)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted group', ylabel='Actual group', title='k-NN held-out confusion matrix\nCounts and percentages within each actual group')
    save(fig, 'knn_confusion.png')
    fig, ax = plt.subplots(figsize=(8, 5))
    # Three base x positions represent the three classes. Small offsets place
    # LDA, QDA and k-NN bars next to one another within each class.
    x = np.arange(3)
    for i, name in enumerate(['LDA', 'QDA', 'k-NN']):
        # average=None returns one recall per class rather than averaging away the
        # High-group weakness. Bar labels are calculated from the saved predictions.
        values = recall_score(pred.tier, pred[name], labels=CLASSES, average=None, zero_division=0)
        bars = ax.bar(x + (i-1)*.25, values, .24, label=name, color=COLORS, alpha=[.45,.7,1][i], edgecolor='#102a43', hatch=['//','..',''][i])
        ax.bar_label(bars, labels=[f'{v:.1%}' for v in values], fontsize=9, padding=3)
    counts = pred.tier.value_counts()
    ax.set(xticks=x, xticklabels=[f'{c}\n(n={counts[c]})' for c in CLASSES], ylim=(0,1.17), ylabel='Recall', title='Held-out class recall: fixed LDA, regularized QDA and primary k-NN')
    ax.legend(loc='upper right', ncol=3)
    save(fig, 'recall_comparison.png')
    # The app checks these image hashes plus run/schema IDs before displaying
    # figures, connecting the visual evidence to the same model/results bundle.
    (output / 'manifest.json').write_text(json.dumps({'schema_version': result['schema_version'], 'run_id': result['run_id'], 'files': files}, indent=2) + '\n')


if __name__ == '__main__':
    main()
