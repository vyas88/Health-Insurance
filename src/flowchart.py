"""Readable, editable flowchart source with a PNG export."""
from .data import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 11))
    lanes = [
        ('OFFLINE: development and evaluation', [
            'Read raw CSV; retain source-row IDs',
            'Schema valid? No: stop and inspect\nInvalid target? Drop and report',
            'Remove/report exact duplicates\nAudit identifiers; no proof of person uniqueness',
            'Random 80/20 split, seed 42\nFreeze DEVELOPMENT charge quantiles',
            'Distinct cutoffs and adequate classes?\nNo: stop; document design limitation',
            'Five development folds\nMissing predictors? Fit median/mode on train fold',
            'Scale/encode inside each fold\nTune modest k-NN grid by macro F1',
            'Fit chosen k-NN and fixed benchmarks\non all development records',
            'Evaluate frozen test once\nSave model, predictions, metrics, figures']),
        ('ONLINE: applicant inference', [
            'Load matching model/results/figures\nMismatch? Stop with rebuild instructions',
            'Read six applicant fields',
            'Valid and complete?\nNo: display error; ask for correction',
            'Outside training range?\nFlag unfamiliar values without a new label',
            'Saved preprocessing only\nFind k DEVELOPMENT neighbours',
            'Python class and exact support\nHistorical group summary and preview',
            'AI button clicked?\nNo: retain Python-only result',
            'Configured and response completed?\nNo: neutral availability message',
            'Show AI-assisted interpretation\nCache by profile, context, model and run'])]
    for ax, (title, nodes) in zip(axes, lanes):
        ax.set(xlim=(0,1), ylim=(0,1))
        ax.axis('off')
        ax.set_title(title, fontsize=13, color='#102a43', pad=20)
        for i, node in enumerate(nodes):
            y = .95-i*.108
            ax.text(.5, y, node, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=.65', facecolor='#eff5fa', edgecolor='#1f5d9b'))
            if i < len(nodes)-1:
                ax.annotate('', xy=(.5,y-.07), xytext=(.5,y-.04), arrowprops=dict(arrowstyle='->', color='#102a43'))
    fig.tight_layout()
    fig.savefig(ROOT / 'outputs/flowchart.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
