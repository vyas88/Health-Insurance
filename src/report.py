"""Generate a concise results table and error interpretation from this run."""
import json
from .data import ROOT


def main():
    r=json.loads((ROOT/'outputs/results.json').read_text())
    lines=['# Revised evaluation results', '', f"Run `{r['run_id']}`; schema {r['schema_version']}; Python {r['python']}.",
           '', f"Development n={len(r['split']['development_ids'])}; held-out test n={len(r['split']['test_ids'])}.",
           f"Exact development cutoffs: q1={r['thresholds'][0]!r}, q2={r['thresholds'][1]!r}.",
           '', '| Model | Accuracy | Balanced accuracy | Macro F1 | High recall |', '|---|---:|---:|---:|---:|']
    for name,report in r['models'].items():
        m=report['test']
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {m['macro_f1']:.4f} | {m['per_class']['High']['recall']:.2%} |")
    knn=r['models']['k-NN']; m=knn['test']; cv=knn['selection_cv']; sub=m['high_non_smoker']
    subgroup_text = 'undefined (no eligible records)' if sub['recall'] is None else f"{sub['recall']:.1%}"
    benchmark_text = '; '.join(f"{name}: subgroup n={r['models'][name]['test']['high_non_smoker']['n']}, recall={r['models'][name]['test']['high_non_smoker']['recall']}" for name in ['LDA', 'QDA'])
    lines += ['', f"Selected: `{knn['selected_parameters']}`. Development selection macro F1: {cv['macro_f1']['mean']:.4f} (fold SD {cv['macro_f1']['sd']:.4f}); fold High recall spans {min(cv['recall_High']['folds']):.1%} to {max(cv['recall_High']['folds']):.1%}.",
        '', f"k-NN misses {m['high_to_low']+m['high_to_medium']} High records: {m['high_to_low']} go to Low and {m['high_to_medium']} to Medium. High-cost non-smoker recall is {subgroup_text} on n={sub['n']}. This is a material failure, despite higher aggregate accuracy in this split.",
        '', benchmark_text + '. The profile omits diagnosis, treatment and utilization information. These are plausible missing predictors, not demonstrated causal explanations. Small differences on one split do not establish statistical superiority.',
        '', '## Sensitivity to k (best development selection macro F1 at each k)', '', '| k | Best mean macro F1 |', '|---|---:|']
    for k in [5,11,21,31]:
        scores=[c['macro_f1']['mean'] for c in knn['candidates'] if c['parameters']['model__n_neighbors']==k]
        if scores:
            lines.append(f'| {k} | {max(scores):.4f} |')
    lines += ['', 'Selection scores are optimistic estimates after tuning, not nested-CV estimates. The held-out split was created during this refactor of an already-explored dataset, not external validation.',
              '', 'QDA development covariance diagnostics and any captured warnings are in results.json. No classical hypothesis test validates k-NN.']
    (ROOT/'outputs/RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
