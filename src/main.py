"""Reproduce development selection and frozen holdout evaluation."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import ROOT, FEATURES, NUMERIC, CATEGORIES, CLASSES, build_data, label_costs
from .classify import fit_models, evaluate, SEED

SCHEMA_VERSION = 2


def json_ready(value):
    if isinstance(value, (np.ndarray, pd.Series)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def main():
    data, audit = build_data()
    dev, test = train_test_split(data, test_size=0.2, random_state=SEED)
    dev = dev.sort_values('source_row').reset_index(drop=True)
    test = test.sort_values('source_row').reset_index(drop=True)
    thresholds = dev.charges.quantile([1/3, 2/3]).tolist()
    dev['tier'] = label_costs(dev.charges, thresholds)
    test['tier'] = label_costs(test.charges, thresholds)
    if set(test.tier) != set(CLASSES):
        raise ValueError('Test class missing. Report limitation before choosing another evaluation design.')
    models, reports, folds = fit_models(dev[FEATURES], dev.tier.to_numpy())
    versions = {p: importlib.metadata.version(p) for p in ['scikit-learn', 'numpy', 'scipy', 'pandas', 'joblib', 'matplotlib', 'streamlit', 'openai', 'python-dotenv']}
    source_hash = hashlib.sha256(b''.join((ROOT / 'src' / p).read_bytes() for p in ['data.py', 'classify.py', 'main.py'])).hexdigest()
    run_id = hashlib.sha256(json.dumps([audit['data_sha256'], versions, source_hash, SCHEMA_VERSION], sort_keys=True).encode()).hexdigest()[:16]
    # Diagnose covariance conditioning on development data only.
    benchmark_matrix = models['QDA'].named_steps['preprocess'].transform(dev[FEATURES])
    conditioning = {}
    for label in CLASSES:
        covariance = np.cov(benchmark_matrix[dev.tier == label], rowvar=False)
        regularized = .999 * covariance + .001 * np.eye(covariance.shape[0])
        conditioning[label] = {'raw_rank': int(np.linalg.matrix_rank(covariance)), 'dimensions': 8,
            'regularized_condition_number': float(np.linalg.cond(regularized))}
    reports['QDA']['development_conditioning'] = conditioning
    if any(value['raw_rank'] < 8 for value in conditioning.values()):
        reports['QDA']['warnings'].append('Development class covariance has deficient rank before regularization; class-constant indicators limit Gaussian interpretation.')
    predictions = test[['source_row', 'charges', 'smoker', 'tier']].copy()
    for name, model in models.items():
        pred = model.predict(test[FEATURES])
        predictions[name] = pred
        reports[name]['test'] = evaluate(test.tier, pred, test.smoker)
        reports[name]['split_sizes'] = {'development': len(dev), 'test': len(test)}
    summaries = {}
    for label in CLASSES:
        values = dev.loc[dev.tier == label, 'charges']
        summaries[label] = {'n': len(values), 'min': values.min(), 'q25': values.quantile(.25),
                            'median': values.median(), 'q75': values.quantile(.75), 'max': values.max()}
    result = {'schema_version': SCHEMA_VERSION, 'run_id': run_id, 'source_sha256': source_hash,
        'versions': versions, 'python': platform.python_version(), 'data': audit,
        'class_order': CLASSES, 'raw_schema': {'numeric': NUMERIC, 'categorical': CATEGORIES},
        'thresholds': thresholds, 'boundary_rule': 'Low <= q1; q1 < Medium <= q2; High > q2; open-ended outer groups',
        'split': {'random_state': SEED, 'test_size': .2, 'stratified': False,
            'development_ids': dev.source_row.tolist(), 'test_ids': test.source_row.tolist(),
            'class_counts': {'development': dev.tier.value_counts().to_dict(), 'test': test.tier.value_counts().to_dict()},
            'cv_folds': [{'training_ids': dev.iloc[t].source_row.tolist(), 'validation_ids': dev.iloc[v].source_row.tolist()} for t, v in folds]},
        'models': reports, 'historical_development_costs': summaries,
        'eda': {'partition': 'development', 'n': len(dev), 'mean_vector': dev[NUMERIC].mean().to_dict(),
                'covariance': dev[NUMERIC].cov().to_dict(), 'correlation': dev[NUMERIC].corr().to_dict(),
                'numerical_ranges': {c: [dev[c].min(), dev[c].max()] for c in NUMERIC},
                'category_counts': {c: dev[c].value_counts(dropna=False).to_dict() for c in CATEGORIES},
                'missing_counts': dev[FEATURES].isna().sum().to_dict()}}
    bundle = {key: result[key] for key in ['schema_version', 'run_id', 'versions', 'thresholds', 'class_order', 'raw_schema']}
    bundle.update({'pipeline': models['k-NN'], 'reference': dev, 'features': FEATURES,
                   'selected_parameters': reports['k-NN']['selected_parameters']})
    for directory in ['models', 'outputs']:
        (ROOT / directory).mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / 'models/model.joblib')
    result['model_sha256'] = hashlib.sha256((ROOT / 'models/model.joblib').read_bytes()).hexdigest()
    predictions['run_id'] = run_id
    predictions['schema_version'] = SCHEMA_VERSION
    predictions.to_csv(ROOT / 'outputs/evaluation_predictions.csv', index=False)
    (ROOT / 'outputs/results.json').write_text(json.dumps(json_ready(result), indent=2, allow_nan=False) + '\n')
    from .figures import main as figures
    figures()
    from .flowchart import main as flowchart
    flowchart()
    from .report import main as report
    report()
    print(json.dumps({name: report['test'] for name, report in reports.items()}, indent=2))
    print('Selected:', reports['k-NN']['selected_parameters'], 'Run:', run_id)


if __name__ == '__main__':
    main()
