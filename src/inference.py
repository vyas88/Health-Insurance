"""Load compatible artifacts and explain the exact fitted neighbour model."""
import hashlib
import importlib.metadata
import json
import joblib
import numpy as np
import pandas as pd
from .data import ROOT, FEATURES, CLASSES, CATEGORIES, NUMERIC, validate_predictors


def load_artifacts(root=ROOT):
    try:
        results = json.loads((root / 'outputs/results.json').read_text())
        model_path = root / 'models/model.joblib'
        if results.get('schema_version') != 2 or hashlib.sha256(model_path.read_bytes()).hexdigest() != results['model_sha256']:
            raise ValueError('Model and results version/hash mismatch')
        for package in ['scikit-learn', 'numpy', 'scipy', 'pandas', 'joblib']:
            if results['versions'][package] != importlib.metadata.version(package):
                raise ValueError(f'Incompatible {package} version')
        bundle = joblib.load(model_path)
        manifest = json.loads((root / 'outputs/figures/manifest.json').read_text())
        for key in ['run_id', 'schema_version']:
            if bundle[key] != results[key] or manifest[key] != results[key]:
                raise ValueError('Artifact run mismatch')
        for name, digest in manifest['files'].items():
            if hashlib.sha256((root / 'outputs/figures' / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Figure checksum mismatch')
        if bundle['features'] != FEATURES or bundle['class_order'] != CLASSES or bundle['raw_schema'] != {'numeric': NUMERIC, 'categorical': CATEGORIES}:
            raise ValueError('Input/class schema mismatch')
        if bundle['thresholds'] != results['thresholds'] or bundle['reference'].source_row.tolist() != results['split']['development_ids']:
            raise ValueError('Threshold/reference mismatch')
        if set(bundle['reference'].source_row) & set(results['split']['test_ids']):
            raise ValueError('Test rows found in reference')
        if bundle['pipeline'].named_steps['preprocess'].transform(bundle['reference'][FEATURES].iloc[:1]).shape[1] != 11:
            raise ValueError('Expected 11 encoded features')
        return bundle, results
    except Exception as error:
        raise ValueError(f'Cannot load compatible artifacts ({error}). Install requirements.txt and run python -m src.main.') from error


def neighbour_vote(distances, labels, weights):
    weights_array = np.ones(len(distances))
    if weights == 'distance':
        zero = distances == 0
        weights_array = zero.astype(float) if zero.any() else 1 / distances
    return {label: float(weights_array[np.asarray(labels) == label].sum() / weights_array.sum()) for label in CLASSES}


def assess(bundle, profile):
    row = validate_predictors(pd.DataFrame([profile]), allow_missing=False)
    pipeline = bundle['pipeline']
    model = pipeline.named_steps['model']
    encoded = pipeline.named_steps['preprocess'].transform(row)
    distances, indices = model.kneighbors(encoded)
    neighbours = bundle['reference'].iloc[indices[0]].copy()
    neighbours['distance'] = distances[0]
    support_map = dict(zip(model.classes_, pipeline.predict_proba(row)[0]))
    support = {label: float(support_map[label]) for label in CLASSES}
    manual = neighbour_vote(distances[0], neighbours.tier.to_numpy(), model.weights)
    if not np.allclose(list(support.values()), list(manual.values())):
        raise ValueError('Neighbour support does not match saved pipeline.')
    outside = [name for name in NUMERIC if row[name].iat[0] < bundle['reference'][name].min() or row[name].iat[0] > bundle['reference'][name].max()]
    neighbours['row_label'] = neighbours.source_row.map(lambda value: f'Row {value}')
    preview = neighbours.sort_values(['distance', 'source_row'], kind='stable').head(5)
    return {'profile': profile, 'run_id': bundle['run_id'], 'tier': str(pipeline.predict(row)[0]),
            'support': support, 'k': model.n_neighbors, 'weights': model.weights,
            'metric': 'Manhattan' if model.p == 1 else 'Euclidean', 'outside_training_range': outside,
            'neighbour_counts': neighbours.tier.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
            'preview': preview[['row_label'] + FEATURES + ['tier', 'distance']]}
