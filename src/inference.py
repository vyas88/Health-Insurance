"""Load compatible artifacts and explain the exact fitted neighbour model."""
import hashlib
import importlib.metadata
import json
import joblib
import numpy as np
import pandas as pd
from .data import ROOT, FEATURES, CLASSES, CATEGORIES, NUMERIC, validate_predictors


# Validate saved files as one coherent analysis before exposing them to the
# app. This loader never retrains or silently falls back to an older model.
def load_artifacts(root=ROOT):
    try:
        results = json.loads((root / 'outputs/results.json').read_text())
        model_path = root / 'models/model.joblib'
        # Check the expected schema and model checksum before unpickling the trusted
        # local model. A checksum detects mismatched bytes, not malicious provenance.
        if results.get('schema_version') != 2 or hashlib.sha256(model_path.read_bytes()).hexdigest() != results['model_sha256']:
            raise ValueError('Model and results version/hash mismatch')
        # Exact core versions avoid unsupported cross-version estimator loading.
        # The error explains how to install the environment and regenerate artifacts.
        for package in ['scikit-learn', 'numpy', 'scipy', 'pandas', 'joblib']:
            if results['versions'][package] != importlib.metadata.version(package):
                raise ValueError(f'Incompatible {package} version')
        bundle = joblib.load(model_path)
        # The figure manifest binds each image checksum to the same run/schema.
        # This prevents correct-looking plots from a different evaluation appearing
        # beside the current model metrics.
        manifest = json.loads((root / 'outputs/figures/manifest.json').read_text())
        for key in ['run_id', 'schema_version']:
            if bundle[key] != results[key] or manifest[key] != results[key]:
                raise ValueError('Artifact run mismatch')
        for name, digest in manifest['files'].items():
            if hashlib.sha256((root / 'outputs/figures' / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Figure checksum mismatch')
        # Validate raw field names, category domains and display order as a contract.
        # A pipeline can run successfully yet give misleading output if the app
        # assigns its input columns or class-support labels in a different order.
        if bundle['features'] != FEATURES or bundle['class_order'] != CLASSES or bundle['raw_schema'] != {'numeric': NUMERIC, 'categorical': CATEGORIES}:
            raise ValueError('Input/class schema mismatch')
        if bundle['thresholds'] != results['thresholds'] or bundle['reference'].source_row.tolist() != results['split']['development_ids']:
            raise ValueError('Threshold/reference mismatch')
        # The neighbour reference is a training resource, so any overlap with the
        # test IDs is a leakage error, not something the UI should silently tolerate.
        if set(bundle['reference'].source_row) & set(results['split']['test_ids']):
            raise ValueError('Test rows found in reference')
        if bundle['pipeline'].named_steps['preprocess'].transform(bundle['reference'][FEATURES].iloc[:1]).shape[1] != 11:
            raise ValueError('Expected 11 encoded features')
        return bundle, results
    # Wrap file, version and schema failures in one actionable user-facing error.
    # The chained exception still preserves technical details for debugging.
    except Exception as error:
        raise ValueError(f'Cannot load compatible artifacts ({error}). Install requirements.txt and run python -m src.main.') from error


# Independently reproduce the library vote for validation and explanation.
# Inputs contain all selected k distances and their observed class labels.
# This function checks support; it does not replace the fitted prediction.
def neighbour_vote(distances, labels, weights):
    weights_array = np.ones(len(distances))
    if weights == 'distance':
        # Inverse distance is undefined at zero. Match scikit-learn's convention:
        # when exact matches exist, only those matches receive equal nonzero weight.
        zero = distances == 0
        weights_array = zero.astype(float) if zero.any() else 1 / distances
    # Sum each class's weights and divide by the total. CLASSES also ensures
    # absent groups appear with zero support in the standard display order.
    return {label: float(weights_array[np.asarray(labels) == label].sum() / weights_array.sum()) for label in CLASSES}


# Accept a dictionary of six raw fields and return Python-computed UI facts.
# Do not manually encode the input: reuse the saved training transformation.
def assess(bundle, profile):
    row = validate_predictors(pd.DataFrame([profile]), allow_missing=False)
    pipeline = bundle['pipeline']
    model = pipeline.named_steps['model']
    encoded = pipeline.named_steps['preprocess'].transform(row)
    # kneighbors returns arrays shaped (queries, k); there is one query here.
    # Its integer indices refer to positions in the saved development reference.
    distances, indices = model.kneighbors(encoded)
    neighbours = bundle['reference'].iloc[indices[0]].copy()
    neighbours['distance'] = distances[0]
    # predict_proba uses the estimator's classes_ order, which may differ from
    # Low/Medium/High. Map labels first, then explicitly order the displayed values.
    support_map = dict(zip(model.classes_, pipeline.predict_proba(row)[0]))
    support = {label: float(support_map[label]) for label in CLASSES}
    manual = neighbour_vote(distances[0], neighbours.tier.to_numpy(), model.weights)
    # Floating-point operations can differ by tiny rounding errors. allclose
    # checks numerical agreement without demanding bit-identical arithmetic.
    if not np.allclose(list(support.values()), list(manual.values())):
        raise ValueError('Neighbour support does not match saved pipeline.')
    # Compare with empirical training ranges, not clinical limits. An unfamiliar
    # value warrants caution but does not create a validated typicality category.
    outside = [name for name in NUMERIC if row[name].iat[0] < bundle['reference'][name].min() or row[name].iat[0] > bundle['reference'][name].max()]
    neighbours['row_label'] = neighbours.source_row.map(lambda value: f'Row {value}')
    # Sort the library-selected neighbours only for display; do not select a new
    # set using a different tie rule. All k vote even though only five are shown.
    preview = neighbours.sort_values(['distance', 'source_row'], kind='stable').head(5)
    return {'profile': profile, 'run_id': bundle['run_id'], 'tier': str(pipeline.predict(row)[0]),
            'support': support, 'k': model.n_neighbors, 'weights': model.weights,
            'metric': 'Manhattan' if model.p == 1 else 'Euclidean', 'outside_training_range': outside,
            'neighbour_counts': neighbours.tier.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
            'preview': preview[['row_label'] + FEATURES + ['tier', 'distance']]}
