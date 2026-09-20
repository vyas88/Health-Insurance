"""Validate raw records before splitting; never learn preprocessing here."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'insurance.csv'
NUMERIC = ['age', 'bmi', 'children']
CATEGORIES = {'sex': ['female', 'male'], 'smoker': ['no', 'yes'],
              'region': ['northeast', 'northwest', 'southeast', 'southwest']}
FEATURES = NUMERIC + list(CATEGORIES)
CLASSES = ['Low', 'Medium', 'High']


def validate_predictors(frame, allow_missing=True):
    frame = frame.copy()
    if set(frame.columns) != set(FEATURES):
        raise ValueError('Expected exactly these raw fields: ' + ', '.join(FEATURES))
    for name in NUMERIC:
        original = frame[name]
        values = pd.to_numeric(original, errors='coerce')
        invalid = original.notna() & (values.isna() | ~np.isfinite(values))
        invalid |= values < 0
        if name in ['age', 'children']:
            invalid |= values.notna() & (values % 1 != 0)
        if name == 'bmi':
            invalid |= values <= 0
        if invalid.any():
            raise ValueError(f'Invalid {name}: use finite, nonnegative values; age/children are integers and BMI is positive.')
        frame[name] = values.astype(float)
    for name, domain in CATEGORIES.items():
        if (frame[name].notna() & ~frame[name].isin(domain)).any():
            raise ValueError(f'Unsupported {name}; expected {domain}.')
        frame[name] = frame[name].where(frame[name].notna(), np.nan)
    if not allow_missing and frame.isna().any().any():
        raise ValueError('Complete all six applicant fields.')
    return frame[FEATURES]


def build_data(path=DATA_PATH):
    path = Path(path)
    raw = pd.read_csv(path)
    if set(raw.columns) != set(FEATURES + ['charges']):
        raise ValueError('Unexpected schema. Inspect extra identifiers/repeated-person structure before splitting.')
    missing = raw.isna().sum().astype(int).to_dict()
    duplicates = raw.duplicated()
    duplicate_ids = (raw.index[duplicates] + 2).tolist()
    data = raw.loc[~duplicates].copy()
    data['source_row'] = data.index + 2  # Physical CSV line, including the header.
    charges = pd.to_numeric(data.charges, errors='coerce')
    invalid = charges.isna() | ~np.isfinite(charges) | (charges < 0)
    invalid_ids = data.loc[invalid, 'source_row'].tolist()
    data = data.loc[~invalid].copy()
    data['charges'] = charges.loc[~invalid]
    data[FEATURES] = validate_predictors(data[FEATURES])
    if data.empty:
        raise ValueError('No valid target charges remain.')
    return data.reset_index(drop=True), {
        'source_rows': len(raw), 'clean_rows': len(data), 'missing_counts_raw': missing,
        'missing_predictors_clean': data[FEATURES].isna().sum().astype(int).to_dict(),
        'exact_duplicates_removed': len(duplicate_ids), 'duplicate_source_rows': duplicate_ids,
        'invalid_target_rows_removed': invalid_ids,
        'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'identifier_audit': 'Only seven expected fields; no person identifier. Matching records do not prove a duplicated person; independence cannot be verified.'}


def label_costs(charges, thresholds):
    q1, q2 = thresholds
    if not np.isfinite([q1, q2]).all() or q1 >= q2:
        raise ValueError('Distinct finite development quantiles are required for three groups.')
    values = np.asarray(charges, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Target charges must be finite and nonnegative.')
    return np.where(values <= q1, 'Low', np.where(values <= q2, 'Medium', 'High'))
