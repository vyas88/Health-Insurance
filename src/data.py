"""Validate raw records before splitting; never learn preprocessing here."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

# Resolve paths relative to this module, so commands work independently of
# the terminal location. Data stays local; this module does not download it.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'insurance.csv'
# These are raw input variables, not encoded columns. Children is a count.
# Keep charges out of FEATURES: including the quantity that defines the target
# would give the classifier the answer and invalidate neighbour similarity.
NUMERIC = ['age', 'bmi', 'children']
# Declare the permitted categories in advance. The encoder must not discover
# its schema from held-out records or silently accept an unfamiliar app value.
CATEGORIES = {'sex': ['female', 'male'], 'smoker': ['no', 'yes'],
              'region': ['northeast', 'northwest', 'southeast', 'southwest']}
FEATURES = NUMERIC + list(CATEGORIES)
CLASSES = ['Low', 'Medium', 'High']


# Shared validation for dataset cleaning and applicant inference. Return a
# new DataFrame in a fixed column order so both paths use the same contract.
# Missing values are allowed offline for fold-specific imputation; the online
# form instead requires a complete profile.
def validate_predictors(frame, allow_missing=True):
    # Copying avoids modifying the caller's original table as a side effect.
    frame = frame.copy()
    if set(frame.columns) != set(FEATURES):
        raise ValueError('Expected exactly these raw fields: ' + ', '.join(FEATURES))
    for name in NUMERIC:
        original = frame[name]
        # Coerce nonnumeric text to NaN so it can be identified by a Boolean mask.
        # The original notna() distinguishes invalid text from genuinely missing data.
        values = pd.to_numeric(original, errors='coerce')
        invalid = original.notna() & (values.isna() | ~np.isfinite(values))
        # Combine invalidity conditions with OR. These checks reject impossible input
        # formats or signs, not legitimate extreme observations outside sample ranges.
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
        # Normalize missing categorical values to NumPy NaN, which SimpleImputer
        # recognizes consistently. Do not fill them using the complete dataset.
        frame[name] = frame[name].where(frame[name].notna(), np.nan)
    if not allow_missing and frame.isna().any().any():
        raise ValueError('Complete all six applicant fields.')
    return frame[FEATURES]


# Return cleaned raw records plus a serializable audit dictionary. No means,
# scales or target cutoffs are learned here because the split has not happened.
def build_data(path=DATA_PATH):
    path = Path(path)
    raw = pd.read_csv(path)
    # Extra columns could contain a person identifier or repeated-person grouping.
    # Stop for inspection instead of dropping that information and splitting blindly.
    if set(raw.columns) != set(FEATURES + ['charges']):
        raise ValueError('Unexpected schema. Inspect extra identifiers/repeated-person structure before splitting.')
    missing = raw.isna().sum().astype(int).to_dict()
    # duplicated() marks later exact matches, keeping the first occurrence.
    # The decision concerns matching rows, not proven duplicate individuals.
    duplicates = raw.duplicated()
    # A zero-based DataFrame index becomes a CSV line label by adding two:
    # one for one-based numbering and one for the header. IDs survive filtering.
    duplicate_ids = (raw.index[duplicates] + 2).tolist()
    data = raw.loc[~duplicates].copy()
    data['source_row'] = data.index + 2  # Physical CSV line, including the header.
    charges = pd.to_numeric(data.charges, errors='coerce')
    # A missing, infinite, negative or nonnumeric charge cannot define a valid
    # label. Drop and report these targets; never invent a label by imputation.
    invalid = charges.isna() | ~np.isfinite(charges) | (charges < 0)
    invalid_ids = data.loc[invalid, 'source_row'].tolist()
    data = data.loc[~invalid].copy()
    data['charges'] = charges.loc[~invalid]
    data[FEATURES] = validate_predictors(data[FEATURES])
    if data.empty:
        raise ValueError('No valid target charges remain.')
    # Reset the working index for convenient positional access, while preserving
    # source_row as the audit identity. SHA-256 identifies the exact input bytes;
    # it is a reproducibility fingerprint, not evidence of data quality.
    return data.reset_index(drop=True), {
        'source_rows': len(raw), 'clean_rows': len(data), 'missing_counts_raw': missing,
        'missing_predictors_clean': data[FEATURES].isna().sum().astype(int).to_dict(),
        'exact_duplicates_removed': len(duplicate_ids), 'duplicate_source_rows': duplicate_ids,
        'invalid_target_rows_removed': invalid_ids,
        'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'identifier_audit': 'Only seven expected fields; no person identifier. Matching records do not prove a duplicated person; independence cannot be verified.'}


# Apply already-frozen development thresholds to any valid charge array.
# Keeping this rule in one function prevents training/test boundary drift.
def label_costs(charges, thresholds):
    q1, q2 = thresholds
    if not np.isfinite([q1, q2]).all() or q1 >= q2:
        raise ValueError('Distinct finite development quantiles are required for three groups.')
    values = np.asarray(charges, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Target charges must be finite and nonnegative.')
    # Vectorized conditions assign every valid value, including values outside
    # the development min/max. Equality belongs to the lower group at each cutoff.
    return np.where(values <= q1, 'Low', np.where(values <= q2, 'Medium', 'High'))
