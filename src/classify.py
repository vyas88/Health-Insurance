"""Fold-fitted preprocessing, declared tuning, and fixed-order evaluation."""
import warnings
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, make_scorer, recall_score
from .data import NUMERIC, CATEGORIES, CLASSES

SEED = 42
CV_CONFIG = {'n_splits': 5, 'shuffle': True, 'random_state': SEED}


# Construct an UNFITTED estimator for one named model. GridSearchCV and
# cross_validate clone and fit this whole pipeline within each training fold,
# so validation rows cannot influence imputation or standardization.
def pipeline_for(name):
    # SimpleImputer uses the training median, which is less sensitive to extremes
    # than the mean. StandardScaler then uses training means and standard deviations
    # so age, BMI and children contribute on comparable numerical scales.
    numerical = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    # Mode imputation handles missing labels without treating them as numbers.
    # Full one-hot encoding for k-NN gives symmetric category mismatches and 11
    # dimensions. LDA/QDA drop a reference category to avoid redundant indicators,
    # giving eight dimensions. Indicators remain unscaled in both branches.
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(categories=list(CATEGORIES.values()), drop=None if name == 'k-NN' else 'first',
                                 handle_unknown='error', sparse_output=False))])
    # ColumnTransformer applies the appropriate operations to named column groups
    # and joins the results. No charge-derived or source-ID columns are selected.
    pre = ColumnTransformer([('numerical', numerical, NUMERIC), ('categorical', categorical, list(CATEGORIES))])
    # KD-tree finds nearest points under the selected Minkowski metric without
    # changing the voting rule. LDA fits shared-covariance linear boundaries; QDA
    # allows class-specific covariance, with declared 0.001 regularization to help
    # numerical stability. These are fixed benchmarks, not automatic replacements.
    models = {'k-NN': KNeighborsClassifier(algorithm='kd_tree', n_jobs=1),
              'LDA': LinearDiscriminantAnalysis(solver='svd'),
              'QDA': QuadraticDiscriminantAnalysis(reg_param=0.001)}
    return Pipeline([('preprocess', pre), ('model', models[name])])


# Return the scoring names accepted by scikit-learn. Macro F1 gives each class
# equal weight; accuracy counts all correct labels; balanced accuracy averages
# class recalls. Reporting several measures exposes errors hidden by accuracy.
def scoring():
    scores = {'macro_f1': 'f1_macro', 'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy'}
    # make_scorer adapts recall_score into the estimator/X/y interface expected
    # by CV. A one-class labels list extracts that class's recall. zero_division=0
    # makes an undefined scorer deterministic; it does not prove good performance.
    for label in CLASSES:
        scores['recall_' + label] = make_scorer(recall_score, labels=[label], average='macro', zero_division=0)
    return scores


# Inputs must contain DEVELOPMENT predictors and frozen development labels.
# Return fitted models, selection reports and the exact fold index pairs.
# This function never receives test rows, so tuning cannot use test outcomes.
def fit_models(X, y):
    # Stratification distributes class labels across five shuffled folds.
    # Materializing the splits ensures every candidate and benchmark shares them.
    folds = list(StratifiedKFold(**CV_CONFIG).split(X, y))
    if min(np.sum(y == label) for label in CLASSES) < 5:
        raise ValueError('Each development class needs at least five records for declared CV.')
    # An entirely missing training-fold column has no median or mode to learn.
    # Fail explicitly instead of silently changing the number of encoded features.
    if any(X.iloc[train].isna().all().any() for train, _ in folds):
        raise ValueError('A training fold has an entirely missing predictor; cannot fit declared imputation.')
    # The model__ prefix addresses a parameter on the pipeline's model step.
    # The modest grid limits selection complexity; p=1 is Manhattan and p=2 is
    # Euclidean. A candidate k must fit even the smallest training fold.
    grid = {'model__n_neighbors': [k for k in [5, 11, 21, 31] if k <= min(len(t) for t, _ in folds)],
            'model__weights': ['uniform', 'distance'], 'model__p': [1, 2]}
    # Exact score ties: smaller k, uniform before distance, then Manhattan before Euclidean.
    # GridSearchCV calls this function with candidate scores and expects a row
    # index. Minimizing the negative F1 selects the greatest F1. Remaining tuple
    # items implement the declared deterministic tie rule, not a test-based choice.
    def choose(cv):
        return min(range(len(cv['params'])), key=lambda i: (-cv['mean_test_macro_f1'][i],
            cv['params'][i]['model__n_neighbors'], cv['params'][i]['model__weights'] != 'uniform', cv['params'][i]['model__p']))
    fitted, reports = {}, {}
    # refit=choose fits the selected pipeline on all development rows afterward.
    # error_score=raise makes data or estimator failures visible instead of quietly
    # assigning missing scores. One worker keeps this small analysis predictable.
    search = GridSearchCV(pipeline_for('k-NN'), grid, scoring=scoring(), refit=choose, cv=folds, n_jobs=1, error_score='raise')
    search.fit(X, y)
    # Preserve all candidates and individual fold scores, not just the winner.
    # Their spread helps assess sensitivity to k and stability across splits;
    # these standard deviations describe folds, not confidence intervals.
    candidates = []
    for i, params in enumerate(search.cv_results_['params']):
        candidates.append({'parameters': params, **{key: {'mean': float(search.cv_results_['mean_test_' + key][i]),
            'sd': float(search.cv_results_['std_test_' + key][i]),
            'folds': [float(search.cv_results_[f'split{f}_test_{key}'][i]) for f in range(5)]} for key in scoring()}})
    fitted['k-NN'] = search.best_estimator_
    reports['k-NN'] = {'selected_parameters': search.best_params_, 'candidates': candidates,
        'selection_cv': candidates[search.best_index_], 'warnings': []}
    for name in ['LDA', 'QDA']:
        model = pipeline_for(name)
        # Capture warnings for the report rather than suppressing rank/conditioning
        # information. The context manager restores the previous warning policy.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cv = cross_validate(model, X, y, cv=folds, scoring=scoring(), error_score='raise')
            # cross_validate fits clones, leaving this object unfitted. Fit it once on
            # all development rows so the caller can evaluate the frozen benchmark.
            model.fit(X, y)
        fitted[name] = model
        reports[name] = {'selected_parameters': {'solver': 'svd'} if name == 'LDA' else {'reg_param': 0.001},
            'selection_cv': {key: {'mean': float(cv['test_' + key].mean()), 'sd': float(cv['test_' + key].std()),
                                  'folds': cv['test_' + key].tolist()} for key in scoring()},
            'warnings': sorted(set(str(w.message) for w in caught))}
    for name in reports:
        reports[name].update({'cv': CV_CONFIG, 'evaluation_partition': 'held-out test',
            'preprocessing': 'Training-fold median/mode imputation; numerical z-scores; ' +
                ('full one-hot, 11 dimensions' if name == 'k-NN' else 'reference-category one-hot, 8 dimensions')})
    return fitted, reports, folds


# Compute metrics from a fixed pair of actual/predicted arrays. Nothing is
# fitted here. Confusion rows are actual groups and columns are predictions,
# always in Low, Medium, High order. Precision measures correctness among
# predicted members; recall measures recovery among actual members.
def evaluate(actual, predicted, smoker):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    # Use actual High labels and recorded non-smoking status to define the
    # subgroup, rather than selecting only people already predicted High.
    subgroup = (actual == 'High') & (np.asarray(smoker) == 'no')
    return {'accuracy': accuracy_score(actual, predicted), 'balanced_accuracy': balanced_accuracy_score(actual, predicted),
        'macro_f1': f1_score(actual, predicted, labels=CLASSES, average='macro', zero_division=0),
        'per_class': {k: v for k, v in classification_report(actual, predicted, labels=CLASSES, output_dict=True, zero_division=0).items() if k in CLASSES},
        'confusion_matrix': confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
        'high_to_low': int(np.sum((actual == 'High') & (predicted == 'Low'))),
        'high_to_medium': int(np.sum((actual == 'High') & (predicted == 'Medium'))),
        # An empty subgroup has undefined recall, represented by None (JSON null).
        # It must not be presented as either zero recovery or perfect recovery.
        'high_non_smoker': {'n': int(subgroup.sum()), 'recall': float(np.mean(predicted[subgroup] == 'High')) if subgroup.any() else None},
        'n': len(actual)}
