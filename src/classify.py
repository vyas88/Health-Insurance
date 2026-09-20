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


def pipeline_for(name):
    numerical = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(categories=list(CATEGORIES.values()), drop=None if name == 'k-NN' else 'first',
                                 handle_unknown='error', sparse_output=False))])
    pre = ColumnTransformer([('numerical', numerical, NUMERIC), ('categorical', categorical, list(CATEGORIES))])
    models = {'k-NN': KNeighborsClassifier(algorithm='kd_tree', n_jobs=1),
              'LDA': LinearDiscriminantAnalysis(solver='svd'),
              'QDA': QuadraticDiscriminantAnalysis(reg_param=0.001)}
    return Pipeline([('preprocess', pre), ('model', models[name])])


def scoring():
    scores = {'macro_f1': 'f1_macro', 'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy'}
    for label in CLASSES:
        scores['recall_' + label] = make_scorer(recall_score, labels=[label], average='macro', zero_division=0)
    return scores


def fit_models(X, y):
    folds = list(StratifiedKFold(**CV_CONFIG).split(X, y))
    if min(np.sum(y == label) for label in CLASSES) < 5:
        raise ValueError('Each development class needs at least five records for declared CV.')
    if any(X.iloc[train].isna().all().any() for train, _ in folds):
        raise ValueError('A training fold has an entirely missing predictor; cannot fit declared imputation.')
    grid = {'model__n_neighbors': [k for k in [5, 11, 21, 31] if k <= min(len(t) for t, _ in folds)],
            'model__weights': ['uniform', 'distance'], 'model__p': [1, 2]}
    # Exact score ties: smaller k, uniform before distance, then Manhattan before Euclidean.
    def choose(cv):
        return min(range(len(cv['params'])), key=lambda i: (-cv['mean_test_macro_f1'][i],
            cv['params'][i]['model__n_neighbors'], cv['params'][i]['model__weights'] != 'uniform', cv['params'][i]['model__p']))
    fitted, reports = {}, {}
    search = GridSearchCV(pipeline_for('k-NN'), grid, scoring=scoring(), refit=choose, cv=folds, n_jobs=1, error_score='raise')
    search.fit(X, y)
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cv = cross_validate(model, X, y, cv=folds, scoring=scoring(), error_score='raise')
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


def evaluate(actual, predicted, smoker):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    subgroup = (actual == 'High') & (np.asarray(smoker) == 'no')
    return {'accuracy': accuracy_score(actual, predicted), 'balanced_accuracy': balanced_accuracy_score(actual, predicted),
        'macro_f1': f1_score(actual, predicted, labels=CLASSES, average='macro', zero_division=0),
        'per_class': {k: v for k, v in classification_report(actual, predicted, labels=CLASSES, output_dict=True, zero_division=0).items() if k in CLASSES},
        'confusion_matrix': confusion_matrix(actual, predicted, labels=CLASSES).tolist(),
        'high_to_low': int(np.sum((actual == 'High') & (predicted == 'Low'))),
        'high_to_medium': int(np.sum((actual == 'High') & (predicted == 'Medium'))),
        'high_non_smoker': {'n': int(subgroup.sum()), 'recall': float(np.mean(predicted[subgroup] == 'High')) if subgroup.any() else None},
        'n': len(actual)}
