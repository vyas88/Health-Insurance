import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score


def fit_classifier(F, tier, use_qda):
    lda = LinearDiscriminantAnalysis(solver="svd")
    qda = QuadraticDiscriminantAnalysis(reg_param=0.001)
    model = qda if use_qda else lda
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variables are collinear")
        lda_scores = cross_val_score(lda, F, tier, cv=folds, scoring="accuracy")
        qda_scores = cross_val_score(qda, F, tier, cv=folds, scoring="accuracy")
        scores = qda_scores if use_qda else lda_scores
        predicted = cross_val_predict(model, F, tier, cv=folds)
        fitted = model.fit(F, tier)
    labels = list(tier.cat.categories)

    return fitted, {
        "method": "QDA" if use_qda else "LDA",
        "regularization": 0.001 if use_qda else 0.0,
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_sd": float(scores.std(ddof=1)),
        "cv_accuracy": [float(s) for s in scores],
        "comparison": {
            "lda_cv_mean": float(lda_scores.mean()),
            "lda_cv_sd": float(lda_scores.std(ddof=1)),
            "qda_cv_mean": float(qda_scores.mean()),
            "qda_cv_sd": float(qda_scores.std(ddof=1)),
        },
        "confusion_matrix": confusion_matrix(tier, predicted, labels=labels).tolist(),
        "labels": labels,
        "overall_accuracy": float(accuracy_score(tier, predicted)),
    }
