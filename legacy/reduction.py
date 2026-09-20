import numpy as np
from sklearn.decomposition import PCA


def pca(F):
    model = PCA()
    scores = model.fit_transform(F)
    return {
        "eigenvalues": model.explained_variance_,
        "proportion_variance": model.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(model.explained_variance_ratio_),
        "loadings": model.components_.T,
        "features": list(F.columns),
        "scores": scores[:, :2],
    }
