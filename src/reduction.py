"""Legacy exploratory helper only; unused by the revised classifier and app."""
import numpy as np
from sklearn.decomposition import PCA


# Legacy exploratory helper: principal component analysis rotates the supplied
# feature matrix into orthogonal directions ordered by explained variance.
# It is unsupervised and does not use cost labels. The current app does not
# call this helper, and its output is not the k-NN decision space.
def pca(F):
    # PCA centers its input but does not standardize each feature automatically.
    # The caller is responsible for choosing meaningful scaling/encoding.
    # Normality is not required for the algebraic decomposition.
    model = PCA()
    # fit_transform learns component directions and projects each row onto them.
    # Default PCA retains all available components here, not only two.
    scores = model.fit_transform(F)
    return {
        "eigenvalues": model.explained_variance_,
        "proportion_variance": model.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(model.explained_variance_ratio_),
        # Transpose so rows align with original features and columns with components.
        # These are component-direction coefficients, not causal feature effects.
        "loadings": model.components_.T,
        "features": list(F.columns),
        # Only the first two score columns are returned for an exploratory plot.
        # This slice does not make a two-dimensional model the actual classifier.
        "scores": scores[:, :2],
    }
