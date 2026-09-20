# LEGACY ONLY: these hypothesis tests and distance summaries are preserved
# for study, not used by applicant inference. The current application uses
# src/inference.py. MANOVA here requires the optional statsmodels dependency.
import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from statsmodels.multivariate.manova import MANOVA

# Compare High and Low mean vectors jointly rather than testing one variable
# at a time. The classical two-sample calculation assumes independent groups,
# a common nonsingular covariance and suitable distributional conditions.
def hotelling_t2(C_high, C_low):
    high = np.asarray(C_high, dtype=float)
    low = np.asarray(C_low, dtype=float)
    n1, p = high.shape
    n2 = len(low)
    # Pool the sample covariances using their n-1 degrees-of-freedom weights.
    # The inverse scales mean differences by variation and correlation, not just
    # the original variable units.
    pooled = ((n1 - 1) * np.cov(high, rowvar=False) + (n2 - 1) * np.cov(low, rowvar=False)) / (n1 + n2 - 2)
    difference = high.mean(axis=0) - low.mean(axis=0)
    t2 = n1 * n2 / (n1 + n2) * difference @ np.linalg.inv(pooled) @ difference
    # Convert T-squared into an F statistic for the classical reference test.
    # Small or degenerate samples can invalidate these degrees of freedom.
    df1 = p
    df2 = n1 + n2 - p - 1
    f_stat = df2 / ((n1 + n2 - 2) * p) * t2

    return {
        "t2": float(t2),
        "f": float(f_stat),
        "df1": int(df1),
        "df2": int(df2),
        "p_value": float(f.sf(f_stat, df1, df2)),
    }

# Calculate historical within-tier squared distances from each group centroid.
# These values are not the current k-NN similarity metric or an app assessment.
def mahalanobis(C, tier):
    x = np.asarray(C, dtype=float)
    labels = np.asarray(tier)
    scores = np.empty(len(x))
    for group in np.unique(labels):
        mask = labels == group
        values = x[mask]
        centered = values - values.mean(axis=0)
        inv_cov = np.linalg.inv(np.cov(values, rowvar=False, ddof=1))
        # einsum evaluates x_i^T S^-1 x_i for each centered row without constructing
        # a full row-by-row product matrix. This retains cross-variable covariance.
        scores[mask] = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)

    # The chi-square reference cutoff is an approximation requiring assumptions.
    # Crossing it does not by itself identify illness or prove an anomalous person.
    # The revised user journey deliberately omits this typicality assessment.
    threshold = chi2.ppf(0.975, df=x.shape[1])
    return {
        "scores": scores,
        "outliers": scores > threshold,
        "threshold": float(threshold),
        "count": int(np.sum(scores > threshold)),
    }

# Test whether multivariate means differ across tier groups in the legacy
# analysis. The tier is a grouping factor, not an additional numeric response.
def manova_wilks(C, tier):
    data = pd.DataFrame(C).copy()
    data["tier"] = np.asarray(tier)
    # Construct statsmodels formula syntax: multiple outcomes on the left, group
    # factor on the right. mv_test returns several multivariate statistics;
    # this helper extracts the Wilks lambda row and its approximate F test.
    formula = " + ".join(C.columns) + " ~ tier"
    stat = MANOVA.from_formula(formula, data=data).mv_test().results["tier"]["stat"]
    wilks = stat.loc["Wilks' lambda"]

    return {
        "wilks_lambda": float(wilks["Value"]),
        "f": float(wilks["F Value"]),
        "num_df": float(wilks["Num DF"]),
        "den_df": float(wilks["Den DF"]),
        "p_value": float(wilks["Pr > F"]),
    }
