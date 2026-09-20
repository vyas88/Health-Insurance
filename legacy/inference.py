import numpy as np
import pandas as pd
from scipy.stats import chi2, f
from statsmodels.multivariate.manova import MANOVA

def hotelling_t2(C_high, C_low):
    high = np.asarray(C_high, dtype=float)
    low = np.asarray(C_low, dtype=float)
    n1, p = high.shape
    n2 = len(low)
    pooled = ((n1 - 1) * np.cov(high, rowvar=False) + (n2 - 1) * np.cov(low, rowvar=False)) / (n1 + n2 - 2)
    difference = high.mean(axis=0) - low.mean(axis=0)
    t2 = n1 * n2 / (n1 + n2) * difference @ np.linalg.inv(pooled) @ difference
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

def mahalanobis(C, tier):
    x = np.asarray(C, dtype=float)
    labels = np.asarray(tier)
    scores = np.empty(len(x))
    for group in np.unique(labels):
        mask = labels == group
        values = x[mask]
        centered = values - values.mean(axis=0)
        inv_cov = np.linalg.inv(np.cov(values, rowvar=False, ddof=1))
        scores[mask] = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)

    threshold = chi2.ppf(0.975, df=x.shape[1])
    return {
        "scores": scores,
        "outliers": scores > threshold,
        "threshold": float(threshold),
        "count": int(np.sum(scores > threshold)),
    }

def manova_wilks(C, tier):
    data = pd.DataFrame(C).copy()
    data["tier"] = np.asarray(tier)
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
