import numpy as np
from scipy.stats import chi2, norm


def mardia_test(C):
    x = np.asarray(C, dtype=float)
    n, p = x.shape
    centered = x - x.mean(axis=0)
    inv_cov = np.linalg.inv(np.cov(centered, rowvar=False, ddof=0))
    distances = centered @ inv_cov @ centered.T
    skewness = np.mean(distances**3)
    skew_stat = n * skewness / 6
    skew_df = p * (p + 1) * (p + 2) / 6
    kurtosis = np.mean(np.diag(distances) ** 2)
    kurt_z = (kurtosis - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)

    return {
        "skewness": float(skewness),
        "skewness_chi2": float(skew_stat),
        "skewness_df": float(skew_df),
        "skewness_p_value": float(chi2.sf(skew_stat, skew_df)),
        "kurtosis": float(kurtosis),
        "kurtosis_z": float(kurt_z),
        "kurtosis_p_value": float(2 * norm.sf(abs(kurt_z))),
    }


def box_m(C, tier):
    x = np.asarray(C, dtype=float)
    labels = np.asarray(tier)
    groups = np.unique(labels)
    p = x.shape[1]
    counts = np.array([np.sum(labels == group) for group in groups])
    covs = [np.cov(x[labels == group], rowvar=False, ddof=1) for group in groups]
    total = counts.sum()
    pooled = sum((count - 1) * cov for count, cov in zip(counts, covs)) / (total - len(groups))
    logdet_pooled = np.linalg.slogdet(pooled)[1]
    logdets = np.array([np.linalg.slogdet(cov)[1] for cov in covs])
    m = (total - len(groups)) * logdet_pooled - np.sum((counts - 1) * logdets)
    correction = (
        (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (len(groups) - 1))
        * (np.sum(1 / (counts - 1)) - 1 / (total - len(groups)))
    )
    statistic = (1 - correction) * m
    df = (len(groups) - 1) * p * (p + 1) / 2

    return {
        "m": float(m),
        "chi2": float(statistic),
        "df": float(df),
        "p_value": float(chi2.sf(statistic, df)),
        "group_sizes": {str(group): int(count) for group, count in zip(groups, counts)},
    }
