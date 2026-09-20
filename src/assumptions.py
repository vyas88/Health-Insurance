"""Legacy diagnostics only; not used by the classification workflow."""
import numpy as np
from scipy.stats import chi2, norm

# Historical normality diagnostic only. The active k-NN pipeline does not
# require it. It assumes a usable nonsingular covariance matrix; arbitrary
# constant/duplicated columns can make the inverse fail.
def mardia_test(C):
    '''
    Mardia's test checks whether multiple continuous variables
    jointly follow a multivariate normal distribution.
    H0: Data are multivariate normal.
    H1: Data are not multivariate normal.
    It checks this using multivariate skewness and kurtosis.
    '''
    x = np.asarray(C, dtype=float)
    '''Convert C into a numerical NumPy array.'''
    n, p = x.shape
    '''n = number of observations, p = number of variables.'''
    centered = x - x.mean(axis=0)
    '''Center each variable by subtracting its mean.'''
    inv_cov = np.linalg.inv(np.cov(centered, rowvar=False, ddof=0))
    '''
    Calculate the covariance matrix and its inverse.
    This captures how the variables vary and move together.
    '''
    # This n-by-n matrix includes cross-products between different centered rows.
    # Its diagonal contains squared Mahalanobis distances; off-diagonal entries
    # also contribute to the multivariate skewness calculation.
    distances = centered @ inv_cov @ centered.T
    '''
    Calculate Mahalanobis-based distances.
    These measure multivariate distance while accounting for
    variable scale and correlation.
    '''
    skewness = np.mean(distances**3)
    '''Calculate Mardia's multivariate skewness.'''
    skew_stat = n * skewness / 6
    '''Convert skewness into a chi-square test statistic.'''
    skew_df = p * (p + 1) * (p + 2) / 6
    '''Calculate degrees of freedom for the skewness test.'''
    kurtosis = np.mean(np.diag(distances) ** 2)
    '''
    Calculate Mardia's multivariate kurtosis.
    This checks whether the data have unusually heavy or light tails.
    '''
    kurt_z = (kurtosis - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    '''
    Compare observed kurtosis with the expected value under
    multivariate normality, which is p(p+2), and convert it to a Z-score.
    '''
    # Use real # comments inside dictionaries. A standalone quoted explanation
    # immediately before a quoted key can concatenate into that key in Python,
    # which was the historical defect checked by the regression test.
    # A large p-value is lack of evidence against normality, not proof of it.
    return {
        "skewness": float(skewness),
        "skewness_chi2": float(skew_stat),
        "skewness_df": float(skew_df),
        # Skewness p-value:
        # p > 0.05 -> no strong evidence against multivariate normality.
        # p < 0.05 -> significant skewness; normality assumption is violated.
        "skewness_p_value": float(chi2.sf(skew_stat, skew_df)),

        "kurtosis": float(kurtosis),
        "kurtosis_z": float(kurt_z),
        # Kurtosis p-value:
        # p > 0.05 -> kurtosis is consistent with multivariate normality.
        # p < 0.05 -> significant kurtosis deviation from normality.
        "kurtosis_p_value": float(2 * norm.sf(abs(kurt_z))),
    }

# Legacy comparison of within-group covariance matrices. This test is not
# used to choose the revised classifier and is sensitive to distributional
# assumptions. Every group needs enough observations and usable covariance.
def box_m(C, tier):
    '''Box's M test checks whether the covariance matrices of the groups are equal.
    H0: Covariance matrices are equal across groups.
    H1: At least one group's covariance matrix is different.
    Useful before methods such as MANOVA or LDA.'''
    x = np.asarray(C, dtype=float)
    '''Convert the continuous-variable dataset into a NumPy array.'''
    labels = np.asarray(tier)
    '''Convert Low, Medium and High tier labels into a NumPy array.'''
    groups = np.unique(labels)
    '''Find the unique groups: Low, Medium and High.'''
    p = x.shape[1]
    '''Store the number of variables in C, such as age, bmi and children.'''
    counts = np.array([np.sum(labels == group) for group in groups])
    '''Count how many observations belong to each group.'''
    covs = [np.cov(x[labels == group], rowvar=False, ddof=1) for group in groups]
    '''Calculate a separate covariance matrix for each group.'''
    total = counts.sum()
    '''Calculate the total number of observations across all groups.'''
    pooled = sum((count - 1) * cov for count, cov in zip(counts, covs)) / (total - len(groups))
    '''Create the pooled covariance matrix, representing the combined covariance structure of all groups.'''
    # slogdet computes sign and log absolute determinant more stably than
    # computing a potentially huge/tiny determinant first. This legacy code uses
    # the log value only and assumes valid positive-definite covariance inputs.
    logdet_pooled = np.linalg.slogdet(pooled)[1]
    '''Calculate the log determinant of the pooled covariance matrix.'''
    logdets = np.array([np.linalg.slogdet(cov)[1] for cov in covs])
    '''Calculate the log determinant of each individual group's covariance matrix.'''
    m = (total - len(groups)) * logdet_pooled - np.sum((counts - 1) * logdets)
    '''Calculate Box's M statistic by comparing the pooled covariance with the group covariance matrices.'''
    correction = (
        (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (len(groups) - 1))
        * (np.sum(1 / (counts - 1)) - 1 / (total - len(groups)))
    )
    '''Calculate a correction factor so Box's M can be approximated using a chi-square distribution.'''
    statistic = (1 - correction) * m
    '''Apply the correction to obtain the chi-square-style test statistic.'''
    # There are p(p+1)/2 distinct covariance entries per group. Comparing group
    # matrices gives (groups-1) times that count as the approximation's degrees
    # of freedom. A nonsignificant result does not establish equal covariances.
    df = (len(groups) - 1) * p * (p + 1) / 2
    '''Calculate the degrees of freedom based on the number of groups and variables.'''
    '''Interpretation:
    p-value > 0.05 -> no strong evidence that covariance matrices differ; assumption is reasonably satisfied.
    p-value < 0.05 -> covariance matrices differ significantly; equal-covariance assumption is violated.'''
    return {
        "m": float(m),
        "chi2": float(statistic),
        "df": float(df),
        "p_value": float(chi2.sf(statistic, df)),
        "group_sizes": {str(group): int(count) for group, count in zip(groups, counts)},
    }
