# HISTORICAL SNAPSHOT: existing educational explanations below are preserved.
# This module is not imported by the revised analysis. Some original choices
# are unsuitable for validation; the additional notes identify why.
# Use src/data.py and src/classify.py for the current implementation.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "insurance.csv"

def build_data(path=DATA_PATH):
    """
    Build and prepare the insurance dataset for statistical analysis
    and machine learning.
    Check whether the dataset exists at the given path.
    If it does not exist, stop the program and show an error.
    """
    if not Path(path).exists():

        raise FileNotFoundError(
            f"Dataset not found at {path}. Place insurance.csv in the project's data folder."
        )
    """
    Read the CSV file.
    drop_duplicates():
    Removes duplicate rows.
    reset_index(drop=True):
    Resets row numbers after removing duplicates.
    """
    # Historical duplicate removal loses the original row mapping. The revised
    # loader retains source-row IDs before resetting its working DataFrame index.
    data = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    """
    Count the total number of missing values in the entire dataset.
    """
    missing = int(data.isna().sum().sum())
    """
    If missing values exist, separate numerical and categorical columns
    and fill the missing values.
    """
    # Historical behaviour only: this block can impute charges and learn predictor
    # imputation from all rows. The revised method drops invalid targets and
    # learns predictor imputation within each development training fold instead.
    if missing:
        """
        Identify numerical columns such as:
        age, bmi, children, charges
        """
        numeric = data.select_dtypes(include=np.number).columns
        """
        Identify categorical columns such as:
        sex, smoker and region.
        """
        categorical = data.columns.difference(numeric)
        """
        Fill missing numerical values using the median.
        Example:
        BMI = [22, 25, NaN, 30]
        The missing value is replaced using the median.
        """
        data[numeric] = data[numeric].fillna(data[numeric].median())
        """
        Fill missing categorical values using the mode,
        meaning the most frequently occurring category.
        """
        for col in categorical:

            data[col] = data[col].fillna(data[col].mode().iat[0])
    """
    Store the original insurance charges.
    y_cost becomes the target variable containing
    the actual insurance/medical costs.
    """
    y_cost = data["charges"].copy()
    """
    Take the natural logarithm of insurance charges.
    Insurance charges can be highly skewed because some people
    have extremely large costs.
    Log transformation helps make the distribution more balanced.
    """
    # Log charges were used by the old analysis, not by current neighbour distance.
    # Zero charges would yield negative infinity here; the revised classifier does
    # not need this transformation.
    y_log = np.log(y_cost)
    """
    Divide insurance charges into 3 approximately equal groups:
    Low     -> lowest charges
    Medium  -> middle charges
    High    -> highest charges
    q=3 means divide into 3 quantiles.
    edges stores the actual cutoff values between these groups.
    """
    # This uses the complete dataset to define tiers, unlike the current frozen
    # development-only cutoffs. Do not reuse these edges for revised holdout metrics.
    tier, edges = pd.qcut(
        y_cost,
        q=3,
        labels=["Low", "Medium", "High"],
        retbins=True
    )
    """
    Tell pandas that Low, Medium and High
    are categorical values.
    """
    tier = tier.astype("category")
    """
    Select the continuous numerical variables
    that will be used for statistical analysis.
    """
    # Despite this historical variable name, children is a discrete count, not
    # a continuous measurement. The revised comments and UI make that distinction.
    continuous = ["age", "bmi", "children"]
    """
    Create a StandardScaler.
    StandardScaler converts variables to a common scale
    using approximately:
    z = (value - mean) / standard deviation
    """
    scaler = StandardScaler()
    """
    C = Continuous-variable dataset.
    Standardize:
    - age
    - bmi
    - children
    C is mainly useful for statistical analysis involving
    continuous variables.
    """
    C = pd.DataFrame(
        scaler.fit_transform(data[continuous]),
        columns=continuous
    )
    """
    Convert categorical variables into numerical binary variables.
    Example:
    smoker = yes/no
    becomes something like:
    smoker_yes
    1 = yes
    0 = no
    drop_first=True removes one reference category
    to avoid redundant variables.
    """
    categorical = pd.get_dummies(
        data[["sex", "smoker", "region"]],
        drop_first=True,
        dtype=float
    )
    """
    Combine the original continuous variables
    with the encoded categorical variables.
    The resulting 'full' dataset may contain:
    age
    bmi
    children
    sex_male
    smoker_yes
    region_northwest
    region_southeast
    region_southwest
    """
    full = pd.concat(
        [data[continuous], categorical],
        axis=1
    )
    """
    Create another scaler for the complete feature dataset.
    """
    # The old scaler standardizes all encoded columns on the full dataset.
    # Current k-NN instead scales only numerical variables within folds and leaves
    # full one-hot category indicators unscaled.
    full_scaler = StandardScaler()
    """
    F = Full feature matrix.
    Standardize all features in 'full'.
    Difference:
    C -> only continuous variables
    F -> continuous + encoded categorical variables
    """
    F = pd.DataFrame(
        full_scaler.fit_transform(full),
        columns=full.columns
    )
    """
    Binary indicators are unsuitable for some multivariate
    normality (MVN) tests.
    Also, Low-tier smokers may be constant in some subsets,
    which can cause problems in statistical tests.
    """
    """
    Store metadata inside C.
    imputed_cells:
    Number of missing values that were filled.
    These are attributes only.
    They do NOT become dataframe columns.
    """
    C.attrs["imputed_cells"] = missing
    """
    Store the total number of observations/rows
    in the cleaned dataset.
    """
    C.attrs["n_rows"] = len(data)
    """
    Return everything created by this function.
    C           -> standardized continuous variables
    F           -> standardized full feature matrix
    tier        -> Low / Medium / High cost group
    y_cost      -> original insurance charges
    y_log       -> log-transformed charges
    full_scaler -> scaler used for all features
    edges       -> cutoff values for Low / Medium / High
    """
    return C, F, tier, y_cost, y_log, full_scaler, edges