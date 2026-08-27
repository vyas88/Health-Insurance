from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("/Users/ronitvyas/Downloads/insurance.csv")


def build_data(path=DATA_PATH):
    data = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    missing = int(data.isna().sum().sum())

    if missing:
        numeric = data.select_dtypes(include=np.number).columns
        categorical = data.columns.difference(numeric)
        data[numeric] = data[numeric].fillna(data[numeric].median())
        for col in categorical:
            data[col] = data[col].fillna(data[col].mode().iat[0])

    y_cost = data["charges"].copy()
    y_log = np.log(y_cost)
    tier, edges = pd.qcut(
        y_cost, q=3, labels=["Low", "Medium", "High"], retbins=True
    )
    tier = tier.astype("category")

    continuous = ["age", "bmi", "children"]
    scaler = StandardScaler()
    C = pd.DataFrame(scaler.fit_transform(data[continuous]), columns=continuous)

    categorical = pd.get_dummies(
        data[["sex", "smoker", "region"]], drop_first=True, dtype=float
    )
    full = pd.concat([data[continuous], categorical], axis=1)
    full_scaler = StandardScaler()
    F = pd.DataFrame(full_scaler.fit_transform(full), columns=full.columns)

    # Binary indicators are unsuitable for MVN tests; Low-tier smokers are constant.
    C.attrs["imputed_cells"] = missing
    C.attrs["n_rows"] = len(data)
    return C, F, tier, y_cost, y_log, full_scaler, edges
