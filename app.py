from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
CONTINUOUS = ["age", "bmi", "children"]


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "model.joblib")


def build_row(age, bmi, children, sex, smoker, region, features):
    row = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": int(sex == "male"),
        "smoker_yes": int(smoker == "yes"),
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest"),
    }])
    return row.reindex(columns=features, fill_value=0)


def mahalanobis(point, center, cov):
    difference = point - center
    return float(difference @ np.linalg.solve(cov, difference))


def cost_band(tier, edges):
    bounds = {"Low": (0, 1), "Medium": (1, 2), "High": (2, 3)}
    low, high = bounds[tier]
    return f"${edges[low]:,.0f} to ${edges[high]:,.0f}"


def main():
    st.set_page_config(page_title="Health insurance risk profile", layout="centered")
    bundle = load_bundle()
    classifier = bundle["classifier"]
    scaler = bundle["scaler"]
    features = bundle["features"]

    st.title("Health insurance risk profile")
    st.write("Enter applicant attributes to estimate a health-insurance risk tier and its observed annual medical-cost range using a multivariate QDA model.")
    left, right = st.columns(2)
    age = left.number_input("Age", min_value=18, max_value=64, value=35, step=1)
    bmi = right.number_input("BMI", min_value=15.0, max_value=55.0, value=27.5, step=0.1, format="%.1f")
    children = left.number_input("Children", min_value=0, max_value=5, value=0, step=1)
    sex = right.selectbox("Sex", ["female", "male"])
    smoker = left.selectbox("Smoker", ["no", "yes"])
    region = right.selectbox("Region", ["northwest", "northeast", "southeast", "southwest"])

    if st.button("Predict risk tier", type="primary"):
        row = build_row(age, bmi, children, sex, smoker, region, features)
        scaled = pd.DataFrame(scaler.transform(row), columns=features)
        tier = classifier.predict(scaled)[0]
        point = scaled.loc[0, CONTINUOUS].to_numpy()
        reference = bundle["centroids"][tier]
        center = np.asarray(reference["mean"])
        cov = np.asarray(reference["cov"])
        distance = mahalanobis(point, center, cov)

        st.subheader(f"Predicted tier: {tier}")
        st.metric("Observed annual medical-cost range", cost_band(tier, bundle["tier_edges"]))
        st.caption("A real premium would add risk and expense loadings to this observed cost range.")
        st.metric("Squared Mahalanobis distance", f"{distance:.2f}")
        st.caption("Values above the χ²₀.₉₇₅ (df=3) cutoff of 9.35 are atypical for the predicted tier.")


if __name__ == "__main__":
    main()
