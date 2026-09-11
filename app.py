import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.ai_interpretation import (
    AIInterpretationError,
    build_ai_context,
    create_client,
    generate_ai_interpretation,
    get_ai_settings,
)


ROOT = Path(__file__).resolve().parent
CONTINUOUS = ["age", "bmi", "children"]
ALPHA = 0.05
PAGES = [
    "Risk Assessment",
    "Understand the Model",
    "Evidence & Diagnostics",
    "Methodology",
    "Dataset",
]


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_data
def load_results():
    with (ROOT / "outputs" / "results.json").open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def get_openai_client(api_key):
    return create_client(api_key)


def inject_styles():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #f5f7fb; color: #102a43; }
        [data-testid="stHeader"] { background: rgba(245, 247, 251, 0.92); }
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
        [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p {
            color: #102a43 !important;
        }
        [data-testid="stSidebar"] { background: #102a43; }
        [data-testid="stSidebar"] * { color: #f7fafc; }
        h1, h2, h3 { color: #102a43; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 0.8rem;
        }
        div[data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
        }
        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 1rem;
        }
        [data-testid="stFormSubmitButton"] button,
        .stButton > button {
            background: #1f5d9b;
            border: 1px solid #1f5d9b;
            color: #ffffff !important;
        }
        [data-testid="stFormSubmitButton"] button:hover,
        .stButton > button:hover { background: #174a7a; border-color: #174a7a; }
        .result-label { color: #486581; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.15rem; }
        .result-tier { color: #102a43; font-size: 2.3rem; font-weight: 700; line-height: 1.1; }
        .result-copy { color: #334e68; font-size: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def figure_path(name):
    return str(ROOT / "outputs" / "figures" / name)


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


def format_p(value):
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def technical_detail(rows, title="Technical detail"):
    with st.expander(title):
        st.dataframe(
            pd.DataFrame(rows, columns=["Statistic", "Value"]),
            hide_index=True,
            use_container_width=True,
        )


def predict_applicant(bundle, results, profile):
    row = build_row(**profile, features=bundle["features"])
    scaled = pd.DataFrame(bundle["scaler"].transform(row), columns=bundle["features"])
    classifier = bundle["classifier"]
    tier = str(classifier.predict(scaled)[0])
    probabilities = {
        str(label): float(probability)
        for label, probability in zip(classifier.classes_, classifier.predict_proba(scaled)[0])
    }
    reference = bundle["centroids"][tier]
    distance = mahalanobis(
        scaled.loc[0, CONTINUOUS].to_numpy(),
        np.asarray(reference["mean"]),
        np.asarray(reference["cov"]),
    )
    cutoff = results["mahalanobis"]["threshold"]
    return {
        "profile": profile,
        "tier": tier,
        "probabilities": probabilities,
        "cost_range": cost_band(tier, bundle["tier_edges"]),
        "distance": distance,
        "cutoff": cutoff,
        "status": "Typical" if distance <= cutoff else "Atypical",
    }


def assessment_signature(assessment):
    return json.dumps(assessment["profile"], sort_keys=True)


def show_ai_interpretation(assessment, results):
    st.subheader("Optional AI interpretation")
    st.write("Want a plain-English explanation of this result?")
    signature = assessment_signature(assessment)
    if st.button("Explain my result with AI", key=f"ai_{signature}"):
        api_key, model = get_ai_settings()
        if not api_key:
            st.info("AI interpretation is currently unavailable. The statistical assessment above is still valid and fully available.")
            return
        context = build_ai_context(assessment["profile"], assessment, results)
        try:
            with st.spinner("Preparing a plain-language interpretation..."):
                explanation = generate_ai_interpretation(get_openai_client(api_key), model, context)
            st.session_state["ai_interpretation"] = {"signature": signature, "text": explanation}
        except AIInterpretationError:
            st.info("AI interpretation is currently unavailable. The statistical assessment above is still valid and fully available.")

    response = st.session_state.get("ai_interpretation")
    if response and response["signature"] == signature:
        st.subheader("AI-assisted interpretation")
        st.markdown(response["text"])
        st.caption("AI-assisted interpretation. Quantitative results are generated by the statistical model.")
    st.caption("AI-assisted interpretation is for explanation only. It does not change the statistical result and is not medical advice or an insurance quote.")


def risk_assessment(bundle, results):
    st.title("Multivariate Health Insurance Risk Profiling")
    st.write("A multivariate statistical assessment of applicant characteristics using observed medical-cost patterns.")
    st.info("This analysis uses observed medical costs in the dataset. It is not an insurer's quoted premium.")

    st.subheader("Applicant details")
    with st.form("risk_form"):
        left, right = st.columns(2)
        profile = {
            "age": left.number_input("Age", min_value=18, max_value=64, value=35, step=1),
            "bmi": right.number_input("BMI", min_value=15.0, max_value=55.0, value=27.5, step=0.1, format="%.1f"),
            "children": left.number_input("Children", min_value=0, max_value=5, value=0, step=1),
            "sex": right.selectbox("Sex", ["female", "male"]),
            "smoker": left.selectbox("Smoking status", ["no", "yes"]),
            "region": right.selectbox("Region", ["northwest", "northeast", "southeast", "southwest"]),
        }
        assess = st.form_submit_button("Assess risk", type="primary")

    if assess:
        st.session_state["assessment"] = predict_applicant(bundle, results, profile)
        st.session_state.pop("ai_interpretation", None)

    assessment = st.session_state.get("assessment")
    if not assessment:
        st.caption("The model considers several applicant characteristics together, rather than scoring each one separately.")
        return

    st.divider()
    st.subheader("Risk assessment")
    st.markdown(f'<div class="result-label">PREDICTED RISK GROUP</div><div class="result-tier">{assessment["tier"].upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-copy">The model places this profile in the {assessment["tier"]}-risk group represented in the historical dataset.</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Model support")
        st.caption("How strongly the classifier supports each historical risk group. This is not a probability of illness or a future claim.")
        for label in ["Low", "Medium", "High"]:
            probability = assessment["probabilities"].get(label, 0.0)
            st.write(f"{label}: {probability:.0%}")
            st.progress(int(round(probability * 100)))
    with right:
        st.subheader("Profile typicality")
        st.markdown(f'<div class="result-tier">{assessment["status"].upper()}</div>', unsafe_allow_html=True)
        typicality_text = (
            "The applicant's combined age, BMI and children profile is within the reference range for the predicted group."
            if assessment["status"] == "Typical"
            else "The applicant's combined age, BMI and children profile is relatively unusual for the predicted group."
        )
        st.write(typicality_text)
        technical_detail([
            ("Mahalanobis distance", f"{assessment['distance']:.2f}"),
            ("Reference cutoff", f"{assessment['cutoff']:.2f}"),
        ])

    st.subheader("Observed annual medical-cost range")
    st.metric("Historical range for the predicted group", assessment["cost_range"])
    st.caption("This range summarizes observed annual medical costs in the dataset. It is not a premium quote or a guaranteed future cost.")

    with st.expander("Why this result?"):
        st.markdown(
            "1. The applicant details are converted to the standardized format used during model training.\n"
            "2. The classifier compares the profile with the learned Low, Medium and High historical groups.\n"
            "3. The group with the strongest model support is selected.\n"
            "4. Profile typicality checks how similar the combined continuous profile is to other profiles in that group."
        )
    show_ai_interpretation(assessment, results)


def model_overview(results):
    st.title("Understand the Model")
    st.write("This project uses multivariate statistics to consider several applicant characteristics together, rather than looking at one variable at a time.")
    steps = [
        ("1. Applicant profile", "Age, BMI, children, sex, smoking status and region are collected."),
        ("2. Data preparation", "The details are converted to the same standardized numerical format used to train the model."),
        ("3. Multivariate analysis", "The project examines combined patterns, group differences and data assumptions."),
        ("4. Risk classification", "The selected discriminant model places the profile in a Low, Medium or High observed-cost group."),
        ("5. Profile interpretation", "A typicality check and observed cost range make the classification easier to understand."),
    ]
    for heading, copy in steps:
        st.markdown(f"**{heading}**  \n{copy}")

    st.subheader("Why was QDA selected?")
    st.write(
        "LDA assumes the risk groups have similar covariance structures. The diagnostic evidence indicates that those structures differ, "
        "so QDA is the more appropriate classifier because it allows each group to have its own covariance structure."
    )
    st.caption("The two methods have similar cross-validation performance. QDA is selected for its assumption fit, not because it is automatically more accurate.")
    technical_detail([
        ("Selected classifier", results["classification"]["method"]),
        ("Box's M p-value", format_p(results["box_m"]["p_value"])),
        ("LDA cross-validation accuracy", f"{results['classification']['comparison']['lda_cv_mean']:.1%}"),
        ("QDA cross-validation accuracy", f"{results['classification']['comparison']['qda_cv_mean']:.1%}"),
    ], "Statistical evidence")


def assumptions_tab(results):
    mardia = results["mardia"]
    box = results["box_m"]
    st.subheader("Assumption checks")
    left, right = st.columns(2)
    left.metric("Multivariate normality", "Not supported")
    right.metric("Covariance homogeneity", "Not supported")
    st.write("The continuous variables do not closely follow an exact multivariate normal pattern, and risk groups have different covariance structures.")
    st.write("This does not make the project unusable. It means the results should be interpreted as an empirical analysis of this dataset, and it supports QDA over a common-covariance classifier.")
    technical_detail([
        ("Mardia skewness p-value", format_p(mardia["skewness_p_value"])),
        ("Mardia kurtosis p-value", format_p(mardia["kurtosis_p_value"])),
        ("Box's M statistic", f"{box['m']:.2f}"),
        ("Box's M p-value", format_p(box["p_value"])),
        ("Decision level", "alpha = 0.05"),
    ])
    st.subheader("Data completeness")
    st.metric("Missing values requiring imputation", results["data"]["imputed_cells"])


def pca_tab(results):
    pca = results["pca"]
    components_needed = next(i + 1 for i, value in enumerate(pca["cumulative_variance"]) if value >= 0.90)
    st.subheader("What does PCA tell us?")
    st.write("PCA summarizes broad patterns in the variables into new combined directions called principal components.")
    st.metric("Key finding", f"{components_needed} components are needed to retain at least 90% of variation")
    st.write("The predictors have limited redundancy, so PCA provides limited dimensional compression in this dataset.")
    st.image(figure_path("pca_scree.png"), caption="Scree plot showing the variation explained by each component.", use_container_width=True)
    st.image(figure_path("pca_biplot.png"), caption="PCA biplot by observed risk tier.", use_container_width=True)
    st.subheader("How strongly are the variables related?")
    st.write("Most feature pairs have weak linear relationships, which helps explain why PCA does not compress the dataset dramatically.")
    st.image(figure_path("feature_correlation.png"), caption="Correlation heatmap for the standardized model features.", use_container_width=True)
    loading_table = pd.DataFrame({
        "Feature": pca["features"],
        "PC1 loading": [row[0] for row in pca["loadings"]],
        "PC2 loading": [row[1] for row in pca["loadings"]],
    })
    st.dataframe(loading_table.style.format({"PC1 loading": "{:.3f}", "PC2 loading": "{:.3f}"}), hide_index=True, use_container_width=True)
    technical_detail([
        ("PC1 variance explained", f"{pca['proportion_variance'][0]:.1%}"),
        ("Original features", len(pca["features"])),
        *[(f"PC{i + 1} eigenvalue", f"{value:.3f}") for i, value in enumerate(pca["eigenvalues"])],
    ])


def classification_tab(results):
    classification = results["classification"]
    comparison = classification["comparison"]
    st.subheader("How well does the model separate the risk groups?")
    st.metric("Model performance", f"About {classification['overall_accuracy']:.0%} of historical records were classified correctly")
    st.write("This result comes from five-fold cross-validation, where the model is repeatedly checked on records held back from training.")
    st.subheader("LDA and QDA comparison")
    st.dataframe(pd.DataFrame([
        {"Method": "LDA", "Cross-validation accuracy": comparison["lda_cv_mean"], "Variation across folds": comparison["lda_cv_sd"]},
        {"Method": "QDA", "Cross-validation accuracy": comparison["qda_cv_mean"], "Variation across folds": comparison["qda_cv_sd"]},
    ]).style.format({"Cross-validation accuracy": "{:.1%}", "Variation across folds": "{:.1%}"}), hide_index=True, use_container_width=True)
    st.write("QDA is selected because the covariance structures differ across groups. The performance of LDA and QDA is very similar.")
    st.subheader("Where does the model confuse groups?")
    labels = classification["labels"]
    confusion = pd.DataFrame(classification["confusion_matrix"], index=labels, columns=labels)
    confusion.index.name = "Actual risk group"
    confusion.columns.name = "Predicted risk group"
    st.dataframe(confusion, use_container_width=True)
    st.write("Low and Medium groups are classified relatively well. Some High-risk observations overlap with the Medium group.")
    st.image(figure_path("qda_regions.png"), caption="Illustrative QDA decision regions using PC1 and PC2. The reported classifier uses all eight standardized features.", use_container_width=True)
    technical_detail([
        ("Selected model", classification["method"]),
        ("Overall cross-validation accuracy", f"{classification['overall_accuracy']:.1%}"),
        ("QDA regularization", classification["regularization"]),
    ])


def inference_tab(results):
    hotelling = results["hotelling_t2"]
    manova = results["manova"]
    st.subheader("Do High- and Low-risk groups have different combined profiles?")
    st.metric("Answer", "Yes, strong evidence of a difference")
    st.write("Hotelling's T² tests whether the two groups differ in their combined average profile across age, BMI and children.")
    technical_detail([
        ("T² statistic", f"{hotelling['t2']:.2f}"),
        ("F statistic", f"{hotelling['f']:.2f}"),
        ("Numerator df", hotelling["df1"]),
        ("Denominator df", hotelling["df2"]),
        ("p-value", format_p(hotelling["p_value"])),
        ("Decision", "Reject the null hypothesis"),
    ])
    st.subheader("Do the risk groups differ when variables are considered together?")
    st.metric("Answer", "Yes, strong evidence of a group difference")
    st.write("MANOVA checks whether Low, Medium and High groups differ in their combined continuous-variable profiles.")
    technical_detail([
        ("Wilks' Lambda", f"{manova['wilks_lambda']:.4f}"),
        ("F statistic", f"{manova['f']:.2f}"),
        ("Numerator df", f"{manova['num_df']:.0f}"),
        ("Denominator df", f"{manova['den_df']:.0f}"),
        ("p-value", format_p(manova["p_value"])),
        ("Decision", "Reject the null hypothesis"),
    ])


def mahalanobis_tab(results):
    details = results["mahalanobis"]
    st.subheader("How typical are the historical profiles?")
    st.metric("Profiles outside the reference range", f"{details['count']} of {results['data']['n']}")
    st.write("Mahalanobis distance measures how typical a combined continuous profile is within its own risk group while accounting for relationships among the variables.")
    st.image(figure_path("mahalanobis_hist.png"), caption="Within-tier squared Mahalanobis distances in the historical dataset.", use_container_width=True)
    technical_detail([
        ("Reference cutoff", f"{details['threshold']:.2f}"),
        ("Flagged profiles", details["count"]),
        ("Continuous variables used", "age, BMI, children"),
    ])


def evidence_diagnostics(results):
    st.title("Evidence & Diagnostics")
    st.write("Supporting statistical evidence is available here without interrupting the main risk-assessment journey.")
    assumptions, pca, classification, inference, mahalanobis = st.tabs([
        "Assumptions", "PCA", "Classification", "Inference", "Mahalanobis"
    ])
    with assumptions:
        assumptions_tab(results)
    with pca:
        pca_tab(results)
    with classification:
        classification_tab(results)
    with inference:
        inference_tab(results)
    with mahalanobis:
        mahalanobis_tab(results)


def methodology():
    st.title("Methodology")
    st.write("This page keeps the mathematical reference material available for viva discussion without placing it in the primary user journey.")
    st.subheader("Analysis workflow")
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#486581", fontname="Arial"];
            input [label="Applicant or dataset input"];
            prep [label="Preprocessing and standardization"];
            diagnostics [label="Mardia and Box's M diagnostics"];
            select [shape=diamond, label="Equal covariance supported?"];
            lda [label="LDA"];
            qda [label="QDA"];
            evidence [label="PCA, inference and Mahalanobis analysis"];
            output [label="Risk group and interpretation"];
            input -> prep -> diagnostics -> select;
            select -> lda [label="yes"];
            select -> qda [label="no"];
            lda -> evidence -> output;
            qda -> evidence;
        }
    """, use_container_width=True)
    with st.expander("Step-by-step decision logic"):
        st.markdown(
            "1. Load the dataset and handle any missing values.\n"
            "2. Standardize continuous and encoded predictor variables.\n"
            "3. Run Mardia and Box's M diagnostics.\n"
            "4. Compare LDA and QDA with five-fold cross-validation.\n"
            "5. Select QDA when Box's M rejects equal covariance, otherwise select LDA.\n"
            "6. Produce risk classification, observed cost band and Mahalanobis typicality.\n"
            "7. Use PCA and multivariate inference to interpret the group structure."
        )

    methods = [
        ("PCA", "Summarizes broad directions of variation in several variables.", r"S v = \lambda v", "Used to inspect dimensional structure, not as the production classifier input."),
        ("Mahalanobis distance", "Measures multivariate distance while accounting for covariance.", r"D^2 = (x - \mu)'\Sigma^{-1}(x - \mu)", "Used to assess profile typicality within the predicted group."),
        ("LDA", "Classifies groups under a common covariance assumption.", r"\delta_k(x) = x'\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k'\Sigma^{-1}\mu_k + \log \pi_k", "Retained as the comparison classifier."),
        ("QDA", "Classifies groups with a separate covariance structure for each group.", r"\delta_k(x) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x-\mu_k)'\Sigma_k^{-1}(x-\mu_k) + \log \pi_k", "Selected when equal covariance is not supported."),
        ("Hotelling's T²", "Tests whether two multivariate mean profiles differ.", r"T^2 = \frac{n_1 n_2}{n_1+n_2}(\bar{x}_1-\bar{x}_2)'S_p^{-1}(\bar{x}_1-\bar{x}_2)", "Compares High- and Low-risk profiles."),
        ("MANOVA", "Tests whether several groups differ on a combined set of variables.", r"\Lambda = \frac{|E|}{|E+H|}", "Tests differences across all three risk groups."),
        ("Mardia's test", "Checks how closely continuous variables follow a multivariate normal pattern.", r"b_{1,p} = \frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}[(x_i-\bar{x})'S^{-1}(x_j-\bar{x})]^3", "Used as a multivariate normality diagnostic."),
        ("Box's M", "Checks whether covariance matrices are similar across groups.", r"M = (N-g)\log|S_p| - \sum_{i=1}^{g}(n_i-1)\log|S_i|", "Guides the LDA versus QDA assumption decision."),
    ]
    for name, what, formula, use in methods:
        with st.expander(name):
            st.write(f"What it is: {what}")
            st.write(f"Why it is used here: {use}")
            st.latex(formula)


def dataset(results):
    data = results["data"]
    st.title("Dataset")
    st.write("The model learns Low, Medium and High groups from observed annual medical costs in the insurance dataset.")
    left, middle, right = st.columns(3)
    left.metric("Historical observations", f"{data['n']:,}")
    middle.metric("Original variables", 7)
    right.metric("Model features", 8)
    st.subheader("Variables used")
    st.write("Original dataset variables: age, sex, BMI, children, smoker, region and charges.")
    st.write("The classifier uses eight standardized features: age, BMI, children, sex indicator, smoker indicator and three region indicators.")
    edges = data["tercile_edges"]
    st.subheader("Observed annual medical-cost groups")
    st.dataframe(pd.DataFrame([
        {"Risk group": "Low", "Observed annual medical-cost range": f"${edges[0]:,.0f} to ${edges[1]:,.0f}"},
        {"Risk group": "Medium", "Observed annual medical-cost range": f"${edges[1]:,.0f} to ${edges[2]:,.0f}"},
        {"Risk group": "High", "Observed annual medical-cost range": f"${edges[2]:,.0f} to ${edges[3]:,.0f}"},
    ]), hide_index=True, use_container_width=True)
    st.caption("These groups are dataset-based observed cost bands, not insurance premium quotations.")


def main():
    st.set_page_config(page_title="Multivariate Health Insurance Risk Profiling", page_icon="", layout="wide")
    inject_styles()
    bundle = load_bundle()
    results = load_results()

    with st.sidebar:
        st.title("Risk Profiling")
        page = st.radio("Navigate", PAGES)
        st.caption("University multivariate techniques project")

    pages = {
        "Risk Assessment": lambda: risk_assessment(bundle, results),
        "Understand the Model": lambda: model_overview(results),
        "Evidence & Diagnostics": lambda: evidence_diagnostics(results),
        "Methodology": methodology,
        "Dataset": lambda: dataset(results),
    }
    pages[page]()


if __name__ == "__main__":
    main()
