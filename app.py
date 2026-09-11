import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
CONTINUOUS = ["age", "bmi", "children"]
ALPHA = 0.05
PAGE_OPTIONS = [
    "Risk Assessment",
    "Model Overview",
    "Assumptions & Diagnostics",
    "PCA Analysis",
    "Classification",
    "Statistical Inference",
    "Dataset",
    "Methodology",
]


@st.cache_resource
def load_bundle():
    """Load the saved classifier and preprocessing objects once per app session."""
    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_data
def load_results():
    """Load statistics produced by the backend without retraining the model."""
    with (ROOT / "outputs" / "results.json").open(encoding="utf-8") as file:
        return json.load(file)


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
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def yes_no(supported):
    return "Supported" if supported else "Not supported"


def section_intro(title, text):
    st.title(title)
    st.write(text)


def show_metric_table(rows):
    st.dataframe(pd.DataFrame(rows, columns=["Statistic", "Result"]), hide_index=True, use_container_width=True)


def risk_assessment(bundle, results):
    section_intro(
        "Risk Assessment",
        "Enter applicant information to classify an observed medical-cost risk tier.",
    )
    st.info(
        "This is an analytical risk profile based on observed medical costs in the dataset. "
        "It is not an insurer's quoted premium."
    )
    with st.form("risk_form"):
        left, right = st.columns(2)
        age = left.number_input("Age", min_value=18, max_value=64, value=35, step=1)
        bmi = right.number_input("BMI", min_value=15.0, max_value=55.0, value=27.5, step=0.1, format="%.1f")
        children = left.number_input("Children", min_value=0, max_value=5, value=0, step=1)
        sex = right.selectbox("Sex", ["female", "male"])
        smoker = left.selectbox("Smoker", ["no", "yes"])
        region = right.selectbox("Region", ["northwest", "northeast", "southeast", "southwest"])
        assess = st.form_submit_button("Assess risk", type="primary")

    if not assess:
        st.caption("The model uses age, BMI, children, sex, smoking status and region in the same numeric format used at training.")
        return

    classifier = bundle["classifier"]
    features = bundle["features"]
    row = build_row(age, bmi, children, sex, smoker, region, features)
    scaled = pd.DataFrame(bundle["scaler"].transform(row), columns=features)
    tier = classifier.predict(scaled)[0]
    probabilities = classifier.predict_proba(scaled)[0]
    labels = list(classifier.classes_)

    point = scaled.loc[0, CONTINUOUS].to_numpy()
    reference = bundle["centroids"][tier]
    distance = mahalanobis(point, np.asarray(reference["mean"]), np.asarray(reference["cov"]))
    cutoff = results["mahalanobis"]["threshold"]
    typical = distance <= cutoff

    st.subheader(f"Predicted risk tier: {tier}")
    st.caption(f"Most likely classification: {tier}")
    result_left, result_right = st.columns(2)
    result_left.metric("Observed annual medical-cost range", cost_band(tier, bundle["tier_edges"]))
    result_right.metric("Squared Mahalanobis distance", f"{distance:.2f}")

    st.subheader("Model class probabilities")
    st.caption("These are the classifier's relative supports for the observed Low, Medium and High cost tiers, not insurance claim probabilities.")
    for label, probability in zip(labels, probabilities):
        probability = float(probability)
        st.write(f"{label}: {probability:.0%}")
        st.progress(int(round(probability * 100)))

    st.subheader("Profile typicality")
    status = "Typical relative to predicted tier" if typical else "Atypical relative to predicted tier"
    st.metric("Status", status, help="Compared with the 97.5% chi-square reference cutoff.")
    st.progress(min(int(round(distance / cutoff * 100)), 100))
    st.caption(f"Reference cutoff: {cutoff:.2f}. A value above the cutoff is atypical for the predicted tier.")
    st.write(
        "Mahalanobis distance measures how far the applicant's multivariate profile is from the centre of the predicted tier "
        "while accounting for covariance among age, BMI and children."
    )

    with st.expander("How the model reached this result", expanded=True):
        st.markdown(
            "- The applicant characteristics are converted to the same numerical format used during training.\n"
            "- The trained discriminant classifier compares the applicant with learned Low, Medium and High risk groups.\n"
            "- The tier with the highest model support is selected.\n"
            "- Mahalanobis distance then describes how unusual the combined continuous profile is within that predicted tier."
        )


def model_overview(results):
    section_intro(
        "Multivariate Health Insurance Risk Profiling",
        "Medical-cost estimation using PCA, discriminant analysis, Hotelling's T² and Mahalanobis distance.",
    )
    st.info(
        "This application analyses observed annual medical costs from the dataset. It does not quote or guarantee an insurance premium."
    )
    selected = results["classification"]["method"]
    one, two, three = st.columns(3)
    one.metric("Observations", f"{results['data']['n']:,}")
    two.metric("Selected classifier", selected)
    three.metric("Cross-validation accuracy", f"{results['classification']['overall_accuracy']:.1%}")

    st.subheader("How the analysis works")
    steps = [
        ("1. Applicant data", "Age, BMI, children, sex, smoking status and region are entered."),
        ("2. Preprocessing", "Inputs are converted to the standardized numerical format used by the trained model."),
        ("3. Multivariate analysis", "Diagnostics, PCA and group-difference tests describe the data structure."),
        ("4. Risk classification", "LDA and QDA are compared, then the assumption-appropriate classifier assigns a risk tier."),
        ("5. Risk interpretation", "The observed cost range and Mahalanobis typicality make the classification easier to interpret."),
    ]
    for heading, description in steps:
        st.markdown(f"**{heading}**  \n{description}")

    st.subheader("Key finding")
    st.write(
        f"Box's M rejects covariance homogeneity ({format_p(results['box_m']['p_value'])}), so {selected} is selected because it allows group-specific covariance matrices. "
        "This does not by itself mean that QDA is universally more accurate."
    )


def assumptions_diagnostics(results):
    section_intro(
        "Assumptions & Diagnostics",
        "The diagnostic tests are shown openly because their results guide interpretation and model choice.",
    )
    mardia = results["mardia"]
    box = results["box_m"]
    normality_supported = mardia["skewness_p_value"] >= ALPHA and mardia["kurtosis_p_value"] >= ALPHA
    covariance_supported = box["p_value"] >= ALPHA

    st.subheader("Multivariate normality: Mardia's test")
    st.write("What it checks: whether the continuous variables follow a multivariate normal distribution exactly.")
    show_metric_table([
        ("Mardia skewness statistic", f"{mardia['skewness']:.3f}"),
        ("Skewness chi-square", f"{mardia['skewness_chi2']:.2f}"),
        ("Skewness p-value", format_p(mardia["skewness_p_value"])),
        ("Mardia kurtosis statistic", f"{mardia['kurtosis']:.3f}"),
        ("Kurtosis z statistic", f"{mardia['kurtosis_z']:.2f}"),
        ("Kurtosis p-value", format_p(mardia["kurtosis_p_value"])),
        ("Decision at alpha = 0.05", yes_no(normality_supported)),
    ])
    st.warning(
        "Interpretation: the diagnostic test does not support exact multivariate normality for the continuous variables. "
        "This is an assumption result to report, not a reason to hide the analysis."
    )

    st.subheader("Covariance homogeneity: Box's M")
    st.write("What it checks: whether covariance matrices are approximately equal across Low, Medium and High risk tiers.")
    show_metric_table([
        ("Box's M statistic", f"{box['m']:.2f}"),
        ("Adjusted chi-square", f"{box['chi2']:.2f}"),
        ("Degrees of freedom", f"{box['df']:.0f}"),
        ("p-value", format_p(box["p_value"])),
        ("Decision at alpha = 0.05", yes_no(covariance_supported)),
    ])
    st.warning(
        "Interpretation: covariance matrices differ across risk tiers, so the equal-covariance assumption is not supported. "
        "This supports selecting QDA because QDA allows each tier to have its own covariance matrix."
    )

    st.subheader("Data completeness and outliers")
    left, right, third = st.columns(3)
    left.metric("Observations", f"{results['data']['n']:,}")
    right.metric("Imputed cells", results["data"]["imputed_cells"])
    third.metric("Mahalanobis outliers", results["mahalanobis"]["count"])
    st.write(
        f"What it checks: the histogram shows within-tier squared Mahalanobis distances. "
        f"The reference cutoff is {results['mahalanobis']['threshold']:.2f}; observations above it are flagged as multivariate outliers."
    )
    st.image(figure_path("mahalanobis_hist.png"), caption="Population-level Mahalanobis distance diagnostic.", use_container_width=True)


def pca_analysis(results):
    section_intro(
        "PCA Analysis",
        "Principal component analysis identifies orthogonal directions of variation in the eight standardized model features.",
    )
    pca = results["pca"]
    components_needed = next(index + 1 for index, value in enumerate(pca["cumulative_variance"]) if value >= 0.90)
    one, two, three = st.columns(3)
    one.metric("Original features", len(pca["features"]))
    two.metric("PC1 variance", f"{pca['proportion_variance'][0]:.1%}")
    three.metric("Components for 90%", components_needed)
    st.info(
        f"Interpretation: PC1 explains {pca['proportion_variance'][0]:.1%} of total variance. "
        f"{components_needed} components are needed to retain at least 90%, so PCA is more useful here for understanding structure than aggressive dimensionality reduction."
    )

    table = pd.DataFrame({
        "Component": [f"PC{i}" for i in range(1, len(pca["eigenvalues"]) + 1)],
        "Eigenvalue": pca["eigenvalues"],
        "Variance explained": pca["proportion_variance"],
        "Cumulative variance": pca["cumulative_variance"],
    })
    st.subheader("Eigenvalues and variance explained")
    st.dataframe(
        table.style.format({"Eigenvalue": "{:.3f}", "Variance explained": "{:.1%}", "Cumulative variance": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.image(figure_path("pca_scree.png"), caption="Scree plot and cumulative variance.", use_container_width=True)
    st.image(figure_path("pca_biplot.png"), caption="PCA biplot by observed risk tier.", use_container_width=True)

    st.subheader("PC1 and PC2 loadings")
    loading_table = pd.DataFrame({
        "Feature": pca["features"],
        "PC1 loading": [row[0] for row in pca["loadings"]],
        "PC2 loading": [row[1] for row in pca["loadings"]],
    })
    st.dataframe(loading_table.style.format({"PC1 loading": "{:.3f}", "PC2 loading": "{:.3f}"}), hide_index=True, use_container_width=True)

    st.subheader("Feature correlations")
    st.write("Most off-diagonal correlations are close to zero, which helps explain why PCA provides limited compression.")
    st.image(figure_path("feature_correlation.png"), caption="Correlation heatmap for the standardized model features.", use_container_width=True)


def classification(results):
    section_intro(
        "Discriminant Classification: LDA vs QDA",
        "The classifier assigns observations to Low, Medium or High observed medical-cost tiers.",
    )
    classification_result = results["classification"]
    comparison = classification_result["comparison"]
    st.subheader("Five-fold cross-validation")
    comparison_table = pd.DataFrame([
        {"Method": "LDA", "Mean accuracy": comparison["lda_cv_mean"], "SD": comparison["lda_cv_sd"]},
        {"Method": "QDA", "Mean accuracy": comparison["qda_cv_mean"], "SD": comparison["qda_cv_sd"]},
    ])
    st.dataframe(comparison_table.style.format({"Mean accuracy": "{:.1%}", "SD": "{:.1%}"}), hide_index=True, use_container_width=True)
    st.metric("Selected model", classification_result["method"])
    st.write(
        "Reason for selection: Box's M indicates that covariance matrices differ across risk tiers. "
        "QDA allows each class to have its own covariance matrix, making that assumption more appropriate for these data."
    )
    st.caption("Selection is based on covariance structure, not a claim that QDA is automatically better in every setting.")

    st.subheader("Confusion matrix")
    labels = classification_result["labels"]
    matrix = pd.DataFrame(classification_result["confusion_matrix"], index=labels, columns=labels)
    matrix.index.name = "Actual tier"
    matrix.columns.name = "Predicted tier"
    st.dataframe(matrix, use_container_width=True)
    st.metric("Overall cross-validated accuracy", f"{classification_result['overall_accuracy']:.1%}")
    st.write("What it shows: each cell counts the cross-validated classifications for an actual tier and a predicted tier.")
    st.image(
        figure_path("qda_regions.png"),
        caption="Illustrative QDA decision regions using PC1 and PC2. Illustrative only: the reported classifier uses all eight standardized features.",
        use_container_width=True,
    )


def statistical_inference(results):
    section_intro(
        "Statistical Inference",
        "These tests examine whether observed risk-tier groups differ in their combined continuous-variable profiles.",
    )
    hotelling = results["hotelling_t2"]
    st.subheader("Hotelling's T²: High versus Low risk")
    st.write("H0: the High-risk and Low-risk groups have the same multivariate mean vector.")
    st.write("H1: their multivariate mean vectors differ.")
    show_metric_table([
        ("T² statistic", f"{hotelling['t2']:.2f}"),
        ("F statistic", f"{hotelling['f']:.2f}"),
        ("Numerator df", hotelling["df1"]),
        ("Denominator df", hotelling["df2"]),
        ("p-value", format_p(hotelling["p_value"])),
        ("Decision at alpha = 0.05", "Reject H0" if hotelling["p_value"] < ALPHA else "Do not reject H0"),
    ])
    st.success(
        "Interpretation: the very small p-value provides strong evidence that the High- and Low-risk groups differ in their combined mean profile across age, BMI and children."
    )

    manova = results["manova"]
    st.subheader("MANOVA: Multivariate group difference")
    st.write("What it tests: whether the multivariate response profile differs across the observed risk-tier groups.")
    show_metric_table([
        ("Wilks' Lambda", f"{manova['wilks_lambda']:.4f}"),
        ("F statistic", f"{manova['f']:.2f}"),
        ("Numerator df", f"{manova['num_df']:.0f}"),
        ("Denominator df", f"{manova['den_df']:.0f}"),
        ("p-value", format_p(manova["p_value"])),
        ("Decision at alpha = 0.05", "Reject H0" if manova["p_value"] < ALPHA else "Do not reject H0"),
    ])
    st.success("Interpretation: the multivariate response profile differs across the Low, Medium and High risk tiers.")


def dataset(results):
    section_intro(
        "Dataset",
        "A compact description of the data used to create the observed medical-cost tiers.",
    )
    data = results["data"]
    left, middle, right = st.columns(3)
    left.metric("Dataset", "insurance.csv")
    middle.metric("Observations", f"{data['n']:,}")
    right.metric("Imputed cells", data["imputed_cells"])
    st.subheader("Variables")
    st.write("Original variables: age, sex, BMI, children, smoker, region and charges.")
    st.write("Model variables: age, BMI, children, sex indicator, smoker indicator and three region indicators, for eight standardized features in total.")
    st.write("Risk groups: Low, Medium and High, created from terciles of the observed `charges` variable.")

    edges = data["tercile_edges"]
    tiers = pd.DataFrame([
        {"Observed tier": "Low", "Cost range": f"${edges[0]:,.0f} to ${edges[1]:,.0f}"},
        {"Observed tier": "Medium", "Cost range": f"${edges[1]:,.0f} to ${edges[2]:,.0f}"},
        {"Observed tier": "High", "Cost range": f"${edges[2]:,.0f} to ${edges[3]:,.0f}"},
    ])
    st.subheader("Observed annual medical-cost tiers")
    st.dataframe(tiers, hide_index=True, use_container_width=True)
    st.caption("The deployed model artifact does not contain individual source records, so no raw-record sample is loaded in the app.")


def methodology():
    section_intro(
        "Methodology",
        "A concise viva guide to the multivariate techniques used in this application.",
    )
    methods = [
        ("PCA", "Identify orthogonal directions of variation.", r"S v = \lambda v", "Used to describe feature structure and assess how much dimensional compression is possible."),
        ("Mahalanobis distance", "Measure multivariate distance while accounting for covariance.", r"D^2 = (x - \mu)' \Sigma^{-1} (x - \mu)", "Used to assess how typical an applicant is within the predicted tier."),
        ("Hotelling's T²", "Test whether two multivariate mean vectors differ.", r"T^2 = \frac{n_1 n_2}{n_1+n_2}(\bar{x}_1-\bar{x}_2)'S_p^{-1}(\bar{x}_1-\bar{x}_2)", "Used to compare High- and Low-risk continuous profiles."),
        ("LDA", "Classify groups under a common covariance structure.", r"\delta_k(x) = x'\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k'\Sigma^{-1}\mu_k + \log \pi_k", "Compared with QDA using five-fold cross-validation."),
        ("QDA", "Classify groups while allowing group-specific covariance matrices.", r"\delta_k(x) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x-\mu_k)'\Sigma_k^{-1}(x-\mu_k) + \log \pi_k", "Selected when Box's M rejects the equal-covariance assumption."),
        ("MANOVA", "Assess multivariate differences across several groups.", r"\Lambda = \frac{|E|}{|E+H|}", "Used to test whether the combined continuous-variable profile differs across all three risk tiers."),
    ]
    for name, what, formula, use in methods:
        st.subheader(name)
        st.write(f"What it does: {what}")
        st.latex(formula)
        st.write(f"How this project uses it: {use}")


def main():
    st.set_page_config(page_title="Multivariate Health Insurance Risk Profiling", layout="wide")
    bundle = load_bundle()
    results = load_results()

    with st.sidebar:
        st.title("Risk Profiling")
        page = st.radio("Navigate", PAGE_OPTIONS)
        st.caption("University multivariate techniques project")

    pages = {
        "Risk Assessment": lambda: risk_assessment(bundle, results),
        "Model Overview": lambda: model_overview(results),
        "Assumptions & Diagnostics": lambda: assumptions_diagnostics(results),
        "PCA Analysis": lambda: pca_analysis(results),
        "Classification": lambda: classification(results),
        "Statistical Inference": lambda: statistical_inference(results),
        "Dataset": lambda: dataset(results),
        "Methodology": methodology,
    }
    pages[page]()


if __name__ == "__main__":
    main()
