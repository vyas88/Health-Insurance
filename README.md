# Multivariate Health Insurance Risk Profiling

This repository contains an MSc Computational Statistics and Applied AI project for a Multivariate Techniques course. It examines health-insurance-style medical-cost records and groups applicants into Low, Medium, and High observed medical-cost tiers.

The application is an analytical demonstration, not an actuarial pricing system. It reports historical medical-cost ranges from the dataset and does not provide an insurance quote, medical advice, or an underwriting decision.

## Problem

Insurance-related risk profiles are multivariate: age, BMI, number of children, sex, smoking status, and region are considered together rather than one at a time. The project uses these characteristics to identify the historical cost group that most closely matches an applicant profile.

## Dataset

The analysis uses the [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) dataset published on Kaggle by Miri Choi. The source data contains 1,338 records and the variables `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges`.

After duplicate removal, the analysis uses 1,337 observations. The `charges` variable is split into terciles to form Low, Medium, and High observed medical-cost groups.

The CSV is not committed to this repository. To reproduce the pipeline, download `insurance.csv` from the source and place it at `data/insurance.csv`.

## Methods

The project applies the following multivariate techniques:

- preprocessing, encoding, and standardisation
- Mardia's multivariate normality diagnostics
- Box's M test for covariance homogeneity
- PCA for dimensional structure and visualisation
- LDA and QDA comparison using five-fold cross-validation
- Hotelling's T² test for the High and Low group mean profiles
- MANOVA using Wilks' Lambda for group differences
- Mahalanobis distance for profile typicality

QDA is the reported classifier because Box's M indicates that covariance patterns differ across the risk groups. The reported classifier uses all eight standardised features. PCA is used separately to understand the data structure and to provide illustrative plots.

## Key results

- QDA was selected after the equal-covariance assumption was not supported.
- LDA and QDA achieved similar cross-validation accuracy, at about 84%.
- Mardia's diagnostics did not support exact multivariate normality.
- PC1 explains about 19% of the variation and seven components are needed to retain at least 90%, so PCA offers limited compression for this dataset.
- Hotelling's T² and MANOVA both provide strong evidence that the observed risk groups differ in their combined profiles.

The Streamlit app presents these results alongside an interactive risk assessment. An optional OpenAI explanation feature interprets the Python-generated result in plain language. It never calculates or changes the risk tier, probabilities, statistical tests, or cost range.

## Project structure

```text
.
├── app.py                  Streamlit application
├── data/                   Local dataset location
├── models/                 Saved classifier artefact
├── outputs/                Precomputed results and figures
├── src/                    Statistical pipeline and supporting modules
├── requirements.txt        Python dependencies
└── runtime.txt             Deployment Python version
```

## Running the project

The deployed app uses the saved model, results, and figures. To regenerate the analysis locally, place the source CSV in `data/insurance.csv`.

```bash
pip install -r requirements.txt
python src/main.py
python src/figures.py
streamlit run app.py
```

The AI explanation is optional. For local use, create one root `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.6-luna
```

The `.env` file is ignored by Git and must never be committed.

## References

1. Choi, M. (n.d.). *Medical Cost Personal Datasets*. Kaggle. https://www.kaggle.com/datasets/mirichoi0218/insurance
2. Mardia, K. V. (1970). Measures of multivariate skewness and kurtosis with applications. *Biometrika, 57*(3), 519-530.
3. Box, G. E. P. (1949). A general distribution theory for a class of likelihood criteria. *Biometrika, 36*(3/4), 317-346.
4. Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis* (3rd ed.). Wiley.
