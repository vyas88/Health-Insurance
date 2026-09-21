# Multivariate Classification of Health-Insurance Medical-Cost Groups

How accurately can demographic and lifestyle characteristics classify individuals into Low, Medium and High observed medical-cost groups, and where does classification fail?

This university project uses k-nearest neighbours as its primary Unit 4 multivariate technique, with fixed LDA and regularized QDA benchmarks. It classifies observed medical charges from six jointly considered variables. It is not actuarial pricing, medical assessment, causal inference or a validated forecast of future costs.

## Data and design

The source dataset is [Medical Cost Personal Datasets by Miri Choi](https://www.kaggle.com/datasets/mirichoi0218/insurance). A copy is included at `data/insurance.csv` for reproducibility. There are 1,338 source records, 1,337 after one exact duplicate is removed, and no missing values in this supplied version. Matching records do not prove duplicated people.

The predictors are age, sex, BMI, number of children, smoking status and region. The target is a Low, Medium or High group derived from observed medical charges.

A single random 80/20 split (seed 42) produces 1,069 development and 268 test records. Only development charges define the tercile thresholds. Numerical preprocessing and category encoding fit inside each development CV training fold. k-NN uses 11 encoded dimensions; LDA/QDA use eight with reference-category encoding. Charges never enter predictors. The test set was held out during this refactor, not independent external validation of a previously unexplored dataset.

## Run locally

The saved artifacts were generated with Python 3.11.11 and the pinned dependencies. Clone the repository and start the app:

```bash
git clone https://github.com/vyas88/Health-Insurance.git
cd Health-Insurance
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`. The repository includes the fitted model, results and figures, so retraining is not required to launch the app. Its four pages cover profile assessment, how the model works, results and model comparison, and dataset methodology.

The assessment also displays a group-based annual medical-cost estimate in USD: the median observed charge among development records in the predicted group. The historical minimum-to-maximum range and a simple calculation explanation appear alongside it. This estimate is shared by profiles in the same group; it is not an insurance premium quote, a validated cost forecast or a prediction interval.

To reproduce the analysis and run the checks, use the activated environment:

```bash
python -m src.main
python -m src.flowchart
python -m unittest discover -s tests -v
```

`python -m src.main` generates the development-only model, structured results, held-out predictions and two evaluation figures. `python -m src.figures` regenerates those figures from saved predictions without refitting. The app never trains on opening. Missing or incompatible artifacts produce a rebuild instruction. Dependency versions should match `requirements.txt` when loading the saved model.

## Optional AI interpretation

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`; retain or configure `OPENAI_MODEL`. The existing default is preserved. Equivalent Streamlit secrets are supported. Neither `.env` nor `.streamlit/secrets.toml` should be committed. AI is called only after clicking the explanation button. No key, API failure or incomplete response prevents use of the Python results. Tests mock AI and incur no live requests. AI text can contain errors and cannot replace the authoritative Python cards.

## Files

- `app.py`: navy/white assessment, model explanation, results and methodology pages.
- `data/insurance.csv`: source dataset used by the analysis and tests.
- `src/data.py`, `src/classify.py`, `src/main.py`: validation, fold-fitted selection and frozen evaluation.
- `src/inference.py`: artifact checks and exact neighbour explanations.
- `src/ai_interpretation.py`: optional compact explanation contract.
- `outputs/results.json`, `outputs/evaluation_predictions.csv`: full reproducibility and error audit.
- `models/model.joblib`: versioned development-only pipeline and reference rows.
- `src/figures.py`, `src/flowchart.py`: figure and flowchart sources.
- `METHODOLOGY.md`: equations, pseudo-algorithm, conditions, actual results and limitations.
- `SUBMISSION_CHECKLIST.md`: provisional rubric evidence map and user-owned pending items.
- `VERIFICATION.md`, `tests/`: recorded validation and automated workflow/app checks.
- `legacy/`: previous methodology and preserved local edits, separate from revised evidence.

Submission requirements and outstanding external evidence are tracked separately in [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md). Publishing this repository does not establish a live app deployment or completion of those requirements.

## Saved evaluation results

The selected k-NN model uses 21 neighbours, Euclidean distance and distance-weighted voting. On the 268 held-out records:

| Model | Accuracy | Balanced accuracy | Macro F1 | High-group recall |
|---|---:|---:|---:|---:|
| k-NN | 85.82% | 86.04% | 0.8585 | 75.27% |
| LDA | 83.58% | 83.99% | 0.8350 | 64.52% |
| Regularized QDA | 82.84% | 83.23% | 0.8280 | 64.52% |

k-NN misses 23 High-group records, assigning five to Low and 18 to Medium. Recall for High-cost non-smokers is only 33.3% across 33 records. These limitations matter despite the overall accuracy, and one split does not establish statistical superiority.

See [generated results and error analysis](outputs/RESULTS.md) for the run identifier, exact cutoffs and k sensitivity, and [METHODOLOGY.md](METHODOLOGY.md) for the design and limitations. The analysis entry point regenerates the results report from `outputs/results.json`.

## Reading the code for a viva

The Python files include teaching comments explaining what each stage does, why the
relevant library functions are used, what the inputs and outputs mean, and which
statistical or application errors each safeguard prevents. A useful reading order is:

1. `src/data.py`: raw schema, validation, missing values, duplicate audit and target boundaries.
2. `src/classify.py`: pipelines, encoding, scaling, cross-validation, parameter selection and metrics.
3. `src/main.py`: the complete offline analysis and artifact-generation sequence.
4. `src/inference.py`: compatible artifact loading, neighbour retrieval and exact voting.
5. `app.py` and `src/ai_interpretation.py`: Streamlit reruns, session state, result display and optional AI.
6. `src/figures.py`, `src/flowchart.py` and `src/report.py`: figures and report generation from saved facts.
7. `tests/`: the mathematical and application behaviours checked by each test.

The retained legacy helpers have scope notes explaining their historical purpose
and limitations. Their outputs are separate from the current k-NN evaluation.

Local credentials (`.env` and `.streamlit/secrets.toml`), the virtual environment and
Python caches are excluded from Git. The source data, fitted model and generated
analysis outputs are included.
