# Multivariate Classification of Health-Insurance Medical-Cost Groups

How accurately can demographic and lifestyle characteristics classify individuals into Low, Medium and High observed medical-cost groups, and where does classification fail?

This university project uses k-nearest neighbours as its primary Unit 4 multivariate technique, with fixed LDA and regularized QDA benchmarks. It classifies observed medical charges from six jointly considered variables. It is not actuarial pricing, medical assessment, causal inference or a validated forecast of future costs.

## Data and design

Download [Medical Cost Personal Datasets by Miri Choi](https://www.kaggle.com/datasets/mirichoi0218/insurance) to `data/insurance.csv`. The CSV is local and uncommitted. There are 1,338 source records, 1,337 after one exact duplicate is removed, and no missing values in this supplied version. Matching records do not prove duplicated people.

A single random 80/20 split (seed 42) produces 1,069 development and 268 test records. Only development charges define the tercile thresholds. Numerical preprocessing and category encoding fit inside each development CV training fold. k-NN uses 11 encoded dimensions; LDA/QDA use eight with reference-category encoding. Charges never enter predictors. The test set was held out during this refactor, not independent external validation of a previously unexplored dataset.

## Run locally

Use Python 3.11.11 and the pinned, verified dependencies. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.main
python -m src.flowchart
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

`python -m src.main` generates the development-only model, structured results, held-out predictions and two evaluation figures. `python -m src.figures` regenerates those figures from saved predictions without refitting. The app never trains on opening. Missing or incompatible artifacts produce a rebuild instruction.

## Optional AI interpretation

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`; retain or configure `OPENAI_MODEL`. The existing default is preserved. Equivalent Streamlit secrets are supported. Neither `.env` nor `.streamlit/secrets.toml` should be committed. AI is called only after clicking the explanation button. No key, API failure or incomplete response prevents use of the Python results. Tests mock AI and incur no live requests. AI text can contain errors and cannot replace the authoritative Python cards.

## Files

- `app.py`: navy/white assessment, model explanation, results and methodology pages.
- `src/data.py`, `src/classify.py`, `src/main.py`: validation, fold-fitted selection and frozen evaluation.
- `src/inference.py`: artifact checks and exact neighbour explanations.
- `src/ai_interpretation.py`: optional compact explanation contract.
- `outputs/results.json`, `outputs/evaluation_predictions.csv`: full reproducibility and error audit.
- `models/model.joblib`: versioned development-only pipeline and reference rows.
- `src/figures.py`, `src/flowchart.py`: figure and flowchart sources.
- `METHODOLOGY.md`: equations, pseudo-algorithm, conditions, actual results and limitations.
- `SUBMISSION_CHECKLIST.md`: provisional rubric evidence map and user-owned pending items.
- `legacy/`: previous methodology and preserved local edits, separate from revised evidence.

The rubric PDF was unavailable. Acceptance of existing Streamlit hosting in place of the named Antigravity/GitHub Codespace requirement needs clarification. Public URLs, Developer Pack/tool-use evidence, two LinkedIn posts, a captioned video, three endorsements and a personal reflection with a real peer citation remain user-owned evidence.

## Actual revised results

See [generated results and error analysis](outputs/RESULTS.md) for the current run, selected parameters, exact cutoffs, benchmark table, High errors and k sensitivity. This file is regenerated from results.json by the analysis entry point.

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
and limitations. Original snapshot explanations remain preserved. New explanations
use actual `#` comments, avoiding the adjacent-string dictionary-key defect found
in the earlier Mardia code.

The comment update leaves executable Python syntax trees, fitted artifacts and saved
metrics unchanged. The saved run identifies its original generation source; since
the run fingerprint hashes source bytes, a future regeneration after comment edits
will receive a new run identifier even when the model's numerical results agree.
