# Multivariate Classification of Health-Insurance Medical-Cost Groups

## Question and scope

How accurately can demographic and lifestyle characteristics classify individuals into Low, Medium and High observed medical-cost groups, and where does classification fail?

The profile is multivariate because age, BMI, children (a count), sex, smoker and region enter jointly. This is supervised classification of observed medical charges, not unsupervised clustering, actuarial pricing, causal inference, medical assessment or validated forecasting. The selected course mapping is Unit 4, k-NN in multivariate space, as described in the supplied brief. The rubric PDF was not available for independent inspection.

## Data and cleaning

Source: Miri Choi, [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance), Kaggle. Download the CSV into `data/insurance.csv`. The local CSV remains uncommitted. Its SHA-256 is recorded in `outputs/results.json`.

The supplied file contains 1,338 records, no missing cells and one exact duplicate, leaving 1,337 records. Source-row IDs are original CSV line numbers including the header. Exact matching fields do not establish that two records refer to the same person. The seven-field schema has no person identifier, so repeated-person dependence cannot be ruled out. Unexpected extra fields trigger inspection rather than an unexamined random split; identifiable repeated people would require grouped splitting.

Numeric conversion, finite values, category domains, nonnegative charges, integer age/children and positive BMI are checked. Valid extreme charges are retained. Missing or invalid charges are dropped and reported, never imputed. Invalid predictors stop analysis. Missing predictors remain missing until each training fold fits median numerical and mode categorical imputers. No imputation was actually required in this file. All-missing training-fold predictors stop the workflow.

## Fixed target and validation design

One random 80/20 split with seed 42 assigns 1,069 development and 268 test records. This initial split is not stratified using whole-dataset labels. Rows are then sorted by source-row ID for reproducible neighbour ordering. Only development charges determine linearly interpolated one-third and two-thirds quantiles. Exact floating-point thresholds and the split membership are saved.

Low: charges <= q1. Medium: q1 < charges <= q2. High: charges > q2. The outer groups are open-ended, so valid charges beyond development extrema remain labelled. Coincident quantiles stop execution; inadequate development class sizes or a missing test class also stop with a concrete limitation rather than seed hunting.

Development quantiles define a fixed classification task during CV. Five stratified development folds use shuffle=True and seed 42. Preprocessing is learned separately in each training fold. Development selection chooses among k=[5,11,21,31], weights=[uniform,distance], p=[1,2], excluding unsupported k. Macro F1 is primary; accuracy, balanced accuracy and class recalls are also recorded for each fold and candidate. Exact score ties prefer smaller k, then uniform voting, then p=1. The best selection-CV score is not an unbiased final estimate and is not nested CV.

The primary method is predeclared k-NN. LDA uses fixed SVD; QDA uses fixed reg_param=0.001. Their fold statistics contextualize performance without choosing the deployed model. Frozen fitted models predict the same holdout once; final metrics and plots derive from those saved predictions. No deployment refit includes test rows. This sample was explored before this refactor, so the holdout is not independent external validation or previously unseen data in the broader project history.

## Representation and mathematics

Three raw numerical variables use training-fold z-scores z=(x-mu)/sigma, where mu and sigma are the training mean and population standard deviation. Categorical indicators are not standardized. The declared schema retains two sex, two smoker and four region indicators, giving 11 encoded dimensions. The internal encoder rejects unknown categories, as does the applicant validator. Charges and derived labels never enter the feature matrix.

For encoded profiles z and z_i, Minkowski distance is d_i=(sum_j |z_j-z_ij|^p)^(1/p), where j indexes the 11 dimensions and p=1 gives Manhattan distance, p=2 Euclidean distance. A categorical mismatch changes two indicators: contribution 2 to Manhattan or squared Euclidean distance, equally for every pair of categories. This is a similarity choice, not a uniquely correct medical distance. Full one-hot encoding avoids asymmetric region distances. Linear dependencies do not obstruct k-NN because it does not invert covariance matrices.

For class c, support S_c(x)=sum_{i in N_k(x)} w_i I(y_i=c) / sum_{i in N_k(x)} w_i.

Here x is the submitted profile; N_k(x) is its k nearest development records; y_i is record i's observed group; I is 1 when its condition holds and 0 otherwise; w_i is the voting weight. Uniform voting uses w_i=1. Distance voting uses w_i=1/d_i. If any selected neighbour has d_i=0, only the zero-distance neighbours vote, equally. Prediction is the class with greatest support. This support is not calibrated confidence or a probability of illness, future claims or correct classification.

The selected implementation uses deterministic KD-tree lookup in the pinned scikit-learn version. Exact equal-distance membership at the k boundary may depend on fixed training order and library implementation; no claim is made that such ties have a unique answer. Display sorting is distance then source-row ID within the library-selected k. Class-vote ties use estimator order High, Low, Medium; displayed support is explicitly remapped to Low, Medium, High. Tests reconcile the independently calculated vote with predict_proba, including zero-distance cases. Five shown neighbours are only a preview of k.

LDA and QDA use the same fold-fitted numerical scaling/imputation, but reference-category one-hot encoding (female, no, northeast as references) yields eight dimensions and avoids redundant indicator columns. Gaussian discriminant models remain approximations with categorical predictors. QDA can still face class-constant indicators; rank warnings are captured in results rather than hidden. Regularization does not establish Gaussian assumptions. The initial brute-force lookup hit an installed scikit-learn string-label error; KD-tree avoids that implementation issue without changing the declared distance/grid.

## Numbered pseudo-algorithm

1. Read and validate schema; retain deterministic source IDs. Remove/report exact duplicates and invalid targets. Stop on invalid predictors or unresolved identifiers.
2. Split raw cleaned records once into development/test; compute only development quantiles. Stop on coincident boundaries or inadequate class representation.
3. Label both partitions using the frozen boundaries. Inspect development ranges, category frequencies and class coverage.
4. In each development fold, fit imputation, scaling and encoding only on training rows. Evaluate every declared k-NN configuration and the fixed benchmarks.
5. Select the k-NN configuration by mean macro F1 with the declared tie rule. Fit final models on development only.
6. Predict the held-out test once. Save all predictions, metrics, warnings, split/fold IDs, hashes, versions and exact configuration. Generate figures from saved predictions.
7. Online, check artifact compatibility. Validate six applicant fields; reject missing/unsupported values and flag values outside development numerical ranges without assigning a typicality label.
8. Transform with the saved pipeline; predict class/support; retrieve development neighbours. Show historical group summaries separately from fixed definitions.
9. Only on an explicit button click, send compact Python facts to optional OpenAI interpretation. On absent credentials, failure or incomplete response, retain the fully usable Python result and show a neutral availability message.

## Conditions and limitations

Meaningful scaling and encoding are explicitly defined, not proven medically appropriate. All declared categories and all classes are represented in development data. The saved candidate fold scores allow inspection of k sensitivity and prediction stability. The raw numerical mean vector, covariance/correlation matrices and categorical frequencies describe development records, distinct from the 11 encoded dimensions.

Observed category/range coverage cannot establish population representativeness or independence. Missing diagnoses, treatment intensity, utilization and other cost drivers limit High-group recovery. Subgroup recall is based on a small sample. No statistical superiority claim is warranted from the model differences in one split. The old 84.1% QDA and exploratory k-NN results used different labels/preprocessing/evaluation; they are not targets to reproduce or directly comparable final estimates.

PCA and classical hypothesis tests are omitted because they do not resolve this classification question. Normality tests are not k-NN validation. Legacy methodology is documented separately; old plots/p-values are never used to support the new model.

## Optional AI and assistance disclosure

Python alone computes the authoritative cards. OpenAI receives the submitted profile, run ID, class, support, voting, k/metric, aggregate neighbour counts, frozen definitions, labelled historical group summaries and actual test metrics. It receives neither the training table nor full neighbour records. Configuration retains OPENAI_API_KEY and OPENAI_MODEL through environment or Streamlit secrets; no key is required to classify.

The existing configured model default is preserved. Responses use a 650-token budget, 20-second timeout and at most one SDK retry. Only completed, nonempty responses are shown. Successful text is session-cached by profile/context/model/run. Edits hide old results immediately; artifact changes clear session explanations. AI prose can still be unfaithful despite instructions; no regex or formatting check is claimed to prove mathematical fidelity. Mock tests cover success, no key, incomplete/empty responses, timeout and stale-profile cases. No paid/live call was made during verification.

This refactor used AI assistance for implementation, documentation and verification. The student must review the equations, results, limitations and source attribution and supply their own authentic reflection. No personal experiences or peer endorsements were generated.

## Reproduction and evidence

See README for commands. `src/flowchart.py` is editable flowchart source; `outputs/flowchart.png` separates offline development from online inference. `tests/` checks leakage boundaries, voting, saved metrics, artifact mismatch, legacy dictionary keys and Streamlit journeys. `outputs/results.json` is the complete machine-readable record. `outputs/evaluation_predictions.csv` enables independent metric reconciliation. Only the primary model is persisted; fixed benchmarks are reproducible from the entry point.

Technical sources: [scikit-learn neighbours](https://scikit-learn.org/stable/modules/neighbors.html), [leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html), [LDA/QDA](https://scikit-learn.org/stable/modules/lda_qda.html), [preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html), [selection versus nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).

## Actual revised results

See [generated results and error analysis](outputs/RESULTS.md) for the current run, selected parameters, exact cutoffs, benchmark table, High errors and k sensitivity. This file is regenerated from results.json by the analysis entry point.
