# Verification record

Final analysis run: `16b8293accdaf07c`, schema 2, Python 3.11.11. Dependencies are
pinned in requirements.txt; `python -m pip check` reports no broken requirements.

Ten unittest cases pass, including Streamlit AppTest journeys with multiple assertions:

- Clean counts, duplicate removal, missing/invalid target handling and raw input validation.
- Disjoint development/test source IDs, development-only reference rows and thresholds.
- Exact boundary and beyond-range target labelling; coincident threshold failure.
- Training-fold scaler means, fold membership and imputation statistics; 11 encoded
  features with no charges/targets in preprocessing.
- Uniform, distance and zero-distance votes matched against scikit-learn support.
- Saved-model reload predictions and all model metrics reconciled with the saved CSV.
- Artifact/schema mismatch error and stale session invalidation.
- All four Streamlit pages, valid/invalid profile paths and changed-profile behaviour.
- Mocked AI success, disabled credentials, cache reuse, profile/run changes,
  incomplete/empty response and timeout failure. No live API calls were made.
- Expected Mardia p-value keys, without claiming a complete legacy pipeline rerun.

The focused legacy classifier audit separately confirmed 274 smokers, all in the
old High group, and 172 High non-smokers missed by old QDA CV. These outputs are in
legacy/baseline_audit.json and are not revised performance evidence.

The local Streamlit server was opened in the in-app browser. The assessment,
selected configuration expander, results/figures and methodology navigation were
inspected. A dark-theme label contrast problem was found and fixed with the explicit
light theme. The exported recall figure and flowchart were also visually inspected.

The original system Python 3.9 environment was used for an initial implementation
check, then the identical declared design was rerun in isolated Python 3.11.11 for
final artifacts. No search space, seed or selection criterion was changed in response
to holdout performance. KD-tree lookup replaced a brute-force implementation that
raised a scikit-learn string-label error before evaluation.

Not verified: real OpenAI account/model availability or generated prose quality,
public hosting acceptance, public deployment and rubric PDF wording. The PDF was
not found in the workspace or attachments. Required external/student-owned evidence
is itemized in SUBMISSION_CHECKLIST.md. No paid requests, social messages or public
site deployment were performed.
