# Legacy appendix, not current evidence

The pre-refactor methodology is preserved in Git at 81c1ac2. The copied data module
preserves the user's uncommitted educational explanations. The inference copy
preserves the user's exact local edits; statsmodels is needed only to execute its
optional MANOVA function. Install statsmodels==0.14.4 separately if studying it.
The assumptions snapshot records the original dictionary-key defect; do not execute
it. The retained src/assumptions.py fixes that defect and has a regression test.

results_before_refactor.json and the older tracked figure files are legacy outputs.
They do not validate the revised classifier. No complete legacy pipeline rerun is
claimed. Old full-data preprocessing and target definitions cause validation leakage.
PCA, Mardia, Box's M, Hotelling's T², MANOVA and Mahalanobis are omitted from the
active workflow because they do not answer the selected classification question.
No normality tests are prerequisites for k-NN. PCA decomposition does not require
normality; weak correlation does not prove predictor independence.
