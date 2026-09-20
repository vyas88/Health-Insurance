"""Audit only the old classifier protocol, not the full legacy analysis."""
import json
import warnings
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# This script intentionally reproduces the OLD classifier protocol for audit.
# It is separate from src.main, and its results must not be mixed with revised
# holdout evidence. It runs at script execution and does not launch the app.
root=Path(__file__).resolve().parents[1]
data=pd.read_csv(root/'data/insurance.csv').drop_duplicates().reset_index(drop=True)
# qcut forms approximately equal-frequency groups from ALL charges here.
# This whole-dataset target definition is retained solely to verify old findings.
tier=pd.qcut(data.charges,3,labels=['Low','Medium','High'])
# Recreate the old eight-column representation with reference-category
# indicators. Charges are used for labels, not as a predictor even in this audit.
features=pd.concat([data[['age','bmi','children']],pd.get_dummies(data[['sex','smoker','region']],drop_first=True,dtype=float)],axis=1)
# This full-data fit is the historical preprocessing leakage being audited.
# The current pipeline fits scaling inside each development training fold.
scaled=StandardScaler().fit_transform(features)
# cross_val_predict gives each row a prediction from a fold model that did
# not train on that row, but preprocessing above still used all rows.
# That distinction is why these predictions are not revised validation evidence.
with warnings.catch_warnings(record=True) as caught:
    predictions=cross_val_predict(QuadraticDiscriminantAnalysis(reg_param=.001),scaled,tier,cv=StratifiedKFold(5,shuffle=True,random_state=42))
# Define the missed subgroup from actual High labels and non-smoking status,
# then count how many of those records the old classifier recovered.
mask=(tier=='High') & (data.smoker=='no')
audit={'scope':'Legacy classifier only; full-dataset labels/scaling are leaky and not revised validation evidence',
       'rows':len(data),'smokers':int((data.smoker=='yes').sum()),
       'smokers_in_high':int(((data.smoker=='yes') & (tier=='High')).sum()),
       'high_non_smokers':int(mask.sum()),'high_non_smokers_recovered':int((predictions[mask]=='High').sum()),
       'predicted_high_equals_smoker':bool(((predictions=='High')==(data.smoker=='yes')).all()),
       'warnings':sorted(set(str(w.message) for w in caught))}
# Write only to the labelled legacy appendix, leaving current results intact.
(root/'legacy/baseline_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
