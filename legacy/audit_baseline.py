"""Audit only the old classifier protocol, not the full legacy analysis."""
import json
import warnings
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict

root=Path(__file__).resolve().parents[1]
data=pd.read_csv(root/'data/insurance.csv').drop_duplicates().reset_index(drop=True)
tier=pd.qcut(data.charges,3,labels=['Low','Medium','High'])
features=pd.concat([data[['age','bmi','children']],pd.get_dummies(data[['sex','smoker','region']],drop_first=True,dtype=float)],axis=1)
scaled=StandardScaler().fit_transform(features)
with warnings.catch_warnings(record=True) as caught:
    predictions=cross_val_predict(QuadraticDiscriminantAnalysis(reg_param=.001),scaled,tier,cv=StratifiedKFold(5,shuffle=True,random_state=42))
mask=(tier=='High') & (data.smoker=='no')
audit={'scope':'Legacy classifier only; full-dataset labels/scaling are leaky and not revised validation evidence',
       'rows':len(data),'smokers':int((data.smoker=='yes').sum()),
       'smokers_in_high':int(((data.smoker=='yes') & (tier=='High')).sum()),
       'high_non_smokers':int(mask.sum()),'high_non_smokers_recovered':int((predictions[mask]=='High').sum()),
       'predicted_high_equals_smoker':bool(((predictions=='High')==(data.smoker=='yes')).all()),
       'warnings':sorted(set(str(w.message) for w in caught))}
(root/'legacy/baseline_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
