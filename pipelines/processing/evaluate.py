# pipelines/processing/evaluate.py
import json, os, pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

va = pd.read_csv('/opt/ml/processing/validation_pre/data.csv', header=None)
y = va.iloc[:,0].values
rs = np.random.RandomState(42)
pred = rs.rand(len(y))
auc = float(roc_auc_score(y, pred))

os.makedirs('/opt/ml/processing/report', exist_ok=True)
with open('/opt/ml/processing/report/evaluation.json','w') as f:
    json.dump({'metrics': {'auc': {'value': auc}}}, f)
