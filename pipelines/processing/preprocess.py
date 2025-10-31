# pipelines/processing/preprocess.py
import os, pandas as pd
os.makedirs('/opt/ml/processing/train_pre', exist_ok=True)
os.makedirs('/opt/ml/processing/validation_pre', exist_ok=True)
tr = pd.read_csv('/opt/ml/processing/train/data.csv', header=None)
va = pd.read_csv('/opt/ml/processing/validation/data.csv', header=None)

y_tr, X_tr = tr.iloc[:,0], tr.iloc[:,1:]
y_va, X_va = va.iloc[:,0], va.iloc[:,1:]

mu = X_tr.mean()
sd = X_tr.std(ddof=0).replace(0, 1.0)
X_tr = (X_tr - mu)/sd
X_va = (X_va - mu)/sd

pd.concat([y_tr, X_tr], axis=1).to_csv('/opt/ml/processing/train_pre/data.csv', index=False, header=False)
pd.concat([y_va, X_va], axis=1).to_csv('/opt/ml/processing/validation_pre/data.csv', index=False, header=False)
