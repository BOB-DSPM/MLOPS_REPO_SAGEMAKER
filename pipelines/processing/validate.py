# pipelines/processing/validate.py
import os, json, pandas as pd
tr = pd.read_csv('/opt/ml/processing/train/data.csv', header=None)
va = pd.read_csv('/opt/ml/processing/validation/data.csv', header=None)
os.makedirs('/opt/ml/processing/report', exist_ok=True)
with open('/opt/ml/processing/report/summary.json','w') as f:
    json.dump({'train_rows': len(tr), 'val_rows': len(va)}, f)
