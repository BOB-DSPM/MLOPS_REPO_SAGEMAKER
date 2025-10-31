# pipelines/processing/extract.py
import argparse, os, json, numpy as np, pandas as pd
from urllib.parse import urlparse
import boto3, tempfile

p = argparse.ArgumentParser()
p.add_argument("--s3", required=True)
p.add_argument("--csv", default="")
p.add_argument("--use-feature-store", default="false")
p.add_argument("--feature-group-name", default="")
args = p.parse_args()

os.makedirs("/opt/ml/processing/train", exist_ok=True)
os.makedirs("/opt/ml/processing/validation", exist_ok=True)

def write_csv(path, n=400, m=5):
    X = np.random.randn(n,m)
    y = (X.sum(axis=1) > 0).astype(int)
    pd.DataFrame(np.column_stack([y, X])).to_csv(path, index=False, header=False)

if args.csv:
    s3 = boto3.client("s3")
    u = urlparse(args.csv)
    b, k = u.netloc, u.path.lstrip("/")
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        s3.download_fileobj(b, k, f)
        local = f.name
    df = pd.read_csv(local, header=None)
    tr = df.sample(frac=0.8, random_state=42)
    va = df.drop(tr.index)
    tr.to_csv("/opt/ml/processing/train/data.csv", index=False, header=False)
    va.to_csv("/opt/ml/processing/validation/data.csv", index=False, header=False)
else:
    write_csv("/opt/ml/processing/train/data.csv", n=400)
    write_csv("/opt/ml/processing/validation/data.csv", n=100)
