# ──────────────────────────────────────────────────────────────────────────────
# file: pipelines/processing/extract.py
# - If --external-csv is provided, download CSV from that S3 URI (no header).
# - Otherwise, synthesize a binary classification CSV with label in column 0.
# Outputs recorded as named Processing outputs: "train", "validation"
# ──────────────────────────────────────────────────────────────────────────────
import argparse, os, tempfile, json
import numpy as np
import pandas as pd
import boto3
from urllib.parse import urlparse

def _write_split(df: pd.DataFrame, out_dir: str, train_ratio: float = 0.8, seed: int = 42):
    rs = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rs.shuffle(idx)
    cut = int(len(df) * train_ratio)
    tr_idx, va_idx = idx[:cut], idx[cut:]
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "validation"), exist_ok=True)
    df.iloc[tr_idx].to_csv(os.path.join(out_dir, "train", "data.csv"), index=False, header=False)
    df.iloc[va_idx].to_csv(os.path.join(out_dir, "validation", "data.csv"), index=False, header=False)

def _synthesize(n: int = 500, m: int = 5, seed: int = 42) -> pd.DataFrame:
    rs = np.random.RandomState(seed)
    X = rs.randn(n, m)
    y = (X.sum(axis=1) > 0).astype(int)
    return pd.DataFrame(np.column_stack([y, X]))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--external-csv", default="")
    args = p.parse_args()

    local_base = "/opt/ml/processing"
    out_dir = local_base  # We'll place under train/ and validation/
    if args.external_csv:
        u = urlparse(args.external_csv)
        bucket, key = u.netloc, u.path.lstrip("/")
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            s3.download_fileobj(bucket, key, f)
            local = f.name
        df = pd.read_csv(local, header=None)
    else:
        df = _synthesize(n=600, m=8)

    _write_split(df, out_dir)
    # Processing named outputs are auto-detected by directory names when using SDK v2
    # but to be explicit for SageMaker Pipelines, we rely on the Step definition's Outputs mapping.
