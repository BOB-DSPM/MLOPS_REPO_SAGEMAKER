# ──────────────────────────────────────────────────────────────────────────────
# file: pipelines/processing/extract.py
# ──────────────────────────────────────────────────────────────────────────────
import argparse, os, tempfile
import numpy as np, pandas as pd, boto3
from urllib.parse import urlparse

def _write_split(df: pd.DataFrame, out_dir: str, train_ratio: float = 0.8, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df)); rng.shuffle(idx)
    cut = int(len(df)*train_ratio)
    tr, va = idx[:cut], idx[cut:]
    os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "validation"), exist_ok=True)
    df.iloc[tr].to_csv(os.path.join(out_dir, "train", "data.csv"), index=False, header=False)
    df.iloc[va].to_csv(os.path.join(out_dir, "validation", "data.csv"), index=False, header=False)

def _synthesize(n: int = 600, m: int = 8, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, m)
    y = (X.sum(axis=1) > 0).astype(int)
    return pd.DataFrame(np.column_stack([y, X]))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--external-csv", default="")
    args = p.parse_args()

    if args.external_csv:
        u = urlparse(args.external_csv)
        s3 = boto3.client("s3")
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            s3.download_fileobj(u.netloc, u.path.lstrip("/"), f)
            local = f.name
        df = pd.read_csv(local, header=None)
    else:
        df = _synthesize()

    _write_split(df, "/opt/ml/processing")