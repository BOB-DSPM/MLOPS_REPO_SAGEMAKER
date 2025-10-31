# ──────────────────────────────────────────────────────────────────────────────
# file: pipelines/processing/evaluate.py
# ──────────────────────────────────────────────────────────────────────────────
import argparse, os, io, json, boto3
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

def _read_all_from_s3(prefix: str) -> pd.Series:
    s3 = boto3.client("s3")
    assert prefix.startswith("s3://")
    _, rest = prefix.split("s3://", 1)
    bucket, key_prefix = rest.split("/", 1)
    paginator = s3.get_paginator("list_objects_v2")
    preds = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if not ("part" in k or k.endswith(".out") or k.endswith(".csv")):
                continue
            body = s3.get_object(Bucket=bucket, Key=k)["Body"].read().decode("utf-8")
            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    preds.append(float(line.split(",")[0]))
                except Exception:
                    try:
                        preds.append(float(line))
                    except Exception:
                        pass
    return pd.Series(preds, dtype=float)

def _download_validation_csv(prefix: str, local_path: str) -> None:
    assert prefix.startswith("s3://")
    _, rest = prefix.split("s3://", 1)
    bucket, key_prefix = rest.split("/", 1)
    key = key_prefix.rstrip("/") + "/data.csv"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    boto3.client("s3").download_file(bucket, key, local_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-s3", required=True)
    ap.add_argument("--pred-s3", required=True)
    args = ap.parse_args()

    local_val = "/opt/ml/processing/validation.csv"
    _download_validation_csv(args.validation_s3, local_val)
    val = pd.read_csv(local_val, header=None)
    y_true = val.iloc[:, 0].astype(int).to_numpy()

    y_pred_prob = _read_all_from_s3(args.pred_s3).to_numpy()
    n = min(len(y_true), len(y_pred_prob))
    y_true, y_pred_prob = y_true[:n], y_pred_prob[:n]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        auc = float("nan")
    acc = float(accuracy_score(y_true, y_pred))

    os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
    report = {
        "binary_classification_metrics": {
            "auc": {"value": auc, "standard_deviation": "NaN"},
            "accuracy": {"value": acc, "standard_deviation": "NaN"},
        }
    }
    with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
        json.dump(report, f)
    print("Wrote metrics:", report)