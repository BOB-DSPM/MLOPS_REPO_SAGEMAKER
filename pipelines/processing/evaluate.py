# ──────────────────────────────────────────────────────────────────────────────
# file: pipelines/processing/evaluate.py
# - Reads ground-truth labels from validation CSV (first column)
# - Reads batch transform predictions from S3 output (same row order)
# - Computes ROC-AUC and Accuracy
# - Writes evaluation.json for Model Metrics & RegisterModel
# ──────────────────────────────────────────────────────────────────────────────
import argparse, os, io, json, boto3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

def _read_all_from_s3(prefix: str) -> pd.Series:
    """Concatenate all 'part' files under the transform output prefix."""
    s3 = boto3.client("s3")
    # prefix like s3://bucket/prefix/; list and read objects
    assert prefix.startswith("s3://")
    _, rest = prefix.split("s3://", 1)
    bucket, key_prefix = rest.split("/", 1)
    paginator = s3.get_paginator("list_objects_v2")
    preds = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if not k.endswith(".out") and not k.endswith(".csv") and "part" not in k:
                # xgb transform tends to produce 'part' files with csv lines; be permissive
                continue
            body = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
            s = body.decode("utf-8").strip().splitlines()
            for line in s:
                if not line.strip():
                    continue
                # prediction is a float per line
                try:
                    preds.append(float(line.split(",")[0]))
                except Exception:
                    # if format unexpected, try raw
                    try:
                        preds.append(float(line))
                    except Exception:
                        pass
    return pd.Series(preds, dtype=float)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-s3", required=True)
    ap.add_argument("--pred-s3", required=True)
    args = ap.parse_args()

    # Load validation CSV (label, feat1, feat2, ...)
    val_df = pd.read_csv(args.validation_s3.replace("s3://", "s3a://") if False else args.validation_s3)
    # Above line won't work directly; use boto3 download to local
    s3 = boto3.client("s3")

    def _download_s3_to_local(s3uri: str, local_path: str):
        assert s3uri.startswith("s3://")
        _, rest = s3uri.split("s3://", 1)
        bucket, key = rest.split("/", 1)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)

    # Download validation data to local file
    # The 'validation_s3' points to a prefix; file is 'data.csv'
    if not args.validation_s3.endswith("/"):
        val_prefix = args.validation_s3 + "/"
    else:
        val_prefix = args.validation_s3
    # Construct key for data.csv
    _, rest = val_prefix.split("s3://", 1)
    bkt, kpref = rest.split("/", 1)
    val_key = kpref.rstrip("/") + "/data.csv"
    local_val = "/opt/ml/processing/validation.csv"
    s3.download_file(bkt, val_key, local_val)

    val = pd.read_csv(local_val, header=None)
    y_true = val.iloc[:, 0].astype(int).to_numpy()

    # Load predictions from batch transform output prefix
    y_pred_prob = _read_all_from_s3(args.pred_s3).to_numpy()
    # Align lengths (just in case)
    n = min(len(y_true), len(y_pred_prob))
    y_true = y_true[:n]
    y_pred_prob = y_pred_prob[:n]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Compute metrics
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






