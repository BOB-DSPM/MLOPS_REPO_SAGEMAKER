#!/usr/bin/env python3
# extract.py — Prepare binary:logistic-ready CSVs for XGBoost
# - Reads raw ad click CSV (no header) from S3 or local path
# - Assigns column names
# - Converts all features to numeric
# - Ensures label (clicked) is the FIRST column with values in {0,1}
# - Splits into train/validation and writes to /opt/ml/processing/{train,validation}/data.csv
# - Writes a small preview file for human sanity-check

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from typing import Tuple
from urllib.parse import urlparse, unquote

import boto3
import pandas as pd


# ---------------------------
# Logging
# ---------------------------
def setup_logging() -> None:
    fmt = "[%(levelname)s] %(asctime)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)


# ---------------------------
# S3 helpers
# ---------------------------
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    s3://bucket/key -> (bucket, key)
    """
    u = urlparse(uri)
    if u.scheme != "s3" or not u.netloc:
        raise ValueError(f"Not a valid s3 uri: {uri}")
    return u.netloc, unquote(u.path.lstrip("/"))


def s3_head_object(s3, bucket: str, key: str) -> None:
    s3.head_object(Bucket=bucket, Key=key)


def s3_read_csv_no_header(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    # 원본은 헤더 없음 → header=None
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), header=None)
    return df


# ---------------------------
# Transform
# ---------------------------
RAW_HEADER = [
    "user_id",
    "user_name",
    "age",
    "gender",
    "device",
    "position",
    "category",
    "time_of_day",
    "clicked",  # label (raw last col)
]


def to_numeric_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw columns to numeric-only feature matrix with label first.
    - clicked -> int in {0,1} as first column
    - user_id -> int (NaN -> 0)
    - user_name -> extract trailing digits -> int (NaN -> 0)
    - age -> float (NaN -> -1)
    - categorical cols -> categorical codes (int)
    """
    out = pd.DataFrame()

    # 1) Label first: ensure 0/1
    lab = pd.to_numeric(df["clicked"], errors="coerce").fillna(0).astype(int)
    # clip to {0,1}, then validate
    lab = lab.clip(0, 1)
    uniq = set(lab.unique().tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"Invalid label set after coercion: {uniq}")
    out["clicked"] = lab

    # 2) Numeric features
    out["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(0).astype(int)
    out["user_id_from_name"] = (
        df["user_name"].astype(str).str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
    )
    out["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(-1).astype(float)

    # 3) Categorical to codes (stable deterministic codes per run)
    def cat_codes(series: pd.Series) -> pd.Series:
        # convert None/NaN to empty string, then to categorical codes
        return series.astype(str).fillna("").astype("category").cat.codes.astype(int)

    for col in ["gender", "device", "position", "category", "time_of_day"]:
        out[col] = cat_codes(df[col])

    # All numeric, header will be removed when saving
    return out


# ---------------------------
# I/O (local)
# ---------------------------
PROC_TRAIN_DIR = "/opt/ml/processing/train"
PROC_VALID_DIR = "/opt/ml/processing/validation"


def ensure_dirs() -> None:
    os.makedirs(PROC_TRAIN_DIR, exist_ok=True)
    os.makedirs(PROC_VALID_DIR, exist_ok=True)


def save_csv_no_header(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, header=False)


def save_preview(df: pd.DataFrame, path: str, n: int = 5) -> None:
    with open(path, "w") as f:
        f.write(df.head(n).to_csv(index=False, header=False))


def save_metrics_json(metrics_path: str, **kwargs) -> None:
    with open(metrics_path, "w") as f:
        json.dump(kwargs, f, ensure_ascii=False, indent=2)


# ---------------------------
# Main
# ---------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract & transform dataset for XGBoost (binary:logistic)")
    ap.add_argument("--bucket", required=True, help="Target S3 bucket for pipeline artifacts (not used for reads)")
    ap.add_argument("--prefix", required=True, help="Prefix under the bucket for pipeline artifacts (not used for reads)")
    ap.add_argument(
        "--external-csv",
        required=True,
        help="Source CSV without header. s3://bucket/key or local file path",
    )
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (0<r<1)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before split")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed for shuffling")
    return ap


def main() -> None:
    setup_logging()
    ap = build_argparser()
    args = ap.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-northeast-2"
    s3 = boto3.client("s3", region_name=region)

    # Load raw
    logging.info(f"Loading source CSV: {args.external_csv}")
    if args.external_csv.startswith("s3://"):
        bucket, key = parse_s3_uri(args.external_csv)
        s3_head_object(s3, bucket, key)
        raw = s3_read_csv_no_header(s3, bucket, key)
    else:
        # Local file support for offline testing
        raw = pd.read_csv(args.external_csv, header=None)

    # Assign header and validate shape
    raw.columns = RAW_HEADER
    if raw.shape[1] != len(RAW_HEADER):
        raise ValueError(f"Unexpected column count: got {raw.shape[1]}, expected {len(RAW_HEADER)}")

    logging.info(f"Raw shape: {raw.shape}")

    # Transform -> numeric with label first
    out = to_numeric_block(raw)
    logging.info(f"Transformed shape: {out.shape}")
    # Quick sanity check
    labs = set(out.iloc[:, 0].unique().tolist())
    if not labs <= {0, 1}:
        raise ValueError(f"Label sanity check failed: {labs}")

    # Optional shuffle
    if args.shuffle:
        out = out.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)

    # Split
    n = len(out)
    split = max(1, min(n - 1, int(n * args.train_ratio)))  # ensure both splits non-empty
    train = out.iloc[:split].copy()
    valid = out.iloc[split:].copy()
    logging.info(f"Split sizes: train={len(train)}, valid={len(valid)} (ratio={args.train_ratio})")

    # Save to processing outputs
    ensure_dirs()
    train_path = os.path.join(PROC_TRAIN_DIR, "data.csv")
    valid_path = os.path.join(PROC_VALID_DIR, "data.csv")
    save_csv_no_header(train, train_path)
    save_csv_no_header(valid, valid_path)

    # Human preview & simple metrics
    save_preview(train, os.path.join(PROC_TRAIN_DIR, "preview.txt"), n=5)
    save_metrics_json(
        os.path.join(PROC_TRAIN_DIR, "metrics.json"),
        rows_total=n,
        rows_train=len(train),
        rows_valid=len(valid),
        label_unique=sorted(list(labs)),
        columns=out.shape[1],
    )

    # Final logs to help console debugging
    logging.info("Wrote files:")
    logging.info(f"- {train_path}")
    logging.info(f"- {valid_path}")
    logging.info(f"- {os.path.join(PROC_TRAIN_DIR, 'preview.txt')}")
    logging.info(f"- {os.path.join(PROC_TRAIN_DIR, 'metrics.json')}")
    logging.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("extract.py failed")
        # Non-zero exit so SageMaker marks the Processing step as Failed with the stack trace.
        sys.exit(1)
