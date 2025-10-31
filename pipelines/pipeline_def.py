# ──────────────────────────────────────────────────────────────────────────────
# file: pipelines/pipeline_def.py
# SageMaker Pipelines (Processing → Train → Transform → Evaluate → Register)
# Compatible with sagemaker==2.254.1 (no Endpoint* pipeline steps used)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import os
import json
import boto3
from sagemaker.session import Session
from sagemaker.workflow.parameters import (
    ParameterString, ParameterInteger, ParameterFloat
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.inputs import TrainingInput, TransformInput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.estimator import Estimator
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.model_step import RegisterModel
from sagemaker import image_uris


def env_or(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


# -----------------------------
# Pipeline definition factory
# -----------------------------
def get_pipeline(region: str, role_arn: str) -> Pipeline:
    sess = boto3.Session(region_name=region)
    sm_sess = PipelineSession(boto_session=sess, sagemaker_client=sess.client("sagemaker"))

    # Parameters (can be overridden at start time)
    p_prefix = ParameterString("Prefix", default_value=env_or("PREFIX", "pipelines/exp1"))
    p_bucket = ParameterString("DataBucket", default_value=env_or("DATA_BUCKET", "my-mlops-dev-data"))
    p_use_ext_csv = ParameterString("ExternalCsvUri", default_value=env_or("EXTERNAL_CSV_URI", ""))
    p_process_instance = ParameterString("ProcessingInstanceType", default_value=env_or("SM_INSTANCE_TYPE", "ml.m5.large"))
    p_train_instance = ParameterString("TrainingInstanceType", default_value=env_or("SM_INSTANCE_TYPE", "ml.m5.large"))
    p_transform_instance = ParameterString("TransformInstanceType", default_value=env_or("SM_INSTANCE_TYPE", "ml.m5.large"))
    p_train_max_runtime = ParameterInteger("TrainMaxRuntimeSeconds", default_value=3600)
    p_metric_auc_threshold = ParameterFloat("AUCThreshold", default_value=0.65)

    # Built-in XGBoost image URI (if not provided via env, resolve by SDK)
    default_train_img = image_uris.retrieve(
        region=region, framework="xgboost", version="1.7-1"
    )
    train_image = os.environ.get("TRAIN_IMAGE_URI", default_train_img)

    # ──────────── Step 1: EXTRACT (create or download CSV, split train/val)
    # Produces:
    #   /opt/ml/processing/train/data.csv
    #   /opt/ml/processing/validation/data.csv
    sklearn_proc = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=p_process_instance,
        instance_count=1,
        sagemaker_session=sm_sess,
        base_job_name="extract-csv",
    )

    step_extract = ProcessingStep(
        name="Extract",
        processor=sklearn_proc,
        code="pipelines/processing/extract.py",
        job_arguments=[
            "--bucket", p_bucket,
            "--prefix", p_prefix,
            "--external-csv", p_use_ext_csv,
        ],
        outputs=[
            # The processor will write train/validation under these output paths
            # Pipeline will capture to S3 automatically
        ],
    )

    # Locations for training/validation from previous step
    train_s3 = step_extract.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
    val_s3 = step_extract.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri

    # ──────────── Step 2: TRAIN (Built-in XGBoost)
    estimator = Estimator(
        image_uri=train_image,
        role=role_arn,
        instance_count=1,
        instance_type=p_train_instance,
        max_run=p_train_max_runtime,
        sagemaker_session=sm_sess,
        base_job_name="xgb-train",
        output_path=f"s3://{p_bucket}/%s/model" % p_prefix,  # model artifact location
        enable_sagemaker_metrics=True,
    )
    # Typical XGB binary classification hyperparams (simple demo)
    estimator.set_hyperparameters(
        objective="binary:logistic",
        eval_metric="auc",
        num_round=100,
        max_depth=5,
        eta=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=1,
    )

    train_step = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(s3_data=train_s3, content_type="text/csv"),
            "validation": TrainingInput(s3_data=val_s3, content_type="text/csv"),
        },
    )

    # ──────────── Step 3: TRANSFORM (Batch transform on validation set)
    transformer = train_step.get_transformer(
        instance_count=1,
        instance_type=p_transform_instance,
        accept="text/csv",
        strategy="SingleRecord",
        assemble_with="Line",
        output_path=f"s3://{p_bucket}/%s/transform" % p_prefix,
    )

    transform_input = TransformInput(
        data=val_s3,  # validation CSV incl. label in 1st column
        content_type="text/csv",
        split_type="Line",
        input_filter="$[1:]",    # drop label (1st col) before prediction
    )

    step_transform = TransformStep(
        name="TransformValidation",
        transformer=transformer,
        inputs=transform_input,
    )

    # ──────────── Step 4: EVALUATE (compute AUC from transform output vs labels)
    metrics_pf = PropertyFile(name="EvalReport", output_name="evaluation", path="evaluation.json")

    step_eval = ProcessingStep(
        name="Evaluate",
        processor=sklearn_proc,
        code="pipelines/processing/evaluate.py",
        job_arguments=[
            "--validation-s3", val_s3,
            "--pred-s3", step_transform.properties.TransformOutput.S3OutputPath,
        ],
        outputs=[
            {
                "OutputName": "evaluation",
                "AppManaged": True,
                "S3Output": {
                    "S3Uri": f"s3://{p_bucket}/%s/eval" % p_prefix,
                    "LocalPath": "/opt/ml/processing/evaluation",
                    "S3UploadMode": "EndOfJob",
                },
            },
        ],
        property_files=[metrics_pf],
    )

    # ──────────── Step 5: REGISTER (Model Registry with metrics)
    mpg_name = env_or("MODEL_PACKAGE_GROUP_NAME", "my-mlops-dev-pkg")

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri + "/evaluation.json",
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large", "ml.m5.xlarge"],
        model_package_group_name=mpg_name,
        model_metrics=model_metrics,
    )

    # (Optional) You could add a condition on AUC >= threshold to gate registration.
    # For simplicity, we always register; if you want gating, add a ConditionStep here.

    return Pipeline(
        name="SageMaker-ML-Exp1",
        parameters=[
            p_prefix,
            p_bucket,
            p_use_ext_csv,
            p_process_instance,
            p_train_instance,
            p_transform_instance,
            p_train_max_runtime,
            p_metric_auc_threshold,
        ],
        steps=[step_extract, train_step, step_transform, step_eval, register_step],
        sagemaker_session=sm_sess,
    )


# -----------------------------
# CLI entry
# -----------------------------
def upsert_and_start(wait: bool = False) -> None:
    region = boto3.Session().region_name
    role = os.environ["SM_EXEC_ROLE_ARN"]
    pipe = get_pipeline(region, role)
    pipe.upsert(role_arn=role)

    # Start with defaults (can pass specific param values here if desired)
    execution = pipe.start()
    print("Pipeline execution started:", execution.arn)
    if wait:
        execution.wait()
        print("Execution completed with status:", execution.describe().get("PipelineExecutionStatus"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Upsert and execute the pipeline")
    ap.add_argument("--wait", action="store_true", help="Wait for execution to finish")
    ap.add_argument("--register-only", action="store_true", help="(unused in this version)")
    args = ap.parse_args()

    if args.run:
        upsert_and_start(wait=args.wait)
    else:
        # Just upsert (no run)
        region = boto3.Session().region_name
        role = os.environ["SM_EXEC_ROLE_ARN"]
        p = get_pipeline(region, role)
        p.upsert(role_arn=role)
        print("Pipeline upserted (no execution started).")