# ──────────────────────────────────────────────────────────────────────────────
# file: pipelines/pipeline_def.py
# Flow: Extract(Processing) → Train → CreateModel → Transform → Evaluate → Register
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse
import os
import boto3

from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, TransformInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.transformer import Transformer

from sagemaker.processing import ProcessingOutput
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger,
    ParameterFloat,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.functions import Join  # (Eval JSON 경로 조립에만 사용)

def env_or(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def get_pipeline(region: str, role_arn: str) -> Pipeline:
    boto_sess = boto3.Session(region_name=region)
    sm_sess = PipelineSession(
        boto_session=boto_sess,
        sagemaker_client=boto_sess.client("sagemaker"),
    )

    # ── Parameters
    p_prefix = ParameterString("Prefix", default_value=env_or("PREFIX", "pipelines/exp1"))
    p_bucket = ParameterString("DataBucket", default_value=env_or("DATA_BUCKET", "my-mlops-dev-data"))
    p_use_ext_csv = ParameterString("ExternalCsvUri", default_value=env_or("EXTERNAL_CSV_URI", ""))
    p_process_instance = ParameterString("ProcessingInstanceType", default_value=env_or("SM_INSTANCE_TYPE", "ml.m5.large"))
    p_train_instance = ParameterString("TrainingInstanceType", default_value=env_or("SM_INSTANCE_TYPE", "ml.m5.large"))
    p_transform_instance = ParameterString("TransformInstanceType", default_value=env_or("SM_INSTANCE_TYPE", "ml.m5.large"))
    p_train_max_runtime = ParameterInteger("TrainMaxRuntimeSeconds", default_value=3600)
    p_metric_auc_threshold = ParameterFloat("AUCThreshold", default_value=0.65)

    # ── Training image (XGBoost)
    default_train_img = image_uris.retrieve(region=region, framework="xgboost", version="1.7-1")
    train_image = os.environ.get("TRAIN_IMAGE_URI", default_train_img)

    # ── Step 1: Extract (Processing)
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
        # ✅ 파이프라인 변수 대신 '로컬 파일 경로' 사용 (SDK가 S3로 업로드)
        code="pipelines/processing/extract.py",
        job_arguments=[
            "--bucket", p_bucket,
            "--prefix", p_prefix,
            "--external-csv", p_use_ext_csv,
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
        ],
    )
    train_s3 = step_extract.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
    val_s3   = step_extract.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri

    # ── Step 2: Train
    estimator = Estimator(
        image_uri=train_image,
        role=role_arn,
        instance_count=1,
        instance_type=p_train_instance,
        max_run=p_train_max_runtime,
        sagemaker_session=sm_sess,
        base_job_name="xgb-train",
        enable_sagemaker_metrics=True,
    )
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

    # ── Step 3: CreateModel
    model = Model(
        image_uri=train_image,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        sagemaker_session=sm_sess,
    )

    step_create_model = ModelStep(
        name="CreateModel",
        step_args=model.create(),
    )

    # ── Step 4: Transform (Batch inference on validation set)
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type=p_transform_instance,
        accept="text/csv",
        assemble_with="Line",
        strategy="SingleRecord",
        sagemaker_session=sm_sess,
        output_path=None,  # pipeline execution별 기본 S3 출력 경로 사용
    )
    step_transform = TransformStep(
        name="TransformValidation",
        transformer=transformer,
        inputs=TransformInput(
            data=val_s3,
            content_type="text/csv",
            split_type="Line",
            input_filter="$[1:]",  # 첫 컬럼(label) 제외
        ),
    )

    # ── Step 5: Evaluate
    metrics_pf = PropertyFile(name="EvalReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="Evaluate",
        processor=sklearn_proc,
        # ✅ 로컬 파일 경로 사용
        code="pipelines/processing/evaluate.py",
        job_arguments=[
            "--validation-s3", val_s3,
            "--pred-s3", step_transform.properties.TransformOutput.S3OutputPath,
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        property_files=[metrics_pf],
    )

    # ── Step 6: Register (Model Registry)
    mpg_name = env_or("MODEL_PACKAGE_GROUP_NAME", "my-mlops-dev-pkg")

    # Eval 결과 JSON S3 경로 조립 (pipeline 변수끼리 문자열 덧셈 금지 → Join 사용)
    eval_json_s3 = Join(
        on="/",
        values=[
            step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            "evaluation.json",
        ],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=eval_json_s3,
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,  # 이미지/환경 정의 상속 목적
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large", "ml.m5.xlarge"],
        model_package_group_name=mpg_name,
        model_metrics=model_metrics,
    )

    return Pipeline(
        name="SageMaker-ML-Exp1",
        parameters=[
            p_prefix, p_bucket, p_use_ext_csv,
            p_process_instance, p_train_instance, p_transform_instance,
            p_train_max_runtime, p_metric_auc_threshold,
        ],
        steps=[step_extract, train_step, step_create_model, step_transform, step_eval, register_step],
        sagemaker_session=sm_sess,
    )


def upsert_and_start(wait: bool = False) -> None:
    region = boto3.Session().region_name
    role = os.environ["SM_EXEC_ROLE_ARN"]
    pipe = get_pipeline(region, role)
    pipe.upsert(role_arn=role)
    exe = pipe.start()
    print("Pipeline execution started:", exe.arn)
    if wait:
        exe.wait()
        print("Execution completed:", exe.describe().get("PipelineExecutionStatus"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    if args.run:
        upsert_and_start(wait=args.wait)
    else:
        region = boto3.Session().region_name
        role = os.environ["SM_EXEC_ROLE_ARN"]
        p = get_pipeline(region, role)
        p.upsert(role_arn=role)
        print("Pipeline upserted (no execution started).")
