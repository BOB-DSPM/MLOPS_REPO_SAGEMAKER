# pipelines/pipeline_def.py
from __future__ import annotations
import argparse
import os
import boto3
from pathlib import Path
from botocore.exceptions import ClientError

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
from sagemaker.workflow.functions import Join  # ★ PipelineVariable 결합용

def env_or(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default

def _upload_code(local_path: str, bucket: str, key: str) -> str:
    """로컬 스크립트를 지정 버킷/키로 업로드하고 s3:// URI 반환."""
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

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

    # ── 정적 코드 S3 경로(기본 버킷 회피)
    data_bucket_str = env_or("DATA_BUCKET", "my-mlops-dev-data")
    prefix_str = env_or("PREFIX", "pipelines/exp1")

    repo_root = Path(__file__).resolve().parent.parent
    local_extract = str(repo_root / "pipelines" / "processing" / "extract.py")
    local_evaluate = str(repo_root / "pipelines" / "processing" / "evaluate.py")

    extract_key = f"{prefix_str}/code/extract.py"
    evaluate_key = f"{prefix_str}/code/evaluate.py"

    s3_code_extract = _upload_code(local_extract, data_bucket_str, extract_key)
    s3_code_evaluate = _upload_code(local_evaluate, data_bucket_str, evaluate_key)

    # ── 공통 S3 출력 경로 (전부 "정적 문자열"로, 기본 버킷 사용 금지)
    s3_train_out     = f"s3://{data_bucket_str}/{prefix_str}/processing/train"
    s3_val_out       = f"s3://{data_bucket_str}/{prefix_str}/processing/validation"
    s3_eval_out      = f"s3://{data_bucket_str}/{prefix_str}/processing/evaluation"
    s3_transform_out = f"s3://{data_bucket_str}/{prefix_str}/transform/validation"
    s3_train_output_path = f"s3://{data_bucket_str}/{prefix_str}/training"  # Estimator output_path 용

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
        code=s3_code_extract,
        job_arguments=[
            "--bucket", p_bucket,
            "--prefix", p_prefix,
            "--external-csv", p_use_ext_csv,
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=s3_train_out),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation", destination=s3_val_out),
        ],
    )
    train_s3 = step_extract.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
    val_s3   = step_extract.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri

    # ★ PipelineVariable에 리터럴을 붙일 땐 문자열 포매팅 금지 → Join 사용
    train_s3_file = Join(on="", values=[train_s3, "/data.csv"])
    val_s3_file   = Join(on="", values=[val_s3,   "/data.csv"])

    # ── Step 2: Train (★ output_path를 “정적 문자열”로 강제 지정)
    estimator = Estimator(
        image_uri=train_image,
        role=role_arn,
        instance_count=1,
        instance_type=p_train_instance,
        max_run=p_train_max_runtime,
        sagemaker_session=sm_sess,
        base_job_name="xgb-train",
        enable_sagemaker_metrics=True,
        output_path=s3_train_output_path,  # <- 기본 버킷 경로 생성 방지
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
            # ★ processing 산출 파일(data.csv)로 직접 지정 + 명시적 File 모드
            "train": TrainingInput(s3_data=train_s3_file, content_type="text/csv", input_mode="File"),
            "validation": TrainingInput(s3_data=val_s3_file, content_type="text/csv", input_mode="File"),
        },
    )

    # ── Step 3: CreateModel
    model = Model(
        image_uri=train_image,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        sagemaker_session=sm_sess,
    )
    step_create_model = ModelStep(name="CreateModel", step_args=model.create())

    # ── Step 4: Transform (★ output_path도 정적 문자열)
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type=p_transform_instance,
        accept="text/csv",
        assemble_with="Line",
        strategy="SingleRecord",
        sagemaker_session=sm_sess,
        output_path=s3_transform_out,
    )
    step_transform = TransformStep(
        name="TransformValidation",
        transformer=transformer,
        inputs=TransformInput(
            # ★ validation 파일 경로도 Join 사용
            data=val_s3_file,
            content_type="text/csv",
            split_type="Line",
            input_filter="$[1:]",  # 첫 컬럼(라벨) 제외 (필요 시 유지)
        ),
    )

    # ── Step 5: Evaluate (★ 출력 경로 정적 문자열)
    metrics_pf = PropertyFile(name="EvalReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="Evaluate",
        processor=sklearn_proc,
        code=s3_code_evaluate,
        job_arguments=[
            "--validation-s3", val_s3,  # evaluate.py가 단일 파일을 요구하면 val_s3_file로 교체
            "--pred-s3", step_transform.properties.TransformOutput.S3OutputPath,
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=s3_eval_out)],
        property_files=[metrics_pf],
    )

    # ── Step 6: Register (평가 파일 경로 정적 규칙)
    mpg_name = env_or("MODEL_PACKAGE_GROUP_NAME", "my-mlops-dev-pkg")
    eval_json_uri = f"{s3_eval_out}/evaluation.json"
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=eval_json_uri,
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
    import time, json
    region = boto3.Session().region_name
    role = os.environ["SM_EXEC_ROLE_ARN"]
    sm   = boto3.client("sagemaker", region_name=region)

    pipe = get_pipeline(region, role)
    pipe.upsert(role_arn=role)
    exe = pipe.start()
    arn = exe.arn
    print("Pipeline execution started:", arn)

    if not wait:
        print("[info] --wait 미사용: CodeBuild는 곧바로 성공 종료하고, 파이프라인은 백그라운드에서 실행됩니다.")
        print(f"[open] 콘솔: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/execute/{arn}")
        return

    # 수동 폴링로직 + 실패시 상세 출력
    def _desc():
        return sm.describe_pipeline_execution(PipelineExecutionArn=arn)

    def _list_steps():
        try:
            return sm.list_pipeline_execution_steps(PipelineExecutionArn=arn, SortOrder="Ascending")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "AccessDeniedException":
                print("  [warn] missing permission: sagemaker:ListPipelineExecutionSteps (skipping step diagnostics)")
                return {}
            raise

    # 대기 루프
    while True:
        d = _desc()
        st = d["PipelineExecutionStatus"]
        if st in ("Executing", "Stopping"):
            time.sleep(20)
            continue
        print("[status]", st)
        if st == "Failed":
            fr = d.get("FailureReason")
            if fr:
                print("FailureReason:", fr)
            print(f"[open] 콘솔: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/execute/{arn}")
        if st == "Succeeded":
            print("Execution completed: Succeeded")
            return

        # === 실패/중단: 스텝별 원인/리소스/로그 링크 출력 ===
        print("[error] pipeline failed. printing step diagnostics...\n")
        steps = _list_steps().get("PipelineExecutionSteps", [])
        for s in steps:
            name = s.get("StepName")
            t    = s.get("StepType")
            st2  = s.get("StepStatus")
            meta = s.get("Metadata", {})
            fr   = s.get("FailureReason") or meta.get("FailureReason")
            print(f"--- Step: {name} [{t}] => {st2}")
            if fr:
                print("FailureReason:", fr)

            # 작업별 세부조사
            try:
                if t == "Processing":
                    pj = meta.get("ProcessingJob", {}).get("Arn")
                    if pj:
                        jn = pj.split("/")[-1]
                        dj = sm.describe_processing_job(ProcessingJobName=jn)
                        print("  ProcessingJob:", jn)
                        print("  AppLogs (CloudWatch):", dj.get("ProcessingJobArn"))
                        print("  S3 Outputs:", json.dumps(dj.get("ProcessingOutputConfig", {}), ensure_ascii=False))
                        print(f"  콘솔: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/processing-jobs/{jn}")
                elif t == "Training":
                    tj = meta.get("TrainingJob", {}).get("Arn")
                    if tj:
                        jn = tj.split("/")[-1]
                        dj = sm.describe_training_job(TrainingJobName=jn)
                        print("  TrainingJob:", jn)
                        print("  FinalModelArtifacts:", dj.get("ModelArtifacts", {}))
                        print(f"  콘솔: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/training-jobs/{jn}")
                elif t == "Transform":
                    tj = meta.get("TransformJob", {}).get("Arn")
                    if tj:
                        jn = tj.split("/")[-1]
                        dj = sm.describe_transform_job(TransformJobName=jn)
                        print("  TransformJob:", jn)
                        print("  OutputPath:", dj.get("TransformOutput", {}))
                        print(f"  콘솔: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/transform-jobs/{jn}")
                elif t == "Model":
                    mdl = meta.get("Model", {}).get("Arn")
                    if mdl:
                        print("  ModelArn:", mdl)
                        mn = mdl.split("/")[-1]
                        print(f"  콘솔: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/models/{mn}")
                elif t == "RegisterModel":
                    mp = meta.get("RegisterModel", {}).get("Arn")
                    if mp:
                        print("  ModelPackageArn:", mp)
                        print(f"  콘솔: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/model-packages")
            except Exception as e:
                print("  [warn] detail fetch error:", e)

            print()

        raise SystemExit(1)

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
