# pipelines/pipeline_def.py
import argparse
import os
import time
import boto3

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterFloat, ParameterInteger,
)
from sagemaker.workflow.steps import (
    CacheConfig, ProcessingStep, TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris
from sagemaker.workflow.pipeline_context import PipelineSession


def get_pipeline(region: str, role: str) -> Pipeline:
    sm_sess = PipelineSession()

    # ---- Parameters (환경변수와 매핑) ----
    p_data_bucket   = ParameterString("DataBucket",        default_value=os.environ.get("DATA_BUCKET", ""))
    p_prefix        = ParameterString("Prefix",            default_value=os.environ.get("PREFIX", "pipelines/exp1"))
    p_instance_type = ParameterString("InstanceType",      default_value=os.environ.get("SM_INSTANCE_TYPE", "ml.m5.large"))
    p_endpoint_name = ParameterString("EndpointName",      default_value=os.environ.get("SM_ENDPOINT_NAME", "mlops-endpoint"))

    default_train_image = image_uris.retrieve(framework="xgboost", region=region, version="1.7-1")
    p_train_image  = ParameterString("TrainImage",         default_value=os.environ.get("TRAIN_IMAGE_URI", default_train_image))
    p_external_csv = ParameterString("ExternalCsvUri",     default_value=os.environ.get("EXTERNAL_CSV_URI", ""))
    p_use_fs       = ParameterString("UseFeatureStore",    default_value=os.environ.get("USE_FEATURE_STORE", "false"))
    p_fg_name      = ParameterString("FeatureGroupName",   default_value=os.environ.get("FEATURE_GROUP_NAME", ""))
    p_mpg          = ParameterString("ModelPackageGroupName", default_value=os.environ.get("MODEL_PACKAGE_GROUP_NAME", "model-pkg"))

    p_auc_threshold= ParameterFloat("AucThreshold",        default_value=0.65)
    p_num_round    = ParameterInteger("NumRound",          default_value=50)

    cache = CacheConfig(enable_caching=False, expire_after="PT1H")

    # ---- 1) Extract ----
    extract = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        sagemaker_session=sm_sess,
    )
    extract_args = extract.run(
        code="processing/extract.py",
        source_dir="pipelines",
        inputs=[],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=Join(on="", values=["s3://", p_data_bucket, "/", p_prefix, "/extract/train"]),
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=Join(on="", values=["s3://", p_data_bucket, "/", p_prefix, "/extract/validation"]),
            ),
        ],
        arguments=[
            "--s3",  Join(on="", values=["s3://", p_data_bucket, "/", p_prefix]),
            "--csv", p_external_csv,
            "--use-feature-store", p_use_fs,
            "--feature-group-name", p_fg_name,
        ],
    )
    extract_step = ProcessingStep(name="Extract", step_args=extract_args, cache_config=cache)

    # ---- 2) Validate ----
    validate = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        sagemaker_session=sm_sess,
    )
    evaluation = PropertyFile(name="ValidationSummary", output_name="report", path="summary.json")
    validate_args = validate.run(
        code="processing/validate.py",
        source_dir="pipelines",
        inputs=[
            ProcessingInput(source=extract_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri, destination="/opt/ml/processing/train"),
            ProcessingInput(source=extract_step.properties.ProcessingOutputConfig.Outputs[1].S3Output.S3Uri, destination="/opt/ml/processing/validation"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="report",
                source="/opt/ml/processing/report",
                destination=Join(on="", values=["s3://", p_data_bucket, "/", p_prefix, "/validate/report"]),
            )
        ],
    )
    validate_step = ProcessingStep(name="Validate", step_args=validate_args, property_files=[evaluation], cache_config=cache)

    # ---- 3) Preprocess ----
    preprocess = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        sagemaker_session=sm_sess,
    )
    preprocess_args = preprocess.run(
        code="processing/preprocess.py",
        source_dir="pipelines",
        inputs=[
            ProcessingInput(source=extract_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri, destination="/opt/ml/processing/train"),
            ProcessingInput(source=extract_step.properties.ProcessingOutputConfig.Outputs[1].S3Output.S3Uri, destination="/opt/ml/processing/validation"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_pre",
                source="/opt/ml/processing/train_pre",
                destination=Join(on="", values=["s3://", p_data_bucket, "/", p_prefix, "/preprocess/train_pre"]),
            ),
            ProcessingOutput(
                output_name="validation_pre",
                source="/opt/ml/processing/validation_pre",
                destination=Join(on="", values=["s3://", p_data_bucket, "/", p_prefix, "/preprocess/validation_pre"]),
            ),
        ],
    )
    preprocess_step = ProcessingStep(name="Preprocess", step_args=preprocess_args, cache_config=cache)

    # ---- 4) Train (XGBoost) ----
    train = Estimator(
        image_uri=p_train_image,
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        sagemaker_session=sm_sess,
        output_path=Join(on="", values=["s3://", p_data_bucket, "/", p_prefix, "/model"]),
        enable_network_isolation=False,
    )
    train.set_hyperparameters(objective="binary:logistic", num_round=p_num_round, eval_metric="auc", verbosity=1)
    train_args = train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs[1].S3Output.S3Uri,
                content_type="text/csv"
            ),
        }
    )
    train_step = TrainingStep(name="Train", step_args=train_args, cache_config=cache)

    # ---- 5) Evaluate ----
    eval_proc = ScriptProcessor(
        image_uri=image_uris.retrieve(framework="sklearn", region=region, version="1.2-1"),
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        command=["python3"],
        sagemaker_session=sm_sess,
    )
    eval_report = PropertyFile(name="EvaluationReport", output_name="report", path="evaluation.json")
    eval_args = eval_proc.run(
        code="processing/evaluate.py",
        source_dir="pipelines",
        inputs=[
            ProcessingInput(source=train_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=preprocess_step.properties.ProcessingOutputConfig.Outputs[1].S3Output.S3Uri, destination="/opt/ml/processing/validation_pre"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="report",
                source="/opt/ml/processing/report",
                destination=Join(on="", values=["s3://", p_data_bucket, "/", p_prefix, "/evaluate/report"]),
            )
        ],
    )
    eval_step = ProcessingStep(name="Evaluate", step_args=eval_args, property_files=[eval_report], cache_config=cache)

    # ---- 6) Register (조건부) ----
    reg = RegisterModel(
        name="RegisterModel",
        estimator=train,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=p_mpg,
        approval_status="PendingManualApproval",
    )

    # ---- 7) Deploy (ScriptProcessor + boto3) ----
    model_name = Join(on="-", values=["model", ExecutionVariables.PIPELINE_EXECUTION_ID])
    epc_name   = Join(on="-", values=["epc",   ExecutionVariables.PIPELINE_EXECUTION_ID])

    deployer = ScriptProcessor(
        image_uri=image_uris.retrieve(framework="sklearn", region=region, version="1.2-1"),
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        command=["python3"],
        sagemaker_session=sm_sess,
    )
    deploy_args = deployer.run(
        code="processing/deploy.py",
        source_dir="pipelines",
        inputs=[],
        outputs=[],
        arguments=[
            "--model-name", model_name,
            "--endpoint-config-name", epc_name,
            "--endpoint-name", p_endpoint_name,
            "--image-uri", p_train_image,
            "--model-data", train_step.properties.ModelArtifacts.S3ModelArtifacts,
            "--instance-type", p_instance_type,
            "--initial-instance-count", "1",
            "--exec-role-arn", role,
        ],
    )
    deploy_step = ProcessingStep(name="Deploy", step_args=deploy_args, cache_config=cache)

    cond = ConditionStep(
        name="ModelQualityCheck",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step=eval_step, property_file=eval_report, json_path="metrics.auc.value"),
                right=p_auc_threshold,
            )
        ],
        if_steps=[reg, deploy_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name=os.environ.get("SM_PIPELINE_NAME", "my-mlops-dev-pipeline"),
        parameters=[
            p_data_bucket, p_prefix, p_instance_type, p_endpoint_name,
            p_train_image, p_external_csv, p_use_fs, p_fg_name,
            p_mpg, p_auc_threshold, p_num_round,
        ],
        steps=[extract_step, validate_step, preprocess_step, train_step, eval_step, cond],
        sagemaker_session=sm_sess,
    )
    return pipeline


def upsert_and_start(wait: bool = False):
    region = boto3.Session().region_name
    role = os.environ["SM_EXEC_ROLE_ARN"]
    pipe = get_pipeline(region, role)
    pipe.upsert(role_arn=role)

    ev = os.environ
    params = {}
    if ev.get("DATA_BUCKET"):              params["DataBucket"] = ev["DATA_BUCKET"]
    if ev.get("PREFIX"):                   params["Prefix"] = ev["PREFIX"]
    if ev.get("EXTERNAL_CSV_URI"):         params["ExternalCsvUri"] = ev["EXTERNAL_CSV_URI"]
    if ev.get("USE_FEATURE_STORE"):        params["UseFeatureStore"] = ev["USE_FEATURE_STORE"]
    if ev.get("FEATURE_GROUP_NAME"):       params["FeatureGroupName"] = ev["FEATURE_GROUP_NAME"]
    if ev.get("MODEL_PACKAGE_GROUP_NAME"): params["ModelPackageGroupName"] = ev["MODEL_PACKAGE_GROUP_NAME"]
    if ev.get("TRAIN_IMAGE_URI"):          params["TrainImage"] = ev["TRAIN_IMAGE_URI"]
    if ev.get("SM_INSTANCE_TYPE"):         params["InstanceType"] = ev["SM_INSTANCE_TYPE"]
    if ev.get("SM_ENDPOINT_NAME"):         params["EndpointName"] = ev["SM_ENDPOINT_NAME"]

    exe = pipe.start(parameters=params if params else None)
    print("Started pipeline:", exe.arn)
    print("Parameters passed:", params)

    if wait:
        sm = boto3.client("sagemaker")
        while True:
            desc = sm.describe_pipeline_execution(PipelineExecutionArn=exe.arn)
            status = desc.get("PipelineExecutionStatus")
            print("Pipeline status:", status)
            if status in {"Succeeded", "Failed", "Stopped"}:
                if status != "Succeeded":
                    raise SystemExit(f"Pipeline did not succeed: {status}")
                break
            time.sleep(15)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    if args.run:
        upsert_and_start(wait=args.wait)
    else:
        role = os.environ["SM_EXEC_ROLE_ARN"]
        p = get_pipeline(boto3.Session().region_name, role)
        print(p.definition())
