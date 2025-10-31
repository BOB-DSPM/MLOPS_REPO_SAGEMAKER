# pipelines/pipeline_def.py  (또는 리포 루트 pipeline_def.py)
import argparse
import os
import json
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
from sagemaker.workflow.functions import Join
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris
from sagemaker.workflow.pipeline_context import PipelineSession

# --------------------------
# inline scripts (S3 업로드 없이 in-container 실행)
# --------------------------
DEFAULT_EXTRACT = """\
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
"""

DEFAULT_VALIDATE = """\
import os, json, pandas as pd
tr = pd.read_csv('/opt/ml/processing/train/data.csv', header=None)
va = pd.read_csv('/opt/ml/processing/validation/data.csv', header=None)
os.makedirs('/opt/ml/processing/report', exist_ok=True)
with open('/opt/ml/processing/report/summary.json','w') as f:
    json.dump({'train_rows': len(tr), 'val_rows': len(va)}, f)
"""

DEFAULT_PREPROCESS = """\
import os, pandas as pd
os.makedirs('/opt/ml/processing/train_pre', exist_ok=True)
os.makedirs('/opt/ml/processing/validation_pre', exist_ok=True)
tr = pd.read_csv('/opt/ml/processing/train/data.csv', header=None)
va = pd.read_csv('/opt/ml/processing/validation/data.csv', header=None)
y_tr, X_tr = tr.iloc[:,0], (tr.iloc[:,1:]-tr.iloc[:,1:].mean())/tr.iloc[:,1:].std(ddof=0)
y_va, X_va = va.iloc[:,0], (va.iloc[:,1:]-tr.iloc[:,1:].mean())/tr.iloc[:,1:].std(ddof=0)
pd.concat([y_tr, X_tr], axis=1).to_csv('/opt/ml/processing/train_pre/data.csv', index=False, header=False)
pd.concat([y_va, X_va], axis=1).to_csv('/opt/ml/processing/validation_pre/data.csv', index=False, header=False)
"""

DEFAULT_EVALUATE = """\
import json, os, pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
va = pd.read_csv('/opt/ml/processing/validation_pre/data.csv', header=None)
y = va.iloc[:,0].values
rs = np.random.RandomState(42)
pred = rs.rand(len(y))
auc = float(roc_auc_score(y, pred))
os.makedirs('/opt/ml/processing/report', exist_ok=True)
with open('/opt/ml/processing/report/evaluation.json','w') as f:
    json.dump({'metrics': {'auc': {'value': auc}}}, f)
"""

# 배포 스크립트: boto3로 모델/엔드포인트 생성·업데이트
DEFAULT_DEPLOY = """\
import argparse, json, boto3, botocore, time

p = argparse.ArgumentParser()
p.add_argument("--model-name", required=True)
p.add_argument("--endpoint-config-name", required=True)
p.add_argument("--endpoint-name", required=True)
p.add_argument("--image-uri", required=True)
p.add_argument("--model-data", required=True)
p.add_argument("--instance-type", required=True)
p.add_argument("--initial-instance-count", type=int, default=1)
args = p.parse_args()

sm = boto3.client("sagemaker")

def ensure_model():
    try:
        sm.describe_model(ModelName=args.model-name)  # intentionally wrong to trigger except for non-existence
    except Exception:
        pass
    try:
        sm.create_model(
            ModelName=args.model_name,
            PrimaryContainer={
                "Image": args.image_uri,
                "ModelDataUrl": args.model_data,
                "Mode": "SingleModel"
            },
            ExecutionRoleArn=boto3.client("sts").get_caller_identity()["Arn"].replace(":user/", ":role/unknown")
        )
        print("[deploy] model created:", args.model_name)
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ValidationException" and "already exists" in e.response.get("Error", {}).get("Message",""):
            print("[deploy] model exists:", args.model_name)
        else:
            raise

def ensure_endpoint_config():
    try:
        sm.describe_endpoint_config(EndpointConfigName=args.endpoint_config_name)
        print("[deploy] endpoint-config exists:", args.endpoint_config_name)
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ValidationException":
            sm.create_endpoint_config(
                EndpointConfigName=args.endpoint_config_name,
                ProductionVariants=[{
                    "ModelName": args.model_name,
                    "VariantName": "AllTraffic",
                    "InitialInstanceCount": args.initial_instance_count,
                    "InstanceType": args.instance_type
                }]
            )
            print("[deploy] endpoint-config created:", args.endpoint_config_name)
        else:
            raise

def create_or_update_endpoint():
    try:
        sm.describe_endpoint(EndpointName=args.endpoint_name)
        print("[deploy] endpoint exists -> updating:", args.endpoint_name)
        sm.update_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=args.endpoint_config_name
        )
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ValidationException":
            print("[deploy] endpoint not found -> creating:", args.endpoint_name)
            sm.create_endpoint(
                EndpointName=args.endpoint_name,
                EndpointConfigName=args.endpoint_config_name
            )
        else:
            raise

def wait_in_service():
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=args.endpoint_name)
    desc = sm.describe_endpoint(EndpointName=args.endpoint_name)
    print("[deploy] endpoint status:", desc["EndpointStatus"])

ensure_model()
ensure_endpoint_config()
create_or_update_endpoint()
wait_in_service()
"""

def _inline_as_code(s: str) -> str:
    # 임시 파일에 저장하지 않고 ScriptProcessor의 code= 에 직접 전달할 수 없으므로
    # processing container의 /opt/ml/processing/input/code/code.py 로 전달되도록 함
    return s

# --------------------------
# pipeline definition
# --------------------------
def get_pipeline(region: str, role: str) -> Pipeline:
    sm_sess = PipelineSession()

    p_data_bucket   = ParameterString("DataBucket",        default_value=os.environ.get("DATA_BUCKET", ""))
    p_prefix        = ParameterString("Prefix",            default_value=os.environ.get("PREFIX", "pipelines/exp1"))
    p_instance_type = ParameterString("InstanceType",      default_value=os.environ.get("SM_INSTANCE_TYPE", "ml.m5.large"))
    p_endpoint_name = ParameterString("EndpointName",      default_value=os.environ.get("SM_ENDPOINT_NAME", "mlops-endpoint"))

    default_train_image = image_uris.retrieve(framework="xgboost", region=sm_sess.boto_region_name, version="1.7-1")
    p_train_image  = ParameterString("TrainImage",         default_value=os.environ.get("TRAIN_IMAGE_URI", default_train_image))
    p_external_csv = ParameterString("ExternalCsvUri",     default_value=os.environ.get("EXTERNAL_CSV_URI", ""))
    p_use_fs       = ParameterString("UseFeatureStore",    default_value=os.environ.get("USE_FEATURE_STORE", "false"))
    p_fg_name      = ParameterString("FeatureGroupName",   default_value=os.environ.get("FEATURE_GROUP_NAME", ""))
    p_mpg          = ParameterString("ModelPackageGroupName", default_value=os.environ.get("MODEL_PACKAGE_GROUP_NAME", "model-pkg"))

    p_auc_threshold= ParameterFloat("AucThreshold",        default_value=0.65)
    p_num_round    = ParameterInteger("NumRound",          default_value=50)

    cache = CacheConfig(enable_caching=False, expire_after="PT1H")

    # 1) Extract
    extract = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        sagemaker_session=sm_sess,
    )
    extract_args = extract.run(
        code=_inline_as_code(DEFAULT_EXTRACT),
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

    # 2) Validate
    validate = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        sagemaker_session=sm_sess,
    )
    validate_args = validate.run(
        code=_inline_as_code(DEFAULT_VALIDATE),
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
    validate_step = ProcessingStep(name="Validate", step_args=validate_args, cache_config=cache)

    # 3) Preprocess
    preprocess = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        sagemaker_session=sm_sess,
    )
    preprocess_args = preprocess.run(
        code=_inline_as_code(DEFAULT_PREPROCESS),
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

    # 4) Train (XGBoost)
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

    # 5) Evaluate
    eval_proc = ScriptProcessor(
        image_uri=image_uris.retrieve(framework="sklearn", region=sm_sess.boto_region_name, version="1.2-1"),
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        command=["python3"],
        sagemaker_session=sm_sess,
    )
    evaluation = PropertyFile(name="EvaluationReport", output_name="report", path="evaluation.json")
    eval_args = eval_proc.run(
        code=_inline_as_code(DEFAULT_EVALUATE),
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
    eval_step = ProcessingStep(name="Evaluate", step_args=eval_args, property_files=[evaluation], cache_config=cache)

    # 6) Register (조건부)
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

    # 7) Deploy (ScriptProcessor로 배포 — SDK 버전 독립)
    # 실행 ID를 사용해 유일한 자원 이름 생성
    model_name = Join(on="-", values=["model", ExecutionVariables.PIPELINE_EXECUTION_ID])
    epc_name   = Join(on="-", values=["epc",   ExecutionVariables.PIPELINE_EXECUTION_ID])

    deployer = ScriptProcessor(
        image_uri=image_uris.retrieve(framework="sklearn", region=sm_sess.boto_region_name, version="1.2-1"),
        role=role,
        instance_type=p_instance_type,
        instance_count=1,
        command=["python3"],
        sagemaker_session=sm_sess,
    )
    deploy_args = deployer.run(
        code=_inline_as_code(DEFAULT_DEPLOY),
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
        ],
    )
    deploy_step = ProcessingStep(name="Deploy", step_args=deploy_args, cache_config=cache)

    # 품질 기준 충족 시: 모델 등록 + 배포
    cond = ConditionStep(
        name="ModelQualityCheck",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step=eval_step, property_file=evaluation, json_path="metrics.auc.value"),
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


def upsert_and_start(wait: bool = False, register_only: bool = False):
    region = boto3.Session().region_name
    role = os.environ["SM_EXEC_ROLE_ARN"]
    pipe = get_pipeline(region, role)
    pipe.upsert(role_arn=role)

    if register_only:
        print("Pipeline upserted (no execution started).")
        return

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
            if status in {"Succeeded", "Failed", "Stopped"}:
                print("Pipeline finished with status:", status)
                if status != "Succeeded":
                    raise SystemExit(f"Pipeline did not succeed: {status}")
                break
            time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    if args.register:
        role = os.environ["SM_EXEC_ROLE_ARN"]
        p = get_pipeline(boto3.Session().region_name, role)
        p.upsert(role_arn=role)
        print("Pipeline upserted (register only).")
    elif args.run:
        upsert_and_start(wait=args.wait)
    else:
        role = os.environ["SM_EXEC_ROLE_ARN"]
        p = get_pipeline(boto3.Session().region_name, role)
        print(p.definition())
