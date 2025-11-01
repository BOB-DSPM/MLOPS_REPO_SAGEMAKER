# MLOPS_REPO_SAGEMAKERhahahaasdasd<!-- README.md -->
# MLOps Repo (SageMaker Pipeline)

## 요구 환경변수 (CodeBuild/CodePipeline)
- `AWS_DEFAULT_REGION=ap-northeast-2`
- `SM_EXEC_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole>`
- `DATA_BUCKET=my-mlops-dev-data`
- `MODEL_PACKAGE_GROUP_NAME=my-mlops-dev-pkg`
- `SM_INSTANCE_TYPE=ml.m5.large`  # 처리/학습/배포 공통
- `TRAIN_IMAGE_URI=366743142698.dkr.ecr.ap-northeast-2.amazonaws.com/sagemaker-xgboost:1.7-1` (옵션, 미지정 시 SDK 기본)
- `PREFIX=pipelines/exp1`
- `SM_ENDPOINT_NAME=mlops-endpoint`
- (옵션) `EXTERNAL_CSV_URI=s3://.../train.csv`
- (옵션) `USE_FEATURE_STORE=true|false`
- (옵션) `FEATURE_GROUP_NAME=...`

## 파이프라인 단계
1) Extract → 2) Validate → 3) Preprocess → 4) Train → 5) Evaluate  asdfasdf
AUC ≥ 0.65면 6) Register + 7) Deploy(boto3) 수행

## 로컬 테스트
```bash
export SM_EXEC_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole>
python pipelines/pipeline_def.py --run --wait
asdasdasdasdas