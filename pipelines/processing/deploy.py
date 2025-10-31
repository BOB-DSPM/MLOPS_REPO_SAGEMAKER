# pipelines/processing/deploy.py
import argparse, json, boto3, botocore, time, os

p = argparse.ArgumentParser()
p.add_argument("--model-name", required=True)
p.add_argument("--endpoint-config-name", required=True)
p.add_argument("--endpoint-name", required=True)
p.add_argument("--image-uri", required=True)
p.add_argument("--model-data", required=True)
p.add_argument("--instance-type", required=True)
p.add_argument("--initial-instance-count", type=int, default=1)
p.add_argument("--exec-role-arn", required=True)
args = p.parse_args()

sm = boto3.client("sagemaker")

def ensure_model():
    try:
        sm.describe_model(ModelName=args.model_name)
        print("[deploy] model exists:", args.model_name)
        return
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ValidationException":
            raise
    sm.create_model(
        ModelName=args.model_name,
        PrimaryContainer={
            "Image": args.image_uri,
            "ModelDataUrl": args.model_data,
            "Mode": "SingleModel"
        },
        ExecutionRoleArn=args.exec_role_arn,
    )
    print("[deploy] model created:", args.model_name)

def ensure_endpoint_config():
    try:
        sm.describe_endpoint_config(EndpointConfigName=args.endpoint_config_name)
        print("[deploy] endpoint-config exists:", args.endpoint_config_name)
        return
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ValidationException":
            raise
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

def create_or_update_endpoint():
    try:
        sm.describe_endpoint(EndpointName=args.endpoint_name)
        print("[deploy] endpoint exists -> updating:", args.endpoint_name)
        sm.update_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=args.endpoint_config_name
        )
        return "update"
    except botocore.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ValidationException":
            raise
    print("[deploy] endpoint not found -> creating:", args.endpoint_name)
    sm.create_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=args.endpoint_config_name
    )
    return "create"

def wait_in_service():
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=args.endpoint_name)
    desc = sm.describe_endpoint(EndpointName=args.endpoint_name)
    print("[deploy] endpoint status:", desc["EndpointStatus"])

ensure_model()
ensure_endpoint_config()
op = create_or_update_endpoint()
print("[deploy] endpoint op:", op)
wait_in_service()
