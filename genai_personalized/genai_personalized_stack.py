import os
from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    RemovalPolicy,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_dynamodb as dynamodb,
    aws_apigateway as apigateway,
    cloudformation_include as cfn_inc,
)
from constructs import Construct
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile


class GenAIPersonalizedStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        genai_personalized_bucket = s3.Bucket(self, "genai_personalized")

        trust_policy = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW, actions=["sts:AssumeRole"], resources=["*"]
                )
            ]
        )

        execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            inline_policies={"TrustPolicy": trust_policy},
        )

        execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )
        execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
        )

        dataset_name = "vishesh1412/celebrity-face-image-dataset"
        with tempfile.TemporaryDirectory() as temp_dir:
            destination_path = temp_dir + "celebrity-face-image-dataset"
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)

            self.download_kaggle_dataset(dataset_name, destination_path)

            s3deploy.BucketDeployment(
                self,
                "DeployDataset",
                sources=[s3deploy.Source.asset(destination_path)],
                destination_bucket=genai_personalized_bucket,
                destination_key_prefix="training_images",
            )

        model_metadata_table = dynamodb.Table(
            self,
            "ModelMetadata",
            table_name="model-metadata",
            partition_key=dynamodb.Attribute(
                name="UserID", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="ModelID", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        launch_fine_tuning_job_lambda = lambda_.Function(
            self,
            "LaunchFineTuningJobLambda",
            runtime=lambda_.Runtime.PYTHON_3_8,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambdas/launch_fine_tune_job"),
            environment={
                "EXECUTION_ROLE_ARN": execution_role.role_arn,
                "S3_MODEL_STORE": genai_personalized_bucket.s3_url_for_object(
                    key="model-store"
                ),
                "MODEL_METADATA_TABLE": model_metadata_table.table_name,
                "JUMPSTART_SUBMIT_DIRECTORY": "s3://jumpstart-cache-prod-eu-west-1/source-directory-tarballs/stabilityai/transfer_learning/txt2img/prepack/v1.0.3/sourcedir.tar.gz",
                "JUMPSTART_S3URI": "s3://jumpstart-cache-prod-eu-west-1/stabilityai-training/train-model-txt2img-stabilityai-stable-diffusion-v2-1-base.tar.gz",
            },
        )

        policy_statement = iam.PolicyStatement(
            actions=[
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:InvokeEndpoint",
                "iam:PassRole",
                "dynamodb:*",
                "s3:*",
                "states:StartExecution",
            ],
            resources=[
                "*"
            ],
        )

        launch_fine_tuning_job_lambda.role.add_to_policy(policy_statement)
        launch_fine_tuning_job_step = sfn_tasks.LambdaInvoke(
            self, "LaunchFineTuningJob", lambda_function=launch_fine_tuning_job_lambda
        )

        wait_state = sfn.Wait(
            self, "WaitState", time=sfn.WaitTime.duration(Duration.minutes(15))
        )
        untar_model_artifact_lambda = lambda_.Function(
            self,
            "UntarModelArtifactLambda",
            runtime=lambda_.Runtime.PYTHON_3_8,
            handler="untar_file.handler",
            code=lambda_.Code.from_asset("lambdas/model_artifact_preparation"),
            memory_size=10240,
            timeout=Duration.minutes(15),
            environment={
                "S3_MODEL_STORE_BUCKET_NAME": genai_personalized_bucket.bucket_name,
                "MODEL_METADATA_TABLE": model_metadata_table.table_name,
            },
        )

        untar_model_artifact_lambda.role.add_to_policy(policy_statement)
        untar_model_artifact_step = sfn_tasks.LambdaInvoke(
            self, "UntarModelArtifact", lambda_function=untar_model_artifact_lambda
        )

        upload_model_artifact_lambda = lambda_.Function(
            self,
            "UploadModelArtifactLambda",
            runtime=lambda_.Runtime.PYTHON_3_8,
            handler="upload_model_code.handler",
            code=lambda_.Code.from_asset("lambdas/model_artifact_preparation"),
            environment={
                "S3_MODEL_STORE_BUCKET_NAME": genai_personalized_bucket.bucket_name,
                "MODEL_METADATA_TABLE": model_metadata_table.table_name,
            },
        )
        upload_model_artifact_lambda.role.add_to_policy(policy_statement)
        upload_model_artifact_step = sfn_tasks.LambdaInvoke(
            self, "UploadModelArtifact", lambda_function=upload_model_artifact_lambda
        )

        definition = (
            sfn.Chain.start(launch_fine_tuning_job_step)
            .next(wait_state)
            .next(untar_model_artifact_step)
            .next(upload_model_artifact_step)
            .next(sfn.Pass(self, "FinalState"))
        )

        state_machine = sfn.StateMachine(
            self,
            "GenAIStateMachine",
            definition=definition,
            timeout=Duration.minutes(30),
        )

        genai_personalized_bucket.grant_read(launch_fine_tuning_job_lambda)
        genai_personalized_bucket.grant_write(untar_model_artifact_lambda)

        start_sfn_workflow_lambda = lambda_.Function(
            self,
            "StartSFNWorkflow",
            runtime=lambda_.Runtime.PYTHON_3_8,
            handler="start_sfn_workflow.handler",
            code=lambda_.Code.from_asset("lambdas/http_request_lambdas"),
            environment={
                "S3_TRAINING_DATA": genai_personalized_bucket.s3_url_for_object(
                    key="training_images/Celebrity Faces Dataset/Angelina Jolie/"
                ),
                "STEP_FUNCTION_ARN": state_machine.state_machine_arn,
                "MODEL_METADATA_TABLE": model_metadata_table.table_name,
            },
        )
        start_sfn_workflow_lambda.role.add_to_policy(policy_statement)

        check_model_state_lambda = lambda_.Function(
            self,
            "CheckModelState",
            runtime=lambda_.Runtime.PYTHON_3_8,
            handler="check_model_state.handler",
            code=lambda_.Code.from_asset("lambdas/http_request_lambdas"),
            environment={"MODEL_METADATA_TABLE": model_metadata_table.table_name},
        )
        check_model_state_lambda.role.add_to_policy(policy_statement)

        # ModelExecutionRoleArn and S3PathToModels are specific to account
        import_params = {
            "SageMakerProjectName": "sd-fine-tuned-mme",
            "ModelExecutionRoleArn": execution_role.role_arn,
            "StageName": "dev",
            "EndpointInstanceCount": "1",
            "EndpointInstanceType": "ml.g5.xlarge",
            "S3PathToModels": genai_personalized_bucket.s3_url_for_object(
                key="model-code/fine-tuned-sd/code/"
            ),
            "ContainerImageInference": "763104351884.dkr.ecr.eu-west-1.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117",
        }

        cfn_inc.CfnInclude(
            self,
            "Template",
            template_file="genai_personalized/mme-template.yaml",
            parameters=import_params,
        )

        invoke_endpoint_for_inference_lambda = lambda_.Function(
            self,
            "InvokeEndpointForInference",
            runtime=lambda_.Runtime.PYTHON_3_8,
            handler="invoke_endpoint_for_inference.handler",
            code=lambda_.Code.from_asset("lambdas/http_request_lambdas"),
            environment={
                "MULTI_MODEL_ENDPOINT_NAME": "sd-fine-tuned-mme-dev",
            },
        )
        invoke_endpoint_for_inference_lambda.role.add_to_policy(policy_statement)

        api = apigateway.RestApi(
            self,
            "GenAIPersonalizedAPI",
            rest_api_name="genai-personalized-api",
        )

        catch_all_resource = api.root.add_resource("resource")
        catch_all_resource.add_proxy()

        finetune = api.root.add_resource("finetune")
        model = api.root.add_resource("model")
        inference = api.root.add_resource("inference")
        # ...

        lambda_integration1 = apigateway.LambdaIntegration(start_sfn_workflow_lambda)
        lambda_integration2 = apigateway.LambdaIntegration(check_model_state_lambda)
        lambda_integration3 = apigateway.LambdaIntegration(invoke_endpoint_for_inference_lambda)
        # ...

        finetune.add_method("POST", lambda_integration1)
        model.add_method("GET", lambda_integration2)
        inference.add_method("POST", lambda_integration3)
        # ...

        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)

    def download_kaggle_dataset(self, dataset_name, destination_path):
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_name, path=destination_path, unzip=True)
