# This function is the first in the Step Function workflow and copies an existing 
# training job (training_job_name_to_copy) and just modifies the S3Uri of the training data.
# It also updates the DynamoDB table to set status to 'MODEL_TRAINING' and adds the training job name
import boto3, os, datetime, json


jumpstart_submit_directory = os.environ['JUMPSTART_SUBMIT_DIRECTORY']
jumpstart_s3uri = os.environ['JUMPSTART_S3URI']
execution_role_arn = os.environ['EXECUTION_ROLE_ARN']
s3_model_store = os.environ['S3_MODEL_STORE']
dynamodbTableName = os.environ['MODEL_METADATA_TABLE']
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(dynamodbTableName)
s3_client = boto3.client('s3')

def handler(event, context):

    user_id = event['user_id']
    model_id = event['model_id']
    s3_training_data = event['s3_training_data']
    
    sm = boto3.client('sagemaker')
    job = {
        'TrainingJobName': 'None',
        'TrainingJobArn': 'None',
        'HyperParameters': {
            'adam_beta1': '"0.9"',
            'adam_beta2': '"0.999"',
            'adam_epsilon': '"1e-08"',
            'adam_weight_decay': '"0.01"',
            'batch_size': '"1"',
            'center_crop': '"False"',
            'compute_fid': '"False"',
            'epochs': '"10"',
            'gradient_accumulation_steps': '"1"',
            'learning_rate': '"2e-06"',
            'lr_scheduler': '"constant"',
            'max_grad_norm': '"1.0"',
            'max_steps': '"200"',
            'num_class_images': '"100"',
            'prior_loss_weight': '"1.0"',
            'sagemaker_container_log_level': '20',
            'sagemaker_job_name': '"sd-fine-tune-training-job-model-txt2img-2023-04-20-12-44-39-910"',
            'sagemaker_program': '"transfer_learning.py"',
            'sagemaker_region': '"eu-west-1"',
            'sagemaker_submit_directory': f'"{jumpstart_submit_directory}"',
            'seed': '"0"',
            'with_prior_preservation': '"False"'
        },
        'AlgorithmSpecification': {
            'TrainingImage': '763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04',
            'TrainingInputMode': 'File',
            'MetricDefinitions': [
                {'Name': 'stabilityai-txt2img:train-loss', 'Regex': 'train_avg_loss=([0-9\\.]+)'},
                {'Name': 'fid_score', 'Regex': 'fid_score=([-+]?\\d\\.?\\d*)'}
            ],
            'EnableSageMakerMetricsTimeSeries': False
        },
        'RoleArn': execution_role_arn,
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': 'None',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'CompressionType': 'None',
                'RecordWrapperType': 'None'
            },
            {
                'ChannelName': 'model',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': jumpstart_s3uri,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'application/x-sagemaker-model',
                'CompressionType': 'None',
                'RecordWrapperType': 'None',
                'InputMode': 'File'
            }
        ],
        'OutputDataConfig': {
            'KmsKeyId': '',
            'S3OutputPath': s3_model_store
        },
        'ResourceConfig': {
            'InstanceType': 'ml.g5.2xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 360000
        },
        'EnableNetworkIsolation': False,
        'EnableInterContainerTrafficEncryption': False,
        'EnableManagedSpotTraining': False,
        'DebugHookConfig': {
            'S3OutputPath': s3_model_store,
            'CollectionConfigurations': []
        },
        'ProfilerConfig': {
            'S3OutputPath': s3_model_store,
            'ProfilingIntervalInMilliseconds': 500,
            'DisableProfiler': False
        }
    }

    training_job_prefix = 'sd-fine-tune-training-job-model-txt2img-'
    training_job_name = training_job_prefix+str(datetime.datetime.today()).replace(' ', '-').replace(':', '-').rsplit('.')[0]
    
    job['InputDataConfig'][0]['DataSource']['S3DataSource']['S3Uri'] = s3_training_data
    job['ResourceConfig']['InstanceType'] = 'ml.g5.2xlarge' 

    # Define the JSON data
    json_data = {
        "instance_prompt": "a photo of a zwx person",
        "class_prompt": "a photo of a person"
    }

    # Convert the JSON data to a string
    json_string = json.dumps(json_data)
    parts = s3_training_data.split('/')
    bucket_name = parts[2]
    object_key = '/'.join(parts[3:]) + "dataset_info.json"

    # Upload the JSON data to the S3 bucket
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=json_string
    )

    print(f"Uploaded {object_key} to {s3_training_data}")
    print("Starting training job %s" % training_job_name)

    if 'VpcConfig' in job:
        resp = sm.create_training_job(
            TrainingJobName=training_job_name, AlgorithmSpecification=job['AlgorithmSpecification'], RoleArn=job['RoleArn'],
            InputDataConfig=job['InputDataConfig'], OutputDataConfig=job['OutputDataConfig'],
            ResourceConfig=job['ResourceConfig'], StoppingCondition=job['StoppingCondition'],
            HyperParameters=job['HyperParameters'] if 'HyperParameters' in job else {},
            VpcConfig=job['VpcConfig'],
            Tags=job['Tags'] if 'Tags' in job else [])
    else:
        # Because VpcConfig cannot be empty like HyperParameters or Tags
        resp = sm.create_training_job(
            TrainingJobName=training_job_name, AlgorithmSpecification=job['AlgorithmSpecification'], RoleArn=job['RoleArn'],
            InputDataConfig=job['InputDataConfig'], OutputDataConfig=job['OutputDataConfig'],
            ResourceConfig=job['ResourceConfig'], StoppingCondition=job['StoppingCondition'],
            HyperParameters=job['HyperParameters'] if 'HyperParameters' in job else {},
            Tags=job['Tags'] if 'Tags' in job else [])
    

    print(resp)
    
    response = table.update_item(
        Key={
            'UserID': user_id,
            'ModelID': model_id
        },
        UpdateExpression='SET model_status = :input1, training_job_name = :input2',
        ExpressionAttributeValues={
            ':input1': 'MODEL_TRAINING',
            ':input2': training_job_name
        },
        ReturnValues='UPDATED_NEW'
    )
    print(response)
    
    return {
        'training_job_name': training_job_name,
        'user_id':user_id,
        'model_id':model_id
    }