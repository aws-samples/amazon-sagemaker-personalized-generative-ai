# To use accelerate, the model artifacts need to be untarred.
# This function untars the model artifacts, and copies them to the 'ab-model-store-32987' bucket
# It also updates the status in the DynamoDB table to 'PROCESSING_MODEL_ARTIFACTS'
import boto3
import botocore
import tarfile
import os

from io import BytesIO
s3_client = boto3.client('s3')

s3_model_store_bucket_name = os.environ['S3_MODEL_STORE_BUCKET_NAME']
dynamodbTableName = os.environ['MODEL_METADATA_TABLE']
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(dynamodbTableName)

def handler(event, context):

    # bucket = 'ab-model-store-32987'
    training_job_name = event['Payload']['training_job_name']
    user_id = event['Payload']['user_id']
    model_id = event['Payload']['model_id']
    key = 'model-store/' + training_job_name + '/output/model.tar.gz'
    # key = 'output-fine-tuned-sd/output/' + training_job_name + '/output/model.tar.gz'
    
    response = table.update_item(
        Key={
            'UserID': user_id,
            'ModelID': model_id
        },
        UpdateExpression='SET model_status = :input',
        ExpressionAttributeValues={
            ':input': 'PROCESSING_MODEL_ARTIFACTS'
        },
        ReturnValues='UPDATED_NEW'
    )

    input_tar_file = s3_client.get_object(Bucket = s3_model_store_bucket_name, Key = key)
    input_tar_content = input_tar_file['Body'].read()
    
    # destination_bucket = 'ab-model-unzipped-83681'
    unzipped_artifacts_s3_path = ('/model-artifact/' + training_job_name + '/')

    with tarfile.open(fileobj = BytesIO(input_tar_content)) as tar:
        for tar_resource in tar:
            if (tar_resource.isfile()):
                inner_file_bytes = tar.extractfile(tar_resource).read()
                s3_client.upload_fileobj(BytesIO(inner_file_bytes), Bucket = s3_model_store_bucket_name, Key = (unzipped_artifacts_s3_path+tar_resource.name))
                
    return{
        'unzipped_artifacts_s3_path': unzipped_artifacts_s3_path,
        'user_id':user_id,
        'model_id':model_id
    }