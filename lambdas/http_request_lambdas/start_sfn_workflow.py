import boto3
import json
import uuid
import os
from custom_encoder import CustomEncoder
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodbTableName = os.environ['MODEL_METADATA_TABLE']
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(dynamodbTableName)

stepfunction_client = boto3.client('stepfunctions')

postMethod = 'POST'
finetunePath = '/finetune'

def handler(event, context):
    
    logger.info(event)
    s3_training_data = os.environ['S3_TRAINING_DATA']
    # httpMethod = event['httpMethod']
    # path = event['path']

    body = {
        "s3_training_data": s3_training_data,
        "UserID": str(uuid.uuid4()),
        "ModelID": str(uuid.uuid4())
    }

    httpMethod = 'POST'
    path = '/finetune'
    
    if httpMethod == postMethod and path == finetunePath:
        # response = saveModel(json.loads(event['body']))
        response = saveModel(body)
    else:
        response = buildResponse(404, 'Not Found')
    return response


# Request a new training job. It starts the Step Function workflow and updates the DynamoDB table
def saveModel(requestBody):
    step_function_arn = os.environ['STEP_FUNCTION_ARN']
    model_params = {
        "model_status" : "TRAINING_REQUESTED",
        "target_model_name_mme" : "",
        "training_job_name" : ""
    }
    requestBody.update(model_params)
    
    try:
        table.put_item(Item=requestBody)
        body = {
            'Operation': 'SAVE',
            'Message': 'SUCCESS',
            'Item': requestBody
        }
        
        input_dict = {
            'user_id': requestBody['UserID'],
            'model_id' : requestBody['ModelID'],
            's3_training_data': requestBody['s3_training_data']
        }
        answ = stepfunction_client.start_execution(
            stateMachineArn=step_function_arn,
            input=json.dumps(input_dict))
        
        return buildResponse(200, body)
    
    except:
        logger.exception('Error Handling')


def buildResponse(statusCode, body=None):
    response = {
        'statusCode': statusCode,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
    if body is not None:
        response['body'] = json.dumps(body, cls=CustomEncoder)
    return response
