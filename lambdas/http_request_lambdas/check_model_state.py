# API Gateway Connector for getting model state from DynamoDB, launching a new training job.

import boto3
import json
from custom_encoder import CustomEncoder
import logging
import os
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodbTableName = os.environ['MODEL_METADATA_TABLE']
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(dynamodbTableName)

stepfunction_client = boto3.client('stepfunctions')

getMethod = 'GET'
postMethod = 'POST'
patchMethod = 'PATCH'
deleteMethod = 'DELETE'
healthPath = '/health'
modelPath = '/model'
model = '/models'


def handler(event, context):
    logger.info(event)
    httpMethod = event['httpMethod']
    path = event['path']
    
    if httpMethod == getMethod and path == healthPath:
        response = buildResponse(200)
    elif httpMethod == getMethod and path == modelPath:
        response = getModel(event['queryStringParameters']
                            ['UserID'], event['queryStringParameters']['ModelID'])
    elif httpMethod == postMethod and path == modelPath:
        response = saveModel(json.loads(event['body']))
    # To modify a model, not implemented
    elif httpMethod == patchMethod and path == modelPath:
        requestBody = json.loads(event['body'])
        response = modifyModel(
            requestBody['UserID'], requestBody['updateKey'], requestBody['updateValue'])
    else:
        response = buildResponse(404, 'Not Found')
    return response

# Checks if the UserID and ModelID combination already exists, and its current state
def getModel(UserID, ModelID):
    try:
        response = table.get_item(
            Key={
                'UserID': UserID,
                'ModelID': ModelID
            }
        )
        if 'Item' in response:
            return buildResponse(200, response['Item'])
        else:
            return buildResponse(404, {'Message': 'UserID or ModelID not found'})
    except:
        logger.exception('Error Handling')

def saveModel(requestBody):
    
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
            's3_fine_tuning_images_path': requestBody['s3_fine_tuning_images_path']
        }
        answ = stepfunction_client.start_execution(
            stateMachineArn='arn:aws:states:us-east-1:055107841600:stateMachine:ab-ft-sd-workflow',
            input=json.dumps(input_dict))
        
        return buildResponse(200, body)
    
    except:
        logger.exception('Error Handling')

def modifyModel(UserID, updateKey, updateValue):
    try:
        response = table.update_item(
            Key={
                'UserID': UserID
            },
            UpdateExpression='set %s = :value' (updateKey),
            ExpressionAttributeValues={
                ':value': updateValue
            },
            ReturnValues='UPDATED_NEW'
        )
        body = {
            'Operation': 'UPDATE',
            'Message': 'SUCCESS',
            'UpdatedAttributes': response
        }
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
