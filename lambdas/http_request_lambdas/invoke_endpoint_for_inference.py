# Function that connects to API gateway to get an image from the MME
# The Endpoint's name is in an environment variable (multi_model_endpoint_name) 
# The Model's name is passed as an argument in the API call
# Lastly, some processing is done to return the decoded image

import os
import io
import boto3
import json
from io import BytesIO
import base64
from base64 import b64encode

ENDPOINT_NAME = os.environ['MULTI_MODEL_ENDPOINT_NAME'] #'sd-fine-tuned-mme'
runtime = boto3.client('sagemaker-runtime')
  
def handler(event, context):
    
    TARGET_MODEL = event['queryStringParameters']['model_name']

    payload = '{"prompt" : "a photo of a zwx person"}'
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       Body=payload,
                                       ContentType='application/json',
                                       TargetModel=TARGET_MODEL)

    
    result = response['Body'].read()
    stream = BytesIO(result)
    image = stream.read()
    
    return{
        #"isBase64Encoded": True,
        "statusCode": 200,
        'headers': {"Content-Type": "text/plain",
                    'Access-Control-Allow-Origin': '*'},
        'body': base64.b64encode(image).decode('utf-8')
    }