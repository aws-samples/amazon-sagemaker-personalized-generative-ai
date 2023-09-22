# This function creates the necessary files for Accelerate:
# 1. The serving.properties file 
# 2. The Python file with the inference code
#
# It packages them into a tar.gz, and uploads it to the 'ab-model-code-90823' bucket, which is the S3 prefix 
# where all the models artifacts (.tar.gz) in the Multi-Model endpoint are located
import tarfile
import os.path
import boto3

model_code = """import logging
import os
from time import time
import torch
from diffusers import DiffusionPipeline
import diffusers
from djl_python.inputs import Input
from djl_python.outputs import Output
from typing import Optional
from io import BytesIO
from PIL import Image


def get_torch_dtype_from_str(dtype: str):
    if dtype == "fp16":
        return torch.float16
    raise ValueError(
        f"Invalid data type: {dtype}. DeepSpeed currently only supports fp16 for stable diffusion"
    )

class StableDiffusionService(object):

    def __init__(self):
        self.pipeline = None
        self.initialized = False
        self.logger = logging.getLogger()
        self.model_id_or_path = None
        self.data_type = None
        self.device = None
        self.world_size = None
        self.max_tokens = None
        self.save_image_dir = None

    def initialize(self, properties: dict):
        # model_id can point to huggingface model_id or local directory.
        # If option.model_id points to a s3 bucket, we download it and set model_id to the download directory.
        # Otherwise we assume model artifacts are in the model_dir
        self.model_id_or_path = properties.get("model_id") or properties.get(
            "model_dir")
        self.data_type = get_torch_dtype_from_str(properties.get("dtype"))
        self.max_tokens = int(properties.get("max_tokens", "1024"))
        self.device = int(os.getenv("LOCAL_RANK", "0"))

        if os.path.exists(self.model_id_or_path):
            config_file = os.path.join(self.model_id_or_path,
                                       "model_index.json")
            if not os.path.exists(config_file):
                raise ValueError(
                    f"{self.model_id_or_path} does not contain a model_index.json."
                    f"This is required for loading stable diffusion models from local storage"
                )

        kwargs = {"torch_dtype": torch.float16, "revision": "fp16"}
        
        start = time()
        pipeline = DiffusionPipeline.from_pretrained(self.model_id_or_path,
                                                     device_map='auto',
                                                     low_cpu_mem_usage=True,
                                                     **kwargs
                                                    )
    
        duration = time()-start 
        self.logger.info(f'Loaded model in {duration} seconds')

        self.pipeline = pipeline
        self.initialized = True


    def inference(self, inputs: Input):
        try:
            content_type = inputs.get_property("Content-Type")
            if content_type == "application/json":
                request = inputs.get_as_json()
                prompt = request.pop("prompt")
                params = request.pop("parameters", {})
                start = time()
                result = self.pipeline(prompt, **params)
                duration = time() - start
                self.logger.info(f'Inference took {duration} seconds')
            elif content_type and content_type.startswith("text/"):
                prompt = inputs.get_as_string()
                result = self.pipeline(prompt)
            else:
                # in case an image and a prompt is sent in the input
                init_image = Image.open(BytesIO(
                    inputs.get_as_bytes())).convert("RGB")
                request = inputs.get_as_json("json")
                prompt = request.pop("prompt")
                params = request.pop("parameters", {})
                result = self.pipeline(prompt, image=init_image, **params)

            img = result.images[0]
            buf = BytesIO()
            img.save(buf, format="PNG")
            byte_img = buf.getvalue()
            outputs = Output().add(byte_img).add_property(
                "content-type", "image/png")

        except Exception as e:
            logging.exception("Inference failed")
            outputs = Output().error(str(e))
        return outputs


_service = StableDiffusionService()


def handle(inputs: Input) -> Optional[Output]:
    if not _service.initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    return _service.inference(inputs)"""

s3_client = boto3.client('s3')
s3_model_store_bucket_name = os.environ['S3_MODEL_STORE_BUCKET_NAME']
dynamodbTableName = os.environ['MODEL_METADATA_TABLE']
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(dynamodbTableName)
        
def handler(event,context):
    
    # model_artifact_path = 's3://ab-model-unzipped-83681/' + event['unzipped_artifacts_s3_path']
    model_artifact_path = event['Payload']['unzipped_artifacts_s3_path']
    
    serving_properties = """engine=Python
option.s3url={path}
option.dtype=fp16""".format(path=model_artifact_path)
    
    serving_template = """engine=Python
option.s3url={{ s3url }}
option.dtype=fp16"""
    
    model_file = open("/tmp/model.py", "w+")
    model_file.write(str(model_code))
    model_file.close()
    
    serving_properties_file = open("/tmp/serving.properties", "w+")
    serving_properties_file.write(str(serving_properties))
    serving_properties_file.close()
    
    serving_template_file = open("/tmp/serving.template", "w+")
    serving_template_file.write(str(serving_template))
    serving_template_file.close()
    
    model_file_name = '/tmp/model.py'
    serving_properties_file_name = '/tmp/serving.properties'
    serving_template_file_name = '/tmp/serving.template'
    
    with tarfile.open("/tmp/model.tar.gz", "w|gz") as tf:
        tf.add(f"{model_file_name}")
        tf.add(f"{serving_properties_file_name}")
        tf.add(f"{serving_template_file_name}")
        
    user_id = event['Payload']['user_id']
    model_id = event['Payload']['model_id']
    
    model_name = 'ft-sd-'+user_id+'-'+model_id+'.tar.gz'
    s3_client.upload_file('/tmp/model.tar.gz', s3_model_store_bucket_name, ('model-code/fine-tuned-sd/code/'+model_name))
    
    response = table.update_item(
        Key={
            'UserID': user_id,
            'ModelID': model_id
        },
        UpdateExpression='SET model_status = :input1, target_model_name_mme = :input2',
        ExpressionAttributeValues={
            ':input1': 'AVAILABLE',
            ':input2': model_name
        },
        ReturnValues='UPDATED_NEW'
    )
    
    return {
        'model_name': model_name,
        'user_id' : user_id,
        "model_id" : model_id
    }
