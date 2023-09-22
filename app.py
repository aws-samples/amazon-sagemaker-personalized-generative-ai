#!/usr/bin/env python3
import os

import aws_cdk as cdk
# Set the Kaggle API credentials as environment variables
# os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
# os.environ["KAGGLE_KEY"] = "your_kaggle_api_key"
os.environ["KAGGLE_USERNAME"] = "voantonino"
os.environ["KAGGLE_KEY"] = "0da1e9d995fd7fcf18c3d1b83fa70554"
from genai_personalized.genai_personalized_stack import GenAIPersonalizedStack


app = cdk.App()
GenAIPersonalizedStack(app, "GenAIPersonalizedStackStack",
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.

    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.

    #env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),

    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
    )

app.synth()
