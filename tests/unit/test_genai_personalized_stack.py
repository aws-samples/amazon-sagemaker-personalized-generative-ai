import aws_cdk as core
import aws_cdk.assertions as assertions

from genai_personalized.genai_personalized_stack import GenaiPersonalizedStack

# example tests. To run these tests, uncomment this file along with the example
# resource in genai_personalized/genai_personalized_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = GenaiPersonalizedStack(app, "genai-personalized")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
