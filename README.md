
# Welcome to your CDK Python project!

## setup requirements

* Node 18+
* Install CDK with npm `npm install -g aws-cdk`
* Install Poetry: `https://python-poetry.org/docs/#installation`

Poetry install Linux, macOS, Windows (WSL)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install dependencies with poetry

```
poetry install
```

Setup python env in shell

```
poetry shell
```

At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

To add additional dependencies, for example other CDK libraries, just use `poetry add yourpackage`

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!
