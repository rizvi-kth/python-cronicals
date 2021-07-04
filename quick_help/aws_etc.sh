rz-robot
ACCESS_KEY_ID_LOTUS = ..
SECRET_KEY_ID_LOTUS = ..

[mnd-prod]
ACCESS_KEY_ID = ..
SECRET_KEY_ID = ..








# SAM CLI
# =======
pipenv install aws-sam-cli
sam init --help

# Create an App
cd lotus-etl/Serverless/sam
sam init --runtime python3.7 --name lotus-spider-layer

# Build and local test
# template-build.yaml --> build/template-build.yaml
cd lotus-etl/Serverless/sam/lotus-etl
sam build --template-file template-build.yaml --manifest ./extr_infr_load/requirements.txt
sam local invoke InfrFunction --no-event


# Package and deploy
# build/template.yaml --> template-deploy.yaml
sam package --template-file ./.aws-sam/build/template.yaml --output-template-file template-deploy.yaml --s3-bucket rz-ds-resources --profile rz-robot --region eu-west-1
sam deploy --guided --profile rz-robot
sam deploy --template-file template-deploy.yaml


# AWS CLI
# =======
# Remove a stack
aws cloudformation delete-stack --stack-name lotus-etl-lambda --profile rz-robot





