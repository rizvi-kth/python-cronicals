# First start venv and go to "Serverless/sam/lotus-spider" folder, then use the Makefile
# pipenv shell
# cd Serverless/sam/lotus-spider

clean:
	rm -R .aws-sam
	rm template-deploy.yaml

build:
	# Remember to set the ENV vars in the template-build.yaml file
	sam build --template-file template-build.yaml # --manifest ./hello_world/requirements.txt

local:
	sam local invoke SpiderFunction --no-event  # ./events/bucket_event.json

pack:
	sam package --template-file ./.aws-sam/build/template.yaml --output-template-file template-deploy.yaml --s3-bucket rz-ds-resources --profile rz-robot --region eu-west-1

deploy:
	# sam deploy --guided --profile rz-robot # (Run this first time to create the .toml file)
	sam deploy --template-file template-deploy.yaml

down:
	aws cloudformation delete-stack --stack-name SpyderLayerTest --profile rz-robot


# For the layer commands
# ======================

layer:
	cd ./layer-mypy
	mkdir -p lib-zip/python/lib/python3.7/site-packages
	pip3 install -t lib-zip/python/lib/python3.7/site-packages -r ./requirements.txt
	cd lib-zip
	zip -r python.zip ./python/

sam:
	cd ./sam
	sam init --runtime python3.7 --name sam-layer
	cd sam-layer

# Copy python.zip in lib_zip

lbuild:
	# Remember to set the ENV vars in the template-build.yaml file
	sam build --template-file template-layer-build.yaml # --manifest ./hello_world/requirements.txt

lpack:
	sam package --template-file ./.aws-sam/build/template.yaml --output-template-file template-deploy.yaml --s3-bucket rz-ds-resources --profile rz-robot --region eu-west-1

ldeploy:
	sam deploy --template-file template-deploy.yaml
