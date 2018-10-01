#############################################################################
#	Start Tensorflow/Serving container as an WebAPI for prediction service	#
#############################################################################


docker run -p 8501:8501 -p 8500:8500 
--mount type=bind,source=C:/Users/A547184/Git/Repos/python-cronicals/TensorFlow/Digit-CNN/saved_models,target=/models/my_mnist
-e MODEL_NAME=my_mnist -t tensorflow/serving


# Saved-Model location:
# C:/Users/A547184/Git/Repos/python-cronicals/TensorFlow/Digit-CNN/saved_models


# POST http://host:port/<URI>:<VERB>
# URI: /v1/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]
# VERB: classify|regress|predict


# rRpc Server:
# http://localhost:8500/v1/models/my_digit_pred:predict


# Rest Server:
# http://localhost:8501/v1/models/my_mnist:predict

<#
Sample post content
{
  "instances": [
				{"b64": "iVBORw..."}
               ]
}
{
  "instances": [
				{"images":{"b64": "iVBORw..."}}
               ]
}
{
  "instances": [
				{"b64": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAABHNCSVQICAgIfAhkiAAAAd5JREFUSIntlU+rOVEcxh8/P4tbNoiyEKWUjSRq8i9lR1bjLcxqlvZ2Nt6AV8BLsLO2wGIWmvxNSKQQFiOc792pW3fcMbir+9TZnPM859PTzDnHAIDwi/r3m7A/4B9Ql/6rLeTzeQiCgMViAUVRUK1WsVwuMRwOnwIaoHIOx+MxPB7Pl7nD4YBut3t3w/l8jnK5jHa7/e26akNBEBAIBCDLMvx+P0KhEFKpFDiOw2w2g8vlunkvlwvW6zWcTicAYDqdqgJVG34ni8WCYDCITqeDSCRym1cUBf1+H7Isw2q1QhRFVCoV1X3oFYPnebperyRJElmt1nve52EOh4NWqxUREfE8f9f7kmMhiiLsdju22y16vd6P/qfaxWIxOp1OxBijZDL5o//phplMBiaTCY1GA81mU1NGd7uPjw/qdDqkKApFo1GtOf3AYrFIjDGq1+uP5PTBstksnc9n2u12xHHce4E2m41GoxExxqhWqz2afwxmNBqp1WoRY4wGgwF5vd73An0+HzHGiDFGuVxOz+fQbna73TSZTIgxRoVCgQwGw3uBpVLp1i4cDuv62TQD4/E47ff7p4Gab5pEIgGz2QwAGI1GOB6PWqNfpPoAq0mSJKTTaWw2G13Ahx7gV+gTRo0HGpCkFMQAAAAASUVORK5CYII="}
               ]
}

#>


#############################################
#	Create image from a running container	#
#############################################

docker ps -a
docker container commit -m "Digit-CNN Added" <Container-ID> digit-cnn:1.0


docker run -d --name serving_base tensorflow/serving
docker cp saved_models serving_base:/models/my_mnist
docker commit --change "ENV MODEL_NAME my_mnist" serving_base digit-cnn:1.0
docker kill serving_base
# This will leave you with a Docker image called "digit-cnn:1.0" that you can deploy and will load your model for serving on startup.

#####################################
#	Create image from a Dockerfile	#
#####################################

docker image build -t digit-cnn:2.0 .
docker run -it -p 8501:8501 -p 8500:8500 digit-cnn:3.0
