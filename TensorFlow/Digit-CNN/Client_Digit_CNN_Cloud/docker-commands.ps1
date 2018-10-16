# To create a stack from yml file 
# docker stack deploy -c <file_name> <stack_name>
docker stack deploy -c digit.client.stack.yml dgt


# To remove a stack 
# docker stack rm <stack_name>
docker stack rm dgt

# To see all the services of a stack
#docker stack services <stack-name>
docker stack services dgt
docker stack services --format "{{.ID}}: {{.Mode}} {{.Replicas}}"

# Brows the URL(http://digit:8501/v1/models/my_mnist:predict) from an nginx container
docker exec 5e apt-get update
docker exec 5e apt-get install -y curl wget
docker exec 5e curl -d '{"key1":"value1", "key2":"value2"}' -H "Content-Type: application/json" -X POST http://digit:8501/v1/models/my_mnist:predict
docker exec 5e curl -d '{"key1":"value1", "key2":"value2"}' -H "Content-Type: application/json" -X POST http://dgt_digit.1.pth44b1mrqk2tmkfp0ttmqvd8:8501/v1/models/my_mnist:predict
docker exec 5e curl -d '{"key1":"value1", "key2":"value2"}' -H "Content-Type: application/json" -X POST http://localhost:8501/v1/models/my_mnist:predict
docker exec 5e curl -d '{"key1":"value1", "key2":"value2"}' -H "Content-Type: application/json" -X POST http://digitcnnapi.westeurope.azurecontainer.io:8501/v1/models/my_mnist:predict
{
  "instances": [
				{"b64": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAABHNCSVQICAgIfAhkiAAAAd5JREFUSIntlU+rOVEcxh8/P4tbNoiyEKWUjSRq8i9lR1bjLcxqlvZ2Nt6AV8BLsLO2wGIWmvxNSKQQFiOc792pW3fcMbir+9TZnPM859PTzDnHAIDwi/r3m7A/4B9Ql/6rLeTzeQiCgMViAUVRUK1WsVwuMRwOnwIaoHIOx+MxPB7Pl7nD4YBut3t3w/l8jnK5jHa7/e26akNBEBAIBCDLMvx+P0KhEFKpFDiOw2w2g8vlunkvlwvW6zWcTicAYDqdqgJVG34ni8WCYDCITqeDSCRym1cUBf1+H7Isw2q1QhRFVCoV1X3oFYPnebperyRJElmt1nve52EOh4NWqxUREfE8f9f7kmMhiiLsdju22y16vd6P/qfaxWIxOp1OxBijZDL5o//phplMBiaTCY1GA81mU1NGd7uPjw/qdDqkKApFo1GtOf3AYrFIjDGq1+uP5PTBstksnc9n2u12xHHce4E2m41GoxExxqhWqz2afwxmNBqp1WoRY4wGgwF5vd73An0+HzHGiDFGuVxOz+fQbna73TSZTIgxRoVCgQwGw3uBpVLp1i4cDuv62TQD4/E47ff7p4Gab5pEIgGz2QwAGI1GOB6PWqNfpPoAq0mSJKTTaWw2G13Ahx7gV+gTRo0HGpCkFMQAAAAASUVORK5CYII="}
               ]
}
docker exec 217 curl -d '{"instances": [{"b64": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAABHNCSVQICAgIfAhkiAAAAd5JREFUSIntlU+rOVEcxh8/P4tbNoiyEKWUjSRq8i9lR1bjLcxqlvZ2Nt6AV8BLsLO2wGIWmvxNSKQQFiOc792pW3fcMbir+9TZnPM859PTzDnHAIDwi/r3m7A/4B9Ql/6rLeTzeQiCgMViAUVRUK1WsVwuMRwOnwIaoHIOx+MxPB7Pl7nD4YBut3t3w/l8jnK5jHa7/e26akNBEBAIBCDLMvx+P0KhEFKpFDiOw2w2g8vlunkvlwvW6zWcTicAYDqdqgJVG34ni8WCYDCITqeDSCRym1cUBf1+H7Isw2q1QhRFVCoV1X3oFYPnebperyRJElmt1nve52EOh4NWqxUREfE8f9f7kmMhiiLsdju22y16vd6P/qfaxWIxOp1OxBijZDL5o//phplMBiaTCY1GA81mU1NGd7uPjw/qdDqkKApFo1GtOf3AYrFIjDGq1+uP5PTBstksnc9n2u12xHHce4E2m41GoxExxqhWqz2afwxmNBqp1WoRY4wGgwF5vd73An0+HzHGiDFGuVxOz+fQbna73TSZTIgxRoVCgQwGw3uBpVLp1i4cDuv62TQD4/E47ff7p4Gab5pEIgGz2QwAGI1GOB6PWqNfpPoAq0mSJKTTaWw2G13Ahx7gV+gTRo0HGpCkFMQAAAAASUVORK5CYII="}]}' -H "Content-Type: application/json" -X POST http://digitcnnapi.westeurope.azurecontainer.io:8501/v1/models/my_mnist:predict

# Create an image from container
docker build -t voidrizvi/cnn-client:latest .
# If you forget to tag
docker tag 0e5574283393 voidrizvi/cnn-client:latest
docker run voidrizvi/cnn-client:latest
docker system prune --volumes


docker commit 475 voidrizvi/digit-cnn-client
docker push voidrizvi/digit-cnn-client
docker run voidrizvi/digit-cnn-client




# Resource Group: Digit-Cnn-App-Group

# Container Service Name: digit-swarm
# DNS Name Prefix: digitapp

# FQDN:         digitappagents.westeurope.cloudapp.azure.com
# Master:       digitappmgmt.westeurope.cloudapp.azure.com

# user: rizvi

# Create RSA key pair
ssh-keygen -t rsa

# Login using key
ssh -i az_dc_id_rsa rizvi@digitappmgmt.westeurope.cloudapp.azure.com
ssh -i ./key_2/az_dc_id1 rizvi@digitappmgmt.westeurope.cloudapp.azure.com

# Remove cached key from known_hosts file
ssh-keygen -f "/home/hasan/.ssh/known_hosts" -R digitappmgmt.westeurope.cloudapp.azure.com


scp -C -i ./key_2/az_dc_id1 -r ./digit.client.stack.yml rizvi@digitappmgmt.westeurope.cloudapp.azure.com:/home/rizvi/