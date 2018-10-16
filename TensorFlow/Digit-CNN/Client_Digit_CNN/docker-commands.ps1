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