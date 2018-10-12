# To create a stack from yml file 
# docker stack deploy -c <file_name> <stack_name>
docker stack deploy -c digit.client.stack.yml stk-dgt

# To remove a stack 
# docker stack rm <stack_name>
docker stack rm stk-dgt

# To see all the services of a stack
#docker stack services <stack-name>
docker stack services stk-dgt
docker stack services --format "{{.ID}}: {{.Mode}} {{.Replicas}}"

# Brows the URL(http://digit:8501/v1/models/my_mnist:predict) from an nginx container
docker exec 5e apt-get update
docker exec 5e apt-get install -y curl wget
docker exec 5e curl -d '{"key1":"value1", "key2":"value2"}' -H "Content-Type: application/json" -X POST http://digit:8501/v1/models/my_mnist:predict
