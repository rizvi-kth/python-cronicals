# Request 
curl localhost:5343/feed -H "Content-Type: application/json" -d '{"url":"http://dmoztools.net", "appid":"madisonTest", "crawlid":"abc123"}' 

# Request status
curl localhost:5343/feed -H "Content-Type: application/json" -d '{"url":"http://dmoztools.net", "appid":"madisonTest", "crawlid":"abc123"}' 


# Exec to Kafka monitor + listen the kafka pipe
python kafkadump.py dump -t demo.crawled_firehose
