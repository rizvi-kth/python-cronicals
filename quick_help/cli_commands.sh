


# scrapy
# ======
scrapy startproject newsposts && cd newsposts
scrapy genspider example https://www.aftonbladet.se/senastenytt
scrapy crawl examplename
scrapy crawl examplename -o newsitems.json

scrapy shell https://www.aftonbladet.se/senastenytt
$(under scrapy shell)
response.css('title')
response.css('title::text').get()
response.css('h3::text')[0].get()
response.css('h3::text').getall()
response.css('.HLf1C p::text').getall()



# Docker
# ======

docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest

# Running Model Archiver Option 1
docker run --rm -p 8080:8080 -p 8081:8081 -v $(pwd)/model-store:/model-store  -v $(pwd)/examples:/examples torchserve:latest bash
model-server@98b1c682955c:~$ torch-model-archiver ...

# Running Model Archiver Option 2
docker run --rm -p 8080:8080 -p 8081:8081 -v $(pwd)/model-store:/model-store  -v $(pwd)/examples:/examples torchserve:latest torch-model-archiver --model-name densenet161 --version 1.0 --model-file /examples/image_classifier/densenet_161/model.py --serialized-file /examples/image_classifier/densenet161-8d451a50.pth --export-path /model-store --extra-files /examples/image_classifier/index_to_name.json --handler image_classifier


docker run --rm -it -p 8080:8080 -p 8081:8081 -v $(pwd)/model_store:/model-store pytorch/torchserve:latest

docker cp ./mnist.mar bbb3ca6da87f:/home/model-server/model-store


