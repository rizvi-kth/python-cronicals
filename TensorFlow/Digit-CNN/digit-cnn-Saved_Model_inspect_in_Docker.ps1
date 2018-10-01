########################################
#   Check the Tensorflow docker image  #
########################################


docker run -it --rm -p 8888:8888 tensorflow/tensorflow:1.10.1-devel-py3  python -c "import tensorflow as tf; print(tf.__version__)"


################################
#   Inspect your Saved-Model   #
################################

docker run -it --rm -p 8888:8888 
--mount type=bind,source=C:/Users/A547184/Documents/_etc/tensor-serving-test/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,target=/mymodels/half_plus_two
tensorflow/tensorflow:1.10.1-devel-py3 bash

find / -name "saved_model_cli"

cd /usr/local/bin/
saved_model_cli show --dir /mymodels/half_plus_two/00000123
saved_model_cli show --dir /mymodels/half_plus_two/00000123 --tag_set serve
saved_model_cli show --dir /mymodels/half_plus_two/00000123 --tag_set serve --signature_def serving_default


docker run -it --rm -p 8888:8888 
--mount type=bind,source=C:/Users/A547184/Git/Repos/python-cronicals/TensorFlow/Digit-CNN/saved_models/1537371118,target=/mymodels/my_mnist
tensorflow/tensorflow:1.10.1-devel-py3 bash


/usr/local/bin/saved_model_cli show --dir /mymodels/my_mnist --all
/usr/local/bin/saved_model_cli show --dir /mymodels/my_mnist 
/usr/local/bin/saved_model_cli show --dir /mymodels/my_mnist --tag_set serve 
/usr/local/bin/saved_model_cli show --dir /mymodels/my_mnist --tag_set serve --signature_def serving_default

###########################
C:\Users\A547184\Git\Repos\python-cronicals\TensorFlow\Digit-CNN\saved_models\1537867745

docker run -it --rm -p 8888:8888 
--mount type=bind,source=C:/Users/A547184/Git/Repos/python-cronicals/TensorFlow/Digit-CNN/saved_models/1537867745,target=/mymodels/my_mnist
tensorflow/tensorflow:1.10.1-devel-py3 bash

/usr/local/bin/saved_model_cli show --dir /mymodels/my_mnist --all

docker run -it --rm -p 8888:8888 --mount type=bind,source=C:/Users/A547184/Git/Repos/python-cronicals/TensorFlow/Digit-CNN/saved_models/1537884991,target=/mymodels/my_mnist tensorflow/tensorflow:1.10.1-devel-py3 /usr/local/bin/saved_model_cli show --dir /mymodels/my_mnist --all




###########################

1. Test string post from RESTapi.

{
  "instances": [
                  {"b64": "iVBORw"},
                  {"b64": "pT4rmN"},
                  {"b64": "w0KGg2"}
                 ]
}
