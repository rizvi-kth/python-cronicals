import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Sample size 28x28
# 55000 data samples
print("Train Samples: {}".format(train_data.shape[0]))
print("Train Outputs: {}".format(train_labels.shape[0]))

# Sample size 28x28
# 10000 test samples
print("Eval Samples: {}".format(eval_data.shape[0]))
print("Eval Outputs: {}".format(eval_labels.shape[0]))

# mnist.train.images[0]
# first_image = mnist.test.images[0]
# first_image = np.array(first_image, dtype='float')
# pixels = first_image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()

# Reshape for input
first_image = mnist.test.images[0]
pixels = first_image.reshape((-1, 28, 28, 1))
print(pixels.shape)


from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


# # channel = grpc.insecure_channel('%s:%d' % (host, port))
# channel = implementations.insecure_channel(host, port)
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
#
# # Read an image
# data = imread(image)
# data = data.astype(np.float32)
# print(data)
#
# start = time.time()
#
# # Call classification model to make prediction on the image
# request = predict_pb2.PredictRequest()
# request.model_spec.name = model
# request.model_spec.signature_name = signature_name
# request.inputs['image'].CopyFrom(make_tensor_proto(data, shape=[1, 28, 28, 1]))
#
# result = stub.Predict(request, 10.0)
#
# end = time.time()
# time_diff = end - start
#
# # Reference:
# # How to access nested values
# # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
# print(result)
# print('time elapased: {}'.format(time_diff))