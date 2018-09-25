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

mnist.train.images[0]
first_image = mnist.test.images[0]
first_image = np.array(first_image, dtype='float')
print(first_image.shape)

# Save the picture
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
plt.imsave('seven.png',pixels, cmap='gray')

# Encode base64 the picture
import base64
input_image = open("seven.png", "rb").read()
# Encode image in b64
encoded_input_string = base64.b64encode(input_image)
input_string = encoded_input_string.decode("utf-8")
print("Base64 encoded string: " )
print(input_string)


# Test

feature_spec = {'x': tf.FixedLenFeature(shape=[1,28,28,1], dtype=tf.float32)}

serialized_tf_example = tf.placeholder(shape=None, dtype=tf.string, name='input_image_tensor')
received_tensors = {'images': serialized_tf_example}


def _img_string_to_tensor(image_string, image_size=(28, 28)):
    image_decoded = tf.image.decode_png(image_string, channels=1)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)


    # Resize to expected
    # image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)

    image_resized = tf.reshape(image_decoded_as_float, [1,28,28,1])

    return image_resized


# fn = lambda image: _img_string_to_tensor(serialized_tf_example)
feature_tensor = _img_string_to_tensor(serialized_tf_example)
feature_spec = {"x": feature_tensor}


serialized_tf_example = input_string
print(feature_spec["x"].shape)



# Reshape for input
# first_image = mnist.test.images[0]
# pixels = first_image.reshape((-1, 28, 28, 1))
# print(pixels.shape)



# from grpc.beta import implementations
# from tensorflow.contrib.util import make_tensor_proto
#
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2


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