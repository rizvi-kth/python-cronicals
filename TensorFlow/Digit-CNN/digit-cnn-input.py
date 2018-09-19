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
first_image = mnist.test.images[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

