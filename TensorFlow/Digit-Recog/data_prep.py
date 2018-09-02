from __future__ import absolute_import, division, print_function

import numpy as np
#import tensorflow as tf

#print("Tensorflow :", tf.__version__)
print("Numpy      :",np.__version__)

# Reset the default graph to avoide grap duplication in Tensorboard.
#tf.reset_default_graph()


from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data in mnist_data/
mnist = input_data.read_data_sets("../../Data/mnist_data/", one_hot=True)

# # ## Test input dimensions
X_batch, y_batch = mnist.test.next_batch(5)

print("Shape X: ", np.shape(X_batch))
print("Shape y: ", np.shape(y_batch))
