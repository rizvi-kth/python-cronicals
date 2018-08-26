import numpy as np
import tensorflow as tf
#from digit_recog import multilayer_dnn

print("Tensorflow :", tf.__version__)
print("Numpy      :", np.__version__)

def multilayer_dnn(X):
    fc1 = tf.layers.dense(X, 256, activation=tf.nn.relu, name="Hidden-1")
    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name="Hidden-2")
    out = tf.layers.dense(fc2, 10, activation=None, name="Output-pre")
    return out, fc1, fc2


# Reset the default graph to avoide grap duplication in Tensorboard.
tf.reset_default_graph()


from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data in mnist_data/
mnist = input_data.read_data_sets("../NN/mnist_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784], name="Input")
y = tf.placeholder(tf.int32, shape=[None, 10], name="Output-org")

logits, fc1, fc2 = multilayer_dnn(X)

# ### Predict any input
# Apply the softmax to calculate the prediction as the *softmax activation* is
# not implemented in the *multilayer_dnn* function.
y_ = tf.nn.softmax(logits, name="Predict")
saver = tf.train.Saver()

with tf.name_scope("Check-Accuracy"):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./digit-recog-checkpoints/model.ckpt")
    print("Model restored.")

    # Predict some test data.
    print("Predicting on test data:")
    X_test_batch, y_test_batch = mnist.test.next_batch(50)
    my_images = {X: X_test_batch}
    y_pred = sess.run(y_, feed_dict=my_images)
    # print(y_eval)

    for index in range(len(y_pred)):
        print("Predicted: ", np.argmax(y_pred[index]), " Actual: ", np.argmax(y_test_batch[index]))

    # Test accuracy again
    acc_test_1 = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
    print("\nTest accuracy: ", acc_test_1, "\n")
