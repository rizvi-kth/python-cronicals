from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

print("Tensorflow :", tf.__version__)
print("Numpy      :", np.__version__)

# Reset the default graph to avoide grap duplication in Tensorboard.
tf.reset_default_graph()


from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data in mnist_data/
mnist = input_data.read_data_sets("../../Data/mnist_data/", one_hot=True)


# # ## Test input dimensions
# X_batch, y_batch = mnist.train.next_batch(100)
# print("Shape X: ", np.shape(X_batch))
# print("Shape y: ", np.shape(y_batch))


def multilayer_dnn(X):
    fc1 = tf.layers.dense(X, 256, activation=tf.nn.relu, name="Hidden-1")
    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name="Hidden-2")
    out = tf.layers.dense(fc2, 10, activation=None, name="Output-pre")
    return out, fc1, fc2


X = tf.placeholder(tf.float32, shape=[None, 784], name="Input")
y = tf.placeholder(tf.int32, shape=[None, 10], name="Output-org")

# ### The final output layer with softmax activation
#
# Do not apply the softmax activation to this layer.
# The *tf.nn.sparse_softmax_cross_entropy_with_logits* will apply the
# softmax activation as well as calculate the cross-entropy as our cost function

logits, fc1, fc2 = multilayer_dnn(X)

# Predict any input
y_ = tf.nn.softmax(logits, name="Predict")
saver = tf.train.Saver()

with tf.name_scope("Loss-Calc"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=y)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope("Optimize"):
    optimizer = tf.train.AdamOptimizer(0.001, name="Adam-Optimizer")
    training_op = optimizer.minimize(loss)


# ### Check correctness and accuracy of the prediction
#
# * Check whether the highest probability output in logits is equal to the y-label
# * Check the accuracy across all predictions (How many predictions did we get right?)

with tf.name_scope("Check-Accuracy"):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()


n_epochs = 5
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):

        num_iterations = mnist.train.num_examples // batch_size

        for iteration in range(num_iterations):

            X_batch, y_batch = mnist.train.next_batch(batch_size)

            _, loss_eval, fc1_eval, fc2_eval, logits_eval = \
                sess.run([training_op, loss, fc1, fc2, logits],
                         feed_dict={X: X_batch, y: y_batch})
            if iteration == num_iterations - 1:
                print("Layer            :  Mean             Standard Deviation")
                print("Fully connected 1: ", fc1_eval.mean(), fc1_eval.std())
                print("Fully connected 2: ", fc2_eval.mean(), fc2_eval.std())

        # Check accuracy on every epoch
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

        print("Epoch:", epoch + 1 , "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        saved_path = saver.save(sess, "./digit-recog-checkpoints/model.ckpt")
        print("Saved checkpoint to ", saved_path)
        print()

        #writer = tf.summary.FileWriter('./digit-recog/run1', sess.graph)
        #writer.close()

    print(" --- Training complete --- ")
    print("")

    # Destroy the training weights
    print("Destroy the trained weights.")
    init.run()

    # Predict some test data.
    print("Predicting on test data:")
    X_test_batch, y_test_batch = mnist.test.next_batch(50)
    my_images = { X: X_test_batch }
    y_pred = sess.run(y_, feed_dict= my_images)
    # print(y_eval)

    for index in range(len(y_pred)):
        print("Predicted: ", np.argmax(y_pred[index]), " Actual: ", np.argmax(y_test_batch[index]))

    # Test accuracy again
    acc_test_1 = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
    print("\nTest accuracy: ", acc_test_1, "\n")


# with tf.Session() as sess:
#     # Restore variables from disk.
#     saver.restore(sess, "/digit-recog-checkpoints/model.ckpt")
#     print("Model restored.")
#
#     # Predict some test data.
#     print("Predicting on test data:")
#     X_test_batch, y_test_batch = mnist.test.next_batch(50)
#     my_images = {X: X_test_batch}
#     y_pred = sess.run(y_, feed_dict=my_images)
#     # print(y_eval)
#
#     for index in range(len(y_pred)):
#         print("Predicted: ", np.argmax(y_pred[index]), " Actual: ", np.argmax(y_test_batch[index]))
#
#     # Test accuracy again
#     acc_test_1 = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
#     print("\nTest accuracy: ", acc_test_1, "\n")
