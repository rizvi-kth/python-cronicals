import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)
# TODO : Implement this Keras Load_data function
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train = x_train.reshape([-1, 784])
x_train = x_train.astype(float)
y_train = y_train.astype(np.int32)

x_test = x_test.reshape([-1, 784])
x_test = x_test.astype(float)
y_test = y_test.astype(np.int32)




# mnist = tf.contrib.learn.datasets.load_dataset("mnist")
# x_train = mnist.train.images # Returns np.array
# y_train = np.asarray(mnist.train.labels, dtype=np.int32)
# x_test = mnist.test.images # Returns np.array
# y_test = np.asarray(mnist.test.labels, dtype=np.int32)


# 55000 data samples
print("Train Samples: {}".format(x_train.shape))
print("Train Outputs: {}".format(y_train.shape))

# Sample size 28x28
# 10000 test samples
print("Eval Samples: {}".format(x_test.shape))
print("Eval Outputs: {}".format(y_test.shape))



def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # conv1 = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)

    # Pooling Layer #1
    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)

    # Pooling Layer #2
    # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer with Dropout
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    input_layer = features["x"]
    fc1 = tf.layers.dense(input_layer, 256, activation=tf.nn.relu, name="Hidden-1")
    fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, name="Hidden-2")
    dropout = tf.layers.dropout(inputs=fc2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name="classes_op"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions= predictions["classes"], name='acc_op')
    metrics = {'accuracy': accuracy}

    # merging and saving them every 100 steps by default for Tensorboard
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss', loss)

    internal_logging_hook = tf.train.LoggingTensorHook({"1_loss": loss, "2_accuracy": accuracy[1]}, every_n_iter=10 )

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks = [internal_logging_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Create the Estimator
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                          model_dir="./checkpoints/log1",
                                          config=tf.estimator.RunConfig(log_step_count_steps=50,
                                                                        save_summary_steps=50),
                                         )

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)
# Prepare input function
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},
                                                    y=y_train,
                                                    batch_size=100,
                                                    num_epochs=None,
                                                    shuffle=True)



# Train the model with external logging hook
# mnist_classifier.train(input_fn=train_input_fn, steps=100, hooks=[logging_hook])
mnist_classifier.train(input_fn=train_input_fn,
                       steps=5000)

# Evaluate the model and print results
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_test},
                                                   y=y_test,
                                                   num_epochs=1,
                                                   shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

