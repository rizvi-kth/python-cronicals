# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer with Dropout
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs={
                                              'predict': tf.estimator.export.PredictOutput(predictions)
                                            }
                                          )


    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions= predictions["classes"], name='acc_op')
    metrics = {'accuracy': accuracy}
    # merging and saving them every 100 steps by default for Tensorboard
    tf.summary.scalar('accuracy', accuracy[1])


    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss', loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



# def main(unused_argv):

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
# train_data = mnist.train.images # Returns np.array
# train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# 55000 data samples
# print("Train Samples: {}".format(train_data.shape[0]))
# print("Train Outputs: {}".format(train_labels.shape[0]))

# Sample size 28x28
# 10000 test samples
print("Eval Samples: {}".format(eval_data.shape[0]))
print("Eval Outputs: {}".format(eval_labels.shape[0]))

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                          model_dir="./digit-cnn/log2",
                                          config=tf.estimator.RunConfig(log_step_count_steps=50,
                                                                        save_summary_steps=50),
                                          )

# # Train the model
# train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
#                                                     y=train_labels,
#                                                     batch_size=100,
#                                                     num_epochs=None,
#                                                     shuffle=True)
# mnist_classifier.train(input_fn=train_input_fn,steps=5000,hooks=[logging_hook])
#
# # Evaluate the model and print results
# eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
#                                                    y=eval_labels,
#                                                    num_epochs=1,
#                                                    shuffle=False)
#
# eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
# print(eval_results)

# # Predict the model and print results
pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, num_epochs=1, shuffle=False)
predict_results = mnist_classifier.predict(input_fn=pred_input_fn)

# #Check all the afiled predictions
# i=0
# for item, org in zip(predict_results,eval_labels):
#     if int(org) != int(item['classes']):
#         print("************ Mismatch at Index: ", i)
#         print("Original   : ", org)
#         print("Prediction : ", item['classes'])
#         print(item['probabilities'])
#     i += 1

def serving_input_receiver_fn():
    # inputs = {
    #     "x": tf.placeholder(tf.float32, shape=[None, 28, 28, 1]),
    # }
    # return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    # receiver_tensors = {
    #     'my_image': tf.placeholder(tf.string, name='tf_image_input')
    # }
    # print(receiver_tensors)
    # inputs = {
    #     "x": tf.placeholder(tf.float32, shape=[None, 28, 28, 1]),
    # }
    # return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=inputs)

    feature_spec = { 'x': tf.FixedLenFeature(shape=[1,28,28,1], dtype=tf.float32)}

    serialized_tf_example = tf.placeholder(shape=[], dtype=tf.string, name='input_image_tensor')
    received_tensors = {'images': serialized_tf_example}

    # features = tf.parse_example(serialized_tf_example, feature_spec)
    # input_img_size = (28,28)
    # fn = lambda image: _img_string_to_tensor(image, input_img_size)
    # features['x'] = tf.map_fn(fn, features['x'], dtype=tf.float32)

    # Cast 1D tensor to a scaler
    scaler_image = tf.reshape(serialized_tf_example, [])
    feature_tensor = _img_string_to_tensor(scaler_image  , image_size=(28, 28))
    feature_spec = {"x": feature_tensor}

    return tf.estimator.export.ServingInputReceiver(feature_spec, received_tensors)


def _img_string_to_tensor(image_string, image_size=(28, 28)):
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

    # Resize to expected
    image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)
    # image_resized = tf.reshape(image_decoded_as_float, [1,28,28,1])
    image_resized.set_shape((28,28,1))

    image_resized = tf.stack(image_resized)

    return image_resized



# Save the model
mnist_classifier.export_savedmodel("./saved_models", serving_input_receiver_fn=serving_input_receiver_fn)


# if __name__ == "__main__":
#     tf.app.run(main)