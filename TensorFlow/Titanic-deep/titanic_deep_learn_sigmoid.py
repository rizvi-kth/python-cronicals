from titanic_Data_Prep import get_tf_feature_columns
from titanic_Data_Prep import get_Clean_Train_Test_Data
import tensorflow as tf
import pandas as pd

# import sys
# for path in sys.path:
#     print(path)


def my_model_fn(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
    tf.summary.scalar('loss', loss)
    # tf.summary.histogram('loss-histogram', loss)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}
    # merging and saving them every 100 steps by default for Tensorboard
    tf.summary.scalar('accuracy', accuracy[1])
    # tf.summary.histogram('accuracy-histogram', accuracy[1])
    # Stack Two histogram together
    # h = tf.stack([loss, accuracy[1]],0)
    # tf.summary.histogram('loss-accuricy', h)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec( mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset



BATCH_SIZE = 10
TRAIN_STEPS = 10000
tf.logging.set_verbosity(tf.logging.INFO)

(train_x, train_y), (test_x, test_y) = get_Clean_Train_Test_Data()

# As we are using tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits) in place of
# tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)  so we need to fix the labels.
# See more on
# https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow
# https://www.kaggle.com/realshijjang/tensorflow-binary-classification-with-sigmoid/notebook
# https://www.tensorflow.org/api_docs/python/tf/losses/sigmoid_cross_entropy
# https://www.tensorflow.org/api_docs/python/tf/losses/sparse_softmax_cross_entropy
train_y = pd.DataFrame(train_y)
test_y = pd.DataFrame(test_y)



my_feature_columns = get_tf_feature_columns()
print("Feature columns: {}".format(len(my_feature_columns)))
for col in my_feature_columns:
    print(type(col))



# Build 2 hidden layer DNN with 10, 10 units respectively.
# Configure to log in a directory
# Configure to log every 50 epoch for Console Output. Make sure to INFO log by tf.logging.set_verbosity(tf.logging.INFO)
# Configure to log every 50 epoch for Tensorboard.
classifier = tf.estimator.Estimator(model_fn=my_model_fn,
                                    model_dir='./titanic_deep_learn_sigmoid/log1',
                                    config=tf.estimator.RunConfig(log_step_count_steps=50, save_summary_steps = 50),
                                    params={
                                        'feature_columns': my_feature_columns,
                                        # Two hidden layers of 10 nodes each.
                                        'hidden_units': [16, 16],
                                        # The model must choose between 3 classes.
                                        'n_classes': 1,
                                    })

# Train the Model.
classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE), steps = TRAIN_STEPS)
# Evaluate the model.
eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, BATCH_SIZE))

print('\nEvaluation accuracy for Test set : {accuracy:0.3f}\n'.format(**eval_result))
