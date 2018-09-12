import argparse
import pandas as pd
import tensorflow as tf
#import numpy as np

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


(train_x, train_y), (test_x, test_y) = load_data()

print("Input Features : {}".format(train_x.keys()))
print("Input Shape : {}".format(train_x.shape))
print("Output Shape : {}".format(train_y.shape))

train_x.head(5)
train_y.head(5)

assert train_x.shape[0] == train_y.shape[0], "Train and Label should have same nr of rows."


# Convert to numpy array
# train_x_np = train_x.values
# train_y_np = train_y.values

# Convert to tensorflow Dataset
# train_X_tf_ds = tf.data.Dataset.from_tensor_slices(train_x_np)
# train_Y_tf_ds = tf.data.Dataset.from_tensor_slices(train_y_np)
# print(type(train_Y_tf_ds))


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


def my_model(features, labels, mode, params):
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
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('loss-histogram', loss)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}
    # merging and saving them every 100 steps by default for Tensorboard
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.histogram('accuracy-histogram', accuracy[1])
    # Stack Two histogram together
    h = tf.stack([loss, accuracy[1]],0)
    tf.summary.histogram('loss-accuricy', h)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec( mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# =====================================================================================================
# To run in TERMINAL Console use the following main declaration
# Set 'Run -> Edit Configurations -> <this_file> -> Parameters' to '--batch_size 10 --train_steps 1000'
# Uncomment the bellow code and INDENT the following code
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def main(argv):
    args = parser.parse_args(argv[1:])
    BATCH_SIZE = args.batch_size
    TRAIN_STEPS = args.train_steps
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # =====================================================================================================
    # To run in PYTHON Console
    # Uncomment the bellow code and UNINDENT the following code
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # tf.logging.set_verbosity(tf.logging.INFO)
    # BATCH_SIZE = 10
    # TRAIN_STEPS = 1000
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # Feature columns describe how to use the input.
    # It is a list of <class 'tensorflow.python.feature_column.feature_column._NumericColumn'>
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    # Configure to log in a directory
    # Configure to log every 50 epoch for Console Output. Make sure to INFO log by tf.logging.set_verbosity(tf.logging.INFO)
    # Configure to log every 50 epoch for Tensorboard.
    classifier = tf.estimator.Estimator(model_fn=my_model,
                                        model_dir='./iris-nn-impl/log1',
                                        config=tf.estimator.RunConfig(log_step_count_steps=50, save_summary_steps = 50),
                                        params={
                                            'feature_columns': my_feature_columns,
                                            # Two hidden layers of 10 nodes each.
                                            'hidden_units': [10, 10],
                                            # The model must choose between 3 classes.
                                            'n_classes': 3,
                                        })

    # Train the Model.
    classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, BATCH_SIZE), steps = TRAIN_STEPS)
    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, BATCH_SIZE))

    print('\nEvaluation accuracy for Test set : {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size= BATCH_SIZE))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SPECIES[class_id], 100 * probability, expec))




# =====================================================================================================
# To run in TERMINAL Console use the following main declaration
# Set 'Run -> Edit Configurations -> <this_file> -> Parameters' to '--batch_size 10 --train_steps 1000'
# Uncheck 'Run with python console'
# Uncomment the bellow code
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
#
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++