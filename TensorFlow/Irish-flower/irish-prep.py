import pandas as pd
import tensorflow as tf
import numpy as np

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

#print(type(train_x))

print(train_x.shape)
print(train_y.shape)

assert train_x.shape[0] == train_y.shape[0], "Train and Label should have same nr of rows."

# Convert to numpy array
train_x_np = train_x.values
train_y_np = train_y.values

# Convert to tensorflow Dataset
train_X_tf_ds = tf.data.Dataset.from_tensor_slices(train_x_np)
train_Y_tf_ds = tf.data.Dataset.from_tensor_slices(train_y_np)

print(train_Y_tf_ds.take(5))
