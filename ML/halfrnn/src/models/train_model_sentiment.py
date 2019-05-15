import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
# tensorboard --logdir ./ --host=127.0.0.1
from time import time

print("Tensorflow version:", tf.__version__)

SEQUENCE_LENGTH = 200
VACAB_FEATURES = 100
VOCAB_SIZE = 60064 # 3575 # 3443


def train_lstm_rnn(train_x, train_y, test_x, test_y):
    # Input_Layer = keras.Input(shape=(1, SEQUENCE_LENGTH))
    # LSTM_Layer_1 = keras.layers.LSTM(SEQUENCE_LENGTH, return_sequences=True)(Input_Layer)

    deep_inputs = keras.layers.Input(shape=(SEQUENCE_LENGTH, ))
    embedding = keras.layers.Embedding(VOCAB_SIZE, VACAB_FEATURES, input_length=SEQUENCE_LENGTH)(deep_inputs)
    dropout = keras.layers.Dropout(0.2)(embedding)
    lstm_1 = keras.layers.LSTM(units=100, return_sequences=True)(dropout)  # batch_input_shape=[None, SEQUENCE_LENGTH, VACAB_FEATURES]
    lstm_2 = keras.layers.LSTM(units=100, return_sequences=False)(lstm_1)  # batch_input_shape=[None, SEQUENCE_LENGTH, VACAB_FEATURES],
    final_dense = keras.layers.Dense(2, activation='softmax')(lstm_2)
    deep_model = keras.Model(inputs=deep_inputs, outputs=final_dense)
    print(deep_model.summary())

    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
    deep_model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

    # history = deep_model.fit(train_x, train_y, batch_size=100,
    #                          epochs=50,
    #                          validation_data=(test_x, test_y))
    history = deep_model.fit(train_x, train_y, batch_size=100,
                             epochs=6,
                             validation_split=.2,
                             callbacks=[tensorboard])

    scores = deep_model.evaluate(test_x, test_y, verbose=0)
    print('Test accuracy:', scores[1])


def prepare_training_data(review_text_array):

    logger = logging.getLogger('__prepare_training_data__')
    # Shuffle the array
    logger.info('Shuffle the array')
    np.random.shuffle(review_text_array)

    # Prepare tokenizer
    logger.info('Prepare tokenizer')
    t = keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(review_text_array[0:, 0])
    text_seq = t.texts_to_sequences(review_text_array[0:, 0])
    vocab_size = len(t.word_index) + 1
    print('Vocabulary size : {}'.format(vocab_size))

    assert(len(t.word_index) + 1 <= VOCAB_SIZE)
    assert(max([len(iner_list) for iner_list in text_seq]) == SEQUENCE_LENGTH)

    # Padding Sequences
    input_seq_padded = keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=SEQUENCE_LENGTH)
    print('Input shape after padding: ', input_seq_padded.shape)
    print('Input type: ', type(input_seq_padded))
    print(input_seq_padded[0:2])

    # Get the maximum element from a Numpy array
    max_element = np.amax(input_seq_padded)
    print('Maximum element: ', max_element)

    # Input data type conversion
    input_seq_normalized = np.asarray(input_seq_padded, dtype=np.float)

    # Output data type conversion
    output = np.asarray(review_text_array[0:, 2], dtype=np.float)
    # As the scores are in 1-10 we need to transform to 0-9 for categorical
    # output -= 1
    output_y = keras.utils.to_categorical(output, num_classes=2)

    return input_seq_normalized, output_y


@click.command()
@click.argument('train_tokens_file', type=click.Path(exists=True))
@click.argument('test_tokens_file', type=click.Path())
def main(train_tokens_file, test_tokens_file):
    # train_tokens_file = 'data/processed/train/reviews_523_tokens_25.npy'  # Check SEQUENCE_LENGTH flag
    # test_tokens_file = 'data/processed/test/test_reviews_556_tokens_25.npy'

    logger = logging.getLogger(__name__)
    logger.info('Loading data for training')

    # Load the file
    logger.info('Load Train file {0!s}'.format(train_tokens_file))
    train_array = np.load(train_tokens_file)
    logger.info('Shape of Train {0!s}'.format(train_array.shape))
    train_x, train_y = prepare_training_data(train_array)

    logger.info('Load Test file {0!s}'.format(test_tokens_file))
    test_array = np.load(test_tokens_file)
    logger.info('Shape of Test {0!s}'.format(test_array.shape))
    test_x, test_y = prepare_training_data(test_array)

    print('Sample Train Input: ')
    print(train_x[0:2])
    print('Sample Train Output: ')
    print(train_y[0:2])

    print('Sample Test Input: ')
    print(test_x[0:2])
    print('Sample Test Output: ')
    print(test_y[0:2])

    train_lstm_rnn(train_x, train_y, test_x, test_y)


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
