# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import numpy as np


# ****** To run in Python Console run this - 1 **********
# import os
# # Confirm the current working directory
# os.getcwd()
# # Use '\\' while changing the directory
# os.chdir(".\\src\\data")

import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re


def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return " ".join(filtered_words)


FILE_COUNT = 20000  # Total Sample count 20000*2
PROCESSED_TOKEN_LENGTH_MAX = 200


def make_nparray_with_reviews(review_directory, sentiment):
    print('Reading directory ', review_directory)
    pos_arr = np.zeros(shape=[0, 3])
    for path, dirs, files in os.walk(review_directory):
        assert (len(files) >= 12500)
        print('Reading %d file contents ' % (len(files)))
        file_counter = 0
        for file in files:

            if file_counter >= FILE_COUNT:
                break

            label = file.split('.')[0].split('_')[1]
            fs = open(os.path.join(path, file), 'r', encoding="utf-8")
            content = fs.read()
            processed_content = preprocess(content)
            file_token_count = len(processed_content.split())
            if file_token_count <= PROCESSED_TOKEN_LENGTH_MAX:
                print('{} ({}): {}'.format(file_counter, file_token_count, processed_content))
                pos_arr = np.append(pos_arr, [[processed_content, label, sentiment]], axis=0)
                file_counter += 1

            fs.close()

    # print(posArr)
    return pos_arr


# def make_nparray_with_reviews(review_directory):
#     print('Reading directory ', review_directory)
#     pos_arr = np.zeros(shape=[0, 2])
#     for path, dirs, files in os.walk(review_directory):
#         assert(len(files) >= 1000)
#         print('Reading %d file contents ' % (len(files)))
#         for file in files:
#             label = file.split('.')[0].split('_')[1]
#             fs = open(os.path.join(path, file), 'r')
#             content = fs.read()
#             fs.close()
#             pos_arr = np.append(pos_arr, [[content, label]], axis=0)
#
#     # print(posArr)
#     return pos_arr


@click.command()
@click.argument('pos_filepath', type=click.Path(exists=True))
@click.argument('neg_filepath', type=click.Path())
def main(pos_filepath, neg_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # ****** To run in Python Console run this - 2 **********
    # train_pos_filepath = '../../notebooks/IMDB_Sample_DS/train/pos'
    # train_neg_filepath = '../../notebooks/IMDB_Sample_DS/train/neg'

    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_fmt = '%(relativeCreated)d - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    pos_array = make_nparray_with_reviews(pos_filepath, 1)
    logger.info('Found positive reviews shape {0!s}' .format(pos_array.shape))
    neg_array = make_nparray_with_reviews(neg_filepath, 0)
    logger.info('Found negative reviews shape {0!s}' .format(neg_array.shape))

    rev1_array = np.concatenate((pos_array, neg_array), axis=0)
    logger.info('Total reviews shape: {0!s}' .format(rev1_array.shape))

    file_name = 'reviews_{}_tokens_{}_sentiment'.format(rev1_array.shape[0], PROCESSED_TOKEN_LENGTH_MAX)
    np.save(file_name, rev1_array)
    logger.info('All reviews saved to : {}'.format(file_name))

    # logger.info('Load the file train_review_np_array.npy')
    # rev2_array = np.load('train_review_np_array.npy')
    # logger.info('Shape of the read array {0!s}'.format(rev2_array.shape))


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()





