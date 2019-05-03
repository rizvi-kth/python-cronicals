import itertools
import os
import math
import numpy as np
import pandas as pd
# pip install -q -U tensorflow==1.7.0
import tensorflow as tf
print("Tensorflow version:", tf.__version__)

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

path_data = "../data/raw/winemag-data_first150k.csv"
vocab_size = 12000 # This is a hyperparameter, experiment with different values for your dataset

# Read data
raw_data_df = pd.read_csv(path_data)
cols = [c for c in raw_data_df.columns]
print("Columns: ", cols)
# raw_data_df['description'].head()

# Feature 1:
f_description = raw_data_df['description']
# Create a tokenizer to preprocess our text descriptions
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(f_description)  # only fit on description feature
# Sparse bag of words (bow) vocab_size vector
description_bow_train = tokenize.texts_to_matrix(f_description)


