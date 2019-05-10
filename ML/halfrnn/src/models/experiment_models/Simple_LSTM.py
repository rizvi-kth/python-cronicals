#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
# pip install -q -U tensorflow==1.7.0
import tensorflow as tf
print("Tensorflow version:", tf.__version__)
from tensorflow import keras
# from keras.utils import plot_model
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import Dense
# from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt



# In[2]:


seq_length = 3
sample_count = 100
data = [[i+j for j in range(seq_length)] for i in range(sample_count)]
# print(data)
target = [[i+j+1 for j in range(seq_length)] for i in range(1,sample_count+1)]
# print(target)


# In[3]:


data = np.array(data, dtype=float)
print(data.shape)
target = np.array(target, dtype=float)
print(target.shape)

# Normalize the Input
data = data/150
target = target/150


# In[4]:


data = data.reshape(sample_count, 1, seq_length) 
print(data.shape)
target = target.reshape(sample_count, 1, seq_length)
print(target.shape)


# In[19]:


Input_Layer = keras.Input(shape=(1, seq_length))

# 1st LSTM layer with 3 (seq_length) nodes
LSTM_Layer_1 = keras.layers.LSTM(seq_length, return_sequences=True)(Input_Layer)
# 2nd LSTM layer with 3 (seq_length) nodes
LSTM_Layer_2 = keras.layers.LSTM(seq_length, return_sequences=True)(LSTM_Layer_1)
# 3rd LSTM layer with 3 (seq_length) nodes
LSTM_Layer_3 = keras.layers.LSTM(seq_length, return_sequences=True)(LSTM_Layer_2)
# 2nd LSTM layer with 3 (seq_length) nodes
LSTM_Layer_4 = keras.layers.LSTM(seq_length, return_sequences=True)(LSTM_Layer_3)
# 3rd LSTM layer with 3 (seq_length) nodes
LSTM_Layer_5 = keras.layers.LSTM(seq_length, return_sequences=True)(LSTM_Layer_4)


Output_Layer = keras.layers.Dense(seq_length, activation='linear')(LSTM_Layer_5)
model = keras.models.Model(inputs=Input_Layer, outputs=Output_Layer)
print(model.summary())

#keras.utils.plot_model(model, to_file='Simple_LSTM.png')


# In[20]:


model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
history = model.fit(data, target, epochs=1000, batch_size=100, validation_data=(data, target))


# In[21]:

prediction_normalized = model.predict(data)
prediction = prediction_normalized * 150
# print(prediction)

# plt.scatter(range(100), prediction.reshape(300)[0:100], c='r')
# plt.scatter(range(100), (target*150).reshape(300)[0:100], c='g')
# plt.show()

plt.plot(history.history['loss'])
plt.show()


