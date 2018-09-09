# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 23:06:55 2018

@author: Rizvi
"""

import pandas as pd
import os
from matplotlib import pyplot as plt
from pprint import pprint

#%matplotlib inline

# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# ### Reading data files 
#___________________________

# If you run F9 - you need to go to (cd) the appropriate folder in the python console 
# otherwise you will get 'File not found' exception.
CSV_PATH1 = os.path.join("..","DataIris", "iris_training.csv")
CSV_PATH2 = os.path.join("..","DataIris", "iris_test.csv")



# Read just 5 rows to see what's there
#df = pd.read_csv(CSV_PATH, nrows=5)
df_train = pd.read_csv(CSV_PATH1)
df_test  = pd.read_csv(CSV_PATH2)


# Specify an Index
#df = pd.read_csv(CSV_PATH, nrows=5, index_col='PassengerId')
# Check the columns
df_train.columns
df_test.columns

df_train.head()
df_test.head()

# Rename the columns
df_train.rename(columns={'120'            :'Feature1', 
                         '4'              :'Feature2',
                         'setosa'         :'Feature3',
                         'versicolor'     :'Feature4',
                         'virginica'      :'Result',
                         }, inplace=True)
# Rename the columns
df_test.rename(columns={ '30'             :'Feature1', 
                         '4'              :'Feature2',
                         'setosa'         :'Feature3',
                         'versicolor'     :'Feature4',
                         'virginica'      :'Result',
                         }, inplace=True)


plt.plot(df_train[['Feature1','Feature2','Feature3','Feature4']])
plt.plot(df_test[['Feature1','Feature2','Feature3','Feature4']])

#df[['Feature3']].plot.hist(bins=100)
#plt.show()
#plt.show()

#df[['Feature3']].plot.hist(bins=100)


