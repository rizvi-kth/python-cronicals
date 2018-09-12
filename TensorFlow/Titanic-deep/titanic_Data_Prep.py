
# coding: utf-8

# In[84]:


import pandas as pd
import numpy as np
import tensorflow as tf
import os
import math
#from sklearn.model_selection import train_test_split

tf.__version__
#import tensorflow as tf


# ### Reading data files

# In[ ]:


CSV_PATH = os.path.join("..","Data","Titanic_data", "titanic.csv")

# Specify an Index
df = pd.read_csv(CSV_PATH, usecols=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'])


# In[83]:


# Check the Dataframe
# print(df.columns)
# df.head(3)
df.info()


# ### NaN treatment
# #### Check: Which columns has one or more null value

# In[79]:


df.isnull().any()


# #### Check: Show all rows with NaN for Age, Cabin and Embarked

# In[75]:


# df.loc[df[['Age']].isnull().any(axis=1),:]
# df.loc[df[['Cabin']].isnull().any(axis=1),:]
df.loc[df[['Embarked']].isnull().any(axis=1),:]


# ### Remove the null values with mean values for AGE

# In[76]:


df.fillna({'Age':df['Age'].mean()}, inplace = True)


# ### Remove the null values with some values for Cabin, Embarked

# In[77]:


df.fillna({'Cabin':'XXX'}, inplace = True)


# In[78]:


df.fillna({'Embarked':'X'}, inplace = True)


# ### Check unique values

# In[80]:


df['Sex'].unique()


# In[81]:


# df['Cabin'].unique()
len(df['Cabin'].unique())


# In[82]:


df['Embarked'].unique()
# len(df['Embarked'].unique())


# ### Split Train and Test Dataframe

# In[93]:


train_df = df.sample(frac=0.8,random_state=200)
test_df = df.drop(train_df.index)


# In[99]:


print("DataFrame Train: {}".format(train_df.shape))
print("DataFrame Test : {}".format(test_df.shape))


# In[112]:


print("Train columns : {}".format([c for c in train_df.columns]))
print("Test columns :  {}".format([c for c in test_df.columns]))
# [print(c) for c in train.columns]


# In[117]:


def load_data(train_df, test_df, y_label='Survived'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    
    train = train_df
    train_x, train_y = train, train.pop(y_label)

    test = test_df
    test_x, test_y = test, test.pop(y_label)

    return (train_x, train_y), (test_x, test_y)

(train_x, train_y), (test_x, test_y) = load_data(train_df, test_df, 'Survived')


# ### Check the Train Test X and Y

# In[133]:


print("Train x shape : {}".format(train_x.shape))
print("Train y shape : {}".format(train_y.shape))
print("Test  x shape : {}".format(test_x.shape))
print("Test  y shape : {}".format(test_y.shape))




# ## Prepare Feature_column for Tensorflow

# ### Process categorical column Sex, Embarked with Categorical-vocabulary-column

# In[145]:


# Converting Sex to Categorical-vocabulary-column
sex = tf.feature_column.categorical_column_with_vocabulary_list(
                                key='Sex',
                                vocabulary_list=['male', 'female'])

# Embedding-dimensions should be 4th root of the number of categories
embedding_dimensions = math.ceil(len(['male', 'female'])**0.25)
# Converting Categorical-vocabulary-column to embedding-column
sex_embedding_column = tf.feature_column.embedding_column(
                            categorical_column=sex,
                            dimension=embedding_dimensions)


# In[146]:


# Converting Sex to Categorical-vocabulary-column
embarked = tf.feature_column.categorical_column_with_vocabulary_list(
                                key='Embarked',
                                vocabulary_list=['S', 'C', 'Q', 'X'])

# Embedding-dimensions should be 4th root of the number of categories
embedding_dimensions = math.ceil(len(['S', 'C', 'Q', 'X'])**0.25)
# Converting Categorical-vocabulary-column to embedding-column
embarked_embedding_column = tf.feature_column.embedding_column(
                                categorical_column=embarked,
                                dimension=embedding_dimensions)


# ### Process categorical column Cabin with Hashed-column

# In[147]:


# Converting Sex to Categorical-vocabulary-column
cabin = tf.feature_column.categorical_column_with_hash_bucket(
        key = "Cabin",
        hash_bucket_size = 500) # The number of categories or greater - len(df['Cabin'].unique())

# Embedding-dimensions should be 4th root of the number of categories
embedding_dimensions = math.ceil(500**0.25)
# Converting Categorical-vocabulary-column to embedding-column
cabin_embedding_column = tf.feature_column.embedding_column(
                                categorical_column=cabin,
                                dimension=embedding_dimensions)


# ### Process Numerical column
# #### 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'

# In[148]:


survived = tf.feature_column.numeric_column(key="Survived")
pclass = tf.feature_column.numeric_column(key="Pclass")
age = tf.feature_column.numeric_column(key="Age")
sibSp = tf.feature_column.numeric_column(key="SibSp")
parch = tf.feature_column.numeric_column(key="Parch")
fare = tf.feature_column.numeric_column(key="Fare")


# ### Prepare feature_column array according to train_x columns

# In[142]:


['- '+ key for key in train_x.keys()]
# train_x.keys


# In[149]:


my_feature_columns = []
my_feature_columns.append(pclass)
my_feature_columns.append(sex_embedding_column)
my_feature_columns.append(age)
my_feature_columns.append(sibSp)
my_feature_columns.append(parch)
my_feature_columns.append(fare)
my_feature_columns.append(cabin_embedding_column)
my_feature_columns.append(embarked_embedding_column)


def get_tf_feature_columns():
    return my_feature_columns