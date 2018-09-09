# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
%matplotlib inline


data = {'score': [234,24,14,27,-74,46,73,-18,59,160]}
test_df = pd.DataFrame(data)
type(test_df)

# Example usage of from_records method
records = [("Espresso", "5$"), ("Flat White", "10$")]
df2 = pd.DataFrame.from_records(records)
df2 = pd.DataFrame.from_records(records, columns=["Coffee", "Price"])
df2["Price"]

# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# ### Reading data files 
#___________________________

# If you run F9 - you need to go to (cd) the appropriate folder in the python console 
# otherwise you will get 'File not found' exception.
CSV_PATH = os.path.join("..","DataTitanic", "titanic.csv")

# Read just 5 rows to see what's there
df = pd.read_csv(CSV_PATH, nrows=5)


# Specify an Index
df = pd.read_csv(CSV_PATH, nrows=5, index_col='PassengerId')
# Check the columns
df.columns
df

# Limit columns
df = pd.read_csv(CSV_PATH, nrows=5, index_col='PassengerId', usecols=['PassengerId','Survived', 'Pclass', 'Sex'])

# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# ### Some basic operations
#___________________________

CSV_PATH = os.path.join("TestData", "titanic.csv")
# Proper data loading
COLS_TO_USE = ['PassengerId','Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Cabin']
df = pd.read_csv(CSV_PATH, usecols=COLS_TO_USE, index_col='PassengerId')
df

# Save for later
df.to_pickle(os.path.join('TestData', 'titanic.pickle'))

# Check some data
df[['Pclass', 'Age']]


# Get unique 'Pclass'
pclass = df['Pclass']
u_pclass = pd.unique(pclass)
print("Nr of class {0} and they are {1}".format(len(u_pclass), u_pclass))

# Filtering a class
df['Pclass']
# Make a group count
class_group = df['Pclass'].value_counts()
class_group
# Count for second class ticket
class_group[2] 

# Other way
count_result = df['Pclass'] == 3
type(count_result)
count_result.value_counts()

# Sorting the values 
df['Fare'].sort_values().head()
# Index is based on 'PassengerId'
df.loc[264,'Fare']
# Index is based on Row-Id
df.iloc[264,4]

df.loc[264,['Fare', 'Cabin']]
df.iloc[264,:5]



# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ### Experiment 1:  Process data column wise
#____________________________________________

CSV_PATH = os.path.join("TestData", "titanic_poluted.csv")
# Proper data loading
COLS_TO_USE = ['PassengerId','Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Cabin']
df_init = pd.read_csv(CSV_PATH, usecols=COLS_TO_USE, index_col='PassengerId')

# Count non-NA cells for each column or row.
print("Initial count(non-NA cells) on Fare {0}".format(df_init['Fare'].count()))
# Convert the column to numeric. This conversion has done the following things
# 1. Converted non-parsable to NaN
# 2. Convertes the single column DataFrame to a DataSeries
ds_pf = pd.to_numeric(df_init['Fare'], errors = "coerce")

# It has become a DataSeries from DataFrame
type(ds_pf)

# Count non-NA cells for each column or row.
print("After perse count(non-NA cells) on Fare {0}".format(ds_pf.count()))

# Detect the NaN ; usuallly they are at the end or statring of the column.
ds_pf.sort_values().tail()

# Apply iloc in a series to check the position of NaN
ds_pf.iloc[260:270]
ds_pf.iloc[263]

# Plot the chart  
ds_pf.plot(kind='bar')

#Convert DataSeries to DataFrame
df_pf = pd.DataFrame(ds_pf)

# NaN is still in the column - So drop it before normalization calculation
df_dr = df_pf.dropna(thresh=1)
df_dr.loc[260:270, :]
df_dr.iloc[260:270,:]

df_dr.sort_values(by=['Fare']).tail()

# Normalizing the columnFare
df_normalized=(df_dr-df_dr.min())/(df_dr.max()-df_dr.min())
df_normalized_dr.loc[260:270, :]

# Plot the normalized values
df_normalized.plot(kind='bar')

# Make the half
'df_normalized_dr = df_dr // 2
'df_normalized_dr.loc[260:270, :]

df_final = df_pf.assign(nFare=df_normalized_dr)
df_final
df_final.iloc[260:270, :]
df_final.iloc[263,:]


