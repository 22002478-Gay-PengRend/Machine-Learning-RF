#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import datasets
import sklearn.model_selection as ms
import sklearn.metrics as sklm
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import pickle

pd.set_option('display.max_rows', 500)
data_path = ['data']

filepath = os.sep.join(data_path + ['bodyPerformance.csv'])
df = pd.read_csv(filepath)

le = preprocessing.LabelEncoder()

#cols = ['gender', 'class']
#df[cols] = df[cols].apply(le.fit_transform)
df['gender'] = le.fit_transform(df['gender'])
np.save('genders.npy', le.classes_)

df['class'] = le.fit_transform(df['class'])
np.save('classes.npy', le.classes_)

df_cleaned = df.drop('class', axis=1)

y = df['class']

#save label encoding for class

X_train, X_test, y_train, y_test = train_test_split(df_cleaned, y, test_size=0.2, random_state=42)


#load the model
filename = 'best_model.pkl'
best_randforest_model = pickle.load(open(filename, 'rb'))

y_pred = best_randforest_model.predict(X_test.values)
print(best_randforest_model.predict(X_test.values))

le.classes_ = np.load('genders.npy',allow_pickle=True)
print(X_test.shape)

input_gender = le.transform(np.expand_dims("M",-1))
print(int(input_gender))

inputs = np.expand_dims([27,int(input_gender),172.3,75.3,21.5,80,130,54.1,18.1,60.0,217.9],0)
print(inputs.shape)

le.classes_ = np.load('classes.npy',allow_pickle=True)
prediction = best_randforest_model.predict(inputs)
print("final pred", np.squeeze(prediction,-1))
print(le.inverse_transform(prediction))


# In[ ]:




