#!/usr/bin/env python
# coding: utf-8

# In[42]:


import sklearn
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import os
import pickle

st.header("Body Performance Class Prediction App")
st.text_input("Enter your Name: ", key="name")

data_path = ['data']
filepath = os.sep.join(data_path + ['bodyPerformance.csv'])
data = pd.read_csv(filepath)

filename = 'best_model.pkl'

#load label encoder
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model

best_randforest_model = pickle.load(open(filename, 'rb'))

if st.checkbox('Show Training Dataframe'):
    data

st.subheader("Please select your gender")
left_column, right_column = st.columns(2)
with left_column:
    inp_gender = st.radio(
        'Your Gender:',
        np.unique(data['gender']))

input_age = st.number_input('Please input Age', min_value=0, max_value=None)
input_height = st.slider('Height(cm)', 0.0, max(data["height_cm"]), 1.0)
input_weight = st.slider('Weight(kg)', 0.0, max(data["weight_kg"]), 1.0)
input_fat = st.slider('body fat %', 0.0, max(data["body fat_%"]), 1.0)
input_diastolic = st.slider('diastolic reading', 0.0, max(data["diastolic"]), 1.0)
input_systolic = st.slider('systolic reading', 0.0, max(data["systolic"]), 1.0)
input_grip = st.slider('Grip Force', 0.0, max(data["gripForce"]), 1.0)
input_sitbend = st.slider('Sit and Reach', 0.0, max(data["sit and bend forward_cm"]), 1.0)
input_situp = st.slider('Sit Ups', 0.0, max(data["sit-ups counts"]), 1.0)
input_broadjump = st.slider('Standing Broad Jump', 0.0, max(data["broad jump_cm"]), 1.0)

#data['gender'] = encoder.fit_transform(data['gender'])
encoder.classes_ = np.load('genders.npy',allow_pickle=True)

if st.button('Make Prediction'):
    input_gender = encoder.transform(np.expand_dims(inp_gender, -1))
    inputs = np.expand_dims(
        [input_age, int(input_gender), input_height, input_weight, input_fat, input_diastolic, input_systolic, input_grip, input_sitbend, input_situp, input_broadjump], 0)
    prediction = best_randforest_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    encoder.classes_ = np.load('classes.npy',allow_pickle=True)
    st.write("Your performance class is: {} ".format(encoder.inverse_transform(prediction)))

    st.write(f"Thank you {st.session_state.name}! I hope you can accept it!")
