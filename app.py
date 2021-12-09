import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# def load_model():
#     with open('model_pickle','rb') as f:
#         mp = pickle.load(f)
#     return mp

df = pd.read_csv('Real estate.csv')


x = df.drop(['Y house price of unit area', 'No'], axis=1)
y = df['Y house price of unit area']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=50)

reg = LinearRegression()
reg.fit(x_train, y_train)


st.write("""
## Real estate price prediction
Input the following values and I will give you the price of the house
""")


x2 = st.slider('House age', 1, 100, 2)
x3 = st.slider('Distance to the nearest MRT station	', 23.38, 6488.02, 0.01)
x4 = st.slider('Number of convenience stores', 1, 10, 2)
x5 = st.slider('Latitude', 24.93, 24.93, 0.01)
x6 = st.slider('Longitude', 121.47, 121.56, 0.01)

y_predict = reg.predict(x_test)
st.write(y_predict)
