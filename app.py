import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

# App title
st.title("Breast Cancer Mortality Risk Prediction")

st.write("Predict whether patient may die within 10 years.")

# Inputs
age = st.number_input("Age at Diagnosis")
tumor_size = st.number_input("Tumor Size")
tumor_stage = st.number_input("Tumor Stage")

# Prediction button
if st.button("Predict"):

    data = np.array([[age, tumor_size, tumor_stage]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("High Mortality Risk")
    else:
        st.success("Lower Mortality Risk")