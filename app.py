import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Breast Cancer Mortality Risk Prediction")

# Inputs
age = st.number_input("Age at Diagnosis")
tumor_size = st.number_input("Tumor Size")
lymph_nodes = st.number_input("Positive Lymph Nodes")

# Predict button
if st.button("Predict"):

    # Input array
    data = np.array([[age, tumor_size, lymph_nodes]])

    # Scale input
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    # Output
    if prediction[0] == 1:
        st.error("High Mortality Risk")
    else:
        st.success("Lower Mortality Risk")
