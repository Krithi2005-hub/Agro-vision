import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
st.title(" AI for Crop Yield Prediction")
st.write("Enter the details below to predict the crop yield:")

# Check if files exist
if not (os.path.exists("yield_model.pkl") and os.path.exists("label_encoder.pkl")):
    st.error(" Model or label encoder not found. Please run train_model.py first.")
    st.stop()

# Load model and encoder
model = joblib.load("yield_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Inputs
rainfall = st.number_input(" Rainfall (mm)", min_value=0.0, value=1000.0)
pesticide = st.number_input(" Pesticide Usage (tonnes)", min_value=0.0, value=2.0)
temperature = st.number_input(" Temperature (°C)", min_value=-10.0, value=25.0)
crop = st.selectbox(" Crop Type", label_encoder.classes_)

if st.button(" Predict Yield"):
    crop_encoded = label_encoder.transform([crop])[0]
    input_array = np.array([[rainfall, pesticide, temperature, crop_encoded]])
    prediction = model.predict(input_array)[0]
    st.success(f" Estimated Yield: {prediction:.2f} hg/ha")
