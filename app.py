import streamlit as st
import joblib
import pickle
import numpy as np
import os

# Paths to model files
MODEL_PATH = '../backend/models/crop_recommendation_model.pkl'
ENCODER_PATH = '../backend/models/label_encoder.pkl'

# Function to safely load model
def _safe_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

# Load model
model = _safe_load_model(MODEL_PATH)

# Load LabelEncoder
with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# Streamlit app
st.title("🌱 Crop Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", 0, 140, 90)
P = st.number_input("Phosphorus (P)", 5, 145, 42)
K = st.number_input("Potassium (K)", 5, 205, 43)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 28.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 75.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

# Predict
if st.button("Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    pred_label = model.predict(input_data)
    crop = le.inverse_transform(pred_label)
    st.success(f"🌾 Recommended Crop: {crop[0]}")
