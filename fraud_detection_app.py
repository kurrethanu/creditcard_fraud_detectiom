import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below to check for fraud.")

# Input fields for 30 features
features = []
for i in range(30):
    feature = st.number_input(f"Feature {i+1}", value=0.0, step=0.1, format="%.6f")
    features.append(feature)

if st.button("Predict"): 
    input_data = np.array([features])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.error("⚠️ The transaction is predicted to be FRAUDULENT.")
    else:
        st.success("✅ The transaction is predicted to be LEGITIMATE.")