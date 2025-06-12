import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example input (replace with real data: 30 features including Time and Amount)
new_transaction = np.array([[0.0] * 30])  # Replace with actual values

# Preprocess and predict
scaled_input = scaler.transform(new_transaction)
prediction = model.predict(scaled_input)

print("Prediction:", "Fraud" if prediction[0] == 1 else "Not Fraud")