
# app.py

import streamlit as st
from breast_cancer_model import load_data, preprocess_data, train_model

# Load data
data = load_data("BreastCancer.csv")
X, y, scaler = preprocess_data(data)

# Train model
model = train_model(X, y)

# Streamlit app
st.title("Breast Cancer Prediction Web App")

# Sidebar for user input
st.sidebar.header("Enter Patient Data")
user_input = {}
for col in data.columns[:-1]:
    user_input[col] = st.sidebar.number_input(f"{col}", value=float(data[col].mean()))

# Convert input to DataFrame
import pandas as pd
input_df = pd.DataFrame([user_input])

# Predict button
if st.sidebar.button("Predict"):
    # Scale the input
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_input)[0]
    prediction_label = "Malignant" if prediction == 1 else "Benign"
    
    # Display result
    st.write("### Prediction Result")
    st.write(f"The model predicts the diagnosis as **{prediction_label}**.")

    # Show probabilities
    probabilities = model.predict_proba(scaled_input)[0]
    st.write("### Prediction Probabilities")
    st.write(f"Benign: {probabilities[0]:.2f}, Malignant: {probabilities[1]:.2f}")
