import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the tuned model, selector, and preprocessor
model = joblib.load('rf_model_cbrt_tuned.joblib')
selector = joblib.load('feature_selector_cbrt.joblib')
preprocessor = joblib.load('preprocessor_cbrt.joblib')

# App title
st.title("Salary Predictor")

# Input widgets
work_year = st.slider("Work Year", 2020, 2025, 2025)
experience_level = st.selectbox("Seniority Level", ["EN", "MI", "SE", "EX"])
job_title = st.selectbox("Job Title", ["Data Scientist", "Machine Learning Engineer", "Data Analyst"])
company_location = st.selectbox("Country", ["US", "CA", "GB"])
company_size = st.selectbox("Company Size", ["S", "M", "L"])

# Prediction button
if st.button("Predict Salary"):
    input_data = [[work_year, experience_level, job_title, company_location, company_size]]
    input_df = pd.DataFrame(input_data, columns=['work_year', 'experience_level', 'job_title', 'company_location', 'company_size'])
    input_processed = preprocessor.transform(input_df)
    input_selected = selector.transform(input_processed)
    prediction_cbrt = model.predict(input_selected)[0]
    prediction = np.power(prediction_cbrt, 3)  # Reverse cube root
    st.success(f"Predicted Salary: ${prediction:.2f} USD (MAE: ~$45,366, R²: 0.2777)")

# Optional: Add a note
st.write("Note: This is a demo model with approximate accuracy. MAE reflects mean absolute error.")