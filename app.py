import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or pickle

# Load the trained model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Make sure 'model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()


# Title
st.title("Loan Prediction App")

# Sidebar inputs
st.sidebar.header("Applicant Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=3000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=66)
loan_term = st.sidebar.selectbox("Loan Term (months)", [360, 180, 120, 60])
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Convert to DataFrame for prediction
input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Preprocessing should match the training preprocessing
def preprocess(df):
    df = df.copy()
    # Categorical to Numerical
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int) # Ensure '3+' is handled correctly
    
    # It's crucial that the columns are in the same order as in the training data
    # Ensure all expected columns are present, fill with 0 or a median if appropriate, though sidebar prevents this.
    
    return df

input_processed = preprocess(input_data)

# --- THIS IS THE CRITICAL MISSING STEP ---
# Scale the numerical features
# IMPORTANT: Make sure you only scale the columns that were scaled during training
# For example, if you only scaled income and loan amount:
try:
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Dependents'] # Adjust this list
    input_processed[numerical_cols] = scaler.transform(input_processed[numerical_cols])
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.warning("Please ensure the columns being scaled match those from training.")
    st.stop()
# -----------------------------------------


# Predict button
if st.button("Predict Loan Approval"):
    try:
        prediction = model.predict(input_processed)[0]
        if prediction == 'Y':
            st.success("✅ Loan will be Approved!")
        else:
            st.error("❌ Loan will be Rejected.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
