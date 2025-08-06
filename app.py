import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Optional: Load a preprocessor if used during training
# preprocessor = joblib.load("preprocessor.pkl")

st.title("🏦 Loan Eligibility Prediction App")

st.write("Enter applicant details below to check loan eligibility.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0, value=360)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Preprocess input
def preprocess_input():
    return np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        property_area
    ]], dtype=object)

# Predict button
if st.button("Predict Loan Status"):
    input_data = preprocess_input()

    # If you used a preprocessor in training, apply it
    # input_data = preprocessor.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == "Y":
        st.success("✅ Loan will likely be Approved!")
    else:
        st.error("❌ Loan will likely be Rejected.")
