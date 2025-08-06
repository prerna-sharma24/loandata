import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 🎨 Inject custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/premium-vector/loan-logo-design-icon-vector_999827-1718.jpg?w=2000"); /* You can replace this with any direct image link */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model and scaler
model = joblib.load('model.pkl')  
scaler = joblib.load('scaler.pkl')

# Title
st.title("🏦 Loan Prediction App")

# Sidebar inputs
st.sidebar.header("📝 Applicant Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
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

# Preprocessing function
def preprocess(df):
    df = df.copy()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)
    return df

input_processed = preprocess(input_data)

# Visualize user financial inputs
st.subheader("📊 Applicant Financial Summary")

# Create bar chart
fig, ax = plt.subplots()
bars = ax.bar(
    ['Applicant Income', 'Coapplicant Income', 'Loan Amount'],
    [applicant_income, coapplicant_income, loan_amount],
    color=['skyblue', 'orange', 'green']
)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{yval:.0f}', ha='center', va='bottom')

ax.set_ylabel("Amount")
ax.set_title("Income & Loan Overview")

st.pyplot(fig)

# Predict button
if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(input_processed)[0]
    result_text = "✅ Loan will be Approved!" if prediction == 'Y' else "❌ Loan will be Rejected."
    
    if prediction == 'Y':
        st.success(result_text)
    else:
        st.error(result_text)

    # ----- PDF generation -----
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "Loan Prediction Report")
    c.line(50, 745, 550, 745)
    
    c.drawString(50, 720, f"Prediction Result: {result_text}")
    c.drawString(50, 700, f"Gender: {gender}")
    c.drawString(50, 685, f"Married: {married}")
    c.drawString(50, 670, f"Dependents: {dependents}")
    c.drawString(50, 655, f"Education: {education}")
    c.drawString(50, 640, f"Self Employed: {self_employed}")
    c.drawString(50, 625, f"Applicant Income: {applicant_income}")
    c.drawString(50, 610, f"Coapplicant Income: {coapplicant_income}")
    c.drawString(50, 595, f"Loan Amount: {loan_amount}")
    c.drawString(50, 580, f"Loan Term: {loan_term}")
    c.drawString(50, 565, f"Credit History: {credit_history}")
    c.drawString(50, 550, f"Property Area: {property_area}")
    
    c.save()
    buffer.seek(0)

    st.download_button(
        label="📄 Download Result as PDF",
        data=buffer,
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )
