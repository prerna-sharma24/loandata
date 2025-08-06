import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import base64
import datetime

# ===== Optional: increase layout width for nicer card visuals =====
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# ===== Custom CSS: background image + bordered title card =====
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/255379/pexels-photo-255379.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    /* Title card with border and shadow */
    .title-card {
        border: 3px solid rgba(255,255,255,0.9);
        background: rgba(0,0,0,0.35);
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.4);
        color: #ffffff;
        display: inline-block;
        margin-bottom: 10px;
    }

    .title-card h1 {
        margin: 0;
        padding: 0;
        font-size: 30px;
    }

    /* Make sidebar inputs slightly translucent for style */
    .css-1d391kg { background-color: rgba(255,255,255,0.85); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== Bordered title (rendered as HTML so it has an outer border) =====
st.markdown(
    """
    <div class="title-card">
      <h1>🏦 Loan Prediction App</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model and scaler - ensure model.pkl and scaler.pkl exist in working dir
# If these are heavy or not present, wrap in try/except and warn user.
try:
    model = joblib.load('model.pkl')
except Exception as e:
    st.warning("Could not load model.pkl. Prediction will not work until model file is present.")
    model = None

try:
    scaler = joblib.load('scaler.pkl')
except Exception:
    scaler = None  # Not used in code below, but left for completeness

# Sidebar inputs
st.sidebar.header("📝 Applicant Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0.0, format="%.2f")
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0.0, format="%.2f")
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, format="%.2f")
loan_term = st.sidebar.selectbox("Loan Term (months)", [360, 180, 120, 60])
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Convert to DataFrame for prediction & for download
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
    # If your model expects scaled values, apply scaler here (if you have it)
    # if scaler is not None:
    #     numeric_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
    #     df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

input_processed = preprocess(input_data)

# Layout: left column for visualization & result, right column for data preview & downloads
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Applicant Financial Summary")

    # Let user choose whether to show the visualization (so it shows after data entry)
    if st.checkbox("Show visualization for current input"):
        # Create bar chart
        fig, ax = plt.subplots()
        bars = ax.bar(
            ['Applicant Income', 'Coapplicant Income', 'Loan Amount'],
            [applicant_income, coapplicant_income, loan_amount]
        )

        # Add value labels
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(1, yval*0.02),
                    f'{yval:.2f}', ha='center', va='bottom')

        ax.set_ylabel("Amount")
        ax.set_title("Income & Loan Overview")
        st.pyplot(fig)

        # Also show a quick KPI strip
        total_income = applicant_income + coapplicant_income
        cols = st.columns(3)
        cols[0].metric("Applicant Income", f"{applicant_income:.2f}")
        cols[1].metric("Coapplicant Income", f"{coapplicant_income:.2f}")
        cols[2].metric("Total Income", f"{total_income:.2f}")

with col2:
    st.subheader("🧾 Entered Data Preview")
    st.dataframe(input_data, use_container_width=True)

    # Button to download entered data as CSV
    csv_buffer = BytesIO()
    input_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    st.download_button(
        label="⬇️ Download entered data (CSV)",
        data=csv_buffer,
        file_name=f"loan_input_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Predict button and result
if st.button("🔍 Predict Loan Approval"):
    if model is None:
        st.error("Model is not loaded. Put model.pkl in the app directory.")
    else:
        # If your model expects a numpy array or scaled values, ensure format matches
        try:
            pred = model.predict(input_processed)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            pred = None

        if pred is not None:
            result_text = "✅ Loan will be Approved!" if pred == 'Y' or pred == 'Yes' or pred == 1 else "❌ Loan will be Rejected."
            if pred == 'Y' or pred == 'Yes' or pred == 1:
                st.success(result_text)
            else:
                st.error(result_text)

            # ----- PDF generation (reportlab) -----
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, 760, "Loan Prediction Report")
            c.setFont("Helvetica", 11)
            c.line(50, 755, 550, 755)

            y = 730
            c.drawString(50, y, f"Prediction Result: {result_text}")
            y -= 18
            c.drawString(50, y, f"Gender: {gender}")
            y -= 14
            c.drawString(50, y, f"Married: {married}")
            y -= 14
            c.drawString(50, y, f"Dependents: {dependents}")
            y -= 14
            c.drawString(50, y, f"Education: {education}")
            y -= 14
            c.drawString(50, y, f"Self Employed: {self_employed}")
            y -= 14
            c.drawString(50, y, f"Applicant Income: {applicant_income}")
            y -= 14
            c.drawString(50, y, f"Coapplicant Income: {coapplicant_income}")
            y -= 14
            c.drawString(50, y, f"Loan Amount: {loan_amount}")
            y -= 14
            c.drawString(50, y, f"Loan Term (months): {loan_term}")
            y -= 14
            c.drawString(50, y, f"Credit History: {credit_history}")
            y -= 14
            c.drawString(50, y, f"Property Area: {property_area}")
            y -= 28

            # Add timestamp and footer
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(50, y, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.save()
            buffer.seek(0)

            st.download_button(
                label="📄 Download Result as PDF",
                data=buffer,
                file_name=f"loan_prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

# Small note / instructions
st.markdown(
    """
    <div style="margin-top:10px; color: white; opacity:0.9">
    *Tip:* Use **Show visualization** to preview the financial chart before downloading the input data or generating the PDF.
    </div>
    """,
    unsafe_allow_html=True,
)
