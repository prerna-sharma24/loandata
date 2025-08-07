# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import datetime

st.set_page_config(page_title="Loan Prediction App", layout="wide")

# ------------------ Styling & background ------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.vecteezy.com/vector-art/23221109-banking-and-finance-concept-digital-connect-system-financial-and-banking-technology-with-integrated-circles-glowing-line-icons-and-on-blue-background-vector-design");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    .panel {
        background: rgba(0,0,0,0.45);
        color: white;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
    }

    .panel-light {
        background: rgba(255,255,255,0.92);
        color: #111;
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }

    .title-card {
        border: 2px solid rgba(255,255,255,0.7);
        padding: 12px 18px;
        border-radius: 12px;
        display: inline-block;
        background: rgba(0,0,0,0.35);
        color: white;
        margin-bottom: 10px;
    }

    /* Ensure Streamlit dataframes readable on light panels */
    .stDataFrame table {
        color: #111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title-card"><h2 style="margin:0">🏦 Loan Prediction App</h2></div>', unsafe_allow_html=True)

# ------------------ Load model (optional) ------------------
try:
    model = joblib.load("model.pkl")
except Exception:
    model = None
    st.warning("Model (model.pkl) not found. Prediction will fail until model file is present.")

# ------------------ Sidebar inputs ------------------
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

# ------------------ Visualization then Data Preview ------------------
st.markdown("<div class='panel'><h3 style='margin:6px 0'>📊 Applicant Financial Summary</h3></div>", unsafe_allow_html=True)
show_vis = st.checkbox("Show visualization for current input", value=True)

if show_vis:
    # Bar chart
    fig, ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(
        ['Applicant Income', 'Coapplicant Income', 'Loan Amount'],
        [applicant_income, coapplicant_income, loan_amount]
    )
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(1, yval*0.02), f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("Amount")
    ax.set_title("Income & Loan Overview")
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
    st.pyplot(fig)

    # KPI row (white panels for readability)
    total_income = applicant_income + coapplicant_income
    k1, k2, k3 = st.columns(3)
    k1.markdown('<div class="panel-light"><h4 style="margin:4px">Applicant Income</h4><h3 style="margin:4px">{:.2f}</h3></div>'.format(applicant_income), unsafe_allow_html=True)
    k2.markdown('<div class="panel-light"><h4 style="margin:4px">Coapplicant Income</h4><h3 style="margin:4px">{:.2f}</h3></div>'.format(coapplicant_income), unsafe_allow_html=True)
    k3.markdown('<div class="panel-light"><h4 style="margin:4px">Total Income</h4><h3 style="margin:4px">{:.2f}</h3></div>'.format(total_income), unsafe_allow_html=True)

    # Entered Data Preview and CSV download (shown after the visual)
    st.empty()
    st.markdown(" ")
    st.markdown("<div class='panel-light'><h4 style='margin:6px 0'>🧾 Entered Data Preview</h4>", unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True)
    csv_buf = BytesIO()
    input_data.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    st.download_button(
        label="⬇️ Download entered data (CSV)",
        data=csv_buf,
        file_name=f"loan_input_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Compact preview when visualization is hidden
    st.markdown("<div class='panel-light'><h4 style='margin:6px 0'>🧾 Entered Data Preview</h4>", unsafe_allow_html=True)
    st.dataframe(input_data, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Prediction & PDF generation ------------------

predict_col, _ = st.columns([1,3])

with predict_col:
    if st.button("🔍 Predict Loan Approval"):
        if model is None:
            st.error("Model is not loaded. Please add model.pkl to the app folder.")
        else:
            try:
                pred = model.predict(input_processed)[0]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                pred = None

            if pred is not None:
                approved = (pred == 'Y' or pred == 'Yes' or pred == 1)
                if approved:
                    st.success("✅ Loan will be Approved!")
                else:
                    st.error("❌ Loan will be Rejected.")

                # ----- PDF generation using fpdf (IN-MEMORY) -----
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(0, 10, "Loan Prediction Report", ln=True, align="C")
                pdf.ln(4)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 8, f"Prediction Result: {'Approved' if approved else 'Rejected'}")
                pdf.ln(2)
                for col_name, val in input_data.iloc[0].items():
                    pdf.cell(0, 7, f"{col_name}: {val}", ln=True)
                pdf.ln(4)
                pdf.set_font("Arial", size=9)
                pdf.cell(0, 6, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

                # --- Correct in-memory output for FPDF ---
                pdf_bytes = pdf.output(dest='S').encode('latin1')  # returns PDF as bytes
                pdf_buf = BytesIO(pdf_bytes)

                st.download_button(
                    label="📄 Download Result as PDF",
                    data=pdf_buf,
                    file_name=f"loan_prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

# ------------------ Note/Tip ------------------
st.markdown(
    """
    <div style="margin-top:12px; color: rgba(255,255,255,0.9)">
    *Tip:* Toggle **Show visualization** to preview the chart. The entered data preview appears after the visual for a cleaner flow.
    </div>
    """,
    unsafe_allow_html=True,
)









