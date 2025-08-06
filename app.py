import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ===== Load Data and Models =====
df = pd.read_csv("loan_data.csv")

# Load your trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ===== Dashboard Title =====
st.title("📊 Loan Approval Dashboard")

# ===== Top Summary Metrics =====
total_loans = len(df)
approved_loans = df[df["Loan_Status"] == "Y"].shape[0]
rejected_loans = df[df["Loan_Status"] == "N"].shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Loans", total_loans)
col2.metric("Approved Loans", approved_loans)
col3.metric("Rejected Loans", rejected_loans)

st.markdown("---")

# ===== Pie Chart =====
st.subheader("Loan Status Distribution")
status_counts = df["Loan_Status"].value_counts()

fig, ax = plt.subplots()
ax.pie(
    status_counts,
    labels=status_counts.index.map({"Y": "Approved", "N": "Rejected"}),
    autopct="%1.1f%%",
    colors=["#4CAF50", "#F44336"],
    startangle=90,
)
ax.axis("equal")
st.pyplot(fig)

# ===== Display Sample Data Table =====
st.subheader("📋 Sample Loan Data")
st.dataframe(df.head(10))

# ===== Filter by Property Area =====
st.subheader("🔍 Filter by Property Area")
area = st.selectbox("Select Area", df["Property_Area"].unique())
filtered_df = df[df["Property_Area"] == area]
st.write(f"Showing {len(filtered_df)} records from '{area}' area:")
st.dataframe(filtered_df)

# ===== Loan Approval Prediction =====
st.markdown("---")
st.subheader("🧠 Predict Loan Approval")

with st.form("loan_form"):
    st.write("### Enter Applicant Details")

    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=1, value=100)
    loan_term = st.number_input("Loan Term (in days)", min_value=30, value=360)
    credit_history = st.selectbox("Credit History", options=[1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")

    submitted = st.form_submit_button("Predict Loan Approval")

    if submitted:
        # Step 1: Prepare raw input
        raw_input = [[
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history
        ]]

        # Step 2: Scale input
        scaled_input = scaler.transform(raw_input)

        # Step 3: Predict
        prediction = model.predict(scaled_input)

        # Step 4: Display result
        if prediction[0] == 'Y' or prediction[0] == 1:
            st.success("✅ Loan is likely to be Approved!")
        else:
            st.error("❌ Loan is likely to be Rejected.")
update this code without load the dataset
