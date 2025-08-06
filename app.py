import joblib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load
model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")

# Title
st.title("📊 Loan Approval Dashboard")

# Top metrics
total_loans = len(df)
approved_loans = df[df["Loan_Status"] == "Y"].shape[0]
rejected_loans = df[df["Loan_Status"] == "N"].shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Loans", total_loans)
col2.metric("Approved", approved_loans)
col3.metric("Rejected", rejected_loans)

st.markdown("---")

# Pie chart
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

# Show Data Table
st.subheader("📋 Sample Loan Data")
st.dataframe(df.head(10))

# Optional: Filter
st.subheader("🔍 Filter by Property Area")
area = st.selectbox("Select Area", df["Property_Area"].unique())
filtered_df = df[df["Property_Area"] == area]
st.write(f"Showing {len(filtered_df)} records from '{area}' area:")
st.dataframe(filtered_df)

