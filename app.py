import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load trained model
model = joblib.load("models/fraud_detection_rf.pkl")

st.set_page_config(
    page_title="Digital Transaction Fraud Detector",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Digital Transaction Fraud Detector")
st.write(
    "ML-based fraud risk prediction for digital financial transactions."
)

st.divider()

st.subheader("Enter Transaction Details")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

old_balance = st.number_input(
    "Sender Balance Before Transaction",
    min_value=0.0,
    value=5000.0
)

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

hour = st.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)

sender_count = st.number_input(
    "Sender Transaction Count",
    min_value=1,
    value=1
)

receiver_count = st.number_input(
    "Receiver Transaction Count",
    min_value=1,
    value=1
)


if st.button("🔍 Check Transaction"):

    amount_log = np.log1p(amount)

    amount_to_sender_balance = (
        amount / (old_balance + 1)
    )

    input_data = pd.DataFrame({
        "amount_log": [amount_log],
        "amount_to_sender_balance": [amount_to_sender_balance],
        "sender_transaction_count": [sender_count],
        "receiver_transaction_count": [receiver_count],
        "hour": [hour],
        "type": [transaction_type]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=["type"],
        dtype=int
    )

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.error("🚨 HIGH RISK: Potential Fraudulent Transaction")
    else:
        st.success("✅ LOW RISK: Transaction Classified as Legitimate")

    st.metric(
        "Fraud Risk Probability",
        f"{probability * 100:.2f}%"
    )