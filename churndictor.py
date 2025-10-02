import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model (which includes preprocessing pipeline)
model = joblib.load(r"C:\Users\LENOVO\Desktop\github_portfolio\Churn_Prediction_Project_Portfolio\best_churn_model.joblib")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 CHURNDICTOR: Customer Churn Prediction App")
st.markdown("Fill in customer details to predict whether they are likely to churn.")

st.divider()

# --- Collect inputs from user ---
age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

tech_support = st.selectbox("Tech Support", options=["Yes", "No"])
contract_type = st.selectbox("Contract Type", options=["Month-to-Month", "One Year", "Two Year"])
internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])

# --- Predict button ---
if st.button("Predict Churn"):
    # Match training features
    input_dict = {
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Gender": gender,
        "TechSupport": tech_support,
        "ContractType": contract_type,
        "InternetService": internet_service
    }

    input_df = pd.DataFrame([input_dict])

    # Predict using pipeline
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.success(f"🔮 Churn Prediction: {'Yes' if pred == 1 else 'No'}")
    st.info(f"📊 Churn Probability: {prob:.2%}")

    if pred == 1:
        st.warning("⚠️ This customer is likely to churn.")
    else:
        st.balloons()
        st.success("✅ This customer is likely to stay.")
else:
    st.info("👆 Enter details and click Predict")
