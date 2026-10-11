import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page configuration
st.set_page_config(page_title="Telecom Customer Churn Predictor", layout="wide")

st.title("📈 Telecom Customer Churn Prediction App")
st.write("Enter customer details below to estimate churn risk.")

# Load pre-saved model, scaler, and encoders
@st.cache_resource
def load_assets():
    model = joblib.load('rf_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, scaler, encoders

try:
    model, scaler, encoders = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Please ensure 'rf_churn_model.pkl', 'scaler.pkl', and 'encoders.pkl' exist.")
    st.stop()

# Layout inputs in columns
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Is Senior Citizen?", [0, 1])
    Partner = st.selectbox("Has Partner?", ["Yes", "No"])
    Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

with col2:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)

# Submit button
if st.button("🔮 Predict Churn"):
    # Prepare the raw input dictionary
    input_data = {
        'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner, 'Dependents': Dependents,
        'tenure': tenure, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
        'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies, 'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges, 'TotalCharges': TotalCharges
    }

    # Derived features engineered during preprocessing
    input_data['NewCustomer'] = 1 if tenure <= 2 else 0
    input_data['Tenure_MonthlyCharges_Interaction'] = tenure * MonthlyCharges

    df_input = pd.DataFrame([input_data])

    # Encode categorical variables
    for col, encoder in encoders.items():
        df_input[col] = encoder.transform(df_input[col])

    # Scaler expect inputs in the specific column order
    numerical_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
        'NewCustomer', 'Tenure_MonthlyCharges_Interaction'
    ]
    df_input[numerical_features] = scaler.transform(df_input[numerical_features])

    # Arrange columns to match train features order exactly
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'NewCustomer', 'Tenure_MonthlyCharges_Interaction'
    ]
    df_input = df_input[feature_order]

    # Make prediction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    st.markdown("--- ")
    if prediction == 1:
        st.error(f"🚨 **High Risk of Churn!** (Probability: {probability:.2%})")
    else:
        st.success(f"✅ **Low Risk of Churn!** (Probability of churn: {probability:.2%})")
