import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("dementia_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.set_page_config(page_title="Dementia Prediction", layout="centered")
st.title("🧠 Dementia Prediction App")

st.write("Enter patient details below:")

# Input fields
mr_delay = st.number_input("MR Delay", 0)
visit = st.number_input("Visit", 1)
age = st.number_input("Age", 0)
educ = st.number_input("Education (EDUC)", 0)
ses = st.number_input("SES", 0.0)
mmse = st.number_input("MMSE", 0.0)
cdr = st.number_input("CDR", 0.0)
etiv = st.number_input("eTIV", 0.0)
nwbv = st.number_input("nWBV", 0.0)
asf = st.number_input("ASF", 0.0)

gender = st.selectbox("Gender", ["Male", "Female"])

# Encode gender
gender_val = 1 if gender == "Male" else 0

# Predict button
if st.button("🔍 Predict"):

    try:
        # Arrange input in correct order
        input_data = np.array([[mr_delay, gender_val, visit, age, educ, ses,
                                mmse, cdr, etiv, nwbv, asf]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)

        # Output
        if prediction[0] == 1:
            st.error("❌ Dementia Detected")
        else:
            st.success("✅ No Dementia")

    except Exception as e:
        st.error(f"Error: {e}")