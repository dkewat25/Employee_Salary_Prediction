import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("employee_salary_model.pkl")

st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.write("Fill in the employee details below to predict their salary.")

# User input form
with st.form("prediction_form"):
    education = st.selectbox("Education Level", ["PhD", "Masters", "Bachelors", "High School", "Associate"])
    experience = st.slider("Years of Experience", 0, 40, 5)
    role = st.selectbox("Role", ["Data Scientist", "Software Engineer", "Manager", "HR", "Accountant", "Analyst"])
    department = st.selectbox("Department", ["IT", "Finance", "HR", "Marketing", "Operations"])
    location = st.selectbox("Location", ["New York", "San Francisco", "Austin", "Chicago", "Seattle"])

    submit = st.form_submit_button("Predict Salary")

# Process prediction
if submit:
    user_input = pd.DataFrame([{
        "Education": education,
        "Experience": experience,
        "Role": role,
        "Department": department,
        "Location": location
    }])

    prediction = model.predict(user_input)[0]
    st.success(f"💰 Predicted Salary: **${int(prediction):,}** per year")
