import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("C:/Users/hp/OneDrive/Desktop/jupyter notebook/diabetes/diabetes_decision_tree_model_last.pkl")

# Set up the Streamlit app title
st.title("Diabetes Prediction App")

# Create input fields for user data
gender = st.selectbox("Gender", options=["Female", "Male"])
age = st.number_input("Age", min_value=0, max_value=120)
hypertension = st.selectbox("Hypertension", options=[0, 1])
heart_disease = st.selectbox("Heart Disease", options=[0, 1])
smoking_history = st.selectbox("Smoking History", options=["never", "current", "No Info"])
bmi = st.number_input("BMI", format="%.2f")
hba1c_level = st.number_input("HbA1c Level", format="%.1f")
blood_glucose_level = st.number_input("Blood Glucose Level", format="%.1f")

# Prepare the input data
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'smoking_history': [smoking_history],
    'bmi': [bmi],
    'HbA1c_level': [hba1c_level],
    'blood_glucose_level': [blood_glucose_level]
})

# Convert categorical variables to dummy variables
input_data = pd.get_dummies(input_data, drop_first=True)

# Ensure input data matches the model's input
# Add missing columns if any
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("The model predicts: Diabetes")
    else:
        st.success("The model predicts: No Diabetes")







#cd "C:\Users\hp\OneDrive\Desktop\jupyter notebook\diabetes"
#streamlit run interface.py
