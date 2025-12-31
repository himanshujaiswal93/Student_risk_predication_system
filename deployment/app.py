import streamlit as st
import joblib
import numpy as np

# -----------------------
# Load model & scaler
# -----------------------
model = joblib.load('student_risk_logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🎓 Student Risk Prediction App")
st.write("Enter student details to predict risk level.")

# -----------------------
# Inputs
# -----------------------
math = st.number_input("Math Score", 0, 100, 50)
reading = st.number_input("Reading Score", 0, 100, 50)
writing = st.number_input("Writing Score", 0, 100, 50)

gender = st.selectbox("Gender", ["Female", "Male"])
lunch = st.selectbox("Lunch Type", ["Standard", "Free/Reduced"])
test_prep = st.selectbox("Test Preparation", ["None", "Completed"])

race = st.selectbox("Race/Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])

parent_edu = st.selectbox(
    "Parental Level of Education",
    [
        "Some High School",
        "High School",
        "Some College",
        "Associate's Degree",
        "Bachelor's Degree",
        "Master's Degree"
    ]
)

# -----------------------
# Encoding
# -----------------------
gender_male = 1 if gender == "Male" else 0
lunch_standard = 1 if lunch == "Standard" else 0
test_prep_none = 1 if test_prep == "None" else 0

race_B = 1 if race == "Group B" else 0
race_C = 1 if race == "Group C" else 0
race_D = 1 if race == "Group D" else 0
race_E = 1 if race == "Group E" else 0

parent_somehighschool = 1 if parent_edu == "Some High School" else 0
parent_highschool = 1 if parent_edu == "High School" else 0
parent_somecollege = 1 if parent_edu == "Some College" else 0
parent_associate = 1 if parent_edu == "Associate's Degree" else 0
parent_bachelor = 1 if parent_edu == "Bachelor's Degree" else 0
parent_master = 1 if parent_edu == "Master's Degree" else 0

avg_score = (math + reading + writing) / 3

# -----------------------
# FINAL INPUT (17 FEATURES)
# -----------------------
input_data = np.array([[
    math, reading, writing, avg_score,
    gender_male, lunch_standard, test_prep_none,
    race_B, race_C, race_D, race_E,
    parent_associate, parent_bachelor, parent_highschool,
    parent_master, parent_somecollege, parent_somehighschool
]])

input_scaled = scaler.transform(input_data)

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Risk"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.success("🟢 Low Risk Student")
    elif prediction == 1:
        st.warning("🟡 Medium Risk Student")
    else:
        st.error("🔴 High Risk Student")

