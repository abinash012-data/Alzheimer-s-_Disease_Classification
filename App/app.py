import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")
training_columns = list(model.feature_names_in_)

# Preprocessing function
def preprocess_new_data(new_data, training_columns):
    if 'Gender' in new_data.columns:
        new_data['Gender'] = new_data['Gender'].map({'Male':1,'Female':0})
    if 'Ethnicity' in new_data.columns:
        new_data['Ethnicity'] = new_data['Ethnicity'].astype('category').cat.codes
    for col in training_columns:
        if col not in new_data.columns:
            new_data[col] = 0
    new_data = new_data[training_columns]
    return new_data

st.set_page_config(page_title="Alzheimer's Prediction", layout="wide")
st.title("🧠 Alzheimer's Disease Prediction")
st.markdown("Fill in the patient details to predict Alzheimer stage and risk category.")

# ------------------ Vitals & Lifestyle ------------------
st.subheader("Vitals & Lifestyle")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age",50,100,70)
    education = st.slider("Years of Education",0,20,16)
    bmi = st.slider("BMI",10.0,50.0,24.5)
with col2:
    mmse = st.slider("MMSE Score",0,30,25)
    systolic_bp = st.slider("Systolic BP",80,200,120)
    diastolic_bp = st.slider("Diastolic BP",40,120,80)
with col3:
    physical_activity = st.slider("Physical Activity (0-5)",0,5,1)
    diet_quality = st.slider("Diet Quality (0-5)",0,5,3)
    sleep_quality = st.slider("Sleep Quality (hours)",0,24,7)
gender = st.selectbox("Gender", ["Male","Female"])
ethnicity = st.text_input("Ethnicity","Caucasian")

# ------------------ Medical History Expander ------------------
with st.expander("🩺 Medical History"):
    col1, col2, col3 = st.columns(3)
    with col1:
        smoking = st.selectbox("Smoking", ["No","Yes"])
        alcohol = st.selectbox("Alcohol Consumption", ["No","Yes"])
        family_history = st.selectbox("Family History of Alzheimer", ["No","Yes"])
    with col2:
        cardio = st.selectbox("Cardiovascular Disease", ["No","Yes"])
        diabetes = st.selectbox("Diabetes", ["No","Yes"])
        depression = st.selectbox("Depression", ["No","Yes"])
    with col3:
        head_injury = st.selectbox("History of Head Injury", ["No","Yes"])
        hypertension = st.selectbox("Hypertension", ["No","Yes"])
        adl = st.slider("ADL (0-5)",0,5,1)
    functional = st.slider("Functional Assessment (0-5)",0,5,1)

# ------------------ Cognitive & Behavioral Expander ------------------
with st.expander("🧠 Cognitive & Behavioral"):
    col1, col2, col3 = st.columns(3)
    with col1:
        memory_complaints = st.selectbox("Memory Complaints", ["No","Yes"])
        behavioral = st.selectbox("Behavioral Problems", ["No","Yes"])
    with col2:
        confusion = st.selectbox("Confusion", ["No","Yes"])
        disorientation = st.selectbox("Disorientation", ["No","Yes"])
    with col3:
        personality_changes = st.selectbox("Personality Changes", ["No","Yes"])
        difficulty_tasks = st.selectbox("Difficulty Completing Tasks", ["No","Yes"])
    forgetfulness = st.selectbox("Forgetfulness", ["No","Yes"])

# ------------------ Prediction ------------------
if st.button("Predict"):
    # Convert Yes/No to 0/1
    def yesno_to_int(x):
        return 1 if x=="Yes" else 0

    new_patient = pd.DataFrame({
        'Age':[age],'EducationLevel':[education],'BMI':[bmi],
        'MMSE':[mmse],'SystolicBP':[systolic_bp],'DiastolicBP':[diastolic_bp],
        'PhysicalActivity':[physical_activity],'DietQuality':[diet_quality],
        'SleepQuality':[sleep_quality],'Gender':[gender],'Ethnicity':[ethnicity],
        'Smoking':[yesno_to_int(smoking)],'AlcoholConsumption':[yesno_to_int(alcohol)],
        'FamilyHistoryAlzheimers':[yesno_to_int(family_history)],
        'CardiovascularDisease':[yesno_to_int(cardio)],'Diabetes':[yesno_to_int(diabetes)],
        'Depression':[yesno_to_int(depression)],'HeadInjury':[yesno_to_int(head_injury)],
        'Hypertension':[yesno_to_int(hypertension)],'ADL':[adl],'FunctionalAssessment':[functional],
        'MemoryComplaints':[yesno_to_int(memory_complaints)],
        'BehavioralProblems':[yesno_to_int(behavioral)],
        'Confusion':[yesno_to_int(confusion)],'Disorientation':[yesno_to_int(disorientation)],
        'PersonalityChanges':[yesno_to_int(personality_changes)],
        'DifficultyCompletingTasks':[yesno_to_int(difficulty_tasks)],
        'Forgetfulness':[yesno_to_int(forgetfulness)]
    })

    new_patient_processed = preprocess_new_data(new_patient, training_columns)
    stage = model.predict(new_patient_processed)[0]
    prob = float(model.predict_proba(new_patient_processed)[0,1])

    if prob < 0.3:
        risk = "Low Risk"; color="green"
    elif prob < 0.7:
        risk = "Medium Risk"; color="orange"
    else:
        risk = "High Risk"; color="red"

    st.markdown(f"### 🧩 Predicted Alzheimer Stage: **{stage}**")
    st.markdown(f"### 🔴 Risk Category: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
    st.progress(prob)
    st.write(f"Probability of Positive Class: {prob:.2f}")
