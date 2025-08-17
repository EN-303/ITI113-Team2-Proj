import joblib
import streamlit as st
import pandas as pd

st.title('Heart Disease Prediction App v1')

@st.cache_resource
def load_model():
    # model = joblib.load("model.joblib")
    # return joblib.load('model.joblib')
    return joblib.load('model.pkl')

try:
    preprocessor = joblib.load("preprocessor.pkl")
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None
    
def predict(X): 
    if model is None:
        st.error("Model not available. Prediction aborted.")
        return None
    try:
        return model.predict(X)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None
    
# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographics")
    age = st.slider('Age', 20, 120, 30)
    gender = st.radio("Gender", ('Female', 'Male'))
    gender = 0 if gender == 'Female' else 1

    st.subheader("🫀 Chest Pain")
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )
    chest_pain_value = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }[chest_pain]

    st.subheader("📉 Exercise")
    exerciseangina = st.radio('Exercise-Induced Angina', ('No', 'Yes'))
    exerciseangina = 0 if exerciseangina == 'No' else 1
    oldpeak = st.slider(
        'ST depression induced by exercise (oldpeak)',
        min_value=0.0,
        max_value=7.0,
        value=0.0,
        step=0.1
    )
    slope = st.selectbox(
        "Slope of the peak ST segment",
        ["Upsloping", "Flat", "Downsloping"]
    )
    slope_value = {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3
    }[slope]

with col2:
    st.subheader("🩺 Vital Signs & Lab Results")
    restingBP = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 100)
    serumcholestrol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 100)

    fastingbloodsugar = st.radio("Fasting Blood Sugar", ('Below 120 mg/dl', '120 mg/dl and above'))
    fastingbloodsugar = 0 if fastingbloodsugar == 'Below 120 mg/dl' else 1

    st.subheader("📊 ECG & Heart Rate")
    restingrelectro = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T wave abnormality", "Probable or definite left ventricular hypertrophy"]
    )
    restingrelectro_value = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Probable or definite left ventricular hypertrophy": 2
    }[restingrelectro]

    maxheartrate = st.slider('Max Heart Rate Achieved', 71, 202, 100)

    noofmajorvessels = st.slider('Number of Major Vessels (0-3)', 0, 3, 0)


# Feature names
feature_names = [
    "Age", "Gender (0=F, 1=M)", "Chest Pain Type (0=TA, 1=AA, 2=NA, 3=ASY)", "Resting BP", "Serum Cholesterol",
    "Fasting Blood Sugar", "Resting ECG (0=N, 1=Ab, 2=P)", "Max Heart Rate",
    "Exercise Angina (0=N, 1=Y)", "Oldpeak", "Slope (1=Up, 2=Flat, 3=Down)", "Number of Major Vessels (0-3)"
]

# Input vector for prediction
input_vector = [[
    age, gender, chest_pain_value, restingBP, serumcholestrol,
    fastingbloodsugar, restingrelectro_value, maxheartrate,
    exerciseangina, oldpeak, slope_value, noofmajorvessels
]]

input_df = pd.DataFrame(input_vector, columns=[
    'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
    'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
    'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
])

X_transformed = preprocessor.transform(input_df)

# Prediction section
st.markdown("---")
st.subheader("🧪 Prediction")

if st.button("Predict Heart Disease Risk"):
    prediction = predict(X_transformed)
    if prediction is not None:
        if prediction == 0:
            st.success("✅ Prediction: No Heart Disease")
        else:
            st.error("⚠️ Prediction: Potential Heart Disease Detected")

# Display as table
# input_df = pd.DataFrame(input_vector, columns=feature_names)
st.subheader("🧾 Patient Feature Summary")
st.dataframe(input_df, use_container_width=True)

