import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from transformers import pipeline

chatbot = pipeline("text-generation", model="distilgpt2", framework="pt")

# --- Load Symptom Checker Models & Data ---
def load_symptom_model():
    base_path = "C:/Users/dipay/OneDrive/Desktop/AI_Healthcare_Chatbot"
    with open(os.path.join(base_path, "model/svc.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_path, "model/scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_path, "model/features.pkl"), "rb") as f:
        features = pickle.load(f)
    description = pd.read_csv(os.path.join(base_path, "dataset/description.csv"))
    precautions = pd.read_csv(os.path.join(base_path, "dataset/precautions_df.csv"))
    medications = pd.read_csv(os.path.join(base_path, "dataset/medications.csv"))
    diets = pd.read_csv(os.path.join(base_path, "dataset/diets.csv"))
    workout = pd.read_csv(os.path.join(base_path, "dataset/workout_df.csv"))
    return model, scaler, features, description, precautions, medications, diets, workout

# --- Load Heart & Diabetes Models ---
def load_health_models():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    heart_model_path = os.path.join(working_dir, "model", "heart.pkl")
    diabetes_model_path = os.path.join(working_dir, "model", "diabetes.pkl")
    heart_model = pickle.load(open(heart_model_path, 'rb'))
    diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
    return heart_model, diabetes_model

# --- Healthcare Chatbot & Prediction UI ---
def healthcare_assistant_app():
    heart_model, diabetes_model = load_health_models()

    if "show_heart_form" not in st.session_state:
        st.session_state.show_heart_form = False
    if "show_diabetes_form" not in st.session_state:
        st.session_state.show_diabetes_form = False

    def healthcare_chatbot(user_input):
        user_input = user_input.lower()
        if "symptoms" in user_input:
            return "Please consult a doctor for accurate advice."
        elif "appointment" in user_input:
            return "Would you like to schedule an appointment with the doctor?"
        elif "medication" in user_input:
            return "It's important to take prescribed medicines regularly. If you have concerns, consult your doctor."
        elif "heart" in user_input:
            st.session_state.show_heart_form = True
            st.session_state.show_diabetes_form = False
            return "Please fill out the heart disease prediction form below."
        elif "diabetes" in user_input:
            st.session_state.show_diabetes_form = True
            st.session_state.show_heart_form = False
            return "Please fill out the diabetes prediction form below."
        else:
            response = chatbot(user_input, max_length=500, num_return_sequences=1)
            return response[0]['generated_text']

    st.title("Healthcare Assistant Chatbot")

    user_input = st.text_input("How can I assist you today...")
    if st.button("Submit"):
        if user_input:
            st.write("User:", user_input)
            with st.spinner("Processing your query, Please wait..."):
                response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant:", response)
        else:
            st.write("Please enter a message to get a response.")

    if st.session_state.show_heart_form:
        heart_disease_prediction()
    if st.session_state.show_diabetes_form:
        diabetes_prediction()

# --- Symptom Checker ---
def symptom_checker_app():
    model, scaler, features, description, precautions, medications, diets, workout = load_symptom_model()

    def get_info(disease):
        desc = description[description['Disease'] == disease]['Description'].values
        pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten()
        med = medications[medications['Disease'] == disease]['Medication'].values
        diet = diets[diets['Disease'] == disease]['Diet'].values
        wrk = workout[workout['Disease'] == disease]['workout'].values
        return desc, pre, med, diet, wrk

    st.title("AI Healthcare Symptom Checker")
    st.markdown("Enter your symptoms to get a possible diagnosis and health guidance.")

    selected_symptoms = st.multiselect("Select symptoms:", options=features)

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_data = pd.DataFrame([[1 if feature in selected_symptoms else 0 for feature in features]], columns=features)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            predicted_disease = prediction[0]

            st.success(f"I think you have **{predicted_disease}**")

            desc, pre, med, diet, wrk = get_info(predicted_disease)
            if desc:
                st.warning(desc[0])
            if pre.any():
                st.markdown("**Precautions:**")
                for p in pre:
                    st.markdown(f"- {p}")
            if med.any():
                st.markdown("**Medications:**")
                for m in med:
                    st.markdown(f"- {m}")
            if diet.any():
                st.markdown("**Recommended Diet:**")
                for d in diet:
                    st.markdown(f"- {d}")
            if wrk.any():
                st.markdown("**Suggested Workout:**")
                for w in wrk:
                    st.markdown(f"- {w}")
    pass

# --- Diabetes Prediction ---
def diabetes_prediction():
    heart_model, diabetes_model = load_health_models()
    st.write("**Let's do a quick Diabetes Prediction Test**")

    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            pregnancies = st.text_input("Number of Pregnancies")
        with col2:
            glucose = st.text_input("Glucose Level")
        with col3:
            blood_pressure = st.text_input("Blood Pressure (mm Hg)")
        with col1:
            skin_thickness = st.text_input("Skin Thickness (mm)")
        with col2:
            insulin = st.text_input("Insulin Level (mu U/ml)")
        with col3:
            bmi = st.text_input("Body Mass Index (BMI)")
        with col1:
            dpf = st.text_input("Diabetes Pedigree Function")
        with col2:
            age = st.text_input("Age")

        submitted = st.form_submit_button("Predict Diabetes")

        if submitted:
            try:
                user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
                user_input = [float(x) for x in user_input]
                prediction = diabetes_model.predict([user_input])

                if prediction[0] == 1:
                    st.warning("This person has diabetes.")
                    st.subheader("Dietary Recommendations:")
                    st.write("""
                    - Reduce carbohydrate intake
                    - Increase fiber consumption
                    - Stay hydrated
                    - Eat healthy fats
                    - Monitor blood sugar levels regularly
                    """)
                else:
                    st.success("This person does not have diabetes.")

            except ValueError:
                st.error("Please enter valid numeric values for all fields.")

# --- Heart Disease Prediction ---
def heart_disease_prediction():
    heart_model, diabetes_model = load_health_models()
    st.write("**Let's do a quick Heart Disease Prediction Test**")

    with st.form("heart_disease_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input("Age")
        with col2:
            sex = st.text_input("Sex (1=Male, 0=Female)")
        with col3:
            cp = st.text_input("Chest Pain Types (0-3)")
        with col1:
            trestbps = st.text_input("Blood Pressure")
        with col2:
            chol = st.text_input("Serum Cholesterol in mg/dl")
        with col3:
            fbs = st.text_input('Blood Sugar level more than 120 mg/dl (1=Yes, 0=No)')
        with col1:
            restecg = st.text_input('Electrocardiographic results (0-2)')
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
        with col3:
            exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)')
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
        with col3:
            ca = st.text_input('Major vessels colored by fluoroscopy (0-3)')
        with col1:
            thal = st.text_input('Thal: 1=Normal, 2=Fixed Defect, 3=Reversible Defect')

        submitted = st.form_submit_button("Predict Heart Disease")

        if submitted:
            try:
                user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                user_input = [float(x) for x in user_input]

                prediction = heart_model.predict([user_input])

                if prediction[0] == 1:
                    st.warning("This person has heart disease.")
                    st.subheader("Dietary Recommendations:")
                    st.write("""
                    - Eat more Fruits, vegetables, whole grains, lean proteins  
                    - Eat less Salt, red meat, added sugars  
                    - Use healthy fats like Olive oil, avocado  
                    """)
                    st.subheader("Necessary Tests to do right now:")
                    st.write("""
                    1. Electrocardiogram (ECG/EKG) 
                    2. Echocardiogram
                    3. Cardiac MRI/CT Scan
                    4. Coronary Angiography 
                    5. Blood Tests  
                    """)

                else:
                    st.success("This person does not have heart disease.")

                    trestbps = float(trestbps)
                    chol = float(chol)
                    fbs = float(fbs)
                    thalach = float(thalach)

                    normal_ranges = {
                        "Heart Rate": (60, 100),
                        "Cholesterol": (0, 200),
                        "Blood Pressure": (90, 120)
                    }

                    user_values = {
                        "Heart Rate": thalach,
                        "Cholesterol": chol,
                        "Blood Pressure": trestbps
                    }

                    st.subheader("Health Level Check:")

                    # Heart Rate Check
                    if user_values["Heart Rate"] > 100:
                        st.warning(f"Heart Rate: {user_values['Heart Rate']} (Normal: 60-100) - Your heart beat is fast.")
                    elif user_values["Heart Rate"] < 60:
                        st.warning(f"Heart Rate: {user_values['Heart Rate']} (Normal: 60-100) - Your heart beat is comparatively slow.")
                    else:
                        st.success(f"Heart Rate: {user_values['Heart Rate']} (Normal: 60-100)")

                    # Cholesterol Check
                    if user_values["Cholesterol"] > 200:
                        st.warning(f"Cholesterol: {user_values['Cholesterol']} mg/dL (Normal: 0-200) - Cholesterol is high.")
                    elif user_values["Cholesterol"] < 100:
                        st.warning(f"Cholesterol: {user_values['Cholesterol']} mg/dL (Normal: 0-200) - Cholesterol is low.")
                    else:
                        st.success(f"Cholesterol: {user_values['Cholesterol']} mg/dL (Normal: 0-200)")

                    # Blood Pressure Check
                    if 120 <= user_values["Blood Pressure"] < 130:
                        st.warning(f"Blood Pressure: {user_values['Blood Pressure']} mmHg (Normal: 90-120) - Elevated Blood Pressure.")
                    elif 130 <= user_values["Blood Pressure"] < 140:
                        st.error(f"Blood Pressure: {user_values['Blood Pressure']} mmHg (Normal: 90-120) - Hypertension - Stage 1.")
                    elif user_values["Blood Pressure"] >= 140:
                        st.error(f"Blood Pressure: {user_values['Blood Pressure']} mmHg (Normal: 90-120) - Hypertension - Stage 2")
                    else:
                        st.success(f"Blood Pressure: {user_values['Blood Pressure']} mmHg (Normal: 90-120)")

            except ValueError:
                st.error("Please enter valid numeric values for all fields.")

# --- Main Navigation ---
def main():
    st.set_page_config(page_title="AI Healthcare App", layout="centered")
    app = st.sidebar.radio("Navigation", ["Chatbot", "Symptom Checker"])
    if app == "Chatbot":
        healthcare_assistant_app()
    elif app == "Symptom Checker":
        symptom_checker_app()

if __name__ == "__main__":
    main()
