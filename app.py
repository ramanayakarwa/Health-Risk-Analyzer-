import streamlit as st
import pickle
import pandas as pd
from fpdf import FPDF
import datetime
import os

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Health Prediction Portal", layout="wide", page_icon="💖")

st.sidebar.markdown("<h2 style='color:#ff4b4b;'>Health Prediction Portal</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
menu = ["🏠 Home"]
choice = st.sidebar.radio("📍 Navigate:", menu)

# ------------------ Load Models ------------------
heart_rf, heart_features = pickle.load(open("models/heart_rf_model.pkl", "rb"))
diabetes_rf, diabetes_features = pickle.load(open("models/diabetes_rf_model.pkl", "rb"))
stroke_rf, stroke_features = pickle.load(open("models/stroke_rf_model.pkl", "rb"))

# ------------------ Helper Functions ------------------
def risk_category(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "High"

risk_advice = {
    "Heart Disease": {
        "Low": "Maintain a healthy lifestyle and regular checkups.",
        "Moderate": "Monitor diet, exercise regularly, and consult a doctor.",
        "High": "Consult a cardiologist immediately and follow medical advice."
    },
    "Diabetes": {
        "Low": "Continue healthy eating habits and regular exercise.",
        "Moderate": "Monitor blood sugar levels and consult a doctor if needed.",
        "High": "Seek medical advice promptly and follow a treatment plan."
    },
    "Stroke": {
        "Low": "Maintain a healthy lifestyle and regular health checkups.",
        "Moderate": "Monitor blood pressure and cholesterol; consult your doctor.",
        "High": "Immediate medical attention recommended; follow preventive measures."
    }
}

def predict_disease(model, user_input, all_features):
    df = pd.DataFrame([user_input])
    for f in all_features:
        if f not in df.columns:
            df[f] = 0
    df = df[all_features]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    prob = float(model.predict_proba(df)[0][1])
    category = risk_category(prob)
    return prob, category

# ------------------ Enhanced Validation ------------------
def validate_inputs(user_input):
    if not user_input or all(v == "" or v is None for v in user_input.values()):
        return "no_data"

    if any(v == "" or v is None for v in user_input.values()):
        return "empty"

    numeric_values = []
    non_numeric_values = []
    for v in user_input.values():
        try:
            numeric_values.append(float(v))
        except:
            non_numeric_values.append(v)

    if len(numeric_values) > 0 and all(v == 0 for v in numeric_values):
        return "all_zero"

    default_values = ["no", "unknown", "none", "select", "n/a", "none_selected"]
    if non_numeric_values and all(str(v).strip().lower() in default_values for v in non_numeric_values):
        return "default_categorical"

    return "ok"

if "results" not in st.session_state:
    st.session_state["results"] = {
        "Heart Disease": (None, None),
        "Diabetes": (None, None),
        "Stroke": (None, None)
    }

# ================================================================
# 🏠 HOME PAGE
# ================================================================
if choice == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#ff4b4b;'>Health Risk Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; font-size:18px;'>
    Welcome to the <b>Health Prediction Portal</b> — an AI-powered tool that predicts your risks for 
    <b>Heart Disease</b>, <b>Diabetes</b>, and <b>Stroke</b> using advanced machine learning models.  
    </div>
    """, unsafe_allow_html=True)

    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966487.png", width=250)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Start Prediction"):
        st.session_state["show_tabs"] = True
        st.toast("Scroll down to start your health prediction!", icon="💡")

    if st.session_state.get("show_tabs", False):
        st.markdown("---")
        st.markdown("<h3 style='color:#1f77b4;'>🩺 Begin Your Prediction Below</h3>", unsafe_allow_html=True)

        # Create Tabs
        tab1, tab2, tab3 = st.tabs(["Heart Disease", "Diabetes", "Stroke"])

        # ❤️ HEART DISEASE TAB
        with tab1:
            st.subheader("Heart Disease Prediction")
            heart_input_features = [col for col in heart_features if col not in ["Gender_M"]]
            user_input = {}
            for col in heart_input_features:
                user_input[col] = st.number_input(col.replace("_", " ").title(), min_value=0.0, key=f"heart_{col}")

            if st.button("Predict Heart Disease"):
                status = validate_inputs(user_input)
                if status == "no_data":
                    st.warning("⚠️ No patient data entered, so no prediction.")
                elif status == "empty":
                    st.warning("⚠️ Please fill all fields before prediction.")
                elif status == "all_zero":
                    st.warning("⚠️ Please enter at least one non-zero value.")
                elif status == "default_categorical":
                    st.warning("⚠️ Please select meaningful options instead of defaults.")
                else:
                    heart_prob, heart_category = predict_disease(heart_rf, user_input, heart_features)
                    st.session_state["results"]["Heart Disease"] = (heart_prob, heart_category)
                    color = 'red' if heart_category == "High" else 'orange' if heart_category == "Moderate" else 'green'
                    st.markdown(f"<h3 style='color:{color};'>Risk: {heart_prob*100:.2f}% ({heart_category})</h3>", unsafe_allow_html=True)
                    st.info(risk_advice["Heart Disease"][heart_category])

        # 🩸 DIABETES TAB
        with tab2:
            st.subheader("Diabetes Prediction")
            gender_col = next((col for col in diabetes_features if col.lower().startswith("gender_")), None)
            diabetes_input_features = [col for col in diabetes_features if not col.lower().startswith("gender_")]
            user_input = {}

            if gender_col:
                gender_option = st.selectbox("Gender", ["Select", "Male", "Female"], key="diabetes_gender")
                if gender_option == "Select":
                    user_input[gender_col] = "select"
                else:
                    user_input[gender_col] = 1 if gender_option == "Male" else 0

            for col in diabetes_input_features:
                user_input[col] = st.number_input(col.replace("_", " ").title(), min_value=0.0, key=f"diabetes_{col}")

            if st.button("Predict Diabetes"):
                status = validate_inputs(user_input)
                if status == "no_data":
                    st.warning("⚠️ No patient data entered, so no prediction.")
                elif status == "empty":
                    st.warning("⚠️ Please fill all fields before prediction.")
                elif status == "all_zero":
                    st.warning("⚠️ Please enter at least one non-zero value.")
                elif status == "default_categorical":
                    st.warning("⚠️ Please select meaningful options instead of defaults.")
                else:
                    diabetes_prob, diabetes_category = predict_disease(diabetes_rf, user_input, diabetes_features)
                    st.session_state["results"]["Diabetes"] = (diabetes_prob, diabetes_category)
                    color = 'red' if diabetes_category == "High" else 'orange' if diabetes_category == "Moderate" else 'green'
                    st.markdown(f"<h3 style='color:{color};'>Risk: {diabetes_prob*100:.2f}% ({diabetes_category})</h3>", unsafe_allow_html=True)
                    st.info(risk_advice["Diabetes"][diabetes_category])

        # 🧠 STROKE TAB (Correctly Placed and Indented)
        with tab3:
            st.subheader("Stroke Prediction")
            user_input = {}

            # Basic numeric inputs
            user_input["age"] = st.number_input("Age", min_value=0.0, max_value=120.0, step=1.0)
            user_input["avg_glucose_level"] = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
            user_input["bmi"] = st.number_input("BMI (Body Mass Index)", min_value=0.0, step=0.1)
            user_input["hypertension"] = st.number_input("Hypertension (0 = No, 1 = Yes)", min_value=0, max_value=1, step=1)
            user_input["heart_disease"] = st.number_input("Heart Disease (0 = No, 1 = Yes)", min_value=0, max_value=1, step=1)

            # One-hot mappings
            gender = st.selectbox("Gender", ["Select", "Male", "Female"])
            for val in ["Male"]:
                user_input[f"gender_{val}"] = 1 if gender == val else 0

            ever_married = st.selectbox("Ever Married", ["Select", "Yes", "No"])
            for val in ["Yes"]:
                user_input[f"ever_married_{val}"] = 1 if ever_married == val else 0

            work_type = st.selectbox("Work Type", ["Select", "Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
            for val in ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"]:
                user_input[f"work_type_{val}"] = 1 if work_type == val else 0

            residence = st.selectbox("Residence Type", ["Select", "Urban", "Rural"])
            for val in ["Urban"]:
                user_input[f"Residence_type_{val}"] = 1 if residence == val else 0

            smoking = st.selectbox("Smoking Status", ["Select", "formerly smoked", "never smoked", "smokes", "Unknown"])
            for val in ["formerly smoked", "never smoked", "smokes", "Unknown"]:
                user_input[f"smoking_status_{val}"] = 1 if smoking == val else 0

            if st.button("Predict Stroke"):
                status = validate_inputs(user_input)
                if status == "no_data":
                    st.warning("⚠️ No patient data entered, so no prediction.")
                elif status == "empty":
                    st.warning("⚠️ Please fill all fields before prediction.")
                elif status == "all_zero":
                    st.warning("⚠️ Please enter at least one non-zero value.")
                elif status == "default_categorical":
                    st.warning("⚠️ Please select meaningful options instead of defaults.")
                else:
                    stroke_prob, stroke_category = predict_disease(stroke_rf, user_input, stroke_features)
                    st.session_state["results"]["Stroke"] = (stroke_prob, stroke_category)
                    color = 'red' if stroke_category == "High" else 'orange' if stroke_category == "Moderate" else 'green'
                    st.markdown(f"<h3 style='color:{color};'>Risk: {stroke_prob*100:.2f}% ({stroke_category})</h3>", unsafe_allow_html=True)
                    st.info(risk_advice["Stroke"][stroke_category])

# ================================================================
#  PDF DOWNLOAD BUTTON
# ================================================================
def generate_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, "Health Prediction Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    for disease, (prob, category) in results.items():
        if prob is not None:
            pdf.multi_cell(0, 8,
                           f"{disease}:\n"
                           f"  - Risk Probability: {prob*100:.2f}%\n"
                           f"  - Risk Category: {category}\n"
                           f"  - Advice: {risk_advice[disease][category]}\n")
            pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.multi_cell(0, 6, "Disclaimer: This report is for informational purposes only.", align="L")
    pdf_file = "Health_Report.pdf"
    pdf.output(pdf_file)
    return pdf_file

if st.button(" Download Health Report"):
    results = st.session_state["results"]
    if any(v[0] is not None for v in results.values()):
        pdf_file = generate_pdf(results)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download PDF", data=f, file_name="Health_Report.pdf", mime="application/pdf")
    else:
        st.warning("⚠️ Please make at least one prediction before downloading the report.")
