import streamlit as st
import pandas as pd
import joblib
import time

# Page Layout Setup
st.set_page_config(page_title="AI Clinical Prognosis Portal", page_icon="⚡", layout="wide")

# Safe Header without complex string formatting
st.title("🔬 Advanced Patient Analytics System")
st.caption("Evaluating clinical dimensions using high-accuracy Ensemble Learning Architectures")
st.write("---")

# Load Saved Model Safely
model = joblib.load(r"C:\Users\Hp\cancer_rf_model.pkl")

# Layout Split Into Grid Columns
col1, col2 = st.columns(2, gap="large")

with col1:
    # 1. RED CONTAINER LOOK
    st.markdown("<div style='background-color: #ffe4e6; border: 3px solid #f43f5e; border-radius: 15px; padding: 20px; margin-bottom: 20px;'><h3 style='color: #e11d48; margin-top:0; font-weight:800;'>📊 PATIENT AGE VECTOR</h3></div>", unsafe_allow_html=True)
    age = st.slider("Select Patient Age Factor", min_value=1, max_value=120, value=45)
    st.write("##")
    
    # 2. PINK CONTAINER LOOK
    st.markdown("<div style='background-color: #fce7f3; border: 3px solid #ec4899; border-radius: 15px; padding: 20px; margin-bottom: 20px;'><h3 style='color: #db2777; margin-top:0; font-weight:800;'>👥 BIOLOGICAL MATRIX</h3></div>", unsafe_allow_html=True)
    gender = st.radio("Gender Profile Classification", ["Male", "Female"], horizontal=True)
    st.write("##")

    # 3. LAVENDER CONTAINER LOOK
    st.markdown("<div style='background-color: #f3e8ff; border: 3px solid #a855f7; border-radius: 15px; padding: 20px; margin-bottom: 20px;'><h3 style='color: #7c3aed; margin-top:0; font-weight:800;'>📐 STATISTICAL RATIOS</h3></div>", unsafe_allow_html=True)
    st.write("Enter the calculated *Age Survival Ratio* value:")
    age_survival_ratio = st.number_input("Ratio Metric Input", min_value=0.0, max_value=10.0, value=0.53, step=0.01)

with col2:
    # 4. BLACK CONTAINER LOOK (White Text Content)
    st.markdown("<div style='background-color: #1e293b; border: 3px solid #0f172a; border-radius: 15px; padding: 25px; color: white;'><h3 style='color: #38bdf8; margin-top:0; font-weight:800;'>🩺 CLINICAL DIAGNOSTICS & PARAMETERS</h3><p>Fill out the oncology specific metrics below:</p></div>", unsafe_allow_html=True)
    st.write("")
    
    cancer_type = st.selectbox("Select Oncology Variant Class", ["Lung Cancer", "Stomach Cancer", "Oral Cancer", "Ovarian Cancer", "Prostate Cancer", "Colorectal Cancer", "Cervical Cancer"])
    st.write("")
    
    stage = st.select_slider("Tumor Pathological Progression Stage", options=["Stage I", "Stage II", "Stage III", "Stage IV"])
    st.write("")
    
    treatment_type = st.selectbox("Assigned Therapeutic Protocol Node", ["Surgery", "Chemotherapy", "Surgery + Chemotherapy", "Palliative Care", "Targeted Therapy"])
    st.write("")
    
    survival_months = st.number_input("Observed Retention Timeline (Months)", min_value=1, max_value=500, value=24)

# Center Action Execution Section
st.write("##")
_, center_btn_block, _ = st.columns([1, 2, 1])
with center_btn_block:
    submit_btn = st.button("🚀 Execute Analytical Inference Engine")

# Prediction Execution Pipeline
if submit_btn:
    with st.spinner("Processing decisions..."):
        time.sleep(0.5)
        
    raw_sample = pd.DataFrame([{
        'Age': age, 
        'Gender': gender, 
        'Cancer_Type': cancer_type, 
        'Stage': stage, 
        'Treatment_Type': treatment_type, 
        'Survival_Months': survival_months, 
        'Age_Survival_Ratio': age_survival_ratio
    }])
    
    expected_features = model.feature_names_in_
    encoded_sample = pd.get_dummies(raw_sample)
    
    for col in expected_features:
        if col not in encoded_sample.columns:
            encoded_sample[col] = 0
            
    encoded_sample = encoded_sample[expected_features]
    prediction = model.predict(encoded_sample)
    
    st.write("---")
    st.subheader("📋 GENERATED DIAGNOSTIC REPORT")
    
    _, report_output_area, _ = st.columns([1, 2, 1])
    with report_output_area:
        if prediction[0] == 1:
            st.error("🚨 HIGH RISK VECTOR DETECTED\n\nStatus: Deceased\n\nThe array structures heavily align with critical mortality risk quadrants. Immediate protocol audit recommended.")
        else:
            st.success("✅ OPTIMAL RESILIENCE PROFILE\n\nStatus: Survived / Other\n\nStatistical mapping projects favorable prognostic outcome indicators. The current treatment matrix is optimal.")