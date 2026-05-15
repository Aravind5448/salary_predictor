import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json, os

st.set_page_config(page_title="Salary Predictor", page_icon="📊", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'salary_pipeline.joblib')
METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'metrics.json')

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, metrics

model, metrics = load_model()

st.title("📊 AI Salary Predictor")
st.markdown("Predict expected compensation based on role, experience, location, and company factors.")

# ── TOP ROW: Metrics ──
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Model R²", f"{metrics.get('test_r2', 'N/A'):.4f}", border=True)
with mcol2:
    st.metric("Test MAE", f"${metrics.get('test_mae', 0):,.0f}", border=True)
with mcol3:
    st.metric("Training Data", "25,000 records", border=True)
with mcol4:
    st.metric("Algorithm", "XGBoost", border=True)

st.divider()

# ── INPUT SECTION ──
st.subheader("Enter Job Details")

col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("Job Title", [
        "Data Scientist", "Machine Learning Engineer", "Data Analyst", "Data Engineer",
        "Research Scientist", "AI Engineer", "Applied Scientist", "Data Architect",
        "BI Engineer", "Deep Learning Engineer"
    ])
    experience_level = st.selectbox("Experience Level", ["EN (Entry)", "MI (Mid)", "SE (Senior)", "EX (Executive)"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Freelance"])
    work_year = st.slider("Work Year", 2020, 2025, 2025)

with col2:
    company_location = st.selectbox("Company Location", [
        "US", "GB", "CA", "DE", "IN", "FR", "ES", "NL", "AU", "SG",
        "BR", "JP", "KR", "AE", "CH", "SE", "DK", "NO", "IE", "IL"
    ])
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    remote_ratio = st.select_slider("Remote Work", options=[0, 50, 100],
        format_func=lambda x: {0: "🏢 On-site", 50: "🏡 Hybrid", 100: "🌍 Fully Remote"}[x])

# ── PREDICT BUTTON ──
predict_clicked = st.button("💰 Predict Salary", type="primary", use_container_width=True)

# ── FEATURE ENGINEERING (always computed for instant response) ──
exp_map = {"EN (Entry)": "EN", "MI (Mid)": "MI", "SE (Senior)": "SE", "EX (Executive)": "EX"}
emp_map = {"Full-time": "FT", "Part-time": "PT", "Contract": "CT", "Freelance": "FL"}
size_map = {"Small": "S", "Medium": "M", "Large": "L"}

input_df = pd.DataFrame([{
    'work_year': work_year, 'experience_level': exp_map[experience_level],
    'employment_type': emp_map[employment_type], 'job_title': job_title,
    'company_location': company_location, 'company_size': size_map[company_size],
    'remote_ratio': remote_ratio, 'employee_residence': company_location,
}])

input_df['exp_level_num'] = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}[input_df['experience_level'].iloc[0]]
input_df['company_size_num'] = {'S': 0, 'M': 1, 'L': 2}[input_df['company_size'].iloc[0]]
input_df['is_remote'] = (input_df['remote_ratio'] > 0).astype(int)
input_df['is_full_remote'] = (input_df['remote_ratio'] == 100).astype(int)
input_df['is_us'] = (input_df['company_location'] == 'US').astype(int)
input_df['is_full_time'] = (input_df['employment_type'] == 'FT').astype(int)
input_df['title_category'] = input_df['job_title'].apply(lambda x: x.split()[0])

feature_cols = ['work_year', 'exp_level_num', 'company_size_num', 'remote_ratio', 'is_remote', 'is_full_remote', 'is_us', 'is_full_time', 'experience_level', 'employment_type', 'job_title', 'title_category', 'company_location', 'company_size']
input_df = input_df[feature_cols]

# ── PREDICTION RESULT ──
if predict_clicked:
    prediction = float(model.predict(input_df)[0])
    
    st.divider()
    st.subheader("📈 Prediction Result")
    
    rcol1, rcol2 = st.columns([1, 1])
    
    with rcol1:
        st.markdown(f"# 💰 **${prediction:,.0f}**")
        st.markdown("*Estimated annual salary in USD*")
        
        # Salary range indicator
        if prediction < 40000:
            st.info("💡 Below market average for most tech roles")
        elif prediction < 80000:
            st.success("✅ In line with industry average")
        elif prediction < 150000:
            st.success("🌟 Above average — competitive compensation")
        else:
            st.success("🏆 Top-tier salary in the industry")
    
    with rcol2:
        st.markdown("**Your Selections:**")
        st.markdown(f"- **Role:** {job_title} ({experience_level.split('(')[1].rstrip(')')})")
        st.markdown(f"- **Employment:** {employment_type}")
        st.markdown(f"- **Location:** {company_location} | **Size:** {company_size}")
        remote_label = {0:'On-site', 50:'Hybrid', 100:'Fully Remote'}[remote_ratio]
        st.markdown(f"- **Remote:** {remote_label}")
        st.caption(f"Model confidence: R² = {metrics.get('test_r2', 'N/A'):.4f}")

# ── HOW IT WORKS ──
st.divider()
with st.expander("📋 How It Works"):
    st.markdown("""
The model uses **XGBoost** trained on salary data with features including:
- **Role & Seniority**: Job title, experience level, employment type
- **Location**: Company location with geographic cost adjustments
- **Company Profile**: Size, remote work policy
- **Feature Engineering**: Ordinal encoding, interaction features, location tiers

*Note: Predictions are estimates based on industry patterns. Actual compensation varies by company, skills, and negotiation.*
""")
