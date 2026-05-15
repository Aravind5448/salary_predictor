import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json, os, sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from src.pipeline import group_title, group_location

MODEL_PATH = os.path.join(PROJ, 'models', 'salary_pipeline.joblib')
METRICS_PATH = os.path.join(PROJ, 'models', 'metrics.json')

st.set_page_config(page_title="Salary Predictor", page_icon="📊", layout="wide")

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
st.markdown("Predict expected compensation based on role, experience, location & company factors.")

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Model R²", f"{metrics.get('test_r2', 'N/A'):.4f}", border=True)
with mcol2:
    st.metric("Test MAE", f"${metrics.get('test_mae', 0):,.0f}", border=True)
with mcol3:
    st.metric("Training Data", f"{metrics.get('rows', 0):,} rows", border=True)
with mcol4:
    st.metric("Algorithm", "XGBoost", border=True)

st.divider()
st.subheader("Enter Job Details")

col1, col2 = st.columns(2)
with col1:
    job_title = st.text_input("Job Title", "Data Scientist",
        help="Enter your job title. The model groups similar titles automatically.")
    exp_level = st.selectbox("Experience Level", ["EN (Entry)", "MI (Mid)", "SE (Senior)", "EX (Executive)"])
    emp_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Freelance"])
    work_year = st.slider("Work Year", 2020, 2025, 2025)
with col2:
    company_location = st.selectbox("Company Location", [
        "US", "GB", "CA", "DE", "FR", "IN", "AU", "NL", "ES", "BR",
        "JP", "SG", "AE", "CH", "SE", "DK", "NO", "IE", "IL", "ZA"
    ], help="Country where the company is based.")
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    remote_ratio = st.select_slider("Remote Work", options=[0, 50, 100],
        format_func=lambda x: {0: "🏢 On-site", 50: "🏡 Hybrid", 100: "🌍 Remote"}[x])

predict = st.button("💰 Predict Salary", type="primary", use_container_width=True)

# Build features for prediction
exp_map = {"EN (Entry)": "EN", "MI (Mid)": "MI", "SE (Senior)": "SE", "EX (Executive)": "EX"}
emp_map = {"Full-time": "FT", "Part-time": "PT", "Contract": "CT", "Freelance": "FL"}
size_map = {"Small": "S", "Medium": "M", "Large": "L"}

raw = {
    'work_year': work_year,
    'experience_level': exp_map[exp_level],
    'employment_type': emp_map[emp_type],
    'job_title': job_title,
    'company_location': company_location,
    'company_size': size_map[company_size],
    'remote_ratio': remote_ratio,
}

df = pd.DataFrame([raw])
df['title_group'] = df['job_title'].apply(group_title)
df['region'] = df['company_location'].apply(group_location)
df['exp_level'] = df['experience_level'].map({'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}).fillna(1).astype(int)
df['size_score'] = df['company_size'].map({'S': 1, 'M': 2, 'L': 3}).fillna(2).astype(int)
df['remote_bucket'] = df['remote_ratio'].map({0: 'onsite', 50: 'hybrid', 100: 'remote'})
df['is_us'] = (df['company_location'] == 'US').astype(int)
df['is_ft'] = (df['employment_type'] == 'FT').astype(int)
df['is_senior'] = (df['experience_level'].isin(['SE', 'EX'])).astype(int)
df['is_recent'] = (df['work_year'] >= 2023).astype(int)
for kw, col in [('data', 'title_has_data'), ('engineer', 'title_has_engineer'),
                 ('scientist', 'title_has_scientist'), ('analyst', 'title_has_analyst'),
                 ('manager|director|head|chief|vp', 'title_has_manager')]:
    df[col] = df['job_title'].str.lower().str.contains(kw).astype(int)
df['exp_senior'] = df['exp_level'] * df['is_senior']
df['us_exp'] = df['is_us'] * df['exp_level']
df['size_senior'] = df['size_score'] * df['is_senior']

feature_cols = [
    'work_year', 'exp_level', 'size_score', 'remote_ratio',
    'is_us', 'is_ft', 'is_senior', 'is_recent',
    'title_has_data', 'title_has_engineer', 'title_has_scientist',
    'title_has_analyst', 'title_has_manager',
    'exp_senior', 'us_exp', 'size_senior',
    'experience_level', 'employment_type', 'title_group', 'region', 'remote_bucket', 'company_size'
]
input_df = df[feature_cols]

if predict:
    pred_log = model.predict(input_df)[0]
    pred = np.expm1(pred_log)

    st.divider()
    st.subheader("📈 Prediction Result")

    rcol1, rcol2 = st.columns([1, 1])
    with rcol1:
        st.markdown(f"# 💰 **${pred:,.0f}**")
        st.markdown("*Estimated annual salary in USD*")

        if pred < 50000:
            st.info("💡 Entry-level range for most regions")
        elif pred < 100000:
            st.success("✅ Solid mid-level compensation")
        elif pred < 180000:
            st.success("🌟 Senior-level — above market average")
        else:
            st.success("🏆 Top-tier — competitive with leading companies")

    with rcol2:
        st.markdown("**Your Profile:**")
        est_exp = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
        st.markdown(f"- **Role:** {job_title}")
        st.markdown(f"- **Level:** {est_exp[exp_map[exp_level]]} | **Type:** {employment_type}")
        st.markdown(f"- **Location:** {company_location} | **Size:** {company_size}")
        remote_labels = {0: 'On-site', 50: 'Hybrid', 100: 'Remote'}
        st.markdown(f"- **Remote:** {remote_labels[remote_ratio]}")
        st.caption(f"Model R²: {metrics.get('test_r2', 'N/A')} | MAE: ${metrics.get('test_mae', 0):,.0f}")

st.divider()
with st.expander("📋 About This Model"):
    st.markdown("""
    **Data:** [Kaggle — Salaries for Data Science Jobs](https://www.kaggle.com/datasets/adilshamim8/salaries-for-data-science-jobs)
    - **151,445** salary records from **2020–2025**
    - **422** job titles grouped into **12 categories**
    - **97** countries grouped into **11 regions**

    **Features Used (22 total):**
    - **Role signals**: Title keywords (data, engineer, scientist, analyst, manager)
    - **Seniority**: Experience level + interaction features
    - **Location**: US/Canada premium, regional groupings
    - **Company**: Size score, remote policy
    - **Derived**: US×Experience, Size×Seniority, keyword flags

    **Limitations:** Salary depends on company brand, specific skills, education, city-level location, and negotiation — which this data doesn't capture.
    """)
