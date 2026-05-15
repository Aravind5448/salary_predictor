import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json, os, sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJ, 'models', 'xgboost_salary.joblib')
METRICS_PATH = os.path.join(PROJ, 'models', 'metrics.json')

st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, metrics

model, metrics = load_model()

st.title("💰 AI Salary Predictor")
st.markdown("Predict expected annual salary (USD) based on role, experience, location & company.")

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Accuracy (R²)", f"{metrics.get('r2', 'N/A'):.4f}", border=True)
with mcol2:
    st.metric("Avg Error (MAE)", f"${metrics.get('mae', 0):,.0f}", border=True)
with mcol3:
    st.metric("Training Data", f"{metrics.get('rows', 0):,} rows", border=True)
with mcol4:
    st.metric("Algorithm", "XGBoost", border=True)

st.divider()
st.subheader("Job Details")

col1, col2 = st.columns(2)
with col1:
    job_title = st.text_input("Job Title", "Data Scientist",
        help="Any title — the model extracts keywords from it.")
    exp_level = st.selectbox("Experience Level", ["EN (Entry)", "MI (Mid)", "SE (Senior)", "EX (Executive)"],
        help="Entry < Mid < Senior < Executive")
    emp_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Freelance"])
    work_year = st.slider("Work Year", 2020, 2025, 2025)
    company_location = st.selectbox("Company Location", [
        "US", "GB", "CA", "DE", "FR", "IN", "AU", "NL", "ES", "BR",
        "JP", "SG", "AE", "CH", "SE", "DK", "NO", "IE", "IL", "ZA",
        "IT", "PT", "KR", "HK", "AT", "BE", "MY", "MX", "AR", "CO",
        "PL", "CZ", "RO", "HU", "GR", "HR", "RU", "TR", "SA", "NG",
        "KE", "EG", "MA", "TN", "Other"
    ], help="Country where the company is based.")
with col2:
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    remote_ratio = st.select_slider("Remote Work", options=[0, 50, 100],
        format_func=lambda x: {0: "🏢 On-site", 50: "🏡 Hybrid", 100: "🌍 Remote"}[x])
    title_group = st.selectbox("Title Category", [
        "Data_Science_ML", "Software_Engineering", "Data_Analytics", "Data_Engineering",
        "Management", "Executive", "Product_Design", "Research",
        "Marketing_Sales", "Consulting", "Finance", "HR", "Other"
    ], help="Broad category your title falls under. Auto-detected from title if left as default.")
    kw_ml = st.checkbox("AI/ML title keywords (ml, ai, deep learning)", True)
    kw_software = st.checkbox("Software title keywords (developer, backend, frontend)", False)

# ── Buttons ──
col_a, col_b = st.columns([1, 1])
with col_a:
    predict = st.button("💰 Predict Salary", type="primary", use_container_width=True)
with col_b:
    auto_title = st.button("🔍 Auto-detect category", use_container_width=True)

# ── Auto-detect title group ──
def auto_group(title):
    t = title.lower().strip()
    if any(x in t for x in ['data scientist','machine learning','ml engineer','deep learning','ai engineer']): return 'Data_Science_ML'
    if any(x in t for x in ['data engineer','data architect']): return 'Data_Engineering'
    if any(x in t for x in ['data analyst','analyst','business intelligence']): return 'Data_Analytics'
    if any(x in t for x in ['software engineer','software developer','backend','frontend','fullstack','devops','sre']): return 'Software_Engineering'
    if 'engineer' in t: return 'Software_Engineering'
    if any(x in t for x in ['manager','director','head','chief','vp']): return 'Management'
    if any(x in t for x in ['chief ','cfo','cto','ceo']): return 'Executive'
    if any(x in t for x in ['product manager','product designer','ux','ui']): return 'Product_Design'
    if any(x in t for x in ['research','scientist']): return 'Research'
    if any(x in t for x in ['marketing','sales']): return 'Marketing_Sales'
    return 'Other'

if auto_title:
    detected = auto_group(job_title)
    title_group = detected
    st.info(f"Detected category: **{detected}**")

# ── Predict ──
if predict:
    # Build feature vector matching training
    tl = job_title.lower()
    row = {
        'exp_level': {'EN (Entry)': 0, 'MI (Mid)': 1, 'SE (Senior)': 2, 'EX (Executive)': 3}[exp_level],
        'size_score': {'Small': 1, 'Medium': 2, 'Large': 3}[company_size],
        'year_from_2020': work_year - 2020,
        'remote_ratio': remote_ratio,
        'is_us': 1 if company_location == 'US' else 0,
        'is_ft': 1 if emp_type == 'Full-time' else 0,
        'is_senior': 1 if exp_level in ['SE (Senior)', 'EX (Executive)'] else 0,
        'is_remote': 1 if remote_ratio > 0 else 0,
        'is_full_remote': 1 if remote_ratio == 100 else 0,
        'kw_data': 1 if 'data' in tl else 0,
        'kw_engineer': 1 if 'engineer' in tl else 0,
        'kw_scientist': 1 if 'scientist' in tl else 0,
        'kw_analyst': 1 if 'analyst' in tl else 0,
        'kw_manager': 1 if any(x in tl for x in ['manager','director','head','chief','vp']) else 0,
        'kw_ml': 1 if kw_ml else 0,
        'kw_software': 1 if kw_software else 0,
        'title_len': len(job_title),
        'experience_level': {'EN (Entry)': 'EN', 'MI (Mid)': 'MI', 'SE (Senior)': 'SE', 'EX (Executive)': 'EX'}[exp_level],
        'employment_type': {'Full-time': 'FT', 'Part-time': 'PT', 'Contract': 'CT', 'Freelance': 'FL'}[emp_type],
        'title_group': title_group,
        'company_size': {'Small': 'S', 'Medium': 'M', 'Large': 'L'}[company_size],
        'company_location': company_location,
    }

    input_df = pd.DataFrame([row])

    # One-hot encode matching training
    cat_cols = ['experience_level', 'employment_type', 'title_group', 'company_size', 'company_location']
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    # Align columns with training using reindex (no fragmentation)
    with open(os.path.join(PROJ, 'models', 'encoded_columns.json')) as f:
        train_cols = json.load(f)

    input_encoded = input_encoded.reindex(columns=train_cols, fill_value=0)

    pred_log = model.predict(input_encoded)[0]
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
        st.markdown(f"- **Role:** {job_title}")
        exp_labels = {'EN (Entry)': 'Entry', 'MI (Mid)': 'Mid', 'SE (Senior)': 'Senior', 'EX (Executive)': 'Executive'}
        st.markdown(f"- **Level:** {exp_labels[exp_level]} | **Type:** {emp_type}")
        st.markdown(f"- **Location:** {company_location} | **Size:** {company_size}")
        st.markdown(f"- **Category:** {title_group}")
        remote_labels = {0: 'On-site', 50: 'Hybrid', 100: 'Remote'}
        st.markdown(f"- **Remote:** {remote_labels[remote_ratio]}")
        st.caption(f"Model Info — R²: {metrics.get('r2', 'N/A')} | MAE: ${metrics.get('mae', 0):,.0f} | {metrics.get('features', 0)} features")

st.divider()
with st.expander("📋 About This Model"):
    st.markdown("""
    **Data:** [Kaggle — Salaries for Data Science Jobs](https://www.kaggle.com/datasets/adilshamim8/salaries-for-data-science-jobs)
    - **151,445** salary records from **2020–2025**, 95+ countries
    - **422** job titles grouped into categories

    **Features (127 total):**
    - Role keywords (data, engineer, scientist, analyst, manager, ml, software)
    - Experience level, company size, remote policy, location encoded
    - Interaction features (US × experience, remote × seniority, etc.)

    **Limitations:** Dataset is 89.5% US — non-US predictions have wider variance.
    Salary depends on company brand, specific skills, education, and negotiation — not captured here.
    """)
