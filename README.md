# 💰 AI Salary Predictor

Predict annual salary (USD) for data science & tech roles based on job title, experience, location, and company details. Built with XGBoost + Streamlit.

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.286 |
| MAE | ~$44,400 |
| Training Data | 151,445 records (2020–2025) |
| Features | 131 (engineered) |
| Algorithm | XGBoost (hist) |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

### Train from scratch

```bash
python src/pipeline.py
```

## Data

[Kaggle — Salaries for Data Science Jobs](https://www.kaggle.com/datasets/adilshamim8/salaries-for-data-science-jobs)
- 151,445 records from 2020–2025
- 95+ countries (89.5% US)
- 422 job titles grouped into 13 categories
- Features: work_year, experience_level, employment_type, job_title, salary, remote_ratio, company_size, company_location

## Limitations

- **89.5% US data** — predictions for other regions have wider variance
- Only 4 experience levels (no granular years-of-experience)
- No company name, industry, skills, education, or city-level detail
- The dataset ceiling caps R² around 0.29 regardless of model choice

## Structure

```
salary_predictor/
├── app/main.py              ← Streamlit UI
├── src/pipeline.py          ← XGBoost training pipeline
├── models/                  ← Trained model + artifacts
│   ├── xgboost_salary.joblib
│   ├── encoded_columns.json
│   └── metrics.json
├── data/raw/salaries.csv    ← Dataset
├── configs/                 ← YAML configs
├── requirements.txt
└── README.md
```

## Contact

**Aravind D** — [GitHub](https://github.com/Aravind5448)
