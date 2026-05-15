# 📊 AI Salary Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![R²](https://img.shields.io/badge/R%C2%B2-0.95-brightgreen)]()
[![MAE](https://img.shields.io/badge/MAE-%248.7K-success)]()

Predict data science and ML salaries with **94.7% accuracy** using an XGBoost regression model. Built with a production-grade ML pipeline — from feature engineering to deployment.

---

## 🎯 Problem Statement

Data science salaries vary enormously based on role, experience, location, company size, and remote policy. Companies and job seekers lack a reliable, data-driven benchmark for compensation expectations.

**This project provides:**
- Accurate salary predictions (R² = 0.947, MAE = $8.7K)
- Interactive web interface for real-time estimation
- Reproducible training pipeline with feature engineering
- Dockerized deployment ready for cloud hosting

---

## 📈 Model Performance

| Metric | Value | Improvement |
|--------|-------|-------------|
| **R² Score** | **0.9469** | Baseline: 0.28 |
| **MAE** | **$8,703** | Baseline: $45,366 |
| **RMSE** | **$13,275** | — |
| **Cross-val R²** | **0.9482 (±0.001)** | 5-fold CV |
| **Training Data** | 25,000 records | 10 job titles, 20 countries |

### Feature Engineering
- Ordinal encoding for experience level & company size
- Binary indicators: remote, US-based, full-time
- Title category extraction (first-word grouping)
- Log transformation for salary normalization
- Geographic cost-of-living adjustments

---

## 🏗️ Architecture

```
salary_predictor/
├── app/
│   └── main.py                 ← Streamlit inference UI
├── configs/
│   ├── app_config.yaml          ← UI configuration
│   ├── data_config.yaml         ← Data schema
│   └── model_config.yaml        ← Model parameters
├── data/
│   └── raw/salary_data.csv      ← Training dataset
├── models/
│   ├── salary_pipeline.joblib   ← Trained XGBoost pipeline
│   ├── metrics.json             ← Performance metrics
│   └── feature_columns.json     ← Feature schema
├── src/
│   ├── pipeline.py              ← End-to-end training pipeline
│   ├── inference/predict.py     ← Batch inference module
│   └── evaluation/
│       ├── metrics.py           ← Evaluation utilities
│       ├── benchmark.py         ← Model comparison
│       └── explainability.py    ← SHAP/interpretability
├── notebooks/                   ← EDA & training notebooks
├── tests/                       ← Unit tests
├── Dockerfile                   ← Containerized deployment
├── requirements.txt             ← Dependencies
└── README.md                    ← This file
```

---

## 🚀 Quick Start

### Local Setup

```bash
# Clone
git clone https://github.com/Aravind5448/salary_predictor.git
cd salary_predictor

# Install
pip install -r requirements.txt

# Train model
python src/pipeline.py

# Run app
streamlit run app/main.py
```

### Docker

```bash
docker build -t salary-predictor .
docker run -p 8501:8501 salary-predictor
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🧪 Model Training

```bash
python src/pipeline.py
```

The pipeline:
1. Loads and validates raw salary data
2. Engineers features (ordinal encoding, binary flags, title categories)
3. Trains an XGBoost regressor with 5-fold cross-validation
4. Evaluates on held-out test set (20%)
5. Saves model artifacts to `models/`

---

## 🐳 Deployment

### Streamlit Cloud (Recommended)
1. Push to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repo → Set entry point to `app/main.py` → Deploy

### Docker
```bash
docker build -t salary-predictor .
docker run -p 8501:8501 salary-predictor
```

### API Mode (Coming Soon)
FastAPI inference endpoint for programmatic access.

---

## 📊 Data

Synthetic dataset (25,000 records) generated from industry salary distributions based on:
- [Kaggle: Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)
- [Levels.fyi](https://www.levels.fyi) compensation data
- Glassdoor salary reports

**Features:** work_year, experience_level, employment_type, job_title, salary_in_usd, employee_residence, remote_ratio, company_location, company_size

---

## 📬 Contact

**Aravind D** — [GitHub](https://github.com/Aravind5448) · [LinkedIn](https://linkedin.com/in/aravind-dhivakaran-6627hp)

---

*Built as part of ML engineering portfolio — demonstrating end-to-end ML pipeline development, feature engineering, model optimization, and production deployment.*
