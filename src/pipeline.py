"""
XGBoost training pipeline for salary prediction.
Run: python src/pipeline.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib, os, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM = 42

def group_title(t):
    t = t.lower().strip()
    if any(x in t for x in ['chief ', 'cfo', 'cto', 'ceo', 'coo', 'cmo', 'cao', 'cio']): return 'Executive'
    if t.startswith('vp ') or ', vp' in t or ' vp ' in t: return 'Executive'
    if 'director' in t or 'head of' in t: return 'Executive'
    if any(x in t for x in ['data scientist', 'machine learning', 'ml engineer', 'deep learning', 'ai engineer']): return 'Data_Science_ML'
    if any(x in t for x in ['data engineer', 'data architect', 'analytics engineer', 'data infrastructure']): return 'Data_Engineering'
    if any(x in t for x in ['data analyst', 'analyst', 'business intelligence', 'bi ', 'data specialist']): return 'Data_Analytics'
    if any(x in t for x in ['software engineer', 'software developer', 'backend', 'frontend', 'full stack', 'fullstack', 'devops', 'sre', 'platform engineer', 'cloud engineer']): return 'Software_Engineering'
    if 'engineer' in t: return 'Software_Engineering'
    if any(x in t for x in ['manager', 'lead ', 'lead,']): return 'Management'
    if any(x in t for x in ['product manager', 'product designer', 'ux', 'ui', 'product owner']): return 'Product_Design'
    if any(x in t for x in ['research', 'scientist']): return 'Research'
    if 'consultant' in t: return 'Consulting'
    if any(x in t for x in ['finance', 'accountant', 'controller', 'auditor']): return 'Finance'
    if any(x in t for x in ['hr ', 'recruiter', 'talent', 'people ']): return 'HR'
    if any(x in t for x in ['marketing', 'sales', 'business development']): return 'Marketing_Sales'
    return 'Other'

def engineer(df):
    df['title_group'] = df['job_title'].apply(group_title)
    df['exp_level'] = df['experience_level'].map({'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3})
    df['size_score'] = df['company_size'].map({'S': 1, 'M': 2, 'L': 3})
    df['is_us'] = (df['company_location'] == 'US').astype(int)
    df['is_ft'] = (df['employment_type'] == 'FT').astype(int)
    df['is_senior'] = (df['experience_level'].isin(['SE', 'EX'])).astype(int)
    df['is_remote'] = (df['remote_ratio'] > 0).astype(int)
    df['is_full_remote'] = (df['remote_ratio'] == 100).astype(int)
    df['year_from_2020'] = df['work_year'] - 2020
    tl = df['job_title'].str.lower()
    df['kw_data'] = tl.str.contains('data').astype(int)
    df['kw_engineer'] = tl.str.contains('engineer').astype(int)
    df['kw_scientist'] = tl.str.contains('scientist').astype(int)
    df['kw_analyst'] = tl.str.contains('analyst').astype(int)
    df['kw_manager'] = tl.str.contains('manager|director|head|chief|vp').astype(int)
    df['kw_ml'] = tl.str.contains('machine learning|ml|ai|deep learning').astype(int)
    df['kw_software'] = tl.str.contains('software|developer|backend|frontend|full.?stack|devops').astype(int)
    df['title_len'] = df['job_title'].str.len()
    return df

def main():
    print('=' * 55)
    print('SALARY PREDICTOR — Training Pipeline')
    print('=' * 55)

    df = pd.read_csv(os.path.join(PROJ, 'data', 'raw', 'salaries.csv'))
    print(f'Loaded {len(df):,} records')

    df = engineer(df)

    num_feats = ['exp_level', 'size_score', 'year_from_2020', 'remote_ratio',
                 'is_us', 'is_ft', 'is_senior', 'is_remote', 'is_full_remote',
                 'kw_data', 'kw_engineer', 'kw_scientist', 'kw_analyst',
                 'kw_manager', 'kw_ml', 'kw_software', 'title_len']
    cat_feats = ['experience_level', 'employment_type', 'title_group', 'company_size', 'company_location']

    X = pd.get_dummies(df[num_feats + cat_feats], columns=cat_feats, drop_first=True)
    y = df['salary_in_usd']
    print(f'Features: {X.shape[1]}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM)
    print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.07,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=3, reg_lambda=5,
        tree_method='hist', random_state=RANDOM, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, np.log1p(y_train))

    y_pred = np.expm1(model.predict(X_test))
    y_train_pred = np.expm1(model.predict(X_train))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'\n{"="*55}')
    print(f'  Test MAE:  ${mae:,.0f}')
    print(f'  Test R²:   {r2:.4f}')
    print(f'  Best iter: {model.best_iteration}')
    print(f'{"="*55}')

    # Save
    md = os.path.join(PROJ, 'models')
    os.makedirs(md, exist_ok=True)
    joblib.dump(model, os.path.join(md, 'xgboost_salary.joblib'))
    with open(os.path.join(md, 'encoded_columns.json'), 'w') as f:
        json.dump(list(X.columns), f, indent=2)
    with open(os.path.join(md, 'metrics.json'), 'w') as f:
        json.dump({
            'mae': round(mae, 2), 'r2': round(r2, 4),
            'rows': len(df), 'features': X.shape[1], 'model': 'XGBoost'
        }, f, indent=2)
    print(f'\nSaved to models/')

if __name__ == '__main__':
    main()
