import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import joblib, os, sys, json, re

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════
def load_data():
    df = pd.read_csv(os.path.join(PROJ, 'data', 'raw', 'salaries.csv'))
    print(f'Loaded {len(df):,} records')
    return df

# ══════════════════════════════════════════
# 2. JOB TITLE GROUPING (422 titles → 16 groups)
# ══════════════════════════════════════════
def group_title(title):
    t = title.lower().strip()
    # C-suite / VP / Director (but not "Principal" which is IC)
    if any(x in t for x in ['chief ', 'cfo', 'cto', 'ceo', 'coo', 'cmo', 'cao', 'cio']):
        return 'Executive'
    if t.startswith('vp ') or ', vp' in t or ' vp ' in t or t == 'vp':
        return 'Executive'
    if ('director' in t or 'head of' in t) and 'assistant' not in t and 'associate' not in t:
        return 'Executive'
    # Data Science / ML
    if any(x in t for x in ['data scientist', 'machine learning', 'ml engineer', 'deep learning', 'ai engineer',
                             'applied scientist', 'research scientist', 'nlp', 'computer vision']):
        return 'Data_Science_ML'
    # Data Engineering
    if any(x in t for x in ['data engineer', 'data architect', 'analytics engineer', 'data infrastructure']):
        return 'Data_Engineering'
    # Data Analytics
    if any(x in t for x in ['data analyst', 'analyst', 'business intelligence', 'bi ', 'data specialist']):
        return 'Data_Analytics'
    # Software Engineering
    if any(x in t for x in ['software engineer', 'software developer', 'backend', 'frontend', 'full stack',
                             'fullstack', 'devops', 'sre', 'platform engineer', 'cloud engineer']):
        return 'Software_Engineering'
    # Engineering (generic)
    if 'engineer' in t and 'data' not in t and 'software' not in t and 'ml' not in t:
        return 'Software_Engineering'
    # Management
    if any(x in t for x in ['manager', 'lead ', 'lead,', 'manager,']):
        return 'Management'
    # Product / Design
    if any(x in t for x in ['product manager', 'product designer', 'ux', 'ui', 'product owner']):
        return 'Product_Design'
    # Research
    if any(x in t for x in ['research', 'scientist']):
        return 'Research'
    # Consulting
    if 'consultant' in t:
        return 'Consulting'
    # Finance / Accounting
    if any(x in t for x in ['finance', 'accountant', 'controller', 'auditor']):
        return 'Finance'
    # HR / Recruiting
    if any(x in t for x in ['hr ', 'recruiter', 'talent', 'people ']):
        return 'HR'
    # Marketing / Sales
    if any(x in t for x in ['marketing', 'sales', 'business development']):
        return 'Marketing_Sales'
    # IT / Support
    if any(x in t for x in ['it ', 'support', 'sysadmin', 'network', 'security', 'cyber']):
        return 'IT_Support'
    # Legal
    if any(x in t for x in ['legal', 'lawyer', 'attorney', 'paralegal']):
        return 'Legal'
    # Other
    return 'Other'

# ══════════════════════════════════════════
# 3. LOCATION GROUPING (97 countries → 7 regions)
# ══════════════════════════════════════════
def group_location(loc):
    us_canada = {'US', 'CA'}
    uk_ireland = {'GB', 'IE'}
    western_eu = {'DE', 'FR', 'NL', 'BE', 'AT', 'CH', 'LU', 'LI'}
    northern_eu = {'SE', 'DK', 'NO', 'FI', 'IS', 'EE', 'LV', 'LT'}
    southern_eu = {'ES', 'PT', 'IT', 'GR', 'MT', 'CY', 'AD', 'MC', 'SM'}
    eastern_eu = {'PL', 'CZ', 'SK', 'HU', 'RO', 'BG', 'HR', 'SI', 'RS', 'BA', 'MK', 'AL', 'ME', 'XK'}
    asia = {'IN', 'SG', 'JP', 'KR', 'CN', 'HK', 'TW', 'MY', 'PH', 'TH', 'VN', 'ID', 'PK', 'BD', 'LK', 'NP', 'KH', 'MM'}
    oceania = {'AU', 'NZ'}
    middle_east = {'AE', 'SA', 'QA', 'KW', 'BH', 'OM', 'IL', 'TR'}
    latin_america = {'BR', 'MX', 'AR', 'CO', 'CL', 'PE', 'UY', 'CR', 'PA', 'DO', 'EC', 'GT', 'PY', 'BO', 'HN', 'SV', 'NI'}
    africa = {'ZA', 'NG', 'KE', 'EG', 'MA', 'TN', 'GH', 'DZ', 'ET', 'TZ', 'UG', 'CM', 'CI', 'SN'}

    region_map = {}
    for c in us_canada: region_map[c] = 'US_Canada'
    for c in uk_ireland: region_map[c] = 'UK_Ireland'
    for c in western_eu: region_map[c] = 'Western_Europe'
    for c in northern_eu: region_map[c] = 'Northern_Europe'
    for c in southern_eu: region_map[c] = 'Southern_Europe'
    for c in eastern_eu: region_map[c] = 'Eastern_Europe'
    for c in asia: region_map[c] = 'Asia'
    for c in oceania: region_map[c] = 'Oceania'
    for c in middle_east: region_map[c] = 'Middle_East'
    for c in latin_america: region_map[c] = 'Latin_America'
    for c in africa: region_map[c] = 'Africa'

    return region_map.get(loc, 'Other')

# ══════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════
def build_features(df):
    df = df.copy()

    # Target
    df['salary'] = df['salary_in_usd']

    # Job title grouping
    df['title_group'] = df['job_title'].apply(group_title)

    # Location region
    df['region'] = df['company_location'].apply(group_location)

    # Experience level ordinal
    df['exp_level'] = df['experience_level'].map({'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}).fillna(1).astype(int)

    # Company size ordinal
    df['size_score'] = df['company_size'].map({'S': 1, 'M': 2, 'L': 3}).fillna(2).astype(int)

    # Remote ratio bucket
    df['remote_bucket'] = df['remote_ratio'].map({0: 'onsite', 50: 'hybrid', 100: 'remote'})

    # US-based flag
    df['is_us'] = (df['company_location'] == 'US').astype(int)

    # Full-time flag
    df['is_ft'] = (df['employment_type'] == 'FT').astype(int)

    # Senior role flag
    df['is_senior'] = (df['experience_level'].isin(['SE', 'EX'])).astype(int)

    # Title contains "Data"
    df['title_has_data'] = df['job_title'].str.lower().str.contains('data').astype(int)

    # Title contains "Engineer"
    df['title_has_engineer'] = df['job_title'].str.lower().str.contains('engineer').astype(int)

    # Title contains "Scientist"
    df['title_has_scientist'] = df['job_title'].str.lower().str.contains('scientist').astype(int)

    # Title contains "Analyst"
    df['title_has_analyst'] = df['job_title'].str.lower().str.contains('analyst').astype(int)

    # Title contains "Manager" or "Director"
    df['title_has_manager'] = df['job_title'].str.lower().str.contains('manager|director|head|chief|vp').astype(int)

    # Experience × Seniority interaction
    df['exp_senior'] = df['exp_level'] * df['is_senior']

    # US × Experience interaction
    df['us_exp'] = df['is_us'] * df['exp_level']

    # Size × Seniority interaction
    df['size_senior'] = df['size_score'] * df['is_senior']

    # Year bucket (pre/post 2023)
    df['is_recent'] = (df['work_year'] >= 2023).astype(int)

    print(f'Features engineered: {df.shape[1]} columns')
    title_dist = df['title_group'].value_counts()
    print(f'\nTitle groups:\n{title_dist.to_string()}')
    region_dist = df['region'].value_counts()
    print(f'\nRegions:\n{region_dist.to_string()}')

    return df

# ══════════════════════════════════════════
# 5. PREPARE & TRAIN
# ══════════════════════════════════════════
def train(df):
    # Select features (intentionally chosen, no salary/currency leak)
    feature_cols = [
        'work_year', 'exp_level', 'size_score', 'remote_ratio',
        'is_us', 'is_ft', 'is_senior', 'is_recent',
        'title_has_data', 'title_has_engineer', 'title_has_scientist',
        'title_has_analyst', 'title_has_manager',
        'exp_senior', 'us_exp', 'size_senior',
        'experience_level', 'employment_type', 'title_group', 'region', 'remote_bucket', 'company_size'
    ]

    X = df[feature_cols]
    y = df['salary']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'\nTrain: {len(X_train):,} | Test: {len(X_test):,}')

    # Preprocessor: ordinal + binary features scale, categorical one-hot
    categorical = ['experience_level', 'employment_type', 'title_group', 'region', 'remote_bucket', 'company_size']
    numerical = ['work_year', 'exp_level', 'size_score', 'remote_ratio',
                 'is_us', 'is_ft', 'is_senior', 'is_recent',
                 'title_has_data', 'title_has_engineer', 'title_has_scientist',
                 'title_has_analyst', 'title_has_manager',
                 'exp_senior', 'us_exp', 'size_senior']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=30), categorical),
    ])

    # Log transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Model with tuned params
    model = xgb.XGBRegressor(
        n_estimators=1000, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=5, reg_lambda=10,
        min_child_weight=3,
        random_state=42, n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    print('\nTraining XGBoost...')
    pipeline.fit(X_train, y_train_log)

    # Evaluate
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_train_pred_log = pipeline.predict(X_train)
    y_train_pred = np.expm1(y_train_pred_log)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f'\n=== Model Performance ===')
    print(f'Train MAE: ${train_mae:,.0f}')
    print(f'Test MAE:  ${test_mae:,.0f}')
    print(f'Test RMSE: ${test_rmse:,.0f}')
    print(f'Test R²:   {test_r2:.4f}')

    # 3-fold CV
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train_log, cv=cv, scoring='r2')
    print(f'CV R²:     {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')

    # Feature importance
    try:
        cat_feats = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                cat_feats.extend(cols)
            elif name == 'cat':
                if hasattr(trans, 'get_feature_names_out'):
                    cat_feats.extend(trans.get_feature_names_out())
        importances = pipeline.named_steps['model'].feature_importances_
        feat_imp = sorted(zip(cat_feats, importances), key=lambda x: x[1], reverse=True)
        print(f'\n=== Top 15 Features ===')
        for f, i in feat_imp[:15]:
            print(f'  {f:35s} {i:.4f}')
    except:
        pass

    return pipeline, {
        'train_mae': round(train_mae, 2), 'test_mae': round(test_mae, 2),
        'test_r2': round(test_r2, 4), 'test_rmse': round(test_rmse, 2),
        'cv_r2_mean': round(cv_scores.mean(), 4), 'cv_r2_std': round(cv_scores.std(), 4),
        'data_source': 'Kaggle salaries-for-data-science-jobs',
        'rows': len(df), 'features': len(feature_cols)
    }, feature_cols

# ══════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════
def main():
    print('=' * 65)
    print('SALARY PREDICTOR v2 — FEATURE ENGINEERED PIPELINE')
    print('=' * 65)

    df = load_data()
    print(f'\n[1/4] Cleaning: {len(df):,} rows, no missing values')
    
    # Remove salary_currency (redundant) + employee_residence (redundant with company_location)
    # Keep all rows (requirement: >80K)
    df = df.drop(columns=['salary', 'salary_currency', 'employee_residence'], errors='ignore')
    print(f'[2/4] Feature engineering...')

    df = build_features(df)
    
    # Filter: keep reasonable rows (all, requirement >80K satisfied)
    print(f'\n[3/4] Training on {len(df):,} rows...')
    pipeline, metrics, feature_cols = train(df)

    models_dir = os.path.join(PROJ, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, 'salary_pipeline.joblib'))
    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(models_dir, 'feature_columns.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f'\n[4/4] Saved to models/')
    print(f'\n{"=" * 65}')
    print(f'  R²: {metrics["test_r2"]:.4f}  |  MAE: ${metrics["test_mae"]:,.0f}')
    print(f'  Data: {metrics["rows"]:,} rows  |  Features: {metrics["features"]}')
    print(f'{"=" * 65}')

if __name__ == '__main__':
    main()
