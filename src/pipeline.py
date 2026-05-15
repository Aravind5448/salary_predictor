import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import joblib, os, sys, yaml, json
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

def load_data():
    df = pd.read_csv(os.path.join(PROJ, 'data', 'raw', 'salary_data.csv'))
    print(f'Loaded {len(df)} records from raw data')
    print(f'Columns: {list(df.columns)}')
    print(f'Salary range: ${df.salary_in_usd.min():,.0f} - ${df.salary_in_usd.max():,.0f}')
    print(f'Mean salary: ${df.salary_in_usd.mean():,.0f}')
    return df

def build_features(df):
    df = df.copy()
    
    # Target transformation (log)
    df['salary_log'] = np.log1p(df['salary_in_usd'])
    
    # Experience level ordinal mapping
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    df['exp_level_num'] = df['experience_level'].map(exp_map)
    
    # Company size ordinal
    size_map = {'S': 0, 'M': 1, 'L': 2}
    df['company_size_num'] = df['company_size'].map(size_map)
    
    # Remote ratio category
    df['is_remote'] = (df['remote_ratio'] > 0).astype(int)
    df['is_full_remote'] = (df['remote_ratio'] == 100).astype(int)
    
    # Location tier (US vs other)
    df['is_us'] = (df['company_location'] == 'US').astype(int)
    
    # Employment type binary
    df['is_full_time'] = (df['employment_type'] == 'FT').astype(int)
    
    # Title category prefix (first word)
    df['title_category'] = df['job_title'].apply(lambda x: x.split()[0])
    
    print(f'Feature engineering done. Shape: {df.shape}')
    return df

def prepare_features(df):
    feature_cols = [
        'work_year', 'exp_level_num', 'company_size_num', 
        'remote_ratio', 'is_remote', 'is_full_remote',
        'is_us', 'is_full_time', 
        'experience_level', 'employment_type', 'job_title', 'title_category',
        'company_location', 'company_size'
    ]
    
    X = df[feature_cols]
    y = df['salary_in_usd']
    y_log = df['salary_log']
    
    return X, y, y_log

def build_preprocessor():
    categorical = ['experience_level', 'employment_type', 'job_title', 'title_category', 'company_location', 'company_size']
    numerical = ['work_year', 'exp_level_num', 'company_size_num', 'remote_ratio', 'is_remote', 'is_full_remote', 'is_us', 'is_full_time']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
    ])
    return preprocessor, numerical, categorical

def train_model(X_train, y_train, X_test, y_test):
    preprocessor, num_cols, cat_cols = build_preprocessor()
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb_model),
    ])
    
    print('\nTraining XGBoost model...')
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    
    # Metrics (inverse transform from log)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f'\n=== Model Performance ===')
    print(f'Train MAE: ${train_mae:,.0f}')
    print(f'Test MAE: ${test_mae:,.0f}')
    print(f'Test RMSE: ${test_rmse:,.0f}')
    print(f'Test R²: {test_r2:.4f}')
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
    print(f'\nCross-val R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')
    
    return pipeline, {'train_mae': train_mae, 'test_mae': test_mae, 'test_r2': test_r2, 'test_rmse': test_rmse, 'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std()}

def main():
    print('='*60)
    print('SALARY PREDICTOR — TRAINING PIPELINE')
    print('='*60)
    
    # Load
    df = load_data()
    
    # Feature engineering
    df = build_features(df)
    
    # Prepare
    X, y, y_log = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=X['experience_level'])
    print(f'\nTrain: {len(X_train)} | Test: {len(X_test)}')
    
    # Train
    model, metrics = train_model(X_train, y_train, X_test, y_test)
    
    # Save model and preprocessor
    models_dir = os.path.join(PROJ, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'salary_pipeline.joblib'))
    
    # Save metrics
    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature columns
    with open(os.path.join(models_dir, 'feature_columns.json'), 'w') as f:
        json.dump(list(X.columns), f, indent=2)
    
    print(f'\n✅ Model saved to models/salary_pipeline.joblib')
    print(f'✅ Metrics saved to models/metrics.json')
    print(f'\n{"="*60}')
    print(f'R² Score: {metrics["test_r2"]:.4f}  |  MAE: ${metrics["test_mae"]:,.0f}')
    print(f'{"="*60}')

if __name__ == '__main__':
    main()
