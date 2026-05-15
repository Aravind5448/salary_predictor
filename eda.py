import pandas as pd
import numpy as np
import json, os

PROJ = r"C:\Users\ARAVIND\Documents\AntiGravity\Salary Predictor\salary_predictor"
df = pd.read_csv(os.path.join(PROJ, 'data', 'raw', 'salaries.csv'))
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\n=== Data Types ===")
print(df.dtypes)
print(f"\n=== Missing Values ===")
print(df.isnull().sum())
print(f"\n=== Basic Stats ===")
print(df[['salary_in_usd']].describe())
print(f"\n=== Work Years ===")
print(df.work_year.value_counts().sort_index())
print(f"\n=== Experience Levels ===")
print(df.experience_level.value_counts())
print(f"\n=== Employment Types ===")
print(df.employment_type.value_counts())
print(f"\n=== Top 20 Job Titles ===")
print(df.job_title.value_counts().head(20))
print(f"\n=== Company Locations (Top 20) ===")
print(df.company_location.value_counts().head(20))
print(f"\n=== Company Sizes ===")
print(df.company_size.value_counts())
print(f"\n=== Remote Ratio ===")
print(df.remote_ratio.value_counts().sort_index())
print(f"\n=== Salary Stats by Experience ===")
print(df.groupby('experience_level')['salary_in_usd'].describe())
print(f"\n=== Salary Stats by Company Size ===")
print(df.groupby('company_size')['salary_in_usd'].describe())
print(f"\n=== Top 15 Job Titles by Count ===")
top_titles = df.job_title.value_counts().head(15).index
for t in top_titles[:10]:
    subset = df[df.job_title == t]
    print(f"{t:35s} n={len(subset):5d}  median=${subset.salary_in_usd.median():>7,.0f}  mean=${subset.salary_in_usd.mean():>8,.0f}")
