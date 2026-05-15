import kagglehub
import os, shutil

print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("adilshamim8/salaries-for-data-science-jobs")
print(f"Downloaded to: {path}")

files = os.listdir(path)
print(f"Files: {files}")

for f in files:
    full = os.path.join(path, f)
    print(f"  {f} ({os.path.getsize(full)} bytes)")

# Copy to our project
target = r"C:\Users\ARAVIND\Documents\AntiGravity\Salary Predictor\salary_predictor\data\raw"
os.makedirs(target, exist_ok=True)
for f in files:
    shutil.copy2(os.path.join(path, f), os.path.join(target, f))
    print(f"Copied {f} to data/raw/")

# Show first few lines
for f in files:
    if f.endswith('.csv'):
        with open(os.path.join(target, f)) as fh:
            lines = fh.readlines()
            print(f"\n{f} - {len(lines)} rows")
            print(f"Headers: {lines[0].strip()}")
            print(f"Row 1: {lines[1].strip()}")
            print(f"Row 2: {lines[2].strip()}")
