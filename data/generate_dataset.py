import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)
N = 5000

# Patient features
age = np.random.randint(20, 80, N)
gender = np.random.choice(['Male', 'Female'], N)
bmi = np.random.normal(27, 6, N).clip(15, 55)
blood_pressure = np.random.normal(120, 20, N).clip(80, 200)
cholesterol = np.random.normal(200, 40, N).clip(100, 400)
glucose = np.random.normal(100, 30, N).clip(60, 400)
insulin = np.random.normal(80, 50, N).clip(0, 500)
smoking = np.random.choice([0, 1], N, p=[0.7, 0.3])
family_history = np.random.choice([0, 1], N, p=[0.6, 0.4])
exercise = np.random.choice(
    ['None','Light','Moderate','Heavy'],
    N, p=[0.3, 0.3, 0.25, 0.15]
)
diet = np.random.choice(
    ['Poor','Average','Good'],
    N, p=[0.3, 0.4, 0.3]
)
chest_pain = np.random.choice([0, 1], N, p=[0.7, 0.3])
fatigue = np.random.choice([0, 1], N, p=[0.6, 0.4])
alcohol = np.random.choice([0, 1], N, p=[0.65, 0.35])

# Exercise and diet scores
ex_map = {'None': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3}
diet_map = {'Poor': 0, 'Average': 1, 'Good': 2}
ex_score = np.array([ex_map[e] for e in exercise])
diet_score = np.array([diet_map[d] for d in diet])

# Risk score to generate label
risk = (
    0.20 * (age / 80)
    + 0.15 * (bmi / 55)
    + 0.15 * (blood_pressure / 200)
    + 0.10 * (cholesterol / 400)
    + 0.10 * (glucose / 400)
    + 0.08 * smoking
    + 0.08 * family_history
    + 0.06 * chest_pain
    + 0.05 * fatigue
    - 0.05 * (ex_score / 3)
    - 0.04 * (diet_score / 2)
    + np.random.normal(0, 0.05, N)
)

threshold = np.percentile(risk, 60)
disease = (risk > threshold).astype(int)

# Add missing values ~4%
def add_missing(arr, frac=0.04):
    arr = arr.astype(object)
    idx = np.random.choice(len(arr), int(len(arr)*frac), replace=False)
    arr[idx] = np.nan
    return arr

df = pd.DataFrame({
    'age':            age,
    'gender':         gender,
    'bmi':            add_missing(bmi.round(1)),
    'blood_pressure': add_missing(blood_pressure.round(1)),
    'cholesterol':    add_missing(cholesterol.round(1)),
    'glucose':        add_missing(glucose.round(1)),
    'insulin':        add_missing(insulin.round(1)),
    'smoking':        smoking,
    'alcohol':        alcohol,
    'family_history': family_history,
    'chest_pain':     chest_pain,
    'fatigue':        fatigue,
    'exercise':       exercise,
    'diet':           diet,
    'disease':        disease,
})

out = Path("data/disease_data.csv")
df.to_csv(out, index=False)
print(f"Dataset saved -> {out}")
print(f"Rows: {len(df):,} | Disease: {df.disease.sum():,} ({df.disease.mean()*100:.1f}%)")
print(df.head(3).to_string())