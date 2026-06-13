import sys
sys.path.insert(0, ".")
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH       = Path("models/disease_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
METADATA_PATH    = Path("models/model_metadata.json")

def engineer_features(df):
    df = df.copy()
    df['bmi_bp_ratio']          = df['bmi'] / (df['blood_pressure'] + 1)
    df['glucose_insulin_ratio'] = df['glucose'] / (df['insulin'] + 1)
    df['age_risk_score']        = df['age'] * df['bmi'] / 100
    return df

def load_artifacts():
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata     = json.load(open(METADATA_PATH))
    return model, preprocessor, metadata

def risk_label(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.45:
        return "Moderate Risk"
    return "Low Risk"

def predict_single(patient: dict) -> dict:
    model, preprocessor, _ = load_artifacts()
    df = pd.DataFrame([patient])
    df = engineer_features(df)
    X  = preprocessor.transform(df)
    prob  = model.predict_proba(X)[0, 1]
    label = "Disease Likely" if prob >= 0.5 else "Healthy"
    return {
        "label":       label,
        "probability": round(float(prob), 4),
        "risk":        risk_label(prob),
    }

if __name__ == "__main__":
    sample = {
        'age': 55, 'gender': 'Male', 'bmi': 32.0,
        'blood_pressure': 145, 'cholesterol': 250,
        'glucose': 180, 'insulin': 60,
        'smoking': 1, 'alcohol': 1,
        'family_history': 1, 'chest_pain': 1,
        'fatigue': 1, 'exercise': 'None', 'diet': 'Poor',
    }
    print(predict_single(sample))