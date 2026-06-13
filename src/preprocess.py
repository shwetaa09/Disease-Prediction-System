import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

NUMERIC_COLS = [
    'age', 'bmi', 'blood_pressure', 'cholesterol',
    'glucose', 'insulin',
    'bmi_bp_ratio', 'glucose_insulin_ratio', 'age_risk_score',
]
ORDINAL_COLS = ['exercise', 'diet']
BINARY_COLS  = ['smoking', 'alcohol', 'family_history', 'chest_pain', 'fatigue']
NOMINAL_COLS = ['gender']
TARGET_COL   = 'disease'

def engineer_features(df):
    df = df.copy()
    df['bmi_bp_ratio']          = df['bmi'] / (df['blood_pressure'] + 1)
    df['glucose_insulin_ratio'] = df['glucose'] / (df['insulin'] + 1)
    df['age_risk_score']        = df['age'] * df['bmi'] / 100
    return df

def load_and_preprocess(
    path='data/disease_data.csv',
    test_size=0.20,
    random_state=42,
    save_preprocessor=True,
):
    df = pd.read_csv(path)
    print(f"[preprocess] Loaded {df.shape[0]:,} rows x {df.shape[1]} cols")

    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])
    ord_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=[
                ['None','Light','Moderate','Heavy'],
                ['Poor','Average','Good'],
            ],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
    ])
    nom_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
    ])
    bin_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUMERIC_COLS),
        ('ord', ord_pipe, ORDINAL_COLS),
        ('bin', bin_pipe, BINARY_COLS),
        ('nom', nom_pipe, NOMINAL_COLS),
    ], remainder='drop')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
    print(f"[preprocess] Train: {len(X_train):,} | Test: {len(X_test):,}")

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    feature_names = NUMERIC_COLS + ORDINAL_COLS + BINARY_COLS + NOMINAL_COLS

    if save_preprocessor:
        Path('models').mkdir(exist_ok=True)
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        print("[preprocess] Preprocessor saved -> models/preprocessor.pkl")

    return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names

if __name__ == "__main__":
    load_and_preprocess()