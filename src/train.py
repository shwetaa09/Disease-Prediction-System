import sys
sys.path.insert(0, ".")
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.metrics       import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import xgboost as xgb

from src.preprocess import load_and_preprocess

Path('models').mkdir(exist_ok=True)
Path('reports').mkdir(exist_ok=True)

def train_all():
    print("=" * 55)
    print("  Disease Prediction — Model Training")
    print("=" * 55)

    X_tr, X_te, y_tr, y_te, _, feats = \
        load_and_preprocess('data/disease_data.csv')

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42,
            n_jobs=-1, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.1, random_state=42,
            eval_metric='logloss'),
    }

    results = {}
    best_name, best_score, best_model = '', 0, None

    print("\n── Training Models ────────────────────────────")
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        yp  = m.predict(X_te)
        ypr = m.predict_proba(X_te)[:, 1]
        metrics = {
            'accuracy':  round(accuracy_score(y_te, yp),   4),
            'precision': round(precision_score(y_te, yp,
                               zero_division=0),            4),
            'recall':    round(recall_score(y_te, yp,
                               zero_division=0),            4),
            'f1':        round(f1_score(y_te, yp,
                               zero_division=0),            4),
            'roc_auc':   round(roc_auc_score(y_te, ypr),   4),
        }
        results[name] = metrics
        print(f"  {name}: ACC={metrics['accuracy']} "
              f"AUC={metrics['roc_auc']}")

        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_name  = name
            best_model = m

    # Save best model
    joblib.dump(best_model, 'models/disease_model.pkl')
    meta = {
        'best_model':    best_name,
        'metrics':       results[best_name],
        'feature_names': feats,
    }
    json.dump(meta, open('models/model_metadata.json','w'), indent=2)

    print(f"\n  Best Model: {best_name} "
          f"(ROC-AUC={best_score:.4f})")
    print("  Saved -> models/disease_model.pkl")

    # Comparison chart
    df_r = pd.DataFrame(results).T
    df_r.to_csv('reports/model_comparison.csv')

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_r))
    w = 0.15
    colors = ['#4361EE','#7209B7','#F72585','#4CC9F0','#3A0CA3']
    for i, col in enumerate(df_r.columns):
        ax.bar(x + i*w, df_r[col], w,
               label=col.upper(), color=colors[i], alpha=0.85)
    ax.set_xticks(x + w*2)
    ax.set_xticklabels(df_r.index, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title('Model Comparison — Disease Prediction',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig('reports/model_comparison.png', dpi=150)
    plt.close()
    print("  Chart -> reports/model_comparison.png")
    print("\nTraining Complete!")

    return results, best_name

if __name__ == "__main__":
    train_all()