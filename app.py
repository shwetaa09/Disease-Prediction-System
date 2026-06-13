import sys
sys.path.insert(0, ".")
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from pathlib import Path
from src.predict import predict_single, load_artifacts, engineer_features

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #0E1117; }
[data-testid="stSidebar"] { background-color: #1A1D27; }
.result-positive {
    background: linear-gradient(135deg,#2b0d0d,#3a0f0f);
    border: 2px solid #f44336;
    border-radius: 14px; padding: 24px; text-align: center;
}
.result-negative {
    background: linear-gradient(135deg,#0d2b1a,#0f3a22);
    border: 2px solid #00c853;
    border-radius: 14px; padding: 24px; text-align: center;
}
.stButton > button {
    background: linear-gradient(135deg,#7209b7,#f72585);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 2.5rem; font-size: 1rem;
    font-weight: 700; width: 100%;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model():
    return load_artifacts()

def gauge_chart(prob):
    fig, ax = plt.subplots(
        figsize=(5, 3),
        subplot_kw=dict(aspect="equal"),
        facecolor="none"
    )
    ax.set_facecolor("none")
    theta = np.linspace(np.pi, 0, 300)
    r_out, r_in = 1.0, 0.60
    zones = [
        (0.0, 0.45, "#00c853"),
        (0.45, 0.70, "#ff9800"),
        (0.70, 1.0, "#f44336"),
    ]
    for lo, hi, color in zones:
        t = np.linspace(np.pi*(1-lo), np.pi*(1-hi), 100)
        for i in range(len(t)-1):
            ax.fill(
                [0, r_out*np.cos(t[i]), r_out*np.cos(t[i+1])],
                [0, r_out*np.sin(t[i]), r_out*np.sin(t[i+1])],
                color=color, alpha=0.75, zorder=2,
            )
    circle = plt.Circle((0,0), r_in, color="#0E1117", zorder=3)
    ax.add_patch(circle)
    angle = np.pi*(1-prob)
    ax.annotate("",
        xy=(0.85*np.cos(angle), 0.85*np.sin(angle)),
        xytext=(0,0),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5),
        zorder=5,
    )
    color = "#f44336" if prob >= 0.70 else \
            "#ff9800" if prob >= 0.45 else "#00c853"
    ax.text(0, 0.08, f"{prob*100:.1f}%",
            ha="center", va="center",
            fontsize=18, fontweight="bold", color="white", zorder=6)
    ax.text(0, -0.18, "DISEASE RISK",
            ha="center", va="center",
            fontsize=7, color="#9aa0b4", zorder=6)
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-0.3,1.1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

def importance_chart(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.Series(model.feature_importances_,
                    index=feature_names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7,5), facecolor="#12151f")
    ax.set_facecolor("#12151f")
    colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(imp)))
    bars = ax.barh(imp.index, imp.values,
                   color=colors, edgecolor="none", height=0.65)
    for bar, val in zip(bars, imp.values):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center",
                fontsize=9, color="#c8cce0")
    ax.set_title("Feature Importances",
                 color="#e8eaf6", fontsize=13,
                 fontweight="bold", pad=10)
    ax.tick_params(colors="#9aa0b4", labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    plt.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────────
def sidebar_inputs():
    st.sidebar.markdown("## 🏥 Patient Details")
    st.sidebar.markdown("---")

    st.sidebar.markdown("*👤 Personal*")
    age    = st.sidebar.slider("Age", 20, 80, 45)
    gender = st.sidebar.selectbox("Gender", ["Male","Female"])
    bmi    = st.sidebar.slider("BMI", 15.0, 55.0, 27.0, 0.1)

    st.sidebar.markdown("*🩺 Clinical*")
    bp    = st.sidebar.slider("Blood Pressure (mmHg)", 80, 200, 120)
    chol  = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 200)
    gluc  = st.sidebar.slider("Glucose (mg/dL)", 60, 400, 100)
    ins   = st.sidebar.slider("Insulin (μU/mL)", 0, 500, 80)

    st.sidebar.markdown("*🚬 Lifestyle*")
    smoking  = st.sidebar.selectbox("Smoking", ["No","Yes"])
    alcohol  = st.sidebar.selectbox("Alcohol", ["No","Yes"])
    exercise = st.sidebar.selectbox("Exercise",
                   ["None","Light","Moderate","Heavy"])
    diet     = st.sidebar.selectbox("Diet Quality",
                   ["Poor","Average","Good"])

    st.sidebar.markdown("*🧬 Medical History*")
    family  = st.sidebar.selectbox("Family History", ["No","Yes"])
    chest   = st.sidebar.selectbox("Chest Pain", ["No","Yes"])
    fatigue = st.sidebar.selectbox("Fatigue", ["No","Yes"])

    return {
        'age':            age,
        'gender':         gender,
        'bmi':            bmi,
        'blood_pressure': bp,
        'cholesterol':    chol,
        'glucose':        gluc,
        'insulin':        ins,
        'smoking':        1 if smoking == "Yes" else 0,
        'alcohol':        1 if alcohol == "Yes" else 0,
        'family_history': 1 if family  == "Yes" else 0,
        'chest_pain':     1 if chest   == "Yes" else 0,
        'fatigue':        1 if fatigue == "Yes" else 0,
        'exercise':       exercise,
        'diet':           diet,
    }

# ── Main App ──────────────────────────────────────────────
def main():
    col1, col2 = st.columns([1,8])
    with col1:
        st.markdown("<h1 style='font-size:3rem;margin:0'>🏥</h1>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='margin:0;font-size:2rem'>"
                    "Disease Prediction System</h1>",
                    unsafe_allow_html=True)
        st.markdown("<p style='color:#9aa0b4;margin-top:4px'>"
                    "ML-powered disease risk assessment</p>",
                    unsafe_allow_html=True)
    st.markdown("---")

    try:
        model, preprocessor, metadata = get_model()
    except FileNotFoundError:
        st.error("Model not found! Run: python src/train.py")
        st.stop()

    best_name     = metadata.get('best_model', 'Model')
    metrics       = metadata.get('metrics', {})
    feature_names = metadata.get('feature_names', [])

    with st.expander("🔬 Active Model Details", expanded=False):
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, (k,l) in zip(
            [c1,c2,c3,c4,c5],
            [('accuracy','Accuracy'),('precision','Precision'),
             ('recall','Recall'),('f1','F1-Score'),
             ('roc_auc','ROC-AUC')]
        ):
            col.metric(l, f"{metrics.get(k,0):.4f}")
        st.caption(f"*Algorithm:* {best_name}")

    patient = sidebar_inputs()

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("🔍  Predict Disease Risk")

    if btn:
        with st.spinner("Analysing patient data..."):
            result = predict_single(patient)

        prob  = result['probability']
        label = result['label']
        risk  = result['risk']

        if label == "Disease Likely":
            st.markdown(f"""
            <div class="result-positive">
                <h2 style="color:#f44336;font-size:2rem;margin:0">
                ⚠️ DISEASE LIKELY</h2>
                <p style="color:#ffcdd2;font-size:1.1rem;margin-top:8px">
                Risk Level: <b>{risk}</b> &nbsp;|&nbsp;
                Consult a doctor immediately
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <h2 style="color:#00c853;font-size:2rem;margin:0">
                ✅ HEALTHY</h2>
                <p style="color:#b9f6ca;font-size:1.1rem;margin-top:8px">
                Risk Level: <b>{risk}</b> &nbsp;|&nbsp;
                Keep maintaining healthy lifestyle
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns([1,1.4])
        with left:
            st.markdown("#### 📊 Risk Gauge")
            st.pyplot(gauge_chart(prob), use_container_width=True)

        with right:
            st.markdown("#### 🎯 Feature Importances")
            fig = importance_chart(model, feature_names)
            if fig:
                st.pyplot(fig, use_container_width=True)

        st.markdown("#### 📋 Patient Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            basic = {k:v for k,v in patient.items()
                     if k in ['age','gender','bmi',
                               'blood_pressure','cholesterol',
                               'glucose','insulin']}
            st.table(pd.DataFrame.from_dict(
                basic, orient='index', columns=['Value']))
        with col_b:
            lifestyle = {
                'Smoking':        'Yes' if patient['smoking'] else 'No',
                'Alcohol':        'Yes' if patient['alcohol'] else 'No',
                'Family History': 'Yes' if patient['family_history'] else 'No',
                'Chest Pain':     'Yes' if patient['chest_pain'] else 'No',
                'Fatigue':        'Yes' if patient['fatigue'] else 'No',
                'Exercise':       patient['exercise'],
                'Diet':           patient['diet'],
            }
            st.table(pd.DataFrame.from_dict(
                lifestyle, orient='index', columns=['Value']))
    else:
        st.info("👈 Fill patient details and click Predict Disease Risk")
        if Path('data/disease_data.csv').exists():
            df = pd.read_csv('data/disease_data.csv')
            st.markdown("#### 📈 Dataset Overview")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("Disease Cases", f"{df.disease.sum():,}")
            c3.metric("Disease Rate",  f"{df.disease.mean()*100:.1f}%")
            c4.metric("Features",      str(df.shape[1]-1))

if __name__ == "__main__":
    main()