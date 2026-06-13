# 🏥 Disease Prediction System



![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)




![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)




![ML](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)




![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)



> ML-powered system to predict disease risk from patient medical data

## 🎯 What It Does
| Result | Meaning |
|--------|---------|
| ✅ HEALTHY | Low disease risk |
| ⚠️ DISEASE LIKELY | High risk — consult doctor |
| 📊 Risk Score | Probability percentage |

## 🧠 Models Used
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 85% | 0.91 |
| Decision Tree | 82% | 0.87 |
| Random Forest | 88% | 0.94 |
| *XGBoost ✅* | *90%* | *0.96* |

## 📸 App Screenshots

### ✅ Healthy Patient


![Healthy](reports/Screenshot.png)



### ⚠️ High Risk Patient


![Risk](reports/Screenshot2.png)

## 🚀 How To Run
```bash
pip install -r requirements.txt
python data/generate_dataset.py
python src/train.py
streamlit run app.py
