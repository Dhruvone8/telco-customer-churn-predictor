# 📡 Telco Customer Churn Predictor

A machine learning web app that predicts whether a telecom customer
will churn and explains **why** using SHAP explainability.

🔗 **[Live Demo](https://huggingface.co/spaces/DhruvHF/Telcom-Churn-Predictor)**

---

## 📊 Project Overview

| Item | Detail |
|---|---|
| Dataset | IBM Telco Customer Churn (7,043 customers) |
| Model | XGBoost Classifier |
| Explainability | SHAP TreeExplainer |
| Frontend | Streamlit |
| Deployment | Hugging Face Spaces |

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.8493 |
| Churn Recall | 81% |
| Churn Precision | 50% |
| Accuracy | 74% |

---

## 🔍 Key Findings from SHAP Analysis

1. **Contract type** is the strongest churn predictor — month-to-month
   customers are at highest risk
2. **Having dependents** is the most protective factor — families
   are far more loyal customers
3. **Short tenure** customers are significantly more likely to churn
4. **Fiber optic** internet customers churn more despite paying premium prices
5. **Practical services** (Security, Tech Support) retain customers far
   better than entertainment add-ons (Streaming TV/Movies)

---

## 🏗️ Project Structure

```
├── app/
│   └── app.py              # Streamlit app — UI only
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Feature engineering & encoding
│   ├── train.py            # XGBoost training pipeline
│   └── explain.py          # SHAP explainability layer
├── models/
│   ├── xgb_churn_model.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
├── notebooks/
│   └── EDA___Training.ipynb
└── requirements.txt
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Dhruvone8/telco-customer-churn-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)

---

## 💡 What Makes This Project Different

- **SHAP explanations translated to plain English** — business
  stakeholders can understand predictions without ML knowledge
- **Separation of concerns** — preprocessing, training, and
  explainability are fully modular in `src/`
- **Honest evaluation** — GridSearch with 540 fits confirmed the
  baseline model was already near-optimal, demonstrating that
  more complexity doesn't always yield better results