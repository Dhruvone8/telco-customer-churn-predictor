import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import matplotlib.pyplot as plt
import joblib

from src.preprocess import preprocess_single, FEATURE_COLS, SCALE_COLS
from src.explain import get_shap_for_input

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load("models/xgb_churn_model.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    return model, scaler, explainer

model, scaler, explainer = load_artifacts()

# ── Human-readable feature name mapping ──────────────────────
FEATURE_LABELS = {
    'Contract Month-to-month'                  : 'Month-to-month contract',
    'Contract Two year'                        : 'Two-year contract',
    'Contract One year'                        : 'One-year contract',
    'Dependents'                               : 'Has dependents',
    'Tenure Months'                            : 'Time as customer',
    'Monthly Charges'                          : 'Monthly bill amount',
    'Total Charges'                            : 'Total amount paid',
    'Internet Service Fiber optic'             : 'Fiber optic internet',
    'Internet Service DSL'                     : 'DSL internet',
    'Internet Service No'                      : 'No internet service',
    'Online Security'                          : 'Online security service',
    'Tech Support'                             : 'Tech support service',
    'Online Backup'                            : 'Online backup service',
    'Device Protection'                        : 'Device protection service',
    'Payment Method Electronic check'          : 'Pays by electronic check',
    'Payment Method Bank transfer (automatic)' : 'Pays by bank transfer',
    'Payment Method Credit card (automatic)'   : 'Pays by credit card',
    'Payment Method Mailed check'              : 'Pays by mailed check',
    'Paperless Billing'                        : 'Paperless billing',
    'Partner'                                  : 'Has a partner',
    'Senior Citizen'                           : 'Senior citizen',
    'Gender'                                   : 'Gender',
    'Phone Service'                            : 'Phone service',
    'Multiple Lines'                           : 'Multiple phone lines',
    'Streaming TV'                             : 'Streaming TV',
    'Streaming Movies'                         : 'Streaming movies',
    'AddOn Services'                           : 'Number of add-on services',
}

# ── Header ────────────────────────────────────────────────────
st.title("📡 Customer Churn Predictor")
st.markdown("Enter customer details to predict churn probability and understand **why**.")
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header("Customer Details")

st.sidebar.subheader("Demographics")
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner        = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents     = st.sidebar.selectbox("Dependents", ["No", "Yes"])

st.sidebar.subheader("Account Info")
tenure          = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
contract        = st.sidebar.selectbox("Contract",
                    ["Month-to-month", "One year", "Two year"])
paperless       = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment         = st.sidebar.selectbox("Payment Method", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges   = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0,
                    float(round(tenure * monthly_charges, 2)))

st.sidebar.subheader("Services")
phone_service    = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines   = st.sidebar.selectbox("Multiple Lines",
                        ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service",
                        ["DSL", "Fiber optic", "No"])

# Dynamically disable addon services if no internet
internet_options = ["No", "Yes", "No internet service"]
disabled         = (internet_service == "No")

online_security   = st.sidebar.selectbox("Online Security",   internet_options,
                        index=2 if disabled else 0)
online_backup     = st.sidebar.selectbox("Online Backup",     internet_options,
                        index=2 if disabled else 0)
device_protection = st.sidebar.selectbox("Device Protection", internet_options,
                        index=2 if disabled else 0)
tech_support      = st.sidebar.selectbox("Tech Support",      internet_options,
                        index=2 if disabled else 0)
streaming_tv      = st.sidebar.selectbox("Streaming TV",      internet_options,
                        index=2 if disabled else 0)
streaming_movies  = st.sidebar.selectbox("Streaming Movies",  internet_options,
                        index=2 if disabled else 0)

# ── Predict button ────────────────────────────────────────────
st.sidebar.divider()
predict_btn = st.sidebar.button("🔍 Predict Churn", use_container_width=True)

# ── Results ───────────────────────────────────────────────────
if predict_btn:

    # 1. Build raw input dict
    row_dict = {
        'Gender'            : gender,
        'Senior Citizen'    : senior_citizen,
        'Partner'           : partner,
        'Dependents'        : dependents,
        'Tenure Months'     : tenure,
        'Phone Service'     : phone_service,
        'Multiple Lines'    : multiple_lines,
        'Internet Service'  : internet_service,
        'Online Security'   : online_security,
        'Online Backup'     : online_backup,
        'Device Protection' : device_protection,
        'Tech Support'      : tech_support,
        'Streaming TV'      : streaming_tv,
        'Streaming Movies'  : streaming_movies,
        'Contract'          : contract,
        'Paperless Billing' : paperless,
        'Payment Method'    : payment,
        'Monthly Charges'   : monthly_charges,
        'Total Charges'     : total_charges,
    }

    # 2. Preprocess — all logic lives in src/preprocess.py
    input_df = preprocess_single(row_dict, scaler)

    # 3. Predict
    prob       = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= 0.5)
    risk       = "🔴 High" if prob > 0.7 else "🟡 Medium" if prob > 0.4 else "🟢 Low"

    # 4. Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{prob:.1%}")
    col2.metric("Prediction", "Will Churn" if prediction else "Will Stay")
    col3.metric("Risk Level", risk)

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Risk gauge + customer profile ─────────────────────────
    with col_left:
        st.subheader("Risk Gauge")
        fig, ax = plt.subplots(figsize=(6, 1.5))
        color = "#ef4444" if prob > 0.5 else "#22c55e"
        ax.barh([""], [prob],     color=color,  height=0.5)
        ax.barh([""], [1 - prob], left=[prob],
                color="#e5e7eb",  height=0.5)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
        ax.text(prob / 2, 0, f"{prob:.1%}",
                ha='center', va='center', color='white', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Churn Probability")
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Customer Profile")
        addon_count = sum(1 for s in [
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies
        ] if s == "Yes")

        st.markdown(f"""
        | Feature | Value |
        |---|---|
        | Contract | `{contract}` |
        | Tenure | `{tenure} months` |
        | Monthly Charges | `${monthly_charges:.2f}` |
        | Internet Service | `{internet_service}` |
        | Add-on Services | `{addon_count} active` |
        | Payment Method | `{payment}` |
        """)

    # ── Human-readable SHAP explanation ───────────────────────
    with col_right:
        st.subheader("What's driving this prediction?")

        # Get SHAP values — logic lives in src/explain.py
        shap_vals = get_shap_for_input(explainer, input_df)[0]

        # Sort features by absolute SHAP impact
        factors = sorted(
            zip(FEATURE_COLS, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Split into risk vs protective
        risk_factors       = [(f, v) for f, v in factors if v > 0.02][:4]
        protective_factors = [(f, v) for f, v in factors if v < -0.02][:4]

        # Risk factors — red cards
        if risk_factors:
            st.markdown("**🔴 Factors increasing churn risk:**")
            for feat, val in risk_factors:
                label     = FEATURE_LABELS.get(feat, feat)
                intensity = min(int(abs(val) / 0.8 * 100), 100)
                impact    = "High" if intensity > 60 else "Medium" if intensity > 30 else "Low"
                st.markdown(f"""
                <div style="
                    background:#fef2f2;
                    border-left:4px solid #ef4444;
                    padding:10px 14px;
                    border-radius:6px;
                    margin-bottom:8px;
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; color:#991b1b;">⚠️ {label}</span>
                        <span style="font-size:0.8em; color:#dc2626;">{impact} impact</span>
                    </div>
                    <div style="background:#fecaca; border-radius:4px; height:6px; margin-top:6px;">
                        <div style="background:#ef4444; width:{intensity}%; height:6px; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Protective factors — green cards
        if protective_factors:
            st.markdown("**🟢 Factors reducing churn risk:**")
            for feat, val in protective_factors:
                label     = FEATURE_LABELS.get(feat, feat)
                intensity = min(int(abs(val) / 0.8 * 100), 100)
                impact    = "High" if intensity > 60 else "Medium" if intensity > 30 else "Low"
                st.markdown(f"""
                <div style="
                    background:#f0fdf4;
                    border-left:4px solid #22c55e;
                    padding:10px 14px;
                    border-radius:6px;
                    margin-bottom:8px;
                ">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600; color:#166534;">✅ {label}</span>
                        <span style="font-size:0.8em; color:#16a34a;">{impact} impact</span>
                    </div>
                    <div style="background:#bbf7d0; border-radius:4px; height:6px; margin-top:6px;">
                        <div style="background:#22c55e; width:{intensity}%; height:6px; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Plain English summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📋 Summary**")
        top_risk = FEATURE_LABELS.get(
            risk_factors[0][0], risk_factors[0][0]
        ) if risk_factors else "No major risk factors"
        top_prot = FEATURE_LABELS.get(
            protective_factors[0][0], protective_factors[0][0]
        ) if protective_factors else "No strong protective factors"

        st.info(f"""
        The biggest reason this customer **may churn** is: **{top_risk}**

        The biggest reason they **may stay** is: **{top_prot}**
        """)

    st.divider()

    # ── Business recommendation ────────────────────────────────
    st.subheader("Retention Recommendation")
    if prob > 0.7:
        st.error("""
        **High Risk — Act immediately**
        - Offer a discounted upgrade to a 1-year or 2-year contract
        - Assign a dedicated retention agent for a personal call
        - Provide a free month of Online Security or Tech Support
        - Review pricing vs competitors for Fiber optic plans
        """)
    elif prob > 0.4:
        st.warning("""
        **Medium Risk — Monitor closely**
        - Send a proactive satisfaction survey this week
        - Offer a loyalty discount on the next billing cycle
        - Suggest bundling add-on services at a reduced rate
        """)
    else:
        st.success("""
        **Low Risk — Nurture loyalty**
        - Enroll in loyalty rewards program
        - Upsell premium add-on services
        - Send periodic engagement and appreciation offers
        """)

# ── Default landing page ──────────────────────────────────────
else:
    st.info("Fill in customer details in the sidebar and click **Predict Churn**")
    st.subheader("About this app")
    st.markdown("""
    This app uses an **XGBoost classifier** trained on IBM Telco customer data
    with **SHAP explainability** to show exactly why each prediction was made.

    | Metric | Value |
    |---|---|
    | ROC-AUC Score | 0.8493 |
    | Churn Recall | 81% |
    | Churn Precision | 50% |
    | Total Features | 27 |
    | Explainability | SHAP TreeExplainer |
    | Training Samples | 5,625 |

    **Top churn signals found by SHAP:**
    1. Month-to-month contract
    2. Having dependents (protective)
    3. Short tenure
    4. High monthly charges
    5. Fiber optic internet service
    """)