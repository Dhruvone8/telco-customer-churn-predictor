import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────

DROP_COLS = [
    'CustomerID', 'Count', 'Country', 'State', 'City',
    'Zip Code', 'Lat Long', 'Latitude', 'Longitude',
    'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'
]

BINARY_COLS = ['Senior Citizen', 'Partner', 'Dependents',
               'Paperless Billing', 'Phone Service']

SCALE_COLS = ['Tenure Months', 'Monthly Charges', 'Total Charges']

ADDON_COLS = ['Online Security', 'Online Backup', 'Device Protection',
              'Tech Support', 'Streaming TV', 'Streaming Movies']

OHE_COLS = ['Internet Service', 'Contract', 'Payment Method']

ORDINAL_MAP = {'No internet service': 0, 'No': 1, 'Yes': 2}
LINES_MAP   = {'No phone service': 0, 'No': 1, 'Yes': 2}

# Final feature column order — must match X_train exactly
FEATURE_COLS = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents',
    'Tenure Months', 'Phone Service', 'Multiple Lines',
    'Online Security', 'Online Backup', 'Device Protection',
    'Tech Support', 'Streaming TV', 'Streaming Movies',
    'Paperless Billing', 'Monthly Charges', 'Total Charges',
    'AddOn Services',
    'Internet Service DSL', 'Internet Service Fiber optic', 'Internet Service No',
    'Contract Month-to-month', 'Contract One year', 'Contract Two year',
    'Payment Method Bank transfer (automatic)',
    'Payment Method Credit card (automatic)',
    'Payment Method Electronic check',
    'Payment Method Mailed check'
]

def run_full_pipeline(df: pd.DataFrame):
    """
    Runs all preprocessing steps on raw dataframe.
    Returns X, y, scaler.
    """
    df = df.copy()

    # 1. Drop irrelevant columns (ignore missing ones)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 2. Fix TotalCharges and drop NaNs
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df = df.dropna(subset=['Total Charges'])

    # 3. Encode target
    df['Churn Value'] = df['Churn Value'].astype(int)

    # 4. Binary encode
    for col in BINARY_COLS:
        df[col] = (df[col] == 'Yes').astype(int)
    df['Gender'] = (df['Gender'] == 'Male').astype(int)

    # 5. Ordinal encode service columns
    for col in ADDON_COLS:
        df[col] = df[col].map(ORDINAL_MAP)
    df['Multiple Lines'] = df['Multiple Lines'].map(LINES_MAP)

    # 6. Engineer AddOn Services feature
    df['AddOn Services'] = df[ADDON_COLS].apply(
        lambda row: (row == 2).sum(), axis=1
    )

    # 7. One-hot encode
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=False, dtype=int)
    df.columns = df.columns.str.replace('_', ' ')

    # 8. Scale numeric columns
    scaler = StandardScaler()
    df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])

    # 9. Split X, y
    X = df.drop(columns=['Churn Value'])[FEATURE_COLS]
    y = df['Churn Value']

    return X, y, scaler


# ── Single-row transform (for Streamlit app) ─────────────────

def preprocess_single(row_dict: dict, scaler: StandardScaler) -> pd.DataFrame:
    """
    Transforms a single customer input dict into model-ready DataFrame.
    row_dict keys must match raw input field names.
    scaler must be the fitted scaler from training.
    """
    binary_map  = {"Yes": 1, "No": 0}
    gender_map  = {"Male": 1, "Female": 0}

    addon_services = [
        row_dict.get('Online Security', 'No'),
        row_dict.get('Online Backup', 'No'),
        row_dict.get('Device Protection', 'No'),
        row_dict.get('Tech Support', 'No'),
        row_dict.get('Streaming TV', 'No'),
        row_dict.get('Streaming Movies', 'No'),
    ]
    addon_count = sum(1 for s in addon_services if s == "Yes")

    internet  = row_dict.get('Internet Service', 'DSL')
    contract  = row_dict.get('Contract', 'Month-to-month')
    payment   = row_dict.get('Payment Method', 'Electronic check')

    data = {
        'Gender'           : gender_map[row_dict['Gender']],
        'Senior Citizen'   : binary_map[row_dict['Senior Citizen']],
        'Partner'          : binary_map[row_dict['Partner']],
        'Dependents'       : binary_map[row_dict['Dependents']],
        'Tenure Months'    : float(row_dict['Tenure Months']),
        'Phone Service'    : binary_map[row_dict['Phone Service']],
        'Multiple Lines'   : LINES_MAP[row_dict['Multiple Lines']],
        'Online Security'  : ORDINAL_MAP[row_dict['Online Security']],
        'Online Backup'    : ORDINAL_MAP[row_dict['Online Backup']],
        'Device Protection': ORDINAL_MAP[row_dict['Device Protection']],
        'Tech Support'     : ORDINAL_MAP[row_dict['Tech Support']],
        'Streaming TV'     : ORDINAL_MAP[row_dict['Streaming TV']],
        'Streaming Movies' : ORDINAL_MAP[row_dict['Streaming Movies']],
        'Paperless Billing': binary_map[row_dict['Paperless Billing']],
        'Monthly Charges'  : float(row_dict['Monthly Charges']),
        'Total Charges'    : float(row_dict['Total Charges']),
        'AddOn Services'   : addon_count,
        'Internet Service DSL'         : 1 if internet == "DSL" else 0,
        'Internet Service Fiber optic' : 1 if internet == "Fiber optic" else 0,
        'Internet Service No'          : 1 if internet == "No" else 0,
        'Contract Month-to-month'      : 1 if contract == "Month-to-month" else 0,
        'Contract One year'            : 1 if contract == "One year" else 0,
        'Contract Two year'            : 1 if contract == "Two year" else 0,
        'Payment Method Bank transfer (automatic)': 1 if payment == "Bank transfer (automatic)" else 0,
        'Payment Method Credit card (automatic)'  : 1 if payment == "Credit card (automatic)" else 0,
        'Payment Method Electronic check'         : 1 if payment == "Electronic check" else 0,
        'Payment Method Mailed check'             : 1 if payment == "Mailed check" else 0,
    }

    df_input = pd.DataFrame([data], columns=FEATURE_COLS)
    df_input[SCALE_COLS] = scaler.transform(df_input[SCALE_COLS])

    return df_input