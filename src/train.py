import joblib
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report
)
from .preprocess import run_full_pipeline


def train(data_path: str, model_dir: str = "../models"):
    """
    Full training pipeline.
    Loads raw CSV, preprocesses, trains XGBoost, evaluates, saves artifacts.
    """

    # 1. Load raw data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Raw shape: {df.shape}")

    # 2. Preprocess
    print("Preprocessing...")
    X, y, scaler = run_full_pipeline(df)
    print(f"  Processed shape: {X.shape}")
    print(f"  Churn rate: {y.mean():.3f}")

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # 4. Compute scale_pos_weight for class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"  scale_pos_weight: {spw:.2f}")

    # 5. Train XGBoost
    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='auc',
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # 6. Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['No Churn', 'Churn']))

    # 7. Save artifacts
    os.makedirs(model_dir, exist_ok=True)

    model_path   = os.path.join(model_dir, "xgb_churn_model.pkl")
    scaler_path  = os.path.join(model_dir, "scaler.pkl")

    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nSaved model  → {model_path}")
    print(f"Saved scaler → {scaler_path}")

    return model, scaler, X_test, y_test


if __name__ == "__main__":
    train(
        data_path="../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_dir="../models"
    )