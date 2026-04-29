import joblib
import os
import shap
import pandas as pd
import matplotlib.pyplot as plt


def build_explainer(model, X_test: pd.DataFrame, model_dir: str = "../models"):
    """
    Creates and saves a SHAP TreeExplainer.
    Returns explainer and shap_values for X_test.
    """
    print("Building SHAP TreeExplainer...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print(f"  SHAP values shape: {shap_values.shape}")

    # Save explainer
    explainer_path = os.path.join(model_dir, "shap_explainer.pkl")
    joblib.dump(explainer, explainer_path)
    print(f"  Saved explainer → {explainer_path}")

    return explainer, shap_values


def plot_global_importance(shap_values, X_test: pd.DataFrame,
                            max_display: int = 15, save_path: str = None):
    """
    Bar chart of mean absolute SHAP values — global feature importance.
    """
    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        plot_type="bar",
        max_display=max_display,
        show=False
    )
    plt.title("Global Feature Importance — Mean |SHAP|")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved → {save_path}")

    plt.show()
    plt.close()


def plot_beeswarm(shap_values, X_test: pd.DataFrame,
                  max_display: int = 15, save_path: str = None):
    """
    Beeswarm plot — shows direction and magnitude of each feature's impact.
    """
    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        max_display=max_display,
        show=False
    )
    plt.title("SHAP Beeswarm — Feature Impact Distribution")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved → {save_path}")

    plt.show()
    plt.close()


def explain_single(explainer, shap_values, X_test: pd.DataFrame,
                   position: int, show: bool = True):
    """
    Waterfall plot for a single customer prediction.
    position: row index in X_test.
    """
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[position],
            base_values   = explainer.expected_value,
            data          = X_test.iloc[position],
            feature_names = X_test.columns.tolist()
        ),
        show=show
    )


def get_shap_for_input(explainer, input_df: pd.DataFrame):
    """
    Returns SHAP values for a single preprocessed input row.
    Used by the Streamlit app.
    """
    shap_vals = explainer.shap_values(input_df)
    return shap_vals