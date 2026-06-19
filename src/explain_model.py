import os
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
matplotlib.use('Agg')  # headless mode


def fix_xgboost_base_score(model):
    """
    Fix XGBoost base_score parsing issue for SHAP compatibility.
    XGBoost sometimes stores base_score as a string like '[3.0423906E1]'
    which SHAP's TreeExplainer cannot parse. This function converts it to float.
    """
    if isinstance(model, xgb.XGBRegressor):
        try:
            base_score_str = model.get_params().get('base_score')
            if isinstance(base_score_str, str):
                # Strip brackets: '[3.0423906E1]' -> '3.0423906E1'
                cleaned = base_score_str.strip('[]')
                # Convert scientific notation to float
                base_score_float = float(cleaned)
                model.set_params(base_score=base_score_float)
        except (ValueError, AttributeError):
            pass  # If conversion fails, continue with original model
    return model


def explain():
    df = pd.read_csv("data/featured_data.csv", parse_dates=["timestamp"])

    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    X = df[feature_cols]

    # sample for SHAP (performance)
    X_sample = X.sample(500, random_state=42)

    print("Computing SHAP values (may take 1-2 min)...")

    # Fix XGBoost base_score if needed
    if isinstance(model, xgb.XGBRegressor):
        model = fix_xgboost_base_score(model)

    # force booster first (avoids base_score crash)
    if hasattr(model, "get_booster"):
        model_for_shap = model.get_booster()
    else:
        model_for_shap = model

    try:
        explainer = shap.TreeExplainer(model_for_shap)
    except Exception as e:
        print("TreeExplainer failed, using fallback SHAP Explainer:", e)
        explainer = shap.Explainer(model, X_sample)

    shap_values = explainer.shap_values(X_sample)

    # -----------------------------
    # PLOT 1: Feature Importance
    # -----------------------------
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title("Top 15 Features — Impact on PM2.5 Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig("models/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: models/shap_importance.png")

    # -----------------------------
    # PLOT 2: SHAP Summary
    # -----------------------------
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=15
    )
    plt.title("SHAP Summary — Feature Impact Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: models/shap_summary.png")

    # -----------------------------
    # Feature importance table
    # -----------------------------
    mean_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_shap": mean_shap
    }).sort_values("mean_shap", ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    importance_df.to_csv("models/shap_importance.csv", index=False)


if __name__ == "__main__":
    explain()
