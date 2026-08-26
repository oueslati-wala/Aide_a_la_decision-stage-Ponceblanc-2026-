"""
eval_models.py
==============

Evaluate the two source-specific models (held-out test split), mirroring
the same unified euro workflow used in train_models.py:

    - classifier: candidate selling price (€) + total cost (€) + final features
      -> P(accept)
    - regressor: coefficient = prix_total / cout_total on accepted quotes,
      then prix = coefficient * cout_total

Final feature set:
    client, produit, quantite, cout_total, season (4-month blocks)
    (+ prix_total for the classifier only)

Both sources are evaluated identically.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

import features


MODELS_DIR = Path("models")
RANDOM_STATE = 42


def _load_json(path: Path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def evaluate_source(source: str) -> dict:
    print("\n" + "=" * 60)
    print(source.upper())
    print("=" * 60)

    df = features.build_source(source)
    directory = MODELS_DIR / source

    classifier = joblib.load(directory / "classifier_best.joblib")
    regressor = joblib.load(directory / "regressor_marge_best.joblib")

    product_clusters = _load_json(directory / "product_clusters.json") or {}
    client_encoding = _load_json(directory / "client_encoding.json") or {}
    regressor_columns = _load_json(directory / "margin_feature_columns.json") or []

    feat_kwargs = {}

    _, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["signe"],
    )

    # ---------------------------------------------------------------
    # CLASSIFIER
    # ---------------------------------------------------------------
    X = features.build_feature_matrix(
        test_df,
        product_clusters,
        client_encoding,
        include_price=True,
        log_price=True,
        include_cost=True,
        log_cost=True,
        **feat_kwargs,
    )
    if hasattr(classifier, "feature_names_in_"):
        X = X.reindex(columns=list(classifier.feature_names_in_), fill_value=np.nan)

    y = test_df["signe"]
    probability = classifier.predict_proba(X)[:, 1]
    prediction = classifier.predict(X)

    accuracy = float(accuracy_score(y, prediction))
    try:
        auc = float(roc_auc_score(y, probability))
    except ValueError:
        auc = None

    report = classification_report(y, prediction, digits=3, output_dict=True)

    print("\nCLASSIFIER")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC-AUC: {auc:.3f}" if auc is not None else "ROC-AUC: n/a")
    print(classification_report(y, prediction, digits=3))
    print(f"Features ({len(X.columns)}): {list(X.columns)}")

    # ---------------------------------------------------------------
    # PRICE REGRESSOR
    # ---------------------------------------------------------------
    reg_df = df[
        (df["signe"] == 1)
        & df["prix_total"].notna()
        & (df["prix_total"] > 0)
        & df["cout_total"].notna()
        & (df["cout_total"] > 0)
    ].copy()
    reg_df["coeff"] = reg_df["prix_total"].astype(float) / reg_df["cout_total"].astype(float)
    reg_df = reg_df[(reg_df["coeff"] >= 0.80) & (reg_df["coeff"] <= 4.00)].copy()

    _, reg_test = train_test_split(reg_df, test_size=0.20, random_state=RANDOM_STATE)

    X_reg = features.build_feature_matrix(
        reg_test,
        product_clusters,
        client_encoding,
        include_cost=True,
        log_cost=True,
        **feat_kwargs,
    )
    X_reg = X_reg.reindex(columns=regressor_columns, fill_value=np.nan)

    pred_coeff = np.clip(regressor.predict(X_reg), 0.80, 4.00)
    cost_test = reg_test["cout_total"].astype(float).to_numpy()
    pred_eur = pred_coeff * cost_test
    actual_coeff = reg_test["coeff"].astype(float).to_numpy()
    actual_eur = reg_test["prix_total"].astype(float).to_numpy()

    mae_eur = float(mean_absolute_error(actual_eur, pred_eur))
    r2_eur = float(r2_score(actual_eur, pred_eur))
    mae_coeff = float(mean_absolute_error(actual_coeff, pred_coeff))
    r2_coeff = float(r2_score(actual_coeff, pred_coeff))

    print(f"\n{source.upper()} PRICE REGRESSOR")
    print(f"MAE (€): {mae_eur:,.2f} €")
    print(f"R² (€): {r2_eur:.3f}")
    print(f"MAE (coefficient): {mae_coeff:.4f}")
    print(f"R² (coefficient): {r2_coeff:.3f}")
    print(f"Actual price mean: {actual_eur.mean():,.2f} €")
    print(f"Predicted price mean: {pred_eur.mean():,.2f} €")

    return {
        "source": source,
        "n_total": int(len(df)),
        "n_test": int(len(test_df)),
        "acceptance_rate": float(df["signe"].mean()),
        "classifier": {
            "accuracy": accuracy,
            "roc_auc": auc,
            "features": list(X.columns),
            "report": report,
        },
        "regressor": {
            "n_rows": int(len(reg_df)),
            "mae_eur": mae_eur,
            "r2_eur": r2_eur,
            "mae_coeff": mae_coeff,
            "r2_coeff": r2_coeff,
            "actual_price_mean": float(actual_eur.mean()),
            "predicted_price_mean": float(pred_eur.mean()),
            "features": list(regressor_columns),
        },
    }


if __name__ == "__main__":
    evaluate_source("ponceblanc")
    evaluate_source("lbfi")
