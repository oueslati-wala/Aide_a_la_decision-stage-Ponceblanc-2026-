"""
eval_models.py
==============

Evaluate the two source-specific models (held-out test split), mirroring
the same unified euro workflow used in train_models.py:

    - classifier: candidate selling price (€) + total cost (€) -> P(accept)
    - regressor: coefficient = prix_total / cout_total on accepted quotes,
      then prix = coefficient * cout_total

Both sources are evaluated identically; there is no separate taux_marge
path anymore.
"""

from pathlib import Path
import json

import joblib
import numpy as np

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


def evaluate_source(source):

    print("\n" + "=" * 60)
    print(source.upper())
    print("=" * 60)

    df = features.build_source(source)

    directory = MODELS_DIR / source

    classifier = joblib.load(directory / "classifier_best.joblib")
    regressor = joblib.load(directory / "regressor_marge_best.joblib")

    product_clusters = json.loads(
        (directory / "product_clusters.json").read_text(encoding="utf-8")
    )

    client_encoding = json.loads(
        (directory / "client_encoding.json").read_text(encoding="utf-8")
    )

    regressor_columns = json.loads(
        (directory / "margin_feature_columns.json").read_text()
    )

    _, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["signe"],
    )

    # ---------------------------------------------------------------
    # CLASSIFIER
    # Same euro feature set for both sources: candidate price + cost.
    # ---------------------------------------------------------------

    X = features.build_feature_matrix(
        test_df,
        product_clusters,
        client_encoding,
        include_price=True,
        log_price=True,
        include_cost=True,
        log_cost=True,
    )

    X = X.reindex(
        columns=list(classifier.feature_names_in_),
        fill_value=np.nan,
    )

    y = test_df["signe"]

    probability = classifier.predict_proba(X)[:, 1]
    prediction = classifier.predict(X)

    print("\nCLASSIFIER")
    print(f"Accuracy: {accuracy_score(y, prediction):.3f}")

    try:
        print(f"ROC-AUC: {roc_auc_score(y, probability):.3f}")
    except ValueError:
        print("ROC-AUC: n/a")

    print(classification_report(y, prediction, digits=3))

    # ---------------------------------------------------------------
    # PRICE REGRESSOR
    # Predicts coefficient = prix_total / cout_total on accepted quotes;
    # price is coefficient * cout_total. Same logic for both sources.
    # ---------------------------------------------------------------

    reg_df = df[
        (df["signe"] == 1)
        & df["prix_total"].notna()
        & (df["prix_total"] > 0)
        & df["cout_total"].notna()
        & (df["cout_total"] > 0)
    ].copy()

    reg_df["coeff"] = (
        reg_df["prix_total"].astype(float) / reg_df["cout_total"].astype(float)
    )
    reg_df = reg_df[(reg_df["coeff"] >= 0.80) & (reg_df["coeff"] <= 4.00)].copy()

    _, reg_test = train_test_split(
        reg_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    X_reg = features.build_feature_matrix(
        reg_test,
        product_clusters,
        client_encoding,
        include_cost=True,
        log_cost=True,
    )

    X_reg = X_reg.reindex(
        columns=regressor_columns,
        fill_value=np.nan,
    )

    pred_coeff = np.clip(regressor.predict(X_reg), 0.80, 4.00)
    cost_test = reg_test["cout_total"].astype(float).to_numpy()
    pred_eur = pred_coeff * cost_test

    actual_coeff = reg_test["coeff"].astype(float).to_numpy()
    actual_eur = reg_test["prix_total"].astype(float).to_numpy()

    print(f"\n{source.upper()} PRICE REGRESSOR")
    print(f"MAE (€): {mean_absolute_error(actual_eur, pred_eur):,.2f} €")
    print(f"R² (€): {r2_score(actual_eur, pred_eur):.3f}")
    print(f"MAE (coefficient): {mean_absolute_error(actual_coeff, pred_coeff):.4f}")
    print(f"R² (coefficient): {r2_score(actual_coeff, pred_coeff):.3f}")
    print(f"Actual price mean: {actual_eur.mean():,.2f} €")
    print(f"Predicted price mean: {pred_eur.mean():,.2f} €")


if __name__ == "__main__":

    evaluate_source("ponceblanc")
    evaluate_source("lbfi")
