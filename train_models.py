"""
train_models.py
===============

Train separate models for Ponceblanc and LBFI.

Both sources now use the same euro logic when cost data is available:
    - classifier sees candidate selling price in euros + total cost in euros
      + all available context features (matiere, format, commercial, calendar,
      dimensions when present; historical delai is NOT a feature)
    - price regressor predicts coefficient = prix / coût on accepted quotes;
      price is recovered as coefficient × coût
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

import features


MODELS_DIR = Path("models")
RANDOM_STATE = 42


def split_data(df):

    try:

        return train_test_split(
            df,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=df["signe"],
        )

    except ValueError:

        return train_test_split(
            df,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )


def _fit_extra_encodings(train_df):
    """Fit encodings for matière only (final feature list)."""
    matiere_enc = features.fit_categorical_encoding(
        train_df["matiere"] if "matiere" in train_df.columns else pd.Series(dtype=object),
        min_count=8,
    )
    return {
        "matiere_encoding": matiere_enc,
    }


def _feature_kwargs(encodings):
    return dict(
        matiere_encoding=encodings.get("matiere_encoding"),
    )


# ---------------------------------------------------------------------
# CLASSIFIER
# ---------------------------------------------------------------------

def train_classifier(
    source,
    train_df,
    test_df,
    product_clusters,
    client_encoding,
    encodings,
    output_dir,
):
    kwargs = _feature_kwargs(encodings)

    X_train = features.build_feature_matrix(
        train_df,
        product_clusters,
        client_encoding,
        include_price=True,
        log_price=True,
        include_cost=True,
        log_cost=True,
        **kwargs,
    )

    X_test = features.build_feature_matrix(
        test_df,
        product_clusters,
        client_encoding,
        include_price=True,
        log_price=True,
        include_cost=True,
        log_cost=True,
        **kwargs,
    )

    all_nan = [
        col
        for col in X_train.columns
        if X_train[col].isna().all()
    ]

    if all_nan:
        X_train = X_train.drop(columns=all_nan)
        X_test = X_test.drop(columns=all_nan)

    classifier = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
    )

    classifier.fit(
        X_train,
        train_df["signe"],
    )

    probability = classifier.predict_proba(X_test)[:, 1]
    prediction = classifier.predict(X_test)

    accuracy = float(accuracy_score(test_df["signe"], prediction))

    try:
        auc = float(roc_auc_score(test_df["signe"], probability))
    except ValueError:
        auc = None

    joblib.dump(classifier, output_dir / "classifier_best.joblib")

    print(f"  classifier accuracy = {accuracy:.3f}")
    if auc is not None:
        print(f"  classifier ROC-AUC = {auc:.3f}")
    print(f"  classifier features ({len(X_train.columns)}): {list(X_train.columns)}")

    return classifier, list(X_train.columns), accuracy, auc


# ---------------------------------------------------------------------
# PRICE REGRESSOR
# ---------------------------------------------------------------------

def train_price_regressor(
    df,
    product_clusters,
    client_encoding,
    encodings,
    output_dir,
):

    reg_df = df[
        (df["signe"] == 1)
        & df["prix_total"].notna()
        & (df["prix_total"] > 0)
        & df["cout_total"].notna()
        & (df["cout_total"] > 0)
    ].copy()

    reg_df["coeff"] = (
        reg_df["prix_total"].astype(float)
        / reg_df["cout_total"].astype(float)
    )

    before = len(reg_df)
    reg_df = reg_df[
        (reg_df["coeff"] >= 0.80)
        & (reg_df["coeff"] <= 4.00)
    ].copy()
    print(
        f"  accepted rows for price model: {len(reg_df)} "
        f"(dropped {before - len(reg_df)} outlier coeffs)"
    )

    if len(reg_df) < 20:
        print("  [WARN] Too few rows for price regressor.")
        return None

    train_df, test_df = train_test_split(
        reg_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    kwargs = _feature_kwargs(encodings)

    X_train = features.build_feature_matrix(
        train_df,
        product_clusters,
        client_encoding,
        include_cost=True,
        log_cost=True,
        **kwargs,
    )

    X_test = features.build_feature_matrix(
        test_df,
        product_clusters,
        client_encoding,
        include_cost=True,
        log_cost=True,
        **kwargs,
    )

    y_train = train_df["coeff"].astype(float)
    y_test_coeff = test_df["coeff"].astype(float)
    y_test_price = test_df["prix_total"].astype(float)
    cost_test = test_df["cout_total"].astype(float)

    params = dict(
        max_depth=5,
        learning_rate=0.08,
        max_iter=250,
        min_samples_leaf=15,
        random_state=RANDOM_STATE,
    )

    best = HistGradientBoostingRegressor(loss="squared_error", **params)
    lower = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **params)
    upper = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **params)

    best.fit(X_train, y_train)
    lower.fit(X_train, y_train)
    upper.fit(X_train, y_train)

    pred_coeff = np.clip(best.predict(X_test), 0.80, 4.00)
    pred_eur = pred_coeff * cost_test.values

    mae = float(mean_absolute_error(y_test_price, pred_eur))
    mae_coeff = float(mean_absolute_error(y_test_coeff, pred_coeff))

    print(f"  price MAE = {mae:,.2f} €  |  coeff MAE = {mae_coeff:.3f}")
    print(
        f"  coeff train median = {float(train_df['coeff'].median()):.2f}  "
        f"p10={float(train_df['coeff'].quantile(0.10)):.2f}  "
        f"p90={float(train_df['coeff'].quantile(0.90)):.2f}"
    )

    joblib.dump(best, output_dir / "regressor_marge_best.joblib")
    joblib.dump(lower, output_dir / "regressor_marge_lower.joblib")
    joblib.dump(upper, output_dir / "regressor_marge_upper.joblib")

    with open(output_dir / "margin_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f)

    return {
        "target": "coefficient",
        "mode": "coeff_times_cost",
        "target_transform": "identity",
        "mae": mae,
        "mae_coeff": mae_coeff,
        "n_rows": int(len(reg_df)),
        "features": list(X_train.columns),
        "coeff_training": {
            "min": float(reg_df["coeff"].min()),
            "p10": float(reg_df["coeff"].quantile(0.10)),
            "median": float(reg_df["coeff"].median()),
            "p90": float(reg_df["coeff"].quantile(0.90)),
            "max": float(reg_df["coeff"].max()),
        },
        "cost_training_range_eur": {
            "min": float(reg_df["cout_total"].min()),
            "median": float(reg_df["cout_total"].median()),
            "max": float(reg_df["cout_total"].max()),
        },
    }


# ---------------------------------------------------------------------
# SOURCE TRAINING
# ---------------------------------------------------------------------

def train_source(source):

    print("\n" + "=" * 60)
    print(f"TRAINING {source.upper()}")
    print("=" * 60)

    df = features.build_source(source)

    if len(df) < 30:
        print(f"[SKIP] only {len(df)} rows")
        return

    print(f"rows = {len(df)}")
    print(f"acceptance = {df['signe'].mean():.1%}")

    for col in ["matiere", "month"]:
        if col in df.columns:
            nn = int(df[col].notna().sum())
            print(f"  {col}: {nn}/{len(df)} ({100*nn/len(df):.0f}%)")

    output_dir = MODELS_DIR / source
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = split_data(df)

    product_clusters = features.fit_product_clusters(train_df["produit"])
    client_encoding = features.fit_client_encoding(train_df["client"], train_df["signe"])
    encodings = _fit_extra_encodings(train_df)

    with open(output_dir / "product_clusters.json", "w", encoding="utf-8") as f:
        json.dump(product_clusters, f, ensure_ascii=False, indent=2)

    with open(output_dir / "client_encoding.json", "w", encoding="utf-8") as f:
        json.dump(client_encoding, f, ensure_ascii=False, indent=2)

    for name, mapping in encodings.items():
        with open(output_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    classifier, classifier_features, accuracy, auc = train_classifier(
        source,
        train_df,
        test_df,
        product_clusters,
        client_encoding,
        encodings,
        output_dir,
    )

    reg_metrics = train_price_regressor(
        df,
        product_clusters,
        client_encoding,
        encodings,
        output_dir,
    )

    target = "coefficient"
    mode = "coeff_times_cost"
    transform = "identity"

    metrics = {
        "source": source,
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "acceptance_rate": float(df["signe"].mean()),
        "classifier": {
            "accuracy": accuracy,
            "roc_auc": auc,
            "features": classifier_features,
        },
        "margin_regressor": reg_metrics,
        "regressor_target": target,
        "regressor_mode": mode,
        "target_transform": transform,
        "extra_encodings": list(encodings.keys()),
    }

    with open(output_dir / "regressor_target.json", "w", encoding="utf-8") as f:
        json.dump(
            {"target": target, "mode": mode, "target_transform": transform},
            f,
            indent=2,
        )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Models saved to {output_dir}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_source("ponceblanc")
    train_source("lbfi")


if __name__ == "__main__":
    main()
