"""
train_models.py
===============

Train separate models for Ponceblanc and LBFI.

Both sources now use the same euro logic when cost data is available:
    - classifier sees candidate selling price in euros + total cost in euros
      + unit price (log1p(prix/qty)) + calendar season features
      (historical delai is NOT a feature)
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
    """No extra categorical encodings in the final feature set."""
    return {}


def _feature_kwargs(encodings):
    return {}


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
    include_unit_price=True,
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
        include_unit_price=include_unit_price,
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
        include_unit_price=include_unit_price,
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

    n_train = len(train_df)
    classifier = HistGradientBoostingClassifier(
        max_depth=5 if n_train < 2000 else 6,
        learning_rate=0.06 if n_train < 2000 else 0.08,
        max_iter=250,
        min_samples_leaf=max(20, n_train // 80),
        l2_regularization=1.0 if n_train < 2000 else 0.1,
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

def _fit_coeff_priors(train_df: pd.DataFrame, min_n: int = 4) -> dict:
    """Median coefficient by client / product (train accepted only)."""
    global_med = float(train_df["coeff"].median())
    by_client = {}
    for client, g in train_df.groupby(train_df["client"].astype(str).str.upper()):
        if len(g) >= min_n:
            by_client[str(client)] = float(g["coeff"].median())
    by_produit = {}
    for prod, g in train_df.groupby(train_df["produit"].astype(str).str.upper()):
        if len(g) >= 3:
            by_produit[str(prod)] = float(g["coeff"].median())
    return {
        "global": global_med,
        "by_client": by_client,
        "by_produit": by_produit,
        "min_n_client": min_n,
        "min_n_produit": 3,
    }


def _apply_coeff_priors(df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Add client_coeff_prior / produit_coeff_prior columns."""
    global_med = float(priors.get("global", 1.3))
    by_c = priors.get("by_client") or {}
    by_p = priors.get("by_produit") or {}
    clients = df["client"].astype(str).str.upper()
    prods = df["produit"].astype(str).str.upper()
    out = pd.DataFrame(index=df.index)
    out["client_coeff_prior"] = clients.map(lambda c: by_c.get(c, global_med)).astype(float)
    out["produit_coeff_prior"] = prods.map(lambda p: by_p.get(p, global_med)).astype(float)
    # Hierarchical: prefer client× implicitly via average of both priors
    out["hier_coeff_prior"] = (out["client_coeff_prior"] + out["produit_coeff_prior"]) / 2.0
    return out


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
    # Adaptive clip: keep the central mass; hard floor/ceiling for sanity
    lo = max(0.90, float(reg_df["coeff"].quantile(0.05)))
    hi = min(3.00, float(reg_df["coeff"].quantile(0.95)))
    if hi <= lo:
        lo, hi = 0.90, 3.00
    reg_df = reg_df[(reg_df["coeff"] >= lo) & (reg_df["coeff"] <= hi)].copy()
    print(
        f"  accepted rows for price model: {len(reg_df)} "
        f"(dropped {before - len(reg_df)} outlier coeffs; clip [{lo:.2f}, {hi:.2f}])"
    )

    if len(reg_df) < 20:
        print("  [WARN] Too few rows for price regressor.")
        return None

    train_df, test_df = train_test_split(
        reg_df,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # Hierarchical priors fitted on train only (no leakage)
    priors = _fit_coeff_priors(train_df)
    with open(output_dir / "coeff_priors.json", "w", encoding="utf-8") as f:
        json.dump(priors, f, ensure_ascii=False, indent=2)

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
    X_train = pd.concat([X_train, _apply_coeff_priors(train_df, priors)], axis=1)
    X_test = pd.concat([X_test, _apply_coeff_priors(test_df, priors)], axis=1)

    # Log-target stabilizes heavy tails; inverse = exp at predict time
    y_train_log = np.log(train_df["coeff"].astype(float).clip(lower=1e-3))
    y_test_coeff = test_df["coeff"].astype(float)
    y_test_price = test_df["prix_total"].astype(float)
    cost_test = test_df["cout_total"].astype(float)

    small_n = len(train_df) < 400
    if small_n:
        # Strong regularization — avoid overfitting sparse accepted quotes
        params = dict(
            max_depth=3,
            learning_rate=0.05,
            max_iter=150,
            min_samples_leaf=max(15, len(train_df) // 15),
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
        print(f"  small-n mode (n_train={len(train_df)}): depth=3, strong L2")
    else:
        params = dict(
            max_depth=5,
            learning_rate=0.08,
            max_iter=250,
            min_samples_leaf=15,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        )

    best = HistGradientBoostingRegressor(loss="squared_error", **params)
    lower = HistGradientBoostingRegressor(loss="quantile", quantile=0.15, **params)
    upper = HistGradientBoostingRegressor(loss="quantile", quantile=0.85, **params)

    best.fit(X_train, y_train_log)
    lower.fit(X_train, y_train_log)
    upper.fit(X_train, y_train_log)

    def _to_coeff(log_pred):
        return np.clip(np.exp(log_pred), lo, hi)

    pred_model = _to_coeff(best.predict(X_test))
    pred_hier = X_test["hier_coeff_prior"].to_numpy()
    baseline_eur = float(priors["global"]) * cost_test.values
    mae_glob = float(mean_absolute_error(y_test_price, baseline_eur))

    # Small-n: lean hard on client/product median coefficients.
    # A free GBM overfits ~200 accepted quotes and loses to a flat median.
    blend = 0.0
    if small_n:
        blend = 0.85
        pred_coeff = np.clip(blend * pred_hier + (1.0 - blend) * pred_model, lo, hi)
        mae_blend = float(mean_absolute_error(y_test_price, pred_coeff * cost_test.values))
        if mae_blend > mae_glob * 1.01:
            blend = 1.0
            pred_coeff = np.clip(pred_hier, lo, hi)
            print(f"  pure hierarchical priors (model residual hurt hold-out MAE)")
        else:
            print(f"  blend hierarchical {blend:.0%} + model {1-blend:.0%}")
    else:
        pred_coeff = pred_model

    pred_eur = pred_coeff * cost_test.values

    mae = float(mean_absolute_error(y_test_price, pred_eur))
    mae_coeff = float(mean_absolute_error(y_test_coeff, pred_coeff))
    mae_baseline = mae_glob
    within_20 = float(
        (np.abs(y_test_price.values - pred_eur) / np.maximum(y_test_price.values, 1e-6) <= 0.20).mean()
    )

    print(f"  price MAE = {mae:,.2f} €  |  coeff MAE = {mae_coeff:.3f}  |  within±20% = {within_20:.0%}")
    print(f"  baseline (global median coeff) MAE = {mae_baseline:,.2f} €")
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

    # Realistic commercial band from accepted-quote distribution (not the
    # outlier clip). Used by recommend_price so expected_margin / balanced
    # stay inside the historical mass instead of drifting to the clip ceiling.
    p50 = float(reg_df["coeff"].median())
    p75 = float(reg_df["coeff"].quantile(0.75))
    p90 = float(reg_df["coeff"].quantile(0.90))
    meta = {
        "target": "coefficient",
        "mode": "coeff_times_cost",
        "target_transform": "log",
        "coeff_clip": [lo, hi],
        "coeff_p50": p50,
        "coeff_p75": p75,
        "coeff_p90": p90,
        "small_n_blend": blend,
        "uses_coeff_priors": True,
    }
    with open(output_dir / "regressor_target.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "target": "coefficient",
        "mode": "coeff_times_cost",
        "target_transform": "log",
        "mae": mae,
        "mae_coeff": mae_coeff,
        "mae_baseline_eur": mae_baseline,
        "within_20pct": within_20,
        "n_rows": int(len(reg_df)),
        "features": list(X_train.columns),
        "coeff_clip": [lo, hi],
        "small_n_blend": blend,
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

    for col in ["month"]:
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
        if mapping:
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

    target = (reg_metrics or {}).get("target", "coefficient")
    mode = (reg_metrics or {}).get("mode", "coeff_times_cost")
    transform = (reg_metrics or {}).get("target_transform", "log")

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

    # regressor_target.json already written inside train_price_regressor when successful
    if reg_metrics is None:
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
