"""
eval_models.py
==============

Évalue les modèles Ponceblanc et LBFI sur un jeu de test (20 % hold-out).

Objectifs pour un public non data-science :
  - métriques techniques (AUC, MAE…)
  - comparaisons à des baselines simples (hasard, toujours la majorité…)
  - calibration des probabilités (« quand le modèle dit 70 %, ça arrive vraiment ~70 % ? »)
  - erreurs de prix en % et par tranche de coût
  - verdict de confiance en français clair

Même logique métier que train_models.py :
  - classifieur : P(acceptation | client, produit, qty, coût, prix candidat)
  - régresseur  : coefficient = prix/coût sur devis acceptés → prix = coeff × coût
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split

import features


MODELS_DIR = Path("models")
RANDOM_STATE = 42


def _load_json(path: Path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _interpret_auc(auc: float | None) -> str:
    if auc is None:
        return "Non calculable (une seule classe dans le test)."
    if auc < 0.55:
        return "Presque comme le hasard — le modèle ne discrimine pas mieux qu'un pile-ou-face."
    if auc < 0.65:
        return "Faible — un peu mieux que le hasard, utile seulement avec prudence."
    if auc < 0.75:
        return "Correct — meilleur que le hasard de façon nette ; utile en aide à la décision."
    if auc < 0.85:
        return "Bon — le modèle sépare bien acceptés et refusés dans la plupart des cas."
    return "Très bon — discrimination forte sur le jeu de test."


def _interpret_accuracy(acc: float, baseline: float) -> str:
    gain = acc - baseline
    if gain < 0.02:
        return (
            f"À peine mieux que « toujours prédire la classe majoritaire » "
            f"(baseline {baseline:.0%}). Peu d'apport."
        )
    if gain < 0.08:
        return (
            f"Meilleur de {gain:.0%} points que la baseline « toujours la majorité » "
            f"({baseline:.0%}). Apport modéré."
        )
    return (
        f"Meilleur de {gain:.0%} points que la baseline « toujours la majorité » "
        f"({baseline:.0%}). Apport clair."
    )


def _trust_level(
    auc: float | None,
    mae_coeff: float | None,
    n_reg: int,
    within_20: float | None = None,
    beats_baseline: bool | None = None,
) -> dict:
    """
    Score de confiance simple 0–100 et libellé pour non-experts.
    """
    score = 50
    reasons = []

    if auc is not None:
        if auc >= 0.75:
            score += 20
            reasons.append("Le classifieur sépare bien acceptés et refusés (AUC ≥ 0,75).")
        elif auc >= 0.65:
            score += 10
            reasons.append("Le classifieur est correct mais pas excellent (AUC 0,65–0,75).")
        else:
            score -= 15
            reasons.append("Le classifieur est faible (AUC < 0,65) — proche du hasard.")

    if mae_coeff is not None:
        if mae_coeff <= 0.12:
            score += 15
            reasons.append("Erreur moyenne sur le coefficient de prix très faible (≤ 0,12).")
        elif mae_coeff <= 0.22:
            score += 10
            reasons.append("Erreur moyenne sur le coefficient bonne (≤ 0,22).")
        elif mae_coeff <= 0.32:
            score += 4
            reasons.append("Erreur moyenne sur le coefficient correcte (≤ 0,32) — prix reco utilisables avec marge.")
        elif mae_coeff <= 0.45:
            score -= 2
            reasons.append("Erreur moyenne sur le coefficient élevée — prix reco à prendre avec marge.")
        else:
            score -= 12
            reasons.append("Erreur moyenne sur le coefficient très élevée — prix reco peu fiables.")

    if within_20 is not None:
        if within_20 >= 0.85:
            score += 10
            reasons.append(f"Dans {within_20:.0%} des cas, le prix prédit est à ±20 % du vrai prix accepté.")
        elif within_20 >= 0.65:
            score += 6
            reasons.append(f"Dans {within_20:.0%} des cas, le prix prédit est à ±20 % du vrai prix accepté.")
        elif within_20 < 0.50:
            score -= 8
            reasons.append(
                f"Seulement {within_20:.0%} des prix prédits sont à ±20 % du vrai prix — dispersion forte."
            )

    if beats_baseline is True:
        score += 8
        reasons.append(
            "Le régresseur égale ou bat la règle simple « coefficient médian » sur l'erreur en euros."
        )
    elif beats_baseline is False:
        score -= 4
        reasons.append(
            "Le régresseur reste un peu derrière une règle simple (coefficient médian) — "
            "prix reco à croiser avec l'historique."
        )

    if n_reg < 200:
        score -= 10
        reasons.append(f"Peu de devis acceptés avec coût pour entraîner le prix ({n_reg}).")
    elif n_reg >= 1000:
        score += 8
        reasons.append(f"Beaucoup de devis acceptés avec coût ({n_reg}) — régresseur mieux ancré.")

    score = int(max(0, min(100, score)))
    if score >= 80:
        label = "Confiance élevée"
        color = "green"
        advice = (
            "Vous pouvez vous appuyer sur les recommandations pour les profils "
            "habituels (clients / produits fréquents). Gardez votre jugement "
            "sur les cas rares ou les montants extrêmes."
        )
    elif score >= 55:
        label = "Confiance moyenne"
        color = "orange"
        advice = (
            "Utile comme aide, pas comme décision automatique. Vérifiez toujours "
            "l'historique comparable affiché à côté de la prédiction. "
            "Le volume de devis acceptés avec coût peut limiter la précision du prix."
        )
    else:
        label = "Confiance faible"
        color = "red"
        advice = (
            "Les chiffres techniques sont trop faibles pour une confiance forte. "
            "Utilisez l'outil surtout pour explorer l'historique, pas pour fixer le prix seul."
        )

    return {
        "score": score,
        "label": label,
        "color": color,
        "advice": advice,
        "reasons": reasons,
    }


def _calibration_bins(y_true, proba, n_bins: int = 5) -> list[dict]:
    """Fiabilité : pour chaque tranche de proba prédite, quel % réel d'acceptation."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (proba >= lo) & (proba <= hi)
        else:
            mask = (proba >= lo) & (proba < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred_mean = float(proba[mask].mean())
        actual_rate = float(y_true[mask].mean())
        rows.append({
            "tranche_proba": f"{lo:.0%}–{hi:.0%}",
            "n": n,
            "proba_moyenne": round(pred_mean, 3),
            "taux_reel": round(actual_rate, 3),
            "ecart": round(actual_rate - pred_mean, 3),
        })
    return rows


def _price_error_by_cost_band(cost, actual_eur, pred_eur) -> list[dict]:
    cost = np.asarray(cost, dtype=float)
    actual_eur = np.asarray(actual_eur, dtype=float)
    pred_eur = np.asarray(pred_eur, dtype=float)
    if len(cost) < 5:
        return []
    qs = np.quantile(cost, [0, 0.33, 0.66, 1.0])
    labels = ["Coût bas", "Coût moyen", "Coût élevé"]
    rows = []
    for i in range(3):
        lo, hi = qs[i], qs[i + 1]
        if i == 2:
            mask = (cost >= lo) & (cost <= hi)
        else:
            mask = (cost >= lo) & (cost < hi)
        n = int(mask.sum())
        if n < 3:
            continue
        mae = float(np.mean(np.abs(actual_eur[mask] - pred_eur[mask])))
        mape = float(np.mean(np.abs(actual_eur[mask] - pred_eur[mask]) / np.maximum(actual_eur[mask], 1e-6)))
        rows.append({
            "bande": labels[i],
            "cout_min": round(float(lo), 0),
            "cout_max": round(float(hi), 0),
            "n": n,
            "mae_eur": round(mae, 0),
            "mape": round(mape, 3),
        })
    return rows


def evaluate_source(source: str, verbose: bool = True) -> dict:
    if verbose:
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
    coeff_priors = _load_json(directory / "coeff_priors.json") or {}
    reg_meta = _load_json(directory / "regressor_target.json") or {}
    target_transform = reg_meta.get("target_transform", "identity")
    coeff_clip = reg_meta.get("coeff_clip") or [0.80, 4.00]
    small_n_blend = float(reg_meta.get("small_n_blend") or 0.0)
    lo_clip, hi_clip = float(coeff_clip[0]), float(coeff_clip[1])

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
        include_unit_price=True,
        **feat_kwargs,
    )
    if hasattr(classifier, "feature_names_in_"):
        X = X.reindex(columns=list(classifier.feature_names_in_), fill_value=np.nan)

    y = test_df["signe"].astype(int)
    probability = classifier.predict_proba(X)[:, 1]
    prediction = classifier.predict(X)

    accuracy = float(accuracy_score(y, prediction))
    try:
        auc = float(roc_auc_score(y, probability))
    except ValueError:
        auc = None

    report = classification_report(y, prediction, digits=3, output_dict=True)
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Baselines
    majority_class = int(y.mode().iloc[0]) if len(y) else 0
    baseline_acc = float((y == majority_class).mean())
    try:
        brier = float(brier_score_loss(y, probability))
    except Exception:
        brier = None

    calibration = _calibration_bins(y, probability, n_bins=5)

    # Frequent vs rare clients (by historical count in full df)
    client_counts = df["client"].astype(str).value_counts()
    test_clients = test_df["client"].astype(str)
    freq_mask = test_clients.map(lambda c: client_counts.get(c, 0) >= 10)
    rare_mask = ~freq_mask
    segment_auc = {}
    for name, mask in (("clients_frequents", freq_mask), ("clients_rares", rare_mask)):
        if mask.sum() >= 20 and y[mask].nunique() == 2:
            try:
                segment_auc[name] = {
                    "n": int(mask.sum()),
                    "auc": float(roc_auc_score(y[mask], probability[mask])),
                    "accuracy": float(accuracy_score(y[mask], prediction[mask])),
                }
            except ValueError:
                pass

    if verbose:
        print("\nCLASSIFIER")
        print(f"Accuracy: {accuracy:.3f}  |  baseline majorité: {baseline_acc:.3f}")
        print(f"ROC-AUC: {auc:.3f}" if auc is not None else "ROC-AUC: n/a")
        print(f"  → {_interpret_auc(auc)}")
        print(f"  → {_interpret_accuracy(accuracy, baseline_acc)}")
        print(classification_report(y, prediction, digits=3))
        print(f"Confusion  TN={tn} FP={fp}  FN={fn} TP={tp}")
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
    reg_df = reg_df[(reg_df["coeff"] >= lo_clip) & (reg_df["coeff"] <= hi_clip)].copy()

    _, reg_test = train_test_split(reg_df, test_size=0.20, random_state=RANDOM_STATE)

    X_reg = features.build_feature_matrix(
        reg_test,
        product_clusters,
        client_encoding,
        include_cost=True,
        log_cost=True,
        **feat_kwargs,
    )
    # Hierarchical prior features (same as train)
    if coeff_priors:
        global_med = float(coeff_priors.get("global", 1.3))
        by_c = coeff_priors.get("by_client") or {}
        by_p = coeff_priors.get("by_produit") or {}
        clients = reg_test["client"].astype(str).str.upper()
        prods = reg_test["produit"].astype(str).str.upper()
        X_reg = X_reg.copy()
        X_reg["client_coeff_prior"] = clients.map(lambda c: by_c.get(c, global_med)).astype(float).values
        X_reg["produit_coeff_prior"] = prods.map(lambda p: by_p.get(p, global_med)).astype(float).values
        X_reg["hier_coeff_prior"] = (
            X_reg["client_coeff_prior"] + X_reg["produit_coeff_prior"]
        ) / 2.0
    X_reg = X_reg.reindex(columns=regressor_columns, fill_value=np.nan)

    raw_pred = regressor.predict(X_reg)
    if target_transform == "log":
        pred_model = np.clip(np.exp(raw_pred), lo_clip, hi_clip)
    else:
        pred_model = np.clip(raw_pred, lo_clip, hi_clip)

    if small_n_blend > 0 and "hier_coeff_prior" in X_reg.columns:
        pred_hier = X_reg["hier_coeff_prior"].to_numpy()
        pred_coeff = np.clip(
            small_n_blend * pred_hier + (1.0 - small_n_blend) * pred_model,
            lo_clip,
            hi_clip,
        )
    else:
        pred_coeff = pred_model

    cost_test = reg_test["cout_total"].astype(float).to_numpy()
    pred_eur = pred_coeff * cost_test
    actual_coeff = reg_test["coeff"].astype(float).to_numpy()
    actual_eur = reg_test["prix_total"].astype(float).to_numpy()

    mae_eur = float(mean_absolute_error(actual_eur, pred_eur))
    r2_eur = float(r2_score(actual_eur, pred_eur))
    mae_coeff = float(mean_absolute_error(actual_coeff, pred_coeff))
    r2_coeff = float(r2_score(actual_coeff, pred_coeff))
    try:
        mape = float(mean_absolute_percentage_error(actual_eur, pred_eur))
    except Exception:
        mape = float(np.mean(np.abs(actual_eur - pred_eur) / np.maximum(actual_eur, 1e-6)))

    # Median absolute % error (plus robuste)
    pct_err = np.abs(actual_eur - pred_eur) / np.maximum(actual_eur, 1e-6)
    median_ape = float(np.median(pct_err))
    within_10 = float((pct_err <= 0.10).mean())
    within_20 = float((pct_err <= 0.20).mean())

    # Baseline : médiane du coefficient d'entraînement
    med_coeff = float(reg_df["coeff"].median())
    baseline_pred_eur = med_coeff * cost_test
    baseline_mae_eur = float(mean_absolute_error(actual_eur, baseline_pred_eur))
    baseline_mape = float(np.mean(np.abs(actual_eur - baseline_pred_eur) / np.maximum(actual_eur, 1e-6)))

    cost_bands = _price_error_by_cost_band(cost_test, actual_eur, pred_eur)

    if verbose:
        print(f"\n{source.upper()} PRICE REGRESSOR")
        print(f"MAE (€): {mae_eur:,.2f} €  |  MAPE: {mape:.1%}  |  médiane |err|: {median_ape:.1%}")
        print(f"Dans ±10 % du vrai prix : {within_10:.0%} des cas test")
        print(f"Dans ±20 % du vrai prix : {within_20:.0%} des cas test")
        print(f"R² (€): {r2_eur:.3f}  |  MAE coeff: {mae_coeff:.4f}  |  R² coeff: {r2_coeff:.3f}")
        print(f"Baseline (coeff médian={med_coeff:.3f}) MAE: {baseline_mae_eur:,.2f} €  MAPE: {baseline_mape:.1%}")
        print(f"Actual price mean: {actual_eur.mean():,.2f} €")
        print(f"Predicted price mean: {pred_eur.mean():,.2f} €")

    # Treat within 2% of the simple median rule as "matches baseline" (not a failure)
    if baseline_mae_eur and baseline_mae_eur > 0:
        if mae_eur <= baseline_mae_eur * 1.02:
            beats_baseline = True
        else:
            beats_baseline = False
    else:
        beats_baseline = None
    trust = _trust_level(
        auc,
        mae_coeff,
        int(len(reg_df)),
        within_20=within_20,
        beats_baseline=beats_baseline,
    )
    if verbose:
        print(f"\nVERDICT: {trust['label']} ({trust['score']}/100)")
        for r in trust["reasons"]:
            print(f"  · {r}")
        print(f"  → {trust['advice']}")

    return {
        "source": source,
        "n_total": int(len(df)),
        "n_test": int(len(test_df)),
        "acceptance_rate": float(df["signe"].mean()),
        "classifier": {
            "accuracy": accuracy,
            "roc_auc": auc,
            "baseline_accuracy": baseline_acc,
            "brier_score": brier,
            "features": list(X.columns),
            "report": report,
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "calibration": calibration,
            "segments": segment_auc,
            "interpret_auc": _interpret_auc(auc),
            "interpret_accuracy": _interpret_accuracy(accuracy, baseline_acc),
        },
        "regressor": {
            "n_rows": int(len(reg_df)),
            "n_test": int(len(reg_test)),
            "mae_eur": mae_eur,
            "r2_eur": r2_eur,
            "mae_coeff": mae_coeff,
            "r2_coeff": r2_coeff,
            "mape": mape,
            "median_ape": median_ape,
            "within_10pct": within_10,
            "within_20pct": within_20,
            "baseline_mae_eur": baseline_mae_eur,
            "baseline_mape": baseline_mape,
            "baseline_coeff": med_coeff,
            "actual_price_mean": float(actual_eur.mean()),
            "predicted_price_mean": float(pred_eur.mean()),
            "cost_bands": cost_bands,
            "features": list(regressor_columns),
        },
        "trust": trust,
    }


if __name__ == "__main__":
    for src in ("ponceblanc", "lbfi"):
        evaluate_source(src)
