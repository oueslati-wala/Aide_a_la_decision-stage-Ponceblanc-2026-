"""
predict.py
==========

Prediction API.

Both Ponceblanc and LBFI use the same euro workflow:
    cout_total  = total cost in €
    prix_total  = total selling price in €

The regressor predicts a prix/coût coefficient on accepted quotes; the price
is then derived as coefficient × cout_total. Margin rates / taux_marge are
never inputs or targets of the models — they exist only as a display-time
derivation in the UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import features


MODELS_DIR = Path("models")


class QuoteEstimator:

    def __init__(
        self,
        models_dir: str | Path = MODELS_DIR,
    ):
        self.models_dir = Path(models_dir)
        self._cache = {}

    # -----------------------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------------------

    def _load_source(
        self,
        source: str,
    ) -> dict:
        source = source.strip().lower()

        if source not in features.VALID_SOURCES:
            raise ValueError(
                f"source must be one of {features.VALID_SOURCES}"
            )

        cache_key = source
        directory = self.models_dir / source

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not directory.is_dir():
            raise FileNotFoundError(
                f"No models found for {source!r}.\n"
                f"Expected: {directory}\n"
                "Run train_models.py first."
            )

        target_file = directory / "regressor_target.json"

        if target_file.is_file():
            metadata = json.loads(
                target_file.read_text(encoding="utf-8")
            )
        else:
            metadata = {
                "target": "taux_marge",
                "mode": "taux_marge",
                "target_transform": "identity",
            }

        metrics_file = directory / "metrics.json"
        metrics = {}

        if metrics_file.is_file():
            metrics = json.loads(
                metrics_file.read_text(encoding="utf-8")
            )

        bundle = {
            "classifier": joblib.load(directory / "classifier_best.joblib"),
            "reg_best": joblib.load(directory / "regressor_marge_best.joblib"),
            "reg_lower": joblib.load(directory / "regressor_marge_lower.joblib"),
            "reg_upper": joblib.load(directory / "regressor_marge_upper.joblib"),
            "product_clusters": json.loads(
                (directory / "product_clusters.json").read_text(encoding="utf-8")
            ),
            "client_encoding": json.loads(
                (directory / "client_encoding.json").read_text(encoding="utf-8")
            ),
            "regressor_features": json.loads(
                (directory / "margin_feature_columns.json").read_text()
            ),
            "regressor_target": metadata.get("target"),
            "regressor_mode": metadata.get("mode"),
            "target_transform": metadata.get("target_transform", "identity"),
            "metrics": metrics,
        }

        self._cache[cache_key] = bundle
        return bundle

    # -----------------------------------------------------------------
    # ROW
    # -----------------------------------------------------------------

    @staticmethod
    def _make_row(
        client,
        produit,
        quantite,
        source,
        *,
        cout_total=None,
        prix_total=None,
        taux_marge=None,
    ):
        return pd.DataFrame(
            [
                {
                    "client": str(client).strip().upper(),
                    "produit": str(produit).strip().upper(),
                    "quantite": float(quantite),
                    "source": source,
                    "cout_total": cout_total,
                    "prix_total": prix_total,
                    "taux_marge": taux_marge,
                }
            ]
        )

    # -----------------------------------------------------------------
    # CLASSIFIER
    # -----------------------------------------------------------------

    def predict_acceptance_proba(
        self,
        client: str,
        produit: str,
        quantite: float,
        source: str,
        *,
        cout_total: float | None = None,
        prix_total: float | None = None,
        taux_marge: float | None = None,
    ) -> float:
        source = source.strip().lower()
        bundle = self._load_source(source)

        # Euro features: candidate selling price + total cost.
        if cout_total is None or cout_total <= 0:
            raise ValueError(f"{source} needs cout_total > 0 €.")
        if prix_total is None or prix_total <= 0:
            raise ValueError(f"{source} acceptance prediction needs prix_total > 0 €.")

        row = self._make_row(
            client,
            produit,
            quantite,
            source,
            cout_total=float(cout_total),
            prix_total=float(prix_total),
        )

        X = features.build_feature_matrix(
            row,
            bundle["product_clusters"],
            bundle["client_encoding"],
            include_price=True,
            log_price=True,
            include_cost=True,
            log_cost=True,
        )

        classifier = bundle["classifier"]

        if hasattr(classifier, "feature_names_in_"):
            X = X.reindex(
                columns=list(classifier.feature_names_in_),
                fill_value=np.nan,
            )

        return float(classifier.predict_proba(X)[0, 1])

    # -----------------------------------------------------------------
    # RECOMMENDED PRICE
    # -----------------------------------------------------------------

    def recommend_price(
        self,
        client: str,
        produit: str,
        quantite: float,
        source: str,
        *,
        cout_total: float | None = None,
    ) -> dict:
        source = source.strip().lower()
        bundle = self._load_source(source)

        if cout_total is None or cout_total <= 0:
            raise ValueError(f"{source} price recommendation needs cout_total > 0 €.")

        cost = float(cout_total)

        # Model predicts prix/cout coefficient on accepted quotes; convert
        # back to euros via prix = coeff * cout_total.
        row = self._make_row(
            client,
            produit,
            quantite,
            source,
            cout_total=cost,
        )
        X = features.build_feature_matrix(
            row,
            bundle["product_clusters"],
            bundle["client_encoding"],
            include_cost=True,
            log_cost=True,
        )
        X = X.reindex(
            columns=bundle["regressor_features"],
            fill_value=np.nan,
        )
        c_lo = float(np.clip(bundle["reg_lower"].predict(X)[0], 0.80, 4.00))
        c_med = float(np.clip(bundle["reg_best"].predict(X)[0], 0.80, 4.00))
        c_hi = float(np.clip(bundle["reg_upper"].predict(X)[0], 0.80, 4.00))
        c_lo, c_hi = min(c_lo, c_med), max(c_hi, c_med)
        lower, median, upper = c_lo * cost, c_med * cost, c_hi * cost
        coeff_median = c_med

        # Business floor: never recommend below cost
        lower = max(lower, cost)
        median = max(median, cost)
        upper = max(upper, cost)

        return {
            "target": "prix_total",
            "mode": "coeff_times_cost",
            "prix_lower": float(lower),
            "prix_median": float(median),
            "prix_upper": float(upper),
            "cost_eur": cost,
            "coeff_median": float(coeff_median) if coeff_median is not None else None,
            "taux_lower": float(lower / cost) if cost > 0 else None,
            "taux_median": float(median / cost) if cost > 0 else None,
            "taux_upper": float(upper / cost) if cost > 0 else None,
        }

    # -----------------------------------------------------------------
    # BACKWARD COMPATIBILITY & MARGIN OPTIMIZATION
    # -----------------------------------------------------------------

    def recommend_margin(
        self,
        client,
        produit,
        quantite,
        source,
        achats=None,
    ):
        # Backward-compatible alias: achats is treated as cout_total.
        return self.recommend_price(
            client,
            produit,
            quantite,
            source,
            cout_total=achats,
        )

    # -----------------------------------------------------------------
    # EURO PRICE SCENARIOS (both sources)
    # -----------------------------------------------------------------

    def optimize_price(
        self,
        client: str,
        produit: str,
        quantite: float,
        *,
        cout_total: float | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        step: float = 50.0,
        source: str = "lbfi",
    ) -> pd.DataFrame:
        """Génère la grille de sensibilité sur le prix de vente total en euros."""
        source = source.strip().lower()

        if cout_total is None or cout_total <= 0:
            raise ValueError("cout_total must be > 0 €.")

        recommendation = self.recommend_price(
            client,
            produit,
            quantite,
            source,
            cout_total=cout_total,
        )

        center = float(recommendation["prix_median"])

        if min_price is None:
            min_price = max(cout_total, center * 0.60)

        if max_price is None:
            max_price = center * 1.40

        if step <= 0:
            raise ValueError("step must be > 0.")

        prices = np.arange(min_price, max_price + step * 0.5, step)

        rows = []
        for price in prices:
            p_val = float(price)
            probability = self.predict_acceptance_proba(
                client,
                produit,
                quantite,
                source,
                cout_total=cout_total,
                prix_total=p_val,
            )
            rows.append(
                {
                    "prix_total": p_val,
                    "acceptance_proba": probability,
                }
            )

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # PRICE WITH MAXIMUM ACCEPTANCE PROBABILITY
    # -----------------------------------------------------------------

    def max_acceptance_price(
        self,
        client: str,
        produit: str,
        quantite: float,
        *,
        cout_total: float,
        source: str = "ponceblanc",
        min_price: float | None = None,
        max_price: float | None = None,
        step: float | None = None,
        max_coeff: float = 3.0,
    ) -> dict:
        """
        Price on the grid with the highest predicted P(accept).

        If several prices share that maximum probability, the highest price
        among them is returned (prefer more revenue at the same acceptance).
        """
        source = source.strip().lower()
        cost = float(cout_total)
        if cost <= 0:
            raise ValueError("cout_total must be > 0 €.")

        recommendation = self.recommend_price(
            client,
            produit,
            quantite,
            source,
            cout_total=cost,
        )
        center = float(recommendation["prix_median"])

        if min_price is None:
            min_price = float(cost)
        if max_price is None:
            max_price = float(
                min(
                    cost * max_coeff,
                    max(center * 2.5, cost * 1.5),
                    500_000.0,
                )
            )
        if step is None:
            span = max_price - min_price
            step = max(10.0, float(round(span / 60.0, -1) or 10.0))
        if step <= 0:
            raise ValueError("step must be > 0.")

        prices = np.arange(min_price, max_price + step * 0.5, step)
        best_proba = -1.0
        best_price = float(center)

        for price in prices:
            p_val = float(price)
            proba = self.predict_acceptance_proba(
                client,
                produit,
                quantite,
                source,
                cout_total=cost,
                prix_total=p_val,
            )
            # Prefer higher proba; on ties, prefer higher price
            if proba > best_proba + 1e-12 or (
                abs(proba - best_proba) <= 1e-12 and p_val > best_price
            ):
                best_proba = proba
                best_price = p_val

        return {
            "prix": float(best_price),
            "proba": float(best_proba),
            "coeff": float(best_price / cost) if cost > 0 else None,
            "grid_min": float(min_price),
            "grid_max": float(max_price),
            "step": float(step),
        }

    def max_price_for_threshold(
        self,
        client: str,
        produit: str,
        quantite: float,
        *,
        cout_total: float,
        threshold: float = 0.50,
        source: str = "ponceblanc",
        min_price: float | None = None,
        max_price: float | None = None,
        step: float | None = None,
        max_coeff: float = 3.0,
    ) -> dict:
        """Highest price on the grid with P(accept) >= threshold.

        On the feasible set, returns the maximum price (most ambitious revenue
        that still clears the acceptance bar).
        """
        source = source.strip().lower()
        cost = float(cout_total)
        if cost <= 0:
            raise ValueError("cout_total must be > 0 €.")
        thr = float(threshold)
        if not (0.0 < thr <= 1.0):
            raise ValueError("threshold must be in (0, 1].")

        recommendation = self.recommend_price(
            client, produit, quantite, source,
            cout_total=cost,
        )
        center = float(recommendation["prix_median"])

        if min_price is None:
            min_price = float(cost)
        if max_price is None:
            max_price = float(
                min(cost * max_coeff, max(center * 2.5, cost * 1.5), 500_000.0)
            )
        if step is None:
            span = max_price - min_price
            step = max(10.0, float(round(span / 60.0, -1) or 10.0))
        if step <= 0:
            raise ValueError("step must be > 0.")

        best_price = None
        best_proba = -1.0
        fallback_price = float(center)
        fallback_proba = -1.0

        for price in np.arange(min_price, max_price + step * 0.5, step):
            p_val = float(price)
            proba = self.predict_acceptance_proba(
                client, produit, quantite, source,
                cout_total=cost, prix_total=p_val,
            )
            if proba >= fallback_proba:
                fallback_proba = proba
                fallback_price = p_val
            if proba + 1e-12 >= thr and (best_price is None or p_val >= best_price):
                best_price = p_val
                best_proba = proba

        if best_price is not None:
            return {
                "found": True,
                "prix": float(best_price),
                "proba": float(best_proba),
                "threshold": thr,
                "coeff": float(best_price / cost) if cost > 0 else None,
            }
        return {
            "found": False,
            "prix": float(fallback_price),
            "proba": float(fallback_proba),
            "threshold": thr,
            "coeff": float(fallback_price / cost) if cost > 0 else None,
        }

