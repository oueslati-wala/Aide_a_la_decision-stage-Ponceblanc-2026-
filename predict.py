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
        self._context_cache = {}

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

        def _load_json(name: str):
            path = directory / name
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
            return None

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
            "matiere_encoding": _load_json("matiere_encoding.json"),
            "format_encoding": _load_json("format_encoding.json"),
            "commercial_encoding": _load_json("commercial_encoding.json"),
            "commercial_target_encoding": _load_json("commercial_target_encoding.json"),
            "regressor_target": metadata.get("target"),
            "regressor_mode": metadata.get("mode"),
            "target_transform": metadata.get("target_transform", "identity"),
            "metrics": metrics,
        }

        self._cache[cache_key] = bundle
        return bundle

    @staticmethod
    def _feature_kwargs(bundle: dict) -> dict:
        return dict(
            matiere_encoding=bundle.get("matiere_encoding"),
            format_encoding=bundle.get("format_encoding"),
            commercial_encoding=bundle.get("commercial_encoding"),
            commercial_target_encoding=bundle.get("commercial_target_encoding"),
        )

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
        saison=None,
        matiere=None,
        format_dims=None,
        fournisseur=None,
        pression_concurrentielle=None,
        marge_cible=None,
        delai_livraison=None,
        month=None,
        year=None,
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
                    "saison": saison,
                    "matiere": matiere,
                    "format_dims": format_dims,
                    "fournisseur": fournisseur,
                    # commercial is the trained column name (Ponceblanc Commercial)
                    "commercial": fournisseur,
                    "pression_concurrentielle": pression_concurrentielle,
                    "marge_cible": marge_cible,
                    "delai_livraison": delai_livraison,
                    "month": month,
                    "year": year,
                }
            ]
        )

    def _context_adjustments(
        self,
        source: str,
        *,
        saison: str | None = None,
        pression_concurrentielle: float | int | None = None,
        marge_cible: float | int | None = None,
        delai_livraison: float | int | None = None,
        matiere: str | None = None,
        format_dims: str | None = None,
        fournisseur: str | None = None,
    ) -> tuple[float, float]:
        """No post-model heuristics. Season comes from month/trimester in the feature matrix."""
        return (1.0, 1.0)

    # -----------------------------------------------------------------
    # CLASSIFIER
    # -----------------------------------------------------------------

    def _predict_acceptance_batch(
        self,
        rows: pd.DataFrame,
        source: str,
        *,
        saison: str | None = None,
        matiere: str | None = None,
        format_dims: str | None = None,
        fournisseur: str | None = None,
        pression_concurrentielle: float | int | None = None,
        marge_cible: float | int | None = None,
        delai_livraison: float | int | None = None,
    ) -> np.ndarray:
        source = source.strip().lower()
        bundle = self._load_source(source)

        # Ensure extra columns exist on the batch rows for encoding
        for col, val in (
            ("matiere", matiere),
            ("format_dims", format_dims),
            ("commercial", fournisseur),
        ):
            if col not in rows.columns:
                rows = rows.copy()
                rows[col] = val

        X = features.build_feature_matrix(
            rows,
            bundle["product_clusters"],
            bundle["client_encoding"],
            include_price=True,
            log_price=True,
            include_cost=True,
            log_cost=True,
            **self._feature_kwargs(bundle),
        )

        classifier = bundle["classifier"]
        if hasattr(classifier, "feature_names_in_"):
            X = X.reindex(
                columns=list(classifier.feature_names_in_),
                fill_value=np.nan,
            )

        base_proba = classifier.predict_proba(X)[:, 1]
        _, proba_mult = self._context_adjustments(
            source,
            saison=saison,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible,
            delai_livraison=delai_livraison,
            matiere=matiere,
            format_dims=format_dims,
            fournisseur=fournisseur,
        )
        return np.clip(base_proba * proba_mult, 0.0, 0.9999)

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
        saison: str | None = None,
        matiere: str | None = None,
        format_dims: str | None = None,
        fournisseur: str | None = None,
        pression_concurrentielle: float | int | None = None,
        marge_cible: float | int | None = None,
        delai_livraison: float | int | None = None,
        month: int | float | None = None,
        year: int | float | None = None,
    ) -> float:
        source = source.strip().lower()

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
            saison=saison,
            matiere=matiere,
            format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible,
            delai_livraison=delai_livraison,
            month=month,
            year=year,
        )

        return float(self._predict_acceptance_batch(
            row,
            source,
            saison=saison,
            matiere=matiere,
            format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible,
            delai_livraison=delai_livraison,
        )[0])

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
        saison: str | None = None,
        matiere: str | None = None,
        format_dims: str | None = None,
        fournisseur: str | None = None,
        pression_concurrentielle: float | int | None = None,
        marge_cible: float | int | None = None,
        delai_livraison: float | int | None = None,
        month: int | float | None = None,
        year: int | float | None = None,
    ) -> dict:
        source = source.strip().lower()
        bundle = self._load_source(source)

        if cout_total is None or cout_total <= 0:
            raise ValueError(f"{source} price recommendation needs cout_total > 0 €.")

        cost = float(cout_total)
        price_mult, _ = self._context_adjustments(
            source,
            saison=saison,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible,
            delai_livraison=delai_livraison,
            matiere=matiere,
            format_dims=format_dims,
            fournisseur=fournisseur,
        )

        # Fallback: legacy coefficient-based recommendation.
        row = self._make_row(
            client,
            produit,
            quantite,
            source,
            cout_total=cost,
            saison=saison,
            matiere=matiere,
            format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible,
            delai_livraison=delai_livraison,
            month=month,
            year=year,
        )
        # Map UI fields onto the columns the trained encodings expect
        if "commercial" not in row.columns:
            row = row.copy()
            row["commercial"] = fournisseur
        X = features.build_feature_matrix(
            row,
            bundle["product_clusters"],
            bundle["client_encoding"],
            include_cost=True,
            log_cost=True,
            **self._feature_kwargs(bundle),
        )
        X = X.reindex(
            columns=bundle["regressor_features"],
            fill_value=np.nan,
        )
        c_lo = float(np.clip(bundle["reg_lower"].predict(X)[0], 0.80, 4.00))
        c_med = float(np.clip(bundle["reg_best"].predict(X)[0], 0.80, 4.00))
        c_hi = float(np.clip(bundle["reg_upper"].predict(X)[0], 0.80, 4.00))
        base_lower = max(cost, c_lo * cost)
        base_median = max(cost, c_med * cost)
        base_upper = max(cost, c_hi * cost)
        base_lower *= price_mult
        base_median *= price_mult
        base_upper *= price_mult

        # New logic: optimize expected margin contribution by evaluating a price grid.
        # This is done in a single batch to avoid repeated feature-building and
        # classifier calls for every candidate price.
        price_grid = np.linspace(
            max(cost * 0.90, cost),
            max(cost * 2.50, base_upper * 1.10),
            8,
        )

        rows = [
            self._make_row(
                client,
                produit,
                quantite,
                source,
                cout_total=cost,
                prix_total=float(price),
                saison=saison,
                matiere=matiere,
                format_dims=format_dims,
                fournisseur=fournisseur,
                pression_concurrentielle=pression_concurrentielle,
                marge_cible=marge_cible,
                delai_livraison=delai_livraison,
                month=month,
                year=year,
            )
            for price in price_grid
        ]
        rows_df = pd.concat(rows, ignore_index=True)
        probabilities = self._predict_acceptance_batch(
            rows_df,
            source,
            saison=saison,
            matiere=matiere,
            format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible,
            delai_livraison=delai_livraison,
        )

        margins = price_grid - cost
        valid = margins > 0
        scores = np.full_like(price_grid, -np.inf, dtype=float)
        scores[valid] = probabilities[valid] * margins[valid]

        idx = int(np.nanargmax(scores)) if np.any(valid) else 0
        best_price = float(price_grid[idx])
        best_proba = float(probabilities[idx])
        best_score = float(scores[idx]) if valid[idx] else -np.inf

        if not np.any(valid):
            best_price = float(base_median)
            best_proba = 0.5
            best_score = -np.inf

        best_price = float(np.clip(best_price * price_mult, cost * 0.90, max(cost * 2.50, base_upper * 1.10)))

        lower = max(cost, min(base_lower, best_price * 0.95))
        median = max(cost, best_price)
        upper = max(cost, max(base_upper, best_price * 1.05))

        return {
            "target": "prix_total",
            "mode": "expected_margin_grid",
            "prix_lower": float(lower),
            "prix_median": float(median),
            "prix_upper": float(upper),
            "cost_eur": cost,
            "coeff_median": float(median / cost) if cost > 0 else None,
            "taux_lower": float((lower - cost) / cost) if cost > 0 else None,
            "taux_median": float((median - cost) / cost) if cost > 0 else None,
            "taux_upper": float((upper - cost) / cost) if cost > 0 else None,
            "acceptance_probability": float(best_proba),
            "expected_margin_score": float(best_score),
            "legacy_coeff_fallback": {
                "coeff_low": float(c_lo),
                "coeff_median": float(c_med),
                "coeff_high": float(c_hi),
            },
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


    def _batch_price_grid(
        self,
        client: str,
        produit: str,
        quantite: float,
        source: str,
        *,
        cout_total: float,
        prices: np.ndarray,
        saison: str | None = None,
        matiere: str | None = None,
        format_dims: str | None = None,
        fournisseur: str | None = None,
        pression_concurrentielle: float | int | None = None,
        marge_cible: float | int | None = None,
        delai_livraison: float | int | None = None,
        month: int | float | None = None,
        year: int | float | None = None,
    ) -> np.ndarray:
        """Single batched classifier call for many candidate prices."""
        cost = float(cout_total)
        rows = [
            self._make_row(
                client, produit, quantite, source,
                cout_total=cost,
                prix_total=float(p),
                saison=saison,
                matiere=matiere,
                format_dims=format_dims,
                fournisseur=fournisseur,
                pression_concurrentielle=pression_concurrentielle,
                marge_cible=marge_cible,
                delai_livraison=delai_livraison,
                month=month,
                year=year,
            )
            for p in prices
        ]
        rows_df = pd.concat(rows, ignore_index=True)
        return self._predict_acceptance_batch(
            rows_df,
            source,
            saison=saison,
            matiere=matiere,
            format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible,
            delai_livraison=delai_livraison,
        )

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
        saison: str | None = None,
        matiere: str | None = None,
        format_dims: str | None = None,
        fournisseur: str | None = None,
        pression_concurrentielle: float | int | None = None,
        marge_cible: float | int | None = None,
        delai_livraison: float | int | None = None,
        month: int | float | None = None,
        year: int | float | None = None,
    ) -> pd.DataFrame:
        """Sensitivity grid — one batched classifier call."""
        source = source.strip().lower()
        if cout_total is None or cout_total <= 0:
            raise ValueError("cout_total must be > 0 €.")

        recommendation = self.recommend_price(
            client, produit, quantite, source,
            cout_total=cout_total,
            saison=saison, matiere=matiere, format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible, delai_livraison=delai_livraison,
            month=month, year=year,
        )
        center = float(recommendation["prix_median"])
        if min_price is None:
            min_price = max(cout_total, center * 0.60)
        if max_price is None:
            max_price = center * 1.40
        if step <= 0:
            raise ValueError("step must be > 0.")

        # Cap number of points for UI speed (max ~25)
        n = int(np.ceil((max_price - min_price) / step)) + 1
        if n > 25:
            step = (max_price - min_price) / 24.0
        prices = np.arange(min_price, max_price + step * 0.5, step)
        if len(prices) == 0:
            prices = np.array([center])

        probs = self._batch_price_grid(
            client, produit, quantite, source,
            cout_total=float(cout_total),
            prices=prices,
            saison=saison, matiere=matiere, format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible, delai_livraison=delai_livraison,
            month=month, year=year,
        )
        return pd.DataFrame({"prix_total": prices.astype(float), "acceptance_proba": probs})

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
        saison: str | None = None,
        matiere: str | None = None,
        format_dims: str | None = None,
        fournisseur: str | None = None,
        pression_concurrentielle: float | int | None = None,
        marge_cible: float | int | None = None,
        delai_livraison: float | int | None = None,
        month: int | float | None = None,
        year: int | float | None = None,
    ) -> dict:
        """Highest acceptance on a coarse grid (batched)."""
        source = source.strip().lower()
        cost = float(cout_total)
        if cost <= 0:
            raise ValueError("cout_total must be > 0 €.")

        recommendation = self.recommend_price(
            client, produit, quantite, source,
            cout_total=cost,
            saison=saison, matiere=matiere, format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible, delai_livraison=delai_livraison,
            month=month, year=year,
        )
        center = float(recommendation["prix_median"])
        if min_price is None:
            min_price = float(cost)
        if max_price is None:
            max_price = float(min(cost * max_coeff, max(center * 2.5, cost * 1.5), 500_000.0))
        # Coarse grid: at most 20 points
        if step is None:
            step = max(10.0, (max_price - min_price) / 19.0)
        prices = np.linspace(min_price, max_price, num=min(20, max(5, int((max_price - min_price) / step) + 1)))
        probs = self._batch_price_grid(
            client, produit, quantite, source,
            cout_total=cost, prices=prices,
            saison=saison, matiere=matiere, format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible, delai_livraison=delai_livraison,
            month=month, year=year,
        )
        # Prefer higher proba; on ties, higher price
        best_i = int(np.argmax(probs + prices / (prices.max() + 1.0) * 1e-9))
        best_price = float(prices[best_i])
        best_proba = float(probs[best_i])
        return {
            "prix": best_price,
            "proba": best_proba,
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
        saison: str | None = None,
        matiere: str | None = None,
        format_dims: str | None = None,
        fournisseur: str | None = None,
        pression_concurrentielle: float | int | None = None,
        marge_cible: float | int | None = None,
        delai_livraison: float | int | None = None,
        month: int | float | None = None,
        year: int | float | None = None,
    ) -> dict:
        """Highest price with P(accept) >= threshold (batched coarse grid)."""
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
            saison=saison, matiere=matiere, format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible, delai_livraison=delai_livraison,
            month=month, year=year,
        )
        center = float(recommendation["prix_median"])
        if min_price is None:
            min_price = float(cost)
        if max_price is None:
            max_price = float(min(cost * max_coeff, max(center * 2.5, cost * 1.5), 500_000.0))
        if step is None:
            step = max(10.0, (max_price - min_price) / 19.0)
        prices = np.linspace(min_price, max_price, num=min(20, max(5, int((max_price - min_price) / step) + 1)))
        probs = self._batch_price_grid(
            client, produit, quantite, source,
            cout_total=cost, prices=prices,
            saison=saison, matiere=matiere, format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible, delai_livraison=delai_livraison,
            month=month, year=year,
        )
        feasible = probs + 1e-12 >= thr
        if np.any(feasible):
            # highest price among feasible
            idx = int(np.where(feasible)[0][np.argmax(prices[feasible])])
            return {
                "found": True,
                "prix": float(prices[idx]),
                "proba": float(probs[idx]),
                "threshold": thr,
                "coeff": float(prices[idx] / cost) if cost > 0 else None,
            }
        # fallback: max acceptance
        best_i = int(np.argmax(probs))
        return {
            "found": False,
            "prix": float(prices[best_i]),
            "proba": float(probs[best_i]),
            "threshold": thr,
            "coeff": float(prices[best_i] / cost) if cost > 0 else None,
        }
