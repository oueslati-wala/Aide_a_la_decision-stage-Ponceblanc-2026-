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

# Soft post-model decay on P(accept) for unrealistic coefficients.
# Starts after the historical mass (≈ p90) so expected_margin is not forced
# onto a hard cliff; only extreme coeffs are progressively zeroed for display
# and for grid scores above the realistic band.
# Defaults; overridden per-source from training quantiles when available.
ACCEPTANCE_DECAY_START = 1.90
ACCEPTANCE_DECAY_ZERO = 2.80


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
            "coeff_priors": _load_json("coeff_priors.json") or {},
            "regressor_target": metadata.get("target"),
            "regressor_mode": metadata.get("mode"),
            "target_transform": metadata.get("target_transform", "identity"),
            "coeff_clip": metadata.get("coeff_clip") or [0.80, 4.00],
            "coeff_p50": metadata.get("coeff_p50"),
            "coeff_p75": metadata.get("coeff_p75"),
            "coeff_p90": metadata.get("coeff_p90"),
            "small_n_blend": float(metadata.get("small_n_blend") or 0.0),
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
        """No post-model heuristics. Season comes from 4-month blocks (season_4m) in the feature matrix."""
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
            include_unit_price=True,
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

        # Soft decay above historical mass (see module-level constants /
        # per-source p90). Keeps extreme coeffs from looking "acceptable".
        decay_start, decay_zero = self._decay_bounds(source)
        coeff = (
            pd.to_numeric(rows["prix_total"], errors="coerce")
            / pd.to_numeric(rows["cout_total"], errors="coerce").replace(0, np.nan)
        ).to_numpy(dtype=float)
        span = max(decay_zero - decay_start, 1e-6)
        decay_frac = np.clip((coeff - decay_start) / span, 0.0, 1.0)
        decay_mult = np.where(np.isnan(coeff), 1.0, 1.0 - decay_frac)

        return np.clip(base_proba * proba_mult * decay_mult, 0.0, 0.9999)

    def _decay_bounds(self, source: str) -> tuple[float, float]:
        """Decay starts just above historical p90; zeros toward the outlier clip."""
        try:
            bundle = self._load_source(source)
        except Exception:
            return ACCEPTANCE_DECAY_START, ACCEPTANCE_DECAY_ZERO
        p90 = bundle.get("coeff_p90")
        hi_c = (bundle.get("coeff_clip") or [0.8, 4.0])[-1]
        if p90 is None:
            # Fallbacks from last training quantiles
            p90 = {"ponceblanc": 1.80, "lbfi": 1.30}.get(source, ACCEPTANCE_DECAY_START)
        start = float(p90) * 1.02
        zero = min(float(hi_c), float(p90) * 1.45)
        if zero <= start:
            zero = start + 0.35
        return start, zero

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
        pricing_mode: str = "balanced",
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
        """
        Recommend a total selling price.

        pricing_mode
        ------------
        "balanced" (default) — ambitious but realistic:
            start from the regressor coeff, then maximise P(accept)×margin
            only up to ~+12–15 % above that (and never above the source's
            historical p75 accepted coeff). Middle ground between pure
            history and full expected-margin aggression.
        "regressor"
            prix = predicted coefficient × cost (historical median).
        "expected_margin"
            maximises P(accept)×(prix−coût) up to the trained coeff_clip
            (tends highest).
        """
        source = source.strip().lower()
        mode = (pricing_mode or "balanced").strip().lower()
        if mode not in ("regressor", "expected_margin", "balanced"):
            mode = "balanced"
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
        priors = bundle.get("coeff_priors") or {}
        if priors:
            global_med = float(priors.get("global", 1.3))
            by_c = priors.get("by_client") or {}
            by_p = priors.get("by_produit") or {}
            c_key = str(client).strip().upper()
            p_key = str(features.normalize_produit(str(produit).strip())).upper()
            c_prior = float(by_c.get(c_key, global_med))
            p_prior = float(by_p.get(p_key, global_med))
            X = X.copy()
            X["client_coeff_prior"] = c_prior
            X["produit_coeff_prior"] = p_prior
            X["hier_coeff_prior"] = (c_prior + p_prior) / 2.0
        X = X.reindex(
            columns=bundle["regressor_features"],
            fill_value=np.nan,
        )
        lo_c, hi_c = bundle.get("coeff_clip") or [0.80, 4.00]
        lo_c, hi_c = float(lo_c), float(hi_c)
        transform = bundle.get("target_transform", "identity")
        blend = float(bundle.get("small_n_blend") or 0.0)

        def _decode(raw: float) -> float:
            val = float(np.exp(raw)) if transform == "log" else float(raw)
            return float(np.clip(val, lo_c, hi_c))

        c_lo_m = _decode(bundle["reg_lower"].predict(X)[0])
        c_med_m = _decode(bundle["reg_best"].predict(X)[0])
        c_hi_m = _decode(bundle["reg_upper"].predict(X)[0])
        if blend > 0 and "hier_coeff_prior" in X.columns:
            hier = float(X["hier_coeff_prior"].iloc[0])
            c_lo = float(np.clip(blend * hier + (1 - blend) * c_lo_m, lo_c, hi_c))
            c_med = float(np.clip(blend * hier + (1 - blend) * c_med_m, lo_c, hi_c))
            c_hi = float(np.clip(blend * hier + (1 - blend) * c_hi_m, lo_c, hi_c))
            if c_lo > c_med:
                c_lo = c_med * 0.92
            if c_hi < c_med:
                c_hi = c_med * 1.08
        else:
            c_lo, c_med, c_hi = c_lo_m, c_med_m, c_hi_m

        reg_lower = max(cost, c_lo * cost) * price_mult
        reg_median = max(cost, c_med * cost) * price_mult
        reg_upper = max(cost, c_hi * cost) * price_mult
        reg_lower = min(reg_lower, reg_median)
        reg_upper = max(reg_upper, reg_median)

        # Realistic commercial ceilings from training quantiles (not outlier clip).
        # PB ≈ p75 1.5 / p90 1.80 ; LBFI ≈ p75 1.22 / p90 1.30
        priors_global = float((bundle.get("coeff_priors") or {}).get("global") or c_med)
        hist_p75 = float(
            bundle.get("coeff_p75")
            or {"ponceblanc": 1.50, "lbfi": 1.22}.get(source, priors_global * 1.08)
        )
        hist_p90 = float(
            bundle.get("coeff_p90")
            or {"ponceblanc": 1.80, "lbfi": 1.30}.get(source, priors_global * 1.15)
        )
        # Never search above the outlier clip either
        realistic_hi = min(float(hi_c), float(hist_p90))

        def _grid_pick(grid_lo: float, grid_hi: float, n: int = 10):
            grid_lo = max(float(grid_lo), cost * 1.02)
            grid_hi = max(grid_lo * 1.01, float(grid_hi))
            price_grid = np.linspace(grid_lo, grid_hi, n)
            rows = [
                self._make_row(
                    client, produit, quantite, source,
                    cout_total=cost, prix_total=float(price),
                    saison=saison, matiere=matiere, format_dims=format_dims,
                    fournisseur=fournisseur,
                    pression_concurrentielle=pression_concurrentielle,
                    marge_cible=marge_cible, delai_livraison=delai_livraison,
                    month=month, year=year,
                )
                for price in price_grid
            ]
            rows_df = pd.concat(rows, ignore_index=True)
            probabilities = self._predict_acceptance_batch(
                rows_df, source,
                saison=saison, matiere=matiere, format_dims=format_dims,
                fournisseur=fournisseur,
                pression_concurrentielle=pression_concurrentielle,
                marge_cible=marge_cible, delai_livraison=delai_livraison,
            )
            margins = price_grid - cost
            valid = margins > 0
            scores = np.full_like(price_grid, -np.inf, dtype=float)
            scores[valid] = probabilities[valid] * margins[valid]
            idx = int(np.nanargmax(scores)) if np.any(valid) else int(
                np.argmin(np.abs(price_grid - reg_median))
            )
            return (
                float(price_grid[idx]),
                float(probabilities[idx]),
                float(scores[idx]) if valid[idx] else float("nan"),
            )

        if mode == "expected_margin":
            # Cap at historical p90 — NOT the outlier clip (was causing ~1.8
            # or the clip ceiling every time when paired with hard decay).
            try:
                median, best_proba, best_score = _grid_pick(
                    cost * max(lo_c, 1.02),
                    cost * realistic_hi * price_mult,
                    n=14,
                )
            except Exception:
                median, best_proba, best_score = float(reg_median), float("nan"), float("nan")
            lower = max(cost, min(reg_lower, median * 0.95))
            upper = max(cost, max(reg_upper, median * 1.05))
            result_mode = "expected_margin"
        elif mode == "balanced":
            # Ambitious but realistic: between regressor median and min(+12 %, p75).
            cap_price = cost * min(hist_p75, realistic_hi) * price_mult
            band_hi = max(float(reg_median), min(float(reg_median) * 1.12, float(cap_price)))
            if band_hi <= reg_median * 1.001:
                band_hi = max(reg_median * 1.05, min(cap_price, reg_median * 1.12))
            band_lo = float(reg_median)
            try:
                median, best_proba, best_score = _grid_pick(band_lo, band_hi, n=8)
            except Exception:
                target_coeff = min(c_med * 1.08, hist_p75)
                median = max(cost, target_coeff * cost) * price_mult
                best_proba, best_score = float("nan"), float("nan")
                try:
                    best_proba = float(
                        self.predict_acceptance_proba(
                            client=client, produit=produit, quantite=quantite,
                            source=source, cout_total=cost, prix_total=median,
                            saison=saison, matiere=matiere, format_dims=format_dims,
                            fournisseur=fournisseur,
                            pression_concurrentielle=pression_concurrentielle,
                            marge_cible=marge_cible, delai_livraison=delai_livraison,
                            month=month, year=year,
                        )
                    )
                except Exception:
                    pass
            lower = max(cost, min(reg_lower, median * 0.95))
            upper = max(cost, max(reg_upper, median * 1.05, band_hi))
            result_mode = "balanced"
        else:
            lower, median, upper = float(reg_lower), float(reg_median), float(reg_upper)
            best_score = float("nan")
            try:
                best_proba = float(
                    self.predict_acceptance_proba(
                        client=client, produit=produit, quantite=quantite,
                        source=source, cout_total=cost, prix_total=median,
                        saison=saison, matiere=matiere, format_dims=format_dims,
                        fournisseur=fournisseur,
                        pression_concurrentielle=pression_concurrentielle,
                        marge_cible=marge_cible, delai_livraison=delai_livraison,
                        month=month, year=year,
                    )
                )
            except Exception:
                best_proba = float("nan")
            result_mode = "regressor_coeff"

        return {
            "target": "prix_total",
            "mode": result_mode,
            "pricing_mode": mode,
            "prix_lower": float(lower),
            "prix_median": float(median),
            "prix_upper": float(upper),
            "cost_eur": cost,
            "coeff_median": float(median / cost) if cost > 0 else None,
            "taux_lower": float((lower - cost) / cost) if cost > 0 else None,
            "taux_median": float((median - cost) / cost) if cost > 0 else None,
            "taux_upper": float((upper - cost) / cost) if cost > 0 else None,
            "acceptance_probability": float(best_proba) if np.isfinite(best_proba) else None,
            "expected_margin_score": float(best_score) if np.isfinite(best_score) else None,
            "regressor_coeff": {
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
        cost = float(cout_total)
        if min_price is None:
            # From near cost — explore low margins too
            min_price = max(cost * 1.02, cost)
        if max_price is None:
            # Wide range so the UI can probe high coeffs
            # (up to ~4× cost or 3× recommendation, hard-capped at 500k €).
            max_price = float(min(max(cost * 4.0, center * 3.0), 500_000.0))
        if step <= 0:
            raise ValueError("step must be > 0.")

        # Cap number of points for UI speed (max ~40)
        n = int(np.ceil((max_price - min_price) / step)) + 1
        if n > 40:
            step = (max_price - min_price) / 39.0
        prices = np.arange(min_price, max_price + step * 0.5, step)
        if len(prices) == 0:
            prices = np.array([center])

        probs = self._batch_price_grid(
            client, produit, quantite, source,
            cout_total=cost,
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

    # -----------------------------------------------------------------
    # CLIENT-ACCEPTANCE STRATEGIC PRICING (business policy layer)
    # -----------------------------------------------------------------
    #
    # This block is intentionally separate from recommend_price(). It does
    # NOT change what the model has learned; it applies a portfolio-level
    # business decision on top of it: for clients whose smoothed historical
    # acceptance rate sits in the bottom slice of the source's client base,
    # quote a lower price aimed at a target P(accept), trading margin on
    # THIS quote for a higher chance of winning the client -- on the
    # assumption that the margin gap is recovered elsewhere in the
    # portfolio. Nothing here is presented as "learned by the model".

    def client_acceptance_rate(
        self,
        client: str,
        source: str,
    ) -> dict:
        """
        Smoothed historical acceptance rate for a client (same client_encoding
        used by the classifier), plus the source-wide global rate for context.
        Does not require cout_total / prix_total -- just looks up the client.
        """
        source = source.strip().lower()
        bundle = self._load_source(source)
        ce = bundle["client_encoding"]
        global_rate = float(ce.get("__GLOBAL__", 0.0))
        client_key = str(client).strip().upper()
        known = client_key in ce
        rate = float(ce.get(client_key, global_rate))
        return {
            "client": client_key,
            "known": known,
            "client_rate": rate,
            "global_rate": global_rate,
        }

    def low_acceptance_threshold(
        self,
        source: str,
        percentile: float = 25.0,
    ) -> float | None:
        """
        Data-driven cutoff: the `percentile`-th percentile of smoothed
        acceptance rates across all clients seen for this source. Clients
        at or below this rate are considered "low acceptance". Returns
        None if there is no client encoding to draw from.
        """
        source = source.strip().lower()
        bundle = self._load_source(source)
        ce = bundle["client_encoding"]
        rates = np.array(
            [v for k, v in ce.items() if k != "__GLOBAL__"],
            dtype=float,
        )
        if rates.size == 0:
            return None
        return float(np.percentile(rates, percentile))

    def list_low_acceptance_clients(
        self,
        source: str,
        percentile: float = 25.0,
        min_display: int = 0,
    ) -> pd.DataFrame:
        """
        Inspection helper: shows exactly which clients fall at/below the
        percentile threshold and at what smoothed rate, so the threshold
        can be sanity-checked against real names before it drives pricing.
        """
        source = source.strip().lower()
        bundle = self._load_source(source)
        ce = bundle["client_encoding"]
        threshold = self.low_acceptance_threshold(source, percentile)
        rows = [
            {"client": k, "client_rate": float(v)}
            for k, v in ce.items()
            if k != "__GLOBAL__"
        ]
        df = pd.DataFrame(rows).sort_values("client_rate")
        if threshold is not None:
            df = df[df["client_rate"] <= threshold]
        if min_display:
            df = df.head(min_display)
        df["threshold_used"] = threshold
        df["global_rate"] = ce.get("__GLOBAL__")
        return df.reset_index(drop=True)

    def is_low_acceptance_client(
        self,
        client: str,
        source: str,
        percentile: float = 25.0,
    ) -> tuple[bool, float, float | None]:
        info = self.client_acceptance_rate(client, source)
        threshold = self.low_acceptance_threshold(source, percentile)
        if threshold is None:
            return False, info["client_rate"], None
        return (info["client_rate"] <= threshold), info["client_rate"], threshold

    def recommend_price_strategic(
        self,
        client: str,
        produit: str,
        quantite: float,
        source: str,
        *,
        cout_total: float,
        pricing_mode: str = "balanced",
        low_acceptance_percentile: float = 25.0,
        max_discount_pct: float = 0.40,
        min_discount_pct: float = 0.15,
        min_coeff: float = 1.05,
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
        """
        Client-acquisition pricing strategy — portfolio policy layer, not a
        model prediction.

        For most clients this returns the same recommendation as
        recommend_price(), with strategic_pricing_applied=False.

        For a client whose smoothed acceptance rate falls at/below the
        `low_acceptance_percentile` of this source's client base, it
        **forces a lower margin** by applying a severity-based discount
        on the baseline recommended price:

            severity = (threshold − client_rate) / threshold   ∈ [0, 1]
            discount = min_discount + severity × (max_discount − min_discount)

        Defaults: 15 % minimum discount → up to 40 % for the worst clients.
        Price is never allowed below cost × min_coeff (default 1.05).

        This does NOT rely on the classifier's price elasticity (which can
        be weak or inverted for some clients). It is an explicit business
        rule: trade margin on this quote to improve the chance of winning
        a historically hard client.
        """
        source = source.strip().lower()
        cost = float(cout_total)
        if cost <= 0:
            raise ValueError(f"{source} needs cout_total > 0 €.")

        baseline = self.recommend_price(
            client, produit, quantite, source,
            cout_total=cost,
            pricing_mode=pricing_mode,
            saison=saison, matiere=matiere, format_dims=format_dims,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible, delai_livraison=delai_livraison,
            month=month, year=year,
        )

        is_low, client_rate, threshold = self.is_low_acceptance_client(
            client, source, low_acceptance_percentile
        )

        result = dict(baseline)
        result["client_acceptance_rate"] = client_rate
        result["low_acceptance_threshold"] = threshold
        result["strategic_pricing_applied"] = False

        if not is_low:
            return result

        baseline_price = float(baseline["prix_median"])
        if baseline_price <= cost:
            result["strategic_note"] = (
                "Client flagged as low-acceptance, but baseline price is "
                "already at/near cost — no discount room."
            )
            return result

        # Severity: 0 at the threshold, 1 when rate → 0
        if threshold and threshold > 0:
            severity = float(np.clip((threshold - client_rate) / threshold, 0.0, 1.0))
        else:
            severity = 0.5

        min_d = float(min_discount_pct)
        max_d = float(max_discount_pct)
        if max_d < min_d:
            max_d = min_d
        discount = min_d + severity * (max_d - min_d)

        # Floor: never go below cost × min_coeff
        floor_price = cost * float(min_coeff)
        strategic_price = max(floor_price, baseline_price * (1.0 - discount))

        # If floor already ≥ baseline, nothing to do
        if strategic_price >= baseline_price - 1e-6:
            result["strategic_note"] = (
                "Client flagged as low-acceptance, but baseline is already "
                "near the cost floor — no further discount applied."
            )
            return result

        actual_discount = (baseline_price - strategic_price) / baseline_price

        # Recompute P(accept) at the strategic price for display
        try:
            strategic_proba = float(
                self.predict_acceptance_proba(
                    client=client,
                    produit=produit,
                    quantite=quantite,
                    source=source,
                    cout_total=cost,
                    prix_total=strategic_price,
                    month=month,
                    year=year,
                )
            )
        except Exception:
            strategic_proba = result.get("acceptance_probability")

        result.update({
            "prix_median": float(strategic_price),
            "prix_lower": float(min(float(result.get("prix_lower", strategic_price)), strategic_price)),
            "prix_upper": float(result.get("prix_upper", strategic_price)),
            "acceptance_probability": strategic_proba,
            "coeff_median": float(strategic_price / cost) if cost > 0 else None,
            "taux_median": float((strategic_price - cost) / cost) if cost > 0 else None,
            "strategic_pricing_applied": True,
            "strategic_severity": float(severity),
            "strategic_baseline_price": baseline_price,
            "strategic_discount_pct": float(actual_discount),
            "strategic_target_proba": None,
            "strategic_target_reached": None,
        })
        return result