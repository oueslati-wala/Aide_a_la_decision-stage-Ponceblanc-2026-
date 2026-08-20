"""
vis.py — Estimateur d'offres (Streamlit)
========================================

Decision-support UI for quote pricing (Ponceblanc & LBFI).

Both sources use the same euro workflow:
    inputs  → client, produit, quantite, cout_total (€)
    outputs → prix_total recommandé (€), P(acceptation), scenarios, sensitivity

Model for both sources: regressor predicts a prix/coût coefficient on
accepted quotes, then prix = coefficient × coût. Margin rates shown in the
UI are derived for display only and are never model inputs.

Run from this directory:
    streamlit run vis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import features
from predict import QuoteEstimator

st.set_page_config(page_title="Estimateur d'offres", layout="wide")

MODELS_DIR = Path("models")

NBSP = "\u202f"  # narrow no-break space, standard French thousands separator


def fmt_eur(x: float | int | None, decimals: int = 0) -> str:
    """
    Format a euro amount to match the underlying data convention: "." is the
    decimal separator (never re-formatted to ","), and a space is used for
    thousands grouping. No comma is used anywhere, since a comma reads as a
    decimal point to a French user and previously caused amounts to be
    misread/mistyped by 1000x (e.g. "19,172 €" misread as ~19€ instead of
    19172€).
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    formatted = f"{x:,.{decimals}f}"  # US-style: comma thousands, period decimal
    formatted = formatted.replace(",", NBSP)  # swap thousands sep only; keep "."
    return f"{formatted} €"


@st.cache_resource
def get_estimator(_version: int = 9):
    return QuoteEstimator()


@st.cache_data
def get_history(source: str) -> pd.DataFrame:
    try:
        df = features.build_source(source)
        if df is None:
            return pd.DataFrame()
        return df
    except FileNotFoundError as e:
        st.error(str(e))
        return pd.DataFrame()


@st.cache_data
def get_metrics(source: str) -> dict | None:
    path = MODELS_DIR / source / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def get_encodings(source: str) -> tuple[dict | None, dict | None]:
    d = MODELS_DIR / source
    pc, ce = None, None

    if (d / "product_clusters.json").is_file():
        pc = json.loads((d / "product_clusters.json").read_text(encoding="utf-8"))

    if (d / "client_encoding.json").is_file():
        ce = json.loads((d / "client_encoding.json").read_text(encoding="utf-8"))

    return pc, ce


def clean_dropdown_values(
    history: pd.DataFrame,
    column: str,
) -> list[str]:
    if history.empty or column not in history.columns:
        return []

    values = history[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    values = values[values.str.lower() != "nan"]

    return sorted(values.unique().tolist())


def smooth_curve(curve: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = curve.copy()
    out["acceptance_proba_smooth"] = (
        out["acceptance_proba"]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    return out


def compute_margin(prix_total: float | None, cout_total: float | None) -> dict:
    """
    Derive a margin rate for LBFI from price and cost, purely for display —
    never fed back into the model (LBFI's model stays euro-only by design).

    - marge_eur: profit in euros (prix - coût)
    - coefficient: prix / coût, same scale/convention as Ponceblanc's
      taux_marge (roughly 0.8-3.0), so the two sources read comparably
    - taux_pct: markup as a percentage of cost, (prix - coût) / coût
    """
    if prix_total is None or cout_total is None or cout_total <= 0:
        return {"marge_eur": None, "coefficient": None, "taux_pct": None}

    marge_eur = float(prix_total) - float(cout_total)
    coefficient = float(prix_total) / float(cout_total)
    taux_pct = marge_eur / float(cout_total)

    return {
        "marge_eur": marge_eur,
        "coefficient": coefficient,
        "taux_pct": taux_pct,
    }


def fmt_coefficient(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}"


def fmt_pct_signed(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.0%}"


def interpret_proba(p: float) -> str:
    if p >= 0.70:
        return "Élevée — historique favorable pour ce profil."
    if p >= 0.45:
        return "Moyenne — zone d'incertitude ; montant et volume pèsent beaucoup."
    if p >= 0.25:
        return "Faible — profil souvent refusé ; ajuster le prix peut aider."
    return "Très faible — peu d'offres similaires ont été acceptées."


def probability_label(p: float) -> str:
    if p >= 0.70:
        return "🟢 Forte"
    if p >= 0.45:
        return "🟡 Moyenne"
    if p >= 0.25:
        return "🟠 Faible"
    return "🔴 Très faible"


def risk_label(p: float) -> str:
    if p >= 0.60:
        return "🟢 Risque bas"
    if p >= 0.40:
        return "🟡 Risque moyen"
    if p >= 0.25:
        return "🟠 Risque élevé"
    return "🔴 Risque très élevé"


def filter_similar(
    history: pd.DataFrame,
    client: str,
    produit: str,
    quantite: float,
    qty_tol: float = 0.5,
) -> pd.DataFrame:
    if history.empty:
        return history

    sub = history.copy()

    if client and client.strip():
        sub = sub[
            sub["client"].astype(str).str.contains(client.strip().upper(), na=False)
        ]

    if produit and produit.strip():
        sub = sub[
            sub["produit"].astype(str).str.contains(produit.strip().upper(), na=False)
        ]

    if len(sub) == 0:
        return sub

    sub = sub.assign(_qty_dist=(sub["quantite"] - quantite).abs())
    return sub.sort_values("_qty_dist")


def source_stats(df: pd.DataFrame, source: str) -> dict:
    acc = df[df["signe"] == 1]
    rej = df[df["signe"] == 0]

    out = {
        "n": len(df),
        "n_acc": len(acc),
        "n_rej": len(rej),
        "rate": float(df["signe"].mean()) if len(df) else 0.0,
    }

    margin_cost = df["cout_total"] if "cout_total" in df.columns else pd.Series(dtype=float)
    margin_price = df["prix_total"] if "prix_total" in df.columns else pd.Series(dtype=float)
    valid = margin_cost.notna() & (margin_cost > 0) & margin_price.notna() & (margin_price > 0)
    coeff = pd.Series(np.nan, index=df.index)
    if len(valid):
        coeff.loc[valid] = margin_price[valid] / margin_cost[valid]

    acc_coeff = coeff.loc[acc.index].dropna() if len(acc) else pd.Series(dtype=float)
    rej_coeff = coeff.loc[rej.index].dropna() if len(rej) else pd.Series(dtype=float)

    out.update({
        "prix_acc_med": float(acc["prix_total"].median()) if len(acc) and "prix_total" in acc and acc["prix_total"].notna().any() else None,
        "prix_acc_p10": float(acc["prix_total"].quantile(0.10)) if len(acc) and "prix_total" in acc and acc["prix_total"].notna().any() else None,
        "prix_acc_p90": float(acc["prix_total"].quantile(0.90)) if len(acc) and "prix_total" in acc and acc["prix_total"].notna().any() else None,
        "prix_rej_med": float(rej["prix_total"].median()) if len(rej) and "prix_total" in rej and rej["prix_total"].notna().any() else None,
        "coeff_acc_med": float(acc_coeff.median()) if len(acc_coeff) else None,
        "coeff_rej_med": float(rej_coeff.median()) if len(rej_coeff) else None,
    })

    return out


page = st.sidebar.radio(
    "Navigation",
    [
        "Aide à la décision",
        "Performance des modèles",
        "Guide",
    ],
    index=0,
)

est = get_estimator()


# ===========================================================================
# PAGE 1 — Decision support
# ===========================================================================

if page == "Aide à la décision":

    st.title("Aide à la décision")

    source = st.radio(
        "Source",
        ["ponceblanc", "lbfi"],
        horizontal=True,
        format_func=lambda s: "Ponceblanc" if s == "ponceblanc" else "LBFI",
    )

    # Production model for both sources: coefficient × cost → prix (€).
    history = get_history(source)
    metrics = get_metrics(source)
    product_clusters, client_encoding = get_encodings(source)

    

    st.header("1. Profil de l'offre")

    known_clients = clean_dropdown_values(history, "client")
    known_produits = clean_dropdown_values(history, "produit")

    # Defaults chosen so the recommended price lands in a comfortable
    # acceptance zone (~70–80%) for a frequent client/product pair.
    if source == "ponceblanc":
        preferred_client, preferred_produit = "GERFLOR", "NUANCIER"
        default_qty, default_cost = 1000, 6000.0
    else:
        preferred_client, preferred_produit = "ROUGE_GORGE", "ECHANTILLONNAGE"
        default_qty, default_cost = 400, 800.0

    c1, c2 = st.columns(2)

    with c1:
        default_c = (
            preferred_client
            if preferred_client in known_clients
            else (known_clients[0] if known_clients else preferred_client)
        )
        client_options = list(known_clients)
        if default_c not in client_options:
            client_options.insert(0, default_c)

        client = st.selectbox(
            "Client",
            options=client_options,
            index=client_options.index(default_c),
            accept_new_options=True,
            key=f"client_{source}",
            help="Tapez pour filtrer la liste, ou saisissez un client nouveau.",
        )
        client = str(client).strip()

        default_p = (
            preferred_produit
            if preferred_produit in known_produits
            else (known_produits[0] if known_produits else preferred_produit)
        )
        prod_options = list(known_produits)
        if default_p not in prod_options:
            prod_options.insert(0, default_p)

        produit = st.selectbox(
            "Produit / type",
            options=prod_options,
            index=prod_options.index(default_p),
            accept_new_options=True,
            key=f"produit_{source}",
            help="Tapez pour filtrer la liste, ou saisissez un produit nouveau.",
        )
        produit = str(produit).strip()

    with c2:
        quantite = st.number_input(
            "Quantité (Nb exemplaires)",
            min_value=1,
            value=default_qty,
            step=50,
            key=f"quantite_{source}",
        )

        cout_total = st.number_input(
            "Coût total du devis (€)",
            min_value=0.0,
            value=default_cost,
            step=10.0,
            key=f"cout_{source}",
            help=(
                "Achat + fabrication + transport pour l'ensemble du devis. "
                "Saisissez le montant en chiffres seuls, sans espace ni virgule "
                "de séparation des milliers (ex. 19172, pas 19 172 ni 19,172)."
            ),
        )
        prix_candidat = None

    with st.expander("Filtres pour l'historique comparable (n'entraînent pas le modèle)"):
        f1, f2, f3 = st.columns(3)
        with f1:
            qty_band = st.slider(
                "Tolérance quantité pour « similaire »",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
            )
        with f2:
            only_accepted = st.checkbox("Historique : acceptés seulement", value=False)
        with f3:
            min_n_hint = st.number_input(
                "Seuil d'alerte N similaires",
                min_value=1,
                value=10,
            )

    similar = filter_similar(history, client, produit, quantite)

    if not similar.empty and qty_band < 1.0:
        lo_q = quantite * (1 - qty_band)
        hi_q = quantite * (1 + qty_band)
        similar = similar[(similar["quantite"] >= lo_q) & (similar["quantite"] <= hi_q)]

    if only_accepted and not similar.empty:
        similar = similar[similar["signe"] == 1]

    stats = source_stats(similar, source)

    st.header("2. Proposition du modèle")

    if cout_total <= 0:
        st.warning("Renseignez un coût total supérieur à 0 €.")
        st.stop()

    try:
        rec = est.recommend_price(
            client=client,
            produit=produit,
            quantite=quantite,
            source=source,
            cout_total=cout_total,
        )

        prix_recommande = float(rec["prix_median"])

        proba_reco = est.predict_acceptance_proba(
            client=client,
            produit=produit,
            quantite=quantite,
            source=source,
            cout_total=cout_total,
            prix_total=prix_recommande,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prix recommandé", fmt_eur(prix_recommande))
        m2.metric("Coût total", fmt_eur(cout_total))
        marge_reco = compute_margin(prix_recommande, cout_total)
        m3.metric(
            "Marge à la reco",
            fmt_eur(marge_reco["marge_eur"]),
            delta=fmt_pct_signed(marge_reco["taux_pct"]),
            help="Marge en € et en % du coût. Coefficient (prix/coût) : "
                 f"{fmt_coefficient(marge_reco['coefficient'])}.",
        )
        m4.metric("P(acceptation) à la reco", f"{proba_reco:.0%}", help=interpret_proba(proba_reco))

        st.success(
            f"{probability_label(proba_reco)} — Prix recommandé : **{fmt_eur(prix_recommande)}**"
        )
        st.caption(
            f"Intervalle indicatif du modèle : {fmt_eur(rec['prix_lower'])} — {fmt_eur(rec['prix_upper'])}."
        )

    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        st.stop()

    # Explanations
    st.subheader("Pourquoi ces chiffres ?")
    client_key = client.strip().upper()
    produit_key = produit.strip().upper()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Signaux que le modèle utilise**")
        bullets = []

        if client_encoding:
            client_rate = client_encoding.get(client_key, client_encoding.get("__GLOBAL__"))
            global_rate = client_encoding.get("__GLOBAL__", metrics["acceptance_rate"] if metrics else None)
            if client_key in client_encoding:
                bullets.append(f"- **Client `{client_key}`** : taux historique lissé ≈ **{client_rate:.0%}**.")
            else:
                bullets.append(f"- **Client `{client_key}` inconnu** : taux global appliqué (≈ **{global_rate:.0%}**).")

        if product_clusters:
            if produit_key in product_clusters:
                bullets.append(f"- **Produit `{produit_key}`** : cluster #{product_clusters[produit_key]}.")
            else:
                bullets.append(f"- **Produit `{produit_key}`** : cluster « autres » (produit rare/nouveau).")

        bullets.append(f"- **Quantité {quantite}** : prend en compte l'effet de volume.")
        bullets.append(f"- **Coût total {fmt_eur(cout_total)}** : entrée du régresseur (coeff × coût → prix).")

        st.markdown("\n".join(bullets))

    with col_b:
        st.markdown("**Ce que la recommandation représente**")
        label = "Ponceblanc" if source == "ponceblanc" else "LBFI"
        st.markdown(
            f"""
Le modèle **{label}** prédit un **coefficient prix/coût**, converti en **prix total en euros**.
- **{fmt_eur(prix_recommande)}** = montant typique parmi les devis **acceptés** pour ce coût et ce profil.
- Le coût entre en feature du régresseur ; le taux de marge n'est pas une entrée.
"""
        )

    # Historical proof
    st.subheader("3. Preuves historiques pour ce profil")
    if stats["n"] == 0:
        st.warning("Aucun devis historique comparable trouvé après filtres.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Devis similaires", stats["n"])
        k2.metric("Acceptés", f"{stats['n_acc']} ({stats['rate']:.0%})")

        k3.metric("Prix médian ACCEPTÉS", fmt_eur(stats['prix_acc_med']) if stats['prix_acc_med'] else "n/a")
        k4.metric("Prix médian REFUSÉS", fmt_eur(stats['prix_rej_med']) if stats['prix_rej_med'] else "n/a")

        k5, k6 = st.columns(2)
        k5.metric(
            "Coefficient marge médian ACCEPTÉS",
            fmt_coefficient(stats['coeff_acc_med']),
            help="Médiane de prix / coût sur les devis acceptés similaires.",
        )
        k6.metric(
            "Coefficient marge médian REFUSÉS",
            fmt_coefficient(stats['coeff_rej_med']),
            help="Médiane de prix / coût sur les devis refusés similaires.",
        )

    # Scenarios
    st.subheader("4. Scénarios de décision")

    # Hard bounds: models were trained on realistic quote ranges.
    # Anything outside this is treated as out-of-distribution.
    MAX_COEFF = 5.0
    MAX_PRICE = max(cout_total * MAX_COEFF, prix_recommande * 2.5)
    MAX_PRICE = min(MAX_PRICE, 500_000.0)  # absolute ceiling for the UI
    MIN_PRICE = 0.0

    def _scenario_row(name: str, p_val: float) -> dict:
        p_val = float(p_val)
        if p_val <= 0:
            return {
                "Scénario": name,
                "Prix de vente total (€)": p_val,
                "Marge (€)": None,
                "Taux de marge": None,
                "P(acceptation)": "n/a",
                "Risque": "—",
                "Lecture": "Saisissez un prix supérieur à 0 €.",
            }
        if p_val < cout_total:
            margin = compute_margin(p_val, cout_total)
            return {
                "Scénario": name,
                "Prix de vente total (€)": p_val,
                "Marge (€)": margin["marge_eur"],
                "Taux de marge": margin["coefficient"],
                "P(acceptation)": "n/a",
                "Risque": "—",
                "Lecture": f"⚠️ Prix inférieur au coût total ({fmt_eur(cout_total)}).",
            }

        coeff = p_val / cout_total if cout_total > 0 else None
        # Out-of-distribution: extreme markup or price the model never saw
        if coeff is not None and coeff > MAX_COEFF:
            margin = compute_margin(p_val, cout_total)
            return {
                "Scénario": name,
                "Prix de vente total (€)": p_val,
                "Marge (€)": margin["marge_eur"],
                "Taux de marge": margin["coefficient"],
                "P(acceptation)": "~0%",
                "Risque": "🔴 Hors zone",
                "Lecture": (
                    f"⚠️ Prix hors plage réaliste (coeff {coeff:.1f}× > {MAX_COEFF:.0f}×). "
                    "Le modèle n'a pas d'historique comparable — ne pas faire confiance à la proba."
                ),
            }

        proba = est.predict_acceptance_proba(
            client=client,
            produit=produit,
            quantite=quantite,
            source=source,
            cout_total=cout_total,
            prix_total=p_val,
        )
        margin = compute_margin(p_val, cout_total)
        return {
            "Scénario": name,
            "Prix de vente total (€)": p_val,
            "Marge (€)": margin["marge_eur"],
            "Taux de marge": margin["coefficient"],
            "P(acceptation)": f"{proba:.0%}",
            "Risque": risk_label(proba),
            "Lecture": interpret_proba(proba),
        }

    # Ambitieux = highest price that still keeps P(accept) at least as high
    # as the recommended price (push revenue without worsening acceptance vs reco).
    # Meilleure P(accept) = price that maximises acceptance probability.
    max_acc = None
    ambitieux_price = round(prix_recommande * 1.10, -1)
    try:
        max_acc = est.max_acceptance_price(
            client=client,
            produit=produit,
            quantite=quantite,
            cout_total=cout_total,
            source=source,
            max_coeff=min(MAX_COEFF, 3.0),
        )
        ambitieux = est.max_price_for_threshold(
            client=client,
            produit=produit,
            quantite=quantite,
            cout_total=cout_total,
            threshold=float(proba_reco),
            source=source,
            max_coeff=min(MAX_COEFF, 3.0),
        )
        if ambitieux.get("found"):
            ambitieux_price = float(ambitieux["prix"])
        else:
            # No grid point matches reco proba: fall back to max-acceptance price
            ambitieux_price = float(ambitieux["prix"])
    except (ValueError, FileNotFoundError) as exc:
        st.warning(f"Scénarios optimisés indisponibles : {exc}")

    st.caption(
        "**Recommandé** = prix typique accepté (régresseur). "
        "**Ambitieux** = prix **le plus haut** dont la P(accept) reste ≥ celle du recommandé "
        f"({proba_reco:.0%}). "
        "**Meilleure P(accept)** = prix qui maximise la proba (à égalité : le plus haut)."
    )

    anchor_prices = {
        "Prudent (viser l'acceptation)": max(cout_total, round(prix_recommande * 0.90, -1)),
        "Recommandé (modèle)": prix_recommande,
        "Ambitieux (plus élevé)": ambitieux_price,
    }
    if max_acc is not None:
        anchor_prices[
            f"Meilleure P(accept) ({max_acc['proba']:.0%})"
        ] = float(max_acc["prix"])

    # Session key is source+cost specific so a typed value for one quote
    # does not leak into another (and absurd leftovers get wiped).
    choice_key = f"votre_choix_prix_{source}"
    cost_key = f"votre_choix_cost_{source}"
    default_choice = float(max(cout_total, round(prix_recommande, -1)))

    if (
        choice_key not in st.session_state
        or cost_key not in st.session_state
        or abs(float(st.session_state.get(cost_key, 0)) - float(cout_total)) > 1.0
        or float(st.session_state.get(choice_key, 0)) > MAX_PRICE
        or float(st.session_state.get(choice_key, 0)) < 0
    ):
        st.session_state[choice_key] = default_choice
        st.session_state[cost_key] = float(cout_total)

    rows = [_scenario_row(name, p_val) for name, p_val in anchor_prices.items()]
    rows.append(_scenario_row("Votre choix", st.session_state[choice_key]))

    display_df = pd.DataFrame(rows)

    target_note = "Le régresseur actif prédit le **coefficient / prix** (coût en feature)."
    st.caption(
        "Modifiez le prix de la ligne **Votre choix** pour tester un montant précis ; les autres lignes "
        f"sont fixées par le modèle. {target_note} "
        f"Plafond UI : {fmt_eur(MAX_PRICE)} (≈ {MAX_COEFF:.0f}× le coût)."
    )

    edited_df = st.data_editor(
        display_df,
        column_config={
            "Scénario": st.column_config.TextColumn(disabled=True),
            "Prix de vente total (€)": st.column_config.NumberColumn(
                min_value=MIN_PRICE,
                max_value=float(MAX_PRICE),
                step=10.0,
                format="%.2f €",
            ),
            "Marge (€)": st.column_config.NumberColumn(disabled=True, format="%.2f €"),
            "Taux de marge": st.column_config.NumberColumn(disabled=True, format="%.2f"),
            "P(acceptation)": st.column_config.TextColumn(disabled=True),
            "Risque": st.column_config.TextColumn(disabled=True),
            "Lecture": st.column_config.TextColumn(disabled=True),
        },
        disabled=["Scénario", "Marge (€)", "Taux de marge", "P(acceptation)", "Risque", "Lecture"],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key=f"scenario_editor_{source}",
    )

    new_choice_price = float(
        edited_df.loc[edited_df["Scénario"] == "Votre choix", "Prix de vente total (€)"].iloc[0]
    )
    # Clamp anything the widget still lets through
    new_choice_price = max(MIN_PRICE, min(float(MAX_PRICE), new_choice_price))
    if abs(new_choice_price - float(st.session_state[choice_key])) > 0.01:
        st.session_state[choice_key] = new_choice_price
        st.rerun()


    # Sensibilité
    st.subheader("5. Sensibilité")
    st.caption(
        f"Le coût total reste fixé à **{fmt_eur(cout_total)}**. La courbe fait varier "
        f"uniquement le **prix de vente total (€)** facturé au client, et montre "
        f"comment la probabilité d'acceptation évolue selon ce prix."
    )
    step_val = st.number_input("Pas de variation du prix de vente (€)", min_value=5.0, value=50.0, step=10.0)
    try:
        curve = est.optimize_price(
            client=client,
            produit=produit,
            quantite=quantite,
            cout_total=cout_total,
            step=step_val,
            source=source,
        )
        chart = curve.set_index("prix_total")[["acceptance_proba"]].rename(
            columns={"acceptance_proba": "P(acceptation)"}
        )
        chart.index.name = "Prix de vente total (€)"
        st.line_chart(chart)
    except (ValueError, AttributeError) as exc:
        st.error(str(exc))

    # History Table
    st.subheader("6. Devis historiques comparables")
    if similar.empty:
        st.info("Aucun devis comparable.")
    else:
        cols = [c for c in ["devis_code", "client", "produit", "quantite", "cout_total", "prix_total", "signe"] if c in similar.columns]
        disp = similar[cols].head(20).copy()
        if "cout_total" in disp:
            disp["cout_total"] = disp["cout_total"].map(fmt_eur)
        if "prix_total" in disp:
            disp["prix_total"] = disp["prix_total"].map(fmt_eur)
        if "signe" in disp:
            disp["signe"] = disp["signe"].map({1: "Accepté", 0: "Refusé"})
        disp = disp.rename(columns={"devis_code": "Devis", "client": "Client", "produit": "Produit", "quantite": "Quantité", "cout_total": "Coût total", "prix_total": "Prix de vente total", "signe": "Résultat"})
        st.dataframe(disp, width="stretch", hide_index=True)


# ===========================================================================
# PAGE 2 — Performance
# ===========================================================================

elif page == "Performance des modèles":

    st.title("Performance des modèles")

    for src in ("ponceblanc", "lbfi"):
        title = {
            "ponceblanc": "PONCEBLANC",
            "lbfi": "LBFI",
        }.get(src, src.upper())
        st.subheader(title)
        metrics = get_metrics(src)

        if not metrics:
            st.warning(f"metrics.json introuvable pour {src}.")
            continue

        classifier = metrics.get("classifier") or {}
        regressor = metrics.get("margin_regressor") or {}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Devis utilisables", f"{metrics.get('n_total', 0):,}")
        c2.metric("Taux acceptation global", f"{metrics.get('acceptance_rate', 0):.0%}")
        c3.metric("Accuracy (test)", f"{classifier.get('accuracy', 0):.1%}")
        auc = classifier.get("roc_auc")
        c4.metric("ROC-AUC (test)", f"{auc:.3f}" if auc else "n/a")

        mode = regressor.get("mode") or metrics.get("regressor_mode") or "—"
        target = regressor.get("target") or metrics.get("regressor_target") or "—"
        st.write(f"Cible régresseur : **{target}** — mode **{mode}**")
        mae = regressor.get("mae")
        mae_c = regressor.get("mae_coeff")
        if mae is not None:
            st.write(f"MAE prix (sur acceptés) : **{fmt_eur(mae, decimals=0)}**")
        if mae_c is not None:
            st.write(f"MAE coefficient : **{mae_c:.3f}**")
        n_reg = regressor.get("n_rows")
        if n_reg is not None:
            st.caption(f"Lignes utilisées pour le régresseur (acceptés avec coût) : {n_reg:,}")
        st.caption(
            "Entrées du régresseur : quantite, cluster produit, client encodé, cout_total. "
            "Le coefficient / taux n'est pas une entrée — il est prédit puis multiplié par le coût."
        )

        st.divider()


# ===========================================================================
# PAGE 3 — Guide
# ===========================================================================

else:

    st.title("Guide d'utilisation")

    st.markdown(
        """
## Objectif

Aider le commercial à **choisir un prix de vente total** (en euros) pour un devis,
en s'appuyant sur l'historique des offres acceptées / refusées de la même source
(Ponceblanc ou LBFI).

## Ce que vous saisissez

| Champ | Rôle |
|-------|------|
| **Source** | Ponceblanc ou LBFI (modèles entraînés séparément) |
| **Client** | Nom client (liste historique + saisie libre) |
| **Produit / type** | Type de produit |
| **Quantité** | Nombre d'exemplaires |
| **Coût total (€)** | Achat + fabrication + transport pour tout le devis |

## Ce que le modèle renvoie

1. **Prix recommandé (€)** — montant typique parmi les devis **acceptés** pour un profil proche (coût, client, produit, volume).
2. **P(acceptation)** — probabilité qu'une offre à ce prix soit signée, d'après le classifieur.
3. **Scénarios** — Prudent / Recommandé / Ambitieux / Meilleure P(accept), plus une ligne **Votre choix** éditable.
4. **Courbe de sensibilité** — P(acceptation) en fonction du prix de vente, coût fixé.
5. **Devis historiques comparables** — extrait filtré pour justification.

## Principe technique (commun aux deux sources)

- **Régresseur** : prédit un **coefficient** (prix / coût), puis  
  `prix_recommandé = coefficient × coût_total`.
- **Classifieur** : estime P(acceptation | client, produit, quantité, coût, prix candidat).
- Les taux de marge affichés sont **uniquement pour lecture** ; ils ne sont pas des entrées du modèle.

## Limites à garder en tête

- Peu d'historique pour un client / produit rare → confiance plus faible.
- Prix très éloignés du coût (coeff > ~3–5×) sont hors zone d'entraînement.
- Le modèle ne remplace pas le jugement commercial ; il quantifie l'habitude historique.

## Lancer l'application

```bash
cd <dossier_du_projet>
pip install -r requirements.txt
streamlit run vis.py
```
"""
    )

    st.caption("Estimateur d'offres — modèles entraînés sur l'historique devis Ponceblanc & LBFI.")
