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

st.set_page_config(page_title="Estimateur d'offres", layout="wide", page_icon="🧾")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #FBF6EF;
    --bg-alt: #F2E9DB;
    --card: #FFFDF9;
    --card-border: #E7D9C4;
    --ink: #2B241C;
    --ink-soft: #746553;
    --ink-faint: #A6987F;
    --accent: #A9552F;
    --accent-dark: #7A3C1F;
    --accent-soft: #F0DAC1;
    --sage: #64735A;
    --sage-soft: #DEE5D3;
    --amber: #B3822F;
    --amber-soft: #F3E3BE;
    --rose: #A2483E;
    --rose-soft: #F0D6CF;
    --radius-lg: 20px;
    --radius-md: 14px;
    --shadow: 0 1px 2px rgba(43, 36, 28, 0.04), 0 8px 24px rgba(43, 36, 28, 0.06);
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
    background: var(--bg-alt);
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }

h1, h2, h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}
h1 { font-weight: 600 !important; }
h2 {
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    margin-top: 2.2rem !important;
    padding-top: 0.6rem;
    border-top: 1px solid var(--card-border);
}
h2::before {
    content: "";
    display: block;
    width: 34px;
    height: 3px;
    background: var(--accent);
    border-radius: 3px;
    margin-bottom: 0.6rem;
}
h3 { font-weight: 600 !important; }

.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.35rem;
}
.hero-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--ink);
    line-height: 1.08;
    margin: 0 0 0.5rem 0;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1.02rem;
    color: var(--ink-soft);
    max-width: 640px;
    margin-bottom: 1.6rem;
}

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 1rem 1.1rem;
    box-shadow: var(--shadow);
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ink-soft) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--card-border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow);
    padding: 0.4rem 0.4rem;
}

[data-testid="stExpander"] {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow);
}

[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border: 1px solid var(--card-border);
    border-radius: var(--radius-md);
    overflow: hidden;
}

[data-testid="stAlertContentSuccess"] {
    background: var(--sage-soft) !important;
    border-left: 3px solid var(--sage) !important;
    border-radius: var(--radius-md);
    color: var(--ink) !important;
}
[data-testid="stAlertContentWarning"] {
    background: var(--amber-soft) !important;
    border-left: 3px solid var(--amber) !important;
    border-radius: var(--radius-md);
    color: var(--ink) !important;
}
[data-testid="stAlertContentError"] {
    background: var(--rose-soft) !important;
    border-left: 3px solid var(--rose) !important;
    border-radius: var(--radius-md);
    color: var(--ink) !important;
}
[data-testid="stAlertContentInfo"] {
    background: var(--accent-soft) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: var(--radius-md);
    color: var(--ink) !important;
}

[data-testid="stRadio"] > div { gap: 0.5rem; }
[data-testid="stRadio"] label {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    transition: all 0.15s ease;
}
[data-testid="stRadio"] label:hover { border-color: var(--accent); }

[data-baseweb="input"], [data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: var(--card-border) !important;
}

.stButton > button, .stDownloadButton > button {
    background: var(--accent);
    color: #FFF8F0;
    border: none;
    border-radius: 999px;
    padding: 0.5rem 1.3rem;
    font-weight: 600;
    box-shadow: var(--shadow);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--accent-dark);
    color: #FFF8F0;
}

hr { border-color: var(--card-border) !important; }

[data-testid="stCaptionContainer"] {
    color: var(--ink-faint) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
def get_estimator(_version: int = 16):
    return QuoteEstimator()


@st.cache_data
def get_history(source: str, _version: int = 16) -> pd.DataFrame:
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



@st.cache_data(show_spinner=False)
def cached_recommend(
    source: str,
    client: str,
    produit: str,
    quantite: float,
    cout_total: float,
    month: int,
    year: int,
    _version: int = 16,
):
    """Cache price recommendation for identical inputs (fast UI)."""
    est = get_estimator()
    rec = est.recommend_price(
        client=client,
        produit=produit,
        quantite=quantite,
        source=source,
        cout_total=cout_total,
        month=month,
        year=year,
    )
    proba = est.predict_acceptance_proba(
        client=client,
        produit=produit,
        quantite=quantite,
        source=source,
        cout_total=cout_total,
        prix_total=float(rec["prix_median"]),
        month=month,
        year=year,
    )
    return rec, float(proba)


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

    st.markdown(
        """
        <div class="hero-eyebrow">Ponceblanc · LBFI — Pricing devis</div>
        <div class="hero-title">Aide à la décision</div>
        <div class="hero-sub">Un prix recommandé et une probabilité d'acceptation, calculés
        à partir de l'historique des devis signés et refusés.</div>
        """,
        unsafe_allow_html=True,
    )

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
        preferred_client, preferred_produit = "GERFLOR", "LIASSE"
        default_qty, default_cost = 1000, 6000.0
    else:
        preferred_client, preferred_produit = "ROUGE_GORGE", "ECHANTILLONNAGE"
        default_qty, default_cost = 400, 800.0

    col_a, col_b, col_c = st.columns(3)

    with col_a:
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
            help="Tapez pour filtrer la liste, ou saisissez un client.",
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
            help="Tapez pour filtrer la liste, ou saisissez un produit.",
        )
        produit = features.normalize_produit(str(produit).strip())
        if not isinstance(produit, str):
            produit = str(produit)

        matiere = None
        format_ = None
        fournisseur = None
        # How many historical rows for this (normalized) product type
        if not history.empty and "produit" in history.columns and produit:
            _n_prod = int((history["produit"].astype(str).str.upper() == str(produit).upper()).sum())
            _aliases = [k for k, v in features.PRODUIT_ALIASES.items() if v == str(produit).upper()]
            _alias_txt = f" (variantes fusionnées : {', '.join(_aliases)})" if _aliases else ""
            st.caption(f"**{produit}** : **{_n_prod}** devis dans l'historique de cette source{_alias_txt}.")

        quantite = st.number_input(
                    "Quantité (Nb exemplaires)",
                    min_value=1,
                    value=default_qty,
                    step=50,
                    key=f"quantite_{source}",
                )


    with col_b:
        cout_achat = st.number_input(
            "Coût achat (€)",
            min_value=0.0,
            value=default_cost * 0.40,
            step=10.0,
            key=f"achat_{source}",
        )
        cout_fabrication = st.number_input(
            "Coût fabrication (€)",
            min_value=0.0,
            value=default_cost * 0.25,
            step=10.0,
            key=f"fabrication_{source}",
        )
        cout_transport = st.number_input(
            "Coût transport (€)",
            min_value=0.0,
            value=default_cost * 0.15,
            step=10.0,
            key=f"transport_{source}",
        )
        cout_total = float(cout_achat + cout_fabrication + cout_transport)
        st.number_input(
            "Coût total calculé (€)",
            min_value=0.0,
            value=float(cout_total),
            step=10.0,
            key=f"cout_{source}",
            disabled=True,
            help="Total = achat + fabrication + transport.",
        )

    with col_c:
        
        # --- Date ↔ saison (4 mois) keep in sync ---
        season_labels = {
            1: "S1 — Janvier à Avril",
            2: "S2 — Mai à Août",
            3: "S3 — Septembre à Décembre",
        }
        season_mid_month = {1: 2, 2: 6, 3: 10}  # representative month for the model
        date_key = f"date_{source}"
        season_key = f"season4m_{source}"

        def _season_from_month(m: int) -> int:
            return min(3, max(1, (int(m) - 1) // 4 + 1))

        # Init session defaults once
        if date_key not in st.session_state:
            st.session_state[date_key] = __import__("datetime").date.today()
        if season_key not in st.session_state:
            st.session_state[season_key] = _season_from_month(st.session_state[date_key].month)

        # Detect which control the user last changed via dedicated flags
        def _on_date_change():
            d = st.session_state.get(date_key)
            if d is not None:
                st.session_state[season_key] = _season_from_month(d.month)

        def _on_season_change():
            s = int(st.session_state.get(season_key, 1))
            d = st.session_state.get(date_key)
            if d is not None:
                # keep year, move day to mid-month of selected season
                mid = season_mid_month[s]
                import datetime as _dt
                day = min(d.day, 28)
                st.session_state[date_key] = _dt.date(d.year, mid, day)

        date_devis = st.date_input(
            "Date du devis",
            key=date_key,
            on_change=_on_date_change,
            help="Change la saison automatiquement (S1/S2/S3).",
        )
        season_4m = st.selectbox(
            "Saison (4 mois)",
            options=[1, 2, 3],
            format_func=lambda s: season_labels[s],
            key=season_key,
            on_change=_on_season_change,
            help="Change la date vers le milieu de la saison (même année).",
        )
        month = season_mid_month[int(season_4m)]
        year = int(date_devis.year) if date_devis else 2024

        # No manual heuristics — season comes only from month/trimester above
        saison = None
        pression_concurrentielle = None
        marge_cible = None
        delai_livraison = None

        prix_candidat = None

    st.caption(
        "Entrées modèle : client, type de produit, matière, quantité, "
        "coûts (achat + fabrication + transport), date → mois/trimestre (saison)."
    )

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

    if stats["n"] < min_n_hint:
        st.warning(
            f"Seulement {stats['n']} devis similaires après filtres (seuil : {min_n_hint}). "
            "Le prix est donc basé sur un échantillon faible, et la recommandation doit être interprétée avec prudence."
        )

    st.header("2. Proposition du modèle")

    if cout_total <= 0:
        st.warning("Renseignez un coût total supérieur à 0 €.")
        st.stop()

    try:
        rec, proba_reco = cached_recommend(
            source=source,
            client=client,
            produit=produit,
            quantite=float(quantite),
            cout_total=float(cout_total),
            month=int(month),
            year=int(year),
        )
        prix_recommande = float(rec["prix_median"])

        prix_unitaire_reco = (
            prix_recommande / float(quantite) if quantite and float(quantite) > 0 else 0.0
        )

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prix recommandé (total)", fmt_eur(prix_recommande))
            m2.metric(
                "Prix unitaire (reco)",
                f"{prix_unitaire_reco:,.4f} €".replace(",", " "),
                help="Prix total recommandé ÷ quantité",
            )
            m3.metric("Coût total", fmt_eur(cout_total))
            marge_reco = compute_margin(prix_recommande, cout_total)
            m4.metric("P(acceptation)", f"{proba_reco:.0%}", help=interpret_proba(proba_reco))

            st.success(
                f"{probability_label(proba_reco)} — Prix recommandé : "
                f"**{fmt_eur(prix_recommande)}** "
                f"({prix_unitaire_reco:,.4f} € / exemplaire)".replace(",", " ")
            )
            st.caption(
                f"Intervalle : {fmt_eur(rec['prix_lower'])} — {fmt_eur(rec['prix_upper'])}."
            )

    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        st.stop()

    # Explanations
    st.subheader("Pourquoi ces chiffres ?")
    client_key = client.strip().upper()
    produit_key = produit.strip().upper()

    explain_card = st.container(border=True)
    col_a, col_b = explain_card.columns(2)

    with col_a:
        st.markdown("**Signaux du modèle**")
        bullets = []
        if client_encoding:
            client_rate = client_encoding.get(client_key, client_encoding.get("__GLOBAL__"))
            global_rate = client_encoding.get("__GLOBAL__", metrics["acceptance_rate"] if metrics else None)
            if client_key in client_encoding:
                bullets.append(f"- **Client {client_key}** : taux d'acceptation lissé ≈ **{client_rate:.0%}**.")
            else:
                bullets.append(f"- **Client {client_key} inconnu** : taux global ≈ **{global_rate:.0%}**.")
        if product_clusters:
            if produit_key in product_clusters:
                bullets.append(f"- **Type produit {produit_key}** : cluster #{product_clusters[produit_key]}.")
            else:
                bullets.append(f"- **Type produit {produit_key}** : rare → profil plus volatil.")
        bullets.append(f"- **Quantité** : {quantite}")
        bullets.append(f"- **Coût total** : {fmt_eur(cout_total)} (achat + fab + transport)")
        season_code = int((month - 1) // 4) + 1
        season_names = {1: "S1 Jan–Avr", 2: "S2 Mai–Août", 3: "S3 Sep–Déc"}
        bullets.append(f"- **Saison** : {season_names.get(season_code, season_code)} (année {year})")
        st.markdown("\n".join(bullets))

    with col_b:
        label = "Ponceblanc" if source == "ponceblanc" else "LBFI"
        auc = (metrics or {}).get("classifier", {}).get("roc_auc")
        auc_txt = f"{auc:.3f}" if auc is not None else "n/a"
        st.markdown("**Lecture**")
        st.markdown(
            f"""
Modèle **{label}** (ROC-AUC test ≈ **{auc_txt}**).

Entrées **uniquement** :
client, type de produit, quantité, coût total, date → saison 4 mois (S1/S2/S3).

**Non utilisés** : matière, délai offre→décision, format, commercial, heuristiques.

- Prix recommandé : **{fmt_eur(prix_recommande)}**
- P(acceptation) : **{proba_reco:.0%}**
- Intervalle : **{fmt_eur(rec['prix_lower'])} — {fmt_eur(rec['prix_upper'])}**
"""
        )

    # Historical proof
    st.subheader("3. Preuves historiques pour ce profil")
    if stats["n"] == 0:
        st.warning("Aucun devis historique comparable trouvé après filtres.")
    else:
        with st.container(border=True):
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
    # --- Saisie prix de vente unitaire (→ total = unitaire × quantité) ---
    st.subheader("4. Votre prix de vente unitaire")
    pu_key = f"prix_unitaire_saisie_{source}"
    default_pu = float(prix_recommande / quantite) if quantite and quantite > 0 else 0.0
    if pu_key not in st.session_state:
        st.session_state[pu_key] = round(default_pu, 4)

    # When recommended price / qty change a lot, refresh default if user hasn't locked
    cost_sync_key = f"pu_sync_cost_{source}"
    if (
        cost_sync_key not in st.session_state
        or abs(float(st.session_state.get(cost_sync_key, 0)) - float(cout_total)) > 1.0
    ):
        st.session_state[pu_key] = round(default_pu, 4)
        st.session_state[cost_sync_key] = float(cout_total)

    col_pu, col_pt = st.columns(2)
    with col_pu:
        prix_unitaire_saisi = st.number_input(
            "Prix de vente unitaire (€ / exemplaire)",
            min_value=0.0,
            value=float(st.session_state[pu_key]),
            step=0.01,
            format="%.4f",
            key=f"pu_input_{source}",
            help="Saisissez le prix de vente par exemplaire. Le total = unitaire × quantité.",
        )
        st.session_state[pu_key] = float(prix_unitaire_saisi)
    with col_pt:
        prix_total_from_unit = float(prix_unitaire_saisi) * float(quantite)
        st.metric("Prix de vente total correspondant", fmt_eur(prix_total_from_unit))
        st.caption(f"= {prix_unitaire_saisi:.4f} € × {int(quantite)} exemplaires")

    # Drive "Votre choix" from unit price
    choice_key = f"votre_choix_prix_{source}"
    st.session_state[choice_key] = float(prix_total_from_unit)

    st.subheader("5. Scénarios de décision")


    # Hard bounds: models were trained on realistic quote ranges.
    # Anything outside this is treated as out-of-distribution.
    MAX_COEFF = 5.0
    MAX_PRICE = max(cout_total * MAX_COEFF, prix_recommande * 2.5)
    MAX_PRICE = min(MAX_PRICE, 500_000.0)  # absolute ceiling for the UI
    MIN_PRICE = 0.0

    def _scenario_row(name: str, p_val: float) -> dict:
        p_val = float(p_val)
        if p_val <= 0:
            pu = p_val / float(quantite) if quantite and float(quantite) > 0 else 0.0
            return {
                "Scénario": name,
                "Prix total (€)": p_val,
                "Prix unitaire (€)": round(pu, 4),
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
                "Prix total (€)": p_val,
                "Prix unitaire (€)": round(p_val / float(quantite), 4) if quantite and float(quantite) > 0 else 0.0,
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
                "Prix total (€)": p_val,
                "Prix unitaire (€)": round(p_val / float(quantite), 4) if quantite and float(quantite) > 0 else 0.0,
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
            saison=saison,
            matiere=matiere,
            format_dims=format_,
            fournisseur=fournisseur,
            pression_concurrentielle=pression_concurrentielle,
            marge_cible=marge_cible,
            month=month,
            year=year,
            delai_livraison=delai_livraison,
        )
        margin = compute_margin(p_val, cout_total)
        return {
            "Scénario": name,
            "Prix total (€)": p_val,
                "Prix unitaire (€)": round(p_val / float(quantite), 4) if quantite and float(quantite) > 0 else 0.0,
            "Marge (€)": margin["marge_eur"],
            "Taux de marge": margin["coefficient"],
            "P(acceptation)": f"{proba:.0%}",
            "Risque": risk_label(proba),
            "Lecture": interpret_proba(proba),
        }

    # Fast scenarios: one batched grid, no nested max_* loops
    max_acc = None
    ambitieux_price = round(prix_recommande * 1.10, -1)
    try:
        grid_min = float(cout_total)
        grid_max = float(min(cout_total * min(MAX_COEFF, 3.0), max(prix_recommande * 2.0, cout_total * 1.5), 500_000.0))
        grid_prices = np.linspace(grid_min, grid_max, 16)
        rows_grid = []
        for gp in grid_prices:
            rows_grid.append(
                est._make_row(
                    client, produit, quantite, source,
                    cout_total=cout_total, prix_total=float(gp),
                    matiere=matiere, month=month, year=year,
                )
            )
        grid_df = pd.concat(rows_grid, ignore_index=True)
        grid_probs = est._predict_acceptance_batch(grid_df, source, matiere=matiere)
        # max acceptance
        best_i = int(np.argmax(grid_probs + grid_prices / (grid_prices.max() + 1) * 1e-9))
        max_acc = {
            "prix": float(grid_prices[best_i]),
            "proba": float(grid_probs[best_i]),
        }
        # ambitious: highest price with proba >= reco proba
        feasible = grid_probs + 1e-12 >= float(proba_reco)
        if np.any(feasible):
            fi = int(np.where(feasible)[0][np.argmax(grid_prices[feasible])])
            ambitieux_price = float(grid_prices[fi])
        else:
            ambitieux_price = float(max_acc["prix"])
    except Exception as exc:
        st.warning(f"Scénarios optimisés indisponibles : {exc}")

    st.caption(
        "**Recommandé** = prix typique accepté (régresseur + grille). "
        "**Ambitieux** = prix le plus haut dont la P(accept) reste ≥ celle du recommandé "
        f"({proba_reco:.0%}). "
        "**Meilleure P(accept)** = prix qui maximise la proba."
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

    # "Votre choix" follows the unit selling price input above
    choice_key = f"votre_choix_prix_{source}"
    st.session_state[choice_key] = float(prix_total_from_unit)

    # Batch scenario probabilities (one classifier call)
    scenario_items = list(anchor_prices.items()) + [("Votre choix (via PU)", st.session_state[choice_key])]
    scen_prices = np.array([float(v) for _, v in scenario_items], dtype=float)
    try:
        scen_rows = [
            est._make_row(
                client, produit, quantite, source,
                cout_total=cout_total, prix_total=float(pv),
                matiere=matiere, month=month, year=year,
            )
            for pv in scen_prices
        ]
        scen_probs = est._predict_acceptance_batch(
            pd.concat(scen_rows, ignore_index=True), source, matiere=matiere
        )
    except Exception:
        scen_probs = np.full(len(scen_prices), np.nan)

    rows = []
    for i, (name, p_val) in enumerate(scenario_items):
        p_val = float(p_val)
        margin = compute_margin(p_val, cout_total)
        if p_val < cout_total:
            pu = p_val / float(quantite) if quantite and float(quantite) > 0 else 0.0
            rows.append({
                "Scénario": name,
                "Prix total (€)": p_val,
                "Prix unitaire (€)": round(pu, 4),
                "Marge (€)": margin["marge_eur"],
                "Taux de marge": margin["coefficient"],
                "P(acceptation)": "n/a",
                "Risque": "—",
                "Lecture": f"⚠️ Prix inférieur au coût ({fmt_eur(cout_total)}).",
            })
            continue
        proba = float(scen_probs[i]) if np.isfinite(scen_probs[i]) else 0.0
        pu = p_val / float(quantite) if quantite and float(quantite) > 0 else 0.0
        rows.append({
            "Scénario": name,
            "Prix total (€)": p_val,
            "Prix unitaire (€)": round(pu, 4),
            "Marge (€)": margin["marge_eur"],
            "Taux de marge": margin["coefficient"],
            "P(acceptation)": f"{proba:.0%}",
            "Risque": risk_label(proba),
            "Lecture": interpret_proba(proba),
        })

    display_df = pd.DataFrame(rows)

    st.caption(
        "Modifiez le prix de la ligne **Votre choix** pour tester un montant précis. "
        f"Plafond UI : {fmt_eur(MAX_PRICE)}."
    )

    edited_df = st.data_editor(
        display_df,
        column_config={
            "Scénario": st.column_config.TextColumn(disabled=True),
            "Prix total (€)": st.column_config.NumberColumn(
                min_value=MIN_PRICE,
                max_value=float(MAX_PRICE),
                step=10.0,
                format="%.2f €",
            ),
            "Prix unitaire (€)": st.column_config.NumberColumn(
                disabled=True,
                format="%.4f €",
            ),
            "Marge (€)": st.column_config.NumberColumn(disabled=True, format="%.2f €"),
            "Taux de marge": st.column_config.NumberColumn(disabled=True, format="%.2f"),
            "P(acceptation)": st.column_config.TextColumn(disabled=True),
            "Risque": st.column_config.TextColumn(disabled=True),
            "Lecture": st.column_config.TextColumn(disabled=True),
        },
        disabled=["Scénario", "Prix unitaire (€)", "Marge (€)", "Taux de marge", "P(acceptation)", "Risque", "Lecture"],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key=f"scenario_editor_{source}",
    )

    new_choice_price = float(
        edited_df.loc[
            edited_df["Scénario"].astype(str).str.startswith("Votre choix"),
            "Prix total (€)",
        ].iloc[0]
    )
    new_choice_price = max(MIN_PRICE, min(float(MAX_PRICE), new_choice_price))
    if abs(new_choice_price - float(st.session_state[choice_key])) > 0.01:
        st.session_state[choice_key] = new_choice_price
        # keep unit price in sync
        if quantite and float(quantite) > 0:
            st.session_state[pu_key] = round(new_choice_price / float(quantite), 4)
        st.rerun()


    # Sensibilité — only compute when expander is open (saves a full grid on every tweak)
    st.subheader("6. Sensibilité")
    with st.expander("Afficher la courbe P(acceptation) vs prix", expanded=False):
        st.caption(
            f"Coût fixé à **{fmt_eur(cout_total)}**. Variation du prix de vente total uniquement."
        )
        step_val = st.number_input(
            "Pas de variation du prix (€)", min_value=5.0, value=100.0, step=10.0,
            key=f"step_sens_{source}",
        )
        try:
            curve = est.optimize_price(
                client=client,
                produit=produit,
                quantite=quantite,
                cout_total=cout_total,
                step=step_val,
                source=source,
                matiere=matiere,
                month=month,
                year=year,
            )
            chart = curve.set_index("prix_total")[["acceptance_proba"]].rename(
                columns={"acceptance_proba": "P(acceptation)"}
            )
            chart.index.name = "Prix de vente total (€)"
            st.line_chart(chart)
        except (ValueError, AttributeError) as exc:
            st.error(str(exc))

    # History Table
    st.subheader("7. Devis historiques comparables")
    if similar.empty:
        st.info("Aucun devis comparable après filtres (client / produit / quantité).")
    else:
        n_sim = len(similar)
        n_acc = int((similar["signe"] == 1).sum()) if "signe" in similar.columns else 0
        n_rej = int((similar["signe"] == 0).sum()) if "signe" in similar.columns else 0
        n_clients = int(similar["client"].nunique()) if "client" in similar.columns else 0
        n_produits = int(similar["produit"].nunique()) if "produit" in similar.columns else 0
        qty_min = float(similar["quantite"].min()) if "quantite" in similar.columns else None
        qty_max = float(similar["quantite"].max()) if "quantite" in similar.columns else None

        # Same product type only (exact) vs current filter set
        exact_prod = similar[similar["produit"].astype(str).str.upper() == str(produit).upper()] if "produit" in similar.columns else similar
        n_exact = len(exact_prod)

        # Two rows so values (esp. qty range) are not truncated
        r1 = st.columns(3)
        r1[0].metric("Similaires", f"{n_sim}")
        r1[1].metric("Même produit", f"{n_exact}")
        r1[2].metric("Clients distincts", f"{n_clients}")

        r2 = st.columns(3)
        r2[0].metric("Acceptés", f"{n_acc}")
        r2[1].metric("Refusés", f"{n_rej}")
        if qty_min is not None:
            r2[2].metric(
                "Quantité",
                f"{qty_min:,.0f} – {qty_max:,.0f}".replace(",", " "),
            )
        else:
            r2[2].metric("Quantité", "—")

        st.caption(
            f"Tableau : jusqu'à **20** lignes sur **{n_sim}** devis similaires "
            f"(client ≈ « {client} », produit ≈ « {produit} », "
            f"quantité ±{qty_band:.0%}). "
            f"Types produit distincts dans ce filtre : **{n_produits}**."
        )

        cols = [c for c in ["devis_code", "client", "produit", "quantite", "cout_total", "prix_total", "prix_unitaire", "signe"] if c in similar.columns]
        disp = similar[cols].head(20).copy()
        if "cout_total" in disp:
            disp["cout_total"] = disp["cout_total"].map(fmt_eur)
        if "prix_total" in disp:
            disp["prix_total"] = disp["prix_total"].map(fmt_eur)
        if "prix_unitaire" in disp:
            disp["prix_unitaire"] = disp["prix_unitaire"].map(lambda v: f"{float(v):,.4f} €".replace(",", " ") if pd.notna(v) else "—")
        if "signe" in disp:
            disp["signe"] = disp["signe"].map({1: "Accepté", 0: "Refusé"})
        disp = disp.rename(columns={
            "devis_code": "Devis",
            "client": "Client",
            "produit": "Produit",
            "quantite": "Quantité",
            "cout_total": "Coût total",
            "prix_total": "Prix total",
            "prix_unitaire": "Prix unitaire",
            "signe": "Résultat",
        })
        st.dataframe(disp, width="stretch", hide_index=True)
        if n_sim > 20:
            st.caption(f"… et {n_sim - 20} autres non affichés (affinez les filtres pour réduire).")


# ===========================================================================
# PAGE 2 — Performance
# ===========================================================================

elif page == "Performance des modèles":

    st.title("Performance des modèles")
    st.caption(
        "Évaluation alignée sur `eval_models.py` : split test 20 %, "
        "mêmes features finales (client, produit, quantité, coût, saison 4 mois)."
    )

    if "live_eval_results" not in st.session_state:
        st.session_state.live_eval_results = {}

    run_live = st.checkbox(
        "Recalculer sur le jeu de test (eval_models) — plus lent",
        value=False,
        help="Lance l'évaluation et **conserve** les résultats à l'écran jusqu'au prochain recalcul.",
    )
    c_run, c_clear = st.columns([1, 1])

    if run_live:
        try:
            import eval_models
            for src in ("ponceblanc", "lbfi"):
                with st.spinner(f"Évaluation {src}…"):
                    st.session_state.live_eval_results[src] = eval_models.evaluate_source(src)
            st.success("Évaluation terminée — résultats conservés sur cet écran.")
            # Uncheck is not possible programmatically for checkbox easily; user can leave it
        except Exception as exc:
            st.error(f"Évaluation live impossible : {exc}")

    if st.session_state.live_eval_results:
        st.caption("Affichage des **derniers résultats live** sauvegardés en session.")
    else:
        st.caption("Affichage depuis **metrics.json** (entraînement). Cochez la case pour recalculer.")

    for src in ("ponceblanc", "lbfi"):
        title = "PONCEBLANC" if src == "ponceblanc" else "LBFI"
        st.subheader(title)

        live = st.session_state.live_eval_results.get(src)

        metrics = get_metrics(src)
        if live is None and not metrics:
            st.warning(f"metrics.json introuvable pour {src}. Lancez train_models.py.")
            continue

        # Prefer live eval numbers when available
        if live is not None:
            n_total = live["n_total"]
            acc_rate = live["acceptance_rate"]
            clf = live["classifier"]
            reg = live["regressor"]
        else:
            n_total = metrics.get("n_total", 0)
            acc_rate = metrics.get("acceptance_rate", 0)
            clf = metrics.get("classifier") or {}
            reg_m = metrics.get("margin_regressor") or {}
            reg = {
                "n_rows": reg_m.get("n_rows"),
                "mae_eur": reg_m.get("mae"),
                "r2_eur": None,
                "mae_coeff": reg_m.get("mae_coeff"),
                "r2_coeff": None,
                "actual_price_mean": None,
                "predicted_price_mean": None,
                "features": reg_m.get("features") or [],
            }

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Devis utilisables", f"{n_total:,}")
            c2.metric("Taux acceptation", f"{acc_rate:.0%}")
            c3.metric("Accuracy (test)", f"{clf.get('accuracy', 0):.1%}")
            auc = clf.get("roc_auc")
            c4.metric("ROC-AUC (test)", f"{auc:.3f}" if auc is not None else "n/a")

        st.markdown("#### Classifieur — P(acceptation)")
        report = clf.get("report")
        if report:
            # compact table like classification_report
            rows = []
            for label in ("0", "1"):
                if label in report:
                    r = report[label]
                    rows.append({
                        "Classe": "Refusé" if label == "0" else "Accepté",
                        "Precision": f"{r['precision']:.3f}",
                        "Recall": f"{r['recall']:.3f}",
                        "F1": f"{r['f1-score']:.3f}"
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        feats = clf.get("features") or []
        if feats:
            st.caption("Features classifieur : " + ", ".join(feats))

        st.markdown("#### Régresseur de prix (coefficient × coût)")
        st.write(
            "Cible : **coefficient** = prix / coût sur devis **acceptés** "
            "(mode `coeff_times_cost`). Le prix affiché = coefficient × coût."
        )
        r1, r2, r3, r4 = st.columns(4)
        if reg.get("mae_eur") is not None:
            r1.metric("MAE prix (€)", fmt_eur(reg["mae_eur"], decimals=0))
        if reg.get("r2_eur") is not None:
            r2.metric("R² prix (€)", f"{reg['r2_eur']:.3f}")
        if reg.get("mae_coeff") is not None:
            r3.metric("MAE coefficient", f"{reg['mae_coeff']:.4f}")
        if reg.get("r2_coeff") is not None:
            r4.metric("R² coefficient", f"{reg['r2_coeff']:.3f}")

        means = []
        if reg.get("actual_price_mean") is not None:
            means.append(f"Prix réel moyen (test) : **{fmt_eur(reg['actual_price_mean'])}**")
        if reg.get("predicted_price_mean") is not None:
            means.append(f"Prix prédit moyen (test) : **{fmt_eur(reg['predicted_price_mean'])}**")
        if reg.get("n_rows") is not None:
            means.append(f"Lignes régresseur (acceptés avec coût) : **{reg['n_rows']:,}**")
        if means:
            st.markdown(" · ".join(means))

        reg_feats = reg.get("features") or []
        if reg_feats:
            st.caption("Features régresseur : " + ", ".join(reg_feats))

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
| **Coût (€)** | Achat + fabrication + transport → total |
| **Date** | → saison 4 mois (S1 Jan–Avr / S2 Mai–Août / S3 Sep–Déc) |

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

## Entrées du modèle (liste finale)

| Champ | Rôle |
|-------|------|
| Client | taux d'acceptation lissé |
| Type de produit | cluster |
| Quantité | volume |
| Coût total | achat + fabrication + transport |
| Date → saison 4 mois | S1 / S2 / S3 |
| Prix candidat | classifieur uniquement (P(accept)) |

**Non utilisés** : délai offre→décision (ne pilote pas la décision client), format, commercial, heuristiques pression / marge cible.

## Limites

- Peu d'historique pour un client / produit rare → confiance plus faible.
- Prix très éloignés du coût (coeff > ~3–5×) sont hors zone d'entraînement.
- Le modèle ne remplace pas le jugement commercial.

"""
    )

    st.caption("Estimateur d'offres — modèles entraînés sur l'historique devis Ponceblanc & LBFI.")