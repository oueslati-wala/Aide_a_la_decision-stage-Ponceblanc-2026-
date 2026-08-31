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
def get_estimator(_version: int = 17):
    return QuoteEstimator()


@st.cache_data
def get_history(source: str, _version: int = 17) -> pd.DataFrame:
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
    month: int | None,
    year: int | None,
    low_acceptance_percentile: float = 25.0,
    pricing_mode: str = "expected_margin",
    _version: int = 24,
):
    """Cache price recommendation for identical inputs (fast UI).
    month/year may be None when the user disables date/season.

    Uses recommend_price_strategic so that clients in the bottom slice of
    historical acceptance rates get a deliberately lower margin (higher
    chance of winning the quote). The percentile defines that slice.

    pricing_mode: fixed to "expected_margin" — the UI no longer exposes a
    choice. predict.py constrains its coefficient search to the realistic
    band [REALISTIC_COEFF_MIN, REALISTIC_COEFF_MAX] = [1.20, 1.80].
    """
    est = get_estimator()
    rec = est.recommend_price_strategic(
        client=client,
        produit=produit,
        quantite=quantite,
        source=source,
        cout_total=cout_total,
        month=month,
        year=year,
        low_acceptance_percentile=float(low_acceptance_percentile),
        pricing_mode=pricing_mode,
    )
    # Prefer the probability already computed on the strategic price when available
    if rec.get("acceptance_probability") is not None:
        proba = float(rec["acceptance_probability"])
    else:
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
    """
    Rank historical quotes by proximity (priority order):
      1. same client + same product + close quantity
      2. same client + same product
      3. same client only
    """
    if history.empty:
        return history

    sub = history.copy()
    client_key = (client or "").strip().upper()
    produit_key = (produit or "").strip().upper()
    qty = float(quantite) if quantite is not None else 0.0

    if not client_key:
        return sub.iloc[0:0].copy()

    client_s = sub["client"].astype(str).str.upper()
    mask_client_exact = client_s == client_key
    mask_client = mask_client_exact if mask_client_exact.any() else client_s.str.contains(client_key, na=False)
    sub = sub.loc[mask_client].copy()
    if sub.empty:
        return sub

    prod_s = sub["produit"].astype(str).str.upper()
    if produit_key:
        mask_prod = prod_s == produit_key
        if not mask_prod.any():
            mask_prod = prod_s.str.contains(produit_key, na=False)
    else:
        mask_prod = pd.Series(False, index=sub.index)

    qty_s = pd.to_numeric(sub["quantite"], errors="coerce")
    qty_dist = (qty_s - qty).abs()
    if qty > 0 and qty_tol is not None:
        lo, hi = qty * (1.0 - float(qty_tol)), qty * (1.0 + float(qty_tol))
        mask_qty = qty_s.between(lo, hi)
    else:
        mask_qty = pd.Series(False, index=sub.index)

    priority = pd.Series(3, index=sub.index, dtype=int)  # 3 = client only
    priority.loc[mask_prod] = 2                          # 2 = client + product
    priority.loc[mask_prod & mask_qty] = 1                # 1 = client + product + qty

    labels = {
        1: "1 · Client + produit + qté",
        2: "2 · Client + produit",
        3: "3 · Même client",
    }
    sub = sub.assign(
        _priority=priority.to_numpy(),
        _qty_dist=qty_dist.fillna(1e18).to_numpy(),
        Proximité=priority.map(labels).to_numpy(),
    )
    return sub.sort_values(["_priority", "_qty_dist"], ascending=[True, True])



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

        # --- Date / saison optionnelle ---
        season_labels = {
            1: "S1 — Janvier à Avril",
            2: "S2 — Mai à Août",
            3: "S3 — Septembre à Décembre",
        }
        season_mid_month = {1: 2, 2: 6, 3: 10}  # representative month for the model
        date_key = f"date_{source}"
        season_key = f"season4m_{source}"
        use_season_key = f"use_season_{source}"

        def _season_from_month(m: int) -> int:
            return min(3, max(1, (int(m) - 1) // 4 + 1))

        if date_key not in st.session_state:
            st.session_state[date_key] = __import__("datetime").date.today()
        if season_key not in st.session_state:
            st.session_state[season_key] = _season_from_month(
                st.session_state[date_key].month
            )
        if use_season_key not in st.session_state:
            st.session_state[use_season_key] = True

        use_season = st.toggle(
            "Prendre en compte la date / saison",
            key=use_season_key,
            help=(
                "Désactivez pour ignorer le calendrier dans l'estimation. "
                "Date et saison deviennent grisées et ne sont plus envoyées au modèle."
            ),
        )

        def _on_date_change():
            d = st.session_state.get(date_key)
            if d is not None:
                st.session_state[season_key] = _season_from_month(d.month)

        def _on_season_change():
            s = int(st.session_state.get(season_key, 1))
            d = st.session_state.get(date_key)
            if d is not None:
                mid = season_mid_month[s]
                import datetime as _dt
                day = min(d.day, 28)
                st.session_state[date_key] = _dt.date(d.year, mid, day)

        date_devis = st.date_input(
            "Date du devis",
            key=date_key,
            on_change=_on_date_change,
            disabled=not use_season,
            help="Change la saison automatiquement (S1/S2/S3).",
        )
        season_4m = st.selectbox(
            "Saison (4 mois)",
            options=[1, 2, 3],
            format_func=lambda s: season_labels[s],
            key=season_key,
            on_change=_on_season_change,
            disabled=not use_season,
            help="Change la date vers le milieu de la saison (même année).",
        )

        if use_season:
            month = season_mid_month[int(season_4m)]
            year = int(date_devis.year) if date_devis else 2024
        else:
            # Explicitly unknown → feature matrix sets month_known=0
            month = None
            year = None

        saison = None
        pression_concurrentielle = None
        marge_cible = None
        delai_livraison = None
        prix_candidat = None

    if use_season:
        st.caption(
            "Entrées modèle : client, type de produit, quantité, "
            "coûts (achat + fabrication + transport), date → saison 4 mois."
        )
    else:
        st.caption(
            "Entrées modèle : client, type de produit, quantité, "
            "coûts (achat + fabrication + transport). "
            "**Date / saison ignorées** pour cette estimation."
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

    with st.expander(
        "Stratégie clients à faible acceptation (ajuste la marge recommandée)",
        expanded=False,
    ):
        st.markdown(
            "Pour les clients dont le **taux d'acceptation historique** est parmi "
            "les plus bas de la source, le système **force une marge plus basse** "
            "(remise de 15 % à 40 % selon la sévérité) afin d'augmenter la chance "
            "de gagner le devis. C'est une **décision de portefeuille**, pas un "
            "apprentissage du modèle : on accepte moins de marge sur ce devis pour "
            "mieux convertir le client, en misant sur le reste du portefeuille."
        )
        low_acc_pct = st.slider(
            "Seuil « faible acceptation » (percentile)",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            key=f"low_acc_pct_{source}",
            help=(
                "Ex. 25 = les 25 % de clients avec le taux d'acceptation le plus bas "
                "sont considérés « faibles ». Plus le curseur est haut, plus de clients "
                "reçoivent une marge réduite. Plus il est bas, seule une minorité "
                "très difficile est ciblée."
            ),
        )
        st.caption(
            f"Les clients dans le **bas {int(low_acc_pct)} %** des taux d'acceptation "
            "lissés de cette source reçoivent une **remise forcée** sur le prix "
            "recommandé : **15 % minimum**, jusqu'à **40 %** pour les clients les "
            "plus difficiles (proportionnelle à l'écart au seuil). "
            "Le prix ne descend jamais sous coût × 1,05. "
            "Les scénarios ajoutent alors une ligne **Acquisition (marge mini ≈ +10 %)**."
        )

    # Pricing mode is fixed to "expected_margin" (maximise P(acceptation) x
    # marge) -- the earlier 3-way "Mode de pricing" selector was removed.
    # The coefficient search is constrained to a realistic band, 1,20x–1,80x
    # le coût, enforced in predict.py via REALISTIC_COEFF_MIN/MAX.
    pricing_mode = "expected_margin"
    st.caption(
        "Mode de pricing : **Marge espérée (P × marge)** — le prix maximise "
        "P(acceptation) × marge, dans une plage de coefficient réaliste "
        "**1,20× – 1,80×** le coût."
    )

    similar = filter_similar(history, client, produit, quantite, qty_tol=float(qty_band))

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
            month=int(month) if month is not None else None,
            year=int(year) if year is not None else None,
            low_acceptance_percentile=float(low_acc_pct),
            pricing_mode=pricing_mode,
        )
        prix_recommande = float(rec["prix_median"])
        strategic_applied = bool(rec.get("strategic_pricing_applied"))
        client_acc_rate = rec.get("client_acceptance_rate")
        low_acc_threshold = rec.get("low_acceptance_threshold")
        strategic_baseline = rec.get("strategic_baseline_price")
        strategic_discount = rec.get("strategic_discount_pct")
        strategic_target = rec.get("strategic_target_proba")

        prix_unitaire_reco = (
            prix_recommande / float(quantite) if quantite and float(quantite) > 0 else 0.0
        )

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            marge_reco = compute_margin(prix_recommande, cout_total)
            m1.metric("Prix recommandé (total)", fmt_eur(prix_recommande))
            m2.metric(
                "Prix unitaire (reco)",
                f"{prix_unitaire_reco:,.4f} €".replace(",", " "),
                help="Prix total recommandé ÷ quantité",
            )
            coeff_help = "prix / coût (ex. 1.30 = +30 % sur le coût)"
            if strategic_applied:
                coeff_help += (
                    " — Marge volontairement abaissée (stratégie client à faible acceptation)."
                )
            m3.metric(
                "Coefficient de marge",
                fmt_coefficient(marge_reco["coefficient"]),
                help=coeff_help,
            )
            m4.metric("P(acceptation)", f"{proba_reco:.0%}", help=interpret_proba(proba_reco))
            m5, m6 = st.columns(2)
            m5.metric("Coût total", fmt_eur(cout_total))
            marge_delta = fmt_pct_signed(marge_reco["taux_pct"])
            if strategic_applied and strategic_discount is not None:
                marge_delta = f"stratégie −{strategic_discount:.0%}"
            m6.metric(
                "Marge (€)",
                fmt_eur(marge_reco["marge_eur"]),
                delta=marge_delta,
            )

            mode_txt = "mode marge espérée (P × marge, coeff 1,20×–1,80×)"
            st.success(
                f"{probability_label(proba_reco)} — Prix recommandé : "
                f"**{fmt_eur(prix_recommande)}** "
                f"({prix_unitaire_reco:,.4f} € / exemplaire)".replace(",", " ")
            )
            if strategic_applied:
                st.caption(
                    f"{mode_txt.capitalize()} · marge abaissée (client à faible acceptation). "
                    f"Intervalle : {fmt_eur(rec['prix_lower'])} — {fmt_eur(rec['prix_upper'])}."
                )
            else:
                st.caption(
                    f"{mode_txt.capitalize()} · "
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
        if month is not None:
            season_code = int((month - 1) // 4) + 1
            season_names = {1: "S1 Jan–Avr", 2: "S2 Mai–Août", 3: "S3 Sep–Déc"}
            bullets.append(
                f"- **Saison** : {season_names.get(season_code, season_code)}"
                + (f" (année {year})" if year is not None else "")
            )
        else:
            bullets.append("- **Saison** : *non prise en compte* (bascule désactivée)")
        st.markdown("\n".join(bullets))

    with col_b:
        label = "Ponceblanc" if source == "ponceblanc" else "LBFI"
        auc = (metrics or {}).get("classifier", {}).get("roc_auc")
        auc_txt = f"{auc:.3f}" if auc is not None else "n/a"
        st.markdown("**Lecture**")
        season_line = (
            "client, type de produit, quantité, coût total, date → saison 4 mois (S1/S2/S3)."
            if month is not None
            else "client, type de produit, quantité, coût total (**sans** date/saison)."
        )
        st.markdown(
            f"""
Modèle **{label}** (ROC-AUC test ≈ **{auc_txt}**).

Entrées **uniquement** :
{season_line}

**Non utilisés** : matière, délai offre→décision, format, commercial, heuristiques.

- Prix recommandé : **{fmt_eur(prix_recommande)}**
- P(acceptation) : **{proba_reco:.0%}**
- Intervalle : **{fmt_eur(rec['prix_lower'])} — {fmt_eur(rec['prix_upper'])}**
"""
        )

    # Explicit explanation when margin was lowered for a low-acceptance client
    if strategic_applied:
        rate_txt = (
            f"{client_acc_rate:.0%}" if client_acc_rate is not None else "n/a"
        )
        thr_txt = (
            f"{low_acc_threshold:.0%}" if low_acc_threshold is not None else "n/a"
        )
        base_txt = (
            fmt_eur(strategic_baseline) if strategic_baseline is not None else "n/a"
        )
        disc_txt = (
            f"{strategic_discount:.0%}" if strategic_discount is not None else "n/a"
        )
        st.info(
            f"**Pourquoi la marge est plus basse sur ce devis** — "
            f"le client **{client_key}** a un taux d'acceptation historique lissé "
            f"d'environ **{rate_txt}**. Avec le seuil que vous avez choisi "
            f"(percentile **{int(low_acc_pct)}** → seuil ≈ **{thr_txt}**), "
            f"il est classé parmi les clients à **faible acceptation**. "
            f"Le prix a donc été volontairement abaissé par rapport au prix "
            f"« normal » du modèle (**{base_txt}** → **{fmt_eur(prix_recommande)}**, "
            f"remise ≈ **{disc_txt}**). "
            f"Plus le taux d'acceptation du client est bas, plus la remise est "
            f"forte (15 % minimum → jusqu'à 40 % pour les clients les plus difficiles). "
            f"Objectif : gagner ce devis en acceptant une marge plus faible ici, "
            f"en misant sur le reste du portefeuille. "
            f"Vous pouvez modifier le seuil « faible acceptation » dans l'encart "
            f"ci-dessus pour décider qui est concerné par cette stratégie."
        )
    elif client_acc_rate is not None and low_acc_threshold is not None:
        st.caption(
            f"Client **{client_key}** : taux d'acceptation lissé ≈ **{client_acc_rate:.0%}** "
            f"(seuil « faible » actuel ≈ **{low_acc_threshold:.0%}**, percentile "
            f"**{int(low_acc_pct)}**). Pas de baisse stratégique de marge sur ce devis."
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
    choice_key = f"votre_choix_prix_{source}"
    pu_widget_key = f"pu_input_{source}"
    pu_pending_key = f"pu_pending_{source}"
    cost_sync_key = f"pu_sync_cost_{source}"
    qty_f = float(quantite) if quantite else 1.0
    default_pu = float(prix_recommande / qty_f) if qty_f > 0 else 0.0

    # Apply pending PU from the scenario editor *before* the widget is created
    # (Streamlit forbids writing to a widget key after instantiation).
    if pu_pending_key in st.session_state:
        st.session_state[pu_widget_key] = float(st.session_state.pop(pu_pending_key))

    # Reset unit price when cost profile changes
    if (
        cost_sync_key not in st.session_state
        or abs(float(st.session_state.get(cost_sync_key, 0)) - float(cout_total)) > 1.0
    ):
        st.session_state[cost_sync_key] = float(cout_total)
        st.session_state[pu_widget_key] = round(default_pu, 4)

    if pu_widget_key not in st.session_state:
        st.session_state[pu_widget_key] = round(default_pu, 4)

    col_pu, col_pt = st.columns(2)
    with col_pu:
        prix_unitaire_saisi = st.number_input(
            "Prix de vente unitaire (€ / exemplaire)",
            min_value=0.0,
            step=0.01,
            format="%.4f",
            key=pu_widget_key,
            help="Prix de vente par exemplaire. Le total = unitaire × quantité.",
        )
    prix_total_from_unit = float(prix_unitaire_saisi) * qty_f
    with col_pt:
        st.metric("Prix de vente total correspondant", fmt_eur(prix_total_from_unit))
        st.caption(f"= {float(prix_unitaire_saisi):.4f} € × {int(qty_f)} exemplaires")

    # Always keep choice_key defined for the rest of the page
    st.session_state[choice_key] = float(prix_total_from_unit)

    st.subheader("5. Scénarios de décision")


    # UI bounds — allow high numbers for exploration; extreme coeffs are flagged.
    MAX_COEFF = 8.0
    MAX_PRICE = max(cout_total * MAX_COEFF, prix_recommande * 4.0)
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
    # For low-acceptance clients, densify the low-coeff end of the grid so
    # "Meilleure P" and acquisition scenarios can land near cost.
    try:
        if strategic_applied:
            # denser sampling between cost×1.05 and the strategic price
            low_end = np.linspace(float(cout_total) * 1.05, float(prix_recommande), 10)
            high_end = np.linspace(float(prix_recommande), float(min(cout_total * min(MAX_COEFF, 3.0), prix_recommande * 2.0, 500_000.0)), 8)
            grid_prices = np.unique(np.concatenate([low_end, high_end]))
        else:
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

    if strategic_applied:
        st.caption(
            "**Recommandé** = prix modèle **déjà abaissé** (stratégie client faible acceptation). "
            "**Acquisition** = marge minimale (≈ coût × 1,10) pour tenter de gagner le devis. "
            "**Meilleure P(accept)** = prix qui maximise la proba sur une grille densifiée côté bas."
        )
    else:
        st.caption(
            "**Recommandé** = prix typique accepté (régresseur + grille). "
            "**Ambitieux** = prix le plus haut dont la P(accept) reste ≥ celle du recommandé "
            f"({proba_reco:.0%}). "
            "**Meilleure P(accept)** = prix qui maximise la proba."
        )

    # Acquisition floor: ~10% above cost — only shown for low-acceptance clients
    acquisition_price = max(float(cout_total) * 1.10, float(cout_total) + 1.0)

    if strategic_applied:
        # More aggressive anchors when the client is hard to win
        prudent_price = max(acquisition_price, round(float(prix_recommande) * 0.85, -1))
        anchor_prices = {
            "Acquisition (marge mini)": round(acquisition_price, -1) if acquisition_price >= 100 else round(acquisition_price, 0),
            "Prudent (viser l'acceptation)": prudent_price,
            "Recommandé (stratégie)": prix_recommande,
            "Ambitieux (plus élevé)": ambitieux_price,
        }
    else:
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
    st.session_state[choice_key] = float(prix_total_from_unit)

    # Batch scenario probabilities (one classifier call)
    scenario_items = list(anchor_prices.items()) + [("Votre choix (via PU)", float(st.session_state[choice_key]))]
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
    prev_choice = float(st.session_state.get(choice_key, prix_total_from_unit))
    if abs(new_choice_price - prev_choice) > 0.01:
        st.session_state[choice_key] = new_choice_price
        # Sync unit-price widget on next run (cannot write widget key after instantiation)
        if quantite and float(quantite) > 0:
            st.session_state[pu_pending_key] = round(
                new_choice_price / float(quantite), 4
            )
        st.rerun()


    # Sensibilité — only compute when expander is open (saves a full grid on every tweak)
    st.subheader("6. Sensibilité")
    with st.expander("Afficher la courbe P(acceptation) vs prix", expanded=False):
        st.caption(
            f"Coût fixé à **{fmt_eur(cout_total)}**. "
            "Vous pouvez monter très haut pour tester l’effet sur P(acceptation) "
            "(prédictions du classifieur, pas un taux empirique)."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            step_val = st.number_input(
                "Pas (€)", min_value=5.0, value=100.0, step=10.0,
                key=f"step_sens_{source}",
            )
        with c2:
            sens_max_coeff = st.number_input(
                "Coeff max (× coût)",
                min_value=1.2,
                max_value=10.0,
                value=4.0,
                step=0.5,
                key=f"sens_max_coeff_{source}",
                help="Plafond de la courbe = coût × ce coefficient (ex. 4.0 = +300 % sur le coût).",
            )
        with c3:
            sens_min_coeff = st.number_input(
                "Coeff min (× coût)",
                min_value=1.0,
                max_value=3.0,
                value=1.05,
                step=0.05,
                key=f"sens_min_coeff_{source}",
            )
        try:
            sens_min_price = float(cout_total) * float(sens_min_coeff)
            sens_max_price = float(cout_total) * float(sens_max_coeff)
            sens_max_price = min(sens_max_price, 500_000.0)
            curve = est.optimize_price(
                client=client,
                produit=produit,
                quantite=quantite,
                cout_total=cout_total,
                min_price=sens_min_price,
                max_price=sens_max_price,
                step=step_val,
                source=source,
                matiere=matiere,
                month=month,
                year=year,
            )
            curve = curve.copy()
            curve["coeff"] = curve["prix_total"] / float(cout_total)
            chart = curve.set_index("prix_total")[["acceptance_proba"]].rename(
                columns={"acceptance_proba": "P(acceptation)"}
            )
            chart.index.name = "Prix de vente total (€)"
            st.line_chart(chart)
            st.caption(
                f"Plage : {fmt_eur(sens_min_price)} → {fmt_eur(sens_max_price)} "
                f"(coeff {float(sens_min_coeff):.2f}× → {float(sens_max_coeff):.2f}×). "
                f"{len(curve)} points."
            )
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
        n1 = int((similar["_priority"] == 1).sum()) if "_priority" in similar.columns else 0
        n2 = int((similar["_priority"] == 2).sum()) if "_priority" in similar.columns else 0
        n3 = int((similar["_priority"] == 3).sum()) if "_priority" in similar.columns else 0

        r1 = st.columns(4)
        r1[0].metric("Total", f"{n_sim}")
        r1[1].metric("Client+prod+qté", f"{n1}")
        r1[2].metric("Client+produit", f"{n2}")
        r1[3].metric("Client seul", f"{n3}")

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
            f"Tri par priorité : **1** client+produit+qté (±{qty_band:.0%}) → "
            f"**2** client+produit → **3** même client. "
            f"Affiche les **{n_sim}** devis correspondants."
        )

        # "produit" = product type / name (e.g. LIASSE, COLLECTION) — this is
        # the product reference shown here, never "matiere" (raw material),
        # which is a separate, unrelated field. "reference_client" is the
        # "Référence client" column from Query_tableau_devis_with_costs.xlsx.
        # "taux_marge" is the margin field EXTRACTED DIRECTLY from the
        # Ponceblanc Excel's own "Taux Marge" column (see
        # standardize_ponceblanc in features.py) — nothing is calculated
        # here. LBFI's raw extract never has this field (business rule:
        # LBFI margin is not used — see standardize_lbfi), so it shows "—".
        cols = [c for c in [ "devis_code", "reference_client", "client", "produit", "quantite", "cout_total", "prix_total", "prix_unitaire", "taux_marge", "signe"] if c in similar.columns]
        disp = similar[cols].copy()

        if "cout_total" in disp:
            disp["cout_total"] = disp["cout_total"].map(fmt_eur)
        if "prix_total" in disp:
            disp["prix_total"] = disp["prix_total"].map(fmt_eur)
        if "prix_unitaire" in disp:
            disp["prix_unitaire"] = disp["prix_unitaire"].map(lambda v: f"{float(v):,.4f} €".replace(",", " ") if pd.notna(v) else "—")
        if "taux_marge" in disp:
            disp["taux_marge"] = disp["taux_marge"].map(
                lambda v: fmt_coefficient(float(v)) if pd.notna(v) else "—"
            )
        if "reference_client" in disp:
            disp["reference_client"] = disp["reference_client"].fillna("—")
        if "signe" in disp:
            disp["signe"] = disp["signe"].map({1: "Accepté", 0: "Refusé"})
        disp = disp.rename(columns={
            "devis_code": "Devis",
            "reference_client": "Référence client",
            "client": "Client",
            "produit": "Produit",
            "quantite": "Quantité",
            "cout_total": "Coût total",
            "prix_total": "Prix total",
            "prix_unitaire": "Prix unitaire",
            "taux_marge": "Coefficient (marge)",
            "signe": "Résultat",
        })
        st.dataframe(disp, width="stretch", hide_index=True)
        


# ===========================================================================
# PAGE 2 — Performance
# ===========================================================================

elif page == "Performance des modèles":

    st.title("Performance des modèles")
    st.markdown(
        """
        Cette page répond à une question simple : **peut-on faire confiance aux modèles ?**

        Les chiffres sont calculés sur des devis **que le modèle n'a jamais vus** pendant
        l'entraînement (20 % réservés pour le test). C'est la meilleure façon de mesurer
        s'il généralise, ou s'il a seulement « appris par cœur » l'historique.
        """
    )

    with st.expander("Comment lire cette page (sans jargon)", expanded=True):
        st.markdown(
            """
**Deux jobs distincts**

1. **Classifieur** — « Pour ce prix, quelle chance d'être accepté ? »  
   On regarde surtout le **ROC-AUC** (0,5 = hasard, 1,0 = parfait) et si les
   probabilités annoncées collent à la réalité (calibration).

2. **Régresseur de prix** — « Quel prix de vente typique pour un devis accepté ? »  
   On regarde l'erreur en **euros** et en **%**, et si le modèle bat une règle naïve
   (toujours le coefficient médian historique).

**Verdict de confiance**  
Un score 0–100 résume classifieur + régresseur + volume de données.
Il n'est pas magique : c'est un guide pour décider si l'outil est une aide fiable
ou seulement un explorateur d'historique.
            """
        )

    if "live_eval_results" not in st.session_state:
        st.session_state.live_eval_results = {}

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        do_eval = st.button(
            "Lancer l'évaluation complète",
            type="primary",
            help="Recalcule toutes les métriques + tests de confiance sur le jeu de test.",
        )
    with col_info:
        if st.session_state.live_eval_results:
            st.success("Résultats d'évaluation chargés (session). Relancer pour rafraîchir.")
        else:
            st.info(
                "Sans évaluation live, seuls les chiffres d'entraînement (metrics.json) "
                "sont affichés — moins complets. Cliquez le bouton pour les tests détaillés."
            )

    if do_eval:
        try:
            import eval_models
            for src in ("ponceblanc", "lbfi"):
                with st.spinner(f"Évaluation {src}…"):
                    st.session_state.live_eval_results[src] = eval_models.evaluate_source(
                        src, verbose=False
                    )
            st.success("Évaluation terminée.")
            st.rerun()
        except Exception as exc:
            st.error(f"Évaluation impossible : {exc}")

    # ---- Summary cards for both sources ----
    live_any = bool(st.session_state.live_eval_results)
    summary_rows = []
    for src in ("ponceblanc", "lbfi"):
        live = st.session_state.live_eval_results.get(src)
        metrics = get_metrics(src)
        if live is None and not metrics:
            continue
        if live is not None:
            trust = live.get("trust") or {}
            auc = (live.get("classifier") or {}).get("roc_auc")
            mae_c = (live.get("regressor") or {}).get("mae_coeff")
            within20 = (live.get("regressor") or {}).get("within_20pct")
            n_total = live.get("n_total")
        else:
            trust = {}
            auc = (metrics.get("classifier") or {}).get("roc_auc")
            mae_c = (metrics.get("margin_regressor") or {}).get("mae_coeff")
            within20 = None
            n_total = metrics.get("n_total")
        summary_rows.append({
            "Source": "Ponceblanc" if src == "ponceblanc" else "LBFI",
            "Devis": f"{n_total:,}" if n_total else "—",
            "ROC-AUC": f"{auc:.3f}" if auc is not None else "—",
            "MAE coeff.": f"{mae_c:.3f}" if mae_c is not None else "—",
            "Prix ±20 %": f"{within20:.0%}" if within20 is not None else "—",
            "Confiance": trust.get("label", "— (lancer l'éval.)"),
            "Score": trust.get("score", "—"),
        })

    if summary_rows:
        st.markdown("### Vue d'ensemble")
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    for src in ("ponceblanc", "lbfi"):
        title = "Ponceblanc" if src == "ponceblanc" else "LBFI"
        st.markdown("---")
        st.subheader(title)

        live = st.session_state.live_eval_results.get(src)
        metrics = get_metrics(src)
        if live is None and not metrics:
            st.warning(f"Aucune métrique pour {src}. Lancez `python train_models.py`.")
            continue

        if live is not None:
            n_total = live["n_total"]
            n_test = live.get("n_test")
            acc_rate = live["acceptance_rate"]
            clf = live["classifier"]
            reg = live["regressor"]
            trust = live.get("trust") or {}
        else:
            n_total = metrics.get("n_total", 0)
            n_test = metrics.get("n_test")
            acc_rate = metrics.get("acceptance_rate", 0)
            clf = metrics.get("classifier") or {}
            reg_m = metrics.get("margin_regressor") or {}
            reg = {
                "n_rows": reg_m.get("n_rows"),
                "mae_eur": reg_m.get("mae"),
                "r2_eur": None,
                "mae_coeff": reg_m.get("mae_coeff"),
                "r2_coeff": None,
                "mape": None,
                "median_ape": None,
                "within_10pct": None,
                "within_20pct": None,
                "baseline_mae_eur": None,
                "baseline_mape": None,
                "actual_price_mean": None,
                "predicted_price_mean": None,
                "cost_bands": [],
                "features": reg_m.get("features") or [],
            }
            trust = {}

        # --- Trust banner ---
        if trust:
            color = trust.get("color", "orange")
            if color == "green":
                st.success(
                    f"**{trust.get('label', '')}** — score {trust.get('score', '—')}/100\n\n"
                    f"{trust.get('advice', '')}"
                )
            elif color == "red":
                st.error(
                    f"**{trust.get('label', '')}** — score {trust.get('score', '—')}/100\n\n"
                    f"{trust.get('advice', '')}"
                )
            else:
                st.warning(
                    f"**{trust.get('label', '')}** — score {trust.get('score', '—')}/100\n\n"
                    f"{trust.get('advice', '')}"
                )
            if trust.get("reasons"):
                with st.expander("Détail du score de confiance"):
                    for r in trust["reasons"]:
                        st.markdown(f"- {r}")
        else:
            st.info(
                "Lancez **l'évaluation complète** pour obtenir le verdict de confiance, "
                "la calibration des probabilités et les tests par tranche de coût."
            )

        # --- Volume ---
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Devis utilisables", f"{n_total:,}")
            c2.metric("Taux d'acceptation historique", f"{acc_rate:.0%}")
            c3.metric("Devis de test (jamais vus)", f"{n_test:,}" if n_test else "—")

        # --- Classifier ---
        st.markdown("#### 1. Probabilité d'acceptation (classifieur)")
        auc = clf.get("roc_auc")
        acc = clf.get("accuracy")
        base_acc = clf.get("baseline_accuracy")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "ROC-AUC",
            f"{auc:.3f}" if auc is not None else "n/a",
            help="0,5 = hasard · 0,7 = correct · 0,8+ = bon",
        )
        m2.metric(
            "Précision globale",
            f"{acc:.0%}" if acc is not None else "n/a",
        )
        if base_acc is not None:
            m3.metric(
                "Baseline « toujours la majorité »",
                f"{base_acc:.0%}",
                help="Si on prédisait toujours la classe la plus fréquente, sans modèle.",
            )

        if clf.get("interpret_auc"):
            st.markdown(f"**En clair :** {clf['interpret_auc']}")
        if clf.get("interpret_accuracy"):
            st.caption(clf["interpret_accuracy"])

        # Confusion in plain language
        conf = clf.get("confusion")
        if conf:
            st.markdown(
                f"""
Sur le jeu de test :
- **Bien classés acceptés** (vrais positifs) : **{conf.get('tp', 0)}**
- **Bien classés refusés** (vrais négatifs) : **{conf.get('tn', 0)}**
- **Faux espoirs** (prédit accepté, en réalité refusé) : **{conf.get('fp', 0)}**
- **Occasions manquées** (prédit refusé, en réalité accepté) : **{conf.get('fn', 0)}**
                """
            )

        report = clf.get("report")
        if report:
            rows = []
            for label in ("0", "1"):
                if label in report:
                    r = report[label]
                    rows.append({
                        "Classe": "Refusé" if label == "0" else "Accepté",
                        "Précision": f"{r['precision']:.0%}",
                        "Rappel": f"{r['recall']:.0%}",
                        "F1": f"{r['f1-score']:.2f}",
                    })
            if rows:
                with st.expander("Tableau précision / rappel"):
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                    st.caption(
                        "Précision = parmi les « accepté » prédits, combien l'étaient vraiment. "
                        "Rappel = parmi les vrais acceptés, combien le modèle a trouvés."
                    )

        # Calibration
        cal = clf.get("calibration") or []
        if cal:
            st.markdown("**Calibration des probabilités**")
            st.caption(
                "Quand le modèle annonce ~70 %, le taux réel d'acceptation dans cette tranche "
                "devrait être proche de 70 %. Un écart fort = probabilités mal calibrées."
            )
            cal_df = pd.DataFrame(cal)
            cal_df = cal_df.rename(columns={
                "tranche_proba": "Tranche annoncée",
                "n": "Nb devis test",
                "proba_moyenne": "Proba moyenne",
                "taux_reel": "Taux réel d'acceptation",
                "ecart": "Écart (réel − annoncé)",
            })
            st.dataframe(cal_df, hide_index=True, width="stretch")

        segs = clf.get("segments") or {}
        if segs:
            with st.expander("Clients fréquents vs rares"):
                for name, s in segs.items():
                    label = "Clients fréquents (≥ 10 devis)" if "frequent" in name else "Clients rares"
                    st.markdown(
                        f"- **{label}** : {s['n']} devis test · "
                        f"AUC **{s['auc']:.3f}** · précision **{s['accuracy']:.0%}**"
                    )
                st.caption(
                    "Si l'AUC chute fortement sur les clients rares, méfiez-vous des "
                    "recommandations pour un nouveau client peu présent dans l'historique."
                )

        # --- Regressor ---
        st.markdown("#### 2. Prix recommandé (régresseur coefficient × coût)")
        st.write(
            "Le modèle apprend le **coefficient** prix/coût sur les devis **signés**, "
            "puis affiche `prix = coefficient × coût total`."
        )

        r1, r2, r3, r4 = st.columns(4)
        if reg.get("mae_coeff") is not None:
            r1.metric("Erreur moyenne sur le coefficient", f"{reg['mae_coeff']:.3f}")
        if reg.get("median_ape") is not None:
            r2.metric("Erreur médiane sur le prix", f"{reg['median_ape']:.0%}")
        if reg.get("within_20pct") is not None:
            r3.metric("Prix dans ±20 % du réel", f"{reg['within_20pct']:.0%}")
        if reg.get("within_10pct") is not None:
            r4.metric("Prix dans ±10 % du réel", f"{reg['within_10pct']:.0%}")

        r5, r6, r7, r8 = st.columns(4)
        if reg.get("mae_eur") is not None:
            r5.metric("MAE prix (€)", fmt_eur(reg["mae_eur"], decimals=0))
        if reg.get("mape") is not None:
            r6.metric("MAPE (erreur % moyenne)", f"{reg['mape']:.0%}")
        if reg.get("r2_eur") is not None:
            r7.metric("R² prix", f"{reg['r2_eur']:.2f}")
        if reg.get("n_rows") is not None:
            r8.metric("Devis acceptés avec coût", f"{reg['n_rows']:,}")

        # Baseline comparison
        if reg.get("baseline_mae_eur") is not None and reg.get("mae_eur") is not None:
            gain = reg["baseline_mae_eur"] - reg["mae_eur"]
            if gain > 0:
                st.success(
                    f"Le modèle bat la règle naïve « toujours le coefficient médian "
                    f"({reg.get('baseline_coeff', 0):.2f}) » : "
                    f"MAE {fmt_eur(reg['mae_eur'], decimals=0)} vs "
                    f"{fmt_eur(reg['baseline_mae_eur'], decimals=0)} pour la baseline "
                    f"(gain {fmt_eur(gain, decimals=0)})."
                )
            else:
                st.warning(
                    f"Le modèle ne bat pas clairement la règle naïve "
                    f"(coeff médian {reg.get('baseline_coeff', 0):.2f}). "
                    f"MAE modèle {fmt_eur(reg['mae_eur'], decimals=0)} vs "
                    f"baseline {fmt_eur(reg['baseline_mae_eur'], decimals=0)}."
                )

        means = []
        if reg.get("actual_price_mean") is not None:
            means.append(f"Prix réel moyen (test) : **{fmt_eur(reg['actual_price_mean'])}**")
        if reg.get("predicted_price_mean") is not None:
            means.append(f"Prix prédit moyen (test) : **{fmt_eur(reg['predicted_price_mean'])}**")
        if means:
            st.markdown(" · ".join(means))

        bands = reg.get("cost_bands") or []
        if bands:
            st.markdown("**Erreur selon la taille du devis (coût)**")
            bdf = pd.DataFrame(bands).rename(columns={
                "bande": "Tranche",
                "cout_min": "Coût min €",
                "cout_max": "Coût max €",
                "n": "Nb test",
                "mae_eur": "MAE €",
                "mape": "Erreur % moy.",
            })
            if "Erreur % moy." in bdf.columns:
                bdf["Erreur % moy."] = bdf["Erreur % moy."].map(
                    lambda x: f"{x:.0%}" if pd.notna(x) else "—"
                )
            st.dataframe(bdf, hide_index=True, width="stretch")
            st.caption(
                "Si l'erreur explose sur les gros devis, soyez plus prudent sur les "
                "montants élevés — l'historique y est souvent plus rare."
            )

        with st.expander("Features utilisées (technique)"):
            feats = clf.get("features") or []
            reg_feats = reg.get("features") or []
            if feats:
                st.caption("Classifieur : " + ", ".join(feats))
            if reg_feats:
                st.caption("Régresseur : " + ", ".join(reg_feats))

    st.markdown("---")
    st.markdown(
        """
### Comment décider de faire confiance ?

| Situation | Attitude recommandée |
|-----------|----------------------|
| Score confiance **élevé**, client/produit **fréquents** | S'appuyer sur le prix recommandé + scénarios |
| Score **moyen**, ou client peu connu | Utiliser l'outil + **toujours** regarder l'historique comparable |
| Score **faible**, ou devis hors normes (coeff extrême, coût hors plage) | Explorer l'historique ; le chiffre du modèle n'est qu'indicatif |
| Probabilités mal calibrées (écarts forts dans le tableau) | Lire P(accept) comme un **ordre de grandeur**, pas un % exact |

L'outil **ne remplace pas** le jugement commercial. Il quantifie les habitudes passées.
        """
    )

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
| **Date / saison** | Optionnelle (bascule). Si activée → saison 4 mois (S1/S2/S3) |

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
| Date → saison 4 mois | Optionnelle (bascule UI) ; S1 / S2 / S3 si activée |
| Prix candidat | classifieur uniquement (P(accept)) |

**Non utilisés** : délai offre→décision (ne pilote pas la décision client), format, commercial, heuristiques pression / marge cible.

## Limites

- Peu d'historique pour un client / produit rare → confiance plus faible.
- Prix très éloignés du coût (coeff > ~3–5×) sont hors zone d'entraînement.
- Le modèle ne remplace pas le jugement commercial.

"""
    )

    st.caption("Estimateur d'offres — modèles entraînés sur l'historique devis Ponceblanc & LBFI.")
