"""
chat_assistant.py
==================

Text-chat layer over the existing Estimateur d'offres logic.

Design principle: the assistant NEVER invents a price, a probability, or a
history figure. Every number in its answers comes from a tool call into the
existing QuoteEstimator / features functions already used by vis.py. The LLM
only does two things:
    1. Turn a natural-language request into the right tool call(s).
    2. Turn the tool result(s) into a clear French answer.

Read-only by design: no tool here writes to Odoo, models/, or any file.
It only reads history and calls prediction functions that already exist in
predict.py.

Backend: MiniMax M3 (free) via OpenRouter
  model  = minimax/minimax-m3:free
  base   = https://openrouter.ai/api/v1
  key    = OPENROUTER_API_KEY

Usage from vis.py:

    from chat_assistant import run_chat_turn

    reply = run_chat_turn(messages, source_hint="ponceblanc")

`messages` is a list of {"role": "user"|"assistant", "content": str} dicts,
already including the new user message. The function returns the assistant's
reply text and appends nothing itself — the caller manages session_state.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd

import features
from predict import QuoteEstimator


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# MiniMax M3 free on OpenRouter
CHAT_MODEL = os.environ.get(
    "CHAT_ASSISTANT_MODEL", "minimax/minimax-m3:free"
)
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

MAX_TOOL_ROUNDS = 4  # safety cap on tool-call loops

SYSTEM_PROMPT = """\
Tu es l'assistant intégré à l'outil interne de pricing devis de Ponceblanc / LBFI.

Règles strictes :
- Tu ne dois JAMAIS inventer un prix, une probabilité d'acceptation, ou un
  chiffre d'historique. Utilise TOUJOURS les outils fournis pour obtenir ces
  chiffres ; ne les calcule pas toi-même et ne les estime pas de mémoire.
  Toute réponse contenant un chiffre (€, %) sans appel d'outil est rejetée
  automatiquement par le système — appelle toujours l'outil, même si tu
  penses connaître la réponse d'un tour précédent.
- Pour une demande de recommandation de prix sur l'offre en cours, les
  champs client/produit/quantité/coût/saison/mode de pricing manquants sont
  automatiquement complétés depuis le formulaire affiché à l'écran — appelle
  get_price_recommendation même sans répéter ces valeurs, le résultat sera
  identique à ce que montre l'interface pour la même offre.
- Si une information nécessaire manque (client, produit, quantité, coût,
  source Ponceblanc/LBFI), demande-la avant d'appeler un outil, sauf si un
  défaut raisonnable est évident depuis la conversation.
- Réponds en français, de façon concise et professionnelle, comme à un
  commercial qui n'est pas data scientist. Explique les chiffres en une ou
  deux phrases, pas de jargon ML inutile.
- "source" doit valoir "ponceblanc" ou "lbfi". Si l'utilisateur ne précise
  pas, utilise la source indiquée dans le contexte de l'interface.

CONTRÔLE DU FORMULAIRE — RÈGLE ABSOLUE :
Tu contrôles le formulaire. L'outil update_form_inputs modifie réellement
l'interface. Tu DOIS l'appeler pour tout changement demandé.

Champs modifiables (tous) :
  • client, produit, quantite
  • cout_achat, cout_fabrication, cout_transport, cout_total
  • date_devis (YYYY-MM-DD), saison (1=jan–avr, 2=mai–aoû, 3=sep–déc)
  • use_season (true/false), source (ponceblanc/lbfi)

Interdictions (phrases INTERDITES, ne les écris jamais) :
  ✗ « je ne peux pas modifier le widget »
  ✗ « changez la valeur dans le formulaire »
  ✗ « ni la date ni la saison ne peuvent être modifiées »
  ✗ « je n'ai pas de levier pour les coûts / la quantité »
  ✗ « une fois le formulaire affiché… »
  ✗ toute variante qui prétend que tu ne contrôles pas le formulaire

Comportement :
1. Demande de changer un champ → appelle update_form_inputs immédiatement.
2. Après succès → confirme en 1 phrase ce qui a changé.
3. Propose de recalculer le prix si pertinent.

Maximiser le % d'acceptation :
- Tu peux et tu dois ajuster les champs du formulaire via update_form_inputs.
- Stratégie typique (sans inventer de chiffres) :
  • garder le client / produit demandés (ou ceux déjà dans le formulaire)
  • ne pas inventer des coûts : si l'utilisateur ne donne pas de coûts,
    garde ceux du formulaire (fournis dans le contexte) ou demande-les
  • saison : tu peux la fixer (ex. saison 1) si l'historique le justifie
    via get_seasonality_stats ; sinon laisse la saison actuelle
  • quantity : ne change que si l'utilisateur l'autorise ou si c'est
    explicitement pour tester un scénario
- Ensuite appelle get_price_recommendation pour montrer le P(accept) obtenu.
- Explique clairement que le levier principal sur P(accept) reste le prix
  recommandé (plus bas → acceptation plus haute, marge plus basse).

Autres outils :
- get_seasonality_stats pour volume par mois / saison / année.
- Tu ne modifies rien d'autre (pas d'écriture dans Odoo, pas de fichier).
"""


# ---------------------------------------------------------------------
# ESTIMATOR (cached at module load — mirrors get_estimator() in vis.py)
# ---------------------------------------------------------------------

_estimator: QuoteEstimator | None = None


def _get_estimator() -> QuoteEstimator:
    global _estimator
    if _estimator is None:
        _estimator = QuoteEstimator()
    return _estimator


_history_cache: dict[str, pd.DataFrame] = {}


def _get_history(source: str) -> pd.DataFrame:
    if source not in _history_cache:
        _history_cache[source] = features.build_source(source)
    return _history_cache[source]


# ---------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# (thin wrappers — same calls vis.py already makes)
# ---------------------------------------------------------------------

def _tool_price_recommendation(args: dict) -> dict:
    """Same call the interface makes (recommend_price_strategic), with the
    same month/year/pricing_mode/low_acceptance_percentile — so the chat can
    never diverge from what is shown in '2. Proposition du modèle' for the
    same offer. Missing args are filled from the live form context by
    _fill_args_from_context() before this runs."""
    est = _get_estimator()
    try:
        quantite = float(args["quantite"])
        cout_total = float(args["cout_total"])
        pricing_mode = str(args.get("pricing_mode") or "balanced")
        low_acc_pct = float(args.get("low_acceptance_percentile", 25.0))
        month = args.get("month")
        year = args.get("year")

        rec = est.recommend_price_strategic(
            client=str(args["client"]),
            produit=str(args["produit"]),
            quantite=quantite,
            source=str(args["source"]).lower(),
            cout_total=cout_total,
            pricing_mode=pricing_mode,
            low_acceptance_percentile=low_acc_pct,
            month=int(month) if month is not None else None,
            year=int(year) if year is not None else None,
        )
        prix = float(rec["prix_median"])
        marge_eur = prix - cout_total
        return {
            "prix_recommande_eur": round(prix, 2),
            "prix_unitaire_eur": round(prix / quantite, 4) if quantite else None,
            "prix_min_eur": round(rec["prix_lower"], 2),
            "prix_max_eur": round(rec["prix_upper"], 2),
            "coefficient_marge": rec.get("coeff_median"),
            "marge_eur": round(marge_eur, 2),
            "taux_marge_pct": round(100 * marge_eur / cout_total, 1) if cout_total else None,
            "probabilite_acceptation": rec.get("acceptance_probability"),
            "cout_total_eur": cout_total,
            "pricing_mode_utilise": pricing_mode,
            "remise_strategique_appliquee": rec.get("strategic_pricing_applied", False),
            "remise_pct": rec.get("strategic_discount_pct"),
            "raison_remise": (
                "Client historiquement à faible taux d'acceptation : marge "
                "volontairement réduite pour augmenter la chance de signature."
                if rec.get("strategic_pricing_applied") else None
            ),
        }
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        return {"error": str(exc)}


def _tool_acceptance_probability(args: dict) -> dict:
    est = _get_estimator()
    try:
        month = args.get("month")
        year = args.get("year")
        proba = est.predict_acceptance_proba(
            client=str(args["client"]),
            produit=str(args["produit"]),
            quantite=float(args["quantite"]),
            source=str(args["source"]).lower(),
            cout_total=float(args["cout_total"]),
            prix_total=float(args["prix_total"]),
            month=int(month) if month is not None else None,
            year=int(year) if year is not None else None,
        )
        return {
            "probabilite_acceptation": round(proba, 4),
            "coefficient_marge": round(
                float(args["prix_total"]) / float(args["cout_total"]), 3
            ) if float(args["cout_total"]) > 0 else None,
        }
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return {"error": str(exc)}


def _tool_similar_history(args: dict) -> dict:
    source = str(args["source"]).lower()
    client = str(args.get("client", "")).strip().upper()
    produit_raw = args.get("produit")
    produit = (
        str(features.normalize_produit(str(produit_raw))).upper()
        if produit_raw else None
    )
    quantite = args.get("quantite")
    qty_tol = float(args.get("qty_tolerance", 0.5))

    try:
        df = _get_history(source)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    sub = df.copy()
    if client:
        client_s = sub["client"].astype(str).str.upper()
        mask = client_s == client
        if not mask.any():
            mask = client_s.str.contains(client, na=False)
        sub = sub.loc[mask]

    if sub.empty:
        return {
            "n_devis_trouves": 0,
            "message": f"Aucun historique trouvé pour le client '{client or '?'}' sur {source}.",
        }

    if produit:
        prod_s = sub["produit"].astype(str).str.upper()
        mask_p = prod_s == produit
        if mask_p.any():
            sub = sub.loc[mask_p]

    if quantite:
        qty = float(quantite)
        qty_s = pd.to_numeric(sub["quantite"], errors="coerce")
        lo, hi = qty * (1 - qty_tol), qty * (1 + qty_tol)
        mask_q = qty_s.between(lo, hi)
        if mask_q.any():
            sub = sub.loc[mask_q]

    n = len(sub)
    n_acc = int((sub["signe"] == 1).sum())
    acc = sub[sub["signe"] == 1]
    rej = sub[sub["signe"] == 0]

    valid_cost = sub["cout_total"].notna() & (sub["cout_total"] > 0)
    valid_price = sub["prix_total"].notna() & (sub["prix_total"] > 0)
    valid = valid_cost & valid_price
    coeff = pd.Series(np.nan, index=sub.index)
    coeff.loc[valid] = sub.loc[valid, "prix_total"] / sub.loc[valid, "cout_total"]

    return {
        "n_devis_trouves": n,
        "n_acceptes": n_acc,
        "n_refuses": n - n_acc,
        "taux_acceptation_pct": round(100 * n_acc / n, 1) if n else None,
        "prix_median_acceptes_eur": (
            round(float(acc["prix_total"].median()), 2)
            if len(acc) and acc["prix_total"].notna().any() else None
        ),
        "prix_median_refuses_eur": (
            round(float(rej["prix_total"].median()), 2)
            if len(rej) and rej["prix_total"].notna().any() else None
        ),
        "coefficient_median_accepte": (
            round(float(coeff.loc[acc.index].dropna().median()), 3)
            if len(acc) and coeff.loc[acc.index].notna().any() else None
        ),
    }


def _tool_client_acceptance_rate(args: dict) -> dict:
    est = _get_estimator()
    try:
        info = est.client_acceptance_rate(
            client=str(args["client"]),
            source=str(args["source"]).lower(),
        )
        return {
            "client": info["client"],
            "client_connu_dans_historique": info["known"],
            "taux_acceptation_lisse_pct": round(100 * info["client_rate"], 1),
            "taux_acceptation_global_source_pct": round(100 * info["global_rate"], 1),
        }
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return {"error": str(exc)}


def _tool_update_form_inputs(args: dict) -> dict:
    """Write form fields into Streamlit session_state so the main page updates.

    Uses a pending-* key pattern so the main page can force the value onto the
    widget *before* it is created (avoids Streamlit ignoring late writes to a
    widget key that already owns the state).

    Supported fields:
      client, produit, quantite,
      cout_achat / cout_fabrication / cout_transport / cout_total,
      date_devis (YYYY-MM-DD), saison (1|2|3), use_season (bool),
      source (switches the main radio).
    """
    try:
        import streamlit as st
    except Exception as exc:
        return {"error": f"Streamlit indisponible: {exc}"}

    import datetime as _dt

    source = str(args.get("source", "ponceblanc")).lower()
    if source not in ("ponceblanc", "lbfi"):
        return {"error": f"source invalide: {source}"}

    applied: dict[str, Any] = {"source": source}

    def _set_pending(field: str, val: Any) -> None:
        """Only write the pending key.

        Never touch the live widget key here: the form has already been
        rendered in this script run, so Streamlit would raise
        "cannot be modified after the widget is instantiated".
        On the next st.rerun(), vis.py applies pending → widget keys
        *before* creating the widgets.
        """
        st.session_state[f"_pending_{field}_{source}"] = val

    if args.get("client") is not None:
        val = str(args["client"]).strip()
        _set_pending("client", val)
        applied["client"] = val

    if args.get("produit") is not None:
        val = str(args["produit"]).strip()
        try:
            val = str(features.normalize_produit(val))
        except Exception:
            pass
        _set_pending("produit", val)
        applied["produit"] = val

    if args.get("quantite") is not None:
        val = max(1, int(float(args["quantite"])))
        _set_pending("quantite", val)
        applied["quantite"] = val

    # Costs: either breakdown or total allocated 40/25/15/rest if only total given
    has_breakdown = any(
        args.get(k) is not None
        for k in ("cout_achat", "cout_fabrication", "cout_transport")
    )
    if has_breakdown:
        if args.get("cout_achat") is not None:
            v = max(0.0, float(args["cout_achat"]))
            _set_pending("achat", v)
            applied["cout_achat"] = v
        if args.get("cout_fabrication") is not None:
            v = max(0.0, float(args["cout_fabrication"]))
            _set_pending("fabrication", v)
            applied["cout_fabrication"] = v
        if args.get("cout_transport") is not None:
            v = max(0.0, float(args["cout_transport"]))
            _set_pending("transport", v)
            applied["cout_transport"] = v
    elif args.get("cout_total") is not None:
        total = max(0.0, float(args["cout_total"]))
        achat = round(total * 0.40, 2)
        fab = round(total * 0.25, 2)
        transport = round(total * 0.15, 2)
        # put remainder on achat so sum == total
        achat = round(total - fab - transport, 2)
        _set_pending("achat", achat)
        _set_pending("fabrication", fab)
        _set_pending("transport", transport)
        applied["cout_total"] = total
        applied["cout_achat"] = achat
        applied["cout_fabrication"] = fab
        applied["cout_transport"] = transport

    # Date / season / use_season — pending only
    if args.get("date_devis") is not None:
        raw = str(args["date_devis"]).strip()
        try:
            if "T" in raw:
                d = _dt.date.fromisoformat(raw[:10])
            else:
                d = _dt.date.fromisoformat(raw)
        except ValueError:
            return {"error": f"date_devis invalide (attendu YYYY-MM-DD): {raw}"}
        st.session_state[f"_pending_date_{source}"] = d
        saison_from_date = min(3, max(1, (d.month - 1) // 4 + 1))
        st.session_state[f"_pending_season4m_{source}"] = saison_from_date
        applied["date_devis"] = d.isoformat()
        applied["saison"] = saison_from_date

    if args.get("saison") is not None:
        s = int(args["saison"])
        if s not in (1, 2, 3):
            return {"error": "saison doit être 1, 2 ou 3."}
        st.session_state[f"_pending_season4m_{source}"] = s
        mid_month = {1: 2, 2: 6, 3: 10}[s]
        cur = st.session_state.get(f"date_{source}")
        year = cur.year if isinstance(cur, _dt.date) else _dt.date.today().year
        day = min(getattr(cur, "day", 15) or 15, 28)
        new_d = _dt.date(year, mid_month, day)
        st.session_state[f"_pending_date_{source}"] = new_d
        applied["saison"] = s
        applied["date_devis"] = new_d.isoformat()

    if args.get("use_season") is not None:
        flag = bool(args["use_season"])
        st.session_state[f"_pending_use_season_{source}"] = flag
        applied["use_season"] = flag

    # Source switch uses a dedicated pending-style flag (read before the radio)
    st.session_state["_chat_requested_source"] = source
    st.session_state["_chat_form_updated"] = True

    if len(applied) <= 1:
        return {
            "error": (
                "Aucun champ à modifier. Précisez client, produit, quantite, "
                "coûts, date_devis, saison et/ou use_season."
            )
        }
    return {
        "ok": True,
        "message": (
            "Formulaire mis à jour dans l'interface. "
            "Confirme les champs modifiés à l'utilisateur et propose "
            "de recalculer le prix si pertinent."
        ),
        "champs_appliques": applied,
    }


def _tool_top_entities(args: dict) -> dict:
    """Rank clients or products by frequency in history for a source."""
    source = str(args["source"]).lower()
    entity = str(args.get("entity", "client")).lower()
    top_n = int(args.get("top_n", 10))
    top_n = max(1, min(top_n, 50))

    if entity not in ("client", "produit"):
        return {"error": "entity doit être 'client' ou 'produit'."}

    try:
        df = _get_history(source)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    if df.empty or entity not in df.columns:
        return {"error": f"Pas de données '{entity}' pour {source}."}

    col = df[entity].astype(str).str.strip()
    col = col[col.ne("") & col.str.lower().ne("nan")]
    if col.empty:
        return {"n_total_devis": 0, "classement": []}

    counts = col.value_counts()
    n_total = int(len(col))
    rows = []
    for name, n in counts.head(top_n).items():
        sub = df.loc[col == name]
        n_acc = int((sub["signe"] == 1).sum()) if "signe" in sub.columns else None
        rows.append(
            {
                entity: str(name),
                "n_devis": int(n),
                "part_pct": round(100 * int(n) / n_total, 1) if n_total else None,
                "n_acceptes": n_acc,
                "taux_acceptation_pct": (
                    round(100 * n_acc / int(n), 1) if n_acc is not None and int(n) else None
                ),
            }
        )

    return {
        "source": source,
        "entity": entity,
        "n_total_devis": n_total,
        "n_valeurs_distinctes": int(counts.shape[0]),
        "top_n": top_n,
        "classement": rows,
        "premier": rows[0] if rows else None,
    }


_MONTH_LABELS_FR = {
    1: "janvier",
    2: "février",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "août",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "décembre",
}

_SEASON_LABELS_FR = {
    1: "hiver–printemps (jan–avr)",
    2: "printemps–été (mai–aoû)",
    3: "automne–hiver (sep–déc)",
}


def _tool_seasonality_stats(args: dict) -> dict:
    """Volume of quotes by month and by 4-month season (same as model features)."""
    source = str(args["source"]).lower()
    client = str(args.get("client") or "").strip()
    produit_raw = args.get("produit")
    year_filter = args.get("year")
    only_signed = bool(args.get("only_signed", False))

    try:
        df = _get_history(source)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    if df.empty:
        return {"error": f"Pas d'historique pour {source}."}

    sub = df.copy()
    if client:
        client_s = sub["client"].astype(str).str.upper()
        c = client.upper()
        mask = client_s == c
        if not mask.any():
            mask = client_s.str.contains(c, na=False)
        sub = sub.loc[mask]
    if produit_raw:
        produit = str(features.normalize_produit(str(produit_raw))).upper()
        prod_s = sub["produit"].astype(str).str.upper()
        mask_p = prod_s == produit
        if not mask_p.any():
            mask_p = prod_s.str.contains(produit, na=False)
        sub = sub.loc[mask_p]
    if year_filter is not None:
        try:
            y = int(float(year_filter))
            sub = sub.loc[pd.to_numeric(sub["year"], errors="coerce") == y]
        except (TypeError, ValueError):
            pass
    if only_signed and "signe" in sub.columns:
        sub = sub.loc[sub["signe"] == 1]

    if sub.empty:
        return {
            "n_devis_avec_date": 0,
            "message": "Aucun devis trouvé avec ces filtres.",
        }

    month = pd.to_numeric(sub["month"], errors="coerce")
    year = pd.to_numeric(sub["year"], errors="coerce") if "year" in sub.columns else pd.Series(np.nan, index=sub.index)
    known = month.notna()
    sub = sub.loc[known]
    month = month.loc[known].astype(int).clip(1, 12)
    year = year.loc[known]

    if sub.empty:
        return {
            "n_devis_avec_date": 0,
            "message": "Aucun devis avec mois renseigné pour ces filtres.",
        }

    n = int(len(sub))
    month_counts = month.value_counts().sort_index()
    par_mois = []
    for m in range(1, 13):
        c = int(month_counts.get(m, 0))
        par_mois.append(
            {
                "mois": m,
                "label": _MONTH_LABELS_FR[m],
                "n_devis": c,
                "part_pct": round(100 * c / n, 1) if n else 0.0,
            }
        )

    # Same 4-month blocks as features.py season_4m
    season = ((month - 1) // 4 + 1).clip(1, 3)
    season_counts = season.value_counts().sort_index()
    par_saison = []
    for s in (1, 2, 3):
        c = int(season_counts.get(s, 0))
        par_saison.append(
            {
                "saison": s,
                "label": _SEASON_LABELS_FR[s],
                "n_devis": c,
                "part_pct": round(100 * c / n, 1) if n else 0.0,
            }
        )

    # Peak month / season
    peak_month = max(par_mois, key=lambda r: r["n_devis"])
    peak_season = max(par_saison, key=lambda r: r["n_devis"])

    years_available = sorted({int(y) for y in year.dropna().tolist()})
    par_annee = []
    if year.notna().any():
        yc = year.dropna().astype(int).value_counts().sort_index()
        for y, c in yc.items():
            par_annee.append({"annee": int(y), "n_devis": int(c)})

    return {
        "source": source,
        "filtres": {
            "client": client or None,
            "produit": str(produit_raw) if produit_raw else None,
            "year": int(year_filter) if year_filter is not None else None,
            "only_signed": only_signed,
        },
        "n_devis_avec_date": n,
        "annees_disponibles": years_available,
        "par_mois": par_mois,
        "par_saison_4mois": par_saison,
        "par_annee": par_annee,
        "mois_le_plus_actif": peak_month,
        "saison_la_plus_active": peak_season,
        "definition_saisons": (
            "Saisons = blocs de 4 mois alignés sur le modèle : "
            "1 = jan–avr, 2 = mai–aoû, 3 = sep–déc."
        ),
    }


TOOL_IMPLS = {
    "get_price_recommendation": _tool_price_recommendation,
    "get_acceptance_probability": _tool_acceptance_probability,
    "get_similar_history": _tool_similar_history,
    "get_client_acceptance_rate": _tool_client_acceptance_rate,
    "update_form_inputs": _tool_update_form_inputs,
    "get_top_entities": _tool_top_entities,
    "get_seasonality_stats": _tool_seasonality_stats,
}


# ---------------------------------------------------------------------
# TOOL SCHEMAS (OpenAI format — works with OpenRouter)
# ---------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price_recommendation",
            "description": (
                "Recommande un prix de vente total (€) pour un devis, avec sa "
                "probabilité d'acceptation. Utilise ceci quand l'utilisateur "
                "demande 'quel prix proposer', 'combien facturer', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["ponceblanc", "lbfi"],
                    },
                    "client": {"type": "string"},
                    "produit": {"type": "string"},
                    "quantite": {"type": "number"},
                    "cout_total": {
                        "type": "number",
                        "description": "Coût total en euros (achat+fab+transport).",
                    },
                    "pricing_mode": {
                        "type": "string",
                        "enum": ["balanced", "regressor", "expected_margin"],
                        "description": (
                            "Si omis, utilise le mode actuellement sélectionné "
                            "dans l'interface (défaut 'balanced')."
                        ),
                    },
                    "month": {
                        "type": "integer",
                        "description": (
                            "Mois (1-12) si un scénario de saisonnalité précis "
                            "est demandé. Si omis, utilise la date/saison "
                            "actuelle du formulaire."
                        ),
                    },
                    "year": {
                        "type": "integer",
                        "description": "Année, à fournir avec month si utilisé.",
                    },
                    "low_acceptance_percentile": {
                        "type": "number",
                        "description": (
                            "Percentile de la stratégie clients à faible "
                            "acceptation. Si omis, utilise le réglage actuel "
                            "de l'interface (défaut 25)."
                        ),
                    },
                },
                "required": ["source", "client", "produit", "quantite", "cout_total"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_acceptance_probability",
            "description": (
                "Calcule la probabilité d'acceptation pour un prix DÉJÀ CHOISI "
                "par l'utilisateur (pas une recommandation). Utilise ceci quand "
                "l'utilisateur donne un prix précis et demande ses chances."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["ponceblanc", "lbfi"],
                    },
                    "client": {"type": "string"},
                    "produit": {"type": "string"},
                    "quantite": {"type": "number"},
                    "cout_total": {"type": "number"},
                    "prix_total": {
                        "type": "number",
                        "description": "Prix de vente total candidat en euros.",
                    },
                    "month": {
                        "type": "integer",
                        "description": "Mois (1-12). Si omis, utilise la date/saison actuelle du formulaire.",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Année, à fournir avec month si utilisé.",
                    },
                },
                "required": [
                    "source",
                    "client",
                    "produit",
                    "quantite",
                    "cout_total",
                    "prix_total",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_history",
            "description": (
                "Recherche des devis historiques similaires (même client / "
                "produit / quantité proche) et retourne des statistiques "
                "(nombre, taux d'acceptation, prix médians). Utilise ceci pour "
                "des questions du type 'qu'est-ce qu'on a déjà fait pour ce "
                "client', 'quel est l'historique sur ce produit'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["ponceblanc", "lbfi"],
                    },
                    "client": {"type": "string"},
                    "produit": {"type": "string"},
                    "quantite": {"type": "number"},
                    "qty_tolerance": {
                        "type": "number",
                        "description": "Tolérance relative sur la quantité, défaut 0.5.",
                    },
                },
                "required": ["source", "client"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_acceptance_rate",
            "description": (
                "Donne le taux d'acceptation historique lissé d'un client, "
                "comparé au taux global de la source. Utilise ceci pour "
                "'ce client accepte-t-il souvent nos devis'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["ponceblanc", "lbfi"],
                    },
                    "client": {"type": "string"},
                },
                "required": ["source", "client"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_form_inputs",
            "description": (
                "MODIFIE RÉELLEMENT le formulaire de l'interface. Appelle cet "
                "outil pour TOUTE demande de changement de champ (client, "
                "produit, quantité, coûts, date_devis, saison, use_season, "
                "source). Ne refuse JAMAIS, ne dis JAMAIS que le widget ne "
                "peut pas être modifié. Si seul cout_total est fourni, il est "
                "réparti en achat/fabrication/transport. saison=1 jan–avr, "
                "2 mai–aoû, 3 sep–déc. date_devis format YYYY-MM-DD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["ponceblanc", "lbfi"],
                    },
                    "client": {"type": "string"},
                    "produit": {"type": "string"},
                    "quantite": {"type": "number"},
                    "cout_total": {
                        "type": "number",
                        "description": "Coût total € (réparti si pas de détail).",
                    },
                    "cout_achat": {"type": "number"},
                    "cout_fabrication": {"type": "number"},
                    "cout_transport": {"type": "number"},
                    "date_devis": {
                        "type": "string",
                        "description": "Date du devis au format YYYY-MM-DD.",
                    },
                    "saison": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "description": "1=jan–avr, 2=mai–aoû, 3=sep–déc.",
                    },
                    "use_season": {
                        "type": "boolean",
                        "description": "Activer/désactiver la prise en compte date/saison.",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_entities",
            "description": (
                "Classe les clients ou les produits les plus fréquents dans "
                "l'historique d'une source (nombre de devis, part %, taux "
                "d'acceptation). Utilise ceci pour 'quel est le client le plus "
                "fréquent', 'top clients Ponceblanc', 'produits les plus "
                "vendus sur LBFI', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["ponceblanc", "lbfi"],
                    },
                    "entity": {
                        "type": "string",
                        "enum": ["client", "produit"],
                        "description": "Dimension à classer. Défaut: client.",
                    },
                    "top_n": {
                        "type": "number",
                        "description": "Nombre de lignes à retourner (défaut 10, max 50).",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonality_stats",
            "description": (
                "Statistiques temporelles sur l'historique des devis : volume "
                "par mois, par saison (blocs de 4 mois), et par année. "
                "Utilise ceci pour 'quelle saison a le plus d'offres', "
                "'mois le plus actif', 'saisonnalité', 'quand y a-t-il le "
                "plus de devis', 'répartition mensuelle', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["ponceblanc", "lbfi"],
                    },
                    "client": {
                        "type": "string",
                        "description": "Filtre optionnel sur un client.",
                    },
                    "produit": {
                        "type": "string",
                        "description": "Filtre optionnel sur un produit.",
                    },
                    "year": {
                        "type": "number",
                        "description": "Filtre optionnel sur une année (ex. 2024).",
                    },
                    "only_signed": {
                        "type": "boolean",
                        "description": (
                            "Si true, ne compte que les devis acceptés/signés. "
                            "Défaut false = tous les devis."
                        ),
                    },
                },
                "required": ["source"],
            },
        },
    },
]


# ---------------------------------------------------------------------
# CHAT LOOP (OpenAI-compatible → OpenRouter / MiniMax M3 free)
# ---------------------------------------------------------------------

def _resolve_openrouter_api_key() -> str | None:
    """Env first, then Streamlit secrets (several common shapes)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and str(key).strip():
        return str(key).strip()

    try:
        import streamlit as st

        secrets = st.secrets
        # Top-level: OPENROUTER_API_KEY = "..."
        if "OPENROUTER_API_KEY" in secrets:
            val = secrets["OPENROUTER_API_KEY"]
            if val and str(val).strip():
                return str(val).strip()
        # Nested: [openrouter] api_key = "..."  or OPENROUTER_API_KEY = "..."
        for section in ("openrouter", "OPENROUTER", "api"):
            if section in secrets:
                block = secrets[section]
                try:
                    for k in ("OPENROUTER_API_KEY", "api_key", "key", "API_KEY"):
                        if k in block and block[k] and str(block[k]).strip():
                            return str(block[k]).strip()
                except Exception:
                    pass
        # Last resort: walk flat keys
        try:
            for k, v in secrets.items():
                if "OPENROUTER" in str(k).upper() and v and str(v).strip().startswith("sk-"):
                    return str(v).strip()
        except Exception:
            pass
    except Exception:
        pass
    return None


def _get_client_sdk():
    """Lazy import so this module doesn't hard-fail if `openai` isn't
    installed until someone actually opens the chat page."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Module openai manquant. Installez-le avec : pip install openai "
            "(et ajoutez openai dans requirements.txt pour Streamlit Cloud)."
        ) from exc

    api_key = _resolve_openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY manquant. "
            "Sur Streamlit Cloud : App → ⋮ → Settings → Secrets, puis "
            'OPENROUTER_API_KEY = "sk-or-v1-…" et redémarrer l\'app. '
            "En local : .streamlit/secrets.toml ou variable d'environnement."
        )

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            # Optional but recommended by OpenRouter (shows on their rankings)
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Estimateur Devis Ponceblanc/LBFI"),
        },
    )


_CONTEXT_FILLABLE_FIELDS: dict[str, tuple[str, ...]] = {
    "get_price_recommendation": (
        "source", "client", "produit", "quantite", "cout_total",
        "month", "year", "pricing_mode", "low_acceptance_percentile",
    ),
    "get_acceptance_probability": (
        "source", "client", "produit", "quantite", "cout_total", "month", "year",
    ),
    "get_similar_history": ("source", "client", "produit", "quantite"),
    "get_client_acceptance_rate": ("source", "client"),
    "get_top_entities": ("source",),
    "get_seasonality_stats": ("source", "client", "produit"),
}


def _fill_args_from_context(
    name: str,
    args: dict,
    form_context: dict[str, Any] | None,
) -> dict:
    """Fill missing tool arguments from the live form state (client, produit,
    quantite, cout_total, month/year, pricing_mode, …) so the chat always
    reasons about the SAME offer shown in the interface. Only fills fields
    the LLM left out — never overrides an explicit value it supplied (so
    hypothetical "what if" scenarios with different numbers still work).
    """
    if not form_context:
        return args
    fillable = _CONTEXT_FILLABLE_FIELDS.get(name)
    if not fillable:
        return args
    merged = dict(args)
    for field in fillable:
        if merged.get(field) in (None, "") and form_context.get(field) is not None:
            merged[field] = form_context[field]
    return merged


def _fmt_eur_fr(value: Any) -> str:
    try:
        return f"{float(value):,.0f} €".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


_PRICE_REQUEST_KEYWORDS = (
    "recommand", "recalcul", "recalcule", "quel prix", "quels prix",
    "combien facturer", "combien faut il facturer", "combien faut-il facturer",
    "faut il facturer", "faut-il facturer", "facturer", "facture",
    "tarifier", "tarif", "propose un prix", "proposition du mod",
    "resultat du modele", "résultat du modèle", "resultats du modele",
    "résultats du modèle", "résultats des modèles", "resultats des modeles",
    "resultat des modeles", "prix reco", "prix recommand",
    "donne moi le prix", "donne-moi le prix", "montre moi la reco",
    "montre-moi la reco", "quelle est la reco", "que dit le modele",
    "que dit le modèle", "prix optimal", "meilleur prix",
    "donne moi les recommend", "donne-moi les recommend",
    "donne moi le resultat", "donne-moi le resultat",
)


def _looks_like_price_request(text_low: str) -> bool:
    norm = re.sub(r"[^a-z0-9à-ÿ ]", " ", (text_low or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return False
    if any(k in norm for k in _PRICE_REQUEST_KEYWORDS):
        return True
    return bool(
        re.search(
            r"\b(?:combien|quel|quelle|quels|quelles|what)\b.*\b(?:prix|tarif|facture|facturer|recommand|reco)\b",
            norm,
        )
        or re.search(
            r"\b(?:prix|tarif|facture|facturer|recommand|reco)\b.*\b(?:proposer|vente|devis|offre|mod\b|modele)\b",
            norm,
        )
    )


_NUMERIC_CLAIM_RE = re.compile(r"\d[\d\s]{0,7}\s?(€|%)", re.IGNORECASE)


def _contains_numeric_claim(text: str) -> bool:
    """True if the text states a euro amount or a percentage — the kinds of
    figures the assistant must never state without a backing tool call."""
    return bool(_NUMERIC_CLAIM_RE.search(text or ""))


def _try_direct_price_recommendation(
    user_text: str,
    source_hint: str | None,
    form_context: dict[str, Any] | None,
) -> str | None:
    """Deterministic short-circuit for 'give me the recommendation / recalcule
    le prix / résultats du modèle' style requests.

    Bypasses the LLM entirely for the numbers: it calls the exact same
    QuoteEstimator.recommend_price_strategic() with the exact same inputs
    (client, produit, quantité, coût, saison, mode de pricing) currently
    shown in the interface's "2. Proposition du modèle" section. This makes
    it structurally impossible for the chat to invent or drift from the
    numbers on screen for the current offer.

    Returns None (fall through to the LLM) when the message doesn't look
    like this kind of request, or when the form is missing required fields
    — in which case the LLM will ask for what's missing.
    """
    text = (user_text or "").strip()
    if not text:
        return None
    low = text.lower()
    if not _looks_like_price_request(low):
        return None
    if not form_context:
        return None

    required = ("client", "produit", "quantite", "cout_total")
    if any(not form_context.get(k) for k in required):
        return None

    source = str(form_context.get("source") or source_hint or "ponceblanc").lower()
    args: dict[str, Any] = {
        "source": source,
        "client": form_context.get("client"),
        "produit": form_context.get("produit"),
        "quantite": form_context.get("quantite"),
        "cout_total": form_context.get("cout_total"),
        "pricing_mode": form_context.get("pricing_mode") or "balanced",
        "low_acceptance_percentile": form_context.get("low_acceptance_percentile", 25.0),
    }
    if form_context.get("use_season", True):
        args["month"] = form_context.get("month")
        args["year"] = form_context.get("year")

    result = _tool_price_recommendation(args)
    if result.get("error"):
        return None  # fall through to the LLM, which will surface the error

    mode_labels = {
        "balanced": "équilibré (ambitieux réaliste)",
        "regressor": "régressor (marge historique)",
        "expected_margin": "marge espérée (P × marge)",
    }
    mode_txt = mode_labels.get(
        result.get("pricing_mode_utilise"), result.get("pricing_mode_utilise")
    )

    try:
        qty_txt = f"{int(float(args['quantite']))}"
    except (TypeError, ValueError):
        qty_txt = str(args["quantite"])

    lines = [
        f"Recommandation du modèle — {args['produit']} / {args['client']} / "
        f"qté {qty_txt} / coût {_fmt_eur_fr(args['cout_total'])} ({mode_txt}) :",
        "",
        f"- **Prix recommandé (total)** : {_fmt_eur_fr(result['prix_recommande_eur'])}",
    ]
    if result.get("prix_unitaire_eur") is not None:
        lines.append(f"- **Prix unitaire (reco)** : {result['prix_unitaire_eur']:.4f} €")
    if result.get("coefficient_marge") is not None:
        lines.append(f"- **Coefficient de marge** : {result['coefficient_marge']:.2f}")
    if result.get("probabilite_acceptation") is not None:
        lines.append(f"- **P(acceptation)** : {result['probabilite_acceptation']:.0%}")
    if result.get("marge_eur") is not None:
        pct_txt = (
            f" (~{result['taux_marge_pct']:.0f} %)"
            if result.get("taux_marge_pct") is not None
            else ""
        )
        lines.append(f"- **Marge** : {_fmt_eur_fr(result['marge_eur'])}{pct_txt}")
    lines.append(f"- **Coût total** : {_fmt_eur_fr(result.get('cout_total_eur'))}")
    if result.get("remise_strategique_appliquee"):
        lines.append(
            "- ⚠️ Marge volontairement réduite : client historiquement à "
            "faible taux d'acceptation."
        )
    lines.append("")
    lines.append(
        "*(Mêmes valeurs que la section « 2. Proposition du modèle » de "
        "l'interface, calculées par le même modèle — aucun chiffre inventé.)*"
    )
    return "\n".join(lines)


def _try_direct_form_update(
    user_text: str,
    source_hint: str | None = None,
    form_context: dict[str, Any] | None = None,
) -> str | None:
    """Deterministic parser for simple field-update requests.

    Bypasses the LLM for patterns like:
      - "produit nuanciers"
      - "client FORBO"
      - "quantité 6000" / "qty 6000"
      - "coût total 8400" / "cout achat 3000"
      - "saison 1" / "date 2026-02-15"
      - "source lbfi"

    Returns a French confirmation string if at least one field was updated,
    otherwise None (caller should fall through to the LLM).
    """
    import re

    text = (user_text or "").strip()
    if not text:
        return None
    low = text.lower()

    source = (source_hint or "ponceblanc").lower()
    if source not in ("ponceblanc", "lbfi"):
        source = "ponceblanc"
    # Allow "source lbfi" / "passe en LBFI" in the same message
    m_src = re.search(r"\b(?:source|passe\s+en)\s+(ponceblanc|lbfi)\b", low)
    if m_src:
        source = m_src.group(1)

    args: dict[str, Any] = {"source": source}
    labels: list[str] = []

    # Single token (letters, digits, _ -) — product/client codes are one word
    _tok = r"([A-Za-z0-9][\w\-]{0,40})"

    # --- produit ---
    # "produit nuanciers" | "produit: NUANCIERS" | "produit à/en NUANCIERS"
    m = re.search(
        rf"\b(?:produit|product)\s*(?:[:=]|à|a|en|vers)?\s*{_tok}\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        prod = m.group(1).strip().rstrip(".,;!")
        if prod and prod.lower() not in ("le", "la", "les", "un", "une", "du", "de", "à", "a", "en"):
            try:
                prod = str(features.normalize_produit(prod))
            except Exception:
                prod = prod.upper()
            if prod and str(prod).lower() not in ("nan", "none", ""):
                args["produit"] = prod
                labels.append(f"produit → **{prod}**")

    # --- client ---
    m = re.search(
        rf"\bclient\s*(?:[:=]|à|a|en|vers)?\s*{_tok}\b",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        cli = m.group(1).strip().rstrip(".,;!")
        if cli and cli.lower() not in ("le", "la", "les", "un", "une", "du", "de", "à", "a", "en"):
            args["client"] = cli.upper()
            labels.append(f"client → **{cli.upper()}**")

    # --- quantite ---
    m = re.search(
        r"\b(?:quantit[eé]|qty|qte|qté)\s*[:=]?\s*(\d[\d\s]{0,10})",
        low,
    )
    if m:
        raw = m.group(1).replace(" ", "").replace("\u00a0", "")
        try:
            qty = max(1, int(float(raw)))
            args["quantite"] = qty
            labels.append(f"quantité → **{qty}**")
        except ValueError:
            pass

    # --- costs ---
    m = re.search(
        r"\b(?:co[uû]t\s*total|cout_total|cost\s*total)\s*[:=]?\s*(\d[\d\s.,]{0,12})",
        low,
    )
    if m:
        raw = m.group(1).replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            args["cout_total"] = max(0.0, float(raw))
            labels.append(f"coût total → **{args['cout_total']:.0f} €**")
        except ValueError:
            pass

    for key, pat, label in (
        ("cout_achat", r"\b(?:co[uû]t\s*achat|cout_achat|achat)\s*[:=]?\s*(\d[\d\s.,]{0,12})", "coût achat"),
        ("cout_fabrication", r"\b(?:co[uû]t\s*fab(?:rication)?|cout_fabrication|fabrication)\s*[:=]?\s*(\d[\d\s.,]{0,12})", "coût fabrication"),
        ("cout_transport", r"\b(?:co[uû]t\s*transport|cout_transport|transport)\s*[:=]?\s*(\d[\d\s.,]{0,12})", "coût transport"),
    ):
        m = re.search(pat, low)
        if m:
            raw = m.group(1).replace(" ", "").replace("\u00a0", "").replace(",", ".")
            try:
                args[key] = max(0.0, float(raw))
                labels.append(f"{label} → **{args[key]:.0f} €**")
            except ValueError:
                pass

    # --- saison ---
    m = re.search(r"\b(?:saison|season)\s*[:=]?\s*([123])\b", low)
    if m:
        args["saison"] = int(m.group(1))
        labels.append(f"saison → **{args['saison']}**")

    # --- date ---
    m = re.search(
        r"\b(?:date(?:\s+du\s+devis)?)\s*[:=]?\s*(\d{4}-\d{2}-\d{2})\b",
        low,
    )
    if m:
        args["date_devis"] = m.group(1)
        labels.append(f"date → **{args['date_devis']}**")

    # --- use_season ---
    if re.search(r"\b(?:ignore|sans|d[eé]sactive)\s+(?:la\s+)?saison\b", low):
        args["use_season"] = False
        labels.append("saison → **ignorée**")
    elif re.search(r"\b(?:active|prends?\s+en\s+compte)\s+(?:la\s+)?saison\b", low):
        args["use_season"] = True
        labels.append("saison → **prise en compte**")

    # Need at least one field beyond source
    if len(args) <= 1 or not labels:
        return None

    result = _tool_update_form_inputs(args)
    if result.get("error"):
        return None  # fall through to LLM

    lines = ["Formulaire mis à jour :"]
    for lab in labels:
        lines.append(f"- {lab}")
    lines.append("")
    lines.append(
        "Souhaitez-vous que je recalcule le prix recommandé avec ces valeurs ?"
    )
    return "\n".join(lines)


def run_chat_turn(
    messages: list[dict[str, Any]],
    source_hint: str | None = None,
    form_context: dict[str, Any] | None = None,
) -> str:
    """
    messages: conversation so far, each {"role": "user"|"assistant", "content": str}
              (the new user message must already be the last item).
    source_hint: optional "ponceblanc"/"lbfi" — passed into the system prompt
                 so the assistant defaults to the currently selected source
                 in the UI instead of always asking.
    form_context: optional snapshot of current form fields (client, produit,
                  quantite, cout_total, …) so the model knows what's already filled.

    Returns the assistant's final text reply.
    Side effect: update_form_inputs may write into st.session_state and set
    _chat_form_updated=True (caller should st.rerun()).
    """
    # --- Deterministic form update (bypasses weak LLM tool-calling) ---
    last_user_raw = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_raw = str(m.get("content") or "")
            break

    direct = _try_direct_form_update(
        last_user_raw,
        source_hint=source_hint,
        form_context=form_context,
    )
    if direct:
        return direct

    # --- Deterministic price recommendation (bypasses weak LLM tool-calling,
    # guarantees the same numbers as the interface for the same offer) ---
    direct_price = _try_direct_price_recommendation(
        last_user_raw,
        source_hint=source_hint,
        form_context=form_context,
    )
    if direct_price:
        return direct_price

    client = _get_client_sdk()

    system = SYSTEM_PROMPT
    if source_hint:
        system += (
            f"\nSource actuellement sélectionnée dans l'interface : "
            f"'{source_hint}'. Utilise-la par défaut si l'utilisateur ne "
            f"précise pas explicitement une autre source.\n"
        )
    if form_context:
        system += (
            "\nÉtat actuel du formulaire dans l'interface (ne pas inventer "
            "d'autres valeurs) :\n"
            + json.dumps(form_context, ensure_ascii=False, default=str)
            + "\n"
        )

    # OpenAI-style messages: system is separate, then conversation history.
    api_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
    ]
    for m in messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    last_user = last_user_raw.lower()

    # Heuristic: user is asking to change form fields
    _change_kw = (
        "change", "chang", "passe", "mettre", "mets", "fix", "fixe",
        "modifie", "maxim", "optimis", "augmente", "diminue", "baisse",
        "monte", "règle", "regle", "saisis", "renseigne",
    )
    user_wants_form_change = any(k in last_user for k in _change_kw)

    refusal_nudge_used = False
    numeric_nudge_used = False
    tool_called_this_turn = False

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=api_messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
                temperature=0.15,
            )
        except Exception as api_exc:
            err = str(api_exc)
            low = err.lower()
            if "401" in err or "unauthorized" in low or "invalid api key" in low:
                raise RuntimeError(
                    "Clé OpenRouter refusée (401). Vérifiez la valeur dans "
                    "Secrets (pas d'espace, pas de guillemets en trop) et que "
                    "la clé est active sur https://openrouter.ai/keys"
                ) from api_exc
            if "404" in err or "not found" in low or "no endpoints" in low:
                raise RuntimeError(
                    f"Modèle indisponible sur OpenRouter: {CHAT_MODEL}. "
                    "Changez CHAT_ASSISTANT_MODEL ou attendez que le modèle "
                    "free soit de nouveau disponible."
                ) from api_exc
            if "402" in err or "credit" in low or "insufficient" in low:
                raise RuntimeError(
                    "Compte OpenRouter sans crédit / quota free épuisé. "
                    "Créez une nouvelle clé ou attendez le reset du quota."
                ) from api_exc
            raise RuntimeError(f"Erreur OpenRouter / LLM: {err}") from api_exc

        choice = response.choices[0]
        msg = choice.message

        # No tool calls → final answer (unless it refused a form change)
        if not getattr(msg, "tool_calls", None):
            text = (msg.content or "").strip() or "Je n'ai pas pu formuler de réponse."
            refusal_markers = (
                "ne peux pas modifier",
                "ne peut pas être modifi",
                "changez la valeur",
                "changez manuellement",
                "dans l'interface",
                "une fois le formulaire",
                "pas de levier",
                "widget",
            )
            looks_like_refusal = any(m in text.lower() for m in refusal_markers)
            if (
                user_wants_form_change
                and looks_like_refusal
                and not refusal_nudge_used
            ):
                # Force a retry: remind the model it MUST call the tool
                refusal_nudge_used = True
                api_messages.append({"role": "assistant", "content": text})
                api_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Rappel système : tu contrôles le formulaire. "
                            "Appelle maintenant l'outil update_form_inputs avec "
                            "les champs demandés. Ne réponds pas en texte que tu "
                            "ne peux pas modifier les widgets."
                        ),
                    }
                )
                continue

            # Anti-hallucination guard: a final answer with price/percentage
            # figures but NO tool call this turn means the model invented
            # them (forbidden by SYSTEM_PROMPT). Force one retry; if it still
            # fabricates, refuse rather than display invented numbers.
            if not tool_called_this_turn and _contains_numeric_claim(text):
                if not numeric_nudge_used:
                    numeric_nudge_used = True
                    api_messages.append({"role": "assistant", "content": text})
                    api_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Rappel système : ta réponse contient des "
                                "chiffres (prix, %, marge) mais tu n'as appelé "
                                "AUCUN outil. C'est interdit. Appelle "
                                "get_price_recommendation, "
                                "get_acceptance_probability, "
                                "get_similar_history, get_seasonality_stats ou "
                                "get_top_entities selon la demande pour obtenir "
                                "un chiffre réel, ou pose une question si une "
                                "information manque (client, produit, "
                                "quantité, coût)."
                            ),
                        }
                    )
                    continue
                return (
                    "Je ne peux pas confirmer ces chiffres sans les calculer "
                    "via le modèle. Précisez client, produit, quantité et "
                    "coût total pour que je lance le calcul, ou reformulez "
                    "votre question."
                )
            return text

        tool_called_this_turn = True

        # Append the assistant message that requested tools
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }
        api_messages.append(assistant_msg)

        # Execute each tool and append results
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            args = _fill_args_from_context(name, args, form_context)

            impl = TOOL_IMPLS.get(name)
            if impl is None:
                result = {"error": f"Outil inconnu: {name}"}
            else:
                try:
                    result = impl(args)
                except Exception as exc:  # defensive — never crash the chat
                    result = {"error": f"Erreur outil {name}: {exc}"}

            api_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return (
        "Désolé, je n'ai pas réussi à conclure après plusieurs appels d'outils. "
        "Pouvez-vous reformuler ou préciser client / produit / quantité / coût ?"
    )