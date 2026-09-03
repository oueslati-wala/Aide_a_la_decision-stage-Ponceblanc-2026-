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
- Si une information nécessaire manque (client, produit, quantité, coût,
  source Ponceblanc/LBFI), demande-la avant d'appeler un outil, sauf si un
  défaut raisonnable est évident depuis la conversation.
- Réponds en français, de façon concise et professionnelle, comme à un
  commercial qui n'est pas data scientist. Explique les chiffres en une ou
  deux phrases, pas de jargon ML inutile.
- "source" doit valoir "ponceblanc" ou "lbfi". Si l'utilisateur ne précise
  pas, utilise la source indiquée dans le contexte de l'interface.
- Tu peux remplir ou modifier les champs du formulaire de la page
  (client, produit, quantité, coûts) via l'outil update_form_inputs quand
  l'utilisateur te le demande ou te donne ces infos pour un devis.
- Tu as accès à la saisonnalité de l'historique (outil get_seasonality_stats) :
  volume de devis par mois, par saison (blocs de 4 mois : jan–avr / mai–aoû /
  sep–déc) et par année. Utilise-le pour toute question sur les périodes,
  mois ou saisons les plus actives.
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
    est = _get_estimator()
    try:
        rec = est.recommend_price_strategic(
            client=str(args["client"]),
            produit=str(args["produit"]),
            quantite=float(args["quantite"]),
            source=str(args["source"]).lower(),
            cout_total=float(args["cout_total"]),
            pricing_mode=str(args.get("pricing_mode", "balanced")),
        )
        return {
            "prix_recommande_eur": round(rec["prix_median"], 2),
            "prix_min_eur": round(rec["prix_lower"], 2),
            "prix_max_eur": round(rec["prix_upper"], 2),
            "coefficient_marge": rec.get("coeff_median"),
            "probabilite_acceptation": rec.get("acceptance_probability"),
            "remise_strategique_appliquee": rec.get("strategic_pricing_applied", False),
            "remise_pct": rec.get("strategic_discount_pct"),
            "raison_remise": (
                "Client historiquement à faible taux d'acceptation : marge "
                "volontairement réduite pour augmenter la chance de signature."
                if rec.get("strategic_pricing_applied") else None
            ),
        }
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return {"error": str(exc)}


def _tool_acceptance_probability(args: dict) -> dict:
    est = _get_estimator()
    try:
        proba = est.predict_acceptance_proba(
            client=str(args["client"]),
            produit=str(args["produit"]),
            quantite=float(args["quantite"]),
            source=str(args["source"]).lower(),
            cout_total=float(args["cout_total"]),
            prix_total=float(args["prix_total"]),
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
    """Write form fields into Streamlit session_state so the main page updates."""
    try:
        import streamlit as st
    except Exception as exc:
        return {"error": f"Streamlit indisponible: {exc}"}

    source = str(args.get("source", "ponceblanc")).lower()
    if source not in ("ponceblanc", "lbfi"):
        return {"error": f"source invalide: {source}"}

    applied: dict[str, Any] = {"source": source}

    if args.get("client") is not None:
        val = str(args["client"]).strip()
        st.session_state[f"client_{source}"] = val
        applied["client"] = val

    if args.get("produit") is not None:
        val = str(args["produit"]).strip()
        try:
            val = str(features.normalize_produit(val))
        except Exception:
            pass
        st.session_state[f"produit_{source}"] = val
        applied["produit"] = val

    if args.get("quantite") is not None:
        val = max(1, int(float(args["quantite"])))
        st.session_state[f"quantite_{source}"] = val
        applied["quantite"] = val

    # Costs: either breakdown or total allocated 40/25/15/rest if only total given
    has_breakdown = any(
        args.get(k) is not None
        for k in ("cout_achat", "cout_fabrication", "cout_transport")
    )
    if has_breakdown:
        if args.get("cout_achat") is not None:
            v = max(0.0, float(args["cout_achat"]))
            st.session_state[f"achat_{source}"] = v
            applied["cout_achat"] = v
        if args.get("cout_fabrication") is not None:
            v = max(0.0, float(args["cout_fabrication"]))
            st.session_state[f"fabrication_{source}"] = v
            applied["cout_fabrication"] = v
        if args.get("cout_transport") is not None:
            v = max(0.0, float(args["cout_transport"]))
            st.session_state[f"transport_{source}"] = v
            applied["cout_transport"] = v
    elif args.get("cout_total") is not None:
        total = max(0.0, float(args["cout_total"]))
        achat = round(total * 0.40, 2)
        fab = round(total * 0.25, 2)
        transport = round(total * 0.15, 2)
        # put remainder on achat so sum == total
        achat = round(total - fab - transport, 2)
        st.session_state[f"achat_{source}"] = achat
        st.session_state[f"fabrication_{source}"] = fab
        st.session_state[f"transport_{source}"] = transport
        applied["cout_total"] = total
        applied["cout_achat"] = achat
        applied["cout_fabrication"] = fab
        applied["cout_transport"] = transport

    # Switch the UI source radio if needed (main page key)
    st.session_state["_chat_requested_source"] = source
    st.session_state["_chat_form_updated"] = True

    if len(applied) <= 1:
        return {
            "error": "Aucun champ à modifier. Précisez client, produit, quantite et/ou coûts."
        }
    return {
        "ok": True,
        "message": "Formulaire mis à jour dans l'interface.",
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
                        "description": "Défaut 'balanced' si non précisé.",
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
                "Remplit ou modifie les champs du formulaire de la page "
                "(client, produit, quantité, coûts). Utilise ceci quand "
                "l'utilisateur donne les paramètres d'un devis ou demande "
                "de préremplir / changer le formulaire. Si seul cout_total "
                "est fourni, il est réparti en achat/fabrication/transport."
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

def _get_client_sdk():
    """Lazy import so this module doesn't hard-fail if `openai` isn't
    installed until someone actually opens the chat page."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Module openai manquant. Installez-le avec : pip install openai"
        ) from exc

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENROUTER_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY manquant. Créez une clé gratuite sur "
            "https://openrouter.ai/keys , puis définissez la variable "
            "d'environnement OPENROUTER_API_KEY ou ajoutez-la dans "
            ".streamlit/secrets.toml (clé: OPENROUTER_API_KEY)."
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

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=api_messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.2,
        )

        choice = response.choices[0]
        msg = choice.message

        # No tool calls → final answer
        if not getattr(msg, "tool_calls", None):
            return (msg.content or "").strip() or "Je n'ai pas pu formuler de réponse."

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
