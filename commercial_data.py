"""
commercial_data.py
===================

Disk-based port of the "Tableau de bord Commercial" logic
(dashboard_commercial_only.py, separate repo) so the pricing-tool chat
assistant can answer commercial-dashboard-style questions — ABC client
classification, KPI formulas (CA devisé/signé, taux de succès, commande
moyenne), CA breakdowns, delay to dossier opening — without requiring a
human to manually upload the Excel file each session.

Source file: Query_tableau_devis.xlsx (sheets 'LBFI' / 'PONCEBLANC'),
bundled on disk. This is a SEPARATE export from
Query_tableau_devis_with_costs.xlsx used by features.py / predict.py for
the pricing models: no cost columns, but it does have 'Commercial' and a
slightly different column set / cleaning history. Do NOT blend the two casually
— tools built on this module report "tableau de bord commercial" numbers,
which can differ slightly from the ML pipeline's "historique devis" numbers
(different source export, different cleaning rules, different row counts).

Every function here is a direct, non-approximated port of
dashboard_commercial_only.py's load_data() / calculer_classification_abc_par_annee()
/ extraire_kpis_annee() — same business rules, same numbers a human would see
on that dashboard page for the same file.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent

COMMERCIAL_XLSX_PATH = os.environ.get(
    "COMMERCIAL_XLSX",
    str(_BASE / "Query_tableau_devis.xlsx"),
)
_FALLBACK_COMMERCIAL_XLSX = str(_BASE / "Query_tableau_devis.xlsx")

VALID_SOURCES = ("ponceblanc", "lbfi")
_SHEET_NAMES = {"ponceblanc": "PONCEBLANC", "lbfi": "LBFI"}

_MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
    7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre",
    12: "Décembre",
}


# ---------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------

def _resolve_path() -> str:
    for p in (COMMERCIAL_XLSX_PATH, _FALLBACK_COMMERCIAL_XLSX):
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Query_tableau_devis.xlsx introuvable (données du tableau de bord "
        f"commercial). Chemins essayés : {COMMERCIAL_XLSX_PATH!r}, "
        f"{_FALLBACK_COMMERCIAL_XLSX!r}. Ajoutez le fichier au dépôt "
        "(ex. data/Query_tableau_devis.xlsx) pour activer les outils "
        "commerciaux du chat."
    )


def load_data(source: str) -> pd.DataFrame:
    """Faithful port of dashboard_commercial_only.load_data(): same cleaning
    rules, same column names, same output shape a human sees on the
    commercial dashboard — just read from disk instead of an upload."""
    source = source.strip().lower()
    if source not in VALID_SOURCES:
        raise ValueError(f"source doit être {VALID_SOURCES}, reçu {source!r}")
    bu_name = _SHEET_NAMES[source]

    path = _resolve_path()
    df = pd.read_excel(path, sheet_name=bu_name, engine="openpyxl")
    df.columns = df.columns.str.strip()

    if bu_name == "PONCEBLANC":
        df = df.rename(columns={
            "Dates": "Date devis",
            "Taux Marge": "Taux de marge",
            "Date Ouverture": "Date ouverture dossier fab",
            "Signé": "Signé?",
        })

    if "Date devis" in df.columns:
        s_numeric = pd.to_numeric(df["Date devis"], errors="coerce")
        df["Dates_Propres"] = pd.to_datetime(
            s_numeric, unit="D", origin="1899-12-30", errors="coerce"
        )
        df["Dates_Propres"] = df["Dates_Propres"].fillna(
            pd.to_datetime(df["Date devis"], errors="coerce")
        )
        df = df.dropna(subset=["Dates_Propres"])

    if "Dates_Propres" in df.columns:
        df["Mois_Num"] = df["Dates_Propres"].dt.month.fillna(1).astype(int)
        df["Année Devis"] = df["Dates_Propres"].dt.year.fillna(2026).astype(int)
    else:
        df["Mois_Num"] = 1
        df["Année Devis"] = 2026

    df["Mois_Nom"] = df["Mois_Num"].map(_MOIS_FR).fillna("Janvier")

    df["Prix total"] = pd.to_numeric(df["Prix total"], errors="coerce")

    if "Taux de marge" in df.columns:
        df["Taux de marge"] = pd.to_numeric(df["Taux de marge"], errors="coerce")
        df["Taux de marge"] = df["Taux de marge"].where(
            df["Taux de marge"].between(0, 100)
        )

    if "Nb exemplaires" in df.columns:
        df["Nb exemplaires"] = pd.to_numeric(df["Nb exemplaires"], errors="coerce")
        df["Prix unitaire"] = df["Prix total"] / df["Nb exemplaires"].replace(0, pd.NA)
        df["Prix unitaire"] = df["Prix unitaire"].where(df["Prix unitaire"] > 0)

    df["Prix total"] = df["Prix total"].fillna(0)

    if "Signé ?" in df.columns:
        df = df.rename(columns={"Signé ?": "Signé?"})
    if "Signé?" in df.columns:
        df["Signé?"] = df["Signé?"].astype(str).str.strip().str.upper()
        df["Signé?"] = df["Signé?"].map(
            {"VRAI": "O", "TRUE": "O", "O": "O", "FAUX": "N", "FALSE": "N", "N": "N"}
        ).fillna("N")

    if "Date ouverture dossier fab" in df.columns and "Dates_Propres" in df.columns:
        date_ouv_num = pd.to_numeric(df["Date ouverture dossier fab"], errors="coerce")
        date_ouv = pd.to_datetime(
            date_ouv_num, unit="D", origin="1899-12-30", errors="coerce"
        )
        date_ouv = date_ouv.fillna(
            pd.to_datetime(df["Date ouverture dossier fab"], errors="coerce")
        )
        df["Délai devis ouverture"] = (date_ouv - df["Dates_Propres"]).dt.days
        df["Délai devis ouverture"] = df["Délai devis ouverture"].where(
            df["Délai devis ouverture"] >= 0
        )

    if "Numéro de devis" in df.columns:
        df = df.rename(columns={"Numéro de devis": "DEVIS N°"})

    return df


# ---------------------------------------------------------------------
# ABC CLASSIFICATION (verbatim port)
# ---------------------------------------------------------------------

def calculer_classification_abc_par_annee(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty or "Année Devis" not in dataframe.columns:
        dataframe = dataframe.copy()
        dataframe["Catégorie Client ABC"] = "NOUVEAU CLIENT"
        return dataframe

    mask_signe_global = dataframe["Signé?"].astype(str).str.upper().str.strip() == "O"
    annees_triees = sorted(dataframe["Année Devis"].unique())

    clients_connus_avant = {}
    clients_cumul: set = set()
    for annee in annees_triees:
        clients_connus_avant[annee] = frozenset(clients_cumul)
        nouveaux = set(
            dataframe.loc[
                (dataframe["Année Devis"] == annee) & mask_signe_global,
                "Nom Client",
            ].dropna().unique()
        )
        clients_cumul |= nouveaux

    categories = pd.Series("NOUVEAU CLIENT", index=dataframe.index)

    for annee in annees_triees:
        mask_annee = dataframe["Année Devis"] == annee
        df_annee = dataframe[mask_annee]
        connus = clients_connus_avant[annee]

        df_annee_signe = df_annee[
            mask_signe_global.reindex(df_annee.index, fill_value=False)
        ]
        ca_par_client = (
            df_annee_signe.groupby("Nom Client")["Prix total"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        ca_par_client_connus = ca_par_client[ca_par_client["Nom Client"].isin(connus)]

        total_ca = ca_par_client_connus["Prix total"].sum()
        dict_abc = {}

        if total_ca > 0:
            ca_par_client_connus = ca_par_client_connus.copy()
            ca_par_client_connus["CA_Cumule_Pct"] = (
                ca_par_client_connus["Prix total"].cumsum() / total_ca * 100
            )
            for _, row in ca_par_client_connus.iterrows():
                client = row["Nom Client"]
                pct = row["CA_Cumule_Pct"]
                if pct <= 80.001:
                    dict_abc[client] = "GRAND COMPTE"
                elif pct <= 90.001:
                    dict_abc[client] = "CLIENT INTERMEDIAIRE"
                else:
                    dict_abc[client] = "PETITS CLIENTS"

        for idx, row in df_annee.iterrows():
            client = row["Nom Client"]
            if client not in connus:
                categories.at[idx] = "NOUVEAU CLIENT"
            elif client in dict_abc:
                categories.at[idx] = dict_abc[client]
            else:
                categories.at[idx] = "PETITS CLIENTS"

    dataframe = dataframe.copy()
    dataframe["Catégorie Client ABC"] = categories
    return dataframe


# ---------------------------------------------------------------------
# KPI FORMULA (verbatim port)
# ---------------------------------------------------------------------

def extraire_kpis_annee(dataframe: pd.DataFrame, annee_cible: int) -> dict | None:
    df_cible = dataframe[dataframe["Année Devis"] == annee_cible]
    col_id = "DEVIS N°" if "DEVIS N°" in dataframe.columns else dataframe.columns[0]

    if df_cible.empty:
        return None

    mask_signe = df_cible["Signé?"].astype(str).str.upper().str.strip() == "O"

    ca_devis = df_cible["Prix total"].sum()
    ca_signe = df_cible[mask_signe]["Prix total"].sum()
    tx_succes_ca = (ca_signe / ca_devis * 100) if ca_devis > 0 else 0

    vol_devis = df_cible[col_id].nunique()
    vol_signe = df_cible[mask_signe][col_id].nunique()
    tx_succes_vol = (vol_signe / vol_devis * 100) if vol_devis > 0 else 0

    cmd_moy_tous = (ca_devis / vol_devis) if vol_devis > 0 else 0
    cmd_moy_signe = (ca_signe / vol_signe) if vol_signe > 0 else 0

    return {
        "ca_devis": float(ca_devis),
        "ca_signe": float(ca_signe),
        "tx_ca": float(tx_succes_ca),
        "vol_devis": int(vol_devis),
        "vol_signe": int(vol_signe),
        "tx_vol": float(tx_succes_vol),
        "cmd_moy_tous": float(cmd_moy_tous),
        "cmd_moy_signe": float(cmd_moy_signe),
    }


# ---------------------------------------------------------------------
# CACHED ACCESS (mirrors the _history_cache pattern in chat_assistant.py)
# ---------------------------------------------------------------------

_data_cache: dict[str, pd.DataFrame] = {}


def get_commercial_df(source: str) -> pd.DataFrame:
    """Loaded + ABC-classified DataFrame for one source, cached per process."""
    source = source.strip().lower()
    if source not in _data_cache:
        df = load_data(source)
        df = calculer_classification_abc_par_annee(df)
        _data_cache[source] = df
    return _data_cache[source]