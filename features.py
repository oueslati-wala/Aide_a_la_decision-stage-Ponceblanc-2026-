"""
features.py
===========

Data preparation for Ponceblanc and LBFI.

IMPORTANT
---------
The two sources are completely separated.

Both sources use the same EURO workflow when cost data is available:

    Inputs:
        - client
        - produit
        - quantite
        - cout_total (€)
        - candidate prix_total (€), when evaluating acceptance

    Target for the price model:
        - prix_total (€)

    Margin rates / coefficients are display-only and never fed into the models.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


_BASE = Path(__file__).resolve().parent

# Preferred locations (package-relative). Override with env vars if needed.
UNIFIED_XLSX_PATH = os.environ.get(
    "DEVIS_XLSX",
    str(_BASE / "data" / "Query_tableau_devis_with_costs.xlsx"),
)

_FALLBACK_XLSX = str(_BASE / "Query_tableau_devis_with_costs.xlsx")

PONCEBLANC_CSV_PATH = os.environ.get(
    "PONCEBLANC_CSV",
    str(_BASE / "data" / "Query_tableau_devis_with_costs(PONCEBLANC).csv"),
)

LBFI_CSV_PATH = os.environ.get(
    "LBFI_CSV",
    str(_BASE / "data" / "Query_tableau_devis_with_costs(LBFI).csv"),
)

_FALLBACK_PONCE_CSV = str(_BASE / "Query_tableau_devis_with_costs(PONCEBLANC).csv")
_FALLBACK_LBFI_CSV = str(_BASE / "Query_tableau_devis_with_costs(LBFI).csv")


UNIFIED_COLUMNS = [
    "devis_code",
    "source",
    "client",
    "produit",
    "quantite",
    "taux_marge",
    "prix_total",
    "prix_unitaire",
    "delai_jours",
    "cout_total",
    "signe",
]

VALID_SOURCES = ("ponceblanc", "lbfi")


# ---------------------------------------------------------------------
# PATH HELPERS
# ---------------------------------------------------------------------

def _resolve_path(*candidates: str) -> str | None:
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _resolve_xlsx() -> str | None:
    return _resolve_path(
        UNIFIED_XLSX_PATH,
        _FALLBACK_XLSX,
    )


def _resolve_csv(
    preferred: str,
    *fallbacks: str,
) -> str:

    path = _resolve_path(preferred, *fallbacks)

    if path:
        return path

    raise FileNotFoundError(
        "CSV file not found.\n"
        + "\n".join(
            f"  tried: {p}"
            for p in (preferred, *fallbacks)
        )
    )


# ---------------------------------------------------------------------
# READING
# ---------------------------------------------------------------------

def _read_csv_robust(
    path: str,
    encodings: list[str],
    separators: list[str],
) -> pd.DataFrame:

    last_error = None

    for encoding in encodings:
        for separator in separators:
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=separator,
                )

                if df.shape[1] > 1:
                    return df

            except Exception as exc:
                last_error = exc

    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
            )

            if df.shape[1] > 1:
                return df

        except Exception as exc:
            last_error = exc

    raise ValueError(
        f"Could not parse {path!r}. "
        f"Last error: {last_error}"
    )


def _find_col(
    columns,
    must_contain: list[str],
    label: str,
):

    for column in columns:

        low = str(column).strip().lower()

        if all(part in low for part in must_contain):
            return column

    raise KeyError(
        f"Could not find {label!r}. "
        f"Expected header containing {must_contain!r}. "
        f"Actual columns: {list(columns)}"
    )


def _find_sheet_name(
    sheet_names: list[str],
    source: str,
) -> str:

    keys = {
        "ponceblanc": [
            "ponceblanc",
            "ponce blanc",
            "pb",
        ],
        "lbfi": [
            "lbfi",
            "lb fi",
        ],
    }[source]

    for name in sheet_names:

        low = str(name).strip().lower()

        if any(key in low for key in keys):
            return name

    raise KeyError(
        f"No Excel sheet found for {source!r}. "
        f"Available sheets: {sheet_names}"
    )


def _read_xlsx_sheet(
    source: str,
    xlsx_path: str | None = None,
) -> pd.DataFrame:

    path = xlsx_path or _resolve_xlsx()

    if not path:
        raise FileNotFoundError(
            "Query_tableau_devis_with_costs.xlsx was not found."
        )

    xl = pd.ExcelFile(
        path,
        engine="openpyxl",
    )

    sheet = _find_sheet_name(
        xl.sheet_names,
        source,
    )

    # Quiet by default (Streamlit); set DEVIS_VERBOSE=1 to log sheet resolution.
    if os.environ.get("DEVIS_VERBOSE"):
        print(f"[XLSX] {source.upper()} <- {sheet!r}")

    return pd.read_excel(
        xl,
        sheet_name=sheet,
        engine="openpyxl",
    )


def load_raw_ponceblanc(
    csv_path: str | None = None,
) -> pd.DataFrame:

    if csv_path is None:

        xlsx = _resolve_xlsx()

        if xlsx:
            return _read_xlsx_sheet(
                "ponceblanc",
                xlsx,
            )

    path = csv_path or _resolve_csv(
        PONCEBLANC_CSV_PATH,
        _FALLBACK_PONCE_CSV,
    )

    return _read_csv_robust(
        path,
        [
            "utf-8-sig",
            "utf-8",
            "latin1",
            "cp1252",
        ],
        [",", ";"],
    )


def load_raw_lbfi(
    csv_path: str | None = None,
) -> pd.DataFrame:

    if csv_path is None:

        xlsx = _resolve_xlsx()

        if xlsx:
            return _read_xlsx_sheet(
                "lbfi",
                xlsx,
            )

    path = csv_path or _resolve_csv(
        LBFI_CSV_PATH,
        _FALLBACK_LBFI_CSV,
    )

    return _read_csv_robust(
        path,
        [
            "latin1",
            "cp1252",
            "utf-8-sig",
            "utf-8",
        ],
        [";", ","],
    )


# ---------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------

def _parse_fr_number(
    series: pd.Series,
) -> pd.Series:

    text = (
        series
        .astype(str)
        .str.strip()
    )

    text = text.str.replace(
        r"\s*euros?\s*",
        "",
        regex=True,
        case=False,
    )

    text = text.str.replace(
        ",",
        ".",
        regex=False,
    )

    text = text.str.replace(
        r"[^0-9.\-]",
        "",
        regex=True,
    )

    return pd.to_numeric(
        text,
        errors="coerce",
    )


def _clean_taux_marge(
    series: pd.Series,
) -> pd.Series:

    parsed = _parse_fr_number(series)

    return parsed.where(
        (parsed >= 0)
        & (parsed <= 10)
    )


def _clean_text(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan})
    )


# ---------------------------------------------------------------------
# PONCEBLANC
# ---------------------------------------------------------------------

def standardize_ponceblanc(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = pd.DataFrame()

    devis_col = _find_col(
        df.columns,
        ["devis n"],
        "DEVIS N°",
    )

    delai_col = _find_col(
        df.columns,
        ["devis ouverture"],
        "Délai devis ouverture",
    )

    signe_col = _find_col(
        df.columns,
        ["sign", "2"],
        "Signé2",
    )

    out["devis_code"] = (
        df[devis_col]
        .astype(str)
        .str.strip()
    )

    out["source"] = "ponceblanc"

    out["client"] = _clean_text(
        df["Nom Client"]
    )

    out["produit"] = _clean_text(
        df["Type de produit"]
    )

    out["quantite"] = _parse_fr_number(
        df["Nb exemplaires"]
    )

    out["taux_marge"] = _clean_taux_marge(
        df["Taux Marge"]
    )

    out["prix_total"] = _parse_fr_number(
        df["Prix total"]
    )

    out["prix_unitaire"] = _parse_fr_number(
        df["Prix Unitaire"]
    )

    out["delai_jours"] = _parse_fr_number(
        df[delai_col]
    )

    # Prefer pre-computed total cost; else sum the three cost parts when present.
    if "Total coût" in df.columns:
        out["cout_total"] = _parse_fr_number(df["Total coût"])
    else:
        cout_total = None
        for part in (
            "Total coût achat",
            "Total coût fabrication",
            "Total coût transport",
        ):
            if part in df.columns:
                parsed = _parse_fr_number(df[part]).fillna(0)
                cout_total = parsed if cout_total is None else cout_total + parsed
        out["cout_total"] = cout_total if cout_total is not None else np.nan

    out["signe"] = pd.to_numeric(
        df[signe_col],
        errors="coerce",
    )

    return out


# ---------------------------------------------------------------------
# LBFI
# ---------------------------------------------------------------------

def standardize_lbfi(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    LBFI is EURO ONLY.

    IMPORTANT:
        We intentionally DO NOT find or read "Taux de marge".

    The model receives:
        cout_total = achat + fabrication + transport
        prix_total = selling price

    Both are absolute EUR amounts.
    """

    out = pd.DataFrame()

    devis_col = _find_col(
        df.columns,
        ["num", "devis"],
        "Numéro de devis",
    )

    achat_col = _find_col(
        df.columns,
        ["co", "t achat"],
        "Total coût achat",
    )

    fab_col = _find_col(
        df.columns,
        ["co", "t fabrication"],
        "Total coût fabrication",
    )

    transport_col = _find_col(
        df.columns,
        ["co", "t transport"],
        "Total coût transport",
    )

    signe_col = _find_col(
        df.columns,
        ["sign"],
        "Signé ?",
    )

    out["devis_code"] = (
        df[devis_col]
        .astype(str)
        .str.strip()
    )

    out["source"] = "lbfi"

    out["client"] = _clean_text(
        df["Nom Client"]
    )

    out["produit"] = _clean_text(
        df["Type de produit"]
    )

    out["quantite"] = _parse_fr_number(
        df["Nb exemplaires"]
    )

    # ==============================================================
    # CRITICAL:
    # LBFI margin is NOT used.
    # ==============================================================
    out["taux_marge"] = np.nan

    out["prix_total"] = _parse_fr_number(
        df["Prix total"]
    )

    out["prix_unitaire"] = (
        out["prix_total"]
        / out["quantite"].replace(0, np.nan)
    )

    out["delai_jours"] = np.nan

    cout_achat = _parse_fr_number(
        df[achat_col]
    )

    cout_fabrication = _parse_fr_number(
        df[fab_col]
    )

    cout_transport = _parse_fr_number(
        df[transport_col]
    ).fillna(0)

    out["cout_total"] = (
        cout_achat
        + cout_fabrication
        + cout_transport
    )

    raw_signe = df[signe_col]

    mapped = raw_signe.map(
        {
            "VRAI": 1,
            "FAUX": 0,
            "Vrai": 1,
            "Faux": 0,
            "vrai": 1,
            "faux": 0,
            True: 1,
            False: 0,
            1: 1,
            0: 0,
            1.0: 1,
            0.0: 0,
            "1": 1,
            "0": 0,
            "TRUE": 1,
            "FALSE": 0,
            "True": 1,
            "False": 0,
        }
    )

    mapped = mapped.fillna(
        pd.to_numeric(
            raw_signe,
            errors="coerce",
        )
    )

    out["signe"] = mapped

    return out


# ---------------------------------------------------------------------
# FINAL CLEANING
# ---------------------------------------------------------------------

def _finalize(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df[UNIFIED_COLUMNS].copy()

    df = df.dropna(
        subset=[
            "signe",
            "quantite",
        ]
    )

    df["signe"] = pd.to_numeric(
        df["signe"],
        errors="coerce",
    )

    df["quantite"] = pd.to_numeric(
        df["quantite"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "signe",
            "quantite",
        ]
    )

    df["signe"] = df["signe"].astype(int)

    df = df[
        df["quantite"] > 0
    ]

    df = df[
        df["client"].notna()
        & (
            df["client"]
            .astype(str)
            .str.len()
            > 0
        )
    ]

    df = df[
        df["produit"].notna()
        & (
            df["produit"]
            .astype(str)
            .str.len()
            > 0
        )
    ]

    # Sanity checks for euro values.
    for col, lo, hi in (
        ("cout_total", 0, 500_000),
        ("prix_total", 0, 500_000),
        ("prix_unitaire", 0, 50_000),
    ):

        bad = (
            df[col].notna()
            & ~df[col].between(lo, hi)
        )

        df.loc[bad, col] = np.nan

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# PUBLIC DATA API
# ---------------------------------------------------------------------

def build_source(
    source: str,
    csv_path: str | None = None,
) -> pd.DataFrame:

    source = source.strip().lower()

    if source not in VALID_SOURCES:
        raise ValueError(
            f"source must be one of "
            f"{VALID_SOURCES}, got {source!r}"
        )

    if source == "ponceblanc":

        raw = load_raw_ponceblanc(
            csv_path
        )

        standardized = standardize_ponceblanc(
            raw
        )

    else:

        raw = load_raw_lbfi(
            csv_path
        )

        standardized = standardize_lbfi(
            raw
        )

    return _finalize(
        standardized
    )


def build_both_separately(
    ponce_csv: str | None = None,
    lbfi_csv: str | None = None,
) -> dict[str, pd.DataFrame]:

    return {
        "ponceblanc": build_source(
            "ponceblanc",
            ponce_csv,
        ),
        "lbfi": build_source(
            "lbfi",
            lbfi_csv,
        ),
    }


# ---------------------------------------------------------------------
# ENCODINGS
# ---------------------------------------------------------------------

def fit_product_clusters(
    produit_series: pd.Series,
    min_count: int = 15,
) -> dict:

    counts = produit_series.value_counts()

    frequent = counts[
        counts >= min_count
    ].index.tolist()

    mapping = {
        product: index
        for index, product
        in enumerate(frequent)
    }

    mapping["__OTHER__"] = len(mapping)

    return mapping


def apply_product_clusters(
    produit_series: pd.Series,
    mapping: dict,
) -> pd.Series:

    other = mapping["__OTHER__"]

    return produit_series.map(
        lambda value:
            mapping.get(value, other)
    )


def fit_client_encoding(
    client_series: pd.Series,
    signe_series: pd.Series,
    smoothing: int = 10,
) -> dict:

    global_rate = float(
        signe_series.mean()
    )

    temp = pd.DataFrame(
        {
            "client": client_series,
            "signe": signe_series,
        }
    )

    grouped = (
        temp
        .groupby("client")["signe"]
    )

    counts = grouped.count()
    means = grouped.mean()

    smoothed = (
        means * counts
        + global_rate * smoothing
    ) / (
        counts + smoothing
    )

    mapping = smoothed.to_dict()

    mapping["__GLOBAL__"] = global_rate

    return mapping


def apply_client_encoding(
    client_series: pd.Series,
    mapping: dict,
) -> pd.Series:

    default = mapping["__GLOBAL__"]

    return client_series.map(
        lambda client:
            mapping.get(client, default)
    )


# ---------------------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    product_clusters: dict,
    client_encoding: dict,
    *,
    include_taux_marge: bool = False,
    include_price: bool = False,
    log_cost: bool = False,
    log_price: bool = False,
    include_cost: bool = False,
) -> pd.DataFrame:

    X = pd.DataFrame(
        index=df.index
    )

    X["quantite"] = pd.to_numeric(
        df["quantite"],
        errors="coerce",
    )

    X["produit_cluster"] = (
        apply_product_clusters(
            df["produit"],
            product_clusters,
        )
    )

    X["client_encoded"] = (
        apply_client_encoding(
            df["client"],
            client_encoding,
        )
    )

    if include_cost:

        cost = pd.to_numeric(
            df["cout_total"],
            errors="coerce",
        )

        if log_cost:
            cost = np.log1p(
                cost.clip(lower=0)
            )

        X["cout_total"] = cost

    if include_price:

        price = pd.to_numeric(
            df["prix_total"],
            errors="coerce",
        )

        if log_price:
            price = np.log1p(
                price.clip(lower=0)
            )

        X["prix_total"] = price

    # Only Ponceblanc uses this.
    if include_taux_marge:

        X["taux_marge"] = pd.to_numeric(
            df["taux_marge"],
            errors="coerce",
        )

    return X