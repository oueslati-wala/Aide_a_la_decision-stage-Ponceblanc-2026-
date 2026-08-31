"""
features.py
===========

Data preparation for Ponceblanc and LBFI (sources fully separated).

Final model feature set
-----------------------
    - client          (smoothed acceptance rate)
    - produit         (cluster)
    - quantite
    - cout_total      (achat + fabrication + transport, €)
    - season          (4-month blocks: Jan–Apr / May–Aug / Sep–Dec)
    - prix_total      (classifier only: candidate selling price)

Explicitly excluded
-------------------
    - historical delai_jours (offer → decision): does not drive client decision;
      also leaks the label on Ponceblanc
    - format, commercial, dimensions
    - manual UI heuristics

Target for the price model: coefficient = prix_total / cout_total on accepted
quotes; displayed price = coefficient × cout_total.
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
    "reference_client",
    "produit",
    "quantite",
    "taux_marge",
    "prix_total",
    "prix_unitaire",
    "delai_jours",
    "cout_total",
    "signe",
    # Extra signals when available in the source extract
    "matiere",
    "format_dims",
    "commercial",
    "month",
    "year",
    "longueur",
    "largeur",
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


def _find_col_optional(
    columns,
    must_contain: list[str],
):
    """Same matching as _find_col, but returns None instead of raising when
    no header matches -- used for fields that may not exist in every raw
    extract (e.g. a source-specific column) without breaking the pipeline."""

    for column in columns:

        low = str(column).strip().lower()

        if all(part in low for part in must_contain):
            return column

    return None


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
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )


# Singular / spelling variants → canonical product type (training + UI)
PRODUIT_ALIASES = {
    "LIASSES": "LIASSE",
    "COLLECTIONS": "COLLECTION",
    "ECHANTILLONS": "ECHANTILLONNAGE",
    "ECHANTILLON": "ECHANTILLONNAGE",
    "ECHANTILLONAGES": "ECHANTILLONNAGE",
    "ECHANTILLONNAGES": "ECHANTILLONNAGE",
    "PONCEBLANVC": "PONCEBLANC",
    "PONCE BLANC": "PONCEBLANC",
    "NUANCIERS": "NUANCIER",
    "BOITES": "BOITE",
    "VALISES": "VALISE",
    "PANNEAU": "PANNEAUX",
    "CLASSEURS": "CLASSEUR",
    "CUSTODE": "CUSTODES",
    "CALENDRIERS": "CALENDRIER",
    "NUMERIQUE": "NUMÉRIQUE",
    "PLVS": "PLV",
}


def normalize_produit(value) -> str | float:
    """Map plural / typo variants to a single canonical product label."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    key = str(value).strip().upper()
    if key in ("", "NAN", "NONE"):
        return np.nan
    return PRODUIT_ALIASES.get(key, key)


def normalize_produit_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_produit)


def _normalize_format(series: pd.Series) -> pd.Series:
    """Normalize FORMAT strings: collapse spaces/x/X variants."""
    s = series.astype(str).str.strip().str.upper()
    s = s.str.replace(r"\s*[xX×]\s*", "X", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    return s


def _excel_serial_to_month_year(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Handle Excel serial dates or already-parsed datetimes."""
    # Try datetime first
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    # Excel serial numbers (rough range 2000-2035 → ~36526–62000)
    numeric = pd.to_numeric(series, errors="coerce")
    serial_mask = numeric.notna() & (numeric > 30000) & (numeric < 70000) & dt.isna()
    if serial_mask.any():
        # Excel epoch 1899-12-30
        origin = pd.Timestamp("1899-12-30")
        dt.loc[serial_mask] = origin + pd.to_timedelta(numeric.loc[serial_mask], unit="D")
    month = dt.dt.month
    year = dt.dt.year
    return month, year


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

    ref_client_col = _find_col_optional(
        df.columns,
        ["référence", "client"],
    )
    if ref_client_col is not None:
        out["reference_client"] = (
            df[ref_client_col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
    else:
        out["reference_client"] = np.nan

    out["produit"] = normalize_produit_series(
        _clean_text(df["Type de produit"])
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

    # --- Extra signals (Ponceblanc-rich) ---
    if "Matière" in df.columns:
        out["matiere"] = _clean_text(df["Matière"])
    else:
        out["matiere"] = np.nan

    if "FORMAT" in df.columns:
        out["format_dims"] = _normalize_format(df["FORMAT"])
    else:
        out["format_dims"] = np.nan

    if "Commercial" in df.columns:
        out["commercial"] = _clean_text(df["Commercial"])
    else:
        out["commercial"] = np.nan

    # Month / year from Mois devis / Année Devis or Dates serial
    if "Mois devis" in df.columns and "Année Devis" in df.columns:
        out["month"] = pd.to_numeric(df["Mois devis"], errors="coerce")
        out["year"] = pd.to_numeric(df["Année Devis"], errors="coerce")
    elif "Dates" in df.columns:
        m, y = _excel_serial_to_month_year(df["Dates"])
        out["month"] = m
        out["year"] = y
    else:
        out["month"] = np.nan
        out["year"] = np.nan

    # No longueur/largeur on Ponceblanc extract
    out["longueur"] = np.nan
    out["largeur"] = np.nan

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
    Extra signals when present: Date devis → month/year, Longueur, Largeur.
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

    ref_client_col = _find_col_optional(
        df.columns,
        ["référence", "client"],
    )
    if ref_client_col is not None:
        out["reference_client"] = (
            df[ref_client_col]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )
    else:
        out["reference_client"] = np.nan

    out["produit"] = normalize_produit_series(
        _clean_text(df["Type de produit"])
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

    # --- Extra signals (LBFI) ---
    out["matiere"] = np.nan  # not in extract
    out["format_dims"] = np.nan
    out["commercial"] = np.nan

    if "Date devis" in df.columns:
        m, y = _excel_serial_to_month_year(df["Date devis"])
        out["month"] = m
        out["year"] = y
    else:
        out["month"] = np.nan
        out["year"] = np.nan

    if "Longueur" in df.columns:
        out["longueur"] = pd.to_numeric(df["Longueur"], errors="coerce")
    else:
        out["longueur"] = np.nan

    if "Largeur" in df.columns:
        out["largeur"] = pd.to_numeric(df["Largeur"], errors="coerce")
    else:
        out["largeur"] = np.nan

    return out


# ---------------------------------------------------------------------
# FINAL CLEANING
# ---------------------------------------------------------------------

def _finalize(
    df: pd.DataFrame,
) -> pd.DataFrame:

    # Ensure all unified columns exist
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

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

    # Clip dimensions to realistic range
    for col, lo, hi in (
        ("longueur", 0, 50_000),
        ("largeur", 0, 50_000),
        ("month", 1, 12),
        ("year", 2015, 2035),
        ("delai_jours", 0, 365),
    ):
        if col in df.columns:
            bad = df[col].notna() & ~df[col].between(lo, hi)
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


def fit_categorical_encoding(
    series: pd.Series,
    min_count: int = 10,
) -> dict:
    """
    Integer codes for categorical columns.
    Rare values → __OTHER__; missing → __MISSING__.
    """
    cleaned = series.fillna("__MISSING__").astype(str).str.strip().str.upper()
    cleaned = cleaned.replace({"": "__MISSING__", "NAN": "__MISSING__", "NONE": "__MISSING__"})
    counts = cleaned.value_counts()
    frequent = counts[counts >= min_count].index.tolist()
    # Always keep __MISSING__ as its own code if it exists
    mapping = {}
    idx = 0
    for val in frequent:
        mapping[val] = idx
        idx += 1
    if "__MISSING__" not in mapping:
        mapping["__MISSING__"] = idx
        idx += 1
    mapping["__OTHER__"] = idx
    return mapping


def apply_categorical_encoding(
    series: pd.Series,
    mapping: dict,
) -> pd.Series:
    other = mapping["__OTHER__"]
    missing = mapping.get("__MISSING__", other)

    def _map(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return missing
        key = str(v).strip().upper()
        if key in ("", "NAN", "NONE"):
            return missing
        return mapping.get(key, other)

    return series.map(_map)


def fit_target_encoding(
    series: pd.Series,
    signe_series: pd.Series,
    smoothing: int = 10,
) -> dict:
    """Smoothed acceptance rate by category (same idea as client encoding)."""
    global_rate = float(signe_series.mean())
    cleaned = series.fillna("__MISSING__").astype(str).str.strip().str.upper()
    cleaned = cleaned.replace({"": "__MISSING__", "NAN": "__MISSING__", "NONE": "__MISSING__"})
    temp = pd.DataFrame({"cat": cleaned, "signe": signe_series})
    grouped = temp.groupby("cat")["signe"]
    counts = grouped.count()
    means = grouped.mean()
    smoothed = (means * counts + global_rate * smoothing) / (counts + smoothing)
    mapping = smoothed.to_dict()
    mapping["__GLOBAL__"] = global_rate
    return mapping


def apply_target_encoding(
    series: pd.Series,
    mapping: dict,
) -> pd.Series:
    default = mapping["__GLOBAL__"]

    def _map(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return mapping.get("__MISSING__", default)
        key = str(v).strip().upper()
        if key in ("", "NAN", "NONE"):
            return mapping.get("__MISSING__", default)
        return mapping.get(key, default)

    return series.map(_map)


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
    # Kept for backward compatibility with older call sites (ignored)
    matiere_encoding: dict | None = None,
    format_encoding: dict | None = None,
    commercial_encoding: dict | None = None,
    commercial_target_encoding: dict | None = None,
) -> pd.DataFrame:
    """
    Final feature set:
        client, produit, quantite, cout_total,
        season from date (4-month blocks + month sin/cos)
        + prix_total when include_price (classifier only)

    NOT used: matiere, historical delai_jours, format, commercial, dimensions.
    """
    X = pd.DataFrame(index=df.index)

    X["quantite"] = pd.to_numeric(df["quantite"], errors="coerce")
    X["produit_cluster"] = apply_product_clusters(df["produit"], product_clusters)
    X["client_encoded"] = apply_client_encoding(df["client"], client_encoding)

    # Season: 4-month blocks (1=Jan–Apr, 2=May–Aug, 3=Sep–Dec)
    # Missing month (UI toggle off) → NaN on all calendar features so the model
    # does not silently use mid-year / S2.
    month = (
        pd.to_numeric(df["month"], errors="coerce")
        if "month" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    known = month.notna()
    month_num = month.clip(1, 12)
    X["month"] = month_num.astype(float)  # stays NaN when unknown
    X["month_sin"] = pd.Series(
        np.where(known, np.sin(2 * np.pi * month_num / 12.0), np.nan),
        index=df.index,
        dtype=float,
    )
    X["month_cos"] = pd.Series(
        np.where(known, np.cos(2 * np.pi * month_num / 12.0), np.nan),
        index=df.index,
        dtype=float,
    )
    X["month_known"] = known.astype(float)
    X["season_4m"] = pd.Series(
        np.where(known, ((month_num - 1) // 4 + 1).clip(1, 3), np.nan),
        index=df.index,
        dtype=float,
    )

    if include_cost:
        cost = pd.to_numeric(df["cout_total"], errors="coerce")
        if log_cost:
            cost = np.log1p(cost.clip(lower=0))
        X["cout_total"] = cost

    if include_price:
        price = pd.to_numeric(df["prix_total"], errors="coerce")
        if log_price:
            price = np.log1p(price.clip(lower=0))
        X["prix_total"] = price

    if include_taux_marge:
        X["taux_marge"] = pd.to_numeric(df["taux_marge"], errors="coerce")

    return X