"""Build a one-row-per-patient registry from a document/visit-level dataframe.

The source dataframe has one row per clinical document (a note tied to a
visit), so the same patient (``client_guid``) appears many times. This
module collapses that into a patient registry: one row per patient with
their most current demographics plus a roll-up of their documents/visits.
"""

from __future__ import annotations

import hashlib

import pandas as pd

DOCUMENT_FIELDS = [
    "document_guid",
    "client_guid",
    "client_idcode",
    "client_touchedwhen",
    "clientvisit_admitdtm",
    "client_displayname",
    "client_firstname",
    "client_lastname",
    "client_universalnumber",
    "client_dob",
    "client_gendercode",
    "client_racecode",
    "client_title",
    "client_applicsource",
    "document_description",
    "body_analysed",
    "updatetime",
    "clientvisit_visitidcode",
    "client_createdwhen",
    "clientvisit_typecode",
    "clientvisit_visitstatus",
    "clientvisit_internalvisitstatus",
    "clientvisit_carelevelcode",
    "clientvisit_serviceguid",
    "clientvisit_touchedby",
    "clientvisit_touchedwhen",
    "clientvisit_providerdisplayname_analysed",
    "clientaddress_city",
    "clientaddress_postalcode",
    "client_languagecode",
    "client_religioncode",
]

# Fields that describe the patient rather than a specific document/visit.
# The most recently touched row's values are used for these.
_DEMOGRAPHIC_FIELDS = [
    "client_idcode",
    "client_displayname",
    "client_firstname",
    "client_lastname",
    "client_universalnumber",
    "client_dob",
    "client_gendercode",
    "client_racecode",
    "client_title",
    "client_applicsource",
    "clientaddress_city",
    "clientaddress_postalcode",
    "client_languagecode",
    "client_religioncode",
]

# Direct identifiers dropped (or transformed) when de-identifying.
_DIRECT_IDENTIFIER_FIELDS = [
    "client_idcode",
    "client_displayname",
    "client_firstname",
    "client_lastname",
    "client_universalnumber",
    "client_dob",
]

_DATETIME_FIELDS = [
    "client_touchedwhen",
    "clientvisit_admitdtm",
    "updatetime",
    "client_createdwhen",
    "clientvisit_touchedwhen",
]


def build_patient_registry(
    documents: pd.DataFrame,
    *,
    deidentify: bool = False,
    salt: str = "",
) -> pd.DataFrame:
    """Collapse a document-level dataframe into a one-row-per-patient registry.

    Args:
        documents: dataframe with (at least) a ``client_guid`` column, one
            row per document/visit, using the field names in ``DOCUMENT_FIELDS``.
        deidentify: if True, replace ``client_guid`` with a salted hash and
            drop/generalize direct identifiers (name, DOB, ID codes).
        salt: salt mixed into the hash used when ``deidentify=True``. Callers
            doing this for real should supply a private, per-project salt
            rather than the default empty string.

    Returns:
        A dataframe with one row per patient: demographics as of their most
        recently touched record, plus document/visit roll-up columns
        (``n_documents``, ``n_visits``, ``first_visit_dtm``, ``last_visit_dtm``,
        ``visit_types``, ``care_levels``, ``providers``).
    """
    if "client_guid" not in documents.columns:
        raise KeyError("documents dataframe must include a 'client_guid' column")

    df = documents.copy()
    for col in _DATETIME_FIELDS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    recency_col = next(
        (c for c in ("client_touchedwhen", "updatetime") if c in df.columns), None
    )
    latest_rows = df.sort_values(recency_col, ascending=False) if recency_col else df
    latest_rows = latest_rows.drop_duplicates(subset="client_guid", keep="first")

    demo_cols = [c for c in _DEMOGRAPHIC_FIELDS if c in df.columns]
    registry = latest_rows.set_index("client_guid")[demo_cols].copy()

    grouped = df.groupby("client_guid")

    if "document_guid" in df.columns:
        registry["n_documents"] = grouped["document_guid"].nunique()
    else:
        registry["n_documents"] = grouped.size()

    if "clientvisit_visitidcode" in df.columns:
        registry["n_visits"] = grouped["clientvisit_visitidcode"].nunique()

    if "clientvisit_admitdtm" in df.columns:
        registry["first_visit_dtm"] = grouped["clientvisit_admitdtm"].min()
        registry["last_visit_dtm"] = grouped["clientvisit_admitdtm"].max()

    if "client_createdwhen" in df.columns:
        registry["first_document_dtm"] = grouped["client_createdwhen"].min()

    if recency_col:
        registry["last_touched_dtm"] = grouped[recency_col].max()

    for src_col, out_col in (
        ("clientvisit_typecode", "visit_types"),
        ("clientvisit_carelevelcode", "care_levels"),
        ("clientvisit_providerdisplayname_analysed", "providers"),
    ):
        if src_col in df.columns:
            registry[out_col] = grouped[src_col].apply(_sorted_unique)

    registry = registry.reset_index()

    if deidentify:
        registry = _deidentify(registry, salt=salt)

    return registry


def _sorted_unique(series: pd.Series) -> list:
    return sorted({value for value in series.dropna()})


def _deidentify(registry: pd.DataFrame, *, salt: str) -> pd.DataFrame:
    """Replace direct identifiers with a hashed ID and generalized fields."""
    registry = registry.copy()

    registry.insert(0, "patient_id", registry["client_guid"].map(lambda g: _hash_id(g, salt)))
    registry = registry.drop(columns=["client_guid"])

    if "client_dob" in registry.columns:
        registry["birth_year"] = pd.to_datetime(registry["client_dob"], errors="coerce").dt.year

    if "clientaddress_postalcode" in registry.columns:
        registry["clientaddress_postalcode"] = (
            registry["clientaddress_postalcode"].astype("string").str.slice(0, 3)
        )

    drop_cols = [c for c in _DIRECT_IDENTIFIER_FIELDS if c in registry.columns]
    return registry.drop(columns=drop_cols)


def _hash_id(client_guid: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}{client_guid}".encode("utf-8")).hexdigest()
    return digest[:16]
