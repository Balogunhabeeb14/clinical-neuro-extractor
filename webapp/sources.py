"""Resolve the documents dataframe the registry is built from.

Two sources, checked in priority order:

1. An uploaded CSV - an explicit user action for one request, always wins.
2. The most recent batch CSV written by the existing CogStack ingestion
   pipeline (``ingest.py``) under ``REGISTRY_RAW_DATA_DIR``.
3. A live CogStack query, reached only when no batch CSV is available or a
   refresh is explicitly requested.

The live query deliberately does not re-implement CogStack authentication -
it imports the same ``cs_core_v1`` module (and its module-level ``cs``
cohort-searcher client) that ``ingest.py`` already uses, so credentials and
connection setup stay in the one place that already works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from clinical_neuro_extractor.registry import DOCUMENT_FIELDS

RAW_DATA_DIR = os.environ.get("REGISTRY_RAW_DATA_DIR")
COGSTACK_UTIL_PATH = os.environ.get("COGSTACK_UTIL_PATH")
COGSTACK_INDEX = os.environ.get("COGSTACK_INDEX", "epr_documents")
COGSTACK_SEARCH_STRING = os.environ.get(
    "COGSTACK_SEARCH_STRING",
    'document_description:("Neuropsychology" AND ("report" OR "scores" OR "assessment"))',
)


class CogStackUnavailable(RuntimeError):
    """The CogStack cohort-searcher client couldn't be imported, or the query failed."""


def _align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every DOCUMENT_FIELDS column is present, in order; extras kept after."""
    df = df.copy()
    for col in DOCUMENT_FIELDS:
        if col not in df.columns:
            df[col] = pd.NA
    ordered = DOCUMENT_FIELDS + [c for c in df.columns if c not in DOCUMENT_FIELDS]
    return df[ordered]


def load_csv(path_or_file) -> pd.DataFrame:
    """Load a documents dataframe from a CSV path or a file-like object (e.g. an upload)."""
    df = pd.read_csv(path_or_file, dtype=str, keep_default_na=True)
    return _align_columns(df)


def find_latest_batch_csv(raw_data_dir: Optional[str] = None) -> Optional[Path]:
    """Return the most recent ``df_*.csv`` written by ingest.py's batch job, if any."""
    directory = raw_data_dir or RAW_DATA_DIR
    if not directory:
        return None
    candidates = sorted(Path(directory).glob("df_*.csv"))
    return candidates[-1] if candidates else None


def fetch_from_cogstack(
    *,
    index_name: str = COGSTACK_INDEX,
    search_string: str = COGSTACK_SEARCH_STRING,
    fields_list: Optional[list] = None,
    util_path: Optional[str] = None,
) -> pd.DataFrame:
    """Run a live CogStack query via the existing cs_core_v1 cohort-searcher client."""
    path = util_path or COGSTACK_UTIL_PATH
    if path and path not in sys.path:
        sys.path.insert(0, path)

    try:
        from cs_core_v1 import cs
    except ImportError as exc:
        raise CogStackUnavailable(
            "Could not import cs_core_v1 (the CogStack cohort-searcher client). "
            "Set COGSTACK_UTIL_PATH to the directory containing it."
        ) from exc

    try:
        docs = cs.cohort_searcher_no_terms(
            index_name=index_name,
            fields_list=fields_list or DOCUMENT_FIELDS,
            search_string=search_string,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as CogStackUnavailable, not a 500
        raise CogStackUnavailable(f"CogStack query failed: {exc}") from exc

    return _align_columns(pd.DataFrame(docs))


def resolve_documents(
    *,
    uploaded_file=None,
    raw_data_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> tuple:
    """Resolve the documents dataframe from the highest-priority available source.

    Returns ``(dataframe, source_label, warning_message_or_None)``.
    """
    if uploaded_file is not None:
        return load_csv(uploaded_file), "uploaded CSV", None

    latest_csv = None if force_refresh else find_latest_batch_csv(raw_data_dir)
    if latest_csv is not None:
        return load_csv(latest_csv), f"batch CSV ({latest_csv.name})", None

    try:
        return fetch_from_cogstack(), "live CogStack query", None
    except CogStackUnavailable as exc:
        fallback_csv = find_latest_batch_csv(raw_data_dir)
        if fallback_csv is not None:
            return (
                load_csv(fallback_csv),
                f"batch CSV ({fallback_csv.name})",
                f"CogStack refresh failed ({exc}); showing the last batch CSV instead.",
            )
        raise
