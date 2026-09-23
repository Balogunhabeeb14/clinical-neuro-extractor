"""LLM-based extraction of structured scores from assessment ``body_analysed`` text.

Assessment documents (``document_description == "assessment"``) hold
battery/domain/test/subtest scores, but as free text pulled out of a PDF
table - labels and values often end up on separate lines in an order that
regexes can't reliably reconstruct (see the module docstring discussion in
the repo's development history). Rather than a brittle rule-based parser,
this module asks a locally-hosted Ollama model to read the text and return
structured rows - no clinical text is sent to a cloud API.

Requires the ``llm`` extra (``pip install -e ".[llm]"``): ``ollama`` and
``pydantic``, plus a running Ollama server with the model pulled
(``ollama pull llama3.1``).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel

from .ollama_client import DEFAULT_MODEL, OllamaUnavailable, extract_structured

__all__ = [
    "DEFAULT_MODEL",
    "OllamaUnavailable",
    "ExtractedMeasure",
    "AssessmentExtraction",
    "extract_assessment_measures",
    "extract_assessment_measures_batch",
]

_SYSTEM_PROMPT = """\
You extract structured neuropsychological test scores from clinical
assessment text. The text was pulled out of a PDF score sheet, so labels \
and their values are often on separate lines and the reading order can be \
irregular - use your judgement about which value(s) belong to which label.

For every individual score you can identify, extract:
- battery: the named test battery/instrument (e.g. "WAIS-IV", "WMS-IV", \
  "TOPF", "Trail Making Test"), if identifiable.
- domain: the cognitive domain the score falls under (e.g. "General \
  Intellectual Functioning", "Memory", "Executive Functions", "Language"), \
  based on the section the score appeared in.
- test: the specific test/subtest/sub-scale name (e.g. "Vocabulary", \
  "Trails A", "Logical Memory-I", "Full scale IQ").
- subtest: a finer-grained subdivision of `test` if the text has one \
  (e.g. a subscale within a subtest); otherwise null.
- metric: what kind of number this is (e.g. "raw score", "age scaled \
  score", "index score", "percentile", "seconds", "IQ"), based on column \
  headers or labels near the value.
- value: the value exactly as written (keep as a string - it may be a \
  number, a range like "50th-75th percentile", or a qualitative rating \
  like "impaired" or "Superior").

Only extract scores you can actually find in the text - do not invent or \
infer values that are not present. If a field cannot be determined, leave \
it null rather than guessing. Extract every distinct score, even if \
several appear for the same test (e.g. raw score and scaled score are two \
separate rows unless they are more naturally one row with two metrics -
use your judgement, but prefer one row per number when in doubt).
"""


class ExtractedMeasure(BaseModel):
    battery: Optional[str] = None
    domain: Optional[str] = None
    test: Optional[str] = None
    subtest: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[str] = None


class AssessmentExtraction(BaseModel):
    measures: list[ExtractedMeasure]


def extract_assessment_measures(
    body_analysed: str,
    *,
    client=None,
    model: str = DEFAULT_MODEL,
    host: Optional[str] = None,
) -> pd.DataFrame:
    """Extract structured measures from one assessment document's free text.

    ``client`` is an ``ollama.Client`` (or a test double exposing the same
    ``.chat(...)`` shape); when omitted one is built pointed at ``host`` (or
    the ``OLLAMA_HOST`` env var, or the local default). Raises
    :class:`~clinical_neuro_extractor.ollama_client.OllamaUnavailable` if the
    server can't be reached or its response doesn't match the schema.

    Returns a dataframe with columns battery/domain/test/subtest/metric/value,
    one row per score the model identified. Empty if none were found.
    """
    extraction = extract_structured(
        system=_SYSTEM_PROMPT,
        user=body_analysed,
        schema=AssessmentExtraction,
        client=client,
        model=model,
        host=host,
    )

    measures = extraction.measures
    if not measures:
        return pd.DataFrame(
            columns=["battery", "domain", "test", "subtest", "metric", "value"]
        )
    return pd.DataFrame([m.model_dump() for m in measures])


def extract_assessment_measures_batch(
    documents: pd.DataFrame,
    *,
    client=None,
    model: str = DEFAULT_MODEL,
    host: Optional[str] = None,
) -> pd.DataFrame:
    """Run :func:`extract_assessment_measures` over every assessment document.

    Expects ``documents`` to already be filtered to (or to include)
    ``document_description == "assessment"`` rows with ``document_guid``,
    ``client_guid``, and ``body_analysed`` columns. A document that fails to
    extract does not abort the batch - it's recorded with an ``error``
    column set instead, so no document silently disappears.
    """
    assessments = documents[
        documents["document_description"].str.strip().str.lower() == "assessment"
    ]

    rows = []
    for _, doc in assessments.iterrows():
        try:
            measures = extract_assessment_measures(
                doc["body_analysed"], client=client, model=model, host=host
            )
        except Exception as exc:  # noqa: BLE001 - one bad document (incl. OllamaUnavailable) shouldn't abort the batch
            rows.append(
                {
                    "document_guid": doc.get("document_guid"),
                    "client_guid": doc.get("client_guid"),
                    "error": str(exc),
                }
            )
            continue

        if measures.empty:
            continue
        measures.insert(0, "client_guid", doc.get("client_guid"))
        measures.insert(0, "document_guid", doc.get("document_guid"))
        rows.append(measures)

    if not rows:
        return pd.DataFrame(
            columns=[
                "document_guid",
                "client_guid",
                "battery",
                "domain",
                "test",
                "subtest",
                "metric",
                "value",
                "error",
            ]
        )

    frames = [r if isinstance(r, pd.DataFrame) else pd.DataFrame([r]) for r in rows]
    return pd.concat(frames, ignore_index=True)
