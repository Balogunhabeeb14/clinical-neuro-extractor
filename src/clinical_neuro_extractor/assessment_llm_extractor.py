"""LLM-based extraction of structured scores from assessment ``body_analysed`` text.

Assessment documents (``document_description == "assessment"``) hold
battery/domain/test/subtest scores, but as free text pulled out of a PDF
table - labels and values often end up on separate lines in an order that
regexes can't reliably reconstruct (see the module docstring discussion in
the repo's development history). Rather than a brittle rule-based parser,
this module asks Claude to read the text and return structured rows.

Requires the ``llm`` extra (``pip install -e ".[llm]"``): ``anthropic`` and
``pydantic``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    import anthropic

DEFAULT_MODEL = "claude-opus-5"

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
    client: "anthropic.Anthropic | None" = None,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Extract structured measures from one assessment document's free text.

    Returns a dataframe with columns battery/domain/test/subtest/metric/value,
    one row per score Claude identified. Empty if none were found.
    """
    if client is None:
        import anthropic as anthropic_module

        client = anthropic_module.Anthropic()

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": body_analysed}],
        output_format=AssessmentExtraction,
    )

    measures = response.parsed_output.measures
    if not measures:
        return pd.DataFrame(
            columns=["battery", "domain", "test", "subtest", "metric", "value"]
        )
    return pd.DataFrame([m.model_dump() for m in measures])


def extract_assessment_measures_batch(
    documents: pd.DataFrame,
    *,
    client: "anthropic.Anthropic | None" = None,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Run :func:`extract_assessment_measures` over every assessment document.

    Expects ``documents`` to already be filtered to (or to include)
    ``document_description == "assessment"`` rows with ``document_guid``,
    ``client_guid``, and ``body_analysed`` columns. A document that fails to
    extract does not abort the batch - it's recorded with an ``error``
    column set instead, so no document silently disappears.
    """
    if client is None:
        import anthropic as anthropic_module

        client = anthropic_module.Anthropic()

    assessments = documents[
        documents["document_description"].str.strip().str.lower() == "assessment"
    ]

    rows = []
    for _, doc in assessments.iterrows():
        try:
            measures = extract_assessment_measures(
                doc["body_analysed"], client=client, model=model
            )
        except Exception as exc:  # noqa: BLE001 - one bad document shouldn't abort the batch
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
