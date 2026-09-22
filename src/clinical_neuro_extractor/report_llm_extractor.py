"""LLM-based extraction of clinical detail from report ``body_analysed`` text.

:mod:`clinical_neuro_extractor.report_sections` splits a report into its
narrative sections (REASON FOR REFERRAL, BACKGROUND INFORMATION, ...), but
the clinically useful facts within those sections - diagnosis, laterality,
treatment type, whether the assessment was pre- or post-treatment,
demographic context (occupation, handedness, education, ...), and each
cognitive domain's overall rating - are stated in free prose, not a fixed
format. This module asks Claude to read the report and pull those out as
structured fields, the same approach used for assessment score sheets in
:mod:`clinical_neuro_extractor.assessment_llm_extractor`.

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
You extract clinical details from a neuropsychological report's free text.

Only extract what the report actually states - if a field is not \
mentioned, leave it null rather than guessing or inferring. Reports vary \
widely: many say nothing about treatment at all (e.g. a pre-surgical \
baseline assessment), so a null treatment_type/treatment_timing is a \
normal, correct result, not a failure.

Fields to extract:

- diagnosis: the patient's primary diagnosis or presenting condition as \
  stated (e.g. "temporal lobe epilepsy", "left medial temporal lobe \
  tumour, likely DNET"). If several are mentioned, summarize concisely; \
  keep the clinical wording used in the text.
- diagnosis_laterality: "left", "right", "bilateral", or null if not \
  stated or not applicable.
- treatment_type: the type of treatment referenced (e.g. "surgery", \
  "radiotherapy", "chemotherapy", "medication", "none" if the text \
  explicitly says no treatment yet, or null if treatment is not \
  discussed at all).
- treatment_timing: whether this assessment was conducted "pre-treatment" \
  or "post-treatment" relative to the treatment_type, based on how the \
  report frames it (e.g. "for pre-surgical assessment" => pre-treatment; \
  "following his resection" => post-treatment). Null if not determinable.
- treatment_details: any further free-text detail about the treatment \
  (procedure name, date, laterality of surgery, medication name, etc.), \
  or null.
- reason_for_referral_summary: a one-sentence summary of why the patient \
  was referred, drawn from the REASON FOR REFERRAL section if present.
- occupation: the patient's stated occupation/job, or null.
- handedness: "left", "right", "ambidextrous", or null if not stated.
- education: any stated educational background (degree, school, level), \
  or null.
- marital_status: if stated, else null.
- living_situation: who the patient lives with / living arrangement, if \
  stated, else null.
- other_demographic_notes: any other demographic/background detail worth \
  keeping that doesn't fit the fields above, or null.
- domain_summaries: for each cognitive domain the report gives an overall \
  rating for (e.g. Intellectual Functioning, Memory, Executive Functions, \
  Language, Attention, Visuospatial - use whatever domain headings or \
  clear topic the report actually uses), extract:
  - domain: the domain name as the report frames it.
  - overall_level: the overall descriptive level given (e.g. "superior", \
    "high average", "average", "low average", "borderline", "impaired"), \
    or null if a level isn't clearly stated.
  - notes: a brief supporting phrase from the text, or null.
  Only include a domain_summaries entry for domains the report actually \
  discusses - do not invent entries for domains it doesn't mention.
"""


class DomainSummary(BaseModel):
    domain: Optional[str] = None
    overall_level: Optional[str] = None
    notes: Optional[str] = None


class ReportExtraction(BaseModel):
    diagnosis: Optional[str] = None
    diagnosis_laterality: Optional[str] = None
    treatment_type: Optional[str] = None
    treatment_timing: Optional[str] = None
    treatment_details: Optional[str] = None
    reason_for_referral_summary: Optional[str] = None
    occupation: Optional[str] = None
    handedness: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = None
    living_situation: Optional[str] = None
    other_demographic_notes: Optional[str] = None
    domain_summaries: list[DomainSummary] = []


_SCALAR_FIELDS = [
    "diagnosis",
    "diagnosis_laterality",
    "treatment_type",
    "treatment_timing",
    "treatment_details",
    "reason_for_referral_summary",
    "occupation",
    "handedness",
    "education",
    "marital_status",
    "living_situation",
    "other_demographic_notes",
]


def extract_report_details(
    body_analysed: str,
    *,
    client: "anthropic.Anthropic | None" = None,
    model: str = DEFAULT_MODEL,
) -> ReportExtraction:
    """Extract clinical/demographic detail from one report's free text."""
    if client is None:
        import anthropic as anthropic_module

        client = anthropic_module.Anthropic()

    response = client.messages.parse(
        model=model,
        max_tokens=16000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": body_analysed}],
        output_format=ReportExtraction,
    )
    return response.parsed_output


def extract_report_details_batch(
    documents: pd.DataFrame,
    *,
    client: "anthropic.Anthropic | None" = None,
    model: str = DEFAULT_MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run :func:`extract_report_details` over every report document.

    Expects ``documents`` to already be filtered to (or to include)
    ``document_description == "report"`` rows with ``document_guid``,
    ``client_guid``, and ``body_analysed`` columns.

    Returns two dataframes:

    - ``details``: one row per document with the scalar fields
      (diagnosis, treatment_type, treatment_timing, occupation, ...) plus
      an ``error`` column (set instead of the fields when extraction
      failed for that document - it is never silently dropped).
    - ``domain_summaries``: long format, one row per (document, domain)
      with ``overall_level``/``notes``.
    """
    if client is None:
        import anthropic as anthropic_module

        client = anthropic_module.Anthropic()

    reports = documents[
        documents["document_description"].str.strip().str.lower() == "report"
    ]

    detail_rows = []
    domain_rows = []
    for _, doc in reports.iterrows():
        document_guid = doc.get("document_guid")
        client_guid = doc.get("client_guid")
        try:
            extraction = extract_report_details(
                doc["body_analysed"], client=client, model=model
            )
        except Exception as exc:  # noqa: BLE001 - one bad document shouldn't abort the batch
            detail_rows.append(
                {
                    "document_guid": document_guid,
                    "client_guid": client_guid,
                    "error": str(exc),
                }
            )
            continue

        row = {"document_guid": document_guid, "client_guid": client_guid}
        row.update({field: getattr(extraction, field) for field in _SCALAR_FIELDS})
        row["error"] = None
        detail_rows.append(row)

        for summary in extraction.domain_summaries:
            domain_rows.append(
                {
                    "document_guid": document_guid,
                    "client_guid": client_guid,
                    **summary.model_dump(),
                }
            )

    details_columns = ["document_guid", "client_guid", *_SCALAR_FIELDS, "error"]
    details = pd.DataFrame(detail_rows, columns=details_columns)

    domain_columns = ["document_guid", "client_guid", "domain", "overall_level", "notes"]
    domain_summaries = pd.DataFrame(domain_rows, columns=domain_columns)

    return details, domain_summaries
