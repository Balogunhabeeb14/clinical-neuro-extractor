"""Split a report-type ``body_analysed`` free text into its narrative sections.

Unlike assessment score sheets, neuropsychological reports have a reliable
structure: a fixed set of section headings (REASON FOR REFERRAL, BACKGROUND
INFORMATION, PRESENTATION, ASSESSMENT FINDINGS, CONCLUSIONS, ...), each on
its own line, sometimes with a trailing colon. Within ASSESSMENT FINDINGS,
sub-headings name cognitive domains (e.g. "MEMORY FUNCTIONS") - the same
domains recognized by :mod:`clinical_neuro_extractor.domains`.

This is a best-effort segmenter for a specific, observed report layout -
extend ``SECTION_ALIASES`` as new heading spellings turn up in real reports.
"""

from __future__ import annotations

import pandas as pd

from .domains import canonical_domain

# canonical section slug -> raw header spellings seen in reports (any case)
SECTION_ALIASES: dict[str, list[str]] = {
    "reason_for_referral": ["reason for referral"],
    "background_information": ["background information", "background", "history"],
    "presentation": ["presentation"],
    "assessment_findings": ["assessment findings", "results", "test results"],
    "conclusions": ["conclusions", "conclusion", "summary and conclusions"],
    "recommendations": ["recommendations"],
    "summary": ["summary"],
    "impression": ["impression"],
    "mental_state_examination": ["mental state examination"],
}

_SECTION_LOOKUP: dict[str, str] = {
    alias: slug for slug, aliases in SECTION_ALIASES.items() for alias in aliases
}


def _canonical_section(line: str) -> str | None:
    normalized = " ".join(line.strip().lower().rstrip(":").split())
    return _SECTION_LOOKUP.get(normalized)


def parse_report_sections(text: str) -> pd.DataFrame:
    """Split report free text into (section, subsection, text) rows.

    Text before the first recognized section heading is returned under the
    section ``None`` (report header/demographics block, signature block,
    etc.) so no text is silently dropped. Within ``assessment_findings``,
    text is further split into cognitive-domain subsections when domain
    headings (see :mod:`clinical_neuro_extractor.domains`) are present.
    """
    lines = [line.strip() for line in text.splitlines()]

    rows: list[dict] = []
    section: str | None = None
    subsection: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            rows.append({"section": section, "subsection": subsection, "text": content})
        buffer.clear()

    for line in lines:
        if not line:
            continue

        new_section = _canonical_section(line)
        if new_section is not None:
            flush()
            section = new_section
            subsection = None
            continue

        if section == "assessment_findings":
            new_subsection = canonical_domain(line)
            if new_subsection is not None:
                flush()
                subsection = new_subsection
                continue

        buffer.append(line)

    flush()
    return pd.DataFrame(rows, columns=["section", "subsection", "text"])


def parse_report_sections_batch(documents: pd.DataFrame) -> pd.DataFrame:
    """Run :func:`parse_report_sections` over every report document.

    Expects ``documents`` to already be filtered to (or to include)
    ``document_description == "report"`` rows with ``document_guid``,
    ``client_guid``, and ``body_analysed`` columns.
    """
    reports = documents[
        documents["document_description"].str.strip().str.lower() == "report"
    ]

    frames = []
    for _, doc in reports.iterrows():
        sections = parse_report_sections(doc["body_analysed"])
        if sections.empty:
            continue
        sections.insert(0, "client_guid", doc.get("client_guid"))
        sections.insert(0, "document_guid", doc.get("document_guid"))
        frames.append(sections)

    if not frames:
        return pd.DataFrame(
            columns=["document_guid", "client_guid", "section", "subsection", "text"]
        )
    return pd.concat(frames, ignore_index=True)
