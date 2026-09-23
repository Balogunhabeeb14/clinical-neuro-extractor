import pandas as pd

from clinical_neuro_extractor.report_sections import (
    parse_report_sections,
    parse_report_sections_batch,
)

SAMPLE_REPORT = """Department of Neuropsychology

Neuropsychological Report

PRIVATE AND CONFIDENTIAL

NEUROPSYCHOLOGICAL REPORT

NAME: XXXXXXXXX
HOSPITAL NO: F4XXXXXX
DOB: 23XXXXX
WARD: IP
DATE SEEN: 28XXXXX
PSYCHOLOGY NO:
EXXXXX
REFERRED BY:
Dr.XXXXXX
Consultant Neurologist
XXXXXX Hospital

REASON FOR REFERRAL:
He was referred for a neuropsychological assessment, having temporal lobe epilepsy, with neuroimaging showing a left medial temporal lobe tumour, a likely DNET.

BACKGROUND INFORMATION:
He is a copy writer, working in advertising, having done an English degree in Leeds.
He is left handed for writing and playing the guitar. He uses his right hand for computing mouse work.

PRESENTATION:
He was helpful and generally well focused, although some occasions where there was a loss of concentration.
ASSESSMENT FINDINGS:

INTELLECTUAL FUNCTIONS
His full scale IQ was well within the superior range (FSIQ = 125).

MEMORY FUNCTIONS
Overall, his verbal memory is in the high average range, consistent with his verbal intellectual functioning.

CONCLUSIONS:
He has superior range non-verbal abilities, with verbal abilities in the high average range.

XXXXXX XXXXXXXX
Head of Clinical Neuropsychology Department
Regional Neurosciences Centre
Neuropsychology Department
PRIVATE AND CONFIDENTIAL
XXXXXXXXX FXXXXXX                                                                     Page 2 of 2
"""


def test_recognizes_top_level_sections():
    sections = parse_report_sections(SAMPLE_REPORT)
    found = set(sections["section"].dropna())
    assert {
        "reason_for_referral",
        "background_information",
        "presentation",
        "assessment_findings",
        "conclusions",
    } <= found


def test_reason_for_referral_text():
    sections = parse_report_sections(SAMPLE_REPORT)
    row = sections[sections["section"] == "reason_for_referral"].iloc[0]
    assert "temporal lobe epilepsy" in row["text"]


def test_assessment_findings_split_into_domains():
    sections = parse_report_sections(SAMPLE_REPORT)
    findings = sections[sections["section"] == "assessment_findings"]

    intellectual = findings[findings["subsection"] == "intellectual_functioning"]
    assert len(intellectual) == 1
    assert "superior range" in intellectual.iloc[0]["text"]

    memory = findings[findings["subsection"] == "memory"]
    assert len(memory) == 1
    assert "verbal memory" in memory.iloc[0]["text"]


def test_header_text_before_first_section_is_kept():
    sections = parse_report_sections(SAMPLE_REPORT)
    preamble = sections[sections["section"].isna()]
    assert not preamble.empty
    assert any("NEUROPSYCHOLOGICAL REPORT" in t for t in preamble["text"])


def test_parse_report_sections_batch_filters_and_tags_documents():
    documents = pd.DataFrame(
        [
            {
                "document_guid": "doc-1",
                "client_guid": "pt-1",
                "document_description": "report",
                "body_analysed": SAMPLE_REPORT,
            },
            {
                "document_guid": "doc-2",
                "client_guid": "pt-1",
                "document_description": "assessment",
                "body_analysed": "some score sheet text",
            },
        ]
    )

    result = parse_report_sections_batch(documents)
    assert set(result["document_guid"]) == {"doc-1"}
    assert set(result["client_guid"]) == {"pt-1"}
    assert "conclusions" in set(result["section"])
