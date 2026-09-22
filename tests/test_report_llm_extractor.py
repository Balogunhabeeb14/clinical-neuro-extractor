import pandas as pd
import pytest

pytest.importorskip("anthropic")
pytest.importorskip("pydantic")

from clinical_neuro_extractor.report_llm_extractor import (
    DomainSummary,
    ReportExtraction,
    extract_report_details,
    extract_report_details_batch,
)


class _FakeResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class _FakeMessages:
    def __init__(self, parsed_output):
        self._parsed_output = parsed_output
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._parsed_output)


class _FakeClient:
    def __init__(self, parsed_output):
        self.messages = _FakeMessages(parsed_output)


def _sample_extraction() -> ReportExtraction:
    return ReportExtraction(
        diagnosis="left medial temporal lobe tumour, likely DNET",
        diagnosis_laterality="left",
        treatment_type=None,
        treatment_timing=None,
        treatment_details=None,
        reason_for_referral_summary="Referred for neuropsychological assessment given temporal lobe epilepsy and a left temporal tumour.",
        occupation="copy writer",
        handedness="left",
        education="English degree, Leeds",
        marital_status=None,
        living_situation=None,
        other_demographic_notes=None,
        domain_summaries=[
            DomainSummary(domain="Intellectual Functioning", overall_level="superior", notes="FSIQ = 125"),
            DomainSummary(domain="Memory", overall_level="high average", notes="verbal memory high average"),
        ],
    )


def test_extract_report_details_returns_pydantic_object():
    client = _FakeClient(_sample_extraction())
    result = extract_report_details("some report text", client=client)

    assert result.diagnosis_laterality == "left"
    assert result.occupation == "copy writer"
    assert len(result.domain_summaries) == 2
    assert client.messages.calls[0]["output_format"] is ReportExtraction


def test_batch_filters_to_report_documents_and_splits_domain_summaries():
    client = _FakeClient(_sample_extraction())

    documents = pd.DataFrame(
        [
            {
                "document_guid": "doc-1",
                "client_guid": "pt-1",
                "document_description": "report",
                "body_analysed": "narrative report text",
            },
            {
                "document_guid": "doc-2",
                "client_guid": "pt-1",
                "document_description": "assessment",
                "body_analysed": "score sheet text",
            },
        ]
    )

    details, domain_summaries = extract_report_details_batch(documents, client=client)

    assert list(details["document_guid"]) == ["doc-1"]
    assert details.iloc[0]["diagnosis_laterality"] == "left"
    assert details.iloc[0]["occupation"] == "copy writer"
    assert details.iloc[0]["error"] is None

    assert list(domain_summaries["document_guid"]) == ["doc-1", "doc-1"]
    assert set(domain_summaries["domain"]) == {"Intellectual Functioning", "Memory"}
    assert len(client.messages.calls) == 1


def test_batch_records_errors_without_aborting():
    class _RaisingMessages:
        def parse(self, **kwargs):
            raise RuntimeError("boom")

    class _RaisingClient:
        def __init__(self):
            self.messages = _RaisingMessages()

    documents = pd.DataFrame(
        [
            {
                "document_guid": "doc-1",
                "client_guid": "pt-1",
                "document_description": "report",
                "body_analysed": "narrative report text",
            }
        ]
    )

    details, domain_summaries = extract_report_details_batch(documents, client=_RaisingClient())

    assert len(details) == 1
    assert details.iloc[0]["error"] == "boom"
    assert domain_summaries.empty
