import pandas as pd
import pytest

pytest.importorskip("anthropic")
pytest.importorskip("pydantic")

from clinical_neuro_extractor.assessment_llm_extractor import (
    AssessmentExtraction,
    ExtractedMeasure,
    extract_assessment_measures,
    extract_assessment_measures_batch,
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


def test_extract_assessment_measures_returns_dataframe():
    parsed = AssessmentExtraction(
        measures=[
            ExtractedMeasure(
                battery="WAIS-IV",
                domain="General Intellectual Functioning",
                test="Vocabulary",
                subtest=None,
                metric="age scaled score",
                value="12",
            ),
            ExtractedMeasure(
                battery="Trail Making Test",
                domain="Executive Functions",
                test="Trails A",
                subtest=None,
                metric="seconds",
                value="57",
            ),
        ]
    )
    client = _FakeClient(parsed)

    result = extract_assessment_measures("some assessment text", client=client)

    assert len(result) == 2
    assert list(result.columns) == [
        "battery",
        "domain",
        "test",
        "subtest",
        "metric",
        "value",
    ]
    assert result.iloc[0]["test"] == "Vocabulary"
    assert result.iloc[1]["value"] == "57"
    assert client.messages.calls[0]["output_format"] is AssessmentExtraction


def test_extract_assessment_measures_empty_result():
    client = _FakeClient(AssessmentExtraction(measures=[]))
    result = extract_assessment_measures("nothing to see here", client=client)
    assert result.empty


def test_batch_filters_to_assessment_documents_and_tags_ids():
    parsed = AssessmentExtraction(
        measures=[ExtractedMeasure(test="Vocabulary", value="12")]
    )
    client = _FakeClient(parsed)

    documents = pd.DataFrame(
        [
            {
                "document_guid": "doc-1",
                "client_guid": "pt-1",
                "document_description": "assessment",
                "body_analysed": "score sheet text",
            },
            {
                "document_guid": "doc-2",
                "client_guid": "pt-1",
                "document_description": "report",
                "body_analysed": "narrative report text",
            },
        ]
    )

    result = extract_assessment_measures_batch(documents, client=client)

    assert list(result["document_guid"]) == ["doc-1"]
    assert list(result["client_guid"]) == ["pt-1"]
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
                "document_description": "assessment",
                "body_analysed": "score sheet text",
            }
        ]
    )

    result = extract_assessment_measures_batch(documents, client=_RaisingClient())

    assert len(result) == 1
    assert result.iloc[0]["error"] == "boom"
