import json

import pandas as pd
import pytest

flask = pytest.importorskip("flask")  # webapp.sources dependency
pytest.importorskip("pydantic")

from automation import extract_on_new_data as mod
from clinical_neuro_extractor.assessment_llm_extractor import AssessmentExtraction, ExtractedMeasure
from clinical_neuro_extractor.report_llm_extractor import ReportExtraction

CSV_TEXT = """document_guid,client_guid,document_description,body_analysed
doc-1,pt-1,report,hello
doc-2,pt-1,assessment,world
doc-3,pt-2,report,another
"""


class _FakeClient:
    """Branches on the requested schema's title so one fake serves both extractors."""

    def __init__(self):
        self.calls = []
        self._assessment = AssessmentExtraction(
            measures=[ExtractedMeasure(battery="b", domain="d", test="t", metric="m", value="1")]
        ).model_dump_json()
        self._report = ReportExtraction(diagnosis="some diagnosis").model_dump_json()

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        title = kwargs["format"]["title"]
        if title == "AssessmentExtraction":
            return {"message": {"content": self._assessment}}
        if title == "ReportExtraction":
            return {"message": {"content": self._report}}
        raise AssertionError(f"unexpected schema {title}")


# ---------------------------------------------------------------- find_new_csvs / mark_processed

def test_find_new_csvs_returns_all_when_no_state(tmp_path):
    (tmp_path / "df_20240101_000000.csv").write_text(CSV_TEXT)
    (tmp_path / "df_20240201_000000.csv").write_text(CSV_TEXT)

    found = mod.find_new_csvs(tmp_path, tmp_path / "state")

    assert [p.name for p in found] == ["df_20240101_000000.csv", "df_20240201_000000.csv"]


def test_mark_processed_excludes_file_from_later_finds(tmp_path):
    a = tmp_path / "df_20240101_000000.csv"
    b = tmp_path / "df_20240201_000000.csv"
    a.write_text(CSV_TEXT)
    b.write_text(CSV_TEXT)
    state_file = tmp_path / "state"

    mod.mark_processed(a, state_file)
    found = mod.find_new_csvs(tmp_path, state_file)

    assert [p.name for p in found] == ["df_20240201_000000.csv"]


# ---------------------------------------------------------------- process_csv

def test_process_csv_writes_expected_outputs(tmp_path):
    csv_path = tmp_path / "df_20240101_000000.csv"
    csv_path.write_text(CSV_TEXT)
    output_dir = tmp_path / "out"
    client = _FakeClient()

    summary = mod.process_csv(csv_path, output_dir=output_dir, model="llama3.1", host=None, client=client)

    batch_dir = output_dir / "df_20240101_000000"
    assert (batch_dir / "registry.csv").exists()
    assert (batch_dir / "assessment_measures.csv").exists()
    assert (batch_dir / "report_details.csv").exists()
    assert (batch_dir / "report_domain_summaries.csv").exists()

    assert summary["n_documents"] == 3
    assert summary["n_patients"] == 2
    # one assessment doc, one call each for the two report docs
    assert len(client.calls) == 3

    registry = pd.read_csv(batch_dir / "registry.csv")
    assert set(registry["client_guid"]) == {"pt-1", "pt-2"}


# ---------------------------------------------------------------- run_once orchestration

def test_run_once_starts_waits_and_stops_ollama_around_processing(tmp_path, monkeypatch):
    csv_path = tmp_path / "raw" / "df_20240101_000000.csv"
    csv_path.parent.mkdir()
    csv_path.write_text(CSV_TEXT)

    events = []
    monkeypatch.setattr(mod, "start_ollama", lambda compose_file: events.append("start"))
    monkeypatch.setattr(mod, "wait_for_ollama", lambda host, **kw: events.append("wait"))
    monkeypatch.setattr(mod, "stop_ollama", lambda compose_file: events.append("stop"))

    client = _FakeClient()
    summaries = mod.run_once(
        raw_data_dir=csv_path.parent,
        state_file=tmp_path / "state",
        output_dir=tmp_path / "out",
        client=client,
    )

    assert events == ["start", "wait", "stop"]
    assert len(summaries) == 1
    assert (tmp_path / "state").read_text().strip() == "df_20240101_000000.csv"


def test_run_once_skips_docker_lifecycle_when_no_new_files(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    calls = []
    monkeypatch.setattr(mod, "start_ollama", lambda compose_file: calls.append("start"))
    monkeypatch.setattr(mod, "stop_ollama", lambda compose_file: calls.append("stop"))

    summaries = mod.run_once(raw_data_dir=raw_dir, state_file=tmp_path / "state", output_dir=tmp_path / "out")

    assert summaries == []
    assert calls == []


def test_run_once_no_docker_flag_never_touches_docker(tmp_path, monkeypatch):
    csv_path = tmp_path / "raw" / "df_20240101_000000.csv"
    csv_path.parent.mkdir()
    csv_path.write_text(CSV_TEXT)

    def _boom(*a, **k):
        raise AssertionError("docker should not be invoked when use_docker=False")

    monkeypatch.setattr(mod, "start_ollama", _boom)
    monkeypatch.setattr(mod, "wait_for_ollama", _boom)
    monkeypatch.setattr(mod, "stop_ollama", _boom)

    summaries = mod.run_once(
        raw_data_dir=csv_path.parent,
        state_file=tmp_path / "state",
        output_dir=tmp_path / "out",
        use_docker=False,
        client=_FakeClient(),
    )

    assert len(summaries) == 1


def test_run_once_stops_ollama_even_if_wait_times_out(tmp_path, monkeypatch):
    csv_path = tmp_path / "raw" / "df_20240101_000000.csv"
    csv_path.parent.mkdir()
    csv_path.write_text(CSV_TEXT)

    events = []
    monkeypatch.setattr(mod, "start_ollama", lambda compose_file: events.append("start"))

    def _timeout(host, **kw):
        events.append("wait-fail")
        raise TimeoutError("never came up")

    monkeypatch.setattr(mod, "wait_for_ollama", _timeout)
    monkeypatch.setattr(mod, "stop_ollama", lambda compose_file: events.append("stop"))

    with pytest.raises(TimeoutError):
        mod.run_once(raw_data_dir=csv_path.parent, state_file=tmp_path / "state", output_dir=tmp_path / "out")

    assert events == ["start", "wait-fail", "stop"]
    # nothing was processed, so the file should not be marked done
    assert not (tmp_path / "state").exists()


def test_run_once_does_not_mark_processed_on_extraction_failure(tmp_path, monkeypatch):
    csv_path = tmp_path / "raw" / "df_20240101_000000.csv"
    csv_path.parent.mkdir()
    csv_path.write_text(CSV_TEXT)

    monkeypatch.setattr(mod, "start_ollama", lambda compose_file: None)
    monkeypatch.setattr(mod, "wait_for_ollama", lambda host, **kw: None)
    monkeypatch.setattr(mod, "stop_ollama", lambda compose_file: None)

    def _raise_process_csv(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "process_csv", _raise_process_csv)

    summaries = mod.run_once(raw_data_dir=csv_path.parent, state_file=tmp_path / "state", output_dir=tmp_path / "out")

    assert summaries == []
    assert not (tmp_path / "state").exists()
