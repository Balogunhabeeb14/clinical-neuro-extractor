import json

import pytest

pytest.importorskip("pydantic")

from clinical_neuro_extractor.assessment_llm_extractor import AssessmentExtraction, ExtractedMeasure
from evals.assessment_extraction import runner


class _FakeClient:
    """Always returns one canned measure - runner-logic tests don't need real content."""

    def __init__(self):
        self.calls = 0
        self._content = AssessmentExtraction(
            measures=[ExtractedMeasure(battery="b", domain="d", test="t", metric="m", value="1")]
        ).model_dump_json()

    def chat(self, **kwargs):
        self.calls += 1
        return {"message": {"content": self._content}}


class _RaisingClient:
    def chat(self, **kwargs):
        raise RuntimeError("no route to host")


@pytest.fixture(autouse=True)
def _isolated_flow_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FLOW_DIR", tmp_path)
    monkeypatch.setattr(runner, "CASES_PATH", tmp_path / "cases.jsonl")
    cases = [
        {
            "id": "case-a",
            "tags": ["synthetic"],
            "input": "some assessment text",
            "expected": [{"battery": "b", "domain": "d", "test": "t", "metric": "m", "value": "1"}],
        },
        {
            "id": "case-b",
            "tags": ["synthetic"],
            "input": "other assessment text",
            "expected": [{"battery": "b", "domain": "d", "test": "t", "metric": "m", "value": "9"}],
        },
    ]
    with open(tmp_path / "cases.jsonl", "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    return tmp_path


def test_run_writes_one_result_row_per_case(tmp_path):
    client = _FakeClient()

    results_path = runner.run(variant="baseline", client=client)

    rows = [json.loads(line) for line in open(results_path)]
    assert len(rows) == 2
    assert {r["prompt_id"] for r in rows} == {"case-a", "case-b"}
    assert client.calls == 2
    # case-a's value matches expected -> perfect recall; case-b's doesn't -> 0 recall
    by_id = {r["prompt_id"]: r for r in rows}
    assert by_id["case-a"]["grade"]["recall"] == 1.0
    assert by_id["case-b"]["grade"]["recall"] == 0.0


def test_run_is_resumable_and_skips_completed_cases(tmp_path):
    client = _FakeClient()
    runner.run(variant="baseline", client=client)
    assert client.calls == 2

    # second run with a fresh client should make no new calls - everything's done
    client2 = _FakeClient()
    runner.run(variant="baseline", client=client2)
    assert client2.calls == 0


def test_run_writes_errors_sidecar_without_aborting(tmp_path):
    client = _RaisingClient()

    results_path = runner.run(variant="baseline", client=client)

    rows = [json.loads(line) for line in open(results_path)]
    assert rows == []  # both cases failed, nothing scorable

    errors_path = runner.FLOW_DIR / "baseline" / "errors.jsonl"
    errors = [json.loads(line) for line in open(errors_path)]
    assert len(errors) == 2
    assert all("no route to host" in e["error"] for e in errors)


def test_run_writes_trace_files(tmp_path):
    client = _FakeClient()
    runner.run(variant="baseline", client=client)

    trace = runner.FLOW_DIR / "baseline" / "traces" / "case-a_rep0.json"
    assert trace.exists()
    turns = json.loads(trace.read_text())
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "some assessment text"


def test_summarize_runs_without_error_on_populated_and_empty_results(capsys):
    runner.summarize(variant="baseline")
    out = capsys.readouterr().out
    assert "No results yet" in out

    client = _FakeClient()
    runner.run(variant="baseline", client=client)
    runner.summarize(variant="baseline")
    out = capsys.readouterr().out
    assert "case-a" in out
    assert "MEAN" in out
