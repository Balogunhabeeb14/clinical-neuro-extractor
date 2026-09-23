import io
import sys

import pandas as pd
import pytest

flask = pytest.importorskip("flask")

from webapp import sources


CSV_TEXT = """document_guid,client_guid,document_description,body_analysed
doc-1,pt-1,report,hello
doc-2,pt-1,assessment,world
"""


def test_load_csv_aligns_document_fields(tmp_path):
    path = tmp_path / "df_20240101_000000.csv"
    path.write_text(CSV_TEXT)

    df = sources.load_csv(path)

    assert list(df.columns[:4]) == [
        "document_guid",
        "client_guid",
        "client_idcode",
        "client_touchedwhen",
    ]
    assert len(df) == 2
    assert df.loc[0, "document_description"] == "report"


def test_find_latest_batch_csv_picks_most_recent_by_name(tmp_path):
    (tmp_path / "df_20240101_000000.csv").write_text(CSV_TEXT)
    latest = tmp_path / "df_20240601_000000.csv"
    latest.write_text(CSV_TEXT)

    found = sources.find_latest_batch_csv(str(tmp_path))

    assert found == latest


def test_find_latest_batch_csv_returns_none_when_missing(tmp_path):
    assert sources.find_latest_batch_csv(str(tmp_path)) is None
    assert sources.find_latest_batch_csv(None) is None


def test_resolve_documents_prefers_uploaded_file(tmp_path):
    (tmp_path / "df_20240101_000000.csv").write_text(CSV_TEXT)
    upload = io.BytesIO(CSV_TEXT.encode())
    upload.filename = "upload.csv"

    df, label, warning = sources.resolve_documents(
        uploaded_file=upload, raw_data_dir=str(tmp_path)
    )

    assert label == "uploaded CSV"
    assert warning is None
    assert len(df) == 2


def test_resolve_documents_falls_back_to_batch_csv(tmp_path):
    batch = tmp_path / "df_20240101_000000.csv"
    batch.write_text(CSV_TEXT)

    df, label, warning = sources.resolve_documents(raw_data_dir=str(tmp_path))

    assert label == f"batch CSV ({batch.name})"
    assert warning is None
    assert len(df) == 2


def test_resolve_documents_raises_when_no_csv_and_no_cogstack_module(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "COGSTACK_UTIL_PATH", None)

    with pytest.raises(sources.CogStackUnavailable):
        sources.resolve_documents(raw_data_dir=str(tmp_path))


def test_fetch_from_cogstack_uses_existing_cs_core_v1_client(monkeypatch):
    calls = []

    class _FakeCs:
        def cohort_searcher_no_terms(self, *, index_name, fields_list, search_string):
            calls.append((index_name, fields_list, search_string))
            return [{"document_guid": "doc-1", "client_guid": "pt-1"}]

    fake_module = type(sys)("cs_core_v1")
    fake_module.cs = _FakeCs()
    monkeypatch.setitem(sys.modules, "cs_core_v1", fake_module)

    df = sources.fetch_from_cogstack()

    assert len(calls) == 1
    assert calls[0][0] == sources.COGSTACK_INDEX
    assert isinstance(df, pd.DataFrame)
    assert df.loc[0, "document_guid"] == "doc-1"


def test_fetch_from_cogstack_wraps_query_errors(monkeypatch):
    class _FailingCs:
        def cohort_searcher_no_terms(self, **kwargs):
            raise RuntimeError("connection refused")

    fake_module = type(sys)("cs_core_v1")
    fake_module.cs = _FailingCs()
    monkeypatch.setitem(sys.modules, "cs_core_v1", fake_module)

    with pytest.raises(sources.CogStackUnavailable):
        sources.fetch_from_cogstack()


def test_resolve_documents_falls_back_to_batch_csv_when_refresh_fails(tmp_path, monkeypatch):
    batch = tmp_path / "df_20240101_000000.csv"
    batch.write_text(CSV_TEXT)
    monkeypatch.setattr(sources, "COGSTACK_UTIL_PATH", None)

    df, label, warning = sources.resolve_documents(
        raw_data_dir=str(tmp_path), force_refresh=True
    )

    assert label == f"batch CSV ({batch.name})"
    assert warning is not None and "CogStack refresh failed" in warning
