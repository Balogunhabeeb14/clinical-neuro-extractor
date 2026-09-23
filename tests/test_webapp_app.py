import io

import pytest

flask = pytest.importorskip("flask")

from webapp.app import create_app


CSV_TEXT = """document_guid,client_guid,document_description,body_analysed,client_touchedwhen
doc-1,pt-1,report,hello,2024-01-01
doc-2,pt-1,assessment,world,2024-01-02
doc-3,pt-2,report,another,2024-02-01
"""


@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_index_get_with_no_sources_shows_error(client, monkeypatch):
    import webapp.sources as sources

    monkeypatch.setattr(sources, "RAW_DATA_DIR", None)
    monkeypatch.setattr(sources, "COGSTACK_UTIL_PATH", None)

    resp = client.get("/")

    assert resp.status_code == 200
    assert b"Couldn" in resp.data  # "Couldn't load documents."


def test_index_post_with_uploaded_csv_renders_registry(client):
    data = {
        "csv_file": (io.BytesIO(CSV_TEXT.encode()), "documents.csv"),
    }
    resp = client.post("/", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "pt-1" in body
    assert "pt-2" in body
    assert "uploaded CSV" in body


def test_index_post_with_deidentify_hides_client_guid(client):
    data = {
        "csv_file": (io.BytesIO(CSV_TEXT.encode()), "documents.csv"),
        "deidentify": "1",
    }
    resp = client.post("/", data=data, content_type="multipart/form-data")

    body = resp.data.decode()
    assert "pt-1" not in body
    assert "patient_id" in body
