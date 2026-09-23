import pytest

pytest.importorskip("pydantic")

from pydantic import BaseModel

from clinical_neuro_extractor.ollama_client import (
    DEFAULT_HOST,
    OllamaUnavailable,
    extract_structured,
)


class _Schema(BaseModel):
    name: str
    value: int


class _FakeClient:
    def __init__(self, content: str):
        self._content = content
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self._content}}


def test_extract_structured_validates_response_against_schema():
    client = _FakeClient('{"name": "vocab", "value": 12}')

    result = extract_structured(
        system="sys", user="usr", schema=_Schema, client=client, model="llama3.1"
    )

    assert result == _Schema(name="vocab", value=12)
    call = client.calls[0]
    assert call["model"] == "llama3.1"
    assert call["format"] == _Schema.model_json_schema()
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_extract_structured_wraps_transport_errors():
    class _RaisingClient:
        def chat(self, **kwargs):
            raise ConnectionError("no route to host")

    with pytest.raises(OllamaUnavailable, match="Ollama request failed"):
        extract_structured(system="sys", user="usr", schema=_Schema, client=_RaisingClient())


def test_extract_structured_wraps_schema_mismatch():
    client = _FakeClient("not valid json")

    with pytest.raises(OllamaUnavailable, match="didn't match the expected schema"):
        extract_structured(system="sys", user="usr", schema=_Schema, client=client)


def test_get_client_requires_ollama_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ollama":
            raise ImportError("no module named ollama")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from clinical_neuro_extractor.ollama_client import get_client

    with pytest.raises(OllamaUnavailable, match="ollama' package isn't installed"):
        get_client()


def test_default_host_is_local():
    assert DEFAULT_HOST == "http://localhost:11434"
