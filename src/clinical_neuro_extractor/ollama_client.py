"""Structured extraction via a locally-hosted Ollama model.

Clinical text never leaves your infrastructure: no cloud API, no API key.
Requires a running Ollama server (https://ollama.com) with the target model
pulled, e.g. ``ollama pull llama3.1``.

Point at a non-default host with the ``OLLAMA_HOST`` environment variable
(default ``http://localhost:11434``).
"""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"

T = TypeVar("T", bound=BaseModel)


class OllamaUnavailable(RuntimeError):
    """The Ollama server couldn't be reached, or its response was unusable."""


def get_client(host: str | None = None):
    """Build an ``ollama.Client`` pointed at ``host`` (or ``OLLAMA_HOST``, or the default)."""
    try:
        import ollama
    except ImportError as exc:
        raise OllamaUnavailable(
            "The 'ollama' package isn't installed. Install the 'llm' extra "
            "(pip install -e \".[llm]\") and make sure an Ollama server is running."
        ) from exc

    return ollama.Client(host=host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST))


def extract_structured(
    *,
    system: str,
    user: str,
    schema: Type[T],
    client=None,
    model: str = DEFAULT_MODEL,
    host: str | None = None,
) -> T:
    """Call a local Ollama model and validate its JSON response against ``schema``."""
    if client is None:
        client = get_client(host)

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema.model_json_schema(),
            options={"temperature": 0},
        )
    except Exception as exc:  # noqa: BLE001 - connection errors, model not pulled, etc.
        raise OllamaUnavailable(f"Ollama request failed: {exc}") from exc

    content = response["message"]["content"]
    try:
        return schema.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise OllamaUnavailable(
            f"Model response didn't match the expected schema: {exc}"
        ) from exc
