"""Grade assessment_llm_extractor output against an eval case's expected measures.

Programmatic, not judge-based: ground truth (`evals/assessment_extraction/cases.jsonl`)
is a known list of (battery, domain, test, subtest, metric, value) tuples,
so scoring is a field-comparison problem, not something that needs a
second model's judgment.

A row counts as "found" when its normalized ``(test, value)`` matches an
expected row's - that pair is the most stable, least paraphrase-prone
identifier a case has. ``domain``/``battery``/``metric`` correctness is
scored separately, conditioned on that match, since those fields are more
prone to reasonable rewording (e.g. "IQ" vs "index score") than the test
name and the value itself.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


def _normalize(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_value(value: Optional[str]) -> str:
    """Normalize a score value: strip a trailing prorated '*', collapse
    whitespace, and drop a trailing '.0' on whole numbers (so 97 == 97.0)."""
    text = _normalize(value).rstrip("*").strip()
    match = re.fullmatch(r"(-?\d+)\.0", text)
    return match.group(1) if match else text


def grade_measures(expected: list[dict], extracted: pd.DataFrame) -> dict:
    """Compare extracted measures to a case's expected ground truth.

    Returns a dict with ``recall``, ``precision``, and ``field_accuracy``
    (each in [0, 1]) plus ``n_expected``/``n_extracted``/``n_matched`` for
    debugging. An empty expected list scores recall=1.0 (nothing to miss);
    an empty extraction with a non-empty expected list scores recall=0.0
    and precision=0.0 (nothing extracted, nothing to be right about).
    """
    extracted_rows = extracted.to_dict(orient="records") if not extracted.empty else []

    def key(row: dict) -> tuple:
        return (_normalize(row.get("test")), _normalize_value(row.get("value")))

    expected_by_key: dict[tuple, list[dict]] = {}
    for row in expected:
        expected_by_key.setdefault(key(row), []).append(row)
    claimed = {k: [False] * len(v) for k, v in expected_by_key.items()}

    matched_pairs = []
    for erow in extracted_rows:
        k = key(erow)
        candidates = expected_by_key.get(k)
        if not candidates:
            continue
        slot = next((i for i, done in enumerate(claimed[k]) if not done), None)
        if slot is None:
            continue
        claimed[k][slot] = True
        matched_pairs.append((candidates[slot], erow))

    n_expected = len(expected)
    n_extracted = len(extracted_rows)
    n_matched = len(matched_pairs)

    recall = (n_matched / n_expected) if n_expected else 1.0
    precision = (n_matched / n_extracted) if n_extracted else (1.0 if n_expected == 0 else 0.0)

    if matched_pairs:
        correct_fields = sum(
            1
            for exp, ext in matched_pairs
            if _normalize(exp.get("domain")) == _normalize(ext.get("domain"))
            and _normalize(exp.get("battery")) == _normalize(ext.get("battery"))
            and _normalize(exp.get("metric")) == _normalize(ext.get("metric"))
        )
        field_accuracy = correct_fields / len(matched_pairs)
    else:
        field_accuracy = 0.0 if n_expected else 1.0

    return {
        "recall": recall,
        "precision": precision,
        "field_accuracy": field_accuracy,
        "n_expected": n_expected,
        "n_extracted": n_extracted,
        "n_matched": n_matched,
    }
