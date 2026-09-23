"""Run assessment_llm_extractor over cases.jsonl and grade each result.

Runs entirely against your local Ollama server - no cloud API, no clinical
text leaves your infrastructure.

Usage:
    python -m evals.assessment_extraction.runner
    python -m evals.assessment_extraction.runner --model llama3.1 --host http://localhost:11434
    python -m evals.assessment_extraction.runner --variant v1-bigger-model --model llama3.1:70b
    python -m evals.assessment_extraction.runner --summary-only  # just print from existing results

Writes ``evals/assessment_extraction/<variant>/results.jsonl`` (one row per
case/rep), ``errors.jsonl`` (failed attempts), and ``traces/<id>_rep<k>.json``
(the input text and extracted rows for that case). Safe to re-run: already
completed (case, rep) pairs are skipped, so a crash mid-run only costs the
cases still outstanding.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from clinical_neuro_extractor.assessment_llm_extractor import (
    OllamaUnavailable,
    extract_assessment_measures,
)

from .grader import grade_measures

FLOW_DIR = Path(__file__).parent
CASES_PATH = FLOW_DIR / "cases.jsonl"


def load_cases() -> list[dict]:
    cases = []
    with open(CASES_PATH) as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def _load_done(results_path: Path) -> set:
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                done.add((row["prompt_id"], row.get("rep", 0)))
    return done


def run(
    variant: str = "baseline",
    model: str = "llama3.1",
    host: Optional[str] = None,
    reps: int = 1,
    client=None,
) -> Path:
    """Run the eval, writing results under ``evals/assessment_extraction/<variant>/``.

    ``client`` lets tests inject a fake Ollama client; real runs leave it
    unset and let the extractor build one from ``host``.
    """
    cases = load_cases()
    variant_dir = FLOW_DIR / variant
    traces_dir = variant_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    results_path = variant_dir / "results.jsonl"
    errors_path = variant_dir / "errors.jsonl"
    done = _load_done(results_path)

    with open(results_path, "a") as results_f, open(errors_path, "a") as errors_f:
        for case in cases:
            for rep in range(reps):
                if (case["id"], rep) in done:
                    continue

                start = time.monotonic()
                try:
                    extracted = extract_assessment_measures(
                        case["input"], client=client, model=model, host=host
                    )
                    status, error = "ok", None
                except OllamaUnavailable as exc:
                    extracted, status, error = pd.DataFrame(), "error", str(exc)
                latency_s = time.monotonic() - start

                trace_path = traces_dir / f"{case['id']}_rep{rep}.json"
                trace_path.write_text(
                    json.dumps(
                        [
                            {"role": "user", "content": case["input"]},
                            {
                                "role": "assistant",
                                "content": (
                                    extracted.to_json(orient="records")
                                    if status == "ok"
                                    else f"ERROR: {error}"
                                ),
                            },
                        ],
                        indent=2,
                    )
                )

                if status == "error":
                    errors_f.write(
                        json.dumps(
                            {
                                "prompt_id": case["id"],
                                "rep": rep,
                                "failure_class": "harness-or-serving-error",
                                "error": error,
                                "model": model,
                            }
                        )
                        + "\n"
                    )
                    errors_f.flush()
                    continue

                grade = grade_measures(case["expected"], extracted)
                results_f.write(
                    json.dumps(
                        {
                            "prompt_id": case["id"],
                            "tags": case["tags"],
                            "rep": rep,
                            "status": status,
                            "grade": grade,
                            "model": model,
                            "latency_s": latency_s,
                        }
                    )
                    + "\n"
                )
                results_f.flush()

    return results_path


def summarize(variant: str = "baseline") -> None:
    """Print a per-case and aggregate summary from an existing results.jsonl."""
    results_path = FLOW_DIR / variant / "results.jsonl"
    if not results_path.exists():
        print(f"No results yet at {results_path}")
        return

    rows = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        print(f"{results_path} is empty")
        return

    print(f"{'case':<28} {'recall':>7} {'precision':>10} {'field_acc':>10}")
    for row in rows:
        g = row["grade"]
        print(
            f"{row['prompt_id']:<28} {g['recall']:>7.2f} {g['precision']:>10.2f} "
            f"{g['field_accuracy']:>10.2f}"
        )

    n = len(rows)
    avg = lambda key: sum(r["grade"][key] for r in rows) / n  # noqa: E731
    print("-" * 58)
    print(
        f"{'MEAN (' + str(n) + ' cases)':<28} {avg('recall'):>7.2f} "
        f"{avg('precision'):>10.2f} {avg('field_accuracy'):>10.2f}"
    )

    errors_path = FLOW_DIR / variant / "errors.jsonl"
    if errors_path.exists() and errors_path.stat().st_size > 0:
        n_errors = sum(1 for _ in open(errors_path) if _.strip())
        print(f"\n{n_errors} case(s) failed to extract - see {errors_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--host", default=None, help="Ollama host (default: OLLAMA_HOST env var, else localhost:11434)")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true", help="Print from existing results.jsonl without running")
    args = parser.parse_args()

    if not args.summary_only:
        run(variant=args.variant, model=args.model, host=args.host, reps=args.reps)
    summarize(variant=args.variant)


if __name__ == "__main__":
    main()
