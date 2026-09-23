#!/usr/bin/env python3
"""Run extraction automatically when ingest.py writes a new batch CSV.

Intended usage: add this as the next step in the same cron job as
ingest.py, e.g.

    0 3 1 * * cd /path/to/repo && python ingest.py && python -m automation.extract_on_new_data --once

Each run: finds any df_*.csv under REGISTRY_RAW_DATA_DIR not yet processed
(tracked in a small state file), starts the Ollama Docker container
on-demand (`docker compose up -d ollama`), waits for it to be ready, runs
the patient registry + assessment/report extractors over each new file,
writes the results to OUTPUT_DIR, then stops the Ollama container again -
so it isn't holding memory/CPU between batches. A crash partway through
doesn't reprocess already-completed files (the state file is updated after
each one succeeds), and a file that fails to process is logged and
retried on the next run rather than marked done.

Use --watch instead of --once to run as a standalone polling daemon if you
don't control the scheduler that runs ingest.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from clinical_neuro_extractor.assessment_llm_extractor import extract_assessment_measures_batch
from clinical_neuro_extractor.registry import build_patient_registry
from clinical_neuro_extractor.report_llm_extractor import extract_report_details_batch
from webapp.sources import load_csv

logger = logging.getLogger("extract_on_new_data")

DEFAULT_STATE_FILE = Path(__file__).parent / ".last_processed"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "extraction_output"
DEFAULT_COMPOSE_FILE = Path(__file__).parent.parent / "docker-compose.yml"


def find_new_csvs(raw_data_dir: Path, state_file: Path) -> list[Path]:
    """Return df_*.csv files under raw_data_dir not yet recorded as processed."""
    processed = set()
    if state_file.exists():
        processed = set(state_file.read_text().splitlines())

    candidates = sorted(Path(raw_data_dir).glob("df_*.csv"))
    return [p for p in candidates if p.name not in processed]


def mark_processed(path: Path, state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "a") as f:
        f.write(path.name + "\n")


def start_ollama(compose_file: Path) -> None:
    logger.info("Starting Ollama container")
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d", "ollama"],
        check=True,
    )


def stop_ollama(compose_file: Path) -> None:
    logger.info("Stopping Ollama container")
    subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "stop", "ollama"],
        check=True,
    )


def wait_for_ollama(host: str, timeout: float = 120.0, poll_interval: float = 2.0) -> None:
    """Block until Ollama's API responds, or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
        time.sleep(poll_interval)
    raise TimeoutError(f"Ollama at {host} did not become ready within {timeout}s: {last_error}")


def process_csv(
    csv_path: Path,
    *,
    output_dir: Path,
    model: str,
    host: Optional[str],
    client=None,
) -> dict:
    """Run the registry + LLM extractors over one batch CSV, writing CSV outputs.

    Returns a summary dict (patient/document/measure counts) for logging.
    """
    documents = load_csv(csv_path)

    batch_dir = output_dir / csv_path.stem
    batch_dir.mkdir(parents=True, exist_ok=True)

    registry = build_patient_registry(documents)
    registry.to_csv(batch_dir / "registry.csv", index=False)

    measures = extract_assessment_measures_batch(documents, client=client, model=model, host=host)
    measures.to_csv(batch_dir / "assessment_measures.csv", index=False)

    details, domain_summaries = extract_report_details_batch(
        documents, client=client, model=model, host=host
    )
    details.to_csv(batch_dir / "report_details.csv", index=False)
    domain_summaries.to_csv(batch_dir / "report_domain_summaries.csv", index=False)

    return {
        "csv": csv_path.name,
        "n_documents": len(documents),
        "n_patients": len(registry),
        "n_assessment_measures": len(measures),
        "n_report_details": len(details),
        "n_report_errors": int((details["error"].notna()).sum()) if "error" in details else 0,
        "n_assessment_errors": int((measures.get("error", pd.Series(dtype=object)).notna()).sum()),
    }


def run_once(
    *,
    raw_data_dir: Path,
    state_file: Path = DEFAULT_STATE_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    compose_file: Path = DEFAULT_COMPOSE_FILE,
    model: str = "llama3.1",
    host: Optional[str] = None,
    use_docker: bool = True,
    client=None,
) -> list[dict]:
    """Process every unprocessed batch CSV once. Returns a summary per file processed."""
    new_csvs = find_new_csvs(raw_data_dir, state_file)
    if not new_csvs:
        logger.info("No new batch CSVs under %s", raw_data_dir)
        return []

    logger.info("Found %d new batch CSV(s): %s", len(new_csvs), [p.name for p in new_csvs])

    summaries = []
    try:
        if use_docker:
            start_ollama(compose_file)
            wait_for_ollama(host or "http://localhost:11434")

        for csv_path in new_csvs:
            try:
                summary = process_csv(
                    csv_path, output_dir=output_dir, model=model, host=host, client=client
                )
            except Exception:
                logger.exception("Failed to process %s - will retry on next run", csv_path.name)
                continue
            summaries.append(summary)
            mark_processed(csv_path, state_file)
            logger.info("Processed %s: %s", csv_path.name, json.dumps(summary))
    finally:
        # Always try to stop what we started, even if start/wait_for_ollama
        # failed or a file's processing raised something unexpected - never
        # leave the container running past this function on our way out.
        # Swallow a stop failure so it doesn't mask whatever original error
        # is already propagating.
        if use_docker:
            try:
                stop_ollama(compose_file)
            except Exception:
                logger.exception("Failed to stop Ollama container during cleanup")

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-data-dir", required=True, help="Directory ingest.py writes df_*.csv into")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--compose-file", type=Path, default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--host", default=None, help="Ollama host (default: OLLAMA_HOST env var, else localhost:11434)")
    parser.add_argument("--no-docker", action="store_true", help="Assume Ollama is already running; don't start/stop it")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Process any new files once, then exit")
    mode.add_argument("--watch", action="store_true", help="Poll for new files continuously")
    parser.add_argument("--poll-seconds", type=int, default=300, help="Polling interval for --watch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    kwargs = dict(
        raw_data_dir=args.raw_data_dir,
        state_file=args.state_file,
        output_dir=args.output_dir,
        compose_file=args.compose_file,
        model=args.model,
        host=args.host,
        use_docker=not args.no_docker,
    )

    if args.once:
        run_once(**kwargs)
        return

    logger.info("Watching %s every %ds (Ctrl+C to stop)", args.raw_data_dir, args.poll_seconds)
    while True:
        run_once(**kwargs)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
