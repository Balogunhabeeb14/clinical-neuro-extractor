# Assessment extractor eval

Measures `clinical_neuro_extractor.assessment_llm_extractor` against a known
set of cases. Runs entirely against your local Ollama server - nothing is
sent to a cloud API.

## Cases

`cases.jsonl` has 10 cases (1 real de-identified sample + 9 synthetic,
anchored on its format/difficulty - see the case review page from when this
was built) with expected `battery`/`domain`/`test`/`subtest`/`metric`/`value`
extractions.

## Grading

Programmatic, not judge-based - see `grader.py`. A row counts as "found"
when its normalized `(test, value)` matches an expected row's. Three
metrics per case:

- **recall** - fraction of expected measures found. The headline number.
- **precision** - fraction of extracted rows that were actually expected
  (catches hallucinated/invented measures).
- **field_accuracy** - among matched rows, fraction with the right
  `domain`/`battery`/`metric` too (these are more paraphrase-prone than
  `test`/`value`, so scored separately rather than folded into recall).

## Running it

```bash
pip install -e ".[dev,llm]"
ollama pull llama3.1          # or whatever model you're pointing at
python -m evals.assessment_extraction.runner
```

Options: `--model`, `--host` (defaults to `OLLAMA_HOST` env var, else
`http://localhost:11434`), `--variant` (compare models/prompts by writing to
different subdirectories, e.g. `--variant v1-qwen --model qwen2.5`),
`--reps` (repeat each case N times), `--summary-only` (just print from
existing results without re-running).

Safe to re-run/interrupt: already-completed `(case, rep)` pairs are skipped,
so a crash mid-run only costs what hadn't finished yet.

## Output

`<variant>/results.jsonl` (per-case grade + latency), `<variant>/errors.jsonl`
(failed attempts - a bad case never silently becomes a zero score), and
`<variant>/traces/<id>_rep<k>.json` (input text + extracted rows, for
spot-checking). All generated output is gitignored; `cases.jsonl` is the
only file meant to be committed.

The runner prints a per-case and mean summary table itself - no extra
tooling needed to get numbers out of a run.
