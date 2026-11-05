# Cross-language evidence alignment

## Full GPT-6.1 Terra run

The full run uses **`gpt-6.1-terra` directly through the OpenAI Responses API** with
`OPENAI_API_KEY` from the project environment. The endpoint is fixed to
`https://api.openai.com/v1`; it does not route through OpenRouter or substitute
another model. Source and target languages are restricted to **EN, DE, ES, FR,
and ZH**. Arabic and Russian are excluded from this run.

The frozen workload retains all **10,691 published QAs** in its QA index:
**8,931** have eligible exact source passages, **1,750** have no exact passage,
and **10** belong to the quarantined chemistry record `EP-1732767-A1_es`.
The eligible collection contains **636 documents**, **4,930 distinct source
passages**, **1,532 document-pair request groups**, and **12,855 target-language
alignments**. Chemistry uses only accepted BASF direct translation pairs.

Prepare the full workload from the verified releases and frozen pilot inventory:

```bash
.venv/bin/python -m clir_bench.alignment.terra_data \
  --output data/evidence_alignment/terra-20261105
```

Run or resume with the same command:

```bash
.venv/bin/python -m clir_bench.alignment.terra \
  --run-dir data/evidence_alignment/terra-20261105 \
  --workers 4
```

Requests use **medium reasoning**, a **32,768-token output limit**, and at most
three attempts. Each request includes both complete document versions, the
source passages, and their linked questions. The model must return the smallest
sufficient contiguous target quotation without changing the source text.
Existing exact source-language evidence is retained without a model call.
Matching saved attempts are reused on resume; immutable inputs and request hashes
prevent results from being attached to different questions or passages.

The run directory contains the following durable outputs:

| File | Contents and update timing |
|---|---|
| `tasks.jsonl`, `qa_index.jsonl`, `preparation.json` | Frozen requests, all published QA identities, and verified provenance/counts. |
| `attempts/` | Raw request-attempt checkpoints, responses, usage, and errors. |
| `alignment.sqlite` | Authoritative transactional state, using SQLite WAL and full synchronization. |
| `calls.csv` | One record per completed API attempt, flushed and fsynced before a retry or subsequent request. |
| `qa_updates.csv` | Changed QA rows appended after each completed task, with increasing `revision`; flushed and fsynced before returning. |
| `qa_evidence.csv` | Complete 10,691-row QA snapshot at initialization/recovery and on pause or completion. |
| `results.csv` | Current alignment-result snapshot, refreshed with the QA snapshot. |
| `run.json` | Runner PID, progress, model settings, status, and recorded usage costs. |

`evidence_all_languages` is JSON keyed by language, for example
`{"en": [...], "fr": [...]}`. Original exact evidence and accepted exact target
quotations retain source identifiers and provenance. Missing, uncertain, invalid,
and pending targets remain explicit in `evidence_alignment_status`; they do not
receive fabricated evidence. Compact metadata preserves model/persona/mode and
the immutable published QA lookup without repeating full source documents in
every CSV row.

For the latest QA state while the run is active, overlay the row with the largest
`revision` for each `qa_id` from `qa_updates.csv` onto `qa_evidence.csv`. Multiple
journal revisions represent updates to the same QA, not additional questions.
Reopening the run rebuilds CSV journals and snapshots from committed SQLite
state, then replays matching raw attempts as needed.

To pause gracefully, send **SIGTERM** to the PID recorded in `run.json`
(`kill -TERM <pid>`). The runner stops scheduling new calls, checkpoints in-flight
responses, and exports the latest full snapshots before exiting. Ctrl-C/SIGINT
uses the same shutdown path. Resume with the command above.

An exact substring check establishes quotation fidelity, not semantic correctness.
New cross-language matches are labeled **`pending_review`** until their support,
qualifications, and minimality have been independently assessed. The full run
preserves the original questions and answers. **No Hugging Face upload or
retrieval chunking is performed by these commands.**

## Historical GPT-6 Luna pilot

The earlier 30-document pilot remains in `data/evidence_alignment/pilot-20261105`.
It used GPT-6 Luna through OpenRouter and included the six available UN languages.
Its commands and pricing below describe that historical pilot only; the full
Terra run above uses separate inputs, checkpoints, and outputs.

### Inputs and selection

`clir_bench.alignment.data` reads the frozen published EUR-Lex and UN releases and
the pinned BASF chemistry snapshot. An anchor is an exact, nonempty evidence
passage in its source document, falling back to an exact answer. No whitespace
normalization or fuzzy matching is allowed. Identical anchors within a document
and language share one result while retaining all original QA links.

The deterministic sample contains ten language-neutral documents per corpus.
Selection covers source languages and short/medium/long texts, UN resolutions
and meetings, and source-finding questions. Chemistry uses accepted BASF direct
translation pairs, not patent-family or transitive language links. The known
mislabeled `EP-1732767-A1_es` endpoint is excluded from source and target selection.

Preparation records the selected documents, exclusions, source provenance, and
all planned requests before any generation. Existing prepared files are immutable;
changing the selection or input texts requires a new run directory.

```bash
.venv/bin/python -m clir_bench.alignment.data \
  --output data/evidence_alignment/pilot-20261105
```

### Generation and checkpoints

Set a working `OPENROUTER_API_KEY` in the project's `.env`. The runner uses
OpenRouter's Responses endpoint with `openai/gpt-6-luna`, checks availability,
and stops on authentication or model-access failures. It never
substitutes a different model.

```bash
.venv/bin/python -m clir_bench.alignment.runner run \
  --run-dir data/evidence_alignment/pilot-20261105
```

Each request supplies both complete document versions and all distinct anchors
for that source/target pair. The OpenRouter Responses API uses medium reasoning, a
32,768-token output limit, and a strict JSON schema. Every anchor returns
`matched`, `not_found`, or `uncertain`; only `matched` includes a quotation.
Quotations must be contiguous, verbatim, and preserve necessary conditions,
exceptions, quantities, and negations. Non-matches are recorded rather than
replaced with generated translations.

The runner records raw responses and usage for every attempt, allows at most
three attempts for transient or invalid responses, and resumes saved work.
Document language mismatches and invalid quotations remain visible in results.
All generation calls are restricted to `tasks.jsonl`, the frozen pilot sample;
`full_tasks.jsonl` is used only for inventory and token-count estimation.

### Review and feasibility gate

```bash
.venv/bin/python -m clir_bench.alignment.report \
  --run-dir data/evidence_alignment/pilot-20261105 \
  --output reports/evidence-alignment-pilot-20261105
```

The report recomputes literal containment against the supplied target text. It
also writes `review_queue.jsonl` with pending exact matches. For each reviewed
match, save a record in `reviews.jsonl` containing its `alignment_id`,
`result_hash`, `reviewer`, `notes`, and Boolean judgments for:

- `semantic_match`: the quote supplies equivalent support.
- `qualifications_preserved`: necessary conditions and exceptions remain intact.
- `minimal`: the quote has no removable padding.
- `language_correct`: the text uses the declared target language.

Use the full source and target texts in `tasks.jsonl` for contextual review. The
queue is a starting point, not an automatic approval. Leave uncertain judgments
unapproved. Hashes bind reviews to the actual quotation and source identities;
changed quotations cannot inherit earlier reviews. Re-run the report after review.
The report writes only exact, reviewed matches to `accepted_alignments.jsonl`.
Pilot semantic reviews are qualitative assistant judgments, not expert annotations.

Each corpus must have at least 95% approved proposed matches and at least 90%
valid coverage over all its alignment tasks. Every retained match must be exact
and pass all four review criteria. Missing results remain in the denominator;
unreviewed results keep the gate pending. Reports separate first-attempt and final
outcomes by corpus and target language. A 30-document pilot does not establish
dataset-wide accuracy.

### Cost estimation

After the pilot passes, count full-workload inputs without running full generation:

```bash
.venv/bin/python -m clir_bench.alignment.runner count \
  --run-dir data/evidence_alignment/pilot-20261105
.venv/bin/python -m clir_bench.alignment.report \
  --run-dir data/evidence_alignment/pilot-20261105 \
  --output reports/evidence-alignment-pilot-20261105
```

Input counts first try the model's token-count endpoint with the same request
payloads. OpenRouter returned 404 for this endpoint in the pilot, so the fallback
uses local `o200k_base` token counts calibrated against actual pilot input usage.
These counts are explicitly labeled approximate, with calibration provenance.
Request hashes prevent stale counts from being reused. Output usage includes
reasoning and protocol retries through the first valid raw response, and is
extrapolated per anchor within corpus, source language, target language, and
document genre; sparse strata fall back explicitly to broader language/corpus
groups. Calls after the first valid response remain in actual spending but do
not inflate projected normal operation.

Full-workload input costs assume no cache-read hits and price all inputs at the
$0.125/M cache-write rate, reflecting observed first-call billing. A conservative
estimate adds 25% to measured output and retry-adjusted input, plus an additional
10% input allowance for calibrated counts. This allowance is a heuristic, not a
statistical confidence interval.

Actual pilot costs prefer OpenRouter's reported `usage.cost` where available and
retain the token-rate calculation separately. They include all recorded attempts
with valid usage. Missing usage
is marked unknown, never zero. Authentication or access failures rejected before
generation are recorded separately and excluded from generated-token costs.
Old prompt versions remain in actual spending, but only matching current request
hashes contribute to the output/retry extrapolation. Standard rates per million tokens are $0.10 input,
$0.01 cached input, $0.125 cache writes, and $0.50 output. For requests above
272,000 input tokens, input/cache rates double and output rates multiply by 1.5.
Optional Batch estimates use half the corresponding rates. Rates were verified
on 2026-11-05 against the [official GPT-6 Luna page](https://developers.openai.com/api/docs/models/gpt-6-luna).

Cost estimates use API usage, not billing invoices. An interrupted run can leave
in-flight requests without saved responses. Such incidents are recorded in
`incidents.jsonl`, keep final pilot spending unknown, and receive a separately
labeled allowance based on the possible request count, largest observed or
estimated input, and the full output limit. This is not a verified billing bound.
Complete saved pilot responses can still support a full-workload estimate.

The report withholds a full-run estimate until generation is complete, every
corpus passes, and all required saved usage and full-workload input counts are
available. Neither reporting nor token counting launches full-dataset alignment
or an upload.

When a cost estimate is requested before quality approval, add `--estimate-cost`
to the report command. This reports the budget separately from the quality gate;
it does not approve or start full-dataset generation.
