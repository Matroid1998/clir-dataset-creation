# CLIR benchmark

Builds and evaluates multilingual cross-language information retrieval benchmarks from
document collections that exist in several languages as human translations of one another.

The domains are chemistry patents (BASF-AI AI4Chem within-document abstract pairs)
and legal documents (EUR-Lex articles and United Nations document blocks). Domains declare
their sources and connect their workflows to the same `clir` commands.

## Why parallel documents

Cross-language retrieval is usually evaluated on machine-translated queries or documents,
which measures the translation as much as the retriever. Patent families give something
better: related document versions in several languages. Chemistry uses explicitly accepted
pairs of abstracts from the same publication; these direct links define which other
language versions are relevant. That relation — which documents are accepted counterparts
of one another — makes cross-lingual retrieval measurable.

## Install

```bash
uv sync                      # pipeline only
uv sync --extra chem         # + chemistry ontology tools and historical ingestion dependencies
uv sync --extra eval         # + the embedding-model stack (compute nodes)
cp .env.example .env         # then fill in the keys you need
```

## Formatting

```bash
uv run ruff format                       # Python
uv run ruff check
uv run python scripts/format_prompts.py  # the prompt templates
```

`scripts/format_prompts.py` hard-wraps the prompt files under
`src/clir_bench/domains` to 100 display columns. It only ever *splits* an
over-long line -- it never joins lines, and it refuses to write unless the
whitespace-stripped text is unchanged and a second pass is a no-op, so the only
thing that can move is where the newlines fall. Blocks shaped like the model's
runtime input (`### ` headers, `[EN] ...` unit headers, `Act:`/`Cite as:`
metadata, quoted source bodies) and the JSON output examples are left verbatim.
Chinese breaks after its own punctuation, never inside a word. `tests/` checks
the templates stay formatted; use `--check` to see what would change.

## Use

```bash
clir domains                 # what is installed
clir config                  # resolved settings and paths

clir ingest gp --limit 500   # import up to 500 accepted AI4Chem abstract pairs
clir corpus stats

clir qac generate --source gp --plan balanced --questions 100
clir qac generate --source gp --plan balanced --questions 4 --dry-run   # inspect the plan first

clir publish corpus
clir publish benchmark --source gp --dry-run

clir eval plan all --sbatch  # prepare the job; run it on a GPU node
clir analyze questions --run latest
clir runs list
```

To publish all generated QAs with supporting evidence, personas, metadata and related
documents, see [generation publishing](docs/generation_publishing.md).

Chemistry uses only
[`BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs`](https://huggingface.co/datasets/BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs).
The existing `gp` command key now names this source. Import keeps records with
`pair_ok=true` and `judge_verdict` equal to `exact` or `high`, with **both**
endpoint languages in `en`, `de`, `fr`, `es` or `zh`. It preserves each endpoint's
full cleaned abstract. `--languages` can further restrict that set;
`--revision` resolves and records an immutable upstream commit.

The corpus, accepted pair records and import manifest are stored under
`data/chemistry/ai4chem_abstract_pairs/`. Historical Google Patents and EPO files
remain on disk but are not active chemistry sources. `input_dataset_repo`
identifies the fixed upstream dataset; `corpus_repo` remains a separate destination
for publishing the derived retrieval corpus.

Retrieval positives are the selected document and its directly accepted paired
documents. Sharing a publication number or being linked through a third language
does not establish an additional positive. The cross-language export retains only
direct partners in other languages and omits queries with no retained partner.

Chemistry generation uses seven personas in English, German, Spanish, French and
Chinese: `lookup`, `process_retrieval`, `conceptual`, `constraint_search`,
`comparison`, `assertion_verification` and `source_finding`. All are enabled by
default; restrict them with `--modes lookup conceptual`, for example. These
replace the old `technical` and `semantic` generation modes. Each call can return
one to three candidates or abstain. Evidence and persona annotations are saved
in the CSV; the best output keeps one candidate per source document, language and
persona. See the [chemistry prompt contract](src/clir_bench/domains/chemistry/qac/prompts/README.md).

Chemistry plans select an existing document in `en`, `de`, `fr`, `es` or `zh`.
That document's language determines both the generation prompt and question
language; `--langs` filters the documents eligible for selection. Generation,
both graders, Jev screening and regrading use only that selected document's
text and publication identifier. The CSV preserves `source_document_id`, so
distinct source documents remain separate even when they share a publication
number and language. Regrading resolves the same document from the saved ID.

Chemistry batch verification uses the supplied `chemistry-clir-exceptional-r1`
rubric: shared faithfulness scores /15 and persona-specific quality scores /25,
with separate mode, support, metadata and batch-diversity checks. A score of 4
means strong or fully successful; every 5 requires a concrete exceptional
justification. Null marks a wholly unassessable criterion and prevents a complete
total. Identify the rubric when comparing scores with earlier chemistry runs.

To select a suitable chemistry persona before generation, use the Jev eligibility decider:

```bash
clir --domain chemistry qac generate --source gp --decider-model jev --questions 100
clir --domain chemistry qac generate --source gp --decider-model jev --questions 10 --dry-run
```

Jev assesses all seven personas independently from the selected source document.
The packaged `decider/jev_eligibility.json` preserves the supplied chemistry
eligibility template. Its 32/40 screening target is a quality forecast;
the default Yes-probability threshold is 0.5 because this prompt supplies no routing policy.
Among eligible personas, the pipeline selects the largest deficit against an equal target mix,
with reproducible tie-breaking from `--seed`. Override positive weights using
`[domains.chemistry.persona_weights]` in `clir.toml`. All-No documents are skipped.
Each selected source document is screened once per invocation; each receives its own
persona assignment. With the decider, `--questions` caps sampled source documents;
skips can reduce output.
Do not combine `--decider-model` with `--modes`. Dry runs make no model calls.

The results CSV retains the decider probabilities, eligible personas, chosen persona and prompt
hash through regrading. A sibling `<output_stem>_eligibility.json` records the prompt snapshot,
source hashes and all assignments, including skips; `--append` retains earlier audit entries.

Chemistry-specific benchmarks:

```bash
clir alias-graph build       # concepts with look-alike hard negatives
clir code-switch build       # swap a term, see whether retrieval survives
clir progressive all         # swap one more term per rung, measure decay
```

Legal generation and model comparisons use the existing EUR-Lex and UN batch pipelines:

```bash
clir --domain legal qac generate --source eurlex --questions 30
# data/legal/eurlex/qac/<timestamp>/results.csv

clir --domain legal qac generate --source un --questions 30
# data/legal/un_parallel/qac/<timestamp>/results.csv

# One invocation creates matching run folders under both corpus QAC roots:
clir --domain legal qac generate --source eurlex un --questions 30 \
  --generation-model provider/model-a --generation-model provider/model-b \
  --verifier-model google/gemini-3.8-flash --output results.csv

clir --domain legal qac generate --source eurlex un \
  --targets-from data/legal/eurlex/qac/comparison_2026-09-21 \
  --generation-model provider/model-c

clir --domain legal qac regrade \
  --input data/legal/un_parallel/qac/comparison_2026-09-21/results.csv

clir --domain legal qac best \
  --input data/legal/un_parallel/qac/comparison_2026-09-21_regrade_v2/results.csv
```

Each corpus run contains one `results.csv` combining its models and one `run.sqlite` state file. The CSV
includes all generated candidates for that corpus, their source text, model and persona, verifier scores,
and `is_best`; the state file records inputs, attempts, and completed stages. `--output`
sets the CSV filename inside the run folder. Omitting `--run-dir` creates a timestamped
folder under `data/legal/eurlex/qac/` for EUR-Lex or `data/legal/un_parallel/qac/` for UN.
When both corpora are selected, they receive the same run ID under their respective QAC roots.
An explicit `--run-dir` for multiple sources is a custom parent containing `eurlex/` and
`un_parallel/` run folders. Regrading infers the sources from the input CSV, creates a
separate run in each corresponding location, and preserves the original.
Best-only CSVs are written only by the explicit `best` command.

Legal runs share validated verifier grades across generators when the exact QA,
source context, verifier prompt/model/settings, and relevant candidate metadata match.
Candidate IDs and generator identities do not prevent reuse. Unseen candidates are
still graded in batches; each generator retains its own rows and candidate IDs.
The cache and reuse provenance live in `run.sqlite` and survive resume. Changed
answers, metadata, or verifier inputs require new grades; failed responses are not cached.
When reuse changes a quality batch, `quality_batch_diversity` is `not_evaluated`:
candidate scores are shared, but a judgment about a different batch is not copied.

For legal generation, `--questions 30` selects 30 target/language/persona cases per model,
split evenly across the selected sources. Each case can yield up to three candidates;
skipped targets or provider failures can reduce the number of successful cases. Every model
receives the same selected targets and contexts. `--targets-from` reuses a previous run's
selections, including its original source counts. Default modes and languages remain those
enabled by each source's batch pipeline; use `--modes` and `--langs` to restrict them.
`--dry-run` inspects the work without model calls or output files. To continue an interrupted
source run, select that source and supply its `--run-dir` with `--resume`; completed compatible
stages are reused. A multi-source run with a custom parent can also repeat its command with `--resume`. `--retries`
is the total attempts per stage (default: three). Context-capacity errors are retained in
the run state, with unknown capacity left unknown.

To request a saved question count per mode, use `--questions-per-mode` instead of
`--questions`. With `--targets-from`, this consumes only the previous run's target
pool, optionally filtered by `--langs` and `--modes`, until each mode reaches the
requested count. Historical run folders and results CSVs are supported. Short or
skipped batches advance to the next existing target; an exhausted pool is reported.
Every generator processes the same document rounds until all generators meet the
quota or the pool is exhausted. Surplus candidates remain in the run state, while
the CSV contains up to the requested count. `--trace` also writes `llm_calls.json` and `trace.md`
with every prompt, response, retry, and parsed grade in the same run folder.

For legal sources, add `--decider-model generator` to let each generator choose its
generation prompt, or `--decider-model jev` to balance personas using independent
Jev yes/no suitability checks:

```bash
clir --domain legal qac generate --source eurlex un --questions 30 \
  --decider-model generator --generation-model gpt-5.6-luna --trace

clir --domain legal qac generate --source un --questions 30 \
  --decider-model jev --generation-model gpt-5.6-luna --trace
```

The decider sees the complete assembled generation payload. New Jev runs assess
each of six personas independently, using the probability threshold pinned in the
eligibility prompt. Among the Yes personas, the pipeline chooses the largest
deficit: `(assignments + 1) * weight / sum_weights - persona_count`. The default
weights in `[domains.legal.persona_weights]` in `clir.toml` are lookup 5, conceptual 5,
fact_pattern 4, source_finding 4, comparison 3, and claim_verification 3. UN uses
practitioner with weight 4 in place of fact_pattern; it maps to the `practitioners` prompts.

One choice per sampled article/block and query language is shared by every generator
candidate. Counts are per corpus run and count assignments once, regardless of
candidate counts or generation failures. These are long-term proportions: a target
with at least one Yes is always used, even when its sole eligible persona is already
overrepresented. No targets are resampled or discarded to balance the mix. An all-No
target makes no generation or verifier calls. Saved question proportions can differ
from assignment proportions when generation yields different numbers of candidates.

Reference-complete sampling remains mandatory for weighted routing. Imported target
plans are checked against the current reference-status indexes before model calls;
unsafe targets or missing verification data fail preflight. The source packets and
eligibility thresholds are preserved. `--questions` counts targets, so suitability
skips can reduce output. Do not combine the decider with `--modes` or
`--questions-per-mode`. Omit the flag to retain explicit mode selection. Reusing a
fixed-mode target plan with a decider deduplicates repeated target/language pairs.

Eligibility calls can run concurrently; persona choices are committed in sampled
order with seed-based tie breaking. Decisions, weights, policy version and count
snapshots are checkpointed in `run.sqlite`, including skips. Resume reconstructs
counts from committed choices without double counting and retains the saved weights.
Historical runs retain their recorded single-winner routing behavior.
CSV rows include the chosen persona, independent Yes probabilities, eligible personas,
threshold, weights and selection policy. The run summary reports assignment counts
and target shares. `--trace` includes raw requests, responses and shared selection
checkpoints linked to CSV rows by `decider_routing_task`.
Invalid decisions are retried and then recorded as failures;
they never silently select a fallback mode. Regrading preserves decisions and only
reruns the verifiers. The original batch module CLIs also accept `--decider-model`.

Jev uses `OPENROUTER_API_KEY`, model `~typesafe/jev-latest`, and
[OpenRouter's Decisions endpoint](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
(`POST https://openrouter.ai/api/alpha/decisions`). Its structured instructions and
criteria are sent directly; new Jev runs request independent `noul` probabilities
for the six personas. The templates are packaged in each legal prompt pack's `decider/` directory.

### Legal prompt versions in local MLflow

New legal generations require minimal, verbatim supporting passages under the
`legal_evidence_spans_v1` contract. Every mode returns source-tagged `evidence_spans`;
EUR-Lex covers every involved article or annex, and UN uses only its generation
target block. Code validates the visible source text and computes offsets. Grading
checks sufficiency and removable padding; entire articles are not valid excerpts.
Answers retain their quotation, translation, or source-identifier rules.

The active bundle is `legal-evidence-spans-v1-20261104`. Existing data and published
datasets are unchanged. To resume an older run, select its original pinned manifest
with `CLIR_PROMPT_MANIFEST`; the previous active snapshot is preserved at
`reports/prompt_versions/legal-decider-score32-optimized-20261007.json`. Saved legacy
candidates remain readable without evidence backfilling. Future exports preserve
every span and map UN block offsets into the complete source document.

The [MLflow Prompt Registry](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.register_prompt)
runs locally with SQLite and requires no account, API key, or server. Install its
optional dependencies with `uv sync --extra prompts`. The default database is
`.clir/prompts.db` at the workspace root; override it with
`CLIR_PROMPT_REGISTRY_URI=sqlite:////absolute/path/to/prompts.db` or `--registry-uri`.

After editing prompts, publish all EUR-Lex and UN templates and activate their
pinned snapshot for subsequent legal runs:

```bash
python -m clir_bench.core.prompt_registry sync \
  --bundle legal-next --label legal-next \
  --output reports/prompt_versions/legal-next.json
python -m clir_bench.core.prompt_registry activate \
  --manifest reports/prompt_versions/legal-next.json \
  --label production
clir --domain legal qac generate --source eurlex un --langs en --questions 100
```

`sync` includes all current legal templates: every language, decider, and supported
legacy mode, including the independent Jev eligibility prompts. Other languages
therefore remain available after activation. To publish a saved snapshot, use
`publish --prompts path/to/prompts.json` with the same bundle, label and output
arguments. The JSON maps logical keys, such as `un/generation/practitioner`, to
exact prompt text; supplemental evaluation prompts may be included.

Each publication records MLflow names, immutable version numbers, text SHA-256,
and exact text in a self-contained manifest. Repeating the same experiment alias
and text is idempotent. Changed text needs a new alias and manifest path; existing
experiment aliases cannot silently move. The SDK supports moving aliases, but
runtime reads pinned manifest text, so alias changes cannot change an ongoing run.

`activate` verifies every pinned version, sets the chosen deployment alias and
atomically writes `.clir/active_prompt_manifest.json`. Later legal loads discover
this file from the workspace root or its subdirectories. Explicit
`CLIR_PROMPT_MANIFEST=/absolute/path/to/manifest.json` selects a different run pin.
Missing prompt keys or invalid hashes fail rather than falling back to edited
local files. Jev JSON remains exact text until its native request is built.

Before first activation, package prompts remain available. For intentional
unregistered development, set `CLIR_PROMPT_SOURCE=local`; it bypasses the default
active snapshot and cannot be combined with an explicit manifest. Tests can use
`CLIR_PROMPT_SOURCE=local python -m pytest`. Keep immutable experiment manifests
with their runs and preserve `.clir/prompts.db` to retain registry history.

Activation updates aliases individually before replacing the local active
snapshot; retry activation if interrupted. Existing active runs retain their
snapshot. Normal publication does not activate a bundle automatically.

For a paired prompt experiment, `python -m clir_bench.domains.legal.qac.screening`
accepts `--selection previous/run/selection.json --prompt-manifest version/manifest.json`
and an independent `--output` directory. It rebuilds and verifies the exact saved source
payloads before making calls, then runs both deciders and all eligible generation modes.
`screening_analysis` exports the native results. `screening_repair --recovery-only`
can recover schema-only failures from recorded responses without new model calls;
repairs retain pinned prompts and their own audit log.

`screening_regrade` replays saved candidates and source packets with a chosen verifier.
By default it also reuses the recorded rubric. To evaluate updated verifier prompts,
pass `--verifier-prompt-manifest path/to/manifest.json`; only the system messages
are replaced. The original generation/decider provenance and the new verifier
manifest are recorded separately. Always use a separate output directory for a
new rubric. Scores from different rubric versions may use different standards
even when the numeric range is unchanged.

The current English verifier rubric, `legal-clir-exceptional-r3`, treats 4 as
strong or fully correct and reserves 5 for specifically justified exceptional
execution. When numerical fidelity is inapplicable, it uses 4 as a compatibility
value and explains this in the reason. Totals retain the /40 schema; compare
audit outcomes separately and identify the rubric when comparing older scores.

To evaluate each persona independently on a completed screening selection:

```bash
python -m clir_bench.domains.legal.qac.screening_eligibility previous/run \
  --output reports/decider_screening/jev_eligibility \
  --prompt-manifest .clir/active_prompt_manifest.json
```

This sends each exact recorded source packet to Jev with six independent `noul`
questions from `decider/jev_eligibility.json`. The current prompt forecasts whether
the best generated question in each mode will score above 31/40 under
`legal-clir-exceptional-r3`. Its locally versioned `routing_policy` sets the default
yes-probability cutoff to 0.55 (inclusive); `--threshold` overrides it. Historical
prompts without a policy use 0.5. The local policy is omitted from the native API
request. This is a prediction before generation, not a guarantee of the realized
grade. No generation or verifier calls are made. The separate output
directory stores resumable calls, a `decisions_only.csv` (one row per
document), a `decisions.csv` row per document/mode, and optional comparisons with
all five existing blinded verifiers. `questions_with_jev_and_grades.csv` preserves
the original questions, grades, and reasons alongside the new decisions.
An observed passing candidate demonstrates feasibility; a failed sampled batch
does not establish that no valid question could be generated. The original
single-choice router remains available for comparison.

For a numeric-cutoff comparison with saved Gemini grades, use:

```bash
python -m clir_bench.domains.legal.qac.screening_cutoff \
  --decisions run/decisions.csv --grades previous/verifier_grades.csv \
  --output run --probability-threshold 0.55
```

It compares the maximum numeric total per document/mode with the strict rule
`best_total > 31`, treats generation skips as negative observations, and reports
audit validity separately. Optional `--split` supplies document-level development
and validation assignments. Threshold proposals use development documents only;
`--selection-objective low_rejection --minimum-high-retention 0.5` favors rejecting
low-scoring modes while retaining at least half the high-scoring development modes.
`--documents` on the Jev replay command restricts calls to an explicit JSON list of
corpus/document IDs for development experiments. Keep validation documents out of
prompt and threshold selection.

`python -m clir_bench.domains.legal.qac.experiment_evaluation` compares completed runs
using one pinned, mode-blind rubric; identical pairs are graded once across versions.
`reports/prompt_experiments/compare_versions.py` writes a standalone HTML report and
paired CSV summaries. Its optional `--mlflow-registry-uri` logs aggregate experiment
metrics alongside the registered prompts. See each command's `--help` and the
[September 2026 experiment design](reports/prompt_experiments/legal_20260930/experiment_design.json)
for the frozen sample, models, criteria and limitations. The
[experiment findings](reports/prompt_experiments/legal_20260930/findings.html)
record the completed reruns, controlled router comparisons, and current evaluation status.

```bash
clir --domain legal qac generate --source eurlex --langs en \
  --modes lookup fact_pattern --questions-per-mode 15 \
  --targets-from data/legal/eurlex/qac/eurlex_luna30_en \
  --generation-model gpt-5.6-luna --verifier-model google/gemini-3.8-flash \
  --run-dir data/legal/eurlex/qac/modes15_new_verifiers --trace
```

Any command's `--help` lists its options; `--dry-run` exists wherever something is written
or published.

## Running models

Encoding a corpus needs a GPU and can exhaust a workstation, so it is opt-in. The normal
path is `clir eval plan`, which writes the exact command and an sbatch script; the run
itself needs `--allow-local` on the machine that should do the work. Analysis reads saved
predictions, so every breakdown, metric and plot can be recomputed without touching a model.

## Layout

```
src/clir_bench/
  core/         domain-independent stages: corpus, qagen, grading, publish, ingest, runs
  evaluation/   retrieval harness, metrics, model loading
  analysis/     per-question, confusion, rescoring, tables (reads saved predictions)
  cli/          command groups
  domains/
    chemistry/   schema, vocabulary, attribution, sources, prompts, benchmarks
    legal/       structured legal corpora, source prompts, batch pipelines, QAC runs
data/           corpora and artifacts (gitignored)
data/legal/eurlex/qac/       EUR-Lex question-generation runs
data/legal/un_parallel/qac/  UN question-generation runs
data/legal/qac/              preserved original combined exports (historical provenance)
reports/runs/   evaluation runs, each self-describing
attic/          one-off scripts, frozen with provenance notes
docs/           ARCHITECTURE.md, MIGRATION.md, licensing and pipeline notes
```

`core/` may not import from `domains/` — a domain reaches the core only as data on its
`DomainSpec`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the contract and
[docs/MIGRATION.md](docs/MIGRATION.md) for the mapping from the previous repo.

## Data licensing

Patent text comes from third-party sources with their own terms, and each dataset published
from here carries the attribution of the source it was actually built from — Google Patents
Public Data is CC BY 4.0 and requires attribution; EPO bulk data has its own conditions.
Publishing a source with no declared attribution is a hard error rather than a default. See
[docs/DATA_LICENSE_AND_CONFIRMATION.md](docs/DATA_LICENSE_AND_CONFIRMATION.md).
