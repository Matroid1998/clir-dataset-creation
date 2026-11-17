# chemistry prompts

This directory is where the domain knowledge lives. The Python code in
`clir_bench` is domain-agnostic plumbing — it batches passages, calls a model,
parses JSON, ranks candidates and writes CSV. What makes the output *chemistry
patent* questions rather than generic questions is entirely the text in these
files: the seven retrieval personas, scientific fidelity and evidence rules,
multilingual worked examples, and grading rubrics.

Changing a rubric here changes the benchmark. Treat these files as data, not
as prose to tidy up — they were tuned against real generator output and the
wording carries weight. The current batch calibration treats 4 as strong or
fully successful and reserves 5 for a specifically justified exceptional result.

Files are package data, addressed through `importlib.resources` via
`clir_bench.core.prompts.PromptPack`. Every directory carries an `__init__.py`
so the package resolves the same way installed, zipped, or from source.

The active generation files use `chemistry-generation-r7`; the seven persona
batch verifiers and shared faithfulness verifier use execution revision
`chemistry-verifier-r7`. They retain the `chemistry-clir-exceptional-r2` scoring
contract. The Jev request is assembled from `decider/jev_eligibility.json` and
`decider/eligibility_cases.json`; the loader adds the companion cases to the
per-persona instruction examples, deduplicating matching `case_id` values. The
eligibility threshold remains 0.5 and the 32/40 quality forecast remains
provisional.

The capped experiment runner is `experiments/chemistry_prompt_loop/run_round.py`.
It snapshots these prompts, uses at most 50 fresh documents per round and retains
all candidates, skips and stage failures. Independent evaluations use a fixed
contract and a fresh evaluator with no conversation history. Run artifacts and
ratings are stored under `reports/prompt_experiments/chemistry_loop_20261014/`.

The r2 rubric declares provisional calibration and should not be described as
calibrated. The single-candidate and legacy technical/semantic quality resources
retain their existing contracts.

## Layout

    generation/lookup/{en,de,fr,es,zh,ja,ru}.txt                 missing technical specification
    generation/process_retrieval/{en,de,fr,es,zh,ja,ru}.txt      connected route or workflow
    generation/conceptual/{en,de,fr,es,zh,ja,ru}.txt             explicit functional relationship
    generation/constraint_search/{en,de,fr,es,zh,ja,ru}.txt      solution meeting requirements
    generation/comparison/{en,de,fr,es,zh,ja,ru}.txt             two sides on a common dimension
    generation/assertion_verification/{en,de,fr,es,zh,ja,ru}.txt scoped assertion check
    generation/source_finding/{en,de,fr,es,zh,ja,ru}.txt         publication identity from a technical clue
    decider/jev_eligibility.json              independent suitability instructions for all seven personas
    decider/eligibility_cases.json            bounded examples merged into the Jev instructions
    verifiers/faithfulness_{batch,single}.txt   is the answer grounded in the passage?
    verifiers/<persona>_batch.txt              quality and contract checks for each new persona
    verifiers/technical_{batch,single}.txt     legacy grading and single concept queries
    verifiers/semantic_batch.txt               legacy semantic grading
    concept_query/{en,de,fr,es,zh,ja,ru}.txt          alias-graph: question whose answer IS the concept
    concept_query_with_term/{en,de,fr,es,zh,ja,ru}.txt alias-graph: same, but must use a given surface form verbatim
    code_switch/nonchem_swap.txt                picks a non-chemistry word to swap (robustness control)

`PromptPack.generation(mode, lang)` reads `generation/<mode>/<lang>.txt`;
`.faithfulness(arity)` and `.quality(mode, arity)` read from `verifiers/`;
everything else is reached with `.custom(...)`.

Japanese and Russian variants retain the English contract instructions, with
native output-language instructions and localized teaching outputs. Teaching
source packets and evidence quotations remain unchanged; translated example
answers carry `answer_is_translation=true`. Actual same-language source answers
are copied verbatim, as required by the existing evidence contract.

## Generation contract

The active chemistry source is
[`BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs`](https://huggingface.co/datasets/BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs),
addressed by the `gp` CLI key. Import retains only `pair_ok=true` records whose
`judge_verdict` is `exact` or `high` and whose two endpoint languages are supported.
Each normalized document contains one endpoint's complete cleaned abstract.
Upstream publication and group identities, revision and accepted direct pair
links remain available as provenance. The group identity is not a relevance
shortcut: accepted pair links need not form a complete graph across languages.

All seven personas are enabled by default for every chemistry generation plan.
Use `--modes` to select a subset; `technical` and `semantic` are no longer
generation modes. Each call returns one to three candidates, or one
`skip_reason` object when the source cannot support that persona. An abstention
produces no candidate rows and does not call the graders. Malformed output
raises an error and is retried under the configured generation retry policy.

The `balanced`, `uniform` and `coverage` plans select existing source documents
in `en`, `de`, `fr`, `es`, `zh`, `ja` or `ru`. The selected document's language determines
the generation prompt and question language. `--langs` restricts which source
documents may be selected. Generation, both graders and the eligibility decider
receive only the selected document's text and publication identifier.

The parser in `../candidates.py` checks exact field sets, types and enums.
It preserves evidence verbatim and retains substring mistakes for review.
Scientific support, literal annotation accuracy, evidence provenance and
translation fidelity are checked by the verifiers. Questions, evidence,
evidence language, translation flags and all
persona annotations survive CSV export and regrading; arrays and booleans use
JSON cell encoding. Publication identifiers accompany the source passage for
source-finding. The CSV also preserves `source_document_id`; regrading resolves
that same source document and supplies only its text to the verifiers.
Best-candidate selection retains one result per source document, question
language and persona. Distinct source documents remain separate even when
their publication number and language are identical.

## Batch verifier contract

The shared faithfulness rubric scores `grounding`, `precision` and
`numerical_fidelity`, each out of 5. It checks the actual question, answer,
evidence, assertion verdict and publication identity against the selected source
document, including connected evidence spans. When numerical fidelity is
inapplicable, the rubric assigns 4 as a compatibility value and explains this in
the reason; inapplicability does not earn exceptional credit.

Every quality rubric has five dimensions, each out of 5. The first three are
specific to the assigned persona; all seven also score `focus_and_economy` and
`linguistic_quality`.

| Persona | First three quality dimensions |
|---|---|
| `lookup` | `technical_search_realism`, `standalone_anchoring`, `information_value` |
| `process_retrieval` | `process_search_realism`, `start_goal_anchoring`, `route_completeness` |
| `conceptual` | `relationship_search_realism`, `subject_anchoring`, `explicit_relationship` |
| `constraint_search` | `requirement_search_realism`, `task_and_constraint_anchoring`, `candidate_fit` |
| `comparison` | `comparative_search_realism`, `two_sided_anchoring`, `common_dimension_value` |
| `assertion_verification` | `checking_search_realism`, `assertion_scope`, `verdict_decisiveness` |
| `source_finding` | `disclosure_search_realism`, `technical_clue_anchoring`, `clue_identity_fit` |

Quality responses contain `candidates` and a separate `batch_diversity`
diagnostic. Each candidate contains exactly `index`, `candidate_id`, `scores`,
`score_notes`, `checks` and `problems`. The score and note maps use the five
persona-specific keys. The checks are exactly `mode`, `support` and `metadata`,
with `pass`, `fail` or `uncertain` values. Every failing or uncertain check needs
a matching prefixed explanation in `problems`. A favorable total cannot replace
these checks, and batch diversity never changes individual scores.

Faithfulness returns a JSON list whose entries contain exactly `index`,
`grounding`, `precision`, `numerical_fidelity` and `reason`; it does not use the
quality response wrapper. Both stages preserve input order. Every quality score
has a criterion-specific note, and a 5 requires the prefix `Exceptional evidence:`.
Each faithfulness 5 needs the corresponding `Grounding exceptional:`,
`Precision exceptional:` or `Numerical fidelity exceptional:` phrase in its reason.

Scores may be null only when missing input makes a whole criterion unassessable.
Known errors and missing submitted answers are assessable failures. The pipeline
computes faithfulness /15, quality /25 and combined /40 totals outside the
prompts; a missing component leaves the corresponding total null rather than
rescaling it or substituting zero. Compare these rubric scores and audit outcomes
separately from historical runs: an entirely correct ordinary result can earn
4 in every dimension without an alleged defect.

CSV export and regrading retain raw verifier responses, `quality_batch_diversity`,
`quality_score_notes_json`, `quality_checks_json`, `quality_problems_json`,
`quality_mode_status`, `quality_support_status`, `quality_metadata_status`,
`quality_audit_status`, `quality_rubric_version` and
`faithfulness_rubric_version`. Best-candidate selection excludes null, blank or
nonfinite combined totals. Candidates with numeric totals rank by score; the
audit checks remain available for review and do not add a separate selection
gate.

## Eligibility decider

`decider/jev_eligibility.json` contains the supplied chemistry eligibility
instructions. The companion `decider/eligibility_cases.json` supplies bounded
teaching cases, grouped by persona at request time and merged by `case_id`. The
effective request serves the same role as the
legal eligibility template: one independent `noul` question per persona. A patent
can support none, one, several or all seven personas. The prompt's 32/40 target
forecasts possible generation quality and is not a measured candidate score.

Enable routing with `clir --domain chemistry qac generate --source gp
--decider-model jev`. The loader replaces the template's state with the same
selected document text and publication identifier supplied to generation, and
uses `~typesafe/jev-latest` through the existing Decisions transport. It excludes
old candidate and grading columns. A response must contain exactly seven finite
probabilities between 0 and 1; they are not normalized across personas. Malformed
responses are retried and remain failures if retries are exhausted.

The probability cutoff defaults to 0.5. A future `routing_policy` in the template
may set `probability_threshold`; local routing policy is never sent to the API.
The current supplied chemistry template has no calibrated routing policy.
The pipeline selects the eligible persona with the largest deficit against its
target share, using equal weights by default and a seeded tie-break. Configure
other positive weights in `[domains.chemistry.persona_weights]`. A sole eligible
persona is always used even if it is overrepresented; an all-No result is skipped.

Eligibility is evaluated once per selected source document, with separate
screening for distinct documents sharing a publication number or language.
Assignments follow sampled order regardless of concurrent completion order and
are counted before generation. `--questions` caps distinct source documents,
and `--limit` further restricts the sampled plan. `--modes` cannot be combined with
the automatic decider. Fixed-mode generation remains available without the flag.

The results CSV includes `decider_*` provenance columns. The sibling
`<output_stem>_eligibility.json` audit saves the prompt snapshot and hash, input
hashes, all probabilities and assignments, including skipped targets. `--append`
adds a new run entry to that audit. Regrading retains the original decider metadata.

## The batch / single split is deliberate — do not merge these

The `_batch` and `_single` verifier prompts must stay separate because they have
**different output contracts**, and the parsers downstream depend on it:

| | input | output |
|---|---|---|
| `faithfulness_batch.txt` | selected source document + actual candidates | a JSON **list** matching the input count, each carrying an `"index"` |
| current persona `*_batch.txt` | selected source document + actual candidates | a JSON **object** containing `"candidates"` and `"batch_diversity"` |
| `*_single.txt` | one passage + **one** QA pair | a single JSON **object**, no `"index"` |

The batch form exists because the main QAC pipeline generates up to three candidates
per document and ranks them, so it grades all three in one call — one request
instead of three, and the grader sees the candidates in a fixed order. The
single form exists for the alias-graph path, which produces exactly one query
per concept and has nothing to rank.

The instruction text differs accordingly and not only in arity. The batch
prompts carry independence guardrails because grading candidates side by side
invites contrast effects. The single prompts instead judge the pair on its own
merits. Legacy technical and semantic batch rubrics retain their historical
three-candidate wording for old datasets.

There is no `semantic_single.txt`: the alias-graph path only ever grades
technical-style concept queries.
