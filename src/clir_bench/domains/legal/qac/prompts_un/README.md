# UN Parallel Corpus prompts

Prompt pack for the UN Parallel Corpus source (`data/legal/un_parallel/`), where
the retrievable unit is a **block** — a contiguous run of lines of one UN
document, packed to a size window at English paragraph boundaries. Junk lines
(mastheads, TOC rows, vote rosters, adoption formulas) act as block boundaries
and fall into the gaps between blocks; section headings open the block that
follows them, so a block carries its section title. It exists as
a separate pack rather than an edit to `prompts_eurlex/` because the central
rule of that pack is **inverted back** here, and because the corpus mixes
genres (resolutions, reports, letters, meeting records) that EU legislation
does not have.

Loaded with `PromptPack("clir_bench.domains.legal.qac.prompts_un")`. Nothing in
`core/` knows about any of this.

    generation/technical/{en,de,fr,es,zh}.txt
    generation/descriptive/{en,de,fr,es,zh}.txt
    generation/conceptual/{en,de,fr,es,zh}.txt
    verifiers/{faithfulness,technical,descriptive,conceptual}_batch.txt

Each language file is a full-file translation with the same convention as
`prompts_eurlex`: the literal markers, JSON keys, and enum values stay in
English verbatim; only the explanatory text is translated. The production
question languages are en, fr, es, zh (the batch driver's default): for
fr/es/zh the payload carries the question-language text of the target and
context blocks (read from the 6-way files by line range) alongside English.
German is not a UN language and is not generated; its prompt files exist as
complete translations should a cross-language run ever be wanted, but no
default run uses them.

## The payload contract

The prompts describe — and therefore pin — the user message the future
`un_context` renderer must emit (the analog of `eurlex_context.render_payload`):

    ### TARGET BLOCK — write the questions about THIS text
    [EN] Document: S/2001/535
      Title: Letter dated 1 June 2001 from the Permanent Representatives ...
    <block text>

    ### REFERENCED DOCUMENTS — other documents CITED by the target block, supplied so you can UNDERSTAND those citations. Context only: never a source of answers, never the subject of a question.
    [EN] Reference: S/RES/1559(2004) — Resolution 1559 (2004) (cited paragraph 3)
    <the cited document's FULL text on whole-fit runs (reference_chars=None);
     otherwise the cited paragraph's block or the opening block, capped at 1500 chars>

Referenced documents are rendered in **every payload language**, one `[XX]`-tagged
block each, exactly like the target block: a French question sends the cited
document in French and in English. The other-language rendering is the block's
own line range read from that language's 6-way file, so it is the same text, not
a translation. A language is emitted only when every block of the cited document
is available in it -- a partial document would mislead -- so English (the pivot)
is always present.

    ### DOCUMENT CONTEXT — surrounding text of the SAME document, supporting context only
    <document opening + neighbouring blocks, or the whole document when it fits the budget>

Both `###` markers are literal strings. Metadata lines are `Document:` (the UN
document symbol) and `Title:` (the document's first content line, masthead
stripped); they get the same "context, not content" treatment as EUR-Lex's
`Act:`/`Location:` lines. Non-English payload variants add the question-language version of each
block alongside English, as in the EUR-Lex flow.

## What changed vs. `prompts_eurlex`, and why

### 1. Two kinds of supporting material — both understanding-only

The payload carries **resolved references** (instruments the target block
cites, found by `un_references.py` regex extraction against the corpus's own
symbol index) and a section EUR-Lex never had — the **surrounding document**.
Unlike EUR-Lex, where a referenced article may *complete* an answer, here
**neither section may ever contribute answer substance**:

- **REFERENCED DOCUMENTS** exist so the model understands what the block's
  citations refer to (what kind of instrument resolution 918 (1994) is, what
  subject it concerns) — the way a footnote helps a reader. A question whose
  answer sits behind a citation is discarded *even when the cited document was
  supplied*; a question answerable from a reference alone is off-target.
- **DOCUMENT CONTEXT** resolves "the Mission", "the present report", "the
  reporting period".

The single rule is the **BLOCK TEST**: the answer must be producible from the
target block alone, with both supporting sections used only to resolve
referring expressions and citations. Enforcement is layered: the faithfulness
verifier caps GROUNDING at 2 when any answer substance comes from a reference
or the context; the quality verifiers cap off-target questions
(`off-target-reference`, `off-target-context`); and `un_batch.pick_best`
refuses to keep any candidate with `faith_grounding < 3` as a target's best
question — a target with no grounded candidate yields no question (rejects
remain auditable in the all-candidates file).

There is consequently no `documents_involved` field. Output is
`{question, answer, question_type}` (technical) or `{question, answer, framing}`
(conceptual).

### 1b. Two technical flavours: `technical` and `descriptive`

`technical` allows official identifiers in the question ("resolution 918
(1994)") — realistic known-item search. `descriptive` forbids them and requires
the instrument to be named by **organ + year + subject** ("the 1994 Security
Council resolution expanding the UN mission in Rwanda"). An identifier is a
language-invariant string that a retriever can match without any cross-lingual
or conceptual work; describing the instrument turns a lookup into a real retrieval
problem — which is what the referenced documents make possible. Both are
fact-extraction modes: same eight categories, same `question_type` field, same
quality columns (`quality_keys('descriptive')` returns the technical keys). Only
the quality verifier differs — `descriptive_batch.txt` adds an IDENTIFIER-LEAK
check (a number in the question caps SPECIFICITY at 2). Faithfulness is shared
and unchanged: the answer must still live wholly in the target block.

### 2. Naming for retrievability is promoted to the top rule

The corpus is institutionally repetitive: thousands of resolutions authorize
troop levels and extend mandates in identical language. Both generation prompts
carry a dedicated NAMING FOR RETRIEVABILITY section ("the most important rule
of this task"): every question must name the situation, country, mission, body,
or instrument; never "this resolution" / "the present report" / a bare
paragraph number. Technical mode may use identifiers from the metadata
(`resolution 918 (1994)`); conceptual mode must anchor with proper nouns and
subject matter instead (identifiers are an identifier-leak there). The quality
verifiers back this with an ANCHOR CHECK (`no-anchor`, specificity ≤ 2) and the
conceptual BOILERPLATE CHECK.

### 3. Two genre rules EU legislation never needed

- **Preambular vs. operative** — resolution preambles ("Reaffirming...") are
  framing, not answer material; questions target operative or substantive
  content.
- **Attribution** — summary records, letters, and reports state what a speaker
  or body *said*. Questions and answers must preserve attribution ("According
  to the Secretary-General's May 1994 report...") and never present an
  attributed claim as established fact. Enforced in the faithfulness verifier
  (ATTRIBUTION RULE, grounding cap 2) and the quality verifiers
  (`missing-attribution`).

### 4. Categories and framings

Technical mode targets eight UN categories (`question_type` slugs):
situation_scope_or_coverage, operative_action,
sanction_condition_or_consequence, date_deadline_or_mandate,
quantity_force_or_finance, reporting_monitoring_or_verification,
finding_event_or_assessment, actor_body_or_procedure. Relative to EUR-Lex,
"Conditions" and "Enforcement" merge into the sanctions category (in UN
practice one machine), freeing a slot for **Findings, Events & Assessments** —
the category reports live on.

Semantic mode uses three framings (`framing` values): **situation**,
**response**, **stakeholder** — the UN reading of EUR-Lex's
problem/solution/application.

### 5. Verifiers

Adapted from the EUR-Lex versions; the JSON keys the graders emit are unchanged
(`grounding`, `precision`, `numerical_fidelity`; the five quality keys), so
`core.grading` needs no modification. The `articles_involved` provenance caps
are replaced by the context-leak cap; new failure_types: `off-target-context`,
`no-anchor`, `missing-attribution`. The unresolved-citation check survives
unchanged in spirit: UN texts cite other resolutions constantly and nothing
resolves them.

The `conceptual` mode was previously named `semantic`. Older mode names and pinned
MLflow manifests remain readable. The rename bundle is saved at
`reports/prompt_versions/un_conceptual_rename_20261006/manifest.json`, based on v4.
The question contract and grading criteria are unchanged by this rename.

## Additional retrieval modes

`comparison`, `claim_verification`, and `source_finding` each have standalone
`generation/<mode>/{en,de,fr,es,zh}.txt` prompts and matching
`verifiers/<mode>_batch.txt` quality rubrics. The six quality rubrics are extracted
verbatim from `ALL_VERIFIERS_EN.md`. They assess multilingual candidates while
writing grading explanations in English.
The generation text is extracted verbatim from the corresponding all-languages
Markdown source at the repository root. Existing corpus-language eligibility and
batch defaults are unchanged; select these modes explicitly with `--modes` or
use either decider, which now supports all six current personas.

The legacy scalar contract preserves `anchor` in these three modes. Comparison also stores `comparison_entities`
(as a JSON array in CSV) and `comparison_aspect`; claim verification stores
`claim` and `claim_status`. Both store `answer_is_translation`. Source finding
stores `source_identifier`, `evidence`, and `evidence_is_translation`; EUR-Lex
also stores `source_article`. Its answer identifies the source, while the
separate contiguous target evidence establishes the clue. The selected article
or block identifies the generation target. These annotations are retained in checkpoints
and CSV export, and passed through to graders and regrading. The shared faithfulness
rubric checks source-finding identity against target metadata and its separate
evidence against the target text.

Prompt bundles are immutable: an older active bundle will not contain these
modes. Use `CLIR_PROMPT_SOURCE=local` to run the edited files, or publish and
activate a new bundle with the existing prompt-registry workflow.


## Explicit evidence spans for future generations

Generation templates declare `EVIDENCE_CONTRACT: legal_evidence_spans_v1`. Every mode and
language returns `evidence_spans`: an ordered list of `{source_ref, language, text}` objects,
plus `answer_is_translation`. The first span supplies the answer passage or source-finding
clue; each further span supplies indispensable support from a different involved EUR-Lex unit.
Exactly one span covers each involved unit. EUR-Lex uses existing article/Cite-as keys;
UN uses `source_ref="target"` and never takes answer substance from context or references.

Quotes stay verbatim in the actual source language. They must be the smallest sufficient
continuous excerpts, with material qualifications retained. Whole-article copying and stitched
passages are invalid; no arbitrary token cap replaces the sufficiency/minimality judgment.
Answers retain their existing quotation, translation, or source-identity roles. Scalar evidence
fields, canonical IDs and character offsets are derived by code, not returned by the model.

Both verifiers inspect span coverage, provenance and semantic sufficiency and judge removable
padding through their existing checks and score dimensions. Their original grade schemas stay
unchanged. Legacy records without the versioned contract remain readable and are not backfilled.
Activate a new immutable prompt bundle after validation; editing these files alone does not
replace already pinned prompts or change existing generated QAs and Hugging Face datasets.
