# EUR-Lex prompts

A variant of the `legal` prompt pack for acts whose **cross-references have been
resolved**. It exists as a separate pack rather than an edit to `legal/` because
the two disagree on a central rule, and that disagreement is the whole point.

Used by `clir_bench.domains.legal.qac.eurlex_generate`, which sends a four-block
payload (target, same-act articles, other-act articles, annexes) instead of a
single passage. Nothing in `core/` knows about any of this.

    generation/lookup/{en,fr,de,es,zh}.txt
    generation/fact_pattern/{en,fr,de,es,zh}.txt
    generation/conceptual/{en,fr,de,es,zh}.txt
    verifiers/{faithfulness,lookup,fact_pattern,conceptual}_batch.txt

The production question languages are en, fr, de, es (the batch driver's
default): the corpus has no zh act versions, so Chinese is not generated. The
zh prompt files exist as complete translations should a cross-language run
(question in a language the corpus lacks) ever be wanted, but no default run
uses them.

## The three modes

EUR-Lex supports `lookup`, `fact_pattern`, and `conceptual`. Retired
`technical`, `semantic`, and `descriptive` modes remain unavailable here.

| Mode | Information need | Extra fields |
|---|---|---|
| `lookup` | One precise legal point in a known substantive regime | `question_type`, `question_cited`, `instrument_short_name`, `anchor` |
| `fact_pattern` | How an explicit rule applies to a concrete case | `question_type`, `particulars` |
| `conceptual` | A stated regulatory problem, operation of a legal response, or a party's role/protection | `framing`, `anchor` |

All modes declare `articles_involved`. They have separate quality rubrics;
the runtime reads each rubric's five criteria and exports those scores.
The conceptual rubric scores search realism, anchoring/time, consequence,
lexical distance, and linguistic quality, alongside the shared faithfulness
rubric. The decider compares all three without a fixed persona preference.

Conceptual mirrors UN's situation/response/stakeholder framings but follows
EUR-Lex evidence rules: supplied references may complete the target's rule,
the target must remain necessary, and the answer is a sufficient contiguous
span. It requires neither a hypothetical client case nor a date on every
standing rule. It distinguishes statutory arrangements from their actual
implementation or effectiveness and does not assume current law. Substantive
institutional mechanisms qualify; pure boilerplate does not.

The new prompts and deciders are versioned in
`reports/prompt_versions/eurlex_conceptual_20261006/manifest.json`, built on
the active v4-derived bundle. Old bundles retain their original two-mode
routing vocabulary and exact text. Production source languages remain
EN/FR/DE/ES; adding the Chinese prompt does not add a Chinese corpus.

### Existing lookup and fact-pattern requirements

Both prompts carry the same three defences against the failure modes a
production run actually produced:

- **the sibling test** — EU legislation repeats the same provision across sister
  acts (the transposition clause in every directive; near-identical
  qualifying-holding rules across UCITS/AIFMD/MiFID/CRD/Solvency II; the same
  "all other companies" residual duty in every anti-dumping regulation). A
  question that would be an equally sensible question about a *different* act
  has no anchor and is discarded.
- **the boilerplate list** — transposition, application, entry-into-force,
  addressee, "binding in its entirety", bare repeal, committee-procedure and
  definitive-collection clauses are never asked about. An article made only of
  these returns a skip.
- **the informativeness test** — an answer that restates the question's own
  words or the pointer it was built from ("in relation to matters to which it
  applies") means the question is empty.

### Skipping

All three prompts answer a boilerplate-only article with

    [{"skip_reason": "transposition clause only"}]

rather than padding out three questions. `eurlex_generate.is_skip` /
`skip_reason` distinguish that from a malformed response; `parse_candidates`
returns no candidates either way, so the target simply contributes no rows.

## What changed relative to the `legal` pack, and why

### 1. The input is four labelled blocks, not one passage

The user message looks like:

    ### TARGET ARTICLE — write the questions about THIS article
    [EN] Article 4 — Placing on the market
    ...
    ### REFERENCED ARTICLES — supporting context only, cited by the target article.
    [EN] Article 3 — Covered products
    ...
    ### REFERENCED ARTICLES FROM OTHER ACTS — supporting context only, cited by the target article.
    [EN] Article 5 — Checks
      Act: Council Regulation (EC) No 21/2004 ...
      Cite as: 32004R0021:5
    ...
    ### REFERENCED ANNEXES — annexes cited by the target article, of this act or of another act ...
    [EN] ANNEX I
      Act: Directive (EU) 2019/904 ...
      Cite as: 32019L0904:anx_1
    ...

All four markers are literal strings emitted by `eurlex_context.render_payload`;
an empty block is rendered as an explicit "— none." line rather than omitted.
The third block holds articles of *other* acts that the target cites by name,
present only when `structure.resolve_external` could pin the cited act down to
one in the corpus (see that module: the year/number order is fixed by the
identifier's shape and never guessed). Each such article carries a `Cite as:`
key, `CELEX:number`, which is how the model refers to it. The fourth block holds
annexes the target cites — of this act or of another act in the corpus — each
with a `Cite as:` key of the form `CELEX:anx_<id>`, declared the same way.

Why separate rather than concatenate: given one undifferentiated blob the model
asks about whichever article reads most interestingly, which is usually not the
one we meant. All three prompts therefore say the question is *about* the target, and
add THE ONE-ARTICLE TEST — "could a reader answer this completely by reading
ONLY the referenced article, never having seen the target? If yes, discard it".
In a production run, *every* multi-article question generated failed that test,
so both prompts carry the real failures as worked examples.

### 2. The cross-reference rule is **inverted**

This is the substantive change. The `legal` pack says:

> **Self-Contained (No Unresolved Cross-References):** … If the substance of the
> answer lies in a provision, annex, or instrument that the passage merely cites
> but does not state, discard the question.

The EUR-Lex pack says instead:

> **Resolved Cross-References Are Allowed and Wanted:** … When that cited article
> is supplied — in the REFERENCED ARTICLES block, the REFERENCED ARTICLES FROM
> OTHER ACTS block, or the REFERENCED ANNEXES block — you MAY follow the
> citation and use its content to complete the answer. … If the answer's
> substance lies behind a citation that was NOT supplied, discard the question.

So the discard rule survives, but its trigger moves from *"is it behind a
citation"* to *"was the cited article actually supplied"*. Questions the old pack
threw away are now the interesting ones — they are the multi-article questions
the reference graph was built to enable.

### 3. `articles_involved`

Every candidate declares which articles a reader genuinely needs:

- answer wholly inside the target → `["4"]`
- answer completed by a followed reference → `["3", "4"]`
- answer completed by an article of another act → `["4", "32004R0021:5"]` — the
  `Cite as` key, never a bare number, so that a bare number can only ever mean
  the target's own act and nothing collides
- answer completed by an annex (of any act) → `["4", "32019L0904:anx_1"]` — annexes
  are always declared by their `Cite as` key
- the target article is always present
- an article merely *cited* by the target, whose content was not used, must **not**
  be listed

Each prompt carries the field being right in the single-article case, right in
the multi-article case, and wrong in **both** directions — under-declared (called
out as a scoring error *even though the answer is correct*) and over-declared
(listing a cited article that contributed nothing). A single example teaches
"list everything".

Identifiers of *other* acts are forbidden in the question in **both** modes: a
query anchored on another act's identifier, CELEX number or `Cite as` key
retrieves that act, not the target. Only `lookup` may name the target's own
instrument, and only in `question_cited`.

The field is a JSON key and stays untranslated in every language variant.

### 4. Output schema

    lookup       {"question", "question_cited", "instrument_short_name",
                  "answer", "question_type", "anchor", "articles_involved"}
    fact_pattern {"question", "answer", "question_type",
                  "particulars", "articles_involved"}
    conceptual   {"question", "answer", "framing", "anchor", "articles_involved"}

`parse_candidates` reads the extra fields **per mode**, so a field belonging to
the other mode cannot leak into a row — the two prompts are near-identical
siblings and a model that has seen both will occasionally emit the wrong one.
`instrument_short_name` is `null` unless a conventional short name is in
established use; a model told to write null sometimes writes the *string*, so
`_short_name` maps `None`, `"null"` and `"none"` alike to `""`.

All three modes write one CSV with one schema; the columns the other mode does not
emit stay empty. `particulars` join on `|`, not `,`, because a particular
routinely contains a comma ("40 tonnes placed on the Spanish market last year").

### 5. Verifiers

`faithfulness_batch.txt` is shared by all three modes:

- the input description covers all four blocks (including REFERENCED ANNEXES)
  and the `Cite as` keys, and the candidates it grades carry their declared
  `articles_involved`;
- the `CROSS-REFERENCE RULE` splits into case (a) *cited article was supplied* —
  following it is legitimate support, not inference — and case (b) *not supplied*
  — score at most 2;
- `GROUNDING` also checks provenance: **cap at 2** if `articles_involved` is
  wrong in either direction, **cap at 1** if the substance came from an article
  never supplied.

Each mode has its own five quality criteria and structured checks. The
conceptual rubric evaluates its framing and anchor without requiring lookup
renderings or fact-pattern particulars. The older modes retain these checks:

| check | `lookup` | `fact_pattern` |
|---|---|---|
| IDENTIFIER-LEAK | ✓ | ✓ |
| ANCHOR CHECK — no regime anchor, only generic actors | ✓ | |
| FACT-PATTERN CHECK — no situation, a bare slot | | ✓ |
| SIBLING TEST — would fit a different act equally well | ✓ | ✓ |
| TERM-SUBSTITUTION — a term of art swapped for a near-synonym | | ✓ |
| BOILERPLATE — the answer is a transposition/entry-into-force/addressee clause | ✓ | ✓ |

The faithfulness grader sees `question`, `answer` and `articles_involved`; the
quality grader also sees the mode metadata. Every
check above is judged from the question text itself rather than from the
generator's own `anchor` / `particulars` self-report.

Both also keep the EUR-LEX TARGET SCOPE rule: a question whose subject matter
lives entirely in a referenced article or annex — of the same act or of another
— scores at most 2 for retrieval quality, because it would retrieve the wrong
document.

## Translations

Each mode ships five full-file translations. The convention, shared with
`prompts_un`:

- **translated**: all prose, headings, rules, self-check items, output field
  descriptions, and the closing trailer — including anything that models the
  wording of the question the model must *output* (question shapes, anchor and
  particular specimens, the forbidden deictic phrases);
- **verbatim English**: the `###` block markers, the `Act:` / `Location:` /
  `Cite as:` keys, every JSON key and `question_type` value, all instrument
  identifiers and CELEX keys, the indented metadata specimen, and **the whole
  worked-examples block**, which is a sample of English input and output;
- the third paragraph declares the output language, and the trailer states that
  the article is supplied in that language *and* in English.

One line = one line across all five files (251 for `lookup`, 209 for
`fact_pattern`), which makes the examples block diffable byte-for-byte against
`en.txt` as a regression check.

## Division of labour on `articles_involved`

The LLM grader judges whether the declaration is *semantically* right. Code in
`eurlex_context.normalise_involved` checks that it is *structurally* valid —
articles the model was never sent are rejected outright, the target is inserted
if omitted, and surface variants (`"Article 3"`, `"3"`, `"3 and 4"`) are
normalised. Rubrics grade that kind of thing badly; a validator does not.

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
