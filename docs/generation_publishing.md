# Publish generated QAs and source corpora

`clir publish generations` preserves every candidate from the specified runs, including all
models, personas/modes, languages, answers, evidence, grading and provenance. It reads legal
SQLite outcomes (not potentially stale CSV exports) and the explicitly listed chemistry CSVs.
The input manifest excludes archive, retry and best-only files.

For a new release, prepare one dataset per corpus. Use `--corpus-scope full` for
the retrieval corpus; the default `referenced` scope contains only QA-associated documents:

```bash
clir publish generations --manifest docs/generation_export.json --corpora eurlex --corpus-scope full \
  --output data/hf_export/eurlex-generated-qa
clir publish generations --manifest docs/generation_export.json --corpora un --corpus-scope full \
  --output data/hf_export/un-generated-qa
clir publish generations --manifest docs/generation_export.json --corpora chemistry --corpus-scope full \
  --output data/hf_export/chemistry-generated-qa
```

Inspect each directory's `README.md`, `report.json` and Parquet files, then upload:

```bash
clir publish generations --upload-from data/hf_export/eurlex-generated-qa \
  --repo MehdiAstaraki/eurlex-generated-qa --public
clir publish generations --upload-from data/hf_export/un-generated-qa \
  --repo MehdiAstaraki/un-generated-qa --public
clir publish generations --upload-from data/hf_export/chemistry-generated-qa \
  --repo MehdiAstaraki/chemistry-generated-qa --public
```

Use `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the environment or project `.env`. Preparing
files requires no token or network access. To prepare and upload in one invocation, add
`--push --repo OWNER/DATASET --public` to a preparation command. Repositories are private
unless `--public` is supplied. Uploading verifies the prepared files' hashes and refuses to
mix a release with existing remote dataset shards; choose a fresh repository for a new release.
An existing output directory is never overwritten. Re-running an interrupted local preparation
is safe because an incomplete bundle is not installed at the requested output path.

The JSON manifest's paths resolve relative to its `root`, which is relative to the manifest
file. List chemistry generator identities explicitly: those CSVs omit the generator model.
`--corpora` selects one or more entries; omitting it combines all entries in a single dataset.

## Expand an existing published corpus

The original November 4 exports used `referenced` scope. Expand an existing repository
without rewriting its published questions, answers, evidence or metadata:

```bash
clir publish expand-corpus \
  --from-export data/hf_export/eurlex-generated-qa \
  --manifest docs/generation_export.json --corpus eurlex \
  --output data/hf_export/eurlex-generated-qa-full-20261104

clir publish expand-corpus \
  --from-export data/hf_export/eurlex-generated-qa \
  --upload-from data/hf_export/eurlex-generated-qa-full-20261104 \
  --repo MehdiAstaraki/eurlex-generated-qa
```

For the UN document-type collection across all six languages:

```bash
clir publish expand-corpus \
  --from-export data/hf_export/un-generated-qa \
  --manifest docs/generation_export.json --corpus un \
  --output data/hf_export/un-generated-qa-genres-20261104

clir publish expand-corpus \
  --from-export data/hf_export/un-generated-qa \
  --upload-from data/hf_export/un-generated-qa-genres-20261104 \
  --repo MehdiAstaraki/un-generated-qa
```

Preparation checks that all previously published source records remain identical and copies
the existing QA, evidence and source-qrels shards byte for byte. If new language versions
become available, their direct source associations are appended in separate qrels shards.
The upload verifies the remote release against the original manifest and replaces the
corpus in one commit, guarded against concurrent remote changes. Repository visibility is
preserved. Only the selected repository is updated; each corpus is prepared/uploaded separately.

## Dataset contents

| Configuration | Contents |
| --- | --- |
| `qas` (default) | Q&A, complete primary `document_text`, persona/mode, languages, generator, evidence, source IDs, scores, and complete original metadata |
| `corpus` | Complete selected local corpus with `full` scope, or related sources and translations with `referenced` scope |
| `evidence` | Supporting text, its origin (`evidence` or `answer`), source ID, match status and exact-match character offsets |
| `source_qrels` | Generation-derived source associations and direct translations, not chunk-level judgments |

Load the QAs with:

```python
from datasets import load_dataset
qas = load_dataset("MehdiAstaraki/eurlex-generated-qa", "qas", split="train")
```

`metadata_json` preserves the entire input row without dropping persona-specific or grading
fields. `persona_metadata_json` exposes routing and persona details together. Source IDs are
language-aware (`eurlex:ELI:LANG`, `un:DOC_ID:LANG`); chemistry retains its existing IDs.
Candidate IDs remain stable across input relocation and grading updates. Repeated identical
candidates are counted and skipped; conflicting metadata for the same candidate aborts export.
Identical questions from different generator models remain distinct candidates.

`multi_article` detects multiple involved article ELIs and retains a true generator declaration.
The raw declaration, resolved article list, article count, cross-act flag and declaration
conflicts are separate fields. Supplied contextual references do not count as involvement.
Unresolved explicitly involved articles abort export, so links cannot silently disappear.
Translation partners do not count as additional involved articles. UN and chemistry have
`multi_article=false` and a null declaration because that annotation does not apply to them.

Exact supporting-text matches receive the first matching span in the original source unit,
using Unicode character offsets with an exclusive end. Whitespace-normalized and unmatched
support have null offsets; they are retained. `supporting_text_origin` makes answer-as-support
fallback explicit without changing the raw evidence field. Matching is not a semantic judgment.

Future legal generations use `evidence_contract=legal_evidence_spans_v1` and an ordered
`evidence_spans` list. Exports preserve every source-tagged span and write one `evidence`
configuration row per span. The span list retains original article/block `start`/`end` and
adds `source_unit_id` plus `document_start`/`document_end` for the exported corpus. UN mapping
uses the recorded block line range and verifies the original block hash, so a repeated quote
in another block cannot receive the span accidentally.

The first span supplies the scalar evidence fields. `support_start`/`support_end` refer to
`support_source_unit_id`, which can identify a supplied reference or another source language;
the QA's `source_unit_id` and `document_text` still identify its primary target. All evidence
languages are loaded even if the manifest omitted them. Missing sources, changed text,
invalid offsets or incomplete multi-article coverage abort an export. Legacy rows keep their
existing behavior and receive no automatic evidence backfill.

Full scope includes the complete locally collected corpus selected by the manifest, not every
document available upstream. The current EUR-Lex selection contains **340,833 article/annex
language records**: 266,494 articles and 74,339 annexes in English, French, German and Spanish.
These come from 14,580 acts processed from the original MultiEURLEX collection; they do not
cover all 35,395 originally downloaded acts. The `unit_types` filter excludes recitals.

Chemistry uses the complete local accepted-pair corpus of title/abstract records. UN
reconstructs each whole document from its original aligned line range, including material
omitted from generation blocks. Its `document_genres` selection uses the generation pipeline's
type classifier and excludes agenda stubs, journals and corrigenda. It includes **14,689 unique
documents**: 6,572 resolutions/decisions, 6,971 meeting records and 1,146 letters/notes verbales.
Across Arabic, English, Spanish, French, Russian and Chinese this produces **88,134 records**.
It is the complete selected type collection, not all 86,307 documents in the six-way corpus.
Generation-only passage-quality, context-length and reference-completeness restrictions do
not limit this retrieval corpus. Omitting `document_genres` restores the unfiltered loader;
an explicitly empty or invalid list is rejected.

UN block IDs and generation passages remain provenance. No 512-token chunks are made.

The `train` split is a storage convention. Construct evaluation splits by source and related
translations, across all generators, before training or evaluating. Keep provenance-derived
source associations separate from evidence-aligned chunk relevance judgments.

## Current input snapshot

`generation_export.json` selects the October 2026 legal comparison and continuation runs,
and the ten current top-level chemistry comparison CSVs. At the initial export it contains
4,050 EUR-Lex, 4,691 UN and 1,950 chemistry candidates. All current EUR-Lex candidates declare
single-article involvement; multi-article support is exercised in tests for future inputs.

The existing `MehdiAstaraki/multilingual_GP` corpus at revision
`1037947810d434fbd07a737cb2d21283398b867b` contains none of the 200 chemistry generation source
IDs, so this exporter includes the matching local chemistry records instead of linking to that
older collection. Source attribution and terms are included in each dataset card.
