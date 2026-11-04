# Publish generated QAs and their related documents

`clir publish generations` preserves every candidate from the specified runs, including all
models, personas/modes, languages, answers, evidence, grading and provenance. It reads legal
SQLite outcomes (not potentially stale CSV exports) and the explicitly listed chemistry CSVs.
The input manifest excludes archive, retry and best-only files.

Prepare one dataset per corpus:

```bash
clir publish generations --manifest docs/generation_export.json --corpora eurlex \
  --output data/hf_export/eurlex-generated-qa
clir publish generations --manifest docs/generation_export.json --corpora un \
  --output data/hf_export/un-generated-qa
clir publish generations --manifest docs/generation_export.json --corpora chemistry \
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

## Dataset contents

| Configuration | Contents |
| --- | --- |
| `qas` (default) | Q&A, complete primary `document_text`, persona/mode, languages, generator, evidence, source IDs, scores, and complete original metadata |
| `corpus` | Unique related documents and available/directly accepted translations, with original source metadata |
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

This release includes **related documents**, not the complete retrieval haystack. EUR-Lex
source units are whole articles; chemistry uses title plus abstract; UN reconstructs the whole
document from its original aligned line range, including material omitted from generation
blocks. UN block IDs and generation passages remain provenance. No 512-token chunks are made.
`--corpus-scope full` optionally exports the full local corpora in the configured languages.

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
