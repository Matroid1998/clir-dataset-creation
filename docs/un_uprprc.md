# Newer UN documents with deterministic multilingual blocks

The UPRPRC adapter adds an optional source for UN question generation. It preserves
the existing UN corpus and defaults. Select the new source explicitly with
`--un-dataset /path/to/manifest.json`; existing runs continue to use their original
source unless they already recorded a new-source manifest.

The default preparation window is **2015–2025**, based on `publication_date`.
This is a target filter, not a promise that every year or document survives validation.
The manifest gives the accepted document/block counts and measured publication
ranges for documents and targets. Complete six-language documents outside this
window remain available as reference context.
Question targets contain **150–450 English whitespace words** and source passages
in all six UN languages: Arabic, Chinese, English, French, Russian and Spanish.
The existing question languages remain English, French, Spanish and Chinese.

## Source inputs and download

Preparation uses two pinned datasets:

| Dataset | Revision | Purpose |
| --- | --- | --- |
| [`bot-yaya/UPRPRC_FTXT_FILEWISE`](https://huggingface.co/datasets/bot-yaya/UPRPRC_FTXT_FILEWISE/tree/67ae6c42e9008777fea18be9a0461e628f4d5ab1) | `67ae6c42e9008777fea18be9a0461e628f4d5ab1` | Complete original documents, publication dates and per-language job numbers. |
| [`bot-yaya/UPRPRC_BILINGUAL`](https://huggingface.co/datasets/bot-yaya/UPRPRC_BILINGUAL/tree/536e033bf67872ba1f06ea44b6a582fb85772760) | `536e033bf67872ba1f06ea44b6a582fb85772760` | Published paragraph index links between English and the other five languages. |

Use the original bilingual paragraph coordinates. The separately published
`BLOCKWISE` dataset does not retain the required original paragraph locations.
Shared document IDs alone establish a document relation, not block correspondence.

Downloading and preparing are separate steps. The full inputs are large
(approximately 26 GB and 78 GB of compressed Parquet respectively); unit and
workflow tests use small local fixtures. The preparer does not download data or
call a model. Download to an input directory that is separate from the existing
active UN data. For example, using the installed Hugging Face client:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "bot-yaya/UPRPRC_FTXT_FILEWISE",
    repo_type="dataset",
    revision="67ae6c42e9008777fea18be9a0461e628f4d5ab1",
    allow_patterns=["*.parquet", "**/*.parquet"],
    local_dir="data/legal/un_uprprc/input/filewise",
)
snapshot_download(
    "bot-yaya/UPRPRC_BILINGUAL",
    repo_type="dataset",
    revision="536e033bf67872ba1f06ea44b6a582fb85772760",
    allow_patterns=["*.parquet", "**/*.parquet"],
    local_dir="data/legal/un_uprprc/input/bilingual",
)
```

The manifest hashes the supplied local input files and records the declared
upstream revisions. It does not attest that local inputs contain the complete
upstream snapshot; retain the pinned download provenance and read the measured
coverage as coverage of the supplied records.

The command accepts a local Parquet file or a directory of shards. Small offline
fixtures can use the Python `prepare_records(filewise_rows, bilingual_rows,
out_dir)` API with iterable records. Filewise `ftxt` and `job_numbers` lists use
the upstream order `ar, zh, en, fr, ru, es, de`. German is not part of the
six-language UN target contract.

## Prepare and audit a staged source

```bash
python -m clir_bench.domains.legal.un.uprprc prepare \
  --filewise data/legal/un_uprprc/input/filewise \
  --bilingual data/legal/un_uprprc/input/bilingual \
  --out-dir data/legal/un_uprprc/staged/2015-2025

python -m clir_bench.domains.legal.un.uprprc audit \
  data/legal/un_uprprc/staged/2015-2025/manifest.json
```

The output directory must be new. Preparation writes `sources.sqlite` and a
versioned `manifest.json` with source revisions, filtering configuration and
checksums. A failed preparation does not install a valid final bundle. Review
the report and audit results before using the manifest for generation. No
existing UN index, translations, saved run or published dataset is replaced.

The builder performs the following mechanical checks without an LLM:

1. Read paragraph slots using the original `text.split("\n\n")`, including empty
   slots. Validate each link's document identity, language job numbers, paragraph
   indices and exact source text round trip.
2. Join the five English pivot link sets into common multiway components. Every
   English paragraph in a target must have coverage in each other language.
3. Require ordered, noncrossing, contiguous source ranges in all six originals.
   Whitespace gaps are retained. A group that is indivisible in one language is
   indivisible in the others.
4. Pack only accepted groups into the English word window. Missing translations,
   invalid links and oversized indivisible groups are excluded from targets;
   the builder does not infer or repair their alignment.
5. Retain complete original texts and exact per-language character ranges plus
   SHA-256 hashes. Unmapped English passages remain context and cannot become
   generation targets.

These checks prove reproducible source correspondence and exact positions.
They do not independently prove that every upstream translation/alignment is
semantically correct. This workflow reuses the published links; it neither asks
an LLM to align passages nor uses an LLM to approve alignment.

## Generate, resume and regrade

Validate source reconstruction and target selection first:

```bash
clir --domain legal qac generate --source un \
  --un-dataset data/legal/un_uprprc/staged/2015-2025/manifest.json \
  --questions 10 --langs en fr es zh --modes lookup --dry-run
```

Generate in a new run directory after reviewing the source audit:

```bash
clir --domain legal qac generate --source un \
  --un-dataset data/legal/un_uprprc/staged/2015-2025/manifest.json \
  --questions 30 --langs en fr es zh --modes lookup \
  --run-dir data/legal/un_uprprc/qac/pilot
```

The normal generation and grading stages call the configured models. Block
definition, translated-block lookup and source reconstruction do not. Within a
prepared bundle, the target and its translations share a stable block ID. For a
non-English question, the existing prompt receives that language and English. Other original languages
remain available through the source index and for export.

Each result retains `source_locations` for all six languages. Those saved
character ranges and document hashes locate the corresponding block in each
exported full original without alignment or translation calls.

These examples select `lookup`, whose checked-in generation prompts cover all
four current question languages. Prompt availability still depends on the
selected mode and any pinned prompt bundle. The adapter does not add missing
`technical` or `descriptive` generation templates; select a supported existing
mode such as `lookup` for the pilot. For new-source selection, omitting `--modes`
uses `lookup`, `conceptual` and `practitioners`; legacy source defaults are retained.

The run pins the manifest and database identities and saves source payload
hashes. Results CSVs also retain the identity so a CSV copied away from its run
database can still be replayed safely. Resume and regrade inherit the original
source when `--un-dataset` is omitted, and refuse a changed source identity:

```bash
clir --domain legal qac generate --source un \
  --run-dir data/legal/un_uprprc/qac/pilot --resume

clir --domain legal qac regrade --source un \
  --input data/legal/un_uprprc/qac/pilot/results.csv \
  --run-dir data/legal/un_uprprc/qac/pilot-regraded
```

Only validated six-language target blocks can be replayed. Supplying a context
block ID, incomplete translation or stale source span fails before generation.
Generation preflight rechecks the selected block's saved paragraph alignment
proof and exact six-language source ranges, rather than trusting its eligibility
flag or database checksum alone.

## References, context and evidence

References are resolved from UN symbols across the prepared document collection.
Namespaced source IDs do not control genre or citation identity: the canonical
symbol does. A document with no eligible question block can still supply cited
context. The current requirements for complete references and whole-document
context fit still apply; extending the source does not bypass them.

Preparation can optionally add the old corpus as reference-only material.
Supply all three legacy paths together when creating a new bundle:

```bash
python -m clir_bench.domains.legal.un.uprprc prepare \
  --filewise data/legal/un_uprprc/input/filewise \
  --bilingual data/legal/un_uprprc/input/bilingual \
  --legacy-docs data/legal/un_parallel/blocks/docs_en.jsonl \
  --legacy-blocks data/legal/un_parallel/blocks/blocks_en.jsonl \
  --legacy-sixway-dir data/legal/un_parallel/6way \
  --out-dir data/legal/un_uprprc/staged/2015-2025-with-legacy
```

Use that bundle's manifest for generation. Its `legacy` member records the
reference paths, and legacy file hashes become part of the pinned run identity.
This does not enable old blocks as targets in a new-source run.

UN answer evidence remains target-only. Full original document context and
referenced documents help interpretation but cannot provide answer substance.
The evidence contract stores the quote's block-relative offsets, original
document character range and full-document hash. It does not invent line
alignment. Export resolves the exact saved range even if the same quote appears
elsewhere in the document. Legacy line-range evidence retains its existing
format. Export also requires the saved new-source identity to match the selected
manifest, including the source versions for languages not used by the question.

To export new-source runs, use the publishing manifest's `dataset_manifest` member
instead of the legacy `documents`/`text_prefix` source paths:

```json
{
  "version": 1,
  "root": ".",
  "corpora": {
    "un": {
      "generation_sqlite": ["qac/pilot/run.sqlite"],
      "dataset_manifest": "staged/2015-2025/manifest.json",
      "languages": ["ar", "en", "es", "fr", "ru", "zh"]
    }
  }
}
```

Save that example as `data/legal/un_uprprc/export.json`, then prepare a local
export using the existing publishing command:

```bash
clir publish generations --manifest data/legal/un_uprprc/export.json \
  --corpora un --corpus-scope full \
  --output data/hf_export/un-uprprc-pilot
```

This command prepares local files. Upload remains a separate publishing step;
see [generation publishing](generation_publishing.md).

Run the small offline integration checks with:

```bash
uv run pytest tests/test_un_uprprc.py tests/test_un_uprprc_workflow.py
```
