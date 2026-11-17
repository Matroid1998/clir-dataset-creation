# Newer EUR-Lex resources

This opt-in workflow discovers original regulations and directives from **2017–2022** and
prepares a separate, immutable EUR-Lex collection. English records from
`joelniklaus/eurlex_resources` identify candidates; the prepared article bodies and ELI
identifiers come from official CELLAR identifiers and Formex in **English, French, German
and Spanish**. Discovery text is never substituted for an official article.

Preparation uses the existing article segmentation, multilingual structural checks and
reference extraction/resolution. It creates no token blocks and uses no LLM for extraction,
alignment, translation lookup or auditing. Question generation and grading continue to use
the existing model stages, prompts and article evidence contract.

## Inventory and a preparation pilot

Create an inventory using the pinned dataset revision:

```bash
python -m clir_bench.domains.legal.structure.eurlex_resources inventory \
  --out-dir data/legal/eurlex_resources/inventory \
  --revision 4d7946ef7f0c5ff5e261e384db8015dfe8e417cb \
  --pilot-size 30 --seed 42
```

Inventory downloads only `data/en/regulation.jsonl.xz` and `data/en/directive.jsonl.xz`.
It excludes records outside the date/type scope, consolidated or noncanonical CELEX IDs,
empty texts and conflicting source variants. Exact duplicate records are counted. The
inventory report records upstream source-date endpoints, year/type counts, rejection reasons and
the deterministic pilot selection.

Prepare official multilingual articles alongside the current structure directory:

```bash
python -m clir_bench.domains.legal.structure.eurlex_resources prepare \
  --inventory data/legal/eurlex_resources/inventory \
  --baseline-structure data/legal/eurlex/structure \
  --out-dir data/legal/eurlex_resources/pilot-30 \
  --pilot-size 30 --seed 42 --workers 4

python -m clir_bench.domains.legal.structure.eurlex_resources audit \
  data/legal/eurlex_resources/pilot-30/manifest.json
```

The pilot size counts **acts**, not questions or articles. Sampling balances year/type
strata and article-count bins using a fixed seed. Preparation removes acts already in the
baseline before selecting new acts, so its selection can differ from the discovery-only
pilot. Some selected acts may fail official retrieval, parsing or multilingual checks;
`preparation_report.json` gives measured counts and the accepted source-date range.

To prepare an exact pool of **5,000 additional QA-eligible articles**, use an article
budget instead of an act count:

```bash
python -m clir_bench.domains.legal.structure.eurlex_resources prepare \
  --inventory data/legal/eurlex_resources/inventory-2022 \
  --baseline-structure data/legal/eurlex/structure \
  --out-dir data/legal/eurlex_resources/qa-5000 \
  --target-articles 5000 --batch-size 50 --workers 8 \
  --exclude-profile data/legal/eurlex_resources/pilot-30/manifest.json
```

This prioritizes acts with more article headings, initially covering each available year
and document type. It processes complete acts in batches until enough articles pass the
existing QA checks. The resulting sample favors larger acts; it is not a census or a
uniform sample of legislation. `--exclude-profile` excludes acts already accepted in a
prior prepared collection and can be repeated. `--cache-from DIRECTORY` can reuse
checksum-verified official responses from another preparation workspace or its `cache`
directory. Transient failures are retried.

The frozen `target_articles.json` permits exactly the requested number of distinct
article IDs as QA targets. All four language versions must exist, including the versions
of references supplied to generation. Additional articles and annexes remain in the
source collection for reference resolution, so the total source record count exceeds the
target budget. No partial collection is published if the budget cannot be met. Progress
and validated source caches remain available for resuming the same command.

Prepared Formex article numbering is checked against the official operative XML
inventory. Modern `GR.SEQ` divisions are adapted for the existing parser; quoted
amendments, tables of contents and annexes do not become primary articles.

The output includes complete original article/annex records, same-act and cross-act
reference artifacts, quarantine records and an explicit `target_acts.json` allowlist.
Baseline article bytes are copied unchanged into the collection for reference resolution.
The baseline directory and default active corpus are not rewritten. Baseline acts remain
available as references; only accepted newer acts can supply new question targets.
Quarantined acts remain in the raw bundle for auditing and are excluded from the usable
article index and published retrieval corpus, including conflicting legacy records.

The default resumable preparation workspace is `.pilot-30.work` beside the output.
Repeat the same command after an interruption to reuse validated official responses and
parsed files. `--work-dir` chooses another workspace. Its input/configuration fingerprints
must match; use fresh paths for a different inventory, baseline or pilot. A completed output
directory is never overwritten. Inventory and preparation can access the network; audit
checks the local declared hashes and structural invariants without model or network calls.

## Use the existing question workflow

Run a model-free preflight first:

```bash
clir --domain legal qac generate --source eurlex \
  --eurlex-dataset data/legal/eurlex_resources/pilot-30/manifest.json \
  --run-dir data/legal/qac/eurlex_resources-pilot \
  --questions 10 --langs en fr de es --modes lookup --dry-run
```

Remove `--dry-run` to generate and grade questions. Existing generator/verifier options
remain available. The preflight creates no run artifacts and performs no model calls.
Each target must pass the current article checks: 600–9000 English characters, nonamending,
outside quarantine and reference-complete. Target and supplied reference articles/annexes
must have all four original language versions even for an English-only question.

This is a preparation pilot, not a claim of complete 2017–2022 coverage. A limited act pool
can leave cross-act references unresolved, which removes affected articles from the question
pool. A valid prepared collection can therefore yield fewer targets than requested.
Inspect the preparation report and dry-run selection before increasing the pilot size.
Question sampling still defaults to at most two targets per act; the prepared article
budget defines the available pool, independently of how many questions a run selects.

Run metadata and exported candidate rows retain `eurlex_dataset_identity` and the
`eurlex-resources-v1` format marker. Identity pins the manifest and every declared source,
reference and quarantine artifact. Resume, regrade and fixed-target replay inherit the
original profile, including when input is a standalone CSV. Changing an unused language
version or a reference artifact still invalidates the pin. A marked newer row without its
saved identity is rejected.

## Publish full original source articles

Use a generation publication manifest with `dataset_manifest` in its EUR-Lex entry:

```json
{
  "version": 1,
  "root": "..",
  "corpora": {
    "eurlex": {
      "generation_sqlite": ["data/legal/qac/eurlex_resources-pilot/run.sqlite"],
      "dataset_manifest": "data/legal/eurlex_resources/pilot-30/manifest.json",
      "languages": ["en", "fr", "de", "es"],
      "unit_types": ["article", "annex"]
    }
  }
}
```

Paths resolve from the publication manifest's `root`. `dataset_manifest` and `articles`
are mutually exclusive. The existing `articles` adapter remains available for legacy
exports. Prepare a separate release with the existing publisher:

```bash
clir --domain legal publish generations --manifest path/to/eurlex-resources-export.json \
  --corpora eurlex --corpus-scope full --output data/hf_export/eurlex-resources-pilot
```

The publication contains full original article/annex text, exact quoted evidence and
document-level translation associations. It preserves baseline reference records and
validates source/ELI/language identities before filtering the requested corpus scope.
Conflicting duplicates and mismatched article identities cannot become translation families.

Every newer QA must match the collection's complete saved identity before publication.
For example, changing French text while leaving English evidence intact still aborts an
export from the old run. Legacy baseline QAs without a profile identity can retain their
original attribution. Pinned profile QAs must target an accepted newer article; baseline
articles and annexes remain reference material. Any supplied identity must match. Preparing
a release writes a new local directory. Upload remains a separate invocation of the existing
publishing command.
