"""Prepare a reviewable Parquet dataset, then optionally upload the exact bundle."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from clir_bench import domains
from clir_bench.core.publish import card_yaml, hf_token

from .evidence import evidence_rows
from .records import QA_SCHEMA, dumps, normalize, read_generations
from .sources import CORPUS_SCHEMA, SourceIndex, load_chemistry, load_eurlex, load_un, targets

QRELS_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("corpus_id", pa.string()),
        ("score", pa.int64()),
        ("relation", pa.string()),
        ("corpus", pa.string()),
        ("language", pa.string()),
    ]
)
EVIDENCE_SCHEMA = pa.schema(
    [
        (key, pa.string())
        for key in (
            "query_id",
            "source_unit_id",
            "language",
            "text",
            "origin",
            "match_status",
            "source_ref",
            "generation_source_id",
        )
    ]
    + [
        (key, pa.int64())
        for key in (
            "start",
            "end",
            "span_index",
            "generation_start",
            "generation_end",
        )
    ]
)


class Shards:
    """Small bounded batches, with explicit schemas even for entirely null fields."""

    def __init__(self, output: Path, name: str, schema: pa.Schema, shard_bytes=128_000_000):
        self.path = output / "data" / name
        self.path.mkdir(parents=True)
        self.schema = schema
        self.shard_bytes = shard_bytes
        self.pending = []
        self.writer = None
        self.size = self.shard = self.count = 0

    def add(self, row):
        self.pending.append(row)
        self.count += 1
        if len(self.pending) >= 128:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        table = pa.Table.from_pylist(self.pending, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path / f"train-{self.shard:05d}.parquet", self.schema, compression="zstd"
            )
        self.writer.write_table(table)
        self.size += table.nbytes
        self.pending.clear()
        if self.size >= self.shard_bytes:
            self.writer.close()
            self.writer = None
            self.shard += 1
            self.size = 0

    def close(self):
        self.flush()
        if self.writer:
            self.writer.close()
            self.writer = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    manifest_path: Path,
    output: Path,
    *,
    scope: str = "referenced",
    corpora: list[str] | None = None,
) -> dict:
    if scope not in {"full", "referenced"}:
        raise ValueError(f"unknown corpus scope: {scope}")
    specification = json.loads(manifest_path.read_text(encoding="utf-8"))
    if specification.get("version") != 1:
        raise ValueError("generation input manifest must have version 1")
    root = (manifest_path.parent / specification.get("root", ".")).resolve()
    specs = specification["corpora"]
    if corpora:
        if set(corpora) - specs.keys():
            raise ValueError("selected corpus is absent from the input manifest")
        specs = {name: spec for name, spec in specs.items() if name in corpora}
    loaders = {"eurlex": load_eurlex, "un": load_un, "chemistry": load_chemistry}
    if not specs or set(specs) - loaders.keys():
        raise ValueError("choose one or more of eurlex, un, chemistry in the manifest")
    if output.exists():
        raise ValueError(f"output already exists; choose a new export directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    def generations():
        for corpus, spec in specs.items():
            yield from read_generations(corpus, spec, root)

    evidence_languages: dict[str, set[str]] = {}
    wanted = targets(generations(), evidence_languages=evidence_languages)
    if set(specs) - wanted.keys():
        raise ValueError("every selected corpus must have generation rows")
    with tempfile.TemporaryDirectory(prefix=".generation-export-", dir=output.parent) as temporary:
        stage = Path(temporary)
        index = SourceIndex(stage / "source-index.sqlite")
        writers = {}
        report = {
            "corpus_scope": scope,
            "queries": Counter(),
            "models": {},
            "personas": {},
            "support_matches": {},
            "multi_article": Counter(),
            "multi_article_conflicts": Counter(),
            "cross_act": Counter(),
            "duplicate_candidates_skipped": 0,
        }
        try:
            for corpus, spec in specs.items():
                print(f"Loading {corpus} source units ({scope})...", flush=True)
                if evidence_languages.get(corpus):
                    spec = {
                        **spec,
                        "languages": list(
                            dict.fromkeys(
                                [
                                    *spec.get("languages", []),
                                    *sorted(evidence_languages[corpus]),
                                ]
                            )
                        ),
                    }
                loaders[corpus](
                    index, spec, root, wanted[corpus] if scope == "referenced" else None
                )
                index.connection.commit()
            for name, schema in (
                ("qas", QA_SCHEMA),
                ("evidence", EVIDENCE_SCHEMA),
                ("source_qrels", QRELS_SCHEMA),
                ("corpus", CORPUS_SCHEMA),
            ):
                writers[name] = Shards(stage, name, schema)
            seen = {}
            for corpus, raw, provenance in generations():
                row = normalize(corpus, raw, provenance, index)
                if row["unresolved_article_references"]:
                    raise ValueError(
                        f"unresolved involved articles for {row['_id']}: "
                        f"{row['unresolved_article_references']}"
                    )
                digest = hashlib.sha256(row["metadata_json"].encode()).hexdigest()
                if row["_id"] in seen:
                    if seen[row["_id"]] != digest:
                        raise ValueError(f"conflicting versions of candidate {row['_id']}")
                    report["duplicate_candidates_skipped"] += 1
                    continue
                seen[row["_id"]] = digest
                writers["qas"].add(row)
                for evidence in evidence_rows(row):
                    writers["evidence"].add(evidence)
                links = {}
                for identifier in row["involved_source_unit_ids"]:
                    doc = index.get(identifier)
                    links[identifier] = (doc, "generation_source")
                    if corpus == "chemistry":
                        partners = (index.get(other) for other in doc["paired_document_ids"])
                    else:
                        partners = index.translations(corpus, doc["source_id"])
                    for partner in partners:
                        if partner is None:
                            raise ValueError(f"missing direct translation partner for {identifier}")
                        links.setdefault(partner["_id"], (partner, "direct_translation"))
                for identifier, (doc, relation) in sorted(links.items()):
                    writers["source_qrels"].add(
                        {
                            "query_id": row["_id"],
                            "corpus_id": identifier,
                            "score": 1,
                            "relation": relation,
                            "corpus": corpus,
                            "language": doc["language"],
                        }
                    )
                report["queries"][corpus] += 1
                for category, field in (
                    ("models", "generator_model_id"),
                    ("personas", "persona"),
                    ("support_matches", "support_match_status"),
                ):
                    report[category].setdefault(corpus, Counter())[row[field]] += 1
                for category, field in (
                    ("multi_article", "multi_article"),
                    ("multi_article_conflicts", "multi_article_conflict"),
                    ("cross_act", "cross_act"),
                ):
                    report[category][corpus] += int(row[field])
            corpus_counts = Counter()
            for doc in index.rows():
                writers["corpus"].add(doc)
                corpus_counts[f"{doc['corpus']}/{doc['language']}"] += 1
            report["corpus_units"] = corpus_counts
            report["eurlex_duplicate_source_records"] = index.duplicates
        finally:
            index.close()
            for writer in writers.values():
                writer.close()
        (stage / "source-index.sqlite").unlink()
        counts = {name: writer.count for name, writer in writers.items()}
        report["configs"] = counts
        (stage / "report.json").write_text(dumps(report) + "\n", encoding="utf-8")
        (stage / "README.md").write_text(dataset_card(specs, scope, counts), encoding="utf-8")
        export_manifest = {
            "version": 1,
            "input_manifest": specification,
            "selected_corpora": list(specs),
            "corpus_scope": scope,
            "input_manifest_sha256": sha256(manifest_path),
            "configs": counts,
            "files": {
                str(p.relative_to(stage)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        (stage / "manifest.json").write_text(dumps(export_manifest) + "\n", encoding="utf-8")
        stage.rename(output)
    return report


def dataset_card(specs: dict, scope: str, counts: dict) -> str:
    attribution = []
    for corpus in specs:
        domain = domains.load("chemistry" if corpus == "chemistry" else "legal")
        attribution.append(domain.attribution_for("gp" if corpus == "chemistry" else corpus))
    scope_note = (
        "Full local corpus for the configured languages; this is not a claim of all upstream data."
        if scope == "full"
        else "Contains documents associated with the published questions and their available "
        "translations. This QA release does not include the complete retrieval haystack."
    )
    return (
        card_yaml(list(counts)).replace(
            "- config_name: qas\n", "- config_name: qas\n  default: true\n"
        )
        + "\n# Multilingual generated QAs and source corpora\n\n"
        + f"""
{scope_note}

All supplied generator candidates are retained, including low-scoring and unmatched evidence.
The `train` split is a storage convention; no leakage-safe evaluation split has been assigned.
Before splitting, group questions from the same article/act, UN document, or patent and its
translation partners to avoid source and generator leakage.

## Configurations

"""
        + "\n".join(f"- `{name}`: {count:,} rows" for name, count in counts.items())
        + """

- `qas`: questions, answers, raw evidence, persona/mode, model, scores, source IDs and involvement.
  `document_text` embeds the complete primary source unit; additional involved units are linked
  through `involved_source_unit_ids` to `corpus`.
  `metadata_json` preserves the complete input row, including persona-specific fields, prompts,
  grading responses, routing decisions and generation references. `provenance_json` identifies
  the input file and row/task. CSV metadata preserves its original string types.
- `evidence`: canonical supporting text: explicit `evidence` where present, otherwise `answer`.
  Legacy rows use this fallback and first-match offsets. For future legal rows with the
  `legal_evidence_spans_v1` contract, emit one record per saved span, identified by `span_index`,
  and validate each against its own source and language. `start` and `end` refer to that
  `source_unit_id`'s corpus text in Unicode code points, end-exclusive. UN offsets are mapped
  through the saved block line range, never a whole-document search. `generation_source_id`,
  `generation_start` and `generation_end` preserve the original article/block coordinates.
  Matching checks text and source identity, not factual entailment or semantic minimality.
- `corpus`: `_id`, text, language, source/parent IDs, unit type and original metadata. EUR-Lex
  units are full articles (or explicitly involved annexes); patent units are title + abstract;
  UN units are entire documents joined from the original aligned lines. UN block IDs/ranges
  survive in QA metadata only. EUR-Lex duplicate ELI/language records use the last occurrence,
  matching the generation loader. Source text and patent markup are retained.
- `source_qrels`: positive **source-unit associations derived from generation provenance**,
  plus existing language versions of each legal unit / directly accepted chemistry translation
  partners. They are not manually judged relevance or chunk-level qrels. `score=1` indicates
  an association; absent judgments do not establish irrelevance. Supplied contextual references
  alone do not create positives. Chemistry partners are not expanded transitively.

`involved_source_unit_ids` includes the primary source and explicitly involved EUR-Lex units;
`involved_article_ids` includes article ELIs only. `multi_article` is true when multiple articles
are resolved or the generator declared it. `multi_article_declared` retains that declaration;
`multi_article_conflict` identifies a disagreement with the resolved article count. `cross_act`
means involvement across acts; `multi_source_unit` also covers an article plus an annex.
A translation into another language does not count as another involved article.
UN and chemistry have no article-involvement annotation, so their `multi_article` is false
and `multi_article_declared` is null; supplied UN references are retained only as metadata.

Future legal QAs retain their complete ordered `evidence_spans` list. Its `start`/`end` remain
relative to the generation source; `document_start`/`document_end` map into the exported corpus
unit. The first span supplies scalar `evidence`, `evidence_language`, `supporting_text` and
`support_start`/`support_end`; `support_source_unit_id` identifies the corpus text those offsets
refer to. It can differ from the QA's target `source_unit_id`, for example when a supplied
EUR-Lex reference provides the direct answer. `document_text` continues to contain the target.
Explicit evidence languages are loaded even when omitted from the manifest's language list.
Source hashes, offsets and multi-article coverage are checked; invalid new-contract evidence
aborts export instead of falling back to the answer. Legacy rows are not backfilled.

These source units have **not yet been chunked**. Create retrieval chunks with a pinned tokenizer
(e.g. BAAI/bge-m3), preserve parent IDs and character ranges, and align evidence to those chunks.
Do not mark every chunk of a source document relevant. Unmatched or translated evidence needs
separate alignment/review before it can establish chunk-level judgments.

`manifest.json` records export inputs and SHA-256 hashes; `report.json` records coverage,
models, personas, evidence matching and multi-article statistics. No upstream license is
assigned to generated material by this exporter.

## Attribution and source terms

"""
        + "\n\n".join(attribution)
        + "\n"
    )


def upload(output: Path, repo_id: str, *, private: bool = False, api=None) -> str:
    """Upload only a validated bundle; refuse to mix it with existing corpus shards."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("files"):
        raise ValueError("not a prepared generation export")
    for relative, details in manifest["files"].items():
        path = output / relative
        if not path.resolve().is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError(f"invalid bundle path: {relative}")
        if sha256(path) != details["sha256"]:
            raise ValueError(f"bundle changed after preparation: {relative}")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token())
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    files = api.list_repo_files(repo_id, repo_type="dataset")
    if any(path.startswith("data/") for path in files):
        raise ValueError(
            "target already contains dataset shards; use a new repo to avoid mixing releases"
        )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output),
        allow_patterns=[*manifest["files"], "manifest.json"],
        commit_message="Publish generated QAs, supporting evidence and source corpora",
    )
    return f"https://huggingface.co/datasets/{repo_id}"
