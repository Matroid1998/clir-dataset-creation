"""Expand a published corpus without rewriting its existing QA release."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .generations import QRELS_SCHEMA, Shards, sha256
from .records import dumps
from .sources import CORPUS_SCHEMA, SourceIndex, load_chemistry, load_eurlex, load_un

_CONFIGS = {"qas", "evidence", "source_qrels", "corpus"}
_REFERENCED_NOTE = (
    "Contains documents associated with the published questions and their available "
    "translations. This QA release does not include the complete retrieval haystack."
)
_FULL_NOTE = (
    "Full local corpus for the configured languages; this is not a claim of all upstream data."
)


def _validated_manifest(output: Path, corpus: str) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("version") != 1
        or manifest.get("selected_corpora") != [corpus]
        or set(manifest.get("configs", {})) != _CONFIGS
        or not manifest.get("files")
    ):
        raise ValueError("expected a prepared single-corpus generation export")
    for relative, details in manifest["files"].items():
        path = output / relative
        if not path.resolve().is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError(f"invalid bundle path: {relative}")
        if path.stat().st_size != details["bytes"] or sha256(path) != details["sha256"]:
            raise ValueError(f"bundle changed after preparation: {relative}")
    actual = {str(path.relative_to(output)) for path in (output / "data").rglob("*.parquet")}
    recorded = {name for name in manifest["files"] if name.startswith("data/")}
    if actual != recorded:
        raise ValueError("bundle contains unrecorded or missing data files")
    for config, expected in manifest["configs"].items():
        paths = sorted((output / "data" / config).glob("*.parquet"))
        if not paths or sum(pq.ParquetFile(path).metadata.num_rows for path in paths) != expected:
            raise ValueError(f"incorrect recorded row count: {config}")
    return manifest


def _rows(output: Path, config: str, columns=None):
    for path in sorted((output / "data" / config).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=columns):
            yield from batch.to_pylist()


def _next_qrels_shard(output: Path) -> int:
    numbers = []
    for path in (output / "data" / "source_qrels").glob("*.parquet"):
        match = re.fullmatch(r"train-(\d+)\.parquet", path.name)
        if not match:
            raise ValueError(f"unsupported existing qrels shard name: {path.name}")
        numbers.append(int(match[1]))
    return max(numbers, default=-1) + 1


def _add_translation_qrels(old_output: Path, stage: Path, index: SourceIndex, corpus: str):
    known = set()
    for row in _rows(old_output, "source_qrels", ["query_id", "corpus_id"]):
        if index.get(row["corpus_id"]) is None:
            raise ValueError(f"existing relevance source is missing: {row['corpus_id']}")
        known.add((row["query_id"], row["corpus_id"]))
    additions_root = stage / ".qrels-additions"
    writer = None
    try:
        for qa in _rows(old_output, "qas", ["_id", "involved_source_unit_ids"]):
            for identifier in qa["involved_source_unit_ids"]:
                source = index.get(identifier)
                if source is None:
                    raise ValueError(f"existing QA source is missing: {identifier}")
                if corpus == "chemistry":
                    partners = (index.get(other) for other in source["paired_document_ids"])
                else:
                    partners = index.translations(corpus, source["source_id"])
                for partner in partners:
                    if partner is None:
                        raise ValueError(f"missing translation partner for {identifier}")
                    pair = (qa["_id"], partner["_id"])
                    if pair in known:
                        continue
                    if writer is None:
                        writer = Shards(additions_root, "source_qrels", QRELS_SCHEMA)
                        writer.shard = _next_qrels_shard(old_output)
                    writer.add(
                        {
                            "query_id": qa["_id"],
                            "corpus_id": partner["_id"],
                            "score": 1,
                            "relation": "direct_translation",
                            "corpus": corpus,
                            "language": partner["language"],
                        }
                    )
                    known.add(pair)
    finally:
        if writer:
            writer.close()
    if writer is None:
        return 0
    for path in writer.path.glob("*.parquet"):
        path.rename(stage / "data" / "source_qrels" / path.name)
    shutil.rmtree(additions_root)
    return writer.count


def _expanded_card(old_card: str, counts: dict, corpus: str, languages: list[str]) -> str:
    if _REFERENCED_NOTE in old_card:
        card = old_card.replace(_REFERENCED_NOTE, _FULL_NOTE, 1)
    elif _FULL_NOTE in old_card:
        card = old_card
    else:
        raise ValueError("could not identify the published dataset card's corpus scope")
    for name in ("corpus", "source_qrels"):
        card, replacements = re.subn(
            rf"(?m)^- `{name}`: [\d,]+ rows$", f"- `{name}`: {counts[name]:,} rows", card
        )
        if replacements != 1:
            raise ValueError(f"could not update dataset card row count: {name}")
    if corpus == "eurlex":
        card = card.replace(
            "units are full articles (or explicitly involved annexes)",
            "units are all locally collected articles and annexes",
        )
    language_note = "Corpus languages: " + ", ".join(f"`{lang}`" for lang in languages) + "."
    if re.search(r"(?m)^Corpus languages: .*\.$", card):
        card = re.sub(r"(?m)^Corpus languages: .*\.$", language_note, card)
    else:
        card = card.replace(_FULL_NOTE, _FULL_NOTE + "\n\n" + language_note, 1)
    return card


def expand_corpus(old_output: Path, manifest_path: Path, output: Path, *, corpus: str) -> dict:
    """Prepare full sources and new translation links, preserving published shard bytes."""
    old_output, manifest_path, output = map(Path, (old_output, manifest_path, output))
    loaders = {"eurlex": load_eurlex, "un": load_un, "chemistry": load_chemistry}
    if corpus not in loaders:
        raise ValueError(f"unknown corpus: {corpus}")
    if output.exists():
        raise ValueError(f"output already exists; choose a new export directory: {output}")
    old_manifest = _validated_manifest(old_output, corpus)
    specification = json.loads(manifest_path.read_text(encoding="utf-8"))
    if specification.get("version") != 1 or corpus not in specification.get("corpora", {}):
        raise ValueError("input manifest must select the corpus and have version 1")
    spec = specification["corpora"][corpus]
    root = (manifest_path.parent / specification.get("root", ".")).resolve()
    report = json.loads((old_output / "report.json").read_text(encoding="utf-8"))
    if report.get("configs") != old_manifest["configs"]:
        raise ValueError("published report and manifest row counts disagree")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corpus-expansion-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for relative in old_manifest["files"]:
            if relative in {"README.md", "report.json"} or relative.startswith("data/corpus/"):
                continue
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_output / relative, destination)
        index_path = stage / "source-index.sqlite"
        index = SourceIndex(index_path)
        writer = None
        try:
            print(f"Loading {corpus} source units (full)...", flush=True)
            loaders[corpus](index, spec, root, None)
            index.connection.commit()
            for previous in _rows(old_output, "corpus"):
                if index.get(previous["_id"]) != previous:
                    raise ValueError(f"published source changed or is missing: {previous['_id']}")
            added_qrels = _add_translation_qrels(old_output, stage, index, corpus)
            writer = Shards(stage, "corpus", CORPUS_SCHEMA)
            counts = Counter()
            print(f"Writing full {corpus} corpus...", flush=True)
            for row in index.rows():
                writer.add(row)
                counts[f"{corpus}/{row['language']}"] += 1
                if writer.count % 10_000 == 0:
                    print(f"  {writer.count:,} corpus records written", flush=True)
            report.update(
                corpus_scope="full",
                corpus_units=dict(counts),
                eurlex_duplicate_source_records=index.duplicates,
                corpus_expansion={
                    "previous_corpus_records": old_manifest["configs"]["corpus"],
                    "source_qrels_added": added_qrels,
                    "previous_manifest_sha256": sha256(old_output / "manifest.json"),
                    "preserved_configs": ["qas", "evidence"],
                    "preserved_existing_qrels": True,
                },
            )
            report["configs"]["corpus"] = writer.count
            report["configs"]["source_qrels"] += added_qrels
        finally:
            index.close()
            if writer:
                writer.close()
        index_path.unlink()
        languages = sorted(key.split("/", 1)[1] for key in report["corpus_units"])
        (stage / "README.md").write_text(
            _expanded_card(
                (old_output / "README.md").read_text(encoding="utf-8"),
                report["configs"],
                corpus,
                languages,
            ),
            encoding="utf-8",
        )
        (stage / "report.json").write_text(dumps(report) + "\n", encoding="utf-8")
        manifest = {
            **old_manifest,
            "input_manifest": specification,
            "input_manifest_sha256": sha256(manifest_path),
            "generation_input_manifest": old_manifest.get(
                "generation_input_manifest", old_manifest["input_manifest"]
            ),
            "previous_manifest_sha256": sha256(old_output / "manifest.json"),
            "corpus_scope": "full",
            "configs": report["configs"],
            "files": {
                str(path.relative_to(stage)): {
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(stage.rglob("*"))
                if path.is_file()
            },
        }
        for relative, details in old_manifest["files"].items():
            if (
                relative.startswith(("data/qas/", "data/evidence/", "data/source_qrels/"))
                and manifest["files"][relative] != details
            ):
                raise ValueError(f"published shard was changed during expansion: {relative}")
        (stage / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
        stage.rename(output)
    return report
