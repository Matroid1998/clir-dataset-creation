"""Import accepted, explicitly paired abstract records from the pinned AI4Chem archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from clir_bench.core import corpus as corpus_io
from clir_bench.core.context import AppContext
from clir_bench.core.domain import CorpusSchema
from clir_bench.domains.chemistry.vocabulary import WORKING_LANGUAGES

DATASET_REPO = "BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs"
SOURCE_VALUE = "ai4chem_abstract_pairs"
SUPPORTED_LANGUAGES = WORKING_LANGUAGES
CORPUS_FILENAME = "multilingual_corpus.csv"
PAIRS_FILENAME = "accepted_pairs.csv"
MANIFEST_FILENAME = "manifest.json"
ACCEPTED_VERDICTS = frozenset(("exact", "high"))
_PAIR_FIELDS = (
    "example_id", "doc_id", "corpus_id", "source", "group_key", "pub_date", "field",
    "language_pair", "source_language", "target_language", "source_text", "target_text",
    "source_char_count", "target_char_count", "source_token_count", "target_token_count",
    "source_sentence_count", "target_sentence_count", "source_tokens_per_sentence",
    "target_tokens_per_sentence", "tokenizer", "selection_rule", "judge_model",
    "judge_verdict", "judge_reason", "pair_ok",
)


def _languages(values: Sequence[str] | None) -> tuple[str, ...]:
    languages = tuple(dict.fromkeys(values if values is not None else SUPPORTED_LANGUAGES))
    if not languages or set(languages) - set(SUPPORTED_LANGUAGES):
        raise ValueError(f"languages must be a nonempty subset of {SUPPORTED_LANGUAGES}")
    return languages


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pair field {key} must be a nonempty string")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pair_rows(path: Path) -> list[dict[str, Any]]:
    """Read a local archive; do not normalize or truncate its abstract strings."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if path.suffix.lower() not in (".jsonl", ".ndjson"):
        raise ValueError("Pair input must be a .parquet, .jsonl or .ndjson file")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Pair input line {line_number} must be a JSON object")  # noqa: TRY004 -- invalid archive data
            rows.append(row)
    return rows


def load_hub_rows(revision: str = "main") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve a moving ref once, then download every train shard at that commit."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(DATASET_REPO, revision=revision)
    resolved = info.sha
    if not isinstance(resolved, str) or re.fullmatch(r"[0-9a-f]{40}", resolved) is None:
        raise ValueError("Hub did not return an immutable dataset commit SHA")
    filenames = sorted(
        sibling.rfilename for sibling in info.siblings
        if re.fullmatch(r"data/train-\d+-of-\d+\.parquet", sibling.rfilename)
    )
    if not filenames:
        raise ValueError("Dataset snapshot contains no train parquet shards")
    rows, files = [], []
    for filename in filenames:
        path = Path(hf_hub_download(
            repo_id=DATASET_REPO, filename=filename, repo_type="dataset", revision=resolved,
        ))
        rows.extend(read_pair_rows(path))
        files.append({"path": filename, "sha256": _sha256(path), "bytes": path.stat().st_size})
    return rows, {
        "origin": "hugging_face", "dataset": DATASET_REPO, "config": "default", "split": "train",
        "requested_revision": revision, "resolved_revision": resolved, "files": files,
    }


def build_corpus(
    pair_rows: Sequence[Mapping[str, Any]], *, schema: CorpusSchema, revision: str,
    languages: Sequence[str] | None = None, limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep accepted edges and deduplicate their endpoints without graph closure."""
    languages = _languages(languages)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("An upstream revision is required")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive accepted-pair count")
    selected = []
    seen_ids: set[str] = set()
    seen_edges: set[tuple[str, frozenset[str]]] = set()
    accepted_count = supported_count = 0
    for row in pair_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every pair row must be an object")  # noqa: TRY004 -- invalid archive data
        pair_id = _text(row, "example_id")
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate or ambiguous pair ID: {pair_id}")
        seen_ids.add(pair_id)
        if type(row.get("pair_ok")) is not bool:
            raise ValueError(f"pair_ok must be a boolean for {pair_id}")
        verdict = _text(row, "judge_verdict")
        if verdict not in ("exact", "high", "medium", "low"):
            raise ValueError(f"Unknown judge_verdict for {pair_id}: {verdict}")
        if not (row["pair_ok"] is True and verdict in ACCEPTED_VERDICTS):
            continue
        accepted_count += 1
        source_language, target_language = _text(row, "source_language"), _text(row, "target_language")
        if source_language == target_language:
            raise ValueError(f"Pair {pair_id} has identical endpoint languages")
        if source_language not in languages or target_language not in languages:
            continue
        supported_count += 1
        for key in ("doc_id", "corpus_id", "source", "group_key", "pub_date", "field",
                    "language_pair", "source_text", "target_text", "judge_model", "judge_reason"):
            _text(row, key)
        if row["field"] != "abstract":
            raise ValueError(f"Pair {pair_id} is not an abstract pair")
        if row["language_pair"] != f"{source_language}-{target_language}":
            raise ValueError(f"Pair {pair_id} language_pair disagrees with its endpoints")
        edge = (row["doc_id"], frozenset((source_language, target_language)))
        if edge in seen_edges:
            raise ValueError(f"Ambiguous repeated publication/language edge: {pair_id}")
        seen_edges.add(edge)
        if limit is None or len(selected) < limit:
            selected.append(dict(row))

    documents: dict[tuple[str, str], dict[str, Any]] = {}
    memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    neighbors: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in selected:
        publication = pair["doc_id"]
        ids = {side: schema.make_doc_id(publication, pair[f"{side}_language"])
               for side in ("source", "target")}
        pair.update(source_document_id=ids["source"], target_document_id=ids["target"])
        for side, other in (("source", "target"), ("target", "source")):
            language, abstract = pair[f"{side}_language"], pair[f"{side}_text"]
            key = (publication, language)
            document = {
                "id": ids[side], "language": language, "title": "", "abstract": abstract,
                "description": "", "first_claim": "", "context": abstract,
                "publication_number": publication, "country_code": pair["source"],
                "publication_date": pair["pub_date"], "source": SOURCE_VALUE, "ipc_codes": "",
                "upstream_dataset": DATASET_REPO, "upstream_revision": revision,
                "upstream_corpus_id": pair["corpus_id"], "upstream_group_key": pair["group_key"],
            }
            if key in documents and documents[key] != document:
                raise ValueError(f"Conflicting text or publication metadata for {publication}/{language}")
            documents[key] = document
            memberships[key].add(pair["example_id"])
            neighbors[key].add(ids[other])
    for key, document in documents.items():
        document["accepted_pair_ids"] = json.dumps(sorted(memberships[key]), ensure_ascii=False)
        document["paired_document_ids"] = json.dumps(sorted(neighbors[key]), ensure_ascii=False)
    corpus_rows = [documents[key] for key in sorted(documents)]
    validate_corpus(corpus_io.group_by_family(corpus_rows, schema), schema)
    counts = {
        "source_pairs": len(pair_rows), "accepted_pairs_all_languages": accepted_count,
        "accepted_pairs_selected_languages": supported_count, "selected_pairs": len(selected),
        "documents": len(corpus_rows), "publications": len({key[0] for key in documents}),
        "patent_families": len({pair["group_key"] for pair in selected}),
        "documents_by_language": dict(sorted(Counter(row["language"] for row in corpus_rows).items())),
        "pairs_by_language": dict(sorted(Counter(row["language_pair"] for row in selected).items())),
        "pairs_by_verdict": dict(sorted(Counter(row["judge_verdict"] for row in selected).items())),
    }
    return corpus_rows, selected, counts


def _id_array(row: Mapping[str, Any], key: str) -> set[str]:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a JSON array for {row.get('id')}")  # noqa: TRY004 -- invalid CSV data
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed {key} for {row.get('id')}") from exc
    if (not isinstance(decoded, list) or not decoded
            or any(not isinstance(item, str) or not item.strip() for item in decoded)
            or len(set(decoded)) != len(decoded)):
        raise ValueError(f"{key} must contain unique, nonempty string IDs for {row.get('id')}")
    return set(decoded)


def validate_corpus(grouped: Mapping[str, Sequence[Mapping[str, Any]]], schema: CorpusSchema) -> None:
    """Reject legacy corpora and malformed or invented pair-neighbor metadata."""
    rows: dict[str, Mapping[str, Any]] = {}
    pair_ids, neighbors = {}, {}
    pair_members: dict[str, set[str]] = defaultdict(set)
    for family, family_rows in grouped.items():
        for row in family_rows:
            if row.get("source") != SOURCE_VALUE or row.get("upstream_dataset") != DATASET_REPO:
                raise ValueError("Chemistry generation requires the accepted AI4Chem abstract-pair corpus")
            if not isinstance(row.get("upstream_revision"), str) or not row["upstream_revision"].strip():
                raise ValueError("Chemistry corpus row is missing upstream_revision")
            abstract = row.get("abstract")
            if not isinstance(abstract, str) or not abstract.strip() or row.get("context") != abstract:
                raise ValueError("Chemistry corpus context must preserve the nonempty abstract exactly")
            language = schema.language_of(row)
            if language not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported chemistry corpus language: {language}")
            publication, document_id = schema.family_of(row), schema.id_of(row)
            if not publication or publication != family:
                raise ValueError("Corpus publication grouping disagrees with its row")
            if not document_id or document_id != schema.make_doc_id(publication, language):
                raise ValueError("Missing or inconsistent chemistry endpoint document ID")
            if document_id in rows:
                raise ValueError(f"Duplicate chemistry endpoint document ID: {document_id}")
            rows[document_id] = row
            pair_ids[document_id] = _id_array(row, "accepted_pair_ids")
            neighbors[document_id] = _id_array(row, "paired_document_ids")
            for pair_id in pair_ids[document_id]:
                pair_members[pair_id].add(document_id)
    for pair_id, members in pair_members.items():
        if len(members) != 2:
            raise ValueError(f"Accepted pair {pair_id} must have exactly two endpoints")
    for document_id, row in rows.items():
        expected = set().union(*(pair_members[pair] for pair in pair_ids[document_id])) - {document_id}
        if neighbors[document_id] != expected:
            raise ValueError(f"Direct neighbors disagree with accepted pair IDs for {document_id}")
        for neighbor_id in neighbors[document_id]:
            other = rows.get(neighbor_id)
            if other is None or neighbor_id == document_id:
                raise ValueError(f"Missing or self paired document for {document_id}")
            if schema.family_of(other) != schema.family_of(row) or schema.language_of(other) == schema.language_of(row):
                raise ValueError(f"Paired documents must share a publication and differ in language: {document_id}")
            if document_id not in neighbors[neighbor_id]:
                raise ValueError(f"Asymmetric paired documents for {document_id}")
            if other["upstream_revision"] != row["upstream_revision"]:
                raise ValueError(f"Paired documents have different upstream revisions: {document_id}")


def _csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (bool, list, dict)) else value


def _publish_files(staged: Path, destination: Path, filenames: Sequence[str]) -> None:
    """Publish complete files with a rollback on failure; commit the manifest last."""
    destination.mkdir(parents=True, exist_ok=True)
    backup = staged / "previous"
    backup.mkdir()
    previous, installed = [], []
    try:
        for filename in filenames:
            target = destination / filename
            if target.exists():
                shutil.copy2(target, backup / filename)
                previous.append(filename)
            os.replace(staged / filename, target)
            installed.append(filename)
    except BaseException:
        for filename in reversed(installed):
            target = destination / filename
            if filename in previous:
                os.replace(backup / filename, target)
            else:
                target.unlink(missing_ok=True)
        raise


def ingest(context: AppContext, args: argparse.Namespace) -> int:
    """Build the sole chemistry corpus from accepted supported-language pair edges."""
    if context.setting("input_dataset_repo", DATASET_REPO) != DATASET_REPO:
        raise ValueError(f"The chemistry input_dataset_repo must be {DATASET_REPO}")
    languages = _languages(getattr(args, "languages", None))
    requested_revision = getattr(args, "revision", None) or "main"
    limit = getattr(args, "limit", None)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive accepted-pair count")
    local_input = getattr(args, "input", None)
    if local_input:
        path = Path(local_input).expanduser().resolve()
        digest = _sha256(path)
        pair_rows = read_pair_rows(path)
        provenance = {
            "origin": "local_input", "dataset": DATASET_REPO, "config": "default", "split": "train",
            "requested_revision": requested_revision, "resolved_revision": f"local-sha256:{digest}",
            "remote_revision_verified": False,
            "files": [{"path": str(path), "sha256": digest, "bytes": path.stat().st_size}],
        }
    else:
        pair_rows, provenance = load_hub_rows(requested_revision)
    corpus_rows, pairs, counts = build_corpus(
        pair_rows, schema=context.schema, revision=provenance["resolved_revision"],
        languages=languages, limit=limit,
    )
    if not pairs:
        raise ValueError("No accepted abstract pairs match the selected languages")
    output_path = context.workspace.corpus_csv("gp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".abstract-pairs-", dir=output_path.parent.parent) as temporary:
        staged = Path(temporary)
        corpus_io.write_rows(staged / output_path.name, corpus_rows, context.schema.fields)
        pair_fields = tuple(dict.fromkeys((*_PAIR_FIELDS, *(key for row in pairs for key in row))))
        corpus_io.write_rows(staged / PAIRS_FILENAME,
                             ({key: _csv_value(value) for key, value in row.items()} for row in pairs), pair_fields)
        manifest = {
            "schema_version": 1, "source": SOURCE_VALUE, "upstream": provenance,
            "acceptance": {"pair_ok": True, "judge_verdict": sorted(ACCEPTED_VERDICTS),
                           "both_languages_in": list(languages)},
            "pair_limit": limit, "counts": counts,
            "artifacts": {name: {"sha256": _sha256(staged / name), "bytes": (staged / name).stat().st_size}
                          for name in (output_path.name, PAIRS_FILENAME)},
        }
        (staged / MANIFEST_FILENAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
        _publish_files(staged, output_path.parent, (output_path.name, PAIRS_FILENAME, MANIFEST_FILENAME))
    print(f"Imported {len(pairs)} accepted abstract pairs -> {len(corpus_rows)} documents: {output_path}")
    return 0


__all__ = ["DATASET_REPO", "SOURCE_VALUE", "build_corpus", "ingest", "load_hub_rows", "validate_corpus"]
