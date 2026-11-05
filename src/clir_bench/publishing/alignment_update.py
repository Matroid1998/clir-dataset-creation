"""Prepare local, additive QA patches containing multilingual evidence.

No Hub writes occur here. A patch contains replacement QA shards and metadata,
while its manifest retains hashes for untouched corpus/evidence/qrels files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .corpus_update import _manifest

LANGUAGES = ("en", "de", "es", "fr", "zh")
CONTRACT = "cross_language_evidence_terra_v1"
SPAN_FIELDS = (
    "source_unit_id", "language", "text", "anchor_id", "origin", "alignment_id",
    "validation_status", "match_status", "alignment_status", "is_original",
    "task_id", "request_hash", "model", "api_model", "provider",
)
STATUS_FIELDS = (
    "source_unit_id", "anchor_id", "status", "alignment_id", "task_id", "request_hash",
    "validation_status", "exact_match", "reason", "model", "api_model", "provider",
)
SPAN_TYPE = pa.struct([(name, pa.bool_() if name == "is_original" else pa.string())
                       for name in SPAN_FIELDS])
STATUS_TYPE = pa.struct([(name, pa.bool_() if name == "exact_match" else pa.string())
                         for name in STATUS_FIELDS])
EVIDENCE_TYPE = pa.struct([(language, pa.list_(SPAN_TYPE)) for language in LANGUAGES])
ALIGNMENT_STATUS_TYPE = pa.struct([(language, pa.list_(STATUS_TYPE)) for language in LANGUAGES])
NEW_FIELDS = [
    pa.field("evidence_all_languages", EVIDENCE_TYPE),
    pa.field("evidence_alignment_status", ALIGNMENT_STATUS_TYPE),
    pa.field("evidence_alignment_eligibility", pa.string()),
    pa.field("evidence_alignment_contract", pa.string()),
    *[pa.field(f"evidence_{language}", pa.list_(pa.string())) for language in LANGUAGES],
]
CSV_FIELDS = [
    "qa_id", "corpus", "question", "answer", "question_language", "source_unit_id",
    "persona", "mode", "generator_model_id", "evidence_alignment_eligibility",
    *[f"evidence_{language}" for language in LANGUAGES],
    "evidence_all_languages", "evidence_alignment_status", "evidence_alignment_contract",
]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _typed_entries(value: dict, fields: tuple[str, ...], boolean: str, label: str) -> dict:
    if not isinstance(value, dict) or set(value) - set(LANGUAGES):
        raise ValueError(f"{label} must have only supported language keys")
    result = {}
    for language in LANGUAGES:
        entries = value.get(language, [])
        if not isinstance(entries, list):
            raise TypeError(f"{label}/{language} must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - set(fields):
                raise ValueError(f"unknown or invalid {label} metadata; refusing to discard fields")
            for key, item in entry.items():
                expected = bool if key == boolean else str
                if item is not None and type(item) is not expected:
                    raise ValueError(f"invalid {label}.{key} type")
        result[language] = entries
    return result


def load_alignment_csv(run_dir: Path) -> tuple[dict[str, dict], dict]:
    """Validate a complete, immutable QA inventory and its final CSV snapshot."""
    run_dir = Path(run_dir)
    csv.field_size_limit(max(csv.field_size_limit(), 20_000_000))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "completed":
        raise ValueError("alignment run is not completed")
    frozen_rows = _jsonl(run_dir / "qa_index.jsonl")
    frozen = {row["qa_id"]: row for row in frozen_rows}
    if len(frozen) != len(frozen_rows) or not frozen:
        raise ValueError("missing or duplicate QA IDs in frozen run inventory")
    with (run_dir / "qa_evidence.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"qa_id", "corpus", "question", "answer", "question_language", "source_unit_id",
                    "evidence_all_languages", "evidence_alignment_status", "eligibility_status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("alignment CSV is missing required columns")
        rows = {}
        for row in reader:
            identity = row["qa_id"]
            if identity in rows or identity not in frozen:
                raise ValueError("alignment CSV contains duplicate or unexpected QA IDs")
            original = frozen[identity]
            for field in ("corpus", "question", "answer", "question_language", "source_unit_id",
                          "eligibility_status"):
                if row[field] != (original.get(field) or ""):
                    raise ValueError(f"alignment CSV changed original QA field: {identity}/{field}")
            evidence = _typed_entries(json.loads(row["evidence_all_languages"]), SPAN_FIELDS,
                                      "is_original", "evidence")
            statuses = _typed_entries(json.loads(row["evidence_alignment_status"]), STATUS_FIELDS,
                                      "exact_match", "status")
            if row["eligibility_status"] != "eligible" and (any(evidence.values()) or any(statuses.values())):
                raise ValueError("excluded QA cannot contain evidence or alignment statuses")
            for language, entries in statuses.items():
                for status in entries:
                    if status.get("status") not in {"source_exact", "matched", "not_found", "uncertain",
                                                    "invalid", "failed", "language_mismatch"}:
                        raise ValueError(f"incomplete or unknown alignment status: {identity}/{language}")
            for target in original.get("expected_targets", []):
                matches = [status for status in statuses[target["language"]]
                           if status.get("task_id") == target["task_id"]
                           and status.get("source_unit_id") == target["source_unit_id"]
                           and status.get("anchor_id") == original.get("anchor_id")]
                if len(matches) != 1:
                    raise ValueError(f"incomplete or duplicate target status: {identity}")
            for language, entries in evidence.items():
                for span in entries:
                    if not span.get("text", "").strip() or span.get("match_status") != "exact":
                        raise ValueError("published evidence must be a nonempty exact source quotation")
                    if span.get("language") not in {None, language}:
                        raise ValueError("evidence language metadata differs from its language key")
                    if span.get("is_original") is True:
                        if (span.get("validation_status") != "source_exact"
                                or span["source_unit_id"] != original["source_unit_id"]
                                or span["text"] != original.get("source_evidence_text")
                                or span.get("anchor_id") != original.get("anchor_id")):
                            raise ValueError("original evidence differs from frozen source anchor")
                    else:
                        matched = [status for status in statuses[language]
                                   if status.get("alignment_id") == span.get("alignment_id")
                                   and status.get("source_unit_id") == span.get("source_unit_id")
                                   and status.get("status") == "matched"
                                   and status.get("exact_match") is True]
                        if (len(matched) != 1 or span.get("validation_status") != "pending_review"
                                or matched[0].get("validation_status") != "pending_review"):
                            raise ValueError("generated evidence lacks its pending-review exact-match status")
            row["evidence_all_languages"] = evidence
            row["evidence_alignment_status"] = statuses
            rows[identity] = row
    if rows.keys() != frozen.keys():
        raise ValueError("alignment CSV is incomplete: QA IDs differ from the frozen run inventory")
    inputs = {name: {"sha256": _hash(run_dir / name), "bytes": (run_dir / name).stat().st_size}
              for name in ("run.json", "qa_index.jsonl", "qa_evidence.csv")}
    return rows, {"model": run.get("model"), "provider": run.get("provider"),
                  "run_status": run["status"], "input_files": inputs,
                  "frozen_qa_count": len(frozen), "prompt_sha256": run.get("prompt_sha256")}


def _source_documents(old_output: Path, manifest: dict, ids: set[str]) -> dict[str, dict]:
    documents = {}
    for relative in sorted(manifest["files"]):
        if relative.startswith("data/corpus/"):
            table = pq.read_table(old_output / relative, columns=["_id", "language", "text"],
                                  filters=[("_id", "in", sorted(ids))])
            for row in table.to_pylist():
                if row["_id"] in documents:
                    raise ValueError("duplicate source corpus ID")
                documents[row["_id"]] = row
    if documents.keys() != ids:
        raise ValueError("aligned evidence refers to a document missing from the published corpus")
    return documents


def _readme(original: str, repo_id: str) -> str:
    section = f"""## Multilingual supporting evidence

The `qas` configuration now includes native nested `evidence_all_languages` and
`evidence_alignment_status` fields with the five keys `en`, `de`, `es`, `fr`, and
`zh`. Each language contains a list of span/status objects; an unavailable language
has an empty list. Original questions, answers, metadata, and corpus records are
preserved. Arabic and Russian corpus versions remain available where previously
published; this alignment run did not create evidence in those languages.

For easy reading, `evidence_en`, `evidence_de`, `evidence_es`, `evidence_fr`, and
`evidence_zh` contain lists of the exact passage strings. A corpus-specific
[readable CSV](exports/qa_evidence_readable.csv) contains these fields alongside
questions and answers. In its readable language columns, one span is plain text,
no span is an empty cell, and multiple spans use a JSON list. Native Parquet
columns always remain lists. No span strings are joined, translated, or normalized.

Original supporting text is marked `source_exact`. Generated spans are literal
matches in the linked target document and remain `pending_review`: exact quotation
does **not** establish semantic equivalence or independent human review. The
language-specific status lists retain unsuccessful/uncertain alignments and their
provenance. `evidence_alignment_eligibility` explains excluded QAs (for example,
missing exact original support or a quarantined language label); such questions
remain in the dataset without invented evidence. The contract is
`{CONTRACT}`. Legacy `evidence`/`evidence_language` columns and the separate
`evidence` configuration keep their previous interpretation.

```python
from datasets import load_dataset

qas = load_dataset("{repo_id}", "qas", split="train")
row = next(row for row in qas if row["evidence_all_languages"]["en"])
print(row["question"])
print(row["evidence_all_languages"]["en"][0]["text"])
print(row["evidence_en"])
```

This update adds supporting passages only. It creates no retrieval chunks or new
relevance judgments, and does not change `source_qrels`.

"""
    lines = original.splitlines(keepends=True)
    start = 0
    if lines and lines[0].strip() == "---":
        start = next((index + 1 for index, line in enumerate(lines[1:], 1)
                      if line.strip() == "---"), 0)
        if not start:
            raise ValueError("unterminated README metadata header")
    title = next((index for index in range(start, len(lines)) if lines[index].startswith("# ")), None)
    insertion = title + 1 if title is not None else start
    return "".join(lines[:insertion]) + "\n" + section + "".join(lines[insertion:])


def prepare_alignment_update(old_output: Path, run_dir: Path, output_dir: Path, *,
                             corpus: str, repo_id: str, revision: str,
                             alignment_data: tuple[dict[str, dict], dict] | None = None) -> dict:
    """Prepare changed files only; preserve every old QA field and untouched hash."""
    old_output, run_dir, output_dir = Path(old_output), Path(run_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("patch directory is not empty; choose a new output directory")
    old_manifest = _manifest(old_output)
    if old_manifest.get("selected_corpora") != [corpus]:
        raise ValueError("source release does not match the requested corpus")
    all_rows, provenance = alignment_data or load_alignment_csv(run_dir)
    aligned = {identity: row for identity, row in all_rows.items() if row["corpus"] == corpus}
    qa_paths = sorted(name for name in old_manifest["files"] if name.startswith("data/qas/"))
    tables = [(name, pq.read_table(old_output / name)) for name in qa_paths]
    original_ids = [identity for _, table in tables for identity in table["_id"].to_pylist()]
    if len(set(original_ids)) != len(original_ids) or set(original_ids) != set(aligned):
        raise ValueError("published QA IDs do not exactly match the alignment CSV corpus")
    new_names = {field.name for field in NEW_FIELDS}
    if any(new_names & set(table.column_names) for _, table in tables):
        raise ValueError("QA release already has alignment fields; refusing a silent overwrite")
    needed = {span["source_unit_id"] for row in aligned.values()
              for entries in row["evidence_all_languages"].values() for span in entries}
    documents = _source_documents(old_output, old_manifest, needed)
    for row in aligned.values():
        for language, entries in row["evidence_all_languages"].items():
            for span in entries:
                document = documents[span["source_unit_id"]]
                if document["language"] != language or span["text"] not in document["text"]:
                    raise ValueError("span is not a verbatim quotation in the published document/language")
    for _, table in tables:
        for original in table.select(["_id", "question", "answer", "corpus", "question_language",
                                      "source_unit_id"]).to_pylist():
            addition = aligned[original["_id"]]
            for key in ("question", "answer", "corpus", "question_language", "source_unit_id"):
                if (original[key] or "") != addition[key]:
                    raise ValueError(f"alignment data changed published QA field: {original['_id']}/{key}")

    output_dir.mkdir(parents=True, exist_ok=True)
    changed = []
    readable = []
    for relative, table in tables:
        additions = []
        originals = table.to_pylist()
        for original in originals:
            row = aligned[original["_id"]]
            evidence = row["evidence_all_languages"]
            addition = {"evidence_all_languages": evidence,
                        "evidence_alignment_status": row["evidence_alignment_status"],
                        "evidence_alignment_eligibility": row["eligibility_status"],
                        "evidence_alignment_contract": CONTRACT,
                        **{f"evidence_{language}": [span["text"] for span in evidence[language]]
                           for language in LANGUAGES}}
            additions.append(addition)
            readable.append({"qa_id": original["_id"], **{key: original.get(key, "") for key in (
                "corpus", "question", "answer", "question_language", "source_unit_id", "persona",
                "mode", "generator_model_id")}, **addition})
        updated = table
        for field in NEW_FIELDS:
            updated = updated.append_column(field, pa.array([row[field.name] for row in additions], type=field.type))
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(updated, destination, compression="zstd")
        restored = pq.read_table(destination)
        if not restored.select(table.column_names).equals(table, check_metadata=True):
            raise ValueError("Parquet round trip changed original QA fields or metadata")
        changed.append(relative)
    csv_path = "exports/qa_evidence_readable.csv"
    (output_dir / "exports").mkdir(parents=True, exist_ok=True)
    with (output_dir / csv_path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in readable:
            cells = {}
            for key, value in row.items():
                if key in {f"evidence_{language}" for language in LANGUAGES}:
                    cells[key] = "" if not value else value[0] if len(value) == 1 else _json(value)
                else:
                    cells[key] = _json(value) if isinstance(value, (dict, list)) else value
            writer.writerow(cells)
    changed.append(csv_path)
    report_block = {
        "contract": CONTRACT, "languages": list(LANGUAGES), "corpus": corpus,
        "base_repo_id": repo_id, "base_revision": revision,
        "base_manifest_sha256": _hash(old_output / "manifest.json"),
        **provenance, "qa_count": len(aligned),
        "eligibility_counts": dict(Counter(row["eligibility_status"] for row in aligned.values())),
        "qa_with_any_evidence": sum(any(row["evidence_all_languages"].values()) for row in aligned.values()),
        "span_counts_by_language": {language: sum(len(row["evidence_all_languages"][language])
                                                  for row in aligned.values()) for language in LANGUAGES},
        "validation_counts": dict(Counter(span["validation_status"] for row in aligned.values()
                                           for entries in row["evidence_all_languages"].values() for span in entries)),
        "status_counts": dict(Counter(status["status"] for row in aligned.values()
                                       for entries in row["evidence_alignment_status"].values() for status in entries)),
        "original_qa_fields_preserved": True, "source_text_exactness_rechecked": True,
        "independent_semantic_review": False, "unchanged_configs": ["corpus", "evidence", "source_qrels"],
        "readable_csv": csv_path,
    }
    old_report = json.loads((old_output / "report.json").read_text(encoding="utf-8"))
    old_report["evidence_alignment"] = report_block
    (output_dir / "report.json").write_text(_json(old_report) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(_readme((old_output / "README.md").read_text(encoding="utf-8"), repo_id), encoding="utf-8")
    changed.extend(["report.json", "README.md"])
    manifest = json.loads(_json(old_manifest))
    manifest["evidence_alignment"] = report_block
    for relative in changed:
        path = output_dir / relative
        manifest["files"][relative] = {"sha256": _hash(path), "bytes": path.stat().st_size}
    (output_dir / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "changed_files": sorted([*changed, "manifest.json"]),
            "evidence_alignment": report_block}
