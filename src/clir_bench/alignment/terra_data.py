"""Freeze the supported-language Terra workload and index every published QA.

This preparation reads verified, immutable exports and the frozen Luna workload.
It never changes prior pilot artifacts, calls a model, or updates Hugging Face.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import (
    QUARANTINE,
    RELEASE_MANIFEST_SHA256,
    RELEASES,
    _load_release,
    digest_file,
    digest_text,
    dumps,
    select_anchor,
    write_immutable,
)

MODEL = "gpt-6.1-terra"
LANGUAGES = frozenset({"en", "de", "es", "fr", "zh"})
CONTRACT = "terra_evidence_alignment_v1"
LUNA_FULL_TASKS_SHA256 = "84db3080ada9971ee398cf427db4ba1c52c1421b5202fea1362def1913b88d9e"
UNIT_TYPES = {"eurlex": {"article", "annex"}, "un": {"document"}, "chemistry": {"title_abstract"}}
COMPACT_METADATA_FIELDS = (
    "generator_model_id", "generator_model_name", "candidate_id", "source_document_id",
    "generation_target_id", "context_language", "question_language", "mode", "persona",
    "persona_metadata_json", "provenance_json", "grading_status", "quality_audit_status",
    "faithfulness_verifier_model", "quality_verifier_model", "candidate_rank", "is_best",
    "answer_is_translation", "evidence_is_translation", "multi_article", "multi_article_declared",
    "multi_article_conflict", "cross_act", "cross_act_declared", "multi_source_unit",
    "num_involved_articles", "involved_article_ids", "involved_source_unit_ids",
    "unresolved_article_references", "faith_overall", "qual_overall", "total_score",
    "support_match_status", "generation_support_match_status", "supporting_text_origin",
    "evidence_language", "support_start", "support_end",
)
EXPECTED_COUNTS = {
    "published_qas": 10691, "eligible_qas": 8931, "source_documents": 636,
    "source_document_language_versions": 639, "distinct_source_spans": 4930,
    "document_pair_groups": 1532, "target_language_alignments": 12855,
}


def filter_tasks(tasks: list[dict]) -> list[dict]:
    """Filter source and target, retaining original IDs and exact source strings."""
    result = []
    seen = set()
    for task in tasks:
        if task["source_language"] not in LANGUAGES or task["target_language"] not in LANGUAGES:
            continue
        if task["source_unit_id"] in QUARANTINE or task["target_unit_id"] in QUARANTINE:
            raise ValueError("The frozen eligible workload contains a quarantined source")
        if task["source_language"] == task["target_language"]:
            raise ValueError("Alignment target must have a different language")
        if task["task_id"] in seen:
            raise ValueError("Duplicate frozen task identifier")
        seen.add(task["task_id"])
        result.append(task)
    return sorted(result, key=lambda task: task["task_id"])


def _task_qa_map(corpus: str, tasks: list[dict], documents: dict[str, dict]) -> dict:
    mappings: dict[str, dict] = {}
    for task in tasks:
        if task["corpus"] != corpus:
            continue
        for side in ("source", "target"):
            unit = task[f"{side}_unit_id"]
            record = documents[unit]
            if task[f"{side}_text"] != record["text"] or task[f"{side}_language"] != record["language"]:
                raise ValueError(f"Frozen task disagrees with the published {side}: {unit}")
            if digest_text(record["text"]) != task[f"{side}_text_sha256"]:
                raise ValueError(f"Frozen task text hash mismatch: {unit}")
            if record["unit_type"] not in UNIT_TYPES[corpus]:
                raise ValueError(f"Unsupported source unit class: {unit}")
        seen_anchor_ids = set()
        if not task["anchors"]:
            raise ValueError("Frozen task has no source anchors")
        for anchor in task["anchors"]:
            if (not isinstance(anchor["text"], str) or not anchor["text"].strip()
                    or anchor["text"] not in task["source_text"]):
                raise ValueError("Frozen anchor is not a nonempty literal source passage")
            expected_id = "a_" + digest_text(dumps([corpus, task["source_unit_id"], anchor["text"]]))
            if anchor["anchor_id"] != expected_id or anchor["anchor_id"] in seen_anchor_ids:
                raise ValueError("Frozen anchor identifier mismatch or duplicate")
            seen_anchor_ids.add(anchor["anchor_id"])
            target = {"language": task["target_language"], "source_unit_id": task["target_unit_id"],
                      "task_id": task["task_id"]}
            for qa_id in anchor["qa_ids"]:
                identity = (task["source_unit_id"], task["source_language"], anchor["anchor_id"], anchor["text"])
                entry = mappings.setdefault(qa_id, {"identity": identity, "targets": []})
                if entry["identity"] != identity:
                    raise ValueError(f"One QA maps to different source anchors: {qa_id}")
                if any(t["source_unit_id"] == target["source_unit_id"] for t in entry["targets"]):
                    raise ValueError(f"Duplicate target for QA {qa_id}")
                entry["targets"].append(target)
    return mappings


def build_qa_index(corpus: str, qas: list[dict], qrels: list[dict], documents: dict[str, dict],
                   tasks: list[dict]) -> list[dict]:
    """Retain all QA identities, distinguishing eligible and excluded source support."""
    mappings = _task_qa_map(corpus, tasks, documents)
    qrel_targets: dict[str, set[str]] = defaultdict(set)
    for relation in qrels:
        target = relation["corpus_id"]
        if (relation["relation"] == "direct_translation" and target not in QUARANTINE
                and documents[target]["language"] in LANGUAGES):
            qrel_targets[relation["query_id"]].add(target)
    seen_qa_ids = set()
    indexed = []
    _, repository, revision = RELEASES[corpus]
    for qa in sorted(qas, key=lambda row: row["_id"]):
        qa_id = qa["_id"]
        if qa_id in seen_qa_ids:
            raise ValueError(f"Duplicate published QA: {qa_id}")
        seen_qa_ids.add(qa_id)
        source = documents[qa["source_unit_id"]]
        language = source["language"]
        anchor = select_anchor(qa, source["text"])
        status = "eligible"
        if qa["source_unit_id"] in QUARANTINE:
            status = "quarantined_source_language"
        elif language not in LANGUAGES:
            status = "unsupported_source_language"
        elif source["unit_type"] not in UNIT_TYPES[corpus]:
            status = "unsupported_source_unit_type"
        elif language != qa["context_language"]:
            status = "source_language_metadata_conflict"
        elif anchor is None:
            status = "no_exact_source_passage"
        elif not qrel_targets[qa_id]:
            status = "no_supported_translation_target"
        mapping = mappings.get(qa_id)
        targets, anchors, multilingual = [], [], {}
        anchor_id = passage = origin = None
        if status == "eligible":
            if mapping is None:
                raise ValueError(f"Eligible published QA is missing from frozen workload: {qa_id}")
            assert anchor is not None
            passage, origin = anchor
            unit, mapped_language, anchor_id, mapped_passage = mapping["identity"]
            if (unit, mapped_language, mapped_passage) != (qa["source_unit_id"], language, passage):
                raise ValueError(f"Frozen anchor differs from exact source evidence/answer: {qa_id}")
            targets = sorted(mapping["targets"], key=lambda item: (item["language"], item["source_unit_id"]))
            if {target["source_unit_id"] for target in targets} != qrel_targets[qa_id]:
                raise ValueError(f"Frozen targets differ from published accepted translations: {qa_id}")
            anchors = [{"anchor_id": anchor_id, "text": passage, "origin": origin,
                        "source_unit_id": unit, "language": language}]
            multilingual = {language: [{
                "language": language, "source_unit_id": unit, "text": passage, "origin": origin,
                "match_status": "exact", "alignment_status": "source_exact", "is_original": True,
            }]}
        elif mapping is not None:
            raise ValueError(f"Ineligible QA appears in frozen alignment workload: {qa_id}")
        metadata = {key: qa[key] for key in COMPACT_METADATA_FIELDS if key in qa}
        metadata["published_source"] = {
            "dataset": repository, "revision": revision, "config": "qas", "split": "train",
            "qa_id": qa_id, "release_manifest_sha256": RELEASE_MANIFEST_SHA256[corpus],
        }
        indexed.append({
            "qa_id": qa_id, "corpus": corpus, "question": qa["question"], "answer": qa["answer"],
            "question_language": qa["question_language"], "source_unit_id": qa["source_unit_id"],
            "source_language": language, "anchor_id": anchor_id, "source_evidence_text": passage,
            "source_evidence_origin": origin, "eligibility_status": status,
            "metadata_json": dumps(metadata), "anchors": anchors, "expected_targets": targets,
            "evidence_all_languages": multilingual,
        })
    if set(mappings) - seen_qa_ids:
        raise ValueError("Frozen workload references a missing published QA")
    return indexed


def _counts(tasks: list[dict], qas: list[dict]) -> dict:
    return {
        "published_qas": len(qas), "eligible_qas": sum(row["eligibility_status"] == "eligible" for row in qas),
        "source_documents": len({(task["corpus"], task["document_id"]) for task in tasks}),
        "source_document_language_versions": len({task["source_unit_id"] for task in tasks}),
        "distinct_source_spans": len({anchor["anchor_id"] for task in tasks for anchor in task["anchors"]}),
        "document_pair_groups": len(tasks),
        "target_language_alignments": sum(len(task["anchors"]) for task in tasks),
    }


def prepare(output_dir: Path, root: Path, *, source_dir: Path | None = None) -> dict:
    root, output_dir = Path(root).resolve(), Path(output_dir)
    source_dir = Path(source_dir) if source_dir is not None else root / "data/evidence_alignment/pilot-20261105"
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("Terra preparation must not overwrite the Luna pilot")
    full_path = source_dir / "full_tasks.jsonl"
    if digest_file(full_path) != LUNA_FULL_TASKS_SHA256:
        raise ValueError("Frozen Luna full_tasks.jsonl SHA-256 does not match the approved workload")
    with full_path.open(encoding="utf-8") as stream:
        frozen_tasks = [json.loads(line) for line in stream if line.strip()]
    tasks = filter_tasks(frozen_tasks)
    qa_index, provenance = [], {}
    inventory_path = source_dir / "inventory.json"
    old_inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else None
    for corpus in RELEASES:
        qas, qrels, documents, source_provenance = _load_release(root, corpus)
        if old_inventory is not None:
            recorded = old_inventory["provenance"][corpus]
            for field in ("dataset", "revision", "manifest_sha256", "verified_files"):
                if recorded[field] != source_provenance[field]:
                    raise ValueError(f"Frozen pilot provenance differs from the approved {corpus} release")
        provenance[corpus] = source_provenance
        qa_index.extend(build_qa_index(corpus, qas, qrels, documents, tasks))
    if len({row["qa_id"] for row in qa_index}) != len(qa_index):
        raise ValueError("Published QA identifiers collide across corpora")
    qa_index.sort(key=lambda row: (row["corpus"], row["qa_id"]))
    counts = _counts(tasks, qa_index)
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Supported-language workload differs from approved inventory: {counts}")
    by_corpus = {corpus: {
        **_counts([task for task in tasks if task["corpus"] == corpus], [qa for qa in qa_index if qa["corpus"] == corpus]),
        "eligibility_statuses": dict(Counter(qa["eligibility_status"] for qa in qa_index if qa["corpus"] == corpus)),
    } for corpus in RELEASES}
    task_bytes = "".join(dumps(task) + "\n" for task in tasks).encode("utf-8")
    qa_bytes = "".join(dumps(qa) + "\n" for qa in qa_index).encode("utf-8")
    manifest = {
        "contract": CONTRACT, "model": MODEL, "transport": "direct_openai", "languages": sorted(LANGUAGES),
        "source_full_tasks_sha256": LUNA_FULL_TASKS_SHA256, "tasks_sha256": digest_text(task_bytes.decode("utf-8")),
        "qa_index_sha256": digest_text(qa_bytes.decode("utf-8")), "counts": counts, "by_corpus": by_corpus,
        "provenance": provenance, "quarantined_source_units": QUARANTINE,
        "source_evidence_policy": "Exact nonempty evidence first, otherwise exact nonempty answer; no normalization. Original entries exist only for eligible source passages.",
        "metadata_policy": "Compact unchanged QA metadata plus immutable published dataset/revision/qa_id lookup; complete documents remain in frozen releases and task inputs.",
        "source_inventory_sha256": digest_file(inventory_path) if inventory_path.exists() else None,
    }
    artifacts = {"tasks.jsonl": task_bytes, "qa_index.jsonl": qa_bytes,
                 "preparation.json": (dumps(manifest) + "\n").encode("utf-8")}
    # Precheck every existing artifact before creating any new one.
    for name, content in artifacts.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Refusing to replace a nonidentical Terra preparation artifact: {path}")
    for name, content in artifacts.items():
        write_immutable(output_dir / name, content)
    return {"output_dir": str(output_dir), "model": MODEL, "languages": sorted(LANGUAGES),
            "counts": counts, "by_corpus": by_corpus,
            "artifacts": {name: digest_file(output_dir / name) for name in artifacts}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    print(dumps(prepare(args.output, args.root, source_dir=args.source_dir)))


if __name__ == "__main__":
    main()
