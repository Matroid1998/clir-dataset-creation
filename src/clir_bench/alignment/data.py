"""Prepare a reproducible evidence-alignment pilot from immutable HF releases.

The source of chemistry text is the pinned BASF Hugging Face snapshot, never
its derived CSV. Published direct-translation links are checked against the
accepted pair graph before preparing any model request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

RELEASES = {
    "eurlex": ("eurlex-generated-qa-full-20261104", "MehdiAstaraki/eurlex-generated-qa", "8b5d1bd5604e7ebb7c27bc6d877b6b3ef712a006"),
    "un": ("un-generated-qa-genres-20261104", "MehdiAstaraki/un-generated-qa", "79b190ab68940cfed6236c372cbfe53404339c05"),
    "chemistry": ("chemistry-generated-qa", "MehdiAstaraki/chemistry-generated-qa", "d390d599ef42870bc5fe2349403ba36924546022"),
}
RELEASE_MANIFEST_SHA256 = {
    "eurlex": "58a799882ddc5f19d7e0d5c67e798742bd65bb9b382cfc24a1831eee5006319e",
    "un": "fbd017a2520caa485de9298420120c95cd7d5ec7c61645aa822f2709bee42dd4",
    "chemistry": "a1ecb5fc04d02a3e11495d30b2fe165f2eb025a1b44793a81d25035a965c2cdd",
}
CHEMISTRY_REPO = "BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs"
CHEMISTRY_REVISION = "24c1f894591a5aa6f745ac233d5091fdbc30ad3c"
CHEMISTRY_FILE = "data/train-00000-of-00001.parquet"
CHEMISTRY_SHA256 = "50304cf8fc8280efbf895136a4084e02951b4a77fc64e179f0a6290120a603c4"
QUARANTINE = {"EP-1732767-A1_es": "Text is German despite the upstream Spanish source-language label."}
SUPPORTED = {"eurlex": {"en", "fr", "de", "es"}, "un": {"en", "fr", "es", "zh", "ar", "ru"}, "chemistry": {"en", "fr", "de", "es", "zh"}}
VERSION = "evidence_alignment_pilot_v1"


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable(path: Path, content: bytes) -> None:
    """Permit exact replay, never change the selected data under an existing run."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to replace nonidentical pilot artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def select_anchor(qa: dict, source_text: str) -> tuple[str, str] | None:
    """Evidence takes precedence; whitespace and Unicode are never normalized."""
    for field in ("evidence", "answer"):
        text = qa.get(field)
        if isinstance(text, str) and text.strip() and text in source_text:
            return text, field
    return None


def _load_release(root: Path, corpus: str) -> tuple[list[dict], list[dict], dict[str, dict], dict]:
    folder, repo, revision = RELEASES[corpus]
    release = root / "data/hf_export" / folder
    if digest_file(release / "manifest.json") != RELEASE_MANIFEST_SHA256[corpus]:
        raise ValueError(f"The local {corpus} manifest differs from the approved release")
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    hashes: dict[str, dict] = {}

    def shards(config: str) -> Iterable[Path]:
        paths = sorted((release / "data" / config).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No {config} shards in {release}")
        for path in paths:
            name = path.relative_to(release).as_posix()
            observed = digest_file(path)
            expected = manifest["files"][name]["sha256"]
            if observed != expected:
                raise ValueError(f"Release hash mismatch: {path}")
            hashes[name] = {"sha256": observed, "bytes": path.stat().st_size}
            yield path

    qas = [row for path in shards("qas") for row in pq.read_table(path).to_pylist()]
    qrels = [row for path in shards("source_qrels") for row in pq.read_table(path).to_pylist()]
    needed = sorted({q["source_unit_id"] for q in qas} | {r["corpus_id"] for r in qrels})
    documents: dict[str, dict] = {}
    for path in shards("corpus"):
        for row in pq.read_table(path, filters=[("_id", "in", needed)]).to_pylist():
            if row["_id"] in documents:
                raise ValueError(f"Duplicate corpus ID: {row['_id']}")
            documents[row["_id"]] = row
    if set(needed) != set(documents):
        raise ValueError(f"Missing referenced {corpus} source documents")
    return qas, qrels, documents, {
        "dataset": repo, "revision": revision, "local_release": str(release.relative_to(root)),
        "manifest_sha256": digest_file(release / "manifest.json"), "verified_files": hashes,
    }


def _chemistry_snapshot() -> tuple[dict[str, str], dict[str, set[str]], dict]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        CHEMISTRY_REPO, CHEMISTRY_FILE, repo_type="dataset", revision=CHEMISTRY_REVISION,
        local_files_only=True,
    ))
    observed = digest_file(path)
    if observed != CHEMISTRY_SHA256:
        raise ValueError("Pinned chemistry HF snapshot failed its SHA-256 check")
    texts: dict[str, str] = {}
    neighbors: dict[str, set[str]] = defaultdict(set)
    accepted = 0
    for row in pq.read_table(path).to_pylist():
        if row["pair_ok"] is not True or row["judge_verdict"] not in {"exact", "high"}:
            continue
        sl, tl = row["source_language"], row["target_language"]
        if sl not in SUPPORTED["chemistry"] or tl not in SUPPORTED["chemistry"]:
            continue
        if sl == tl:
            raise ValueError("Accepted chemistry pair has identical languages")
        accepted += 1
        source, target = f"{row['doc_id']}_{sl}", f"{row['doc_id']}_{tl}"
        for unit, text in ((source, row["source_text"]), (target, row["target_text"])):
            if unit in texts and texts[unit] != text:
                raise ValueError(f"Conflicting HF source texts for {unit}")
            texts[unit] = text
        neighbors[source].add(target)
        neighbors[target].add(source)
    return texts, neighbors, {
        "origin": "hugging_face", "dataset": CHEMISTRY_REPO, "revision": CHEMISTRY_REVISION,
        "filename": CHEMISTRY_FILE, "sha256": observed, "accepted_pairs": accepted,
        "acceptance": "pair_ok is true and judge_verdict in [exact, high]; direct edges only",
    }


def _document_id(corpus: str, row: dict) -> str:
    return row["parent_id"] if corpus == "chemistry" else row["source_id"]


def _genre(corpus: str, row: dict) -> str:
    if corpus != "un":
        return row["unit_type"]
    source = row["source_id"].lower()
    if "sr_" in source or "pv_" in source:
        return "meeting"
    if "/res/" in source or "/dec/" in source:
        return "resolution"
    return "letter"


def _candidate(corpus: str, row: dict) -> dict:
    return {
        "corpus": corpus, "source_unit_id": row["_id"], "document_id": _document_id(corpus, row),
        "source_language": row["language"], "source_text": row["text"], "genre": _genre(corpus, row),
    }


def choose_quantiles(candidates: list[dict], quantiles: Iterable[float], used_document_ids: set[str]) -> list[dict]:
    """Nearest rank by Unicode character length; stable ID resolves every tie."""
    ordered = sorted(candidates, key=lambda c: (len(c["source_text"]), c["source_unit_id"]))
    chosen = []
    for quantile in quantiles:
        if not ordered or not 0 <= quantile <= 1:
            raise ValueError("Quantile selection needs candidates and values in [0, 1]")
        ideal = quantile * (len(ordered) - 1)
        positions = sorted(range(len(ordered)), key=lambda i: (abs(i - ideal), ordered[i]["source_unit_id"]))
        for position in positions:
            row = ordered[position]
            if row["document_id"] not in used_document_ids:
                used_document_ids.add(row["document_id"])
                chosen.append({**row, "selection_quantile": quantile})
                break
        else:
            raise ValueError("Insufficient distinct documents for requested sample")
    return chosen


def select_sample(candidates: dict[str, list[dict]]) -> list[dict]:
    selected = []
    for corpus, rows in candidates.items():
        used: set[str] = set()
        by_language = {language: [r for r in rows if r["source_language"] == language] for language in SUPPORTED[corpus]}
        if corpus == "eurlex":
            quotas = [("en", [.1, .5, .9]), ("fr", [.1, .5, .9]), ("de", [.25, .75]), ("es", [.25, .75])]
        elif corpus == "chemistry":
            quotas = [(language, [.25, .75]) for language in ("en", "fr", "de", "es", "zh")]
        else:
            quotas = [("en", [.1, .5, .9]), ("fr", [.25, .75]), ("es", [.25, .75])]
        for language, quantiles in quotas:
            selected.extend(choose_quantiles(by_language[language], quantiles, used))
        if corpus == "un":
            fixed = [r for r in by_language["zh"] if r["document_id"] == "2003/cedaw/c/sr_627"]
            if len(fixed) != 1:
                raise ValueError("Required UN source-finding meeting is unavailable")
            selected.append({**fixed[0], "selection_reason": "source_finding_meeting"})
            used.add(fixed[0]["document_id"])
            selected.extend(choose_quantiles([r for r in by_language["zh"] if r["genre"] == "meeting"], [.9], used))
            selected.extend(choose_quantiles([r for r in by_language["zh"] if r["genre"] == "resolution"], [.5], used))
        if len(used) != 10:
            raise AssertionError(f"Expected 10 distinct {corpus} documents; selected {len(used)}")
    return selected


def prepare(output_dir: Path, root: Path) -> dict:
    output_dir, root = Path(output_dir), Path(root).resolve()
    all_tasks: list[dict] = []
    candidates: dict[str, list[dict]] = {}
    corpus_stats: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    exclusion_records = []
    discrepancies = []
    for corpus in RELEASES:
        qas, qrels, documents, release_provenance = _load_release(root, corpus)
        provenance[corpus] = release_provenance
        qrels_by_qa: dict[str, set[str]] = defaultdict(set)
        for relation in qrels:
            if relation["relation"] == "direct_translation":
                qrels_by_qa[relation["query_id"]].add(relation["corpus_id"])
        chemistry_texts, chemistry_neighbors = {}, {}
        if corpus == "chemistry":
            chemistry_texts, chemistry_neighbors, upstream = _chemistry_snapshot()
            provenance[corpus]["authoritative_corpus"] = upstream
            for unit, record in documents.items():
                if chemistry_texts.get(unit) != record["text"]:
                    raise ValueError(f"Published chemistry text disagrees with pinned HF snapshot: {unit}")
                # Assign the authoritative source string explicitly, retaining published IDs.
                record["text"] = chemistry_texts[unit]
        anchor_map: dict[tuple[str, str], dict] = {}
        target_map: dict[str, set[str]] = {}
        reasons = Counter()
        eligible_qa_ids = set()
        for qa in sorted(qas, key=lambda q: q["_id"]):
            unit = qa["source_unit_id"]
            source = documents[unit]
            if corpus == "chemistry" and qrels_by_qa[qa["_id"]] != chemistry_neighbors.get(unit, set()):
                raise ValueError(f"Published accepted pair links disagree with HF source: {qa['_id']}")
            if source["language"] != qa["context_language"]:
                raise ValueError(f"QA/source language metadata conflict: {qa['_id']}")
            if qa.get("evidence_language") and qa["evidence_language"] != source["language"]:
                discrepancies.append({"corpus": corpus, "qa_id": qa["_id"], "source_unit_id": unit,
                                      "source_language": source["language"], "evidence_language": qa["evidence_language"],
                                      "action": "Quarantine known source-label errors; otherwise select only a literal source passage and retain original labels in metadata."})
            reason = None
            anchor = select_anchor(qa, source["text"])
            if unit in QUARANTINE:
                reason = "quarantined_source_language"
            elif anchor is None:
                reason = "no_exact_source_passage"
            targets = {target for target in qrels_by_qa[qa["_id"]] if target not in QUARANTINE}
            for target in targets:
                if documents[target]["language"] == source["language"]:
                    raise ValueError("Direct translation has the same language as the source")
                if documents[target]["language"] not in SUPPORTED[corpus]:
                    raise ValueError("Unsupported target language")
                if _document_id(corpus, documents[target]) != _document_id(corpus, source):
                    raise ValueError("Translation target belongs to a different document")
            if not targets and reason is None:
                reason = "no_accepted_target"
            if reason:
                reasons[reason] += 1
                exclusion_records.append({"corpus": corpus, "qa_id": qa["_id"], "source_unit_id": unit, "reason": reason})
                continue
            assert anchor is not None
            if unit in target_map and target_map[unit] != targets:
                raise ValueError("QAs on one source version disagree about accepted translation targets")
            target_map[unit] = targets
            passage, origin = anchor
            key = (unit, passage)
            record = anchor_map.setdefault(key, {
                "anchor_id": "a_" + digest_text(dumps([corpus, unit, passage])), "text": passage,
                "qa_ids": [], "origins": set(), "question_languages": set(), "modes": set(), "qa_links": [],
            })
            record["qa_ids"].append(qa["_id"])
            record["origins"].add(origin)
            record["question_languages"].add(qa["question_language"])
            record["modes"].add(qa["mode"])
            record["qa_links"].append({"qa_id": qa["_id"], "origin": origin, **{k: qa.get(k) for k in ("question", "answer", "generator_model_id", "persona", "mode", "question_language", "context_language", "evidence_language")}})
            eligible_qa_ids.add(qa["_id"])
        by_source: dict[str, list[dict]] = defaultdict(list)
        for (unit, _), anchor in anchor_map.items():
            for key in ("origins", "question_languages", "modes"):
                anchor[key] = sorted(anchor[key])
            by_source[unit].append(anchor)
        candidates[corpus] = [_candidate(corpus, documents[unit]) for unit in sorted(by_source)]
        tasks = []
        for source_unit in sorted(by_source):
            source = documents[source_unit]
            for target_unit in sorted(target_map[source_unit]):
                target = documents[target_unit]
                tasks.append({
                    "task_id": "t_" + digest_text(dumps([VERSION, corpus, source_unit, target_unit])),
                    "corpus": corpus, "source_unit_id": source_unit, "target_unit_id": target_unit,
                    "document_id": _document_id(corpus, source), "source_language": source["language"], "target_language": target["language"],
                    "source_text": source["text"], "target_text": target["text"],
                    "source_title": source["title"], "target_title": target["title"],
                    "source_text_sha256": digest_text(source["text"]), "target_text_sha256": digest_text(target["text"]),
                    "genre": _genre(corpus, source), "anchors": sorted(by_source[source_unit], key=lambda a: a["anchor_id"]),
                })
        corpus_stats[corpus] = {
            "published_qas": len(qas), "eligible_qas": len(eligible_qa_ids), "excluded_qas": dict(sorted(reasons.items())),
            "distinct_source_spans": len(anchor_map), "source_document_language_versions": len(by_source),
            "source_documents": len({_document_id(corpus, documents[unit]) for unit in by_source}),
            "document_pair_groups": len(tasks), "target_language_alignments": sum(len(task["anchors"]) for task in tasks),
            "source_versions_by_language": dict(sorted(Counter(documents[unit]["language"] for unit in by_source).items())),
        }
        all_tasks.extend(tasks)
    selections = select_sample(candidates)
    selected_keys = {(row["corpus"], row["document_id"]) for row in selections}
    pilot_tasks = [task for task in all_tasks if (task["corpus"], task["document_id"]) in selected_keys]
    sample_rows = []
    for selection in selections:
        tasks = [task for task in pilot_tasks if task["corpus"] == selection["corpus"] and task["document_id"] == selection["document_id"]]
        sample_rows.append({
            **{key: value for key, value in selection.items() if key != "source_text"},
            "selection_source_characters": len(selection["source_text"]),
            "source_unit_ids": sorted({task["source_unit_id"] for task in tasks}),
            "target_unit_ids": sorted({task["target_unit_id"] for task in tasks}),
            "source_passages": len({anchor["anchor_id"] for task in tasks for anchor in task["anchors"]}),
            "qa_count": len({qid for task in tasks for anchor in task["anchors"] for qid in anchor["qa_ids"]}),
            "document_pair_groups": len(tasks), "target_language_alignments": sum(len(task["anchors"]) for task in tasks),
        })
    totals = {key: sum(stats[key] for stats in corpus_stats.values()) for key in ("published_qas", "eligible_qas", "distinct_source_spans", "source_documents", "source_document_language_versions", "document_pair_groups", "target_language_alignments")}
    if totals["distinct_source_spans"] != 4930 or totals["target_language_alignments"] != 17957 or totals["document_pair_groups"] != 2024:
        raise ValueError(f"Inventory differs from the approved frozen releases: {totals}")
    inventory = {"contract": VERSION, "corpora": corpus_stats, "totals": totals, "provenance": provenance,
                 "quarantined_source_units": QUARANTINE, "language_metadata_discrepancies": discrepancies,
                 "excluded_qas": exclusion_records,
                 "anchor_policy": "Nonempty exact evidence, otherwise nonempty exact answer. No normalization; deduplicate only within source document-language."}
    sample = {"contract": VERSION, "selection_length_measure": "Unicode character count of the complete source text within each source language",
              "selection_policy": "EUR-Lex EN3/FR3 at 10/50/90%, DE2/ES2 at25/75%; UN EN3/FR2/ES2 similarly plus required Chinese source-finding meeting,90th-percentile Chinese meeting,median Chinese resolution; chemistry2 per EN/FR/DE/ES/ZH at25/75%. Closest rank, source ID tie-break, unique language-neutral documents.",
              "documents": sample_rows,
              "totals": {"documents": len(sample_rows), "document_pair_groups": len(pilot_tasks),
                         "distinct_source_spans": len({a["anchor_id"] for t in pilot_tasks for a in t["anchors"]}),
                         "qas": len({q for t in pilot_tasks for a in t["anchors"] for q in a["qa_ids"]}),
                         "target_language_alignments": sum(len(t["anchors"]) for t in pilot_tasks)}}
    artifacts = {"inventory.json": (dumps(inventory) + "\n").encode("utf-8"), "sample.json": (dumps(sample) + "\n").encode("utf-8"),
                 "tasks.jsonl": "".join(dumps(t) + "\n" for t in pilot_tasks).encode("utf-8"),
                 "full_tasks.jsonl": "".join(dumps(t) + "\n" for t in all_tasks).encode("utf-8")}
    # Check all existing artifacts before writing any new one.
    for name, content in artifacts.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Refusing to alter existing pilot preparation: {path}")
    for name, content in artifacts.items():
        write_immutable(output_dir / name, content)
    return {"output_dir": str(output_dir), "inventory": totals, "pilot": sample["totals"],
            "artifacts": {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--prepare", action="store_true", help="Explicit preparation flag; preparation is the only operation.")
    args = parser.parse_args()
    print(dumps(prepare(args.output, args.root)))


if __name__ == "__main__":
    main()
