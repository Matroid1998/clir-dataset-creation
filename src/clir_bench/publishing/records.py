"""Lossless generation readers and a common, explicitly typed QA schema."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow as pa


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def boolean(value) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1"}:
        return True
    if str(value).lower() in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def identifiers(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.startswith("[") else value.split(",")
    return list(dict.fromkeys(str(v).strip() for v in value if str(v).strip()))


def read_generations(corpus: str, spec: dict, root: Path):
    for source in spec.get("generation_sqlite", []):
        path = root / source
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for task, status, payload, error in connection.execute(
                "SELECT task, status, rows, error FROM outcomes ORDER BY task"
            ):
                for position, row in enumerate(json.loads(payload or "[]")):
                    yield (
                        corpus,
                        row,
                        {
                            "file": source,
                            "task": task,
                            "position": position,
                            "outcome_status": status,
                            "outcome_error": error,
                        },
                    )
    for source in spec.get("generation_csv", []):
        with (root / source["path"]).open(encoding="utf-8", newline="") as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                # The comparison CSVs do not store model identity. Require it explicitly.
                model = source["model"]
                if row.get("generator_model_id") not in (None, "", model):
                    raise ValueError(f"model identity conflict in {source['path']}:{line}")
                yield corpus, row, {"file": source["path"], "line": line, "model": model}


def source_identity(corpus: str, row: dict) -> tuple[str, str]:
    language = row.get("context_language") or row["question_language"]
    if corpus == "eurlex":
        return row.get("target_article_id") or row["target_id"], language
    if corpus == "un":
        return row.get("doc_id") or row["document_id"], language
    return row.get("source_document_id") or row["corpus_id"], language


def unit_id(corpus: str, source: str, language: str) -> str:
    return source if corpus == "chemistry" else f"{corpus}:{source}:{language}"


def article_references(row: dict) -> list[str]:
    explicit = identifiers(row.get("articles_involved_eli"))
    if explicit:
        return explicit
    act = row.get("celex_id") or row.get("document_id", "")
    return [
        ref if ":" in ref else f"{act}:{ref}" for ref in identifiers(row.get("articles_involved"))
    ]


def match_support(support: str, text: str) -> tuple[str, int | None, int | None]:
    """Offsets are original-text Unicode code points, end-exclusive, first match."""
    if not support:
        return "missing", None, None
    start = text.find(support)
    if start >= 0:
        return "exact", start, start + len(support)
    if " ".join(support.split()) in " ".join(text.split()) and support.split():
        # Do not invent original-text offsets from a whitespace-normalized match.
        return "whitespace_normalized", None, None
    return "unmatched", None, None


STRING_FIELDS = [
    "_id",
    "corpus",
    "question",
    "answer",
    "question_language",
    "context_language",
    "mode",
    "persona",
    "generator_model_id",
    "generator_model_name",
    "candidate_id",
    "source_unit_id",
    "source_document_id",
    "generation_target_id",
    "evidence",
    "evidence_language",
    "supporting_text",
    "supporting_text_origin",
    "support_match_status",
    "document_text",
    "document_title",
    "generation_source_text",
    "generation_support_match_status",
    "articles_involved_raw",
    "grading_status",
    "quality_audit_status",
    "faithfulness_verifier_model",
    "quality_verifier_model",
    "persona_metadata_json",
    "metadata_json",
    "provenance_json",
]
QA_SCHEMA = pa.schema(
    [(name, pa.string()) for name in STRING_FIELDS]
    + [
        (name, pa.bool_())
        for name in (
            "is_best",
            "answer_is_translation",
            "evidence_is_translation",
            "multi_article",
            "multi_article_declared",
            "multi_article_conflict",
            "cross_act",
            "cross_act_declared",
            "multi_source_unit",
        )
    ]
    + [
        (name, pa.int64())
        for name in (
            "candidate_rank",
            "support_start",
            "support_end",
            "num_involved_articles",
        )
    ]
    + [
        (name, pa.list_(pa.string()))
        for name in (
            "involved_article_ids",
            "involved_source_unit_ids",
            "unresolved_article_references",
        )
    ]
    + [(name, pa.float64()) for name in ("faith_overall", "qual_overall", "total_score")]
)


def normalize(corpus: str, raw: dict, provenance: dict, index) -> dict:
    source, language = source_identity(corpus, raw)
    document = index.get(unit_id(corpus, source, language))
    if document is None:
        raise ValueError(f"missing source for {corpus}: {source} ({language})")
    model = raw.get("generator_model_id") or provenance.get("model")
    if not model or not raw.get("question", "").strip():
        raise ValueError(f"missing model or question in {provenance}")
    mode = raw.get("mode", "")
    row = {name: str(raw.get(name) or "") for name in STRING_FIELDS}
    row.update(
        corpus=corpus,
        context_language=language,
        generator_model_id=model,
        persona=raw.get("persona") or ("practitioner" if mode == "practitioners" else mode),
        source_unit_id=document["_id"],
        source_document_id=source,
        document_text=document["text"],
        document_title=document["title"],
        generation_target_id=raw.get("target_id") or source,
        metadata_json=dumps(raw),
        provenance_json=dumps(provenance),
        persona_metadata_json=dumps(
            {
                k: v
                for k, v in raw.items()
                if k.startswith(("persona", "decider_"))
                or k
                in {
                    "mode",
                    "anchor",
                    "anchors",
                    "claim",
                    "claim_status",
                    "assertion",
                    "assertion_status",
                    "comparison_entities",
                    "comparison_aspect",
                    "constraints",
                    "objective",
                    "process_kind",
                    "process_start",
                    "process_goal",
                    "relationship_type",
                    "solution_kind",
                    "particulars",
                    "framing",
                    "question_type",
                    "source_identifier",
                }
            }
        ),
    )
    for key in ("is_best", "answer_is_translation", "evidence_is_translation"):
        row[key] = boolean(raw.get(key))
    for key in ("faith_overall", "qual_overall", "total_score"):
        row[key] = float(raw[key]) if raw.get(key) not in (None, "") else None
    row["candidate_rank"] = int(raw["candidate_rank"]) if raw.get("candidate_rank") else None
    row["supporting_text_origin"] = "evidence" if raw.get("evidence") else "answer"
    row["supporting_text"] = raw.get(row["supporting_text_origin"], "") or ""
    row["evidence_language"] = raw.get("evidence_language") or language
    status, start, end = match_support(row["supporting_text"], document["text"])
    row.update(support_match_status=status, support_start=start, support_end=end)
    row["generation_source_text"] = (
        raw.get("document_text")
        or raw.get("target_article_text")
        or raw.get("target_block_text")
        or document["text"]
    )
    row["generation_support_match_status"] = match_support(
        row["supporting_text"], row["generation_source_text"]
    )[0]

    involved = {document["_id"]: document}
    unresolved = []
    if corpus == "eurlex":
        for reference in article_references(raw):
            other = index.resolve_article(reference, language)
            if other:
                involved[other["_id"]] = other
            else:
                unresolved.append(reference)
    articles = sorted({v["source_id"] for v in involved.values() if v["unit_type"] == "article"})
    declared = boolean(raw.get("multi_article")) if corpus == "eurlex" else None
    computed = len(articles) > 1
    row.update(
        articles_involved_raw=str(raw.get("articles_involved") or ""),
        involved_article_ids=articles,
        involved_source_unit_ids=sorted(involved),
        unresolved_article_references=unresolved,
        num_involved_articles=len(articles),
        multi_article=(computed or declared is True) if corpus == "eurlex" else False,
        multi_article_declared=declared,
        multi_article_conflict=(declared is not None and declared != computed),
        cross_act=len({v["parent_id"] for v in involved.values()}) > 1,
        cross_act_declared=boolean(raw.get("cross_act")),
        multi_source_unit=len(involved) > 1,
    )
    # Stable across file relocation, input order, and changes to grading metadata.
    identity = [
        corpus,
        source,
        language,
        row["question_language"],
        mode,
        model,
        row["candidate_id"],
        row["candidate_rank"],
        row["question"],
        row["answer"],
    ]
    row["_id"] = "q_" + hashlib.sha256(dumps(identity).encode()).hexdigest()
    return row
