"""Versioned, source-bound evidence for future legal generation.

Only visible source bodies can establish a quote. Labels, omitted translations,
truncation markers and UN understanding-only context are not evidence sources.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

CONTRACT = "legal_evidence_spans_v1"
MARKER = f"EVIDENCE_CONTRACT: {CONTRACT}"


@dataclass(frozen=True)
class EvidenceSource:
    source_ref: str
    source_id: str
    language: str
    text: str
    full_text: str
    unit_type: str
    doc_id: str = ""
    line_start: int | None = None
    line_end: int | None = None
    original_sha256: str = ""
    original_length: int | None = None


def uses_contract(prompt: str) -> bool:
    return MARKER in prompt.splitlines()


def validate_envelope(data) -> list[dict]:
    """Abstention is distinct from malformed output and exhausted retries."""
    if not isinstance(data, list) or not 1 <= len(data) <= 3:
        raise ValueError("evidence contract requires one to three candidates or one skip_reason")
    if len(data) == 1 and isinstance(data[0], dict) and set(data[0]) == {"skip_reason"}:
        if not isinstance(data[0]["skip_reason"], str) or not data[0]["skip_reason"].strip():
            raise ValueError("skip_reason must be nonempty")
        return []
    for item in data:
        if not isinstance(item, dict) or "skip_reason" in item:
            raise ValueError("mixed skips/candidates or invalid candidate")
        for key in ("question", "answer"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{key} must be nonempty")
        if type(item.get("answer_is_translation")) is not bool:
            raise ValueError("answer_is_translation must be a JSON boolean")
    return data


def validate_spans(
    item: Mapping,
    sources: Mapping[tuple[str, str], EvidenceSource],
    required_refs: Sequence[str],
    *,
    primary_ref: str | None = None,
) -> dict:
    """Return canonical spans and derived compatibility fields.

    Structural failures raise within the generation stage. Semantic sufficiency
    and removable prefix/suffix padding remain verifier judgments.
    """
    raw = item.get("evidence_spans")
    if not isinstance(raw, list) or not raw:
        raise ValueError("evidence_spans must be a nonempty list")
    required = set(required_refs)
    seen: set[str] = set()
    spans = []
    for position, span in enumerate(raw):
        if not isinstance(span, dict) or set(span) != {"source_ref", "language", "text"}:
            raise ValueError("each model evidence span must contain source_ref, language and text")
        for key in ("source_ref", "language", "text"):
            if not isinstance(span[key], str) or not span[key].strip():
                raise ValueError(f"evidence {key} must be a nonempty string")
        reference, language, quote = span["source_ref"], span["language"], span["text"]
        if reference not in required:
            raise ValueError(f"evidence source is not an involved unit: {reference}")
        if reference in seen:
            raise ValueError(f"duplicate evidence source: {reference}")
        seen.add(reference)
        source = sources.get((reference, language))
        if source is None:
            raise ValueError(f"evidence source/language was not supplied: {reference} ({language})")
        if position == 0 and primary_ref is not None and reference != primary_ref:
            raise ValueError("source-finding evidence must start with the target source")
        start = source.text.find(quote)
        if start < 0:
            raise ValueError(f"evidence is not an exact visible source passage: {reference} ({language})")
        # Prefix truncation preserves offsets into the original body. Never
        # derive an offset against a rendered label or synthetic marker.
        end = start + len(quote)
        if source.full_text[start:end] != quote:
            raise ValueError("visible evidence body does not align with its original source")
        original_length = source.original_length if source.original_length is not None else len(source.full_text)
        if (source.unit_type in {"article", "annex"}
                and original_length == len(source.full_text)
                and quote.strip() == source.full_text.strip()):
            raise ValueError("whole-article evidence is not a supporting excerpt; narrow or skip")
        canonical = {
            **span, "source_id": source.source_id, "start": start, "end": end,
            "source_text_sha256": (source.original_sha256
                                   or hashlib.sha256(source.full_text.encode("utf-8")).hexdigest()),
        }
        if source.unit_type == "block":
            if source.line_start is None or source.line_end is None or source.line_start < 1:
                raise ValueError("UN evidence requires its original 1-based line range")
            canonical.update(doc_id=source.doc_id, line_start=source.line_start, line_end=source.line_end)
        spans.append(canonical)
    if seen != required:
        raise ValueError(f"evidence missing for involved sources: {sorted(required - seen)}")
    primary = spans[0]
    return {
        "evidence_contract": CONTRACT,
        "evidence_spans": spans,
        "evidence": primary["text"],
        "evidence_language": primary["language"],
        "evidence_source_id": primary["source_id"],
        "evidence_is_translation": False,
        "answer_is_translation": item["answer_is_translation"],
    }
