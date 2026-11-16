"""Validate saved legal evidence and align generation units to exported documents."""

from __future__ import annotations

import hashlib
import json
import re

import pyarrow as pa

CONTRACT = "legal_evidence_spans_v1"
SPAN_TYPE = pa.struct(
    [
        (key, pa.string())
        for key in (
            "source_ref",
            "source_id",
            "source_unit_id",
            "language",
            "text",
            "source_text_sha256",
            "doc_id",
            "location_kind",
            "document_text_sha256",
        )
    ]
    + [
        (key, pa.int64())
        for key in (
            "start",
            "end",
            "document_start",
            "document_end",
            "line_start",
            "line_end",
            "block_document_start",
            "block_document_end",
        )
    ]
)


def saved_spans(raw: dict) -> list[dict]:
    """Read either native checkpoint lists or JSON-valued CSV fields without backfilling."""
    value = raw.get("evidence_spans")
    if value in (None, ""):
        value = []
    elif isinstance(value, str):
        value = json.loads(value)
    contract = raw.get("evidence_contract") or ""
    if not contract and value == []:
        return []
    if contract != CONTRACT:
        raise ValueError(f"unsupported evidence contract: {contract!r}")
    if not isinstance(value, list) or not value:
        raise ValueError("new-contract QA requires nonempty evidence_spans")
    seen = set()
    for span in value:
        if not isinstance(span, dict):
            raise ValueError("evidence span must be an object")  # noqa: TRY004
        for key in ("source_ref", "source_id", "language", "text"):
            if not isinstance(span.get(key), str) or not span[key].strip():
                raise ValueError(f"evidence span requires nonempty {key}")
        if type(span.get("start")) is not int or type(span.get("end")) is not int:
            raise ValueError("evidence span offsets must be integers")
        if not 0 <= span["start"] < span["end"]:
            raise ValueError("invalid evidence span offsets")
        if span["source_id"] in seen:
            raise ValueError("each involved source must have exactly one evidence span")
        seen.add(span["source_id"])
        digest = span.get("source_text_sha256")
        if digest and (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)):
            raise ValueError("invalid evidence source text hash")
        _character_location(span)
    return value


def _integer(value, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"evidence {name} must be an integer")
    return value


def validate_un_dataset_identity(raw: dict, document: dict) -> None:
    """A frozen UN run can only publish against the same multilingual source store."""
    value = raw.get("un_dataset_identity")
    prepared = json.loads(document["metadata_json"]).get("source_manifest")
    new_source = document["source_id"].startswith("uprprc:")
    if value in (None, ""):
        if new_source:
            raise ValueError("UPRPRC generation is missing its pinned UN dataset identity")
        return
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or not isinstance(prepared, dict):
        raise ValueError("UN generation requires its original manifest-backed source store")  # noqa: TRY004
    for identity in (value, prepared):
        if not isinstance(identity.get("manifest_path"), str) or not identity["manifest_path"].strip():
            raise ValueError("invalid UN dataset manifest identity")
        for key in ("manifest_sha256", "database_sha256"):
            digest = identity.get(key)
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"invalid UN dataset {key}")
    if value != prepared:
        raise ValueError("UN generation dataset identity differs from the exported source store")


def _character_location(span: dict) -> bool:
    """Discriminate explicit character locations without reinterpreting legacy lines."""
    kind = span.get("location_kind")
    fields = ("block_document_start", "block_document_end", "document_text_sha256")
    if kind in (None, ""):
        if any(span.get(key) not in (None, "") for key in fields):
            raise ValueError("incomplete evidence document character location")
        return False
    if kind != "document_chars":
        raise ValueError(f"unsupported evidence location_kind: {kind!r}")
    if any(span.get(key) not in (None, "") for key in ("line_start", "line_end")):
        raise ValueError("evidence cannot mix document character and line ranges")
    first = _integer(span.get("block_document_start"), "block_document_start")
    last = _integer(span.get("block_document_end"), "block_document_end")
    if not 0 <= first < last:
        raise ValueError("invalid evidence document character range")
    if not isinstance(span.get("doc_id"), str) or not span["doc_id"].strip():
        raise ValueError("document character evidence requires its original document ID")
    for key in ("source_text_sha256", "document_text_sha256"):
        digest = span.get(key)
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"invalid evidence {key}")
    return True


def _saved_character_location(raw: dict, span: dict) -> None:
    """Compare the canonical span with the saved target's original source location."""
    if any(raw.get(key) not in (None, "") for key in ("line_start", "line_end")):
        raise ValueError("UN evidence cannot mix saved document character and line ranges")
    for key in ("location_kind", "block_document_start", "block_document_end", "document_text_sha256"):
        if (raw.get(key) not in (None, "")
                and (type(raw[key]) is not type(span[key]) or raw[key] != span[key])):
            raise ValueError("UN evidence character location differs from saved generation target")
    locations = raw.get("source_locations")
    if locations in (None, ""):
        return
    if isinstance(locations, str):
        locations = json.loads(locations)
    if not isinstance(locations, dict) or not isinstance(locations.get(span["language"]), dict):
        raise ValueError("UN saved generation target is missing its language source location")  # noqa: TRY004
    location = locations[span["language"]]
    for source_key, span_key in (
        ("start", "block_document_start"),
        ("end", "block_document_end"),
        ("document_sha256", "document_text_sha256"),
    ):
        if type(location.get(source_key)) is not type(span[span_key]) or location[source_key] != span[span_key]:
            raise ValueError("UN evidence character location differs from saved generation target")


def align_spans(corpus: str, raw: dict, index) -> list[dict]:
    """Keep local offsets and add exact offsets in the exported source unit.

    A UN span is located by its saved block line or character range, never by
    searching for its text across the whole document. Hashes detect stale sources.
    """
    spans = saved_spans(raw)
    if spans and corpus not in {"eurlex", "un"}:
        raise ValueError(f"legal evidence contract is not supported for {corpus}")
    aligned = []
    for span in spans:
        if corpus == "eurlex":
            source = span["source_id"]
        else:
            source = span.get("doc_id")
            target = raw.get("block_id") or raw.get("target_id")
            document = raw.get("doc_id") or raw.get("document_id")
            if not source or source != document or span["source_id"] != target:
                raise ValueError("UN evidence must identify the saved target block and document")
            if span["source_ref"] != "target":
                raise ValueError("UN evidence source_ref must be target")
        identifier = f"{corpus}:{source}:{span['language']}"
        document = index.get(identifier)
        if document is None:
            raise ValueError(f"missing evidence source: {identifier}")
        text = document["text"]
        base = 0
        characters = _character_location(span)
        if characters and corpus != "un":
            raise ValueError("document character evidence is only supported for UN blocks")
        if characters:
            _saved_character_location(raw, span)
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != span["document_text_sha256"]:
                raise ValueError(f"evidence document text changed: {source}")
            first, last = span["block_document_start"], span["block_document_end"]
            if last > len(text):
                raise ValueError("UN evidence block character range lies outside its document")
            base = first
            text = text[first:last]
        elif corpus == "un":
            if (raw.get("location_kind") not in (None, "")
                    or any(raw.get(key) not in (None, "") for key in (
                        "block_document_start", "block_document_end", "document_text_sha256"))):
                raise ValueError("UN saved character location requires document character evidence")
            metadata = json.loads(document["metadata_json"])
            doc_start = _integer(metadata["line_start"], "document line_start")
            doc_end = _integer(metadata["line_end"], "document line_end")
            start_line = _integer(span.get("line_start"), "line_start")
            end_line = _integer(span.get("line_end"), "line_end")
            if not doc_start <= start_line <= end_line <= doc_end:
                raise ValueError("UN evidence block range lies outside its document")
            for key in ("line_start", "line_end"):
                if raw.get(key) not in (None, "") and int(raw[key]) != span[key]:
                    raise ValueError("UN evidence line range differs from saved generation target")
            lines = text.split("\n")
            if len(lines) != doc_end - doc_start + 1:
                raise ValueError("UN document text does not match its saved line range")
            first = start_line - doc_start
            last = end_line - doc_start + 1
            base = sum(len(line) + 1 for line in lines[:first])
            text = "\n".join(lines[first:last])
        digest = span.get("source_text_sha256")
        if digest and hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
            raise ValueError(f"evidence source text changed: {span['source_id']}")
        start, end = span["start"], span["end"]
        if end > len(text) or text[start:end] != span["text"]:
            raise ValueError(f"evidence offsets do not reproduce quotation: {span['source_id']}")
        document_start, document_end = base + start, base + end
        if document["text"][document_start:document_end] != span["text"]:
            raise ValueError("evidence could not be aligned to the exported source unit")
        aligned.append(
            {
                **span,
                "source_unit_id": identifier,
                "document_start": document_start,
                "document_end": document_end,
            }
        )
    return aligned


def evidence_rows(row: dict):
    """One record per new-contract span; preserve the legacy single-support fallback."""
    if row["evidence_spans"]:
        for position, span in enumerate(row["evidence_spans"]):
            yield {
                "query_id": row["_id"],
                "span_index": position,
                "source_unit_id": span["source_unit_id"],
                "source_ref": span["source_ref"],
                "language": span["language"],
                "text": span["text"],
                "origin": "evidence_spans",
                "match_status": "exact",
                "start": span["document_start"],
                "end": span["document_end"],
                "generation_source_id": span["source_id"],
                "generation_start": span["start"],
                "generation_end": span["end"],
            }
    else:
        yield {
            "query_id": row["_id"],
            "source_unit_id": row["source_unit_id"],
            "language": row["evidence_language"],
            "text": row["supporting_text"],
            "origin": row["supporting_text_origin"],
            "match_status": row["support_match_status"],
            "start": row["support_start"],
            "end": row["support_end"],
        }
