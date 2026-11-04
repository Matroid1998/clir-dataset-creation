"""Shared legal candidate annotations and versioned evidence serialization."""

import json
from dataclasses import dataclass, field, fields

from clir_bench.domains.legal.qac.evidence import CONTRACT

MODE_COMPARISON = "comparison"
MODE_CLAIM_VERIFICATION = "claim_verification"
MODE_SOURCE_FINDING = "source_finding"
NEW_MODES = (MODE_COMPARISON, MODE_CLAIM_VERIFICATION, MODE_SOURCE_FINDING)


@dataclass(kw_only=True)
class GenerationMetadata:
    comparison_entities: list[str] = field(default_factory=list)
    comparison_aspect: str = ""
    claim: str = ""
    claim_status: str = ""
    answer_is_translation: bool | None = None
    source_identifier: str = ""
    source_article: str = ""
    evidence: str = ""
    evidence_is_translation: bool | None = None
    evidence_language: str = ""
    evidence_source_id: str = ""
    evidence_contract: str = ""
    evidence_spans: list[dict] = field(default_factory=list)


FIELDS = tuple(f.name for f in fields(GenerationMetadata))


def parse_metadata(item, mode, *, eurlex=False):
    """Keep only annotations belonging to the requested generation contract."""
    result = {}
    if mode == MODE_COMPARISON:
        entities = item.get("comparison_entities", [])
        result.update(
            comparison_entities=([str(value).strip() for value in entities]
                                 if isinstance(entities, list) else []),
            comparison_aspect=str(item.get("comparison_aspect", "")).strip())
    if mode == MODE_CLAIM_VERIFICATION:
        result.update(claim=str(item.get("claim", "")).strip(),
                      claim_status=str(item.get("claim_status", "")).strip())
    if mode in (MODE_COMPARISON, MODE_CLAIM_VERIFICATION):
        flag = item.get("answer_is_translation")
        result["answer_is_translation"] = flag if isinstance(flag, bool) else None
    if mode == MODE_SOURCE_FINDING:
        for key in ("source_identifier", "evidence", *(("source_article",) if eurlex else ())):
            result[key] = str(item.get(key, "")).strip()
        flag = item.get("evidence_is_translation")
        result["evidence_is_translation"] = flag if isinstance(flag, bool) else None
    if item.get("evidence_contract") == CONTRACT:
        # Canonical evidence is already validated during new generation; restore
        # it verbatim during checkpoint replay/regrading, including whitespace.
        for key in ("evidence", "evidence_language", "evidence_source_id", "evidence_contract"):
            result[key] = item.get(key, "")
        result["evidence_spans"] = item.get("evidence_spans", [])
        for key in ("answer_is_translation", "evidence_is_translation"):
            flag = item.get(key)
            result[key] = flag if isinstance(flag, bool) else None
    return result


def metadata(candidate):
    """Native JSON values for checkpoints and verifier input."""
    return {key: getattr(candidate, key) for key in FIELDS}


def row_metadata(candidate):
    """CSV values; JSON preserves entity order and punctuation losslessly."""
    result = metadata(candidate)
    for key in ("comparison_entities", "evidence_spans"):
        values = result[key]
        result[key] = json.dumps(values, ensure_ascii=False) if values else ""
    return {key: value if value is not None else "" for key, value in result.items()}


def restore_metadata(row, mode, *, eurlex=False):
    """Restore annotations when regrading saved CSV rows, including False flags."""
    item = dict(row)
    entities = item.get("comparison_entities")
    if mode == MODE_COMPARISON and isinstance(entities, str):
        item["comparison_entities"] = json.loads(entities) if entities else []
    spans = item.get("evidence_spans")
    if isinstance(spans, str):
        item["evidence_spans"] = json.loads(spans) if spans else []
    for key in ("answer_is_translation", "evidence_is_translation"):
        flag = item.get(key)
        if isinstance(flag, str):
            item[key] = {"true": True, "false": False}.get(flag.lower())
    return parse_metadata(item, mode, eurlex=eurlex)
