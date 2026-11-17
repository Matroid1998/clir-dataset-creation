"""
Question generation engine.

The shape of the pipeline is fixed and domain-independent: for a document and a
target language, generate N candidate questions, grade them all in batched
calls, rank by total score, keep the best. What changes per domain is the
prompts; what changes per dataset is the plan (which documents, which languages,
which mode).

Design properties preserved from the original pipeline, each load-bearing:

* generators and graders see the same evidence: an explicitly selected document,
  or all language versions when the plan selects a whole family;
* candidates are generated in one call and graded in one call per rubric, then
  ranked -- not generated-and-graded one at a time;
* plans control the query language; selecting an individual document requires
  its language to match the question language.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from clir_bench.core import corpus as corpus_io
from clir_bench.core.domain import CorpusSchema
from clir_bench.core.generation_models import SELECTION_FIELDS
from clir_bench.core.grading import (
    MODE_SEMANTIC,
    MODE_TECHNICAL,
    GraderConfig,
    grade_columns,
    grade_faithfulness,
    grade_quality,
    rank_candidates,
)
from clir_bench.core.llm import call_with_retries, chat, parse_json_response
from clir_bench.core.prompts import PromptPack

CANDIDATES_PER_DOCUMENT = 3

# Metadata shared by retrieval personas. Keep structured values intact until CSV export.
CANDIDATE_METADATA_FIELDS = (
    "question_type", "framing", "anchor", "evidence", "evidence_language",
    "answer_is_translation", "process_kind", "process_start", "process_goal",
    "relationship_type", "solution_kind", "objective", "constraints",
    "comparison_entities", "comparison_aspect", "assertion", "assertion_status",
    "source_identifier",
    "candidate_id",
)
_JSON_METADATA_FIELDS = ("constraints", "comparison_entities", "answer_is_translation")

DECIDER_FIELDS = (
    "decider_model", "decider_mode", "decider_reason", "decider_confidence",
    "decider_threshold", "decider_probabilities_yes_json", "decider_eligible_modes_json",
    "decider_selection_policy", "decider_persona_weights_json", "decider_counts_before_json",
    "decider_routing_task", "decider_prompt_sha256",
)

# Query-language selection strategies. The numbering is published in dataset
# metadata, so the values are fixed.
STRATEGY_RANDOM_ANY = 1
STRATEGY_RANDOM_MISSING = 2
STRATEGY_RANDOM_EXISTING = 3
STRATEGY_ALL = 4

STRATEGY_NAMES = {
    STRATEGY_RANDOM_ANY: "random_any",
    STRATEGY_RANDOM_MISSING: "random_missing",
    STRATEGY_RANDOM_EXISTING: "random_existing",
    STRATEGY_ALL: "all",
}


def pick_target_languages(
    strategy: int,
    available: Sequence[str],
    languages: Sequence[str],
    *,
    rng: Optional[random.Random] = None,
) -> list[str]:
    """Choose the language(s) to ask in, given what the document exists in.

    ``random_missing`` is what produces genuinely cross-lingual queries: it asks
    in a language the document is *not* available in, so a retriever cannot
    succeed by lexical overlap with the query's own language.
    """
    chooser = rng or random
    universe = list(languages)
    if not universe:
        raise ValueError("no candidate languages configured for this domain")
    present = set(available) & set(universe)
    missing = [lang for lang in universe if lang not in present]

    if strategy == STRATEGY_RANDOM_ANY:
        return [chooser.choice(universe)]
    if strategy == STRATEGY_RANDOM_MISSING:
        return [chooser.choice(missing or universe)]
    if strategy == STRATEGY_RANDOM_EXISTING:
        existing = [lang for lang in universe if lang in present]
        return [chooser.choice(existing or universe)]
    if strategy == STRATEGY_ALL:
        return list(universe)
    raise ValueError(f"unknown strategy: {strategy}")


@dataclass(frozen=True)
class GenerationConfig:
    """Models and knobs for one generation run."""

    generation_model: str
    grader: GraderConfig
    generation_reasoning_effort: str = "medium"
    candidates: int = CANDIDATES_PER_DOCUMENT
    retries: int = 3
    candidate_parser: Callable[[Any, str], list[dict[str, Any]]] | None = None


@dataclass(frozen=True)
class PlanItem:
    """One unit of work: ask about this document, in this language, this way."""

    family: str
    language: str
    mode: str = MODE_TECHNICAL
    strategy: int = STRATEGY_RANDOM_ANY
    metadata: Mapping[str, Any] = field(default_factory=dict)
    document_id: str = ""

    @property
    def strategy_name(self) -> str:
        return str(self.metadata.get("strategy_name") or STRATEGY_NAMES.get(self.strategy, ""))


def select_plan_rows(
    item: PlanItem, rows: Sequence[Mapping[str, str]], schema: CorpusSchema,
) -> list[Mapping[str, str]]:
    """Resolve an explicitly selected document without substituting another version."""
    if not item.document_id:
        return list(rows)
    selected = [row for row in rows if schema.id_of(row) == item.document_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one source document for {item.document_id!r}, found {len(selected)}")
    if schema.language_of(selected[0]) != item.language:
        raise ValueError("Question language must match the selected source document's language")
    if schema.family_of(selected[0]) != item.family:
        raise ValueError("Selected source document does not belong to the planned family")
    return selected


def generate_candidates(
    client: Any,
    config: GenerationConfig,
    prompt: str,
    passages: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Generate candidate Q/A pairs in one call.

    A domain parser can enforce its own schema and abstention contract. Otherwise
    the legacy question/answer parser retains recognized persona annotations.
    """
    raw = chat(
        client,
        config.generation_model,
        [{"role": "system", "content": prompt}, {"role": "user", "content": passages}],
        reasoning_effort=config.generation_reasoning_effort,
    )
    data = parse_json_response(raw)
    if config.candidate_parser is not None:
        return config.candidate_parser(data, mode)[:config.candidates]
    if isinstance(data, dict):
        data = [data]

    out: list[dict[str, Any]] = []
    for item in list(data)[: config.candidates]:
        if not isinstance(item, Mapping):
            continue
        row = {
            "question": str(item.get("question", "")).strip(),
            "answer": str(item.get("answer", "")).strip(),
        }
        if mode == MODE_SEMANTIC:
            row["framing"] = str(item.get("framing", "other")).strip()
        else:
            row["question_type"] = str(item.get("question_type", "other")).strip()
        row.update({key: item[key] for key in CANDIDATE_METADATA_FIELDS if key in item})
        out.append(row)
    return out


def generate_for_document(
    item: PlanItem,
    rows: Sequence[Mapping[str, str]],
    *,
    schema: CorpusSchema,
    prompts: PromptPack,
    config: GenerationConfig,
    generation_client: Any,
    grading_client: Any,
    language_order: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Generate, grade and rank candidates for one plan item.

    Returns rows sorted best-first. Callers keep the top row per selected
    document, language and mode for publication and may keep the rest for audit.
    """
    rows = select_plan_rows(item, rows, schema)
    passages = corpus_io.build_passages_text(
        rows, schema, order=language_order, include_identifiers=True
    )
    if not passages:
        return []

    qa_pairs = call_with_retries(
        lambda: generate_candidates(
            generation_client,
            config,
            prompts.generation(item.mode, item.language),
            passages,
            item.mode,
        ),
        retries=config.retries,
        label=f"generate {item.family}/{item.language}",
    )
    qa_pairs = [qa for qa in qa_pairs if qa.get("question") and qa.get("answer")]
    if not qa_pairs:
        return []
    qa_pairs = [dict(qa, mode=item.mode, question_language=item.language) for qa in qa_pairs]

    faith = call_with_retries(
        lambda: grade_faithfulness(
            grading_client, config.grader, prompts.faithfulness("batch"), passages, qa_pairs,
            strict=True,
        ),
        retries=config.retries,
        label=f"grade faithfulness {item.family}/{item.language}",
    )
    quality = call_with_retries(
        lambda: grade_quality(
            grading_client,
            config.grader,
            prompts.quality(item.mode, "batch"),
            passages,
            qa_pairs,
            item.mode,
            strict=True,
        ),
        retries=config.retries,
        label=f"grade quality {item.family}/{item.language}",
    )

    context_row = corpus_io.pick_context_row(rows, schema, item.language) or rows[0]
    ranked = rank_candidates(qa_pairs, faith, quality, item.mode)

    out: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked, start=1):
        row: dict[str, Any] = {
            schema.family_field: item.family,
            "corpus_id": schema.id_of(context_row),
            "source_document_id": item.document_id,
            "question_language": item.language,
            "context_language": schema.language_of(context_row),
            "context_languages": corpus_io.serialize_languages(rows, schema, order=language_order),
            "mode": item.mode,
            "strategy": item.strategy,
            "strategy_name": item.strategy_name,
            "candidate_rank": rank,
            "question": candidate.qa.get("question", ""),
            "answer": candidate.qa.get("answer", ""),
            **{key: candidate.qa.get(key, "") for key in CANDIDATE_METADATA_FIELDS},
        }
        row.update(grade_columns(candidate.faith, candidate.quality, item.mode))
        row.update(item.metadata)
        row["generator_model_id"] = config.generation_model
        row["generator_model_name"] = config.generation_model.rsplit("/", 1)[-1]
        out.append(row)
    return out


def grading_fieldnames(
    *, prompts: PromptPack | None = None, modes: Sequence[str] = (),
) -> tuple[str, ...]:
    """Grade and audit columns, including criteria declared by the active rubrics."""
    from clir_bench.core.grading import (
        FAITHFULNESS_FIELDS,
        SEMANTIC_QUALITY_FIELDS,
        TECHNICAL_QUALITY_FIELDS,
        rubric_keys,
    )

    persona_scores = tuple(
        f"qual_{key}"
        for mode in modes if prompts is not None
        for key in rubric_keys(prompts.quality(mode, "batch"))
    )
    scores = tuple(dict.fromkeys(
        FAITHFULNESS_FIELDS + TECHNICAL_QUALITY_FIELDS + SEMANTIC_QUALITY_FIELDS + persona_scores
    ))
    audit = (
        "faith_reason", "qual_failure_type", "qual_reason", "total_score",
        "faithfulness_verifier_response_json", "faithfulness_rubric_version",
        "quality_verifier_response_json", "quality_rubric_version", "quality_audit_status",
        "quality_batch_diversity", "quality_score_notes_json", "quality_checks_json",
        "quality_problems_json", "quality_mode_status", "quality_support_status",
        "quality_metadata_status",
    )
    return scores + audit


def output_fieldnames(
    schema: CorpusSchema, *, prompts: PromptPack | None = None, modes: Sequence[str] = (),
) -> tuple[str, ...]:
    """The QAC CSV schema, covering persona metadata, scores and audit records.

    Pass the domain's prompt pack and modes to include every active rubric's
    criteria when several personas share an output file.
    """
    identity = (
        schema.family_field,
        "corpus_id",
        "source_document_id",
        "question_language",
        "context_language",
        "context_languages",
        "mode",
        "strategy",
        "strategy_name",
        "candidate_rank",
        "generator_model_id",
        "generator_model_name",
        *SELECTION_FIELDS,
        "question",
        "answer",
        *CANDIDATE_METADATA_FIELDS,
    )
    return identity + grading_fieldnames(prompts=prompts, modes=modes) + DECIDER_FIELDS


def normalize_row(row: Mapping[str, Any], fieldnames: Sequence[str]) -> dict[str, Any]:
    """Pad a row to the full schema so mixed-mode rows share one header."""
    return {
        name: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, bool)) else value
        for name in fieldnames
        for value in (row.get(name, ""),)
    }


def candidate_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Restore a CSV candidate's metadata for regrading without leaking prior scores."""
    candidate = {
        key: row[key]
        for key in ("question", "answer", "mode", "question_language", *CANDIDATE_METADATA_FIELDS)
        if key in row and row[key] != ""
    }
    for key in _JSON_METADATA_FIELDS:
        if isinstance(candidate.get(key), str):
            candidate[key] = json.loads(candidate[key])
    return candidate


def select_best(
    rows: Sequence[Mapping[str, Any]], schema: CorpusSchema
) -> list[dict[str, Any]]:
    """Highest-scoring row per selected document, question language and mode.

    Rows with missing, invalid or nonfinite totals remain in the candidate
    export but cannot be selected as a scored best candidate.
    """
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    scores: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        score = _score(row)
        if score is None:
            continue
        key = (
            str(row.get(schema.family_field, "")), str(row.get("question_language", "")),
            str(row.get("mode", "")),
            str(row.get("source_document_id") or ""),
        )
        if key not in best or score > scores[key]:
            best[key] = dict(row)
            scores[key] = score
    return list(best.values())


def _score(row: Mapping[str, Any]) -> float | None:
    try:
        score = float(row.get("total_score"))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def allocate_quotas(total: int, strategies: Sequence[int]) -> dict[int, int]:
    """Split a question budget evenly across strategies.

    ``STRATEGY_ALL`` yields one question per language rather than one per
    document, so its share is handled by the caller when sizing a plan.
    """
    if total <= 0:
        raise ValueError("total questions must be positive")
    if not strategies:
        raise ValueError("no strategies to allocate across")
    base, remainder = divmod(total, len(strategies))
    quotas = {strategy: base for strategy in strategies}
    for strategy in list(strategies)[:remainder]:
        quotas[strategy] += 1
    return quotas


__all__ = [
    "CANDIDATES_PER_DOCUMENT",
    "GenerationConfig",
    "PlanItem",
    "STRATEGY_ALL",
    "STRATEGY_NAMES",
    "STRATEGY_RANDOM_ANY",
    "STRATEGY_RANDOM_EXISTING",
    "STRATEGY_RANDOM_MISSING",
    "allocate_quotas",
    "generate_candidates",
    "generate_for_document",
    "grading_fieldnames",
    "normalize_row",
    "output_fieldnames",
    "pick_target_languages",
    "select_best",
    "select_plan_rows",
]
