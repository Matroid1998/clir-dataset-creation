"""
LLM-as-judge grading.

Six near-identical grading shells existed across the old repo (OpenAI batch,
OpenRouter batch, single-pair OpenAI, single-pair Claude, and two normalizers in
comparison scripts). They differed only in transport and arity, so those are the
two parameters here; the rubric, the score arithmetic, the pad-with-1 fallback
and the field names are defined once.

Legacy batch rubrics return a JSON list, while concept/variant rubrics return a
single JSON object. Current quality rubrics can instead return a compact audit
object with candidates and a batch diversity check. Each prompt's declared
contract determines parsing and validation; these shapes are not interchangeable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from clir_bench.core.llm import (
    chat,
    chat_with_thinking,
    parse_json_response,
)

MODE_TECHNICAL = "technical"
MODE_SEMANTIC = "semantic"
MODE_CONCEPTUAL = "conceptual"
# A fact-extraction mode like technical -- same categories, same quality
# columns -- differing only in that the QUESTION must describe an instrument
# rather than cite it. It therefore shares the technical quality rubric.
MODE_DESCRIPTIVE = "descriptive"
# The two EUR-Lex practitioner modes. Unlike the three above they do NOT share
# the technical rubric: each is graded on criteria its own prompt turns on, so
# each brings its own five quality keys.
MODE_LOOKUP = "lookup"
MODE_FACT_PATTERN = "fact_pattern"

# CSV column names. Load-bearing: published datasets carry these headers.
FAITHFULNESS_FIELDS = (
    "faith_grounding",
    "faith_precision",
    "faith_numerical_fidelity",
    "faith_overall",
)
TECHNICAL_QUALITY_FIELDS = (
    "qual_search_bar_realism",
    "qual_specificity",
    "qual_phrasing_economy",
    "qual_focus",
    "qual_linguistic_quality",
    "qual_overall",
)
SEMANTIC_QUALITY_FIELDS = (
    "qual_search_realism",
    "qual_lexical_distance",
    "qual_conceptual_framing",
    "qual_retrievability",
    "qual_linguistic_quality",
    "qual_overall",
)
LOOKUP_QUALITY_FIELDS = (
    "qual_practitioner_realism",
    "qual_anchoring",
    "qual_informativeness",
    "qual_focus",
    "qual_linguistic_quality",
    "qual_overall",
)
FACT_PATTERN_QUALITY_FIELDS = (
    "qual_situation",
    "qual_regime_fixing",
    "qual_terminology_and_distance",
    "qual_focus",
    "qual_linguistic_quality",
    "qual_overall",
)

# Rubric sub-scores summed into each aggregate.
FAITHFULNESS_KEYS = ("grounding", "precision", "numerical_fidelity")
TECHNICAL_QUALITY_KEYS = (
    "search_bar_realism",
    "specificity",
    "phrasing_economy",
    "focus",
    "linguistic_quality",
)
SEMANTIC_QUALITY_KEYS = (
    "search_realism",
    "lexical_distance",
    "conceptual_framing",
    "retrievability",
    "linguistic_quality",
)
LOOKUP_QUALITY_KEYS = (
    "practitioner_realism",
    "anchoring",
    "informativeness",
    "focus",
    "linguistic_quality",
)
FACT_PATTERN_QUALITY_KEYS = (
    "situation",
    "regime_fixing",
    "terminology_and_distance",
    "focus",
    "linguistic_quality",
)

CONCEPTUAL_QUALITY_KEYS = (
    "search_realism", "anchoring_and_time", "consequence", "lexical_distance", "linguistic_quality",
)
CONCEPTUAL_QUALITY_FIELDS = tuple(f"qual_{key}" for key in CONCEPTUAL_QUALITY_KEYS) + ("qual_overall",)

# Modes whose rubric is not the technical one. A mode absent here falls back to
# the technical five, which is what ``descriptive`` and the chemistry flows want.
_QUALITY_KEYS_BY_MODE = {
    MODE_SEMANTIC: SEMANTIC_QUALITY_KEYS,
    MODE_CONCEPTUAL: CONCEPTUAL_QUALITY_KEYS,
    MODE_LOOKUP: LOOKUP_QUALITY_KEYS,
    MODE_FACT_PATTERN: FACT_PATTERN_QUALITY_KEYS,
}
_QUALITY_FIELDS_BY_MODE = {
    MODE_SEMANTIC: SEMANTIC_QUALITY_FIELDS,
    MODE_CONCEPTUAL: CONCEPTUAL_QUALITY_FIELDS,
    MODE_LOOKUP: LOOKUP_QUALITY_FIELDS,
    MODE_FACT_PATTERN: FACT_PATTERN_QUALITY_FIELDS,
}


# A rubric declares the criteria it scores as ``"<key>": <1-5>`` in its own
# output block. Reading them from the rubric is what keeps a prompt edit from
# silently desynchronising the grade: ``quality_keys`` is keyed by mode NAME,
# but a name is not unique across packs -- "lookup" scores different criteria in
# prompts_eurlex than in prompts_un, and "practitioners" shares none of the
# technical five. When the two disagree the rubric wins, because it is what the
# grader was actually asked to produce.
_RUBRIC_SCORE = re.compile(r'"([a-z_]+)"\s*:\s*<1-5>')
_LEGAL_RUBRIC_VERSIONS = frozenset(("legal-qg-v2.0", "legal-qg-v3.0"))
_CHEMISTRY_RUBRIC_VERSIONS = frozenset((
    "chemistry-clir-exceptional-r1",
    "chemistry-clir-exceptional-r2",
))


def _chemistry_rubric_version(prompt: str) -> str | None:
    """Identify the declared chemistry contract instead of guessing from mode names."""
    versions = set(re.findall(r"^# RUBRIC:\s*(chemistry-\S+)\s*$", prompt, re.MULTILINE))
    if not versions:
        return None
    if len(versions) != 1 or not versions <= _CHEMISTRY_RUBRIC_VERSIONS:
        raise ValueError("Chemistry rubric declares conflicting or unsupported versions")
    return next(iter(versions))


def _quality_output_example(prompt: str) -> Mapping[str, Any] | None:
    """Read the compact quality contract from its JSON example or explicit field list."""
    version = _chemistry_rubric_version(prompt)
    marker = (r"^ILLUSTRATIVE VERIFIER OUTPUT\s*$"
              if version == "chemistry-clir-exceptional-r1" else r"^OUTPUT\s*$")
    output = re.search(marker, prompt, re.MULTILINE)
    # r7 states the compact contract directly instead of including a JSON sample.
    if output is None and version == "chemistry-clir-exceptional-r2":
        output = re.search(r"^OUTPUT CONTRACT\s*$", prompt, re.MULTILINE)
    if output is None:
        return None
    section = prompt[output.end():]
    start = re.search(r'\{\s*"candidates"\s*:', section)
    if start is None:
        declared = re.search(
            r"^scores and score_notes each contain exactly these keys:\s*\n"
            r"([a-z_]+(?:,\s*[a-z_]+){4})\.?$",
            section, re.MULTILINE,
        )
        if declared is None:
            return None
        keys = [key.strip() for key in declared.group(1).rstrip(".").split(",")]
        example = {
            "candidates": [{
                "index": 0, "candidate_id": None,
                "scores": dict.fromkeys(keys, 1),
                "score_notes": dict.fromkeys(keys, ""),
                "checks": {"mode": "pass", "support": "pass", "metadata": "pass"},
                "problems": [],
            }],
            "batch_diversity": "not_applicable",
        }
    else:
        example, _ = json.JSONDecoder().raw_decode(section[start.start():])
    candidates = example.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("Quality output example must contain one candidate template")
    candidate = candidates[0]
    scores = candidate.get("scores") if isinstance(candidate, dict) else None
    if (not isinstance(scores, dict) or len(scores) != 5
            or any(not re.fullmatch(r"[a-z_]+", key) for key in scores)
            or "score_notes" not in candidate or "checks" not in candidate
            or "batch_diversity" not in example):
        raise ValueError("Quality output example must declare five scores, notes, checks, and diversity")
    return example


def _legal_rubric_version(prompt: str) -> str | None:
    """Recognize structured legal contracts without falling back on bad versions."""
    versions = set(re.findall(r"\blegal-qg-v\d+(?:\.\d+)+\b", prompt))
    if not versions:
        return None
    if len(versions) != 1 or not versions <= _LEGAL_RUBRIC_VERSIONS:
        raise ValueError("Legal rubric declares conflicting or unsupported versions")
    return next(iter(versions))


def rubric_keys(prompt: str) -> tuple[str, ...]:
    """The criteria a quality rubric scores, in the order it declares them."""
    example = _quality_output_example(prompt)
    if example is not None:
        return tuple(example["candidates"][0]["scores"])
    nested = re.search(r"^- scores: (?:object with )?exactly ([a-z_, ]+?)(?:\.|,\s+each\b)",
                       prompt, re.MULTILINE)
    if nested:
        return tuple(key.strip() for key in nested.group(1).split(","))
    seen: dict[str, None] = {}
    for key in _RUBRIC_SCORE.findall(prompt or ""):
        seen.setdefault(key, None)
    return tuple(seen)


def quality_keys(mode: str) -> tuple[str, ...]:
    return _QUALITY_KEYS_BY_MODE.get(mode, TECHNICAL_QUALITY_KEYS)


def quality_fields(mode: str) -> tuple[str, ...]:
    return _QUALITY_FIELDS_BY_MODE.get(mode, TECHNICAL_QUALITY_FIELDS)


def _sum_scores(scores: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    values = [scores.get(key, 0) for key in keys]
    return None if any(value is None for value in values) else sum(int(value) for value in values)


def faith_overall(scores: Mapping[str, Any]) -> int | None:
    return _sum_scores(scores, FAITHFULNESS_KEYS)


def quality_overall(scores: Mapping[str, Any], mode: str) -> int | None:
    keys = scores.get("_keys") or quality_keys(mode)
    return _sum_scores(scores, keys)


def total_score(faith: Mapping[str, Any], quality: Mapping[str, Any], mode: str) -> int | None:
    """Faithfulness aggregate plus quality aggregate (the ranking key)."""
    faith_total = faith["overall"] if "overall" in faith else faith_overall(faith)
    quality_total = quality["overall"] if "overall" in quality else quality_overall(quality, mode)
    return None if faith_total is None or quality_total is None else int(faith_total) + int(quality_total)


# --------------------------------------------------------------------------- #
# Grader
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GraderConfig:
    """Which model grades, over which transport, with what knobs."""

    model: str
    reasoning_effort: str = "low"
    thinking_budget_tokens: int = 8000
    thinking_max_tokens: int = 12000
    # Force the Claude extended-thinking transport even for a non-default model.
    use_thinking: bool | None = None

    @property
    def thinking(self) -> bool:
        if self.use_thinking is not None:
            return self.use_thinking
        return self.model.startswith(("anthropic/claude-", "~anthropic/claude-"))


# Rendered between the question and the answer, in this order, when present.
_CANDIDATE_EXTRAS = (
    ("mode", "Mode"),
    ("question_language", "Question language"),
    ("question_cited", "Question (cited rendering)"),
    ("instrument_short_name", "Instrument short name"),
    ("anchor", "Anchor (declared)"),
    ("particulars", "Particulars (declared)"),
    ("question_type", "Question type (declared)"),
    ("framing", "Framing (declared)"),
    ("comparison_entities", "comparison_entities"),
    ("comparison_aspect", "comparison_aspect"),
    ("claim", "claim"),
    ("claim_status", "claim_status"),
    ("answer_is_translation", "answer_is_translation"),
    ("source_identifier", "source_identifier"),
    ("source_article", "source_article"),
    ("evidence", "evidence"),
    ("evidence_language", "evidence_language"),
    ("evidence_source_id", "evidence_source_id"),
    ("evidence_contract", "evidence_contract"),
    ("evidence_spans", "evidence_spans"),
    ("process_kind", "process_kind"),
    ("process_start", "process_start"),
    ("process_goal", "process_goal"),
    ("relationship_type", "relationship_type"),
    ("solution_kind", "solution_kind"),
    ("objective", "objective"),
    ("constraints", "constraints"),
    ("assertion", "assertion"),
    ("assertion_status", "assertion_status"),
    ("evidence_is_translation", "evidence_is_translation"),
)


def candidates_block(qa_pairs: Sequence[Mapping[str, Any]]) -> str:
    """The candidate serialization every grader prompt expects.

    A candidate may carry ``articles_involved`` (the EUR-Lex flow's declaration
    of which articles the answer drew on); when present it is shown so the
    grader can verify it, otherwise the block is question and answer only.

    It may also carry the EUR-Lex modes' own fields -- ``question_cited`` /
    ``instrument_short_name`` / ``anchor`` for ``lookup``, ``particulars`` for
    ``fact_pattern``, and ``question_type`` for both. Those rubrics run
    consistency checks *on* those fields (is the anchor actually in the
    question? is the short name invented? does the cited rendering differ only
    by the identifier?), which is impossible unless the grader is shown them.
    Every extra is optional, so a caller that passes only question/answer gets
    exactly the block it always got.
    """
    lines = []
    for i, qa in enumerate(qa_pairs):
        entry = f"Candidate {i}:\n  Question: {qa.get('question', '')}"
        for key, label in _CANDIDATE_EXTRAS:
            value = qa.get(key)
            if not value and value is not False:
                continue
            if key == "evidence_spans":
                value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (list, tuple)):
                value = "; ".join(str(v) for v in value)
            entry += f"\n  {label}: {value}"
        entry += f"\n  Answer: {qa.get('answer', '')}"
        involved = qa.get("articles_involved")
        if involved:
            declared = ", ".join(involved) if not isinstance(involved, str) else involved
            entry += f"\n  Articles involved (declared): {declared}"
        lines.append(entry)
    return "\n\n".join(lines)


def _invoke(client: Any, config: GraderConfig, system_prompt: str, user_content: str) -> str:
    if config.thinking:
        return chat_with_thinking(
            client,
            config.model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            budget_tokens=config.thinking_budget_tokens,
            max_tokens=config.thinking_max_tokens,
        )
    return chat(
        client,
        config.model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        reasoning_effort=config.reasoning_effort,
    )


def _as_list(data: Any) -> list[dict]:
    if isinstance(data, dict):
        return [data]
    return [item for item in (data or []) if isinstance(item, dict)]


def _normalize_faith(item: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {key: _int_or(item.get(key), 1) for key in FAITHFULNESS_KEYS}
    row["reason"] = str(item.get("reason", "")).strip()
    row["overall"] = faith_overall(row)
    return row


def _normalize_quality(item: Mapping[str, Any], mode: str,
                       keys: Sequence[str] | None = None) -> dict[str, Any]:
    keys = tuple(keys) if keys else quality_keys(mode)
    row: dict[str, Any] = {key: _int_or(item.get(key), 1) for key in keys}
    # Carried so downstream column-writers use the criteria this rubric actually
    # scored rather than re-deriving them from the mode name.
    row["_keys"] = list(keys)
    row["failure_type"] = str(item.get("failure_type", "none")).strip()
    row["reason"] = str(item.get("reason", "")).strip()
    row["overall"] = quality_overall(row, mode)
    return row


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _missing_faith() -> dict[str, Any]:
    row: dict[str, Any] = {key: 1 for key in FAITHFULNESS_KEYS}
    row["reason"] = "missing"
    row["overall"] = faith_overall(row)
    return row


def _missing_quality(mode: str, keys: Sequence[str] | None = None) -> dict[str, Any]:
    keys = tuple(keys) if keys else quality_keys(mode)
    row: dict[str, Any] = {key: 1 for key in keys}
    row["_keys"] = list(keys)
    row["failure_type"] = "missing"
    row["reason"] = "missing"
    row["overall"] = quality_overall(row, mode)
    return row


_VALIDITY_KEYS = (
    "question_premises", "answer_responsiveness", "target_relevance",
    "reference_sufficiency", "legal_temporal_fidelity",
)
_LEGAL_FIELDS = (
    "index", "candidate_id", "rubric_version", "mode", "evidence",
    "target_relation", "reasoning_requirement", "validity", "scores",
    "score_notes", "metadata_checks", "issues", "suggested_repair",
    "suggested_mode", "confidence",
)
_CANDIDATE_INPUT_FIELDS = (
    "candidate_id", "question_language", "question", "answer", "question_type",
    "question_cited", "instrument_short_name", "anchor", "anchors", "particulars",
    "articles_involved", "framing", "question_template", "instrument_slot_base",
    "instrument_slot_cited", "instrument_description",
    "comparison_entities", "comparison_aspect", "claim", "claim_status",
    "answer_is_translation", "source_identifier", "source_article", "evidence",
    "evidence_is_translation", "evidence_language", "evidence_source_id",
    "evidence_contract", "evidence_spans",
)
_CHEMISTRY_CANDIDATE_INPUT_FIELDS = (
    "candidate_id", "mode", "question_language", "question", "answer", "anchor",
    "evidence", "evidence_language", "answer_is_translation", "question_type",
    "process_kind", "process_start", "process_goal", "relationship_type",
    "solution_kind", "objective", "constraints", "comparison_entities",
    "comparison_aspect", "assertion", "assertion_status", "source_identifier",
)


def _chemistry_candidates(qa_pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pass generation fields verbatim, without old scores or model identities.

    Missing generation fields are left missing so the verifier can distinguish
    an assessable candidate failure from missing source evidence.
    """
    candidates = []
    ids = []
    for qa in qa_pairs:
        if not isinstance(qa, Mapping):
            raise ValueError("Chemistry verifier candidates must be objects")  # noqa: TRY004 -- invalid verifier input
        candidate = {key: qa[key] for key in _CHEMISTRY_CANDIDATE_INPUT_FIELDS if key in qa}
        if candidate.get("candidate_id") is not None:
            _string(candidate["candidate_id"], "candidate.candidate_id")
            ids.append(candidate["candidate_id"])
        candidates.append(candidate)
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique within a verifier request")
    return candidates


def _object(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} must have exactly: {', '.join(keys)}")
    return value


def _string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _enum(value: Any, choices: Sequence[str], label: str) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"Invalid {label}: {value!r}")


def _references(value: Any, choices: Sequence[str] | set[str], label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(v, str) or v not in choices for v in value):
        raise ValueError(f"{label} contains unknown references")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate references")


def _strict_items(data: Any, want: int, keys: Sequence[str]) -> list[dict[str, Any]]:
    """Validate the legacy legal contract without changing chemistry's fallback."""
    items = [data] if isinstance(data, dict) and want == 1 else data
    if not isinstance(items, list) or len(items) != want or any(not isinstance(i, dict) for i in items):
        raise ValueError(f"Verifier must return exactly {want} candidate grades")
    indices = [item.get("index") for item in items]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(want)):
        raise ValueError("Verifier candidate indices are missing, duplicated, or out of range")
    for item in items:
        for key in keys:
            value = item.get(key)
            if type(value) is not int or not 1 <= value <= 5:
                raise ValueError(f"Verifier score {key} must be an integer from 1 to 5")
    return sorted(items, key=lambda item: item["index"])


def _normalize_compact_quality(
    data: Any, keys: Sequence[str], candidates: Sequence[Mapping[str, Any]], mode: str,
    *, rubric_version: str | None = None,
) -> list[dict[str, Any]]:
    """Validate compact quality contracts without fabricating scores."""
    _object(data, ("candidates", "batch_diversity"), "Quality response")
    items = data["candidates"]
    if not isinstance(items, list) or len(items) != len(candidates):
        raise ValueError(f"Verifier must return exactly {len(candidates)} candidate grades")
    diversity = data["batch_diversity"]
    if not isinstance(diversity, str):
        raise ValueError("batch_diversity must be a string")  # noqa: TRY004
    if len(candidates) <= 1:
        diversity = "not_applicable"
    elif diversity not in ("pass", "fail", "uncertain"):
        # This audit-only field does not affect candidate grades. Models often
        # answer it in prose despite the requested enum; keep clear judgments
        # and use the neutral value when the wording is ambiguous.
        normalized = diversity.casefold().replace("_", " ")
        if any(term in normalized for term in ("duplicate", "same need", "overlap")):
            diversity = "fail"
        elif any(term in normalized for term in ("distinct", "different need", "satisfactory")):
            diversity = "pass"
        else:
            diversity = "uncertain"
    rows = []
    for index, (item, candidate) in enumerate(zip(items, candidates)):
        _object(item, ("index", "candidate_id", "scores", "score_notes", "checks", "problems"), "Quality grade")
        if (type(item["index"]) is not int or item["index"] != index
                or item["candidate_id"] != candidate.get("candidate_id")):
            raise ValueError("Verifier changed candidate identity or input order")
        _object(item["scores"], keys, "scores")
        _object(item["score_notes"], keys, "score_notes")
        for key, score in item["scores"].items():
            if score is not None and (type(score) is not int or not 1 <= score <= 5):
                raise ValueError(f"Score {key} must be an integer from 1 to 5 or null")
            _string(item["score_notes"][key], f"score_notes.{key}")
            if rubric_version in _CHEMISTRY_RUBRIC_VERSIONS and score == 5:
                note = item["score_notes"][key].strip()
                if not re.match(r"Exceptional evidence:\s*\S", note):
                    raise ValueError(f"Score note {key} requires Exceptional evidence: for a 5")
        _object(item["checks"], ("mode", "support", "metadata"), "checks")
        for key, status in item["checks"].items():
            _enum(status, ("pass", "fail", "uncertain"), f"checks.{key}")
        if not isinstance(item["problems"], list):
            raise ValueError("Quality problems must be an array")  # noqa: TRY004 -- invalid verifier JSON
        for problem in item["problems"]:
            _string(problem, "problem")
        if any(status != "pass" for status in item["checks"].values()) and not item["problems"]:
            raise ValueError("Failed or uncertain quality checks require an explanation in problems")
        if rubric_version in _CHEMISTRY_RUBRIC_VERSIONS:
            for key, status in item["checks"].items():
                if status != "pass" and not any(
                    re.match(rf"{key}:\s*\S", problem.strip()) for problem in item["problems"]
                ):
                    raise ValueError(f"A {status} {key} check requires a {key}: explanation in problems")
            resolution_score = item["scores"][keys[2]]
            support = item["checks"]["support"]
            if support == "fail" and (resolution_score is None or resolution_score > 2):
                raise ValueError("A failed support check requires an answer-resolution score of 1 or 2")
            if support == "uncertain" and resolution_score is not None and resolution_score > 3:
                raise ValueError("An uncertain support check requires an answer-resolution score at most 3 or null")
        row = dict(item["scores"], _keys=list(keys), _response=item,
                   _batch_diversity=diversity, _contract="compact")
        if rubric_version is not None:
            row["_rubric_version"] = rubric_version
        row["overall"] = quality_overall(row, mode)
        row["reason"] = "; ".join(f"{key}: {note}" for key, note in item["score_notes"].items())
        rows.append(row)
    return rows


def _normalize_chemistry_faith(
    data: Any, want: int, rubric_version: str,
) -> list[dict[str, Any]]:
    """Preserve unassessable criteria and enforce the chemistry evidence contract."""
    if not isinstance(data, list) or len(data) != want:
        raise ValueError(f"Chemistry verifier must return exactly {want} candidate grades in an array")
    rows = []
    exceptional_labels = {
        "grounding": "Grounding exceptional:",
        "precision": "Precision exceptional:",
        "numerical_fidelity": "Numerical fidelity exceptional:",
    }
    for index, item in enumerate(data):
        _object(item, ("index", *FAITHFULNESS_KEYS, "reason"), "Faithfulness grade")
        if type(item["index"]) is not int or item["index"] != index:
            raise ValueError("Chemistry verifier changed candidate identity or input order")
        _string(item["reason"], "reason")
        for key in FAITHFULNESS_KEYS:
            score = item[key]
            if score is not None and (type(score) is not int or not 1 <= score <= 5):
                raise ValueError(f"Score {key} must be an integer from 1 to 5 or null")
            if score == 5:
                label = exceptional_labels[key]
                evidence = re.search(
                    rf"{re.escape(label)}\s*(.*?)(?=(?:Grounding|Precision|Numerical fidelity) exceptional:|$)",
                    item["reason"], re.DOTALL,
                )
                if evidence is None or not evidence.group(1).strip():
                    raise ValueError(f"A {key} score of 5 requires {label} evidence in reason")
        row = {key: item[key] for key in FAITHFULNESS_KEYS}
        row.update(reason=item["reason"].strip(), overall=faith_overall(row),
                   _response=item, _rubric_version=rubric_version)
        rows.append(row)
    return rows


def _legal_input(
    prompt: str, qa_pairs: Sequence[Mapping[str, Any]], sources: Sequence[Mapping[str, Any]] | None,
    target_source_id: str | None, policies: Mapping[str, Any] | None, rubric_mode: str | None,
) -> dict[str, Any]:
    declared = re.search(r"^# MODE:\s*([a-z_]+)\s*$", prompt, re.MULTILINE)
    if not declared or (rubric_mode and rubric_mode != declared.group(1)):
        raise ValueError("Legal rubric mode is missing or disagrees with the supplied mode")
    if not sources or not target_source_id or not 1 <= len(qa_pairs) <= 3:
        raise ValueError("Legal grading requires labeled sources, a target ID, and 1–3 candidates")
    source_ids: set[str] = set()
    for source in sources:
        for field in ("source_id", "text"):
            _string(source.get(field), f"source.{field}")
        _enum(source.get("role"), ("target", "reference", "context", "alternative"), "source role")
        if source["source_id"] in source_ids:
            raise ValueError("Duplicate source ID in legal verifier input")
        if "metadata" in source and not isinstance(source["metadata"], dict):
            raise ValueError("Source metadata must be an object")
        source_ids.add(source["source_id"])
    if not any(s["source_id"] == target_source_id and s["role"] == "target" for s in sources):
        raise ValueError("The target source ID must identify a source with role=target")
    candidates = []
    for qa in qa_pairs:
        for field in ("candidate_id", "question_language", "question", "answer"):
            _string(qa.get(field), f"candidate.{field}")
        candidates.append({key: qa[key] for key in _CANDIDATE_INPUT_FIELDS if key in qa})
    if len({qa["candidate_id"] for qa in candidates}) != len(candidates):
        raise ValueError("Candidate IDs must be unique within a verifier request")
    envelope = dict(policies or {})
    protected = {"mode", "target_source_id", "sources", "candidates"}
    if protected & envelope.keys():
        raise ValueError("Policies cannot override verifier identities, sources, or candidates")
    envelope.update(mode=declared.group(1), target_source_id=target_source_id,
                    sources=list(sources), candidates=candidates)
    return envelope


def _normalize_legal_quality(data: Any, prompt: str, envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enforce the prompt's nested audit contract before calculating any scores."""
    candidates = envelope["candidates"]
    version = _legal_rubric_version(prompt)
    if version is None:
        raise ValueError("Structured legal grading requires a declared rubric version")
    keys = rubric_keys(prompt)
    metadata = re.search(r"^Required metadata_checks:\s*([a-z_, ]+)\.", prompt, re.MULTILINE)
    if len(keys) != 5 or len(set(keys)) != 5 or not metadata:
        raise ValueError("Legal rubric must declare its five metrics and required metadata checks")
    metadata_keys = tuple(part.strip() for part in metadata.group(1).split(","))
    issue_codes = ()
    if version == "legal-qg-v3.0":
        controlled = re.search(r"^# CONTROLLED ISSUE CODES\s*\n([a-z0-9_, ]+)", prompt, re.MULTILINE)
        if not controlled:
            raise ValueError("Legal v3 rubric must declare its controlled issue codes")
        issue_codes = tuple(code.strip() for code in controlled.group(1).split(","))
    if not isinstance(data, list) or len(data) != len(candidates):
        raise ValueError("Legal verifier must return one array entry per input candidate")
    sources = {source["source_id"]: source for source in envelope["sources"]}
    rows = []
    for index, (item, candidate) in enumerate(zip(data, candidates)):
        _object(item, _LEGAL_FIELDS, "Legal grade")
        if type(item["index"]) is not int or item["index"] != index or item["candidate_id"] != candidate["candidate_id"]:
            raise ValueError("Legal verifier changed candidate identity or input order")
        if item["rubric_version"] != version or item["mode"] != envelope["mode"]:
            raise ValueError("Legal verifier returned the wrong rubric version or mode")
        _enum(item["target_relation"], ("central", "joint", "incidental", "absent", "uncertain"), "target_relation")
        _enum(item["reasoning_requirement"], ("extraction", "direct_application", "supported_synthesis", "external_or_open_ended", "uncertain"), "reasoning_requirement")
        _enum(item["confidence"], ("high", "medium", "low"), "confidence")
        if item["suggested_repair"] is not None:
            _string(item["suggested_repair"], "suggested_repair")
        if item["suggested_mode"] is not None:
            _enum(item["suggested_mode"], ("eurlex_fact_pattern", "eurlex_lookup", "eurlex_conceptual", "un_lookup", "un_practitioner", "un_semantic", "un_conceptual", "other_persona"), "suggested_mode")
        if not isinstance(item["evidence"], list):
            raise ValueError("Evidence must be an array")  # noqa: TRY004 -- invalid verifier JSON
        if version == "legal-qg-v3.0" and len(item["evidence"]) > 10:
            raise ValueError("Legal v3 evidence must contain at most 10 quotations")
        evidence = {}
        for entry in item["evidence"]:
            _object(entry, ("evidence_id", "source_id", "quote"), "evidence")
            for field in entry:
                _string(entry[field], f"evidence.{field}")
            if entry["evidence_id"] in evidence or entry["source_id"] not in sources:
                raise ValueError("Evidence contains duplicate IDs or unknown source IDs")
            quote = " ".join(entry["quote"].split())
            source_text = " ".join(sources[entry["source_id"]]["text"].split())
            if quote not in source_text:
                raise ValueError(f"Evidence {entry['evidence_id']} is not an exact source quotation")
            evidence[entry["evidence_id"]] = entry
        _object(item["validity"], _VALIDITY_KEYS, "validity")
        for check in item["validity"].values():
            _object(check, ("status", "evidence_ids", "note"), "validity check")
            _enum(check["status"], ("pass", "fail", "uncertain", "not_applicable"), "validity status")
            _references(check["evidence_ids"], set(evidence), "validity evidence_ids")
            _string(check["note"], "validity note")
        relevance = item["validity"]["target_relevance"]
        target = sources[envelope["target_source_id"]]
        target_ids = {target["source_id"]}
        if envelope.get("target_granularity") == "document":
            target_meta = target.get("metadata", {})
            for source in sources.values():
                other_meta = source.get("metadata", {})
                if any(target_meta.get(k) and target_meta[k] == other_meta.get(k) for k in ("document_id", "doc_id", "celex_id")):
                    target_ids.add(source["source_id"])
        if relevance["status"] == "pass" and not any(evidence[e]["source_id"] in target_ids for e in relevance["evidence_ids"]):
            raise ValueError("A target-relevance pass requires evidence from the configured target")
        _object(item["scores"], keys, "scores")
        _object(item["score_notes"], keys, "score_notes")
        for key, score in item["scores"].items():
            if score is not None and (type(score) is not int or not 1 <= score <= 5):
                raise ValueError(f"Score {key} must be an integer from 1 to 5 or null")
            _string(item["score_notes"][key], f"score_notes.{key}")
            if version == "legal-qg-v3.0":
                note = re.fullmatch(r"Basis:\s*(.*?)[.;]\s*Test:\s*(.*?)[.;]\s*Limit:\s*(.*)",
                                    item["score_notes"][key].strip(), re.DOTALL)
                if not note or any(not part.strip() for part in note.groups()):
                    raise ValueError(f"Score note {key} must contain nonempty Basis, Test, and Limit fields")
        checks = item["metadata_checks"]
        if not isinstance(checks, list) or len(checks) != len(metadata_keys):
            raise ValueError("Missing or extra mode metadata checks")
        checked_fields = []
        for check in checks:
            _object(check, ("field", "status", "note"), "metadata check")
            _enum(check["field"], metadata_keys, "metadata field")
            _enum(check["status"], ("ok", "error", "uncertain", "not_applicable"), "metadata status")
            _string(check["note"], "metadata note")
            checked_fields.append(check["field"])
        if len(set(checked_fields)) != len(metadata_keys):
            raise ValueError("Duplicate mode metadata checks")
        if not isinstance(item["issues"], list):
            raise ValueError("Issues must be an array")  # noqa: TRY004 -- invalid verifier JSON
        for issue in item["issues"]:
            _object(issue, ("code", "layer", "severity", "affected_criteria", "evidence_ids", "note"), "issue")
            if not isinstance(issue["code"], str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", issue["code"]):
                raise ValueError("Issue code must be snake_case")
            if version == "legal-qg-v3.0":
                _enum(issue["code"], issue_codes, "issue code")
            _enum(issue["layer"], ("validity", "mode", "metadata", "quality", "input", "collection"), "issue layer")
            _enum(issue["severity"], ("blocking", "repair", "minor", "review"), "issue severity")
            if issue["severity"] == "blocking" and issue["layer"] not in ("validity", "mode"):
                raise ValueError("Only validity or mode issues may be blocking")
            _references(issue["affected_criteria"], keys, "issue affected_criteria")
            _references(issue["evidence_ids"], set(evidence), "issue evidence_ids")
            _string(issue["note"], "issue note")
        if any(value is None for value in item["scores"].values()) and not any(issue["severity"] == "review" for issue in item["issues"]):
            raise ValueError("Unassessable scores require a review issue")
        row = dict(item["scores"], _keys=list(keys), _response=item)
        row["overall"] = quality_overall(row, item["mode"])
        row["reason"] = "; ".join(f"{k}: {v}" for k, v in item["score_notes"].items())
        rows.append(row)
    return rows


def _verification_reuse_identity(qa_pairs, reuse_identity):
    """Evidence-aware candidates must retain their complete verification identity."""
    identity = dict(reuse_identity or {})
    if any(qa.get("evidence_contract") or qa.get("evidence_spans") for qa in qa_pairs):
        identity.pop("deduplicate", None)
    return identity


def grade_faithfulness(
    client: Any,
    config: GraderConfig,
    prompt: str,
    passages: str,
    qa_pairs: Sequence[Mapping[str, Any]],
    *,
    expected: int | None = None,
    strict: bool = False,
    reuse_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Grade answers for grounding, precision and numerical fidelity.

    ``expected`` is the candidate count to return (defaults to ``len(qa_pairs)``).
    Legacy callers retain score-1 padding. Legal runs use ``strict=True`` and
    versioned chemistry rubrics always validate their declared contract, so
    malformed grades raise and can be retried without inventing scores.
    """
    want = expected if expected is not None else len(qa_pairs)
    chemistry_version = _chemistry_rubric_version(prompt)
    if chemistry_version:
        if not re.search(r"^# MODE:\s*chemistry_faithfulness_all_seven_personas\s*$", prompt, re.MULTILINE):
            raise ValueError("Chemistry faithfulness rubric declares the wrong mode")
        if want != len(qa_pairs):
            raise ValueError("Chemistry expected count must match the supplied candidates")
        if not want:
            return []
        chemistry_candidates = _chemistry_candidates(qa_pairs)
        strict = True
    reuse = getattr(client, "reuse_verification", None)
    if strict and reuse is not None and want == len(qa_pairs):
        return reuse({"config": asdict(config), "prompt": prompt, "passages": passages,
                      **_verification_reuse_identity(qa_pairs, reuse_identity)},
                     qa_pairs, lambda unique: grade_faithfulness(
                         SimpleNamespace(chat=client.chat), config, prompt, passages, unique, strict=True))
    if chemistry_version:
        envelope = {"passages": passages, "candidates": chemistry_candidates}
        raw = _invoke(client, config, prompt, json.dumps(envelope, ensure_ascii=False))
        return _normalize_chemistry_faith(parse_json_response(raw), want, chemistry_version)
    raw = _invoke(client, config, prompt, f"{passages}\n\n{candidates_block(qa_pairs)}")
    data = parse_json_response(raw)
    items = (_strict_items(data, want, FAITHFULNESS_KEYS) if strict else
             sorted(_as_list(data)[:want], key=lambda x: x.get("index", 0)))
    rows = [_normalize_faith(item) for item in items]
    if strict:
        for row, item in zip(rows, items):
            row["_response"] = item
    while len(rows) < want:
        rows.append(_missing_faith())
    return rows


def grade_quality(
    client: Any,
    config: GraderConfig,
    prompt: str,
    passages: str,
    qa_pairs: Sequence[Mapping[str, Any]],
    mode: str,
    *,
    expected: int | None = None,
    sources: Sequence[Mapping[str, Any]] | None = None,
    target_source_id: str | None = None,
    policies: Mapping[str, Any] | None = None,
    rubric_mode: str | None = None,
    strict: bool = False,
    reuse_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Grade questions using the rubric's legacy or structured audit contract.

    Structured legal and chemistry prompts always require strict identity and
    score validation, with additional checks from the declared rubric contract.
    The host supplies source boundaries; generator identity and previous scores
    are excluded from the candidate payload.
    """
    want = expected if expected is not None else len(qa_pairs)
    chemistry_version = _chemistry_rubric_version(prompt)
    if chemistry_version:
        declared = re.search(r"^# MODE:\s*([a-z_]+)\s*$", prompt, re.MULTILINE)
        expected_mode = mode if mode.startswith("chemistry_") else f"chemistry_{mode}"
        if not declared or declared.group(1) != expected_mode:
            raise ValueError("Chemistry rubric mode disagrees with the supplied mode")
        if rubric_mode and rubric_mode != declared.group(1):
            raise ValueError("Chemistry rubric mode disagrees with the supplied rubric mode")
        if want != len(qa_pairs):
            raise ValueError("Chemistry expected count must match the supplied candidates")
        if _quality_output_example(prompt) is None:
            raise ValueError("Chemistry quality rubric must declare its compact output contract")
        if not want:
            return []
        chemistry_candidates = _chemistry_candidates(qa_pairs)
        strict = True
    reuse = getattr(client, "reuse_verification", None)
    if strict and reuse is not None and want == len(qa_pairs) and 1 <= want <= 3:
        ids = [qa["candidate_id"] for qa in qa_pairs if qa.get("candidate_id") is not None]
        for identity in ids:
            _string(identity, "candidate.candidate_id")
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate IDs must be unique within a verifier request")
        return reuse({"config": asdict(config), "prompt": prompt, "passages": passages,
                      "mode": mode, "sources": sources, "target_source_id": target_source_id,
                      "policies": policies, "rubric_mode": rubric_mode,
                      **_verification_reuse_identity(qa_pairs, reuse_identity)},
                     qa_pairs, lambda unique: grade_quality(
                         SimpleNamespace(chat=client.chat), config, prompt, passages, unique, mode,
                         sources=sources, target_source_id=target_source_id, policies=policies,
                         rubric_mode=rubric_mode, strict=True))
    if chemistry_version:
        candidates = chemistry_candidates
        envelope = {"passages": passages, "candidates": candidates}
        raw = _invoke(client, config, prompt, json.dumps(envelope, ensure_ascii=False))
        return _normalize_compact_quality(
            parse_json_response(raw), rubric_keys(prompt), candidates, mode,
            rubric_version=chemistry_version,
        )
    if _legal_rubric_version(prompt):
        if want != len(qa_pairs):
            raise ValueError("Legal expected count must match the supplied candidates")
        envelope = _legal_input(prompt, qa_pairs, sources, target_source_id, policies, rubric_mode)
        raw = _invoke(client, config, prompt, json.dumps(envelope, ensure_ascii=False))
        return _normalize_legal_quality(parse_json_response(raw), prompt, envelope)
    if _quality_output_example(prompt) is not None:
        if want != len(qa_pairs) or not 1 <= want <= 3:
            raise ValueError("Legal grading requires 1–3 candidates matching the expected count")
        candidates = [{key: qa[key] for key in _CANDIDATE_INPUT_FIELDS if key in qa} for qa in qa_pairs]
        ids = []
        for candidate in candidates:
            for field in ("question", "answer"):
                _string(candidate.get(field), f"candidate.{field}")
            if candidate.get("candidate_id") is not None:
                _string(candidate["candidate_id"], "candidate.candidate_id")
                ids.append(candidate["candidate_id"])
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate IDs must be unique within a verifier request")
        # Keep every source boundary and identifier in the canonical passage,
        # and serialize all declared candidate metadata without model/score leaks.
        envelope = {"passages": passages, "candidates": candidates}
        if sources and any(qa.get("evidence_contract") or qa.get("evidence_spans")
                           for qa in qa_pairs):
            # Preserve language/body boundaries for evidence-span checks. Older
            # compact contracts keep their original request envelope.
            envelope.update(sources=list(sources), target_source_id=target_source_id,
                            policies=dict(policies or {}))
        raw = _invoke(client, config, prompt, json.dumps(envelope, ensure_ascii=False))
        return _normalize_compact_quality(parse_json_response(raw), rubric_keys(prompt), candidates, mode)
    raw = _invoke(client, config, prompt, f"{passages}\n\n{candidates_block(qa_pairs)}")
    keys = rubric_keys(prompt) or quality_keys(mode)
    data = parse_json_response(raw)
    items = (_strict_items(data, want, keys) if strict else
             sorted(_as_list(data)[:want], key=lambda x: x.get("index", 0)))
    rows = [_normalize_quality(item, mode, keys) for item in items]
    if strict:
        for row, item in zip(rows, items):
            row["_response"] = item
    while len(rows) < want:
        rows.append(_missing_quality(mode, keys))
    return rows


def grade_one(
    client: Any,
    config: GraderConfig,
    faith_prompt: str,
    quality_prompt: str,
    passages: str,
    qa_pair: Mapping[str, str],
    mode: str = MODE_TECHNICAL,
    *,
    strict: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Grade a single pair with the single-object rubric prompts.

    Used by the concept/variant/progressive pipelines, whose prompts return one
    JSON object rather than a list of three.
    """
    faith = grade_faithfulness(
        client, config, faith_prompt, passages, [qa_pair], expected=1, strict=strict
    )[0]
    quality = grade_quality(
        client, config, quality_prompt, passages, [qa_pair], mode, expected=1, strict=strict
    )[0]
    return faith, quality


def grade_columns(
    faith: Mapping[str, Any], quality: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    """Flatten grades into the ``faith_*`` / ``qual_*`` / ``total_score`` columns."""
    row: dict[str, Any] = {}
    for key in FAITHFULNESS_KEYS:
        row[f"faith_{key}"] = faith.get(key, "")
    row["faith_overall"] = faith.get("overall", faith_overall(faith))
    for key in (quality.get("_keys") or quality_keys(mode)):
        row[f"qual_{key}"] = quality.get(key, "")
    row["qual_overall"] = quality.get("overall", quality_overall(quality, mode))
    row["faith_reason"] = faith.get("reason", "")
    row["qual_failure_type"] = quality.get("failure_type", "")
    row["qual_reason"] = quality.get("reason", "")
    row["total_score"] = total_score(faith, quality, mode)
    if "_response" in faith:
        row["faithfulness_verifier_response_json"] = json.dumps(faith["_response"], ensure_ascii=False)
        if "_rubric_version" in faith:
            row["faithfulness_rubric_version"] = faith["_rubric_version"]
    if "_response" in quality:
        audit = quality["_response"]
        row["quality_verifier_response_json"] = json.dumps(audit, ensure_ascii=False)
        row["quality_rubric_version"] = audit.get("rubric_version", "legacy")
        row["quality_audit_status"] = "legacy_not_provided"
        if quality.get("_contract") == "compact":
            row["quality_rubric_version"] = quality.get("_rubric_version", "unversioned_compact")
            row["quality_batch_diversity"] = quality["_batch_diversity"]
            row["quality_score_notes_json"] = json.dumps(audit["score_notes"], ensure_ascii=False)
            row["quality_checks_json"] = json.dumps(audit["checks"], ensure_ascii=False)
            row["quality_problems_json"] = json.dumps(audit["problems"], ensure_ascii=False)
            for key, status in audit["checks"].items():
                row[f"quality_{key}_status"] = status
            if audit["checks"]["mode"] == "fail" or audit["checks"]["support"] == "fail":
                status = "blocking"
            elif "uncertain" in audit["checks"].values() or quality["overall"] is None:
                status = "review"
            elif audit["checks"]["metadata"] == "fail":
                status = "repair"
            else:
                status = "minor_issues" if audit["problems"] else "clear"
            row["quality_audit_status"] = status
        elif audit.get("rubric_version") in _LEGAL_RUBRIC_VERSIONS:
            severities = {issue["severity"] for issue in audit["issues"]}
            validity_statuses = {check["status"] for check in audit["validity"].values()}
            metadata_statuses = {check["status"] for check in audit["metadata_checks"]}
            if "blocking" in severities or "fail" in validity_statuses:
                status = "blocking"
            elif "review" in severities or "uncertain" in validity_statuses or "uncertain" in metadata_statuses or quality["overall"] is None:
                status = "review"
            elif "repair" in severities or "error" in metadata_statuses:
                status = "repair"
            else:
                status = "minor_issues" if "minor" in severities else "clear"
            row["quality_audit_status"] = status
            row["quality_target_relation"] = audit["target_relation"]
            row["quality_reasoning_requirement"] = audit["reasoning_requirement"]
            for key, check in audit["validity"].items():
                row[f"quality_{key}_status"] = check["status"]
    return row


@dataclass(frozen=True)
class GradedCandidate:
    """One generated Q/A pair with its grades, ranked by ``total_score``."""

    qa: Mapping[str, str]
    faith: Mapping[str, Any]
    quality: Mapping[str, Any]
    mode: str

    @property
    def total(self) -> int | None:
        return total_score(self.faith, self.quality, self.mode)


def rank_candidates(
    qa_pairs: Sequence[Mapping[str, str]],
    faith_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> list[GradedCandidate]:
    """Zip candidates with their grades, best total score first."""
    graded = [
        GradedCandidate(qa=qa, faith=faith_rows[i], quality=quality_rows[i], mode=mode)
        for i, qa in enumerate(qa_pairs)
        if i < len(faith_rows) and i < len(quality_rows)
    ]
    return sorted(graded, key=lambda c: (c.total is not None, c.total or 0), reverse=True)


__all__ = [
    "CONCEPTUAL_QUALITY_FIELDS",
    "CONCEPTUAL_QUALITY_KEYS",
    "FACT_PATTERN_QUALITY_FIELDS",
    "FACT_PATTERN_QUALITY_KEYS",
    "FAITHFULNESS_FIELDS",
    "FAITHFULNESS_KEYS",
    "LOOKUP_QUALITY_FIELDS",
    "LOOKUP_QUALITY_KEYS",
    "MODE_CONCEPTUAL",
    "MODE_DESCRIPTIVE",
    "MODE_FACT_PATTERN",
    "MODE_LOOKUP",
    "MODE_SEMANTIC",
    "MODE_TECHNICAL",
    "SEMANTIC_QUALITY_FIELDS",
    "TECHNICAL_QUALITY_FIELDS",
    "GradedCandidate",
    "GraderConfig",
    "candidates_block",
    "faith_overall",
    "grade_columns",
    "grade_faithfulness",
    "grade_one",
    "grade_quality",
    "quality_fields",
    "quality_keys",
    "quality_overall",
    "rank_candidates",
    "rubric_keys",
    "total_score",
]
