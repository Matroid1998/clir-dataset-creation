"""
Build a UN question set: select target blocks, generate, grade, rank, write.

One target block yields three candidates; the best-scoring one is kept. So a
100-query set is 100 target blocks: 100 generation calls plus 200 grading calls.

Selection is stratified by **genre** with user-fixed shares: resolutions &
decisions 50%, meeting records 40%, letters 10% (``GENRE_STRATA``; override
with ``--shares``, or ``--all-genres`` for the unfiltered pool). Within a
stratum the sampling unit is the **block**, ranked by a seeded hash of its
block id -- documents contribute targets in proportion to their content, the
same way the EUR-Lex batch samples articles (``--max-per-doc`` remains a
safety knob, default uncapped).

Filters applied before sampling:

* **genre filter** -- only resolutions/decisions, meeting records, and
  letters become question sources (``genre_for``); everything else, plus the
  logistics classes (agenda stubs, the daily Journal, corrigenda), is
  excluded;
* **whole-fit filter** -- a target qualifies only when its whole document
  PLUS the whole text of every referenced document stays inside the 30k-char
  context budget (``_fits_whole``, checked lazily along the ranked walk), so
  nothing the model sees is ever windowed or truncated; references are then
  rendered as full documents (``--no-fit`` reverts to windowed context and
  capped excerpts);
* only blocks inside the token window (``in_range``) AND free of Layer-2
  shape flags (``usable``) become targets -- the builder records the
  surviving indices per document (``target_idxs``), so pool enumeration
  never scans the 2.6 GB blocks file.

Usage:
    python -m clir_bench.domains.legal.qac.un_batch --n 100 --dry-run
    python -m clir_bench.domains.legal.qac.un_batch --n 100 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clir_bench.domains.legal.qac import un_context as ctx
from clir_bench.domains.legal.qac import un_generate as gen
from clir_bench.domains.legal.qac import un_references as refs
from clir_bench.domains.legal.qac.env import load_env
from clir_bench.domains.legal.un import UN_LANGUAGES
from clir_bench.domains.legal.un import paths as un_paths

DEFAULT_GEN_MODEL = "gpt-5.6-luna"

OUT_DIR = un_paths.QAC_DIR

# Question-source genres and their shares of the set (user-fixed 50/40/10).
# Measured pool under both filters: resolutions 30,043 targets / meeting
# records 31,072 / letters 2,416 -- the letters quota of a 10k set uses ~41%
# of what exists, the other two are barely dented.
GENRE_STRATA: tuple[tuple[str, float], ...] = (
    ("resolution", 0.50),
    ("meeting", 0.40),
    ("letter", 0.10),
)

# The reject floor: a candidate whose faithfulness grader gave GROUNDING <= 2
# was judged NOT fully answerable from the target block (substance drawn from a
# referenced document, the context, or outside knowledge). Such a candidate is
# never kept as a target's best question; if none of the three clears the
# floor, the target yields no question at all. All candidates and grades are
# still written to the all-candidates file, so rejects stay auditable.
MIN_GROUNDING_FOR_BEST = 3

# The fit filter: a target qualifies only when its whole document PLUS the
# whole text of every referenced document stays inside the context budget --
# then nothing the model sees is ever truncated or windowed.
FIT_BUDGET = ctx.DEFAULT_CONTEXT_CHARS
DEFAULT_MODES = (gen.MODE_TECHNICAL, gen.MODE_CONCEPTUAL, gen.MODE_DESCRIPTIVE)
SUPPORTED_MODES = gen.MODES


def genre_for(doc_id: str, title: str) -> str | None:
    """Question-source genre, or None when the document class is ineligible.

    Letters are recognised by their title line and undercounted when that line
    is corrupted -- a safe direction: never misclassifies, only misses.
    """
    if "/res/" in doc_id or "/dec/" in doc_id:
        return "resolution"
    if "sr_" in doc_id or "pv_" in doc_id:
        return "meeting"
    if title.lower().startswith(("letter dated", "note verbale", "identical letters")):
        return "letter"
    return None


def _fits_whole(doc: dict, block_text: str, index: ctx.BlockIndex) -> bool:
    """Whole document + whole referenced documents within the context budget."""
    if doc["char_count"] > FIT_BUDGET:
        return False
    citations = refs.resolve_citations(
        refs.extract_citations(block_text),
        getattr(index, "symbols", None) or index.symbol_map,
        citing_doc_id=doc["doc_id"])
    kept, _ = refs.referenced_docs(citations)
    total = doc["char_count"] + sum(
        index.docs[c.doc_id]["char_count"] for c in kept)
    return total <= FIT_BUDGET


def _cited_doc_ids(index: ctx.BlockIndex, target: Target) -> set[str]:
    """In-corpus documents whose text will travel as this target's references.

    Mirrors the resolution ``un_context.build`` performs, so the preloader and
    the payload agree on exactly which documents are needed.
    """
    block = index.blocks_for(target.doc_id)[target.block_index]
    citations = refs.resolve_citations(
        refs.extract_citations(block.texts["en"], doc_id=target.doc_id),
        getattr(index, "symbols", None) or index.symbol_map,
        citing_doc_id=target.doc_id)
    kept, _ = refs.referenced_docs(citations)
    return {c.doc_id for c in kept if c.doc_id}


@dataclass
class Target:
    doc_id: str
    block_id: str
    block_index: int
    n_blocks: int
    stratum: str
    mode: str
    language: str
    reference_complete: bool = True
    n_unresolved: int = 0
    unresolved_reasons: str = ""    # "reason:count;..." for gated-in blocks


def load_targets(path: Path) -> list[Target]:
    """Read a fixed comparison plan, preserving its documents, order and modes."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("target manifest must contain a nonempty JSON list")
    targets, seen = [], set()
    for position, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"target {position} must be an object")
        try:
            target = Target(**row)
        except TypeError as error:
            raise ValueError(f"invalid target {position}: {error}") from error
        if target.mode not in SUPPORTED_MODES or target.language not in UN_LANGUAGES:
            raise ValueError(f"unsupported mode/language in target {position}")
        if (not isinstance(target.doc_id, str) or not target.doc_id.strip()
                or type(target.block_index) is not int or target.block_index < 0
                or type(target.n_blocks) is not int or target.n_blocks <= target.block_index
                or target.block_id != f"{target.doc_id}#{target.block_index}"):
            raise ValueError(f"invalid block identity in target {position}")
        key = (target.block_id, target.mode, target.language)
        if key in seen:
            raise ValueError(f"duplicate target: {key}")
        seen.add(key)
        targets.append(target)
    return targets


def write_targets(path: Path, targets: Sequence[Target]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(t) for t in targets], ensure_ascii=False, indent=2)
                    + "\n", encoding="utf-8")


def target_from_rows(rows: Sequence[dict[str, Any]]) -> Target:
    """Recover a target from saved candidates, preserving its original selection."""
    row = rows[0]
    return Target(
        doc_id=row.get("doc_id") or row["document_id"],
        block_id=row.get("block_id") or row["target_id"],
        block_index=int(row["block_index"]), n_blocks=int(row["n_blocks"]),
        stratum=row.get("stratum", ""), mode=row["mode"], language=row["question_language"],
        reference_complete=str(row.get("reference_complete", "")).lower() == "true",
        n_unresolved=int(row.get("n_unresolved") or 0),
        unresolved_reasons=row.get("unresolved_reasons", ""),
    )


def prepare_payload(target: Target, index: ctx.BlockIndex, *,
                    context_chars: int = ctx.DEFAULT_CONTEXT_CHARS,
                    reference_chars: int | None = None):
    return index.build(target.doc_id, target.block_index, context_chars=context_chars,
                       reference_chars=reference_chars,
                       languages=ctx.payload_languages(target.language))


def quality_sources(payload: ctx.GenerationPayload, language: str, *,
                    evidence_contract: bool = False) -> list[dict[str, Any]]:
    """Separate answer-bearing target text from understanding-only supplied material."""
    languages = [lg for lg in ctx.payload_languages(language) if payload.target.texts.get(lg)]
    sources = []
    for unit, role in ([(payload.target, "target")]
                       + [(unit, "context") for unit in payload.context_blocks]):
        source = {
            "source_id": unit.block_id, "role": role,
            "text": "\n\n".join(
                (ctx._metadata(unit, lg) + "\n" if role == "target" else "") + unit.texts[lg]
                for lg in languages if unit.texts.get(lg)),
            "metadata": {"doc_id": unit.doc_id, "symbol": unit.symbol, "title": unit.title,
                         "block_index": unit.block_index, "languages": languages},
        }
        if role == "target" and evidence_contract and getattr(payload, "evidence_sources", None):
            visible = [body for body in payload.evidence_sources.values()
                       if body.source_id == unit.block_id]
            source["texts"] = {body.language: body.text for body in visible}
            source["text"] = "\n\n".join(
                f"[{body.language.upper()}]\n{body.text}" for body in visible)
            source["metadata"].update(languages=list(source["texts"]), source_ref="target")
        sources.append(source)
    for unit in payload.references:
        sources.append({
            "source_id": unit.doc_id, "role": "reference",
            "text": "\n\n".join(unit.texts[lg] for lg in languages if unit.texts.get(lg)),
            "metadata": {"doc_id": unit.doc_id, "symbol": unit.symbol, "title": unit.title,
                         "paragraph": unit.paragraph, "part_label": unit.part_label,
                         "languages": languages},
        })
    return sources


def _excluded_doc(doc_id: str) -> bool:
    """Layer 0: document classes that cannot anchor questions.

    Agenda stubs and the Journal are meeting logistics; corrigenda are edit
    instructions ("Replace paragraph X with...") about a *different* document
    -- both sampled corrigendum targets in the verification audit were bad.
    """
    pieces = doc_id.split("/")
    body = pieces[1] if len(pieces) > 1 else ""
    return ("/agenda/" in doc_id or body.startswith("journal")
            or any(p.startswith("corr") for p in pieces[2:]))


def select(index: ctx.BlockIndex, *, n: int, seed: int, languages: Sequence[str],
           modes: Sequence[str], strata=GENRE_STRATA,
           max_per_doc: int = 0, genre_filter: bool = True,
           fit_filter: bool = True, require_complete: bool = True,
           excluded_documents: frozenset[str] = frozenset()) -> list[Target]:
    """Stratified, deterministic, block-level target selection.

    The pool is enumerated from the docs index alone (``target_idxs``) and
    ranked by a seeded hash of the block id; documents contribute in
    proportion to their eligible-block count. The fit check is applied lazily
    while walking the ranked pool -- only candidates actually reached get
    their block text read and citations resolved, so cost scales with ``n``.
    ``max_per_doc`` is a safety cap only; 0 means uncapped.

    ``require_complete`` restricts the pool to reference-complete blocks: a
    block citing anything that could not be resolved -- another document
    outside the corpus, a treaty article, an annex that cannot be pinned down
    -- is not a question target. Needs ``reference_status_en.jsonl``.
    """
    def rank(key: str) -> str:
        return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()

    incomplete = index.incomplete
    if require_complete and incomplete is None:
        raise SystemExit(
            "reference_status_en.jsonl not found: run "
            "`python -m clir_bench.domains.legal.un.references_status` "
            "or pass --allow-incomplete")
    if not require_complete and incomplete is None:
        incomplete = {}

    if not genre_filter:
        strata = (("all", 1.0),)
    pools: dict[str, list[tuple[str, int]]] = {name: [] for name, _ in strata}
    missing_idxs = 0
    for doc_id, doc in index.docs.items():
        if _excluded_doc(doc_id) or doc_id in excluded_documents:
            continue
        idxs = doc.get("target_idxs")
        if idxs is None:
            missing_idxs += 1
            continue
        if fit_filter and doc["char_count"] > FIT_BUDGET:
            continue
        if genre_filter:
            stratum = genre_for(doc_id, doc.get("title", ""))
            if stratum is None:
                continue
        else:
            stratum = "all"
        if require_complete:
            idxs = [idx for idx in idxs
                    if f"{doc_id}#{idx}" not in incomplete]
        pools[stratum].extend((doc_id, idx) for idx in idxs)
    if missing_idxs and not any(pools.values()):
        raise SystemExit(
            "docs index has no target_idxs -- rebuild the blocks "
            "(python -m clir_bench.domains.legal.un.blocks)")

    chosen: list[Target] = []
    per_doc: Counter = Counter()
    quotas = [int(n * share) for _, share in strata]
    remainder_order = sorted(range(len(strata)),
                             key=lambda i: n * strata[i][-1] - quotas[i], reverse=True)
    for position in remainder_order[:n - sum(quotas)]:
        quotas[position] += 1
    for (name, _), want in zip(strata, quotas):
        taken = 0
        for doc_id, idx in sorted(pools[name], key=lambda p: rank(f"{p[0]}#{p[1]}")):
            if taken >= want:
                break
            if max_per_doc and per_doc[doc_id] >= max_per_doc:
                continue
            doc = index.docs[doc_id]
            if fit_filter and not _fits_whole(
                    doc, index.blocks_for(doc_id)[idx].texts["en"], index):
                continue
            per_doc[doc_id] += 1
            position = len(chosen)
            status = incomplete.get(f"{doc_id}#{idx}")
            chosen.append(Target(
                doc_id=doc_id, block_id=f"{doc_id}#{idx}",
                block_index=idx, n_blocks=doc["n_blocks"],
                stratum=name,
                # Alternate deterministically so the set is balanced across
                # modes and question languages rather than randomly lumpy.
                mode=modes[position % len(modes)],
                language=languages[position % len(languages)],
                reference_complete=status is None,
                n_unresolved=(sum(status.get("reasons", {}).values())
                              if status else 0),
                unresolved_reasons=";".join(
                    f"{k}:{v}" for k, v in
                    (status.get("reasons", {}) if status else {}).items()),
            ))
            taken += 1
    return chosen


def run_one(target: Target, index: ctx.BlockIndex, *, gen_model: str,
            grade_model: str, context_chars: int, keep: int,
            reference_chars: int | None = None,
            generation_recorder: Any = None,
            existing_candidates: list[dict[str, Any]] | None = None,
            generation_only: bool = False,
            deduplicate_exact_questions: bool = False,
            checkpoint: Any = None, retries: int = 3,
            payload: ctx.GenerationPayload | None = None) -> list[dict[str, Any]]:
    from clir_bench.core.grading import (
        GraderConfig,
        grade_columns,
        grade_faithfulness,
        grade_quality,
        rank_candidates,
    )
    from clir_bench.core.llm import call_with_retries, client_for

    # reference_chars=None renders each referenced document WHOLE -- safe
    # because the fit filter guaranteed doc + references fit the budget.
    enhanced = checkpoint is not None or existing_candidates is not None
    payload = payload or prepare_payload(target, index, context_chars=context_chars,
                                          reference_chars=reference_chars)
    if payload is None:
        return []
    # The UN quality rubrics are hostile-reviewer prompts of the same family as
    # the EUR-Lex ones: each runs a multi-step procedure and returns, per
    # candidate, an expert rewrite plus a prose flaw note for all five criteria.
    # The defaults (12k total, 8k of it thinking budget) leave ~4k for that and
    # the grader comes back empty or truncated. See eurlex_batch for the same fix.
    grader = GraderConfig(model=grade_model, reasoning_effort="low",
                          thinking_budget_tokens=16000, thinking_max_tokens=32000)
    def stage(name, fn):
        if checkpoint is not None:
            return checkpoint.run(name, fn)
        return call_with_retries(fn, retries=retries, label=name)

    def client(stage_name, model):
        transport = client_for(model)
        return checkpoint.client(stage_name, model, transport) if checkpoint else transport

    def generate_candidates():
        gen_client = client("generation", gen_model)
        if generation_recorder is not None:
            gen_client = generation_recorder.client(
                gen_client.with_options(max_retries=0, timeout=180),
                context={"corpus": "un", "target_id": target.block_id,
                         "doc_id": target.doc_id, "mode": target.mode, "language": target.language})
        return [asdict(candidate) for candidate in gen.generate(
            payload, mode=target.mode, language=target.language, model=gen_model,
            client=gen_client)]

    if existing_candidates is None:
        saved = stage("generation", generate_candidates)
        candidates = [gen.Candidate(**item) for item in saved]
    else:
        candidates = [gen.Candidate(
            **gen.annotations.restore_metadata(row, target.mode, eurlex=False),
            question=row["question"], answer=row["answer"],
            classification=row.get("framing" if target.mode in gen.CONCEPTUAL_MODES else "question_type", ""),
            question_cited=row.get("question_cited", ""), anchor=row.get("anchor", ""),
            anchors=[value for value in row.get("anchors", "").split(gen.ANCHOR_SEP) if value],
        ) for row in existing_candidates]
    if not candidates:
        return []

    if generation_only:
        # Persisted generation data is independent of grading. Emit a compact
        # row that the normal verifier path can restore after the generation
        # barrier, including on resume.
        result = []
        for position, candidate in enumerate(candidates):
            candidate_id = "q_" + hashlib.sha256(json.dumps(
                ["un", target.block_id, target.mode, target.language, gen_model,
                 position, candidate.question, candidate.answer],
                ensure_ascii=False).encode()).hexdigest()[:24]
            result.append({
                "candidate_id": candidate_id,
                "question": candidate.question,
                "answer": candidate.answer,
                "question_type": candidate.classification if target.mode not in gen.CONCEPTUAL_MODES else "",
                "framing": candidate.classification if target.mode in gen.CONCEPTUAL_MODES else "",
                **gen.annotations.row_metadata(candidate),
                "question_cited": candidate.question_cited,
                "anchor": candidate.anchor,
                "anchors": gen.ANCHOR_SEP.join(candidate.anchors),
            })
        return result

    qa = [{"question": c.question, "answer": c.answer, "_candidate_index": position,
           "candidate_id": ((existing_candidates[position].get("candidate_id") if existing_candidates else None)
                            or "q_" + hashlib.sha256(json.dumps(
                                ["un", target.block_id, target.mode, target.language, gen_model,
                                 position, c.question, c.answer], ensure_ascii=False).encode()).hexdigest()[:24]),
           "question_language": target.language} for position, c in enumerate(candidates)]
    evidence_contract = any(getattr(c, "evidence_contract", "")
                            or getattr(c, "evidence_spans", []) for c in candidates)
    for pair, candidate in zip(qa, candidates):
        if target.mode in gen.NEW_MODES or evidence_contract:
            pair.update(mode=target.mode, **gen.annotations.metadata(candidate))
    # The quality rubrics run consistency checks ON the mode's own fields -- is
    # the declared anchor actually a substring of the question, does the cited
    # rendering differ from the base one by nothing but the identifier -- none of
    # which is checkable unless the grader is shown them. Faithfulness keeps the
    # lean pair: it grades the answer against the block and the rest is noise.
    qa_quality = [
        dict({key: value for key, value in pair.items() if not key.startswith("_")}, **{key: value for key, value in (
            ("question_cited", c.question_cited),
            ("anchor", c.anchor),
            ("anchors", list(c.anchors)),
            # conceptual declares ``framing``; the other modes ``question_type``.
            ("framing" if target.mode in gen.CONCEPTUAL_MODES else "question_type",
             c.classification),
        )})
        for pair, c in zip(qa, candidates)
    ]
    faith_prompt = gen.PROMPTS.faithfulness("batch")
    quality_prompt = gen.PROMPTS.quality(target.mode, "batch")
    reuse_identity = ({"deduplicate": "exact_question"}
                      if deduplicate_exact_questions and not evidence_contract else None)
    errors = []
    faith = quality = None
    try:
        faith = stage("faithfulness", lambda: grade_faithfulness(
            client("faithfulness", grade_model), grader, faith_prompt, payload.text, qa,
            strict=enhanced,
            reuse_identity=reuse_identity))
    except Exception as error:
        if not enhanced:
            raise
        errors.append(f"faithfulness: {type(error).__name__}: {error}")
    try:
        quality = stage("quality", lambda: grade_quality(
            client("quality", grade_model), grader, quality_prompt, payload.text, qa_quality,
            target.mode, strict=enhanced, sources=quality_sources(
                payload, target.language, evidence_contract=bool(evidence_contract)),
            target_source_id=target.block_id, rubric_mode="un_practitioner" if target.mode == gen.MODE_PRACTITIONERS else f"un_{target.mode}",
            policies={"target_granularity": "block", "reference_policy": "target_only",
                      "answer_role": ("source_identity" if target.mode == gen.MODE_SOURCE_FINDING
                                      else "evidence_span"), "source_time_policy": "supplied_version",
                      "require_unique_gold": False},
            reuse_identity=reuse_identity))
    except Exception as error:
        if not enhanced:
            raise
        errors.append(f"quality: {type(error).__name__}: {error}")

    if faith is None:
        faith = [{"grounding": None, "precision": None, "numerical_fidelity": None, "overall": None}
                 for _ in candidates]
    if quality is None:
        from clir_bench.core.grading import rubric_keys
        keys = rubric_keys(quality_prompt)
        quality = [dict.fromkeys(keys) | {"overall": None, "_keys": keys} for _ in candidates]

    # Ranked best-first; row order carries the ranking, exactly as in
    # eurlex_batch: the whole list goes to the all-candidates file and the
    # first row of each target to the best-only file.
    ranked = rank_candidates(qa, faith, quality, target.mode)
    rows: list[dict[str, Any]] = []
    for rank, graded in enumerate(ranked[:keep], 1):
        position = graded.qa["_candidate_index"]
        candidate = candidates[position]
        row = {
            "doc_id": target.doc_id,
            "symbol": payload.target.symbol,
            "block_id": target.block_id,
            "block_index": target.block_index,
            "n_blocks": target.n_blocks,
            "line_start": payload.target.line_start,
            "line_end": payload.target.line_end,
            "stratum": target.stratum,
            "context_blocks_supplied": len(payload.context_blocks),
            "context_blocks_dropped": payload.n_context_dropped,
            "question_language": target.language,
            "mode": target.mode,
            "question": candidate.question,
            "answer": candidate.answer,
            **gen.annotations.row_metadata(candidate),
            "question_type": candidate.classification if target.mode not in gen.CONCEPTUAL_MODES else "",
            "framing": candidate.classification if target.mode in gen.CONCEPTUAL_MODES else "",
            # ``lookup`` fills question_cited/anchor; ``practitioners`` fills
            # anchors. The unused ones stay empty.
            "question_cited": candidate.question_cited,
            "anchor": candidate.anchor,
            "anchors": gen.ANCHOR_SEP.join(candidate.anchors),
            # The passage the question was generated from, exactly as the
            # generator saw it (metadata line + block text), so a row can be
            # audited without re-running the payload builder.
            "title": payload.target.title,
            "target_block_text": ctx.unit_source(payload.target, target.language),
            "references_supplied": ",".join(r.symbol for r in payload.references),
            "references_dropped": ",".join(payload.dropped_references),
            "reference_complete": target.reference_complete,
            "n_unresolved": target.n_unresolved,
            "unresolved_reasons": target.unresolved_reasons,
        }
        # Every score the verifiers returned, not just the aggregates: the
        # three faithfulness sub-criteria, the five mode-specific quality
        # sub-criteria, both overalls, the failure type, and both reasons.
        if enhanced and existing_candidates is not None:
            row = dict(existing_candidates[position], **row)
            # A changed rubric must not leave stale metrics or audit conclusions.
            for key in row:
                if key.startswith(("faith_", "qual_", "quality_", "faithfulness_verifier_")):
                    row[key] = ""
        row.update(grade_columns(graded.faith, graded.quality, target.mode))
        if enhanced:
            row.update({
                "corpus": "un", "document_id": target.doc_id, "target_id": target.block_id,
                "document_text_scope": "target_block", "document_text": row["target_block_text"],
                "candidate_id": graded.qa["candidate_id"],
                "candidate_rank": rank if graded.total is not None else "", "is_best": False,
                "generator_model_id": gen_model, "generator_model_name": gen_model.split("/")[-1],
                "faithfulness_verifier_model": grade_model, "quality_verifier_model": grade_model,
                "grading_status": "failed" if errors else "completed", "grading_error": "; ".join(errors),
                "source_payload_sha256": hashlib.sha256(payload.text.encode()).hexdigest(),
                "faithfulness_verifier_prompt": faith_prompt,
                "faithfulness_verifier_prompt_sha256": hashlib.sha256(faith_prompt.encode()).hexdigest(),
                "quality_verifier_prompt": quality_prompt,
                "quality_verifier_prompt_sha256": hashlib.sha256(quality_prompt.encode()).hexdigest(),
            })
        else:
            row = {key: value for key, value in row.items() if key in FIELDS}
        rows.append(row)
    return rows


def pick_best(grouped: dict[str, list[dict[str, Any]]]
              ) -> tuple[list[dict[str, Any]], int]:
    """(best rows, rejected-target count).

    Each group is one target's candidates, ranked best-first. The best is the
    first candidate whose GROUNDING clears ``MIN_GROUNDING_FOR_BEST``; a target
    with no such candidate contributes nothing to the best file.
    """
    best: list[dict[str, Any]] = []
    rejected = 0
    for key in sorted(grouped):
        for row in grouped[key]:
            try:
                grounding = int(row.get("faith_grounding") or 0)
            except (TypeError, ValueError):
                grounding = 0
            if grounding >= MIN_GROUNDING_FOR_BEST:
                best.append(row)
                break
        else:
            rejected += 1
    return best, rejected


# Grade columns are the union of every mode's rubric (grade_columns emits only
# the scoring mode's keys; DictWriter leaves the other modes' cells empty). A
# mode whose keys are missing here fails the write outright, so this must track
# the keys every rubric in ``prompts_un/verifiers`` declares -- which
# ``core.grading.rubric_keys`` reads from the rubric itself. test_rubric_keys
# pins the two together.
FIELDS = ("doc_id", "symbol", "block_id", "block_index", "n_blocks",
          "line_start", "line_end", "stratum",
          "context_blocks_supplied", "context_blocks_dropped",
          "question_language", "mode", "question", "answer",
          "question_type", "framing",
          *gen.annotations.FIELDS,
          "question_cited", "anchor", "anchors",
          "title", "target_block_text",
          "references_supplied", "references_dropped",
          "reference_complete", "n_unresolved", "unresolved_reasons",
          "faith_grounding", "faith_precision", "faith_numerical_fidelity",
          "faith_overall",
          "qual_search_bar_realism", "qual_specificity", "qual_phrasing_economy",
          "qual_focus",
          "qual_search_realism", "qual_lexical_distance",
          "qual_conceptual_framing", "qual_retrievability",
          "qual_practitioner_realism", "qual_anchoring", "qual_informativeness",
          # scored by the revised lookup / practitioners / conceptual rubrics
          "qual_consequence", "qual_anchoring_and_time",
          "qual_linguistic_quality", "qual_overall",
          "faith_reason", "qual_failure_type", "qual_reason", "total_score")


def main(argv: Sequence[str] | None = None, *, index: ctx.BlockIndex | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                        help="target blocks (= queries when keep=1)")
    parser.add_argument("--keep", type=int, default=3,
                        help="candidates written to the all-candidates file")
    # The production configuration: the four question languages the corpus can
    # carry (de is skipped on purpose -- German is not a UN language, so we do
    # not ask in it), all three prompt modes, gpt-5.4-mini generating and
    # Sonnet grading.
    parser.add_argument("--languages", default="en,fr,es,zh")
    parser.add_argument("--modes", default=None,
                        help=f"comma-separated modes (default: {','.join(DEFAULT_MODES)})")
    parser.add_argument("--targets-in", type=Path,
                        help="reuse this exact JSON target list; bypass sampling and ignore --n")
    parser.add_argument("--targets-out", type=Path,
                        help="save the selected target list for identical inputs across models")
    parser.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--decider-model", choices=("generator", "jev"),
                        help="choose a mode with the generator, or balance Jev-eligible personas")
    parser.add_argument("--grade-model", default="google/gemini-3.8-flash")
    parser.add_argument("--generation-cache", type=Path,
                        help="reuse saved responses only for the exact model and canonical messages")
    parser.add_argument("--generation-records", type=Path,
                        help="record generation requests, raw responses, timing and token usage")
    parser.add_argument("--context-chars", type=int, default=ctx.DEFAULT_CONTEXT_CHARS)
    parser.add_argument("--shares", default=None,
                        help="genre shares as resolution,meeting,letter "
                             "(e.g. 0.5,0.4,0.1)")
    parser.add_argument("--all-genres", action="store_true",
                        help="disable the genre filter (uniform pool)")
    parser.add_argument("--no-fit", action="store_true",
                        help="disable the whole-document fit filter; references "
                             "fall back to capped excerpts")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="also sample blocks whose citations are not all resolved")
    parser.add_argument("--max-per-doc", type=int, default=0,
                        help="safety cap on targets per document (0 = uncapped; "
                             "documents contribute proportionally to size)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--blocks", default=None, help="override blocks_en.jsonl path")
    parser.add_argument("--docs", default=None, help="override docs_en.jsonl path")
    parser.add_argument("--out", help="CSV path; defaults to results.csv in a new run folder")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the selected targets and the call budget, make no calls")
    args = parser.parse_args(argv)
    load_env()

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    # A language without a 6-way corpus file (notably German) would silently
    # degrade to an English-only payload: the model would be asked to write in
    # a language it never sees the source in. Refuse instead of degrading.
    unsupported = [l for l in languages if l not in UN_LANGUAGES]
    if unsupported:
        raise SystemExit(
            f"unsupported question language(s) for the UN corpus: {', '.join(unsupported)}. "
            f"The 6-way corpus carries {', '.join(UN_LANGUAGES)} -- a language without a "
            "corpus file has no source text to generate from.")
    if args.decider_model and args.modes is not None:
        parser.error("--decider-model cannot be combined with --modes")
    modes = [x.strip() for x in (args.modes if args.modes is not None else
                                ",".join(DEFAULT_MODES)).split(",") if x.strip()]
    if not modes or any(mode not in SUPPORTED_MODES for mode in modes):
        parser.error(f"unsupported modes; choose from {', '.join(SUPPORTED_MODES)}")

    strata = GENRE_STRATA
    if args.shares:
        values = [float(x) for x in args.shares.split(",")]
        if len(values) != len(GENRE_STRATA):
            raise SystemExit(f"--shares needs {len(GENRE_STRATA)} values "
                             f"({', '.join(n for n, _ in GENRE_STRATA)})")
        strata = tuple(zip((name for name, _ in GENRE_STRATA), values))

    if index is None:
        index = ctx.BlockIndex(blocks_path=args.blocks, docs_path=args.docs)
    if index.incomplete is not None:
        status_mtime = ctx.paths.REFERENCE_STATUS_JSONL.stat().st_mtime
        blocks_file = Path(args.blocks or ctx.paths.BLOCKS_JSONL)
        if blocks_file.exists() and blocks_file.stat().st_mtime > status_mtime:
            print("WARNING: blocks_en.jsonl is newer than reference_status_en.jsonl "
                  "-- re-run clir_bench.domains.legal.un.references_status",
                  file=sys.stderr)
    if args.targets_in:
        targets = load_targets(args.targets_in)
        for target in targets:
            doc = index.docs.get(target.doc_id)
            if doc is None or target.n_blocks != doc["n_blocks"]:
                raise SystemExit(f"manifest target is unavailable or changed: {target.block_id}")
    else:
        targets = select(index, n=args.n, seed=args.seed, languages=languages,
                         modes=modes, strata=strata, max_per_doc=args.max_per_doc,
                         genre_filter=not args.all_genres,
                         fit_filter=not args.no_fit,
                         require_complete=not args.allow_incomplete)
    if args.targets_out:
        write_targets(args.targets_out, targets)

    print(f"selected {len(targets)} target blocks "
          f"across {len({t.doc_id for t in targets})} documents", file=sys.stderr)
    print(f"  by stratum : {dict(Counter(t.stratum for t in targets))}", file=sys.stderr)
    print(f"  by language: {dict(Counter(t.language for t in targets))}", file=sys.stderr)
    print(f"  by mode    : {dict(Counter(t.mode for t in targets))}", file=sys.stderr)
    if index.incomplete is not None:
        print(f"  reference-complete: {sum(1 for t in targets if t.reference_complete)}"
              f"/{len(targets)}; incomplete blocks known to the gate: "
              f"{len(index.incomplete):,}", file=sys.stderr)

    if args.dry_run:
        if args.decider_model:
            print(f"decider: {args.decider_model}; up to {len(targets)} additional calls; "
                  "sampled modes will be replaced by decisions", file=sys.stderr)
        print(f"\ncall budget: {len(targets)} generation + {2 * len(targets)} grading "
              f"= {3 * len(targets)} calls", file=sys.stderr)
        for t in targets[:12]:
            print(f"   {t.block_id:<40} {t.stratum:<8} {t.mode:<9} {t.language}",
                  file=sys.stderr)
        print("   ...", file=sys.stderr)
        return

    from clir_bench.domains.legal.qac import legacy_main
    legacy_main(args, index=index, corpus="un", targets=targets)


def _safe(fn, *a, **kw):
    """A failing block must not abort a 100-block build."""
    try:
        return fn(*a, **kw)
    except Exception as error:  # noqa: BLE001
        print(f"    target failed: {type(error).__name__}: {str(error)[:120]}",
              file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
