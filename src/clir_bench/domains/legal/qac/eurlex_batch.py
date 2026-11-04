"""
Build a EUR-Lex question set: select target articles, generate, grade, rank, write.

One target article yields three candidates; the best-scoring one is kept. So a
100-query set is 100 target articles, and the run costs 100 generation calls plus
200 grading calls.

Selection is stratified by **reference count** -- same-act articles, other
acts' articles and annexes together, since all of them travel with the target
-- and records the
stratum on every row so the resulting set can be re-weighted or split later.
Measured on the reference-complete eligible pool (16.9k articles): 36% cite no
other article and 64% cite at least one (one 21%, two-three 23%, four-plus 19%).
The default mix therefore *under*-samples citing articles (``CROSS_REFERENCE_SHARE``
below): the no-reference control must dominate, and multi-article gold is meant
to be a measured minority rather than the norm.

Filters applied before sampling, each for a reason found the hard way:

* acts in ``quarantine.jsonl`` are excluded -- their article numbering disagrees
  across languages, so an article id there is not reliably the same article in
  all four;
* amending articles are excluded by default -- they quote the text of *another*
  act, so a question drawn from one is about a document that is not in the corpus;
* very short and very long articles are excluded -- the first cannot support
  three distinct questions, the second buries the operative fact;
* articles whose citation graph is not **reference-complete** are excluded by
  default (``reference_status.jsonl`` from ``structure.resolve_external``): if
  any article the target cites could not be resolved -- another act outside the
  corpus, the Treaty, "of that Regulation" -- the generator would be writing
  about text it only half saw. ``--allow-incomplete`` lifts this.

Usage:
    python -m clir_bench.domains.legal.qac.eurlex_batch --n 100 --dry-run
    python -m clir_bench.domains.legal.qac.eurlex_batch --n 100 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clir_bench.domains.legal.qac import eurlex_context as ctx
from clir_bench.domains.legal.qac import eurlex_generate as gen
from clir_bench.domains.legal.qac.env import load_env
from clir_bench.domains.legal.structure import ACT_LANGUAGES
from clir_bench.domains.legal.structure import paths as struct_paths

DEFAULT_GEN_MODEL = "gpt-5.6-luna"

OUT_DIR = struct_paths.EURLEX_DIR / "qac"

# (name, min refs, max refs, share of the set). Only this share of the set is
# drawn from articles that cite another article (same act or another act in the
# corpus); the rest have no references at all and act as the control that must
# never produce a multi-article answer. Cross-referenced questions are the
# interesting minority, not the norm: 64% of the eligible pool cites something,
# so 0.20 is a deliberate under-sampling that keeps multi-article gold a measured
# minority. Raise it with --cross-ref-share when the reference graph is the point.
CROSS_REFERENCE_SHARE = 0.20

DEFAULT_STRATA: tuple[tuple[str, int, int, float], ...] = (
    ("no_refs", 0, 0, 1 - CROSS_REFERENCE_SHARE),
    ("one_ref", 1, 1, CROSS_REFERENCE_SHARE * 0.40),
    ("few_refs", 2, 3, CROSS_REFERENCE_SHARE * 0.35),
    ("many_refs", 4, 99, CROSS_REFERENCE_SHARE * 0.25),
)

MIN_CHARS, MAX_CHARS = 600, 9000


@dataclass
class Target:
    eli_id: str
    celex_id: str
    article_number: str
    n_refs: int            # same-act references
    stratum: str
    mode: str
    language: str
    n_external: int = 0    # resolved references to articles of other acts
    n_annex: int = 0       # resolved annex references (this act or another)
    complete: bool = True  # every citation resolved (reference_status)
    cites_annex: bool = False


def load_targets(path: Path) -> list[Target]:
    """Read an ordered, fixed list of target dictionaries for reuse across models."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("target manifest must be a nonempty JSON list")
    targets, seen = [], set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise TypeError(f"target manifest entry {number} must be an object")
        try:
            target = Target(**row)
        except TypeError as error:
            raise ValueError(f"invalid target manifest entry {number}: {error}") from error
        if any(not isinstance(value, str) or not value.strip()
               for value in (target.eli_id, target.celex_id, target.article_number)):
            raise ValueError(f"target manifest entry {number} needs nonempty article identifiers")
        if target.mode not in gen.MODES:
            raise ValueError(f"unsupported EUR-Lex mode in target manifest: {target.mode}")
        if target.language not in ACT_LANGUAGES:
            raise ValueError(f"unsupported EUR-Lex language in target manifest: {target.language}")
        identity = (target.eli_id, target.mode, target.language)
        if identity in seen:
            raise ValueError(f"duplicate target in manifest: {identity}")
        seen.add(identity)
        targets.append(target)
    return targets


def write_targets(path: Path, targets: Sequence[Target]) -> None:
    """Persist target order, modes and languages without resampling on later runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(target) for target in targets],
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def target_from_rows(rows: Sequence[dict[str, Any]]) -> Target:
    """Recover the original selection metadata without sampling a new article."""
    row = rows[0]
    return Target(
        eli_id=row.get("target_article_id") or row["target_id"],
        celex_id=row.get("celex_id") or row["document_id"],
        article_number=str(row["target_article_number"]),
        n_refs=int(row.get("n_references_available") or 0),
        stratum=row.get("stratum", ""), mode=row["mode"],
        language=row["question_language"],
        n_external=int(row.get("n_external_available") or 0),
        n_annex=int(row.get("n_annex_available") or 0),
        complete=str(row.get("reference_complete", "")).lower() == "true",
        cites_annex=str(row.get("cites_annex", "")).lower() == "true",
    )


def prepare_payload(target: Target, index: ctx.ArticleIndex, *,
                    max_references: int = ctx.DEFAULT_MAX_REFERENCES):
    return index.build(target.eli_id, max_references=max_references,
                       languages=ctx.payload_languages(target.language))


def quality_sources(payload: ctx.GenerationPayload, language: str, *,
                    evidence_contract: bool = False) -> list[dict[str, Any]]:
    """Name each supplied source, preserving the generator's reference truncation."""
    sources = []
    for unit, role, keyed in (
            [(payload.target, "target", False)]
            + [(unit, "reference", False) for unit in payload.references]
            + [(unit, "reference", True) for unit in payload.external_references]
            + [(unit, "reference", True) for unit in payload.annexes]):
        languages = [lg for lg in ctx.payload_languages(language) if unit.texts.get(lg)]
        source = {
            "source_id": unit.eli_id, "role": role,
            "text": "\n\n".join(ctx._block(
                unit, lg, limit=None if role == "target" else ctx.DEFAULT_REFERENCE_CHARS,
                key=ctx.external_key(unit) if keyed else None) for lg in languages),
            "metadata": {"celex_id": unit.celex_id, "article_number": unit.article_number,
                         "unit_type": unit.unit_type, "titles": unit.act_titles,
                         "languages": languages},
        }
        if evidence_contract and getattr(payload, "evidence_sources", None):
            visible = [body for body in payload.evidence_sources.values()
                       if body.source_id == unit.eli_id]
            if not visible:
                continue
            source["texts"] = {body.language: body.text for body in visible}
            source["text"] = "\n\n".join(
                f"[{body.language.upper()}]\n{body.text}" for body in visible)
            source["metadata"].update(
                languages=list(source["texts"]), source_ref=visible[0].source_ref)
        sources.append(source)
    return sources


def _quarantined(path: Path | None = None) -> set[str]:
    path = path or struct_paths.QUARANTINE_JSONL
    if not path.exists():
        return set()
    return {json.loads(line)["celex_id"] for line in path.open(encoding="utf-8")}


def select(index: ctx.ArticleIndex, *, n: int, seed: int, languages: Sequence[str],
           modes: Sequence[str], strata=DEFAULT_STRATA,
           include_amending: bool = False,
           max_per_act: int = 2,
           require_complete: bool = True, excluded_documents: frozenset[str] = frozenset()) -> list[Target]:
    """Stratified, deterministic target selection.

    ``require_complete`` restricts the pool to articles whose citations are all
    resolved (see module docstring); it needs ``reference_status.jsonl`` and
    refuses to silently sample everything when that file is missing.
    """
    quarantine = getattr(index, "quarantine_path", None)
    bad_acts = _quarantined(path=quarantine) if quarantine is not None else _quarantined()
    status = getattr(index, "status", {}) or {}
    if require_complete and not status:
        raise SystemExit(
            "reference_status.jsonl not loaded: run "
            "`python -m clir_bench.domains.legal.structure.resolve_external` "
            "or pass --allow-incomplete")
    external_refs = getattr(index, "external_references", {}) or {}
    annex_refs = getattr(index, "annex_references", {}) or {}
    amending: set[str] = set()
    if not include_amending:
        articles_path = getattr(index, "articles_path", struct_paths.ARTICLES_JSONL)
        with articles_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if (row["unit_type"] == "article" and row["language"] == "en"
                        and row.get("is_amending") and row.get("eli_id")):
                    amending.add(row["eli_id"])

    # A question language the corpus has no version of (zh) is legitimate --
    # the chemistry pipeline asks in languages the document does not exist in
    # by design, and the payload then carries English alone. What a target must
    # actually have is every REQUESTED language the corpus can supply, so any
    # article can serve any assigned language.
    needed = [lg for lg in languages if lg in ctx.ACT_LANGUAGES]

    pools: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for eli, unit in index.by_eli.items():
        # The index also holds cited ANNEX bodies; a question target is always
        # an article.
        if unit.unit_type != "article":
            continue
        if unit.celex_id in bad_acts or unit.celex_id in excluded_documents or eli in amending:
            continue
        text = unit.texts.get("en", "")
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue
        if any(not unit.texts.get(lg) for lg in needed):
            continue
        verdict = status.get(eli, {})
        if require_complete and not verdict.get("complete"):
            continue
        internal = len([t for t in index.references.get(eli, []) if t != eli])
        external = len(external_refs.get(eli, []))
        annexes = len(annex_refs.get(eli, []))
        refs = internal + external + annexes
        for name, low, high, _ in strata:
            if low <= refs <= high:
                # Without a status file nothing is verified complete, so the
                # flag is False rather than assumed -- the row says what we know.
                pools[name].append((eli, internal, external, annexes,
                                    bool(verdict.get("complete")),
                                    bool(verdict.get("cites_annex"))))
                break

    def rank(eli: str) -> str:
        return hashlib.sha256(f"{seed}:{eli}".encode()).hexdigest()

    chosen: list[Target] = []
    per_act: Counter = Counter()
    quotas = [int(n * share) for _, _, _, share in strata]
    remainder_order = sorted(range(len(strata)),
                             key=lambda i: n * strata[i][-1] - quotas[i], reverse=True)
    for position in remainder_order[:n - sum(quotas)]:
        quotas[position] += 1
    for (name, _, _, _), want in zip(strata, quotas):
        pool = sorted(pools.get(name, []), key=lambda x: rank(x[0]))
        taken = 0
        for eli, internal, external, annexes, complete, cites_annex in pool:
            if taken >= want:
                break
            unit = index.by_eli[eli]
            if per_act[unit.celex_id] >= max_per_act:
                continue
            per_act[unit.celex_id] += 1
            taken += 1
            position = len(chosen)
            chosen.append(Target(
                eli_id=eli, celex_id=unit.celex_id,
                article_number=unit.article_number, n_refs=internal, stratum=name,
                # Alternate deterministically so the set is balanced across
                # modes and question languages rather than randomly lumpy.
                mode=modes[position % len(modes)],
                language=languages[position % len(languages)],
                n_external=external, n_annex=annexes,
                complete=complete, cites_annex=cites_annex,
            ))
    return chosen


def run_one(target: Target, index: ctx.ArticleIndex, *, gen_model: str,
            grade_model: str, max_references: int, keep: int,
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

    enhanced = checkpoint is not None or existing_candidates is not None
    payload = payload or prepare_payload(target, index, max_references=max_references)
    if payload is None:
        return []
    # EUR-Lex quality rubrics require source-based checks: each runs a multi-step
    # grading procedure and returns, per candidate, an expert rewrite plus a
    # prose flaw note for every one of the five criteria. That is far more
    # output than the score-only rubrics the defaults were sized for -- 12k
    # total minus an 8k thinking budget leaves ~4k, and the grader came back
    # empty ("Expecting value: line 1 column 1") or truncated mid-JSON
    # ("Unterminated string") on 9 of 30 targets. Give the thinking and the
    # answer room; the faithfulness rubric is unaffected either way.
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
                context={"corpus": "eurlex", "target_id": target.eli_id,
                         "mode": target.mode, "language": target.language})
        return [asdict(candidate) for candidate in gen.generate(
            payload, mode=target.mode, language=target.language, model=gen_model,
            client=gen_client)]

    if existing_candidates is None:
        saved = stage("generation", generate_candidates)
        candidates = [gen.Candidate(**item) for item in saved]
    else:
        candidates = [gen.Candidate(
            **gen.annotations.restore_metadata(row, target.mode, eurlex=True),
            question=row["question"], answer=row["answer"],
            classification=row.get("framing" if target.mode == gen.MODE_CONCEPTUAL else "question_type", ""),
            articles_involved=[value for value in row.get("articles_involved", "").split(",") if value],
            involved_elis=[value for value in row.get("articles_involved_eli", "").split(",") if value],
            rejected_involved=[value for value in row.get("rejected_involved", "").split(",") if value],
            multi_article=str(row.get("multi_article", "")).lower() == "true",
            cross_act=str(row.get("cross_act", "")).lower() == "true",
            question_cited=row.get("question_cited", ""),
            instrument_short_name=row.get("instrument_short_name", ""),
            anchor=row.get("anchor", ""),
            particulars=[value for value in row.get("particulars", "").split(gen.PARTICULAR_SEP) if value],
        ) for row in existing_candidates]
    if not candidates:
        return []

    if generation_only:
        # Keep only the fields needed to reconstruct the candidate for the
        # verifier phase. The generation stage itself stores the complete
        # Candidate objects in run.sqlite, so a resumed run reuses these rows.
        result = []
        for position, candidate in enumerate(candidates):
            candidate_id = "q_" + hashlib.sha256(json.dumps(
                ["eurlex", target.eli_id, target.mode, target.language, gen_model,
                 position, candidate.question, candidate.answer],
                ensure_ascii=False).encode()).hexdigest()[:24]
            result.append({
                "candidate_id": candidate_id,
                "question": candidate.question,
                "answer": candidate.answer,
                "question_type": candidate.classification if target.mode != gen.MODE_CONCEPTUAL else "",
                "framing": candidate.classification if target.mode == gen.MODE_CONCEPTUAL else "",
                **gen.annotations.row_metadata(candidate),
                "articles_involved": ",".join(candidate.articles_involved),
                "articles_involved_eli": ",".join(candidate.involved_elis),
                "rejected_involved": ",".join(candidate.rejected_involved),
                "multi_article": candidate.multi_article,
                "cross_act": candidate.cross_act,
                "question_cited": candidate.question_cited,
                "instrument_short_name": candidate.instrument_short_name,
                "anchor": candidate.anchor,
                "particulars": gen.PARTICULAR_SEP.join(candidate.particulars),
            })
        return result

    # The graders see the declaration too: the faithfulness rubric caps the
    # grade when ``articles_involved`` is wrong, which it can only judge if shown.
    qa = [{"question": c.question, "answer": c.answer, "_candidate_index": position,
           "candidate_id": ((existing_candidates[position].get("candidate_id") if existing_candidates else None)
                            or "q_" + hashlib.sha256(json.dumps(
                                ["eurlex", target.eli_id, target.mode, target.language, gen_model,
                                 position, c.question, c.answer], ensure_ascii=False).encode()).hexdigest()[:24]),
           "question_language": target.language,
           "articles_involved": list(c.articles_involved)} for position, c in enumerate(candidates)]
    evidence_contract = any(getattr(c, "evidence_contract", "")
                            or getattr(c, "evidence_spans", []) for c in candidates)
    for pair, candidate in zip(qa, candidates):
        if target.mode in gen.NEW_MODES or evidence_contract:
            pair.update(mode=target.mode, **gen.annotations.metadata(candidate))
    # The quality rubrics run consistency checks ON the mode's own fields -- is
    # the declared anchor actually present in the question, is the short name
    # invented, does the cited rendering differ from the base one by nothing but
    # the identifier, do the particulars appear in the situation -- none of which
    # is checkable unless the grader is shown them. Faithfulness keeps the lean
    # block: it grades the answer against the text and the extra fields are noise.
    qa_quality = [
        dict({key: value for key, value in pair.items() if not key.startswith("_")}, **{key: value for key, value in (
            ("question_cited", c.question_cited),
            ("instrument_short_name", c.instrument_short_name),
            ("anchor", c.anchor),
            ("particulars", list(c.particulars)),
            ("framing" if target.mode == gen.MODE_CONCEPTUAL else "question_type", c.classification),
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
            target_source_id=target.eli_id, rubric_mode=f"eurlex_{target.mode}",
            policies={"target_granularity": "article", "reference_policy": "target_plus_supplied_references",
                      "answer_role": ("source_identity" if target.mode == gen.MODE_SOURCE_FINDING
                                      else "evidence_span"), "source_time_policy": "supplied_version",
                      "require_unique_gold": False, "fact_pattern_voice": "third_person_client",
                      "fact_pattern_clients": "private_clients"},
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

    # Ranked best-first. Row order carries the ranking, so no rank column is
    # needed: the caller writes the whole list to the all-candidates file and the
    # first row of each target to the best-only file.
    ranked = rank_candidates(qa, faith, quality, target.mode)
    rows: list[dict[str, Any]] = []
    for rank, graded in enumerate(ranked[:keep], 1):
        position = graded.qa["_candidate_index"]
        candidate = candidates[position]
        row = {
            "celex_id": target.celex_id,
            "target_article_id": payload.target.eli_id,
            "target_article_number": target.article_number,
            "stratum": target.stratum,
            "n_references_available": target.n_refs,
            "reference_articles_supplied": ",".join(r.article_number for r in payload.references),
            "reference_articles_dropped": ",".join(payload.dropped_references),
            "n_external_available": target.n_external,
            "external_references_supplied": ",".join(
                ctx.external_key(u) for u in payload.external_references),
            "external_references_dropped": ",".join(payload.dropped_external_references),
            "n_annex_available": target.n_annex,
            "annex_references_supplied": ",".join(
                ctx.external_key(u) for u in payload.annexes),
            "annex_references_dropped": ",".join(payload.dropped_annex_references),
            "reference_complete": target.complete,
            "cites_annex": target.cites_annex,
            "question_language": target.language,
            "mode": target.mode,
            "question": candidate.question,
            "answer": candidate.answer,
            **gen.annotations.row_metadata(candidate),
            "question_type": candidate.classification if target.mode != gen.MODE_CONCEPTUAL else "",
            "framing": candidate.classification if target.mode == gen.MODE_CONCEPTUAL else "",
            # Mode-specific columns, empty in the mode that does not emit them.
            "question_cited": candidate.question_cited,
            "instrument_short_name": candidate.instrument_short_name,
            "anchor": candidate.anchor,
            "particulars": gen.PARTICULAR_SEP.join(candidate.particulars),
            "articles_involved": ",".join(candidate.articles_involved),
            "articles_involved_eli": ",".join(candidate.involved_elis),
            # The source text this question was written from (see rows_for).
            "target_article_text": ctx.unit_source(payload.target, target.language),
            "referenced_articles_text": ctx.referenced_sources(
                candidate.articles_involved, payload, target.language),
            "multi_article": candidate.multi_article,
            "cross_act": candidate.cross_act,
            "rejected_involved": ",".join(candidate.rejected_involved),
            "faith_grounding": graded.faith.get("grounding"),
            "faith_precision": graded.faith.get("precision"),
            "faith_numerical_fidelity": graded.faith.get("numerical_fidelity"),
            "qual_overall": graded.quality.get("overall"),
            "total_score": graded.total,
        }
        if enhanced:
            if existing_candidates is not None:
                original = existing_candidates[position]
                row = dict(original, **row)
                # Inactive old-rubric metrics and audit fields must not survive a regrade.
                for key in row:
                    if key.startswith(("faith_", "qual_", "quality_", "faithfulness_verifier_")):
                        row[key] = ""
            row.update(grade_columns(graded.faith, graded.quality, target.mode))
            row.update({
                "corpus": "eurlex", "document_id": target.celex_id, "target_id": target.eli_id,
                "document_text_scope": "target_article", "document_text": row["target_article_text"],
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
        rows.append(row)
    return rows


# ``n_references_available`` counts same-act references only (as it always
# did); ``stratum`` is binned on same-act + other-act references together.
FIELDS = ("celex_id", "target_article_id", "target_article_number", "stratum",
          "n_references_available", "reference_articles_supplied",
          "reference_articles_dropped", "n_external_available",
          "external_references_supplied", "external_references_dropped",
          "n_annex_available", "annex_references_supplied",
          "annex_references_dropped",
          "reference_complete", "cites_annex", "question_language", "mode",
          "question", "answer", "question_type", "framing",
          # ``lookup`` fills question_cited/instrument_short_name/anchor;
          # ``fact_pattern`` fills particulars. The unused ones stay empty.
          *gen.annotations.FIELDS,
          "question_cited", "instrument_short_name", "anchor", "particulars",
          "articles_involved", "articles_involved_eli",
          "target_article_text", "referenced_articles_text", "multi_article",
          "cross_act", "rejected_involved", "faith_grounding", "faith_precision",
          "faith_numerical_fidelity", "qual_overall", "total_score")


def main(argv: Sequence[str] | None = None, *, index: ctx.ArticleIndex | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                        help="target articles (= queries when keep=1); ignored with --targets-in")
    parser.add_argument("--keep", type=int, default=3,
                        help="candidates written to the all-candidates file")
    # The production configuration: all four corpus languages (zh is skipped
    # on purpose -- no EUR-Lex zh versions exist, so we do not ask in it), all three
    # prompt modes, gpt-5.4-mini generating and Sonnet grading.
    parser.add_argument("--languages", default="en,fr,de,es")
    parser.add_argument("--modes", default=None, help="comma-separated modes (default: lookup,fact_pattern,conceptual)")
    parser.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--decider-model", choices=("generator", "jev"),
                        help="choose a mode with the generator, or balance Jev-eligible personas")
    parser.add_argument("--grade-model", default="google/gemini-3.8-flash")
    parser.add_argument("--generation-cache", type=Path,
                        help="reuse saved responses only for the exact model and canonical messages")
    parser.add_argument("--generation-records", type=Path,
                        help="record generation requests, raw responses, timing and token usage")
    parser.add_argument("--max-references", type=int, default=ctx.DEFAULT_MAX_REFERENCES)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cross-ref-share", type=float, default=CROSS_REFERENCE_SHARE,
                        help="share of targets drawn from articles that cite another article")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--include-amending", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="also sample articles whose citations are not all resolved")
    parser.add_argument("--out", help="CSV path; defaults to results.csv in a new run folder")
    parser.add_argument("--targets-in", type=Path,
                        help="reuse a JSON list of targets; its order, modes, languages and length are authoritative")
    parser.add_argument("--targets-out", type=Path,
                        help="save the exact target list before generation or a dry run")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the selected targets and the call budget, make no calls")
    args = parser.parse_args(argv)
    load_env()

    languages = [x.strip() for x in args.languages.split(",") if x.strip()]
    # Acts are parsed in ACT_LANGUAGES only; any other question language would
    # get an English-only payload and be written from a language the generator
    # never sees the source in. Refuse instead of degrading.
    unsupported = [l for l in languages if l not in ACT_LANGUAGES]
    if unsupported:
        raise SystemExit(
            f"unsupported question language(s) for EUR-Lex: {', '.join(unsupported)}. "
            f"Acts are available in {', '.join(ACT_LANGUAGES)}.")
    if args.decider_model and args.modes is not None:
        parser.error("--decider-model cannot be combined with --modes")
    modes = [x.strip() for x in (args.modes if args.modes is not None else
                                ",".join(gen.DEFAULT_MODES)).split(",") if x.strip()]
    if not languages or not modes:
        parser.error("--languages and --modes must each contain at least one value")
    unsupported_modes = [mode for mode in modes if mode not in gen.MODES]
    if unsupported_modes:
        parser.error(f"unsupported EUR-Lex mode(s): {', '.join(unsupported_modes)}")

    share = args.cross_ref_share
    strata = (("no_refs", 0, 0, 1 - share), ("one_ref", 1, 1, share * 0.40),
              ("few_refs", 2, 3, share * 0.35), ("many_refs", 4, 99, share * 0.25))

    if index is None:
        index = ctx.ArticleIndex()
    if args.targets_in:
        try:
            targets = load_targets(args.targets_in)
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"cannot load target manifest: {error}")
        for target in targets:
            unit = index.by_eli.get(target.eli_id)
            if unit is None or unit.unit_type != "article":
                parser.error(f"unknown target article in manifest: {target.eli_id}")
            if (unit.celex_id, unit.article_number) != (target.celex_id, target.article_number):
                parser.error(f"inconsistent article identifiers in manifest: {target.eli_id}")
            if not unit.texts.get(target.language):
                parser.error(f"target article has no {target.language} source: {target.eli_id}")
    else:
        targets = select(index, n=args.n, seed=args.seed, languages=languages,
                         modes=modes, strata=strata,
                         include_amending=args.include_amending,
                         require_complete=not args.allow_incomplete)
    if args.targets_out:
        write_targets(args.targets_out, targets)

    print(f"selected {len(targets)} target articles "
          f"across {len({t.celex_id for t in targets})} acts", file=sys.stderr)
    print(f"  by stratum : {dict(Counter(t.stratum for t in targets))}", file=sys.stderr)
    print(f"  by language: {dict(Counter(t.language for t in targets))}", file=sys.stderr)
    print(f"  by mode    : {dict(Counter(t.mode for t in targets))}", file=sys.stderr)
    print(f"  with other-act refs: {sum(1 for t in targets if t.n_external)}; "
          f"with annex refs: {sum(1 for t in targets if t.n_annex)}; "
          f"reference-complete: {sum(1 for t in targets if t.complete)}", file=sys.stderr)

    if args.dry_run:
        if args.decider_model:
            print(f"decider: {args.decider_model}; up to {len(targets)} additional calls; "
                  "sampled modes will be replaced by decisions", file=sys.stderr)
        print(f"\ncall budget: {len(targets)} generation + {2 * len(targets)} grading "
              f"= {3 * len(targets)} calls", file=sys.stderr)
        for t in targets[:12]:
            print(f"   {t.celex_id} art {t.article_number:<5} refs={t.n_refs:<3} "
                  f"ext={t.n_external:<3} anx={t.n_annex:<3} {t.stratum:<10} "
                  f"{t.mode:<9} {t.language}", file=sys.stderr)
        print("   ...", file=sys.stderr)
        return

    from clir_bench.domains.legal.qac import legacy_main
    legacy_main(args, index=index, corpus="eurlex", targets=targets)


def _safe(fn, *a, **kw):
    """A failing article must not abort a 100-article build."""
    try:
        return fn(*a, **kw)
    except Exception as error:  # noqa: BLE001
        print(f"    target failed: {type(error).__name__}: {str(error)[:120]}", file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
