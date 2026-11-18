"""Source-bound prompts and the existing QAC parsers used by provider batches."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, fields

from clir_bench.core import corpus, qagen
from clir_bench.core.batch_api import encode
from clir_bench.core.grading import GraderConfig, grade_columns, grade_faithfulness, grade_quality
from clir_bench.core.prompts import PromptPack
from clir_bench.domains.chemistry.qac import decider as patent_decider
from clir_bench.domains.legal.qac import decider, eurlex_batch, un_batch
from clir_bench.domains.legal.qac.evidence import EvidenceSource, uses_contract

LEGAL = {"eurlex": eurlex_batch, "un": un_batch}
LEGAL_UNITS = {
    "eurlex": {
        "target": eurlex_batch.ctx.ArticleUnit, "references": eurlex_batch.ctx.ArticleUnit,
        "external_references": eurlex_batch.ctx.ArticleUnit, "annexes": eurlex_batch.ctx.ArticleUnit,
    },
    "un": {
        "target": un_batch.ctx.BlockUnit, "context_blocks": un_batch.ctx.BlockUnit,
        "references": un_batch.ctx.ReferencedDoc,
    },
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prompt_fingerprint(context, source, languages):
    if source in LEGAL:
        pack = LEGAL[source].gen.PROMPTS
        modes = [decider.generation_mode(mode) for mode in decider.MODES[source]]
        eligibility = decider.eligibility_prompt_text(source)
    else:
        pack = PromptPack(package=context.domain.prompts_package)
        modes = patent_decider.MODES
        eligibility = patent_decider.eligibility_prompt_text()
    prompts = [eligibility, pack.faithfulness("batch")]
    for mode in modes:
        prompts.append(pack.quality(mode, "batch"))
        for language in languages:
            prompt = pack.generation(mode, language)
            if source in LEGAL and not uses_contract(prompt):
                raise ValueError(f"The {source}/{mode}/{language} prompt lacks source-bound evidence")
            prompts.append(prompt)
    return digest(encode(prompts))


def freeze_payload(source, payload, language):
    def unit_data(unit):
        data = asdict(unit)
        for name in ("texts", "headings", "act_titles", "locations", "original_hashes",
                     "original_lengths", "loaded_body_lengths"):
            if name in data:
                data[name] = {key: value for key, value in data[name].items()
                              if key in {language, "en"}}
        return data

    data = {}
    units = LEGAL_UNITS[source]
    for field in fields(payload):
        name = field.name
        value = getattr(payload, name)
        if name == "evidence_sources":
            data[name] = [asdict(entry) for entry in value.values()]
        elif name in units:
            data[name] = [unit_data(unit) for unit in value] if isinstance(value, list) else unit_data(value)
        else:
            data[name] = value
    return data


def thaw_payload(source, data):
    content = dict(data)
    for name, cls in LEGAL_UNITS[source].items():
        value = content[name]
        content[name] = [cls(**unit) for unit in value] if isinstance(value, list) else cls(**value)
    content["evidence_sources"] = {
        (entry["source_ref"], entry["language"]): EvidenceSource(**entry)
        for entry in data["evidence_sources"]
    }
    return LEGAL[source].ctx.GenerationPayload(**content)


def patent_payload(context, row):
    return {
        "text": corpus.build_passages_text([row], context.schema, include_identifiers=True),
        "abstract": row["abstract"], "id": context.schema.id_of(row),
    }


def generation_prompt(source, payload, language, count):
    if source == "gp":
        bodies = [(payload["id"], language, payload["abstract"])]
    else:
        bodies = [
            (item["source_ref"], item["language"], item["text"])
            for item in payload["evidence_sources"]
        ]
    snippets = []
    for ref, lang, body in bodies:
        # Overlap adjacent excerpts so a sentence crossing a boundary stays available.
        for start in range(0, len(body), 1000):
            snippet = body[start:start + 1250]
            if snippet.strip():
                snippets.append({"source_ref": ref, "language": lang, "start": start, "text": snippet})
            if start + 1250 >= len(body):
                break
    if not snippets:
        raise ValueError("Target has no visible evidence")

    def apply(messages):
        system, user = messages
        rule = (
            "\n\nUse the labeled, verbatim evidence excerpts in the user message. Choose the "
            "passages that support the answer before writing the question. Copy evidence from "
            "those passages with the existing source and language annotations."
        )
        if count is not None:
            rule += (
                f" For this run, the response count overrides the earlier count instructions. "
                f"Return exactly {count} distinct question-answer candidates in a single JSON "
                "response, or the existing skip_reason response if the evidence cannot support them."
            )
        return [
            {**system, "content": system["content"] + rule},
            {**user, "content": user["content"] + "\n\nVerbatim evidence excerpts:\n" + encode(snippets)},
        ]

    return apply


def _legal_run(source, target, config, calls, candidates=None):
    module = LEGAL[source]
    limits = ({"max_references": module.ctx.DEFAULT_MAX_REFERENCES} if source == "eurlex"
              else {"context_chars": module.ctx.DEFAULT_CONTEXT_CHARS})
    return module.run_one(
        module.Target(**target["target"]), None, **limits,
        gen_model=target["model"], grade_model=config["verifier_model"], keep=3,
        payload=thaw_payload(source, target["payload"]), batch_calls=calls,
        generation_only=candidates is None, existing_candidates=candidates,
    )


def generate(context, config, target, calls):
    source = config["source"]
    if source in LEGAL:
        rows = _legal_run(source, target, config, calls)
    else:
        prompts = PromptPack(package=context.domain.prompts_package)
        parser = context.domain.qac_candidate_parser
        language, mode = target["language"], target["mode"]
        candidates = qagen.generate_candidates(
            calls.client("generation", target["model"]),
            qagen.GenerationConfig(
                generation_model=target["model"], grader=GraderConfig(config["verifier_model"]),
                generation_reasoning_effort=config["generation_reasoning_effort"],
                candidate_parser=parser,
            ),
            prompts.generation(mode, language), target["payload"]["text"], mode,
        )
        rows = _patent_evidence(target, candidates)
    if not rows:
        return []
    count = config["candidates_per_target"]
    questions = {" ".join(row["question"].casefold().split()) for row in rows}
    if count is not None and (len(rows) != count or len(questions) != count):
        raise ValueError(f"Expected {count} distinct candidates from a single request")
    return rows


def _patent_evidence(target, candidates):
    abstract = target["payload"]["abstract"]
    language = target["language"]
    rows = []
    for position, candidate in enumerate(candidates):
        quote = candidate["evidence"]
        start = abstract.find(quote)
        if start < 0 or candidate["evidence_language"] != language:
            continue
        span = {
            "source_id": target["payload"]["id"], "source_ref": "abstract",
            "language": language, "text": quote, "start": start, "end": start + len(quote),
            "source_text_sha256": digest(abstract),
        }
        candidate_id = "q_" + digest(encode([target["key"], target["model"], language,
                                             target["mode"], position, candidate["question"]]))[:24]
        rows.append(dict(candidate, candidate_id=candidate_id, evidence_spans=[span]))
    if candidates and not rows:
        raise ValueError("Candidates supplied no exact evidence in the selected patent abstract")
    return rows


def grade(context, config, target, candidates, calls):
    source = config["source"]
    if source in LEGAL:
        return _legal_run(source, target, config, calls, candidates)
    prompts = PromptPack(package=context.domain.prompts_package)
    model = config["verifier_model"]
    grader = GraderConfig(model=model, reasoning_effort=config["grading_reasoning_effort"])
    mode, language = target["mode"], target["language"]
    qa = [dict(row, mode=mode, question_language=language) for row in candidates]
    passages = target["payload"]["text"]

    grades = {}
    for stage, function in {
        "faithfulness": lambda: grade_faithfulness(
            calls.client("faithfulness", model), grader, prompts.faithfulness("batch"), passages,
            qa, strict=True,
        ),
        "quality": lambda: grade_quality(
            calls.client("quality", model), grader, prompts.quality(mode, "batch"), passages,
            qa, mode, strict=True,
        ),
    }.items():
        try:
            grades[stage] = calls.run(stage, function)
        except (ValueError, RuntimeError, TypeError, KeyError, IndexError):
            continue
    if len(grades) != 2:
        return []
    rows = []
    for candidate, faith, quality in zip(qa, grades["faithfulness"], grades["quality"]):
        rows.append(dict(
            candidate, **grade_columns(faith, quality, mode),
            corpus="gp", publication_number=target["parent"], document_id=target["parent"],
            target_id=target["key"], source_document_id=target["payload"]["id"],
            corpus_id=target["payload"]["id"], context_language=language, context_languages=language,
            generator_model_id=target["model"], generator_model_name=target["model"].rsplit("/", 1)[-1],
            faithfulness_verifier_model=model, quality_verifier_model=model,
            grading_status="completed",
        ))
    return rows


def passing(row, source):
    if (row.get("grading_status") != "completed" or row.get("total_score") is None
            or not row.get("evidence_spans")):
        return False
    if row.get("quality_audit_status") not in {"clear", "minor_issues"}:
        return False
    if row.get("quality_support_status") not in (None, "", "pass"):
        return False
    return source != "un" or row["faith_grounding"] >= un_batch.MIN_GROUNDING_FOR_BEST
