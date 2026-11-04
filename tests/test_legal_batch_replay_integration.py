"""Replay goes through the existing legal parsers, graders, ranking and exports."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from clir_bench.core import grading, llm, parallel
from clir_bench.domains.legal.qac import eurlex_batch as eb
from clir_bench.domains.legal.qac import eurlex_context as ec
from clir_bench.domains.legal.qac import eurlex_generate as eg
from clir_bench.domains.legal.qac import un_batch as ub
from clir_bench.domains.legal.qac import un_context as uc
from clir_bench.domains.legal.qac import un_generate as ug
from clir_bench.domains.legal.qac.batch_recording import GenerationRecorder, messages_sha256

GENERATOR = "test/generator"
GRADER = "test/grader"


def response(content, model=GENERATOR):
    return ChatCompletion.model_validate({
        "id": "saved-generation" if model == GENERATOR else "fake-grading",
        "object": "chat.completion", "created": 123, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                  "cost": 0.01},
    })


class Client:
    api_key = "integration-test-placeholder"

    def __init__(self, create):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.options = []

    def with_options(self, **options):
        self.options.append(options)
        return self


def case(corpus):
    if corpus == "eurlex":
        target = eb.Target("eli:test:article4", "32000R0001", "4", 1, "one_ref", "lookup", "en")
        article = ec.ArticleUnit(target.eli_id, target.celex_id, "4",
                                 texts={"en": "Compliance officers must report annually."})
        reference = ec.ArticleUnit("eli:test:article3", target.celex_id, "3",
                                   texts={"en": "Auditors check annual compliance reports."})
        payload = ec.GenerationPayload(article, [reference], [],
                                       ec.render_payload(article, [reference], languages=("en",)))
        candidates = [
            {"question": "Who checks annual compliance reports?", "answer": "Auditors",
             "question_type": "definition_actor_or_procedure", "anchor": "compliance reports",
             "question_cited": "Under Regulation1, who checks annual compliance reports?",
             "instrument_short_name": None, "articles_involved": ["4"]},
            {"question": "How often must compliance officers report?", "answer": "annually",
             "question_type": "monitoring_or_reporting", "anchor": "compliance officers",
             "question_cited": "Under Regulation1, how often must compliance officers report?",
             "instrument_short_name": "null", "articles_involved": ["Article 3"]},
        ]
        return eb, eg, target, payload, candidates
    target = ub.Target("1994/s/res/example", "1994/s/res/example#0", 0, 1,
                       "resolution", "conceptual", "en")
    block = uc.BlockUnit(
        block_id=target.block_id, doc_id=target.doc_id, symbol="S/RES/1(1994)",
        title="Exampleland mission", block_index=0, n_blocks=1, line_start=0, line_end=1,
        token_count=20, texts={"en": "Exampleland received support. Peacekeepers protected civilians."},
        in_range=True,
    )
    payload = uc.GenerationPayload(target=block, context_blocks=[], n_context_dropped=0,
                                   references=[], dropped_references=[], text=uc.render_payload(block, []))
    candidates = [
        {"question": "How did Exampleland benefit from international assistance during 1994?",
         "answer": "Exampleland received support", "framing": "stakeholder", "anchor": "Exampleland"},
        {"question": "How did peacekeepers respond to the Exampleland crisis during 1994?",
         "answer": "Peacekeepers protected civilians", "framing": "response", "anchor": "Exampleland"},
    ]
    return ub, ug, target, payload, candidates


def legacy_prompts(monkeypatch, generator, mode):
    """Replay transport tests deliberately exercise the supported flat rubric."""
    original = generator.PROMPTS
    from clir_bench.domains.legal.qac.evidence import MARKER
    prompt = "Legacy quality rubric\n" + "\n".join(
        f'"{key}": <1-5>' for key in grading.quality_keys(mode))
    monkeypatch.setattr(generator, "PROMPTS", SimpleNamespace(
        generation=lambda *args: original.generation(*args).replace(MARKER + "\n", ""),
        faithfulness=original.faithfulness,
        quality=lambda *args: prompt))


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
def test_cached_run_one_uses_existing_parse_grade_rank_and_rows(tmp_path, monkeypatch, corpus):
    batch, generator, target, payload, candidates = case(corpus)
    legacy_prompts(monkeypatch, generator, target.mode)
    messages = generator.build_messages(payload, target.mode, target.language)
    body = response("```json\n" + json.dumps(candidates) + "\n```").model_dump()
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"model": GENERATOR, "messages_sha256": messages_sha256(messages),
                                 "response": body, "source_request_id": "original-request"}) + "\n")
    recorder = GenerationRecorder(cache, tmp_path / "recorded", GENERATOR)
    generator_client = Client(lambda **kwargs: pytest.fail("cache hit contacted generator transport"))
    calls = []

    def grade_transport(**kwargs):
        calls.append(kwargs)
        assert kwargs["model"] == GRADER
        prompt = kwargs["messages"][0]["content"]
        if prompt == generator.PROMPTS.faithfulness("batch"):
            keys = grading.FAITHFULNESS_KEYS
        else:
            assert prompt == generator.PROMPTS.quality(target.mode, "batch")
            keys = grading.rubric_keys(prompt) or grading.quality_keys(target.mode)
        return response(json.dumps([dict(index=index, **{key: score for key in keys})
                                    for index, score in enumerate((2, 5))]), model=GRADER)

    grader_client = Client(grade_transport)
    monkeypatch.setattr(llm, "client_for", lambda model: generator_client if model == GENERATOR else grader_client)
    parsed_calls = []
    original_parse = generator.parse_candidates

    def observe_parse(*args, **kwargs):
        parsed_calls.append(args[0])
        return original_parse(*args, **kwargs)

    monkeypatch.setattr(generator, "parse_candidates", observe_parse)
    builds = []

    def build(*args, **kwargs):
        builds.append((args, kwargs))
        return payload

    options = {"max_references": 6} if corpus == "eurlex" else {"context_chars": 30000}
    rows = batch.run_one(target, SimpleNamespace(build=build), gen_model=GENERATOR,
                         grade_model=GRADER, keep=3, generation_recorder=recorder, **options)

    assert parsed_calls == [candidates]
    assert len(builds) == 1 and builds[0][1]["languages"] == ("en",)
    assert generator_client.options == [{"max_retries": 0, "timeout": 180}]
    assert grader_client.options == []  # The grader was not wrapped in the generator recorder.
    assert len(calls) == 2
    assert all(call["messages"][1]["content"].startswith(payload.text) for call in calls)
    assert all(candidates[0]["question"] in call["messages"][1]["content"] for call in calls)
    assert [row["question"] for row in rows] == [candidates[1]["question"], candidates[0]["question"]]
    assert rows[0]["faith_grounding"] == 5
    assert rows[0]["total_score"] > rows[1]["total_score"]
    assert all(set(row) <= set(batch.FIELDS) for row in rows)
    if corpus == "eurlex":
        # These transformations prove the normal EUR-Lex parser ran on replay.
        assert rows[0]["articles_involved"] == "4,3"
        assert rows[0]["instrument_short_name"] == ""
        assert rows[0]["target_article_id"] == target.eli_id
        assert payload.target.texts["en"] in rows[0]["target_article_text"]
        assert "Articles involved (declared): 4, 3" in calls[0]["messages"][1]["content"]
    else:
        assert rows[0]["framing"] == "response" and rows[0]["question_type"] == ""
        assert rows[0]["anchor"] == "Exampleland"
        assert payload.target.texts["en"] in rows[0]["target_block_text"]
        best, rejected = ub.pick_best({(target.block_id, target.mode, target.language): rows})
        assert best == rows[:1] and rejected == 0
    records = [json.loads(line) for line in (tmp_path / "recorded" / "generation_attempts.jsonl").open()]
    assert len(records) == 1
    recorded = records[0]
    assert recorded["cached"] and recorded["source_request_id"] == "original-request"
    assert recorded["context"]["corpus"] == corpus
    assert recorded["context"]["target_id"] == (target.eli_id if corpus == "eurlex" else target.block_id)
    assert recorded["context"]["mode"] == target.mode and recorded["context"]["language"] == "en"
    assert recorded["messages_sha256"] == messages_sha256(messages)
    assert recorded["usage"]["provider_cost"] == 0


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
def test_cached_skip_still_uses_parser_and_does_not_grade(tmp_path, monkeypatch, corpus):
    batch, generator, target, payload, _ = case(corpus)
    messages = generator.build_messages(payload, target.mode, target.language)
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"model": GENERATOR, "messages_sha256": messages_sha256(messages),
                                 "response": response('[{"skip_reason":"no qualifying content"}]').model_dump()}) + "\n")
    recorder = GenerationRecorder(cache, tmp_path / "recorded", GENERATOR)
    client = Client(lambda **kwargs: pytest.fail("skip called a live generator or a grader"))
    monkeypatch.setattr(llm, "client_for", lambda _: client)
    options = {"max_references": 6} if corpus == "eurlex" else {"context_chars": 30000}
    assert batch.run_one(target, SimpleNamespace(build=lambda *a, **kw: payload),
                         gen_model=GENERATOR, grade_model=GRADER, keep=3,
                         generation_recorder=recorder, **options) == []


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
def test_existing_cli_delegates_selected_targets_to_shared_workflow(tmp_path, monkeypatch, corpus):
    batch, _, first, _, _ = case(corpus)
    second_mode = "fact_pattern" if corpus == "eurlex" else "descriptive"
    targets = [first, replace(first, mode=second_mode), replace(first, language="fr")]
    manifest, out = tmp_path / "targets.json", tmp_path / f"qac_{corpus}_test_model.csv"
    batch.write_targets(manifest, targets)
    monkeypatch.setattr("sys.argv", ["batch", "--targets-in", str(manifest), "--out", str(out),
                                     "--gen-model", GENERATOR, "--grade-model", GRADER])
    monkeypatch.setattr(batch, "load_env", lambda: None)
    monkeypatch.setattr(llm, "client_for", lambda _: Client(lambda **kw: pytest.fail("unexpected network call")))
    monkeypatch.setattr(parallel, "run_tasks", lambda targets, fn, **kw: map(fn, targets))
    if corpus == "eurlex":
        unit = ec.ArticleUnit(first.eli_id, first.celex_id, first.article_number,
                              texts={"en": "Source", "fr": "Source française"})
        monkeypatch.setattr(batch.ctx, "ArticleIndex", lambda: SimpleNamespace(by_eli={first.eli_id: unit}))
    else:
        index = SimpleNamespace(docs={first.doc_id: {"n_blocks": first.n_blocks}}, incomplete=None,
                                preload_translations=lambda *a: None)
        monkeypatch.setattr(batch.ctx, "BlockIndex", lambda **kw: index)
        monkeypatch.setattr(batch, "_cited_doc_ids", lambda *a: set())

    delegated = []
    from clir_bench.domains.legal import qac
    monkeypatch.setattr(qac, "legacy_main", lambda args, **kwargs: delegated.append((args, kwargs)), raising=False)
    batch.main()
    assert len(delegated) == 1
    args, kwargs = delegated[0]
    assert args.out == str(out) and args.gen_model == GENERATOR and args.grade_model == GRADER
    assert kwargs["corpus"] == corpus and kwargs["targets"] == targets
    assert not out.exists()


class MemoryCheckpoint:
    def __init__(self):
        self.completed = {}
        self.transports = []

    def run(self, stage, fn):
        if stage not in self.completed:
            self.completed[stage] = json.loads(json.dumps(fn()))
        return self.completed[stage]

    def client(self, stage, model, client):
        self.transports.append((stage, model))
        return client


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
def test_same_qa_from_different_generators_shares_both_verifiers_and_survives_resume(tmp_path, monkeypatch, corpus):
    from clir_bench.domains.legal.qac.batch_recording import RunState

    batch, generator, target, payload, raw_candidates = case(corpus)
    candidates = (generator.parse_candidates(raw_candidates, payload, target.mode) if corpus == "eurlex"
                  else generator.parse_candidates(raw_candidates, target.mode))
    monkeypatch.setattr(generator, "generate", lambda *args, **kwargs: candidates)
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if prompt == generator.PROMPTS.faithfulness("batch"):
            output = [dict(index=i, **dict.fromkeys(grading.FAITHFULNESS_KEYS, 5)) for i in range(2)]
        else:
            inputs = json.loads(kwargs["messages"][1]["content"])["candidates"]
            keys = grading.rubric_keys(prompt)
            output = {"candidates": [{"index": i, "candidate_id": c["candidate_id"],
                "scores": dict.fromkeys(keys, 4), "score_notes": dict.fromkeys(keys, "Supported"),
                "checks": dict.fromkeys(("mode", "support", "metadata"), "pass"), "problems": []}
                for i, c in enumerate(inputs)], "batch_diversity": "pass"}
        return response(json.dumps(output), model=GRADER)

    monkeypatch.setattr(llm, "client_for", lambda model: Client(transport))
    options = {"max_references": 6} if corpus == "eurlex" else {"context_chars": 30000}
    outputs = []
    for model in ("generator/a", "generator/b", "generator/c"):
        with RunState(tmp_path / "run.sqlite") as state:
            rows = batch.run_one(target, None, payload=payload, gen_model=model, grade_model=GRADER,
                                 keep=3, checkpoint=state.checkpoint(model), **options)
            outputs.append(rows)
            assert len(rows) == 2 and all(r["grading_status"] == "completed" for r in rows)
            for row in rows:
                assert row["generator_model_id"] == model
                assert json.loads(row["quality_verifier_response_json"])["candidate_id"] == row["candidate_id"]
    assert len(calls) == 2  # One faithfulness and one quality request for all three generators.
    assert len({r["candidate_id"] for rows in outputs for r in rows}) == 6
    assert all(r["total_score"] == outputs[0][0]["total_score"] for rows in outputs for r in rows)
    assert all(r["quality_batch_diversity"] == "not_evaluated" for rows in outputs[1:] for r in rows)


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
def test_checkpoint_keeps_questions_when_faithfulness_fails(tmp_path, monkeypatch, corpus):
    batch, generator, target, payload, raw_candidates = case(corpus)
    candidate = generator.parse_candidates(raw_candidates, payload, target.mode)[0] if corpus == "eurlex" else generator.parse_candidates(raw_candidates, target.mode)[0]
    monkeypatch.setattr(generator, "generate", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(llm, "client_for", lambda _: object())
    monkeypatch.setattr(grading, "grade_faithfulness", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("broken grade")))
    quality_calls = []

    def quality(*args, **kwargs):
        quality_calls.append((args, kwargs))
        keys = grading.rubric_keys(args[2])
        return [dict.fromkeys(keys, 4) | {"overall": 20, "_keys": keys}]

    monkeypatch.setattr(grading, "grade_quality", quality)
    checkpoint = MemoryCheckpoint()
    options = {"max_references": 6} if corpus == "eurlex" else {"context_chars": 30000}
    rows = batch.run_one(target, None, payload=payload, gen_model=GENERATOR,
                         grade_model=GRADER, keep=3, checkpoint=checkpoint, **options)
    assert len(checkpoint.completed["generation"]) == len(rows) == 1
    assert rows[0]["question"] == candidate.question
    assert rows[0]["grading_status"] == "failed" and "broken grade" in rows[0]["grading_error"]
    assert rows[0]["faith_grounding"] is None and rows[0]["total_score"] is None
    assert rows[0]["qual_overall"] == 20 and rows[0]["candidate_rank"] == ""
    assert rows[0]["candidate_id"] and rows[0]["document_text"]
    assert batch.target_from_rows(rows) == target
    args, kwargs = quality_calls[0]
    assert kwargs["strict"] and kwargs["rubric_mode"] == f"{corpus}_{target.mode}"
    assert args[4][0]["candidate_id"] == rows[0]["candidate_id"]
    assert kwargs["sources"][0]["role"] == "target"
    assert kwargs["target_source_id"] == rows[0]["target_id"]


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
def test_regrade_preserves_candidate_identity_and_skips_generator(monkeypatch, corpus):
    batch, generator, target, payload, raw_candidates = case(corpus)
    candidates = generator.parse_candidates(raw_candidates, payload, target.mode) if corpus == "eurlex" else generator.parse_candidates(raw_candidates, target.mode)
    monkeypatch.setattr(generator, "generate", lambda *args, **kwargs: candidates)
    requested_models = []
    monkeypatch.setattr(llm, "client_for", lambda model: requested_models.append(model))
    monkeypatch.setattr(grading, "grade_faithfulness", lambda *args, **kwargs: [{"grounding": 5, "precision": 5, "numerical_fidelity": 5, "overall": 15}] * len(candidates))
    monkeypatch.setattr(grading, "grade_quality", lambda *args, **kwargs: [dict.fromkeys(grading.rubric_keys(args[2]), 4) | {"overall": 20, "_keys": grading.rubric_keys(args[2])}] * len(candidates))
    options = {"max_references": 6} if corpus == "eurlex" else {"context_chars": 30000}
    original = batch.run_one(target, None, payload=payload, gen_model=GENERATOR,
                             grade_model=GRADER, keep=3, checkpoint=MemoryCheckpoint(), **options)
    for row in original:
        row["qual_obsolete_metric"] = 5
        row["quality_audit_status"] = "blocking"
    requested_models.clear()
    monkeypatch.setattr(generator, "generate", lambda *args, **kwargs: pytest.fail("regrade generated new questions"))
    regenerated = batch.run_one(target, None, payload=payload, gen_model=GENERATOR,
                                grade_model=GRADER, keep=3, existing_candidates=original,
                                checkpoint=MemoryCheckpoint(), **options)
    assert GENERATOR not in requested_models
    assert [(r["candidate_id"], r["question"], r["answer"]) for r in regenerated] == [(r["candidate_id"], r["question"], r["answer"]) for r in original]
    assert all(r["qual_obsolete_metric"] == r["quality_audit_status"] == "" for r in regenerated)
    assert all(r["grading_status"] == "completed" for r in regenerated)


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
@pytest.mark.parametrize("count", [1, 3, 7, 15, 31])
def test_stratum_rounding_selects_exact_requested_count(monkeypatch, corpus, count):
    if corpus == "eurlex":
        monkeypatch.setattr(eb, "_quarantined", lambda: set())
        units, references = {}, {}
        for ref_count in (0, 1, 2, 4):
            for position in range(40):
                key = f"eli:{ref_count}:{position}"
                units[key] = ec.ArticleUnit(key, f"act:{ref_count}:{position}", "1", texts={"en": "Text " * 200})
                references[key] = [f"ref:{i}" for i in range(ref_count)]
        index = SimpleNamespace(by_eli=units, references=references,
                                status={key: {"complete": True} for key in units})
        selected = eb.select(index, n=count, seed=42, languages=["en"], modes=["lookup"],
                              include_amending=True)
    else:
        docs = {}
        for genre, stem, title in (("resolution", "2020/s/res", "Resolution"),
                                   ("meeting", "2020/s/pv_", "Meeting"),
                                   ("letter", "2020/s/letter", "Letter dated 1 January")):
            for position in range(40):
                key = f"{stem}/{position}"
                docs[key] = {"doc_id": key, "title": title, "n_blocks": 1,
                             "target_idxs": [0], "char_count": 1000}
        index = SimpleNamespace(docs=docs, incomplete={})
        selected = ub.select(index, n=count, seed=42, languages=["en"], modes=["technical"],
                              fit_filter=False)
    assert len(selected) == count
