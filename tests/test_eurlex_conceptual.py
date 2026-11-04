"""Offline EUR-Lex conceptual generation, grading, provenance and replay."""

import csv
import io
import json

import pytest

from clir_bench.core import grading, llm
from clir_bench.domains.legal.qac import decider
from clir_bench.domains.legal.qac import eurlex_batch as batch
from clir_bench.domains.legal.qac import eurlex_context as ctx
from clir_bench.domains.legal.qac import eurlex_generate as gen
from test_legal_batch_replay_integration import Client, MemoryCheckpoint, response

QUALITY_KEYS = {
    "search_realism", "anchoring_and_time", "consequence", "lexical_distance",
    "linguistic_quality",
}
GENERATOR, GRADER = "gpt-test-generator", "gpt-test-grader"


@pytest.fixture(autouse=True)
def local_prompts(monkeypatch):
    """Exercise the new source pack independently of workspace activation."""
    monkeypatch.delenv("CLIR_PROMPT_MANIFEST", raising=False)
    monkeypatch.setenv("CLIR_PROMPT_SOURCE", "local")


@pytest.fixture
def conceptual_case():
    target = batch.Target("eli:test:article4", "32000R0001", "4", 0,
                          "no_refs", "conceptual", "en")
    article = ctx.ArticleUnit(
        target.eli_id, target.celex_id, "4",
        texts={"en": "UCITS management companies shall establish an independent compliance "
                     "function to identify failures to meet their obligations. Investors shall "
                     "receive information about the risks of the proposed investment."},
    )
    payload = ctx.GenerationPayload(
        article, [], [], ctx.render_payload(article, []),
        evidence_sources=ctx.visible_evidence_sources(article, [], languages=("en",)),
    )
    candidates = [
        {"question": "How do UCITS management companies detect failures to meet their obligations?",
         "answer": "UCITS management companies shall establish an independent compliance "
                   "function to identify failures to meet their obligations.",
         "framing": "response", "anchor": "UCITS management companies",
         "articles_involved": ["4"]},
        {"question": "How are investors informed about the risks of a proposed UCITS investment?",
         "answer": "Investors shall receive information about the risks of the proposed investment.",
         "framing": "stakeholder", "anchor": "proposed UCITS investment",
         "articles_involved": ["4"]},
    ]
    for item in candidates:
        item["answer_is_translation"] = False
        item["evidence_spans"] = [{"source_ref": "4", "language": "en", "text": item["answer"]}]
    return target, payload, candidates


def test_conceptual_metadata_survives_parsing_without_other_mode_fields(conceptual_case):
    _, payload, candidates = conceptual_case
    assert gen.MODE_CONCEPTUAL in gen.MODES
    raw = dict(candidates[0], question_type="obligation_or_prohibition",
               question_cited="Under Regulation 1/2000, how does compliance work?",
               instrument_short_name="Invented name", particulars=["decorative client"])
    candidate, = gen.parse_candidates([raw], payload, gen.MODE_CONCEPTUAL)
    assert candidate.classification == raw["framing"]
    assert candidate.anchor == raw["anchor"]
    assert candidate.question_cited == candidate.instrument_short_name == ""
    assert candidate.particulars == []
    row, = gen.rows_for(payload, [candidate], mode=gen.MODE_CONCEPTUAL, language="en")
    assert row["framing"] == "response" and row["question_type"] == ""
    assert row["anchor"] == raw["anchor"]
    assert set(row) <= set(batch.FIELDS)


@pytest.mark.parametrize("mode", ["lookup", "fact_pattern"])
def test_conceptual_framing_does_not_leak_into_existing_modes(conceptual_case, mode):
    _, payload, candidates = conceptual_case
    raw = dict(candidates[0], question_type="obligation_or_prohibition")
    candidate, = gen.parse_candidates([raw], payload, mode)
    row, = gen.rows_for(payload, [candidate], mode=mode, language="en")
    assert row["question_type"] == raw["question_type"] and row["framing"] == ""
    assert row["anchor"] == (raw["anchor"] if mode == "lookup" else "")


def test_conceptual_preserves_reference_keys_and_rejects_unsupplied_sources(conceptual_case):
    _, original, candidates = conceptual_case
    internal = ctx.ArticleUnit("eli:test:article3", original.target.celex_id, "3",
                               texts={"en": "The compliance function assesses identified risks."})
    external = ctx.ArticleUnit("eli:other:article5", "32000R0002", "5",
                               texts={"en": "The procedure includes independent review."})
    annex = ctx.ArticleUnit("http://data.europa.eu/eli/reg/2000/1/anx_1/oj",
                            original.target.celex_id, "", unit_type="annex",
                            texts={"en": "The review records the nature of each risk."})
    payload = ctx.GenerationPayload(
        original.target, [internal], [],
        ctx.render_payload(original.target, [internal], external=[external], annexes=[annex]),
        external_references=[external], annexes=[annex],
    )
    raw = dict(candidates[0], articles_involved=["Article 3", "32000R0002:5",
                                                "32000R0001:anx_1", "32000R0999:9"])
    candidate, = gen.parse_candidates([raw], payload, "conceptual")
    assert candidate.articles_involved == ["4", "3", "32000R0002:5", "32000R0001:anx_1"]
    assert candidate.involved_elis == [unit.eli_id for unit in
                                       (original.target, internal, external, annex)]
    assert candidate.rejected_involved == ["32000R0999:9"]
    assert candidate.multi_article and candidate.cross_act
    row, = gen.rows_for(payload, [candidate], mode="conceptual", language="en")
    for unit in (internal, external, annex):
        assert unit.texts["en"] in row["referenced_articles_text"]
    assert "32000R0999:9" not in row["articles_involved"]


def test_normal_conceptual_batch_and_csv_regrade_use_actual_compact_rubric(
        conceptual_case, monkeypatch):
    target, payload, candidates = conceptual_case
    expected = {item["question"]: item for item in candidates}
    quality_prompt = gen.PROMPTS.quality("conceptual", "batch")
    assert set(grading.rubric_keys(quality_prompt)) == QUALITY_KEYS
    stages, quality_inputs = [], []
    replay = False

    def create(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        content = kwargs["messages"][1]["content"]
        if kwargs["model"] == GENERATOR:
            assert not replay, "Regrading must not regenerate questions"
            stages.append("generation")
            assert prompt == gen.PROMPTS.generation("conceptual", "en")
            assert content == payload.text
            output = candidates
        elif prompt == gen.PROMPTS.faithfulness("batch"):
            stages.append("faithfulness")
            assert kwargs["model"] == GRADER
            assert "Articles involved (declared): 4" in content
            output = [dict(index=index, **dict.fromkeys(grading.FAITHFULNESS_KEYS, 5))
                      for index in range(len(candidates))]
        else:
            stages.append("quality")
            assert kwargs["model"] == GRADER and prompt == quality_prompt
            inputs = json.loads(content)
            assert inputs["passages"] == payload.text
            quality_inputs.append(inputs["candidates"])
            rows = []
            for index, item in enumerate(inputs["candidates"]):
                raw = expected[item["question"]]
                assert item["framing"] == raw["framing"]
                assert item["anchor"] == raw["anchor"]
                assert item.get("question_type", "") == ""
                assert item["articles_involved"] == ["4"]
                assert item["question_language"] == "en"
                assert "generator_model_id" not in item and "total_score" not in item
                score = 5 if raw["framing"] == "stakeholder" else 4
                rows.append({
                    "index": index, "candidate_id": item["candidate_id"],
                    "scores": dict.fromkeys(QUALITY_KEYS, score),
                    "score_notes": dict.fromkeys(QUALITY_KEYS, "The supplied evidence supports this need."),
                    "checks": dict.fromkeys(("mode", "support", "metadata"), "pass"),
                    "problems": [],
                })
            output = {"candidates": rows, "batch_diversity": "pass"}
        return response(json.dumps(output), model=kwargs["model"])

    monkeypatch.setattr(llm, "client_for", lambda _: Client(create))
    options = {"gen_model": GENERATOR, "grade_model": GRADER, "max_references": 6,
               "keep": 3, "retries": 1, "payload": payload}
    rows = batch.run_one(target, None, checkpoint=MemoryCheckpoint(), **options)
    assert stages == ["generation", "faithfulness", "quality"]
    assert [row["framing"] for row in rows] == ["stakeholder", "response"]
    assert [row["qual_overall"] for row in rows] == [25, 20]
    assert [row["total_score"] for row in rows] == [40, 35]
    for row in rows:
        assert row["question_type"] == "" and row["grading_status"] == "completed"
        assert row["quality_audit_status"] == "clear"
        assert row["faith_overall"] == 15
        assert row["quality_verifier_prompt"] == quality_prompt
        assert all(row[f"qual_{key}"] == row["qual_overall"] / 5 for key in QUALITY_KEYS)

    # Exercise the exported string representation, including an obsolete score
    # that must be cleared when a different rubric is used for regrading.
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=[*rows[0], "qual_conceptual_framing"])
    writer.writeheader()
    writer.writerows(dict(row, qual_conceptual_framing=1) for row in rows)
    stream.seek(0)
    saved = list(csv.DictReader(stream))
    replay = True
    regraded = batch.run_one(target, None, existing_candidates=saved, **options)
    assert stages == ["generation", "faithfulness", "quality", "faithfulness", "quality"]
    assert len(quality_inputs) == 2
    for before, after in zip(rows, regraded):
        for key in ("candidate_id", "question", "answer", "framing", "anchor",
                    "articles_involved", "qual_overall", "total_score"):
            assert after[key] == before[key]
        assert after["grading_status"] == "completed"
        assert after["qual_conceptual_framing"] == ""


@pytest.mark.parametrize("backend", ["generator", "jev"])
def test_eurlex_decider_can_choose_conceptual(conceptual_case, monkeypatch, backend):
    _, payload, _ = conceptual_case
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        assert backend == "generator"
        assert kwargs["messages"][0]["content"] == decider.prompt_text("eurlex", "generator")
        assert kwargs["messages"][1]["content"] == payload.text
        return response(json.dumps({"mode": "conceptual", "reason": "The article explains a safeguard."}),
                        model=kwargs["model"])

    def native(request):
        calls.append(request)
        assert backend == "jev" and request["state"] == payload.text
        choices = request["questions"]["mode"]["criteria"]
        assert set(choices) == {*decider.MODES["eurlex"], "skip"}
        return {"answers": {"mode": {"type": "choice", "choice": "conceptual",
                "confidence": 1.0, "probabilities": {key: float(key == "conceptual") for key in choices}}}}

    monkeypatch.setattr(llm, "client_for", lambda _: Client(create))
    monkeypatch.setattr(llm, "decisions", native)
    result = decider.decide("eurlex", payload, backend=backend, model=GENERATOR, retries=1)
    assert result["mode"] == result["generation_mode"] == "conceptual"
    assert len(calls) == 1
