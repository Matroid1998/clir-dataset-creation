"""Evidence remains source-bound through both graders, replay, and verifier reuse."""

import csv
import io
import json
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from clir_bench.core import grading, llm
from clir_bench.domains.legal import qac
from clir_bench.domains.legal.qac import eurlex_generate, un_generate
from clir_bench.domains.legal.qac.batch_recording import RunState, read_run_metadata
from clir_bench.domains.legal.qac.evidence import CONTRACT, MARKER, EvidenceSource, uses_contract
from test_legal_batch_replay_integration import Client, MemoryCheckpoint, case, response


@pytest.fixture(autouse=True)
def local_prompts(monkeypatch):
    monkeypatch.delenv("CLIR_PROMPT_MANIFEST", raising=False)
    monkeypatch.setenv("CLIR_PROMPT_SOURCE", "local")


def evidence_candidate(corpus, mode=None):
    batch, generator, target, payload, raw = case(corpus)
    mode = mode or target.mode
    target = replace(target, mode=mode)
    candidate = (
        generator.parse_candidates(raw, payload, mode)[0]
        if corpus == "eurlex" else generator.parse_candidates(raw, mode)[0]
    )
    full = payload.target.texts["en"]
    quote = "report annually" if corpus == "eurlex" else "received support"
    reference = payload.target.article_number if corpus == "eurlex" else "target"
    source_id = payload.target.eli_id if corpus == "eurlex" else payload.target.block_id
    start = full.index(quote)
    span = {
        "source_ref": reference, "source_id": source_id, "language": "en", "text": quote,
        "start": start, "end": start + len(quote),
    }
    candidate = replace(
        candidate, answer=quote, evidence=quote, evidence_contract=CONTRACT,
        evidence_language="en", evidence_source_id=source_id, evidence_spans=[span],
        evidence_is_translation=False, answer_is_translation=False,
    )
    payload.evidence_sources = {(reference, "en"): EvidenceSource(
        reference, source_id, "en", full, full, "article" if corpus == "eurlex" else "block",
    )}
    return batch, generator, target, payload, candidate


MODES = [("eurlex", mode) for mode in eurlex_generate.MODES] + [
    ("un", mode) for mode in un_generate.MODES
]


@pytest.mark.parametrize("corpus,mode", MODES)
def test_all_modes_pass_evidence_to_both_graders_and_regrade(monkeypatch, corpus, mode):
    batch, generator, target, payload, candidate = evidence_candidate(corpus, mode)
    monkeypatch.setattr(generator, "generate", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(llm, "client_for", lambda _: object())
    calls = []

    def faith(*args, **kwargs):
        calls.append(("faith", args[4], kwargs))
        return [{"grounding": 4, "precision": 4, "numerical_fidelity": 4, "overall": 12}]

    def quality(*args, **kwargs):
        calls.append(("quality", args[4], kwargs))
        keys = grading.rubric_keys(args[2]) or grading.quality_keys(mode)
        return [dict.fromkeys(keys, 4) | {"overall": 20, "_keys": keys}]

    monkeypatch.setattr(grading, "grade_faithfulness", faith)
    monkeypatch.setattr(grading, "grade_quality", quality)
    options = {"max_references": 6} if corpus == "eurlex" else {"context_chars": 30000}
    checkpoint = MemoryCheckpoint()
    rows = batch.run_one(
        target, None, payload=payload, gen_model="model/generator", grade_model="model/grader",
        keep=3, checkpoint=checkpoint, deduplicate_exact_questions=True, **options,
    )
    assert rows[0]["grading_status"] == "completed"
    assert checkpoint.completed["generation"][0]["evidence_spans"] == candidate.evidence_spans
    assert json.loads(rows[0]["evidence_spans"]) == candidate.evidence_spans
    for _, pairs, kwargs in calls:
        assert pairs[0]["evidence_spans"] == candidate.evidence_spans
        assert pairs[0]["evidence_language"] == "en"
        assert pairs[0]["evidence_source_id"] == candidate.evidence_source_id
        assert pairs[0]["evidence_contract"] == CONTRACT
        assert kwargs["reuse_identity"] is None
    assert calls[1][2]["sources"][0]["texts"] == payload.target.texts

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
    saved = list(csv.DictReader(io.StringIO(stream.getvalue())))
    monkeypatch.setattr(generator, "generate", lambda *args, **kwargs: pytest.fail("regenerated QA"))
    calls.clear()
    regraded = batch.run_one(
        target, None, payload=payload, gen_model="model/generator", grade_model="model/grader",
        keep=3, existing_candidates=saved, deduplicate_exact_questions=True, **options,
    )
    assert json.loads(regraded[0]["evidence_spans"]) == candidate.evidence_spans
    assert regraded[0]["evidence"] == candidate.evidence
    assert regraded[0]["evidence_is_translation"] is False
    assert len(calls) == 2 and all(call[1][0]["evidence_spans"] for call in calls)


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
def test_generation_only_preserves_canonical_evidence(monkeypatch, corpus):
    batch, generator, target, payload, candidate = evidence_candidate(corpus)
    monkeypatch.setattr(generator, "generate", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(llm, "client_for", lambda _: object())
    monkeypatch.setattr(grading, "grade_faithfulness", lambda *a, **k: pytest.fail("graded"))
    options = {"max_references": 6} if corpus == "eurlex" else {"context_chars": 30000}
    rows = batch.run_one(
        target, None, payload=payload, gen_model="model/generator", grade_model="model/grader",
        keep=3, checkpoint=MemoryCheckpoint(), generation_only=True, **options,
    )
    assert rows[0]["evidence_contract"] == CONTRACT
    assert json.loads(rows[0]["evidence_spans"]) == candidate.evidence_spans


def test_verifier_sources_preserve_visible_languages_and_custom_reference_truncation():
    batch, _, _, payload, _ = evidence_candidate("eurlex")
    reference = payload.references[0]
    reference.texts["de"] = "This language was not supplied."
    full = reference.texts["en"]
    visible = full[:14]
    payload.evidence_sources[("3", "en")] = EvidenceSource(
        "3", reference.eli_id, "en", visible, full, "article",
    )
    sources = batch.quality_sources(payload, "de", evidence_contract=True)
    source = next(row for row in sources if row["source_id"] == reference.eli_id)
    assert source["texts"] == {"en": visible}
    assert source["metadata"]["languages"] == ["en"]
    assert source["metadata"]["source_ref"] == "3"
    assert "not supplied" not in source["text"]
    assert full not in source["text"]
    legacy = batch.quality_sources(payload, "de")
    assert "texts" not in legacy[0]
    assert "This language was not supplied." in legacy[1]["text"]


def test_faithfulness_span_rendering_is_lossless_json():
    span = {"source_id": "a", "language": "fr", "text": 'Texte « exact »\navec "guillemets".',
            "start": 0, "end": 43}
    rendered = grading.candidates_block([{"question": "Q?", "answer": "A", "evidence_spans": [span]}])
    encoded = rendered.split("  evidence_spans: ", 1)[1].split("\n  Answer:", 1)[0]
    assert json.loads(encoded) == [span]


def test_compact_quality_receives_all_spans_and_source_language_bodies(monkeypatch):
    batch, generator, target, payload, candidate = evidence_candidate("eurlex")
    prompt = generator.PROMPTS.quality(target.mode)
    pair = {"candidate_id": "q", "question_language": "en", "question": candidate.question,
            "answer": candidate.answer, **generator.annotations.metadata(candidate)}
    captured = []
    result = deepcopy(grading._quality_output_example(prompt))
    result["candidates"][0]["candidate_id"] = "q"
    result["candidates"][0]["scores"] = dict.fromkeys(grading.rubric_keys(prompt), 4)
    monkeypatch.setattr(grading, "_invoke", lambda *args: captured.append(json.loads(args[3]))
                        or json.dumps(result))
    sources = batch.quality_sources(payload, "en", evidence_contract=True)
    grading.grade_quality(
        object(), grading.GraderConfig("model/grader"), prompt, payload.text, [pair], target.mode,
        sources=sources, target_source_id=target.eli_id, policies={"reference_policy": "supplied"},
    )
    envelope = captured[0]
    assert envelope["candidates"][0]["evidence_spans"] == candidate.evidence_spans
    assert envelope["sources"] == sources
    assert envelope["target_source_id"] == target.eli_id
    assert envelope["policies"] == {"reference_policy": "supplied"}


@pytest.mark.parametrize("stage", ["faithfulness", "quality"])
def test_same_question_with_different_evidence_cannot_reuse_grade(tmp_path, stage):
    _, generator, target, payload, candidate = evidence_candidate("eurlex")
    prompt = (generator.PROMPTS.faithfulness() if stage == "faithfulness"
              else generator.PROMPTS.quality(target.mode))
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        if stage == "faithfulness":
            result = [{"index": 0, **dict.fromkeys(grading.FAITHFULNESS_KEYS, 4)}]
        else:
            result = deepcopy(grading._quality_output_example(prompt))
            inputs = json.loads(kwargs["messages"][1]["content"])["candidates"]
            result["candidates"][0]["candidate_id"] = inputs[0]["candidate_id"]
            result["candidates"][0]["scores"] = dict.fromkeys(grading.rubric_keys(prompt), 4)
        return response(json.dumps(result))

    with RunState(tmp_path / "run.sqlite") as state:
        for number, quote in enumerate(("report annually", "must report annually", "report annually")):
            pair = {"candidate_id": str(number), "question_language": "en", "question": "How often?",
                    "answer": "annually", "evidence_contract": CONTRACT,
                    "evidence_spans": [{**candidate.evidence_spans[0], "text": quote}]}
            client = state.checkpoint(str(number)).client(stage, "model/grader", Client(transport))
            args = (client, grading.GraderConfig("model/grader"), prompt, payload.text, [pair])
            kwargs = {"strict": True, "reuse_identity": {"deduplicate": "exact_question"}}
            if stage == "faithfulness":
                grading.grade_faithfulness(*args, **kwargs)
            else:
                grading.grade_quality(*args, target.mode, **kwargs)
    assert len(calls) == 2  # The third candidate has exactly the first candidate's evidence.


def test_legacy_exact_question_reuse_remains_available():
    identity = {"deduplicate": "exact_question", "other": "retained"}
    assert grading._verification_reuse_identity([{"question": "Q?"}], identity) == identity
    assert grading._verification_reuse_identity(
        [{"question": "Q?", "evidence_contract": CONTRACT}], identity,
    ) == {"other": "retained"}
    assert identity["deduplicate"] == "exact_question"


@pytest.mark.parametrize("corpus", ["eurlex", "un"])
def test_invalid_evidence_exhausts_generation_budget_and_records_failure(tmp_path, monkeypatch, corpus):
    _, generator, target, payload, candidate = evidence_candidate(corpus)
    mode, language = target.mode, target.language
    prompt = generator.PROMPTS.generation(mode, language)
    assert uses_contract(prompt)
    invalid = {
        "question": candidate.question, "answer": candidate.answer, "answer_is_translation": False,
        "evidence_spans": [{
            "source_ref": "4" if corpus == "eurlex" else "target", "language": "en",
            "text": "This quotation does not occur in the supplied source.",
        }],
    }
    if corpus == "eurlex":
        invalid["articles_involved"] = ["4"]
    calls = []

    def transport(**kwargs):
        assert kwargs["model"] == "model/generator", "invalid evidence reached a grader"
        calls.append(kwargs)
        return response(json.dumps([invalid]), model="model/generator")

    # Only the provider transport is mocked: generation parsing, checkpoint
    # retry handling, persisted outcomes, and resume follow the real workflow.
    monkeypatch.setattr(llm, "client_for", lambda _: Client(transport))
    options = {
        "max_references": 6, "context_chars": 30000, "reference_chars": 4000,
        "keep": 3, "retries": 2, "generation_model": ["model/generator"],
        "verifier_model": "model/grader",
    }
    entry = {"corpus": corpus, "target": asdict(target),
             "payload_sha256": qac._digest(payload.text.encode())}
    prompts = {
        f"{corpus}/generation/{mode}": prompt,
        f"{corpus}/faithfulness": generator.PROMPTS.faithfulness(),
        f"{corpus}/quality/{mode}": generator.PROMPTS.quality(mode),
    }
    directory = tmp_path / corpus
    cases = [(entry, target, payload, "model/generator", None)]

    def execute(saved):
        return qac._execute(
            SimpleNamespace(workers=1), None, "generate", directory, directory / "results.csv",
            saved, options, [entry], prompts, {corpus: None}, cases,
        )

    assert execute({}) == 1
    assert len(calls) == 2
    with RunState(directory / "run.sqlite") as state:
        outcome, = state.outcomes()
        assert outcome["status"] == "failed"
        assert outcome["rows"] == []
        assert "generation failed after 2 attempts" in outcome["error"]
        assert "exact visible source passage" in outcome["error"]
        assert state.get("status") == "completed_with_errors"
        assert state.get("summary")["failed"] == 1
        with state.transaction() as db:
            stages = db.execute("SELECT stage, attempts, status, value FROM stages").fetchall()
        assert stages == [("generation", 2, "failed", None)]
    assert execute(read_run_metadata(directory)) == 1
    assert len(calls) == 2  # Resuming cannot reset the exhausted generation budget.


@pytest.mark.parametrize("generation_prompt,expected", [
    (MARKER + "\nGenerate supporting spans.", "full_candidate_with_evidence-v1"),
    ("Legacy generation contract", "exact_question_within_target-v1"),
])
def test_weighted_run_provenance_names_the_actual_verification_policy(
    tmp_path, monkeypatch, generation_prompt, expected,
):
    recorded = []
    digest = qac._digest

    def capture(value):
        if isinstance(value, dict) and "verification_deduplication_policy" in value:
            recorded.append(value)
        return digest(value)

    monkeypatch.setattr(qac, "_digest", capture)
    assert qac._execute(
        SimpleNamespace(workers=1), None, "generate", tmp_path, tmp_path / "results.csv", {},
        {"persona_selection_policy": "weighted"}, [],
        {"eurlex/generation/lookup": generation_prompt}, {}, [], validate_only=True,
    ) == 0
    assert recorded[0]["verification_deduplication_policy"] == expected
    assert not (tmp_path / "run.sqlite").exists()
