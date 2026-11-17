"""Persona generation, metadata transport, abstention and multilingual resource contracts."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from clir_bench.core import corpus, grading, qagen
from clir_bench.core.prompts import PromptPack
from clir_bench.domains.chemistry.domain import SPEC
from clir_bench.domains.chemistry.qac.candidates import parse_candidates
from clir_bench.domains.chemistry.qac.plans import PLANS
from clir_bench.domains.chemistry.schema import SCHEMA
from clir_bench.domains.chemistry.vocabulary import MODES, WORKING_LANGUAGES

PACK = PromptPack(SPEC.prompts_package)
RUBRIC_VERSION = "chemistry-clir-exceptional-r2"


def verifier_response(prompt, candidates, score=4, *, metadata_status="pass"):
    """Use the supplied verifier contracts, including their audit diagnostics."""
    if prompt == PACK.faithfulness():
        return json.dumps([
            dict.fromkeys(grading.FAITHFULNESS_KEYS, score)
            | {"index": index, "reason": f"Supported at score {score}"}
            for index in range(len(candidates))
        ])
    keys = grading.rubric_keys(prompt)
    return json.dumps({
        "candidates": [
            {
                "index": index,
                "candidate_id": candidate.get("candidate_id"),
                "scores": dict.fromkeys(keys, score),
                "score_notes": {key: f"{key} assessed at score {score}" for key in keys},
                "checks": {"mode": "pass", "support": "pass", "metadata": metadata_status},
                "problems": ([] if metadata_status == "pass" else [
                    "metadata: The evidence has a harmless literal whitespace mismatch.",
                ]),
            }
            for index, candidate in enumerate(candidates)
        ],
        # The multi-candidate integration fixture deliberately repeats one need.
        "batch_diversity": "not_applicable" if len(candidates) == 1 else "fail",
    })


def examples(mode, language="en"):
    prompt = PACK.generation(mode, language)
    # The first array is the schema; r2 supplies multiple worked examples and a skip.
    return [
        json.JSONDecoder().raw_decode(prompt[match.start():])[0]
        for match in re.finditer(r"^\[\n", prompt, re.MULTILINE)
    ][1:]


def config():
    return qagen.GenerationConfig("test", grading.GraderConfig("test"), retries=1,
                                  candidate_parser=SPEC.qac_candidate_parser)


def document(candidate):
    family = candidate.get("source_identifier", "EP-1234567-A1")
    language = candidate["evidence_language"]
    return {"publication_number": family, "id": f"{family}_{language}",
            "language": language, "context": candidate["evidence"]}


def accepted_documents(rows):
    """A symmetric accepted-pair corpus fixture for the chemistry CLI preflight."""
    output = []
    for row in rows:
        peers = [peer for peer in rows if peer["publication_number"] == row["publication_number"]
                 and peer["language"] != row["language"]]
        assert peers, "An accepted source endpoint needs a cross-language peer"
        output.append({
            **row, "abstract": row["context"], "source": "ai4chem_abstract_pairs",
            "upstream_dataset": "BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs",
            "upstream_revision": "test-sha",
            "accepted_pair_ids": json.dumps([
                "pair-" + "-".join(sorted((row["id"], peer["id"]))) for peer in peers
            ]),
            "paired_document_ids": json.dumps([peer["id"] for peer in peers]),
        })
    return output


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("language", WORKING_LANGUAGES)
def test_supplied_worked_examples_and_abstentions_obey_the_parser(mode, language):
    worked = examples(mode, language)
    candidates = [example for example in worked if not (len(example) == 1 and "skip_reason" in example[0])]
    skips = [example for example in worked if len(example) == 1 and "skip_reason" in example[0]]
    assert candidates and skips
    for example in candidates:
        assert parse_candidates(example, mode) == example
    for example in skips:
        assert parse_candidates(example, mode) == []
    assert set(PACK.available_languages(mode)) == set(WORKING_LANGUAGES)
    # Every mode has a matching rubric; generic mode-name defaults may differ.
    keys = grading.rubric_keys(PACK.quality(mode))
    assert len(keys) == 5
    assert all(f"qual_{key}" in qagen.output_fieldnames(SCHEMA, prompts=PACK, modes=MODES)
               for key in keys)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("language", ["en", "ja", "ru"])
def test_generate_grade_and_csv_round_trip_keeps_persona_evidence(
    monkeypatch, tmp_path, mode, count, language,
):
    candidate = examples(mode, language)[0][0]
    if language != "en":
        candidate["evidence"] = candidate["answer"] if mode != "source_finding" else {
            "ja": "PDZドメイン相互作用によって膜タンパク質をウイルス様粒子上に提示する。",
            "ru": "Мембранные белки представлены на вирусоподобных частицах посредством взаимодействия PDZ-доменов.",
        }[language]
        candidate["evidence_language"] = language
        candidate["answer_is_translation"] = False
    candidates = [dict(candidate) for _ in range(count)]
    row = document(candidate)
    item = qagen.PlanItem(row["publication_number"], language, mode)
    calls = []

    def generate(client, model, messages, **kwargs):
        assert messages[0]["content"] == PACK.generation(mode, language)
        assert f"publication_number: {item.family}" in messages[1]["content"]
        assert candidate["evidence"] in messages[1]["content"]
        return json.dumps(candidates)

    def grade(client, settings, prompt, content):
        payload = json.loads(content)
        calls.append(prompt)
        assert f"publication_number: {item.family}" in payload["passages"]
        assert candidate["evidence"] in payload["passages"]
        assert payload["candidates"] == [
            dict(candidate, mode=mode, question_language=language) for _ in range(count)
        ]
        return verifier_response(prompt, payload["candidates"])

    monkeypatch.setattr(qagen, "chat", generate)
    monkeypatch.setattr(grading, "_invoke", grade)
    generated = qagen.generate_for_document(
        item, [row], schema=SCHEMA, prompts=PACK, config=config(),
        generation_client=None, grading_client=None,
    )
    assert len(generated) == count
    assert calls == [PACK.faithfulness(), PACK.quality(mode)]
    fields = qagen.output_fieldnames(SCHEMA, prompts=PACK, modes=MODES)
    path = tmp_path / "results.csv"
    corpus.write_rows(path, [qagen.normalize_row(row, fields) for row in generated], fields)
    for saved in corpus.read_rows(path):
        restored = qagen.candidate_from_row(saved)
        assert restored == dict(candidate, mode=mode, question_language=language)
        assert saved["total_score"] == "32"
        assert saved["faith_overall"] == "12"
        assert saved["qual_overall"] == "20"
        assert saved["faithfulness_rubric_version"] == RUBRIC_VERSION
        assert saved["quality_rubric_version"] == RUBRIC_VERSION
        assert json.loads(saved["faithfulness_verifier_response_json"])["grounding"] == 4
        audit = json.loads(saved["quality_verifier_response_json"])
        assert audit["scores"] == dict.fromkeys(grading.rubric_keys(PACK.quality(mode)), 4)
        for key in audit["scores"]:
            assert saved[f"qual_{key}"] == "4"
        assert json.loads(saved["quality_score_notes_json"]) == audit["score_notes"]
        assert json.loads(saved["quality_checks_json"]) == audit["checks"]
        assert json.loads(saved["quality_problems_json"]) == []
        assert saved["quality_batch_diversity"] == ("not_applicable" if count == 1 else "fail")
        assert saved["quality_audit_status"] == "clear"
        assert all(saved[f"quality_{key}_status"] == "pass" for key in audit["checks"])


def test_abstention_does_not_call_graders(monkeypatch):
    monkeypatch.setattr(qagen, "chat", lambda *a, **k: '[{"skip_reason":"No stated route"}]')
    monkeypatch.setattr(grading, "_invoke", lambda *a: pytest.fail("graded an abstention"))
    row = document(examples("lookup")[0][0])
    assert qagen.generate_for_document(
        qagen.PlanItem(row["publication_number"], "en", "process_retrieval"), [row],
        schema=SCHEMA, prompts=PACK, config=config(), generation_client=None, grading_client=None,
    ) == []


@pytest.mark.parametrize("mutate", [
    lambda item: [],
    lambda item: item,
    lambda item: [item] * 4,
    lambda item: [item, {"skip_reason": "Mixed skip"}],
    lambda item: [dict(item, answer_is_translation="false")],
    lambda item: [dict(item, question_type="material")],
    lambda item: [dict(item, question=None)],
    lambda item: [dict(item, extra="unexpected")],
    lambda item: [{key: value for key, value in item.items() if key != "evidence"}],
])
def test_malformed_generation_is_rejected_for_retry(mutate):
    with pytest.raises(ValueError):
        parse_candidates(mutate(examples("lookup")[0][0]), "lookup")


def test_source_identity_is_checked_and_substring_metadata_reaches_verifier():
    source = examples("source_finding")[0][0]
    with pytest.raises(ValueError, match="equal source_identifier"):
        parse_candidates([dict(source, answer="EP-WRONG")], "source_finding")
    with pytest.raises(ValueError, match="must be false"):
        parse_candidates([dict(source, answer_is_translation=True)], "source_finding")
    comparison = examples("comparison")[0][0]
    reversed_entities = dict(comparison, comparison_entities=comparison["comparison_entities"][::-1])
    assert parse_candidates([reversed_entities], "comparison") == [reversed_entities]
    invalid_anchor = dict(examples("lookup")[0][0], anchor="not in question")
    assert parse_candidates([invalid_anchor], "lookup") == [invalid_anchor]


@pytest.mark.parametrize("plan_name", PLANS)
def test_all_plans_default_to_the_seven_personas_and_reject_old_modes(plan_name):
    row = document(examples("lookup")[0][0])
    context = SimpleNamespace(schema=SCHEMA, languages=SPEC.languages)
    args = SimpleNamespace(seed=42, questions=4, pool=1, modes=None, langs=["en"])
    kwargs = {"context": context, "source": SPEC.source("gp"),
              "grouped": {row["publication_number"]: [row]}, "args": args}
    assert {item.mode for item in PLANS[plan_name](**kwargs)} == set(MODES)
    args.modes = ["technical"]
    with pytest.raises(ValueError, match="Unknown chemistry modes"):
        PLANS[plan_name](**kwargs)


def test_best_selection_keeps_different_personas_for_the_same_document():
    rows = [{"publication_number": "EP-1234567-A1", "question_language": "en", "mode": mode,
             "total_score": score}
            for mode, score in [("lookup", 30), ("lookup", 40), ("conceptual", 35)]]
    assert {(row["mode"], row["total_score"]) for row in qagen.select_best(rows, SCHEMA)} == {
        ("lookup", 40), ("conceptual", 35),
    }


@pytest.mark.parametrize("value", [None, "", "not a score", "NaN", "Infinity", "-Infinity",
                                  float("nan"), float("inf"), -float("inf")])
def test_invalid_and_nonfinite_totals_never_win_best_selection(value):
    unscored = {"publication_number": "EP-1234567-A1", "question_language": "en",
                "mode": "lookup", "total_score": value}
    scored = dict(unscored, total_score="32")
    assert qagen.select_best([unscored], SCHEMA) == []
    assert qagen.select_best([unscored, scored], SCHEMA) == [scored]
    assert qagen.select_best([scored, unscored], SCHEMA) == [scored]


@pytest.mark.parametrize("stage", ["faithfulness", "quality"])
def test_unassessable_candidates_keep_null_totals_and_cannot_become_scored_winners(
    monkeypatch, tmp_path, stage,
):
    first = examples("lookup")[0][0]
    second = dict(first, question=first["question"] + " ")
    row = document(first)
    candidates = [first, second]
    monkeypatch.setattr(qagen, "chat", lambda *a, **k: json.dumps(candidates))

    def grade(client, settings, prompt, content):
        payload = json.loads(content)
        result = json.loads(verifier_response(prompt, payload["candidates"]))
        if stage == "faithfulness" and prompt == PACK.faithfulness():
            result[0]["grounding"] = None
            result[0]["reason"] = "The supplied source unit is truncated before decisive evidence."
        elif stage == "quality" and prompt == PACK.quality("lookup"):
            item = result["candidates"][0]
            key = grading.rubric_keys(prompt)[2]
            item["scores"][key] = None
            item["score_notes"][key] = "The supplied source unit omits decisive evidence."
            item["checks"]["support"] = "uncertain"
            item["problems"] = ["support: The full source unit is needed to assess this answer."]
        return json.dumps(result)

    monkeypatch.setattr(grading, "_invoke", grade)
    generated = qagen.generate_for_document(
        qagen.PlanItem(row["publication_number"], "en", "lookup"), [row],
        schema=SCHEMA, prompts=PACK, config=config(), generation_client=None, grading_client=None,
    )
    scored, unassessable = generated
    assert scored["question"] == second["question"]
    assert scored["total_score"] == 32 and scored["candidate_rank"] == 1
    assert unassessable["question"] == first["question"]
    assert unassessable["total_score"] is None
    assert unassessable["faith_overall" if stage == "faithfulness" else "qual_overall"] is None
    assert qagen.select_best(generated, SCHEMA) == [scored]
    assert qagen.select_best([unassessable], SCHEMA) == []

    fields = qagen.output_fieldnames(SCHEMA, prompts=PACK, modes=MODES)
    path = tmp_path / "unassessable.csv"
    corpus.write_rows(path, [qagen.normalize_row(unassessable, fields)], fields)
    saved = corpus.read_rows(path)
    assert saved[0]["total_score"] == ""
    assert qagen.select_best(saved, SCHEMA) == []


def test_legacy_default_parser_still_supports_semantic_generation(monkeypatch):
    # The shared parser previously referenced MODE_SEMANTIC without importing it.
    candidate = {"question": "Question?", "answer": "Answer", "framing": "solution"}
    monkeypatch.setattr(qagen, "chat", lambda *a, **k: json.dumps([candidate]))
    legacy = qagen.GenerationConfig("test", grading.GraderConfig("test"))
    assert qagen.generate_candidates(None, legacy, "prompt", "passage", "semantic") == [candidate]


def test_cli_generation_and_regrading_preserve_source_metadata_and_replace_grade_audits(
    monkeypatch, tmp_path,
):
    from clir_bench import domains
    from clir_bench.cli import build_parser
    from clir_bench.cli import qac as cli
    from clir_bench.core.config import Settings
    from clir_bench.core.context import AppContext

    context = AppContext.build(Settings(
        project_root=tmp_path, data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports", cluster_dir=tmp_path / "cluster",
    ), SPEC)
    parser = build_parser(context, domains.load_module("chemistry"))
    # The source and prompt share a language; array metadata survives regrading.
    candidate = next(
        candidate
        for example in examples("constraint_search")
        for candidate in example
        if candidate.get("evidence_language") == "en"
    )
    assert candidate["evidence_language"] == "en"
    source = document(candidate)
    source_rows = accepted_documents([
        source, dict(source, id=f"{source['publication_number']}_fr", language="fr",
                     context="Autre version française du document."),
    ])
    corpus.write_rows(context.workspace.corpus_csv("gp"), source_rows, tuple(source_rows[0]))
    monkeypatch.setattr(cli, "client_for", lambda model: None)
    monkeypatch.setattr(qagen, "chat", lambda *a, **k: json.dumps([candidate]))
    score = 4
    metadata_status = "pass"

    def grade(client, settings, prompt, content):
        payload = json.loads(content)
        assert payload["candidates"] == [
            dict(candidate, mode="constraint_search", question_language="en"),
        ]
        assert f"publication_number: {source['publication_number']}" in payload["passages"]
        return verifier_response(prompt, payload["candidates"], score,
                                 metadata_status=metadata_status)

    monkeypatch.setattr(grading, "_invoke", grade)
    output = tmp_path / "generated.csv"
    args = parser.parse_args([
        "qac", "generate", "--source", "gp", "--langs", "en", "--modes", "constraint_search",
        "--questions", "1", "--limit", "1", "--output", str(output),
    ])
    assert args.handler(args, context) == 0
    saved = corpus.read_rows(output)
    assert len(saved) == 1
    assert saved[0]["total_score"] == "32"
    assert saved[0]["quality_audit_status"] == "clear"

    # Simulate an input exported by an older rubric: regrading clears old metrics.
    saved[0]["qual_focus"] = "5"
    saved[0]["qual_failure_type"] = "obsolete rubric diagnosis"
    corpus.write_rows(output, saved, tuple(saved[0]))

    # Regrading must update stored raw responses, not retain the generator run's grades.
    score = 3
    metadata_status = "fail"
    regraded = tmp_path / "regraded.csv"
    args = parser.parse_args([
        "qac", "regrade", "--source", "gp", "--input", str(output), "--output", str(regraded),
    ])
    assert args.handler(args, context) == 0
    saved = corpus.read_rows(regraded)[0]
    assert saved["total_score"] == "24"
    assert saved["faithfulness_rubric_version"] == RUBRIC_VERSION
    assert saved["quality_rubric_version"] == RUBRIC_VERSION
    assert json.loads(saved["faithfulness_verifier_response_json"])["grounding"] == 3
    audit = json.loads(saved["quality_verifier_response_json"])
    assert audit["scores"] == dict.fromkeys(grading.rubric_keys(PACK.quality("constraint_search")), 3)
    assert json.loads(saved["quality_score_notes_json"]) == audit["score_notes"]
    assert json.loads(saved["quality_checks_json"]) == audit["checks"]
    assert json.loads(saved["quality_problems_json"]) == audit["problems"]
    assert saved["quality_metadata_status"] == "fail"
    assert saved["quality_audit_status"] == "repair"
    assert saved["quality_batch_diversity"] == "not_applicable"
    assert saved["qual_focus"] == ""
    assert saved["qual_failure_type"] == ""
    assert qagen.candidate_from_row(saved) == dict(
        candidate, mode="constraint_search", question_language="en",
    )


@pytest.mark.parametrize("use_decider", [False, True])
def test_french_documents_use_french_prompts_and_exact_source_through_regrading(
    monkeypatch, tmp_path, use_decider,
):
    from clir_bench import domains
    from clir_bench.cli import build_parser
    from clir_bench.cli import qac as cli
    from clir_bench.core import llm
    from clir_bench.core.config import Settings
    from clir_bench.core.context import AppContext

    context = AppContext.build(Settings(
        project_root=tmp_path, data_dir=tmp_path / "data", reports_dir=tmp_path / "reports",
        cluster_dir=tmp_path / "cluster",
    ), SPEC)
    parser = build_parser(context, domains.load_module("chemistry"))
    family = "EP-1234567-A1"
    rows = accepted_documents([
        {"publication_number": family, "id": f"{family}_fr", "language": "fr",
         "context": "Le catalyseur A est une zéolite."},
        {"publication_number": family, "id": f"{family}_en", "language": "en",
         "context": "ENGLISH SIBLING MUST NOT REACH MODELS"},
        {"publication_number": family, "id": f"{family}_de", "language": "de",
         "context": "GERMAN SIBLING MUST NOT REACH MODELS"},
    ])
    corpus.write_rows(context.workspace.corpus_csv("gp"), rows, tuple(rows[0]))
    monkeypatch.setattr(cli, "client_for", lambda model: None)
    generation_packets, grade_packets, decider_packets = [], [], []

    def assert_packet(packet):
        assert "[FR]" in packet and "[EN]" not in packet
        assert "ENGLISH SIBLING" not in packet and "GERMAN SIBLING" not in packet
        assert sum(f"catalyseur {label}" in packet for label in ("A", "B")) == 1

    def generate(client, model, messages, **kwargs):
        assert messages[0]["content"] == PACK.generation("lookup", "fr")
        packet = messages[1]["content"]
        assert_packet(packet)
        generation_packets.append(packet)
        label = "A" if "catalyseur A" in packet else "B"
        answer = f"Le catalyseur {label} est une zéolite."
        return json.dumps([{
            "question": f"Quel catalyseur est utilisé pour la réaction {label} ?",
            "answer": answer, "anchor": "catalyseur", "evidence": answer,
            "evidence_language": "fr", "answer_is_translation": False,
            "question_type": "material_or_composition",
        }])

    def grade(client, settings, prompt, content):
        payload = json.loads(content)
        assert_packet(payload["passages"])
        assert payload["candidates"][0]["question_language"] == "fr"
        assert payload["candidates"][0]["evidence"] in payload["passages"]
        grade_packets.append(payload["passages"])
        return verifier_response(prompt, payload["candidates"])

    def decide(request):
        assert_packet(request["state"])
        decider_packets.append(request["state"])
        return {"answers": {
            mode: {"type": "noul", "noul": float(mode == "lookup")} for mode in MODES
        }}

    monkeypatch.setattr(qagen, "chat", generate)
    monkeypatch.setattr(grading, "_invoke", grade)
    monkeypatch.setattr(llm, "decisions", decide)
    output = tmp_path / "french.csv"
    options = ["--decider-model", "jev"] if use_decider else ["--modes", "lookup"]
    args = parser.parse_args([
        "qac", "generate", "--source", "gp", "--langs", "fr", "--questions", "2",
        "--output", str(output), *options,
    ])
    assert args.handler(args, context) == 0
    generated = corpus.read_rows(output)
    assert {row["source_document_id"] for row in generated} == {f"{family}_fr"}
    assert all(row["corpus_id"] == row["source_document_id"] for row in generated)
    assert all(row["question_language"] == row["context_language"] == row["context_languages"] == "fr"
               for row in generated)
    assert len(corpus.read_rows(tmp_path / "french_best.csv")) == 1
    assert len(generation_packets) == 1 and len(grade_packets) == 2
    assert set(grade_packets) == set(generation_packets)
    assert len(decider_packets) == (1 if use_decider else 0)
    if use_decider:
        assert set(decider_packets) == set(generation_packets)

    regraded = tmp_path / "french_regraded.csv"
    args = parser.parse_args([
        "qac", "regrade", "--source", "gp", "--input", str(output), "--output", str(regraded),
    ])
    assert args.handler(args, context) == 0
    assert len(grade_packets) == 4
    assert set(grade_packets) == set(generation_packets)
    assert {row["source_document_id"] for row in corpus.read_rows(regraded)} == {
        f"{family}_fr",
    }
    assert len(corpus.read_rows(tmp_path / "french_regraded_best.csv")) == 1


def test_exact_document_selection_keeps_same_language_siblings_separate(monkeypatch):
    family = "EP-1234567-A1"
    rows = [{"publication_number": family, "id": f"fr-{label}", "language": "fr",
             "context": f"Le catalyseur {label} est une zéolite."} for label in ("A", "B")]

    def generate(client, model, messages, **kwargs):
        assert messages[0]["content"] == PACK.generation("lookup", "fr")
        packet = messages[1]["content"]
        assert sum(row["context"] in packet for row in rows) == 1
        label = "A" if rows[0]["context"] in packet else "B"
        return json.dumps([{
            "question": f"Quel catalyseur est utilisé pour la réaction {label} ?",
            "answer": f"Le catalyseur {label} est une zéolite.", "anchor": "catalyseur",
            "evidence": f"Le catalyseur {label} est une zéolite.", "evidence_language": "fr",
            "answer_is_translation": False, "question_type": "material_or_composition",
        }])

    def grade(client, settings, prompt, content):
        payload = json.loads(content)
        assert sum(row["context"] in payload["passages"] for row in rows) == 1
        assert payload["candidates"][0]["evidence"] in payload["passages"]
        return verifier_response(prompt, payload["candidates"])

    monkeypatch.setattr(qagen, "chat", generate)
    monkeypatch.setattr(grading, "_invoke", grade)
    results = []
    for row in rows:
        results.extend(qagen.generate_for_document(
            qagen.PlanItem(family, "fr", "lookup", document_id=row["id"]), rows,
            schema=SCHEMA, prompts=PACK, config=config(), generation_client=None, grading_client=None,
        ))
    assert {row["source_document_id"] for row in results} == {"fr-A", "fr-B"}
    assert all(row["source_document_id"] == row["corpus_id"] for row in results)
    assert len(qagen.select_best(results, SCHEMA)) == 2
