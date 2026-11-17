"""
A quality rubric and the code that sums its scores must agree on the criteria.

They drifted twice. ``quality_overall`` sums ``quality_keys(mode)`` and
``_normalize_quality`` defaults a missing criterion to 1, so a rubric edited to
score different criteria raises nothing -- every candidate simply collects the
same near-floor quality score and the ranking silently degrades to faithfulness
alone. It is invisible in the CSV, which only carries ``qual_overall``.

Worse, a mode NAME is not unique across packs: `lookup` scores different
criteria in prompts_eurlex than in prompts_un, and UN's `practitioners` shares
none of the technical five. So the mode name cannot be the source of truth --
the rubric is, via ``rubric_keys``.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from clir_bench.core import grading
from clir_bench.core.grading import (
    _normalize_quality,
    grade_columns,
    quality_keys,
    rubric_keys,
)
from clir_bench.core.prompts import PromptPack
from clir_bench.domains.legal.qac import eurlex_batch, un_batch
from clir_bench.domains.legal.qac.generation_metadata import NEW_MODES

PACKS = {
    "un": (PromptPack("clir_bench.domains.legal.qac.prompts_un"),
           ("lookup", "practitioners", "conceptual", "semantic", "technical", "descriptive", *NEW_MODES)),
    "eurlex": (PromptPack("clir_bench.domains.legal.qac.prompts_eurlex"),
               ("lookup", "fact_pattern", "conceptual", *NEW_MODES)),
}
CASES = [(p, m) for p, (_, modes) in PACKS.items() for m in modes]


@pytest.mark.parametrize("pack,mode", CASES)
def test_every_rubric_declares_five_scored_criteria(pack: str, mode: str) -> None:
    keys = rubric_keys(PACKS[pack][0].quality(mode, "batch"))
    assert len(keys) == 5, (pack, mode, keys)
    assert len(set(keys)) == 5


@pytest.mark.parametrize("pack,mode", CASES)
def test_the_grade_is_summed_over_the_criteria_the_rubric_scores(pack: str, mode: str) -> None:
    """The regression: a perfect response must score 25, not silently less."""
    rubric = PACKS[pack][0].quality(mode, "batch")
    keys = rubric_keys(rubric)
    perfect = {k: 5 for k in keys}
    row = _normalize_quality(perfect, mode, keys)
    assert row["overall"] == 25, (pack, mode, row)


def test_the_same_mode_name_scores_differently_in_the_two_packs() -> None:
    """Why the mode name cannot be the source of truth."""
    un = rubric_keys(PACKS["un"][0].quality("lookup", "batch"))
    eurlex = rubric_keys(PACKS["eurlex"][0].quality("lookup", "batch"))
    assert un != eurlex
    assert "consequence" in un and "consequence" not in eurlex
    assert "focus" in eurlex and "focus" not in un


@pytest.mark.parametrize("mode", PACKS["un"][1])
def test_un_csv_has_a_column_for_every_criterion_its_rubrics_score(mode: str) -> None:
    """``FIELDS`` is a fixed tuple and DictWriter raises on an unknown key, so a
    rubric criterion with no column fails the write outright."""
    for key in rubric_keys(PACKS["un"][0].quality(mode, "batch")):
        assert f"qual_{key}" in un_batch.FIELDS, (mode, key)


@pytest.mark.parametrize("pack,mode", CASES)
def test_grade_columns_emits_the_rubric_criteria_not_the_mode_default(pack: str, mode: str) -> None:
    keys = rubric_keys(PACKS[pack][0].quality(mode, "batch"))
    quality = _normalize_quality({k: 4 for k in keys}, mode, keys)
    row = grade_columns({"grounding": 5, "precision": 5, "numerical_fidelity": 5},
                        quality, mode)
    for key in keys:
        assert row[f"qual_{key}"] == 4
    # nothing from the mode-name default leaks in when the rubric disagrees
    for stale in set(quality_keys(mode)) - set(keys):
        assert f"qual_{stale}" not in row


def test_both_batch_drivers_default_to_the_same_generator() -> None:
    assert un_batch.DEFAULT_GEN_MODEL == eurlex_batch.DEFAULT_GEN_MODEL == "auto"


LEGAL_CASES = [("eurlex", "lookup"), ("eurlex", "fact_pattern"), ("eurlex", "conceptual"),
               ("un", "lookup"), ("un", "practitioners"), ("un", "semantic"), ("un", "conceptual"),
               *((source, mode) for source in ("eurlex", "un") for mode in NEW_MODES)]


def _compact_example(pack="eurlex", mode="lookup", count=1):
    prompt = PACKS[pack][0].quality(mode, "batch")
    example = deepcopy(grading._quality_output_example(prompt))
    template = example["candidates"][0]
    # Set the passing fixture explicitly; instructional examples may illustrate failures.
    template["scores"] = dict.fromkeys(template["scores"], 4)
    template["checks"] = dict.fromkeys(("mode", "support", "metadata"), "pass")
    template["problems"] = []
    candidates = [{"candidate_id": f"candidate-{index}", "question_language": "en",
                   "question": f"Which coverage applies to traveller {index}?", "answer": "EUR 30,000",
                   "question_type": "amount_or_threshold", "anchor": "coverage",
                   "anchors": ["coverage", "traveller"], "particulars": ["traveller", "coverage"],
                   "question_cited": "Which coverage applies under Regulation 1/2000?",
                   "instrument_short_name": None, "articles_involved": ["14"],
                   "framing": "response", "question_template": "Which coverage applies to {instrument}?",
                   "instrument_slot_base": "travel insurance", "instrument_slot_cited": "Regulation 1/2000",
                   "instrument_description": "rules governing travel insurance",
                   "generator_model_id": "hidden-model", "total_score": 40}
                  for index in range(count)]
    example["candidates"] = [dict(deepcopy(template), index=index, candidate_id=candidate["candidate_id"])
                             for index, candidate in enumerate(candidates)]
    example["batch_diversity"] = "not_applicable" if count == 1 else "pass"
    return prompt, candidates, example


def _grade_compact(monkeypatch, response, *, pack="eurlex", mode="lookup", count=1):
    prompt, candidates, _ = _compact_example(pack, mode, count)
    monkeypatch.setattr(grading, "_invoke", lambda *args: json.dumps(response))
    return grading.grade_quality(None, grading.GraderConfig("test"), prompt,
                                 "TARGET ARTICLE\nMinimum coverage is EUR 30,000.", candidates, mode, strict=True)


@pytest.mark.parametrize("pack,mode", LEGAL_CASES)
@pytest.mark.parametrize("count", [1, 2, 3])
def test_current_legal_contract_grades_each_pack_and_exports_audits(monkeypatch, pack, mode, count):
    prompt, _, response = _compact_example(pack, mode, count)
    rows = _grade_compact(monkeypatch, response, pack=pack, mode=mode, count=count)
    assert len(rows) == count
    for index, row in enumerate(rows):
        assert row["overall"] == 20
        assert row["_keys"] == list(response["candidates"][index]["scores"])
        columns = grade_columns({"overall": 15}, row, mode)
        assert columns["total_score"] == 35
        assert columns["quality_audit_status"] == "clear"
        assert columns["quality_rubric_version"] == "unversioned_compact"
        assert columns["quality_batch_diversity"] == response["batch_diversity"]
        assert json.loads(columns["quality_verifier_response_json"]) == response["candidates"][index]
        for field in ("score_notes", "checks", "problems"):
            assert json.loads(columns[f"quality_{field}_json"]) == response["candidates"][index][field]
        for key in rubric_keys(prompt):
            assert columns[f"qual_{key}"] == 4


@pytest.mark.parametrize("pack,mode", LEGAL_CASES)
def test_compact_input_keeps_every_declared_field_and_hides_model_scores(monkeypatch, pack, mode):
    prompt, candidates, response = _compact_example(pack, mode, 3)
    seen = []

    def invoke(client, config, system_prompt, content):
        assert system_prompt == prompt
        seen.append(json.loads(content))
        return json.dumps(response)

    monkeypatch.setattr(grading, "_invoke", invoke)
    grading.grade_quality(None, grading.GraderConfig("test"), prompt, "TARGET plus referenced English text",
                          candidates, mode, strict=True)
    assert seen[0]["passages"] == "TARGET plus referenced English text"
    assert seen[0]["candidates"] == [{key: value for key, value in candidate.items()
                                     if key not in ("generator_model_id", "total_score")}
                                    for candidate in candidates]


@pytest.mark.parametrize("mutation", [
    lambda data: data["candidates"].clear(),
    lambda data: data["candidates"].reverse(),
    lambda data: data["candidates"][0].update(index=True),
    lambda data: data["candidates"][0].update(candidate_id="wrong-id"),
    lambda data: data["candidates"][0].update(candidate_id=None),
    lambda data: data["candidates"][0]["scores"].update(focus=True),
    lambda data: data["candidates"][0]["scores"].update(focus=6),
    lambda data: data["candidates"][0]["scores"].update(focus=0),
    lambda data: data["candidates"][0]["scores"].update(focus="4"),
    lambda data: data["candidates"][0]["scores"].pop("focus"),
    lambda data: data["candidates"][0]["score_notes"].update(focus=""),
    lambda data: data["candidates"][0]["checks"].update(mode="ok"),
    lambda data: data["candidates"][0]["checks"].update(support="uncertain"),
    lambda data: data["candidates"][0].update(problems="none"),
    lambda data: data["candidates"][0].update(problems=[None]),
    lambda data: data.update(batch_diversity="not_applicable"),
])
def test_compact_rejects_malformed_scores_identity_and_audits(monkeypatch, mutation):
    _, _, response = _compact_example(count=2)
    mutation(response)
    with pytest.raises(ValueError):
        _grade_compact(monkeypatch, response, count=2)


@pytest.mark.parametrize("changes,problems,status", [
    ({"mode": "fail"}, ["mode: The question is not in English."], "blocking"),
    ({"support": "fail"}, ["support: The span omits an exception."], "blocking"),
    ({"metadata": "fail"}, ["metadata: The reported anchor is missing."], "repair"),
    ({"support": "uncertain"}, ["support: A necessary reference is missing."], "review"),
    ({}, ["focus: The wording includes a minor redundancy."], "minor_issues"),
])
def test_compact_audit_checks_do_not_change_scores(monkeypatch, changes, problems, status):
    _, _, response = _compact_example()
    response["candidates"][0]["checks"].update(changes)
    response["candidates"][0]["problems"] = problems
    row = _grade_compact(monkeypatch, response)[0]
    columns = grade_columns({"overall": 15}, row, "lookup")
    assert columns["quality_audit_status"] == status
    assert columns["total_score"] == 35


def test_compact_null_score_is_preserved_without_a_fabricated_total(monkeypatch):
    _, _, response = _compact_example()
    response["candidates"][0]["scores"]["linguistic_quality"] = None
    response["candidates"][0]["score_notes"]["linguistic_quality"] = "The question's language cannot be assessed."
    row = _grade_compact(monkeypatch, response)[0]
    columns = grade_columns({"overall": 15}, row, "lookup")
    assert columns["qual_linguistic_quality"] is None
    assert columns["qual_overall"] is None
    assert columns["total_score"] is None
    assert columns["quality_audit_status"] == "review"


def test_compact_diversity_failure_is_independent_of_individual_scores(monkeypatch):
    _, _, response = _compact_example(count=3)
    response["batch_diversity"] = "fail"
    for row in _grade_compact(monkeypatch, response, count=3):
        columns = grade_columns({"overall": 15}, row, "lookup")
        assert columns["quality_batch_diversity"] == "fail"
        assert columns["quality_audit_status"] == "clear"
        assert columns["total_score"] == 35


def test_compact_optional_candidate_id_can_remain_null(monkeypatch):
    prompt, candidates, response = _compact_example()
    candidates[0].pop("candidate_id")
    response["candidates"][0]["candidate_id"] = None
    monkeypatch.setattr(grading, "_invoke", lambda *args: json.dumps(response))
    result = grading.grade_quality(None, grading.GraderConfig("test"), prompt, "source", candidates, "lookup")
    assert result[0]["_response"]["candidate_id"] is None


def _legal_example(pack="eurlex", mode="lookup", version=None):
    # Keep historical parser coverage independent of editable production prompts.
    declared_version = version or "legal-qg-v3.0"
    keys = rubric_keys(PACKS[pack][0].quality(mode, "batch"))
    rubric_mode = f"{pack}_{'practitioner' if mode == 'practitioners' else mode}"
    metadata = ["rendering", "instrument_short_name", "anchor", "articles_involved"]
    declaration = (f"object with exactly {', '.join(keys)}. Each value is an integer 1–5 or null."
                   if declared_version == "legal-qg-v2.0" else
                   f"exactly {', '.join(keys)}, each an integer 1–5 or null.")
    prompt = (f"{declared_version}\n# MODE: {rubric_mode}\n- scores: {declaration}\n"
              f"Required metadata_checks: {', '.join(metadata)}.\n"
              "# CONTROLLED ISSUE CODES\ninput_incomplete, metadata_mismatch\n")
    candidate = {
        "candidate_id": "candidate-001", "question_language": "en",
        "question": "What is the minimum coverage?", "answer": "EUR 30,000",
        "anchor": "minimum coverage", "generator_model_id": "must-not-leak",
        "total_score": 39,
    }
    sources = [{"source_id": "target-1", "role": "target",
                "text": "Minimum insurance coverage is EUR 30,000.",
                "metadata": {"document_id": "document-1"}}]
    audit = {
        "index": 0, "candidate_id": candidate["candidate_id"],
        "rubric_version": declared_version, "mode": rubric_mode,
        "evidence": [{"evidence_id": "e1", "source_id": "target-1", "quote": sources[0]["text"]}],
        "target_relation": "central", "reasoning_requirement": "extraction",
        "validity": {key: {"status": "pass", "evidence_ids": ["e1"], "note": "Supported by the supplied target."}
                     for key in grading._VALIDITY_KEYS},
        "scores": {key: 5 for key in keys},
        "score_notes": {key: "Basis: minimum-coverage lookup; Test: asks for a single coverage value; Limit: none"
                        for key in keys},
        "metadata_checks": [{"field": field, "status": "ok", "note": "Consistent with the candidate."} for field in metadata],
        "issues": [], "suggested_repair": None, "suggested_mode": None, "confidence": "high",
    }
    return prompt, candidate, sources, audit


def _grade_legal(monkeypatch, response, *, pack="eurlex", mode="lookup", version=None, **kwargs):
    prompt, candidate, sources, _ = _legal_example(pack, mode, version)
    monkeypatch.setattr(grading, "_invoke", lambda *args: json.dumps(response))
    return grading.grade_quality(None, grading.GraderConfig("test"), prompt, "unused legacy text",
                                 [candidate], mode, sources=sources,
                                 target_source_id="target-1", **kwargs)


@pytest.mark.parametrize("pack,mode", LEGAL_CASES)
def test_historical_legal_contracts_preserve_audits(monkeypatch, pack, mode):
    _, _, _, audit = _legal_example(pack, mode)
    rows = _grade_legal(monkeypatch, [audit], pack=pack, mode=mode)
    assert rows[0]["overall"] == 25
    assert rows[0]["_response"] == audit
    columns = grade_columns({"grounding": 5, "precision": 4, "numerical_fidelity": 3}, rows[0], mode)
    assert columns["total_score"] == 37
    assert columns["quality_audit_status"] == "clear"
    assert columns["quality_rubric_version"] == audit["rubric_version"]
    assert columns["quality_target_relevance_status"] == "pass"
    assert json.loads(columns["quality_verifier_response_json"]) == audit


def test_legal_input_is_structured_and_model_blind(monkeypatch):
    prompt, candidate, sources, audit = _legal_example()
    captured = []

    def invoke(client, config, system_prompt, user_content):
        captured.append(json.loads(user_content))
        return json.dumps([audit])

    monkeypatch.setattr(grading, "_invoke", invoke)
    grading.grade_quality(None, grading.GraderConfig("test"), prompt, "ignored",
                          [candidate], "lookup", sources=sources, target_source_id="target-1",
                          policies={"target_granularity": "article", "answer_role": "evidence_span"})
    envelope = captured[0]
    assert envelope["sources"] == sources
    assert envelope["mode"] == "eurlex_lookup"
    assert envelope["target_granularity"] == "article"
    assert envelope["candidates"][0]["candidate_id"] == candidate["candidate_id"]
    assert "generator_model_id" not in envelope["candidates"][0]
    assert "total_score" not in envelope["candidates"][0]


@pytest.mark.parametrize("version", ["legal-qg-v2.0", "legal-qg-v3.0"])
@pytest.mark.parametrize("mutation", [
    lambda row: row.update(candidate_id="different"),
    lambda row: row.update(index=True),
    lambda row: row.update(rubric_version="old"),
    lambda row: row.update(mode="un_lookup"),
    lambda row: row.update(target_relation="valid"),
    lambda row: row.update(total_score=40),
    lambda row: row["scores"].pop("focus"),
    lambda row: row["scores"].update(focus=True),
    lambda row: row["scores"].update(focus=6),
    lambda row: row["scores"].update(focus="5"),
    lambda row: row["scores"].update(focus=None),
    lambda row: row["score_notes"].update(focus=""),
    lambda row: row["evidence"][0].update(source_id="invented"),
    lambda row: row["evidence"][0].update(quote="Invented exact quotation"),
    lambda row: row["evidence"].append(deepcopy(row["evidence"][0])),
    lambda row: row["validity"]["target_relevance"].update(evidence_ids=[]),
    lambda row: row["validity"]["question_premises"].update(evidence_ids=["invented"]),
    lambda row: row["validity"]["question_premises"].update(status="clear"),
    lambda row: row["metadata_checks"].pop(),
    lambda row: row["metadata_checks"][1].update(field="rendering"),
    lambda row: row["metadata_checks"][0].update(status="pass"),
    lambda row: row.update(suggested_mode="technical"),
])
def test_legal_rejects_invalid_identity_scores_and_audits(monkeypatch, mutation, version):
    _, _, _, audit = _legal_example(version=version)
    mutation(audit)
    with pytest.raises(ValueError):
        _grade_legal(monkeypatch, [audit], version=version)


def test_legal_nulls_remain_unranked_and_repairs_remain_audit_data(monkeypatch):
    _, candidate, _, audit = _legal_example()
    audit["scores"]["focus"] = None
    audit["issues"] = [{"code": "input_incomplete", "layer": "input", "severity": "review",
                        "affected_criteria": ["focus"], "evidence_ids": [], "note": "Additional context is needed."}]
    audit["suggested_repair"] = "Suggested wording, without altering the stored question."
    quality = _grade_legal(monkeypatch, [audit])[0]
    columns = grade_columns({"overall": 15}, quality, "lookup")
    assert columns["qual_focus"] is None
    assert columns["qual_overall"] is None
    assert columns["total_score"] is None
    assert columns["quality_audit_status"] == "review"
    assert candidate["question"] == "What is the minimum coverage?"
    ranked = grading.rank_candidates([candidate, {"question": "complete"}],
                                     [{"overall": 15}, {"overall": 10}],
                                     [quality, {"overall": 20}], "lookup")
    assert ranked[0].qa["question"] == "complete"


def test_legal_requires_explicit_inputs_before_calling_verifier(monkeypatch):
    prompt, candidate, sources, _ = _legal_example()
    monkeypatch.setattr(grading, "_invoke", lambda *args: pytest.fail("Invalid input must fail before API access"))
    config = grading.GraderConfig("test")
    with pytest.raises(ValueError, match="labeled sources"):
        grading.grade_quality(None, config, prompt, "flat passage", [candidate], "lookup")
    with pytest.raises(ValueError, match="candidate_id"):
        grading.grade_quality(None, config, prompt, "", [{"question": "q", "answer": "a"}], "lookup",
                              sources=sources, target_source_id="target-1")
    with pytest.raises(ValueError, match="cannot override"):
        grading.grade_quality(None, config, prompt, "", [candidate], "lookup",
                              sources=sources, target_source_id="target-1", policies={"mode": "injected"})


@pytest.mark.parametrize("version", ["legal-qg-v2.0", "legal-qg-v3.0"])
def test_declared_legal_version_routes_and_flattens_without_legacy_fallback(monkeypatch, version):
    prompt, _, _, audit = _legal_example(version=version)
    assert len(rubric_keys(prompt)) == 5
    quality = _grade_legal(monkeypatch, [audit], version=version)[0]
    columns = grade_columns({"overall": 15}, quality, "lookup")
    assert columns["total_score"] == 40
    assert columns["quality_rubric_version"] == version
    assert columns["quality_audit_status"] == "clear"
    audit["rubric_version"] = "legal-qg-v2.0" if version == "legal-qg-v3.0" else "legal-qg-v3.0"
    with pytest.raises(ValueError, match="wrong rubric version"):
        _grade_legal(monkeypatch, [audit], version=version)


@pytest.mark.parametrize("version", ["legal-qg-v9.0", "legal-qg-v2.0 legal-qg-v3.0"])
def test_unsupported_or_conflicting_legal_versions_fail_before_model_call(monkeypatch, version):
    prompt, candidate, sources, _ = _legal_example()
    prompt = prompt.replace("legal-qg-v3.0", version)
    monkeypatch.setattr(grading, "_invoke", lambda *args: pytest.fail("Invalid version contacted verifier"))
    with pytest.raises(ValueError, match="unsupported versions"):
        grading.grade_quality(None, grading.GraderConfig("test"), prompt, "", [candidate], "lookup",
                              sources=sources, target_source_id="target-1")


@pytest.mark.parametrize("note", [
    "The answer is clear.",
    "Basis: ; Test: one value; Limit: none",
    "Basis: coverage; Test: ; Limit: none",
    "Basis: coverage; Test: one value; Limit: ",
    "Basis: . Test: one value. Limit: none",
    "Basis: coverage. Test: . Limit: none",
    "Basis: coverage. Test: one value. Limit: ",
    "Test: one value. Basis: coverage. Limit: none",
])
def test_legal_v3_score_notes_require_three_nonempty_audit_fields(monkeypatch, note):
    _, _, _, audit = _legal_example()
    audit["score_notes"]["focus"] = note
    with pytest.raises(ValueError, match="nonempty Basis, Test, and Limit"):
        _grade_legal(monkeypatch, [audit])


@pytest.mark.parametrize("separator", ["; ", ". ", ".\n"])
def test_legal_v3_score_notes_accept_sentence_or_semicolon_separators(monkeypatch, separator):
    _, _, _, audit = _legal_example()
    note = separator.join(("Basis: coverage lookup", "Test: asks for one coverage value", "Limit: none"))
    audit["score_notes"]["focus"] = note
    quality = _grade_legal(monkeypatch, [audit])[0]
    assert quality["_response"]["score_notes"]["focus"] == note
    assert quality["overall"] == 25


@pytest.mark.parametrize("version", ["legal-qg-v2.0", "legal-qg-v3.0"])
@pytest.mark.parametrize("defect", ["evidence_count", "issue_code", "score_note"])
def test_v3_only_audit_constraints_preserve_v2_compatibility(monkeypatch, version, defect):
    _, _, _, audit = _legal_example(version=version)
    if defect == "evidence_count":
        audit["evidence"] = [dict(audit["evidence"][0], evidence_id=f"e{i}") for i in range(1, 12)]
    elif defect == "issue_code":
        audit["issues"] = [{"code": "arbitrary_legacy_code", "layer": "quality", "severity": "minor",
                            "affected_criteria": ["focus"], "evidence_ids": [], "note": "A legacy diagnostic."}]
    else:
        audit["score_notes"]["focus"] = "The requested rule is precisely delimited."
    if version == "legal-qg-v3.0":
        with pytest.raises(ValueError):
            _grade_legal(monkeypatch, [audit], version=version)
    else:
        assert _grade_legal(monkeypatch, [audit], version=version)[0]["overall"] == 25


def test_strict_legacy_grades_reject_missing_scores_and_keep_default_compatibility(monkeypatch):
    monkeypatch.setattr(grading, "_invoke", lambda *args: '[]')
    config = grading.GraderConfig("test")
    pairs = [{"question": "q", "answer": "a"}]
    with pytest.raises(ValueError, match="exactly 1"):
        grading.grade_faithfulness(None, config, "legacy", "source", pairs, strict=True)
    with pytest.raises(ValueError, match="exactly 1"):
        grading.grade_quality(None, config, "legacy", "source", pairs, "technical", strict=True)
    assert grading.grade_faithfulness(None, config, "legacy", "source", pairs)[0]["overall"] == 3
    assert grading.grade_quality(None, config, "legacy", "source", pairs, "technical")[0]["overall"] == 5


def test_strict_legacy_grades_keep_raw_response_and_candidate_order(monkeypatch):
    response = [{"index": 1, "grounding": 3, "precision": 3, "numerical_fidelity": 3, "reason": "Second."},
                {"index": 0, "grounding": 5, "precision": 5, "numerical_fidelity": 5, "reason": "First."}]
    monkeypatch.setattr(grading, "_invoke", lambda *args: json.dumps(response))
    rows = grading.grade_faithfulness(None, grading.GraderConfig("test"), "legacy", "source",
                                     [{"question": "q1"}, {"question": "q2"}], strict=True)
    assert [row["overall"] for row in rows] == [15, 9]
    assert rows[0]["_response"] == response[1]
