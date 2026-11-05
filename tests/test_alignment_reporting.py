"""Evidence approval and usage accounting must fail closed."""

import json

import pytest

from clir_bench.alignment.report import (
    REVIEW_FIELDS,
    actual_cost,
    assess,
    cost_usage,
    estimate_cost,
    report,
    result_hash,
)
from clir_bench.alignment.runner import request_hash


def task(identifier="t", corpus="eurlex"):
    return {
        "task_id": identifier, "corpus": corpus, "source_unit_id": "article:en",
        "target_unit_id": "article:fr", "source_language": "en", "target_language": "fr",
        "source_text": "The rule applies unless exempt.",
        "target_text": "La règle s’applique sauf exemption.", "genre": "article",
        "anchors": [{"anchor_id": "a", "text": "The rule applies unless exempt.", "qa_ids": ["q1", "q2"]}],
    }


def result(t):
    return {
        "alignment_id": t["task_id"] + ":a", "task_id": t["task_id"],
        "anchor_id": "a", "corpus": t["corpus"], "source_unit_id": t["source_unit_id"],
        "target_unit_id": t["target_unit_id"], "source_language": "en", "target_language": "fr",
        "source_text": t["anchors"][0]["text"], "target_text": t["target_text"],
        "status": "matched", "exact_match": True, "first_status": "matched",
        "first_exact_match": True,
    }


def review(r, **decisions):
    return {
        "alignment_id": r["alignment_id"], "result_hash": result_hash(r),
        **dict.fromkeys(REVIEW_FIELDS, True), "reviewer": "reviewer-1", "notes": "Checked conditions.",
        **decisions,
    }


def test_exact_quotation_needs_semantic_review():
    t = task()
    assessment, queue = assess([t], [result(t)], [])
    assert assessment["overall"]["gate"] == "pending"
    assert assessment["overall"]["valid_matches"] == 0
    assert queue[0]["result_hash"] == result_hash(result(t))


def test_changed_quote_cannot_inherit_earlier_approval():
    t = task()
    r = result(t)
    approved = review(r)
    r["target_text"] = "La règle s’applique"
    assessment, queue = assess([t], [r], [approved])
    assert assessment["overall"]["reviews"]["stale"] == 1
    assert assessment["overall"]["gate"] == "pending"
    assert queue[0]["result_hash"] != approved["result_hash"]


def test_exact_flag_cannot_rescue_normalized_or_translated_quote():
    t = task()
    r = result(t)
    r["target_text"] = "La règle s’applique  sauf exemption."
    assessment, _ = assess([t], [r], [review(r)])
    assert assessment["overall"]["final_exact_matches"] == 0
    assert assessment["overall"]["gate"] == "failed"


def test_whitespace_only_and_invalid_first_attempts_are_not_exact_matches():
    t = task()
    r = result(t)
    r.update(target_text=" ", first_status="invalid", first_exact_match=True)
    assessment, _ = assess([t], [r], [review(r)])
    assert assessment["overall"]["final_exact_matches"] == 0
    assert assessment["overall"]["first_attempt_exact_matches"] == 0


def test_omitted_exception_fails_review_even_when_quote_is_exact():
    t = task()
    r = result(t)
    r["target_text"] = "La règle s’applique"
    assessment, _ = assess([t], [r], [review(r, qualifications_preserved=False)])
    assert assessment["overall"]["final_exact_matches"] == 1
    assert assessment["overall"]["valid_matches"] == 0
    assert assessment["overall"]["gate"] == "failed"


def test_missing_results_remain_in_coverage_denominator():
    t1, t2 = task("t1"), task("t2")
    r = result(t1)
    assessment, _ = assess([t1, t2], [r], [review(r)])
    assert assessment["overall"]["alignment_tasks"] == 2
    assert assessment["overall"]["valid_coverage"] == 0.5
    assert assessment["overall"]["final_statuses"]["not_run"] == 1
    assert assessment["overall"]["gate"] == "pending"


def test_every_corpus_must_pass_independently():
    t1, t2 = task("t1", "eurlex"), task("t2", "chemistry")
    r1, r2 = result(t1), result(t2)
    r2.update(status="not_found", target_text="", exact_match=False)
    assessment, _ = assess([t1, t2], [r1, r2], [review(r1)])
    assert assessment["by_corpus"]["eurlex"]["gate"] == "passed"
    assert assessment["by_corpus"]["chemistry"]["gate"] == "failed"
    assert assessment["overall"]["gate"] == "failed"


def test_wrong_document_identity_and_duplicate_rows_are_rejected():
    t = task()
    r = result(t)
    with pytest.raises(ValueError, match="duplicate"):
        assess([t], [r, r], [])
    r["target_unit_id"] = "other:fr"
    with pytest.raises(ValueError, match="identity mismatch"):
        assess([t], [r], [])


def test_cache_and_long_context_prices_use_full_request_threshold():
    small = cost_usage({"input_tokens": 272000, "output_tokens": 1000,
                        "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 2000}})
    large = cost_usage({"input_tokens": 272001, "output_tokens": 1000,
                        "input_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 2000}})
    assert small["total_usd"] == pytest.approx((269000 * .1 + 1000 * .01 + 2000 * .125 + 1000 * .5) / 1e6)
    assert large["input_usd"] == pytest.approx((269001 * .1 + 1000 * .01 + 2000 * .125) * 2 / 1e6)
    assert large["output_usd"] == pytest.approx(1000 * .5 * 1.5 / 1e6)
    assert cost_usage({"input_tokens": 10, "output_tokens": 10}, batch=True)["total_usd"] == pytest.approx(3e-6)


def test_retry_usage_counted_and_missing_usage_is_unknown_not_zero():
    usage = {"input_tokens": 100, "output_tokens": 20}
    attempts = [{"task_id": "t", "attempt": i, "response": {"usage": usage}} for i in (1, 2)]
    cost = actual_cost(attempts)
    assert cost["input_tokens"] == 200
    assert cost["total_usd"] == pytest.approx(0.00004)
    attempts.append({"task_id": "t", "attempt": 3, "error": {"type": "Timeout"}})
    cost = actual_cost(attempts)
    assert cost["total_usd"] is None
    assert cost["known_usd"] == pytest.approx(0.00004)
    assert len(cost["attempts_without_usage"]) == 1


def test_provider_cost_is_preferred_and_calculated_cost_remains_visible():
    usage = {"input_tokens": 100, "output_tokens": 20, "cost": .000023}
    cost = actual_cost([{"task_id": "t", "attempt": 1, "response": {"usage": usage}}])
    assert cost["total_usd"] == pytest.approx(.000023)
    assert cost["rate_calculated_usd"] == pytest.approx(.00002)
    assert cost["provider_reported_attempts"] == 1


def test_full_estimate_requires_complete_input_counts():
    assert estimate_cost([task()], [task()], [], [])["status"] == "pending_input_counts"


def test_full_input_count_from_changed_prompt_is_rejected():
    counts = [{"task_id": "t", "model": "gpt-6-luna", "request_hash": "outdated", "input_tokens": 100}]
    with pytest.raises(ValueError, match="different request"):
        estimate_cost([task()], [task()], [], counts)


def test_retry_estimate_does_not_create_false_long_context_surcharge():
    t = task()
    attempts = [{"task_id": "t", "attempt": i, "request_hash": request_hash(t),
                 "response": {"usage": {"input_tokens": 200000, "output_tokens": 100}}}
                for i in (1, 2)]
    counts = [{"task_id": "t", "model": "gpt-6-luna", "request_hash": request_hash(t), "input_tokens": 200000}]
    estimated = estimate_cost([t], [t], attempts, counts)
    assert estimated["standard_usd"] == pytest.approx(.0501)
    assert estimated["batch_usd"] == pytest.approx(.02505)
    assert estimated["conservative_usd"] >= estimated["standard_usd"]


def test_old_requests_count_in_actual_spend_but_not_new_configuration_estimate():
    t = task()
    attempts = [
        {"task_id": "t", "attempt": 1, "request_hash": "old",
         "response": {"usage": {"input_tokens": 1000, "output_tokens": 1000}}},
        {"task_id": "t", "attempt": 1, "request_hash": request_hash(t),
         "response": {"usage": {"input_tokens": 100, "output_tokens": 10}}},
    ]
    counts = [{"task_id": "t", "model": "gpt-6-luna", "request_hash": request_hash(t), "input_tokens": 100}]
    assert actual_cost(attempts)["output_tokens"] == 1010
    assert estimate_cost([t], [t], attempts, counts)["standard_usd"] == pytest.approx(.0000175)


def test_auth_rejections_are_audited_without_poisoning_repaired_run_cost():
    t = task()
    attempts = [
        {"task_id": "t", "attempt": 1, "request_hash": request_hash(t),
         "error": {"type": "AuthenticationError", "status": 401}},
        {"task_id": "t", "attempt": 1, "request_hash": request_hash(t),
         "response": {"usage": {"input_tokens": 100, "output_tokens": 10}}},
    ]
    counts = [{"task_id": "t", "model": "gpt-6-luna", "request_hash": request_hash(t), "input_tokens": 100}]
    actual = actual_cost(attempts)
    assert actual["complete"] is True
    assert len(actual["rejected_before_generation"]) == 1
    assert estimate_cost([t], [t], attempts, counts)["standard_usd"] == pytest.approx(.0000175)


def test_duplicate_calls_after_first_valid_response_do_not_inflate_full_estimate():
    t = task()
    payload = {"source_language_matches": True, "target_language_matches": True,
               "results": [{"anchor_id": "a", "status": "matched", "text": t["target_text"], "reason": ""}]}
    response = {"model": "openai/gpt-6-luna", "status": "completed",
                "output": [{"type": "reasoning", "content": None},
                           {"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}],
                "usage": {"input_tokens": 100, "output_tokens": 10}}
    attempts = [{"task_id": "t", "attempt": i, "request_hash": request_hash(t), "response": response}
                for i in (1, 2, 3)]
    counts = [{"task_id": "t", "model": "openai/gpt-6-luna", "request_hash": request_hash(t), "input_tokens": 100}]
    estimate = estimate_cost([t], [t], attempts, counts)
    assert estimate["excluded_post_success_attempts_from_projection"] == 2
    assert estimate["standard_usd"] == pytest.approx(.0000175)
    assert actual_cost(attempts)["output_tokens"] == 30


def test_calibrated_input_counts_are_labeled_and_get_extra_uncertainty_allowance():
    t = task()
    attempts = [{"task_id": "t", "attempt": 1, "request_hash": request_hash(t),
                 "response": {"usage": {"input_tokens": 100, "output_tokens": 10}}}]
    counts = [{"task_id": "t", "model": "openai/gpt-6-luna", "request_hash": request_hash(t),
               "input_tokens": 100, "method": "calibrated_local_tokenizer"}]
    estimated = estimate_cost([t], [t], attempts, counts)
    assert estimated["input_counts_approximate"] is True
    assert estimated["input_count_methods"] == {"calibrated_local_tokenizer": 1}
    assert estimated["conservative_usd"] == pytest.approx(.0000125 * 1.25 * 1.1 + .000005 * 1.25)


def test_interruption_incident_preserves_recorded_spend_without_claiming_final_total():
    attempts = [{"task_id": "t", "attempt": 1,
                 "response": {"usage": {"input_tokens": 100, "output_tokens": 10, "cost": .00002}}}]
    incidents = [{"type": "interrupted_run", "unrecorded_requests_upper_bound": 3, "billing_unknown": True}]
    cost = actual_cost(attempts, incidents)
    assert cost["recorded_total_usd"] == pytest.approx(.00002)
    assert cost["total_usd"] is None
    assert cost["complete"] is False
    assert cost["recorded_usage_complete"] is True
    assert cost["unrecorded_requests_upper_bound"] == 3
    assert cost["unrecorded_cost_allowance_usd"] > 0


def test_blocked_run_reports_no_measured_result_and_withholds_estimate(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "tasks.jsonl").write_text(json.dumps(task()) + "\n")
    (run_dir / "run.json").write_text(json.dumps({"status": "blocked", "error": "invalid_organization"}))
    output = tmp_path / "report"
    summary = report(run_dir, output)
    assert summary["full_workload_estimate"]["status"] == "withheld"
    assert "No measured pilot cost or quality result" in (output / "report.md").read_text()
    assert summary["assessment"]["overall"]["final_statuses"] == {"not_run": 1}


def test_requested_budget_estimate_does_not_approve_failed_quality(tmp_path):
    from clir_bench.alignment.runner import write_json, write_jsonl

    run_dir = tmp_path / "run"
    t = task()
    r = result(t)
    r["target_text"] = "La règle s’applique"
    write_jsonl(run_dir / "tasks.jsonl", [t])
    write_jsonl(run_dir / "full_tasks.jsonl", [t])
    write_jsonl(run_dir / "results.jsonl", [r])
    write_jsonl(run_dir / "reviews.jsonl", [review(r, qualifications_preserved=False)])
    write_json(run_dir / "run.json", {"status": "completed"})
    write_json(run_dir / "attempts" / "t-1.json", {
        "task_id": "t", "attempt": 1, "request_hash": request_hash(t),
        "response": {"usage": {"input_tokens": 100, "output_tokens": 10}},
    })
    write_jsonl(run_dir / "token_counts.jsonl", [{
        "task_id": "t", "model": "openai/gpt-6-luna",
        "request_hash": request_hash(t), "input_tokens": 100,
    }])
    summary = report(run_dir, tmp_path / "report", estimate_without_quality_gate=True)
    assert summary["assessment"]["overall"]["gate"] == "failed"
    assert summary["full_workload_estimate"]["status"] == "estimated"
    assert summary["full_workload_estimate"]["rollout_approved"] is False
    assert (run_dir / "accepted_alignments.jsonl").read_text() == ""
