"""Local quality reports and usage-based costs for the alignment pilot.

An exact quote is not a semantic review. Unreviewed or stale results cannot pass
the pilot gate, and failed requests with missing usage never become free calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

MODEL = "gpt-6-luna"
PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-6-luna"
REVIEW_FIELDS = (
    "semantic_match", "qualifications_preserved", "minimal", "language_correct",
)
RATES = {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _unique(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError(f"duplicate {label}: {identity}")
        result[identity] = row
    return result


def result_hash(row: dict) -> str:
    """Bind a review to the precise final quotation and its source identity."""
    keys = (
        "alignment_id", "source_unit_id", "target_unit_id", "source_language",
        "target_language", "source_text", "target_text", "status",
    )
    payload = {key: row.get(key) for key in keys}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def cost_usage(usage: dict, *, batch: bool = False) -> dict:
    """Price reported Responses usage in USD, including long-context rates."""
    for key in ("input_tokens", "output_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ValueError(f"missing or invalid usage.{key}")
    tokens = usage["input_tokens"]
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    written = (details.get("cache_write_tokens") or details.get("cache_creation_tokens")
               or usage.get("cache_creation_input_tokens") or 0)
    if any(type(value) is not int or value < 0 for value in (cached, written)):
        raise ValueError("invalid cache token counts")
    if cached + written > tokens:
        raise ValueError("cache tokens exceed total input tokens")
    input_multiplier = 2 if tokens > 272_000 else 1
    output_multiplier = 1.5 if tokens > 272_000 else 1
    tier_multiplier = 0.5 if batch else 1
    input_cost = ((tokens - cached - written) * RATES["input"]
                  + cached * RATES["cached_input"]
                  + written * RATES["cache_write"]) * input_multiplier / 1_000_000
    output_cost = usage["output_tokens"] * RATES["output"] * output_multiplier / 1_000_000
    return {
        "input_tokens": tokens,
        "cached_input_tokens": cached,
        "cache_write_tokens": written,
        "output_tokens": usage["output_tokens"],
        "input_usd": input_cost * tier_multiplier,
        "output_usd": output_cost * tier_multiplier,
        "total_usd": (input_cost + output_cost) * tier_multiplier,
        "long_context": tokens > 272_000,
    }


def actual_cost(attempts: list[dict], incidents: list[dict] | None = None,
                *, max_input_tokens: int | None = None) -> dict:
    """Return known spend and explicitly mark attempts whose billing is unknown."""
    known = []
    unknown = []
    rejected = []
    provider_costs = []
    charged = []
    for attempt in attempts:
        usage = (attempt.get("response") or {}).get("usage")
        if not usage:
            if _rejected_before_generation(attempt):
                rejected.append({"task_id": attempt.get("task_id"), "attempt": attempt.get("attempt"),
                                 "status": attempt["error"]["status"]})
                continue
            unknown.append({"task_id": attempt.get("task_id"), "attempt": attempt.get("attempt")})
            continue
        try:
            calculated = cost_usage(usage)
            known.append(calculated)
        except ValueError:
            unknown.append({"task_id": attempt.get("task_id"), "attempt": attempt.get("attempt")})
            calculated = None
        provider_cost = usage.get("cost")
        if (type(provider_cost) in {int, float} and math.isfinite(provider_cost)
                and provider_cost >= 0):
            provider_costs.append(provider_cost)
            charged.append(provider_cost)
        elif calculated is not None:
            charged.append(calculated["total_usd"])
    billing_incidents = [incident for incident in incidents or [] if incident.get("billing_unknown")]
    unrecorded_count = sum(incident.get("unrecorded_requests_upper_bound", 0)
                          for incident in billing_incidents)
    largest_input = max([row["input_tokens"] for row in known] + [max_input_tokens or 0])
    allowance = None
    if unrecorded_count and largest_input:
        allowance = unrecorded_count * cost_usage({
            "input_tokens": largest_input, "output_tokens": 32768,
            "input_tokens_details": {"cache_write_tokens": largest_input},
        })["total_usd"]
    return {
        "recorded_attempts": len(attempts),
        "attempts_with_usage": len(known),
        "attempts_without_usage": unknown,
        "rejected_before_generation": rejected,
        "recorded_usage_complete": not unknown,
        "complete": not unknown and not billing_incidents,
        "known_usd": sum(charged),
        "recorded_total_usd": sum(charged) if not unknown else None,
        "total_usd": sum(charged) if not unknown and not billing_incidents else None,
        "provider_reported_usd": sum(provider_costs),
        "provider_reported_attempts": len(provider_costs),
        "rate_calculated_usd": sum(row["total_usd"] for row in known),
        "cost_basis": "Provider-reported usage.cost where present; token-rate calculation otherwise.",
        "input_tokens": sum(row["input_tokens"] for row in known),
        "output_tokens": sum(row["output_tokens"] for row in known),
        "cached_input_tokens": sum(row["cached_input_tokens"] for row in known),
        "cache_write_tokens": sum(row["cache_write_tokens"] for row in known),
        "billing_incidents": billing_incidents,
        "unrecorded_requests_upper_bound": unrecorded_count,
        "unrecorded_cost_allowance_usd": allowance,
        "recorded_plus_allowance_usd": sum(charged) + allowance if allowance is not None else None,
        "allowance_assumption": "At most the recorded incident request count, each using the largest "
                                "observed/estimated input, all cache-write input, and the full 32,768 "
                                "output-token limit. This is an allowance, not a verified billing bound.",
    }


def _rejected_before_generation(attempt: dict) -> bool:
    return not attempt.get("response") and (attempt.get("error") or {}).get("status") in {401, 403, 404}


def assess(tasks: list[dict], results: list[dict], reviews: list[dict]) -> tuple[dict, list[dict]]:
    """Validate identity and literal quotes, then apply explicit semantic reviews."""
    result_map = _unique(results, "alignment_id", "alignment result")
    review_map = _unique(reviews, "alignment_id", "semantic review")
    expected = {}
    for task in tasks:
        for anchor in task["anchors"]:
            identity = task["task_id"] + ":" + anchor["anchor_id"]
            if identity in expected:
                raise ValueError(f"duplicate task anchor: {identity}")
            expected[identity] = (task, anchor)
    if result_map.keys() - expected.keys():
        raise ValueError("results include alignments outside the frozen pilot sample")
    if review_map.keys() - result_map.keys():
        raise ValueError("reviews include alignments with no saved result")

    checked = []
    queue = []
    for identity, (task, anchor) in expected.items():
        row = result_map.get(identity)
        if row is None:
            checked.append({"corpus": task["corpus"], "target_language": task["target_language"],
                            "status": "not_run", "first_status": "not_run", "exact": False,
                            "first_exact": False, "proposed": False, "review": "not_applicable",
                            "valid": False})
            continue
        for key in ("task_id", "source_unit_id", "target_unit_id", "source_language",
                    "target_language", "corpus"):
            if row.get(key) != task[key]:
                raise ValueError(f"result identity mismatch: {identity} ({key})")
        if row.get("source_text") != anchor["text"]:
            raise ValueError(f"source passage changed: {identity}")
        quote = row.get("target_text") or ""
        exact = isinstance(quote, str) and bool(quote.strip()) and quote in task["target_text"]
        status = row.get("status", "failed")
        proposed = status in {"matched", "invalid"} and bool(quote)
        review_state = "not_applicable"
        review = review_map.get(identity)
        if status == "matched" and exact:
            if review is None:
                review_state = "pending"
            elif review.get("result_hash") != result_hash(row):
                review_state = "stale"
            elif (not review.get("reviewer")
                  or any(type(review.get(key)) is not bool for key in REVIEW_FIELDS)):
                review_state = "pending"
            else:
                review_state = "pass" if all(review[key] for key in REVIEW_FIELDS) else "fail"
            if review_state in {"pending", "stale"}:
                queue.append({
                    "alignment_id": identity, "result_hash": result_hash(row),
                    "corpus": task["corpus"], "source_language": task["source_language"],
                    "target_language": task["target_language"],
                    "source_text": anchor["text"], "target_text": quote,
                    **{key: None for key in REVIEW_FIELDS}, "reviewer": "", "notes": "",
                })
        checked.append({
            "corpus": task["corpus"], "target_language": task["target_language"],
            "status": status, "first_status": row.get("first_status", "unknown"),
            "exact": exact and status == "matched",
            "first_exact": row.get("first_status") == "matched" and row.get("first_exact_match") is True,
            "proposed": proposed, "review": review_state,
            "valid": status == "matched" and exact and review_state == "pass",
        })

    def summarize(rows: list[dict]) -> dict:
        total = len(rows)
        proposed = sum(row["proposed"] for row in rows)
        valid = sum(row["valid"] for row in rows)
        pending = sum(row["review"] in {"pending", "stale"} for row in rows)
        statuses = Counter(row["status"] for row in rows)
        precision = valid / proposed if proposed else None
        coverage = valid / total if total else 0
        if statuses.get("not_run") or pending:
            gate = "pending"
        elif proposed and precision >= 0.95 and coverage >= 0.90:
            gate = "passed"
        else:
            gate = "failed"
        return {
            "alignment_tasks": total, "first_attempt_statuses": dict(Counter(row["first_status"] for row in rows)),
            "first_attempt_exact_matches": sum(row["first_exact"] for row in rows),
            "final_statuses": dict(statuses), "final_exact_matches": sum(row["exact"] for row in rows),
            "proposed_matches": proposed, "reviews": dict(Counter(row["review"] for row in rows)),
            "pending_reviews": pending, "valid_matches": valid,
            "semantic_precision": precision, "valid_coverage": coverage, "gate": gate,
        }

    corpora = sorted({row["corpus"] for row in checked})
    by_corpus = {corpus: summarize([row for row in checked if row["corpus"] == corpus])
                 for corpus in corpora}
    by_language = {corpus: {
        language: summarize([row for row in checked
                             if row["corpus"] == corpus and row["target_language"] == language])
        for language in sorted({row["target_language"] for row in checked if row["corpus"] == corpus})
    } for corpus in corpora}
    overall = summarize(checked)
    gates = [row["gate"] for row in by_corpus.values()]
    overall["gate"] = ("failed" if "failed" in gates else
                       "pending" if "pending" in gates or not gates else "passed")
    return {"overall": overall, "by_corpus": by_corpus, "by_target_language": by_language}, queue


def estimate_cost(full_tasks: list[dict], pilot_tasks: list[dict], attempts: list[dict],
                  token_counts: list[dict]) -> dict:
    """Extrapolate measured outputs by task strata, using exact full-input counts."""
    from .runner import request_hash, validate_response

    if not full_tasks:
        return {"status": "pending_inventory", "reason": "Full-workload tasks are missing."}
    task_map = _unique(pilot_tasks, "task_id", "pilot task")
    counts = _unique(token_counts, "task_id", "input token count")
    missing = [task["task_id"] for task in full_tasks if task["task_id"] not in counts]
    if missing:
        return {"status": "pending_input_counts", "missing_groups": len(missing)}
    for task in full_tasks:
        count = counts[task["task_id"]]
        if count.get("model") not in {MODEL, "openai/" + MODEL} or type(count.get("input_tokens")) is not int:
            raise ValueError("invalid full-workload token count")
        if count.get("request_hash") != request_hash(task):
            raise ValueError("input token count belongs to a different request")
    measured = defaultdict(list)
    attempts_by_task = defaultdict(list)
    excluded_post_success_attempts = 0
    for attempt in attempts:
        if attempt.get("task_id") not in task_map:
            raise ValueError("usage belongs to a request outside the pilot")
        task = task_map[attempt["task_id"]]
        # Old configurations remain in actual spending but cannot supply the
        # output/retry distribution for the currently counted request payload.
        if attempt.get("request_hash") != request_hash(task):
            continue
        if _rejected_before_generation(attempt):
            continue
        attempts_by_task[attempt["task_id"]].append(attempt)
    for task_id, task_attempts in attempts_by_task.items():
        task = task_map[task_id]
        ordered = sorted(task_attempts, key=lambda row: (row.get("attempt", 0), row.get("started_at", "")))
        effective = []
        for attempt in ordered[:3]:
            effective.append(attempt)
            if attempt.get("response"):
                rows, issues = validate_response(task, attempt["response"])
                if not issues or any(row["status"] == "language_mismatch" for row in rows):
                    break
        excluded_post_success_attempts += len(ordered) - len(effective)
        task_attempts = effective
        # Sum every attempt's output, including retries, rather than only the winning response.
        usages = [(attempt.get("response") or {}).get("usage") for attempt in task_attempts]
        if any(not usage or type(usage.get("output_tokens")) is not int for usage in usages):
            return {"status": "pending_usage", "reason": "Some pilot attempts lack output usage."}
        per_anchor = sum(usage["output_tokens"] for usage in usages) / len(task["anchors"])
        for stratum in _strata(task):
            measured[stratum].append((per_anchor, len(task_attempts)))
    by_corpus = defaultdict(lambda: {
        "request_groups": 0, "alignment_tasks": 0, "input_tokens": 0,
        "estimated_output_tokens": 0.0, "standard_usd": 0.0, "conservative_usd": 0.0,
        "stratum_fallbacks": Counter(),
    })
    for task in full_tasks:
        chosen = next((stratum for stratum in _strata(task) if measured[stratum]), None)
        if chosen is None:
            return {"status": "pending_usage", "reason": f"No pilot usage for {task['corpus']}."}
        observations = measured[chosen]
        output = mean(row[0] for row in observations) * len(task["anchors"])
        retry_multiplier = mean(row[1] for row in observations)
        base_input = counts[task["task_id"]]["input_tokens"]
        # Price each original-size request so retry aggregation cannot trigger a
        # long-context surcharge on a request that was actually short-context.
        base = cost_usage({"input_tokens": base_input, "output_tokens": 0,
                           "input_tokens_details": {"cache_write_tokens": base_input}})["total_usd"]
        out_rate = RATES["output"] * (1.5 if base_input > 272_000 else 1) / 1_000_000
        expected = base * retry_multiplier + output * out_rate
        approximate_input = counts[task["task_id"]].get("method") == "calibrated_local_tokenizer"
        calibration_allowance = 1.10 if approximate_input else 1.0
        conservative = (base * max(1.25, retry_multiplier * 1.25) * calibration_allowance
                        + output * 1.25 * out_rate)
        row = by_corpus[task["corpus"]]
        row["request_groups"] += 1
        row["alignment_tasks"] += len(task["anchors"])
        row["input_tokens"] += base_input
        row["estimated_output_tokens"] += output
        row["standard_usd"] += expected
        row["conservative_usd"] += conservative
        row["stratum_fallbacks"][str(len(chosen))] += 1
    for row in by_corpus.values():
        row["batch_usd"] = row["standard_usd"] * 0.5
        row["conservative_batch_usd"] = row["conservative_usd"] * 0.5
        row["stratum_fallbacks"] = dict(row["stratum_fallbacks"])
    return {
        "status": "estimated", "currency": "USD", "by_corpus": dict(by_corpus),
        "standard_usd": sum(row["standard_usd"] for row in by_corpus.values()),
        "conservative_usd": sum(row["conservative_usd"] for row in by_corpus.values()),
        "batch_usd": sum(row["batch_usd"] for row in by_corpus.values()),
        "conservative_batch_usd": sum(row["conservative_batch_usd"] for row in by_corpus.values()),
        "input_count_methods": dict(Counter(row.get("method", "model_token_endpoint") for row in counts.values())),
        "input_counts_approximate": any(row.get("method") == "calibrated_local_tokenizer" for row in counts.values()),
        "excluded_post_success_attempts_from_projection": excluded_post_success_attempts,
        "method": "Full-request input counts (approximate where calibrated_local_tokenizer is recorded); "
                  "pilot output tokens per anchor including protocol retries through the first valid response; "
                  "pilot attempt counts per group. Strata: corpus/source/target/genre, then "
                  "corpus/source/target, corpus/target, corpus. All input is priced at the cache-write rate "
                  "($0.125/M), with no assumed cache-read savings, matching first-call billing. "
                  "Conservative estimate adds 25% to measured output and retry-adjusted input, plus "
                  "10% input allowance for calibrated counts (a heuristic, not a confidence interval). "
                  "Duplicate calls after the first valid raw response are excluded from projection only; "
                  "actual spending includes them.",
    }


def _strata(task: dict) -> list[tuple]:
    corpus, source, target = task["corpus"], task["source_language"], task["target_language"]
    return [(corpus, source, target, task.get("genre")), (corpus, source, target),
            (corpus, target), (corpus,)]


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _markdown(summary: dict) -> str:
    run = summary["run"]
    assessment = summary["assessment"]
    cost = summary["pilot_cost"]
    lines = ["# Evidence-alignment pilot", "", f"Run status: **{run.get('status', 'not_run')}**.",
             f"Quality gate: **{assessment['overall']['gate']}**.", ""]
    if run.get("error"):
        lines.extend([f"Run issue: `{run['error']}`", ""])
    lines.extend([
        "This is a 30-document feasibility pilot, not a dataset-wide accuracy guarantee.",
        "Semantic decisions are qualitative Codex assistant reviews, not independent expert annotations.", "",
        "| Corpus | Tasks | First exact | Final exact | Reviewed valid | Pending reviews | Coverage | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for corpus, row in assessment["by_corpus"].items():
        lines.append(f"| {corpus} | {row['alignment_tasks']} | {row['first_attempt_exact_matches']} | "
                     f"{row['final_exact_matches']} | {row['valid_matches']} | {row['pending_reviews']} | "
                     f"{_percent(row['valid_coverage'])} | {row['gate']} |")
    lines.extend(["", ("A valid match is an exact quotation with a current semantic review confirming "
                  "equivalent support, preserved qualifications, minimality, and language correctness."), "",
                  "## Outcomes by target language", "",
                  "| Corpus / language | Final outcomes | Reviewed precision | Valid coverage |",
                  "|---|---|---:|---:|"])
    for corpus, languages in assessment["by_target_language"].items():
        for language, row in languages.items():
            statuses = ", ".join(f"{name}: {count}" for name, count in sorted(row["final_statuses"].items()))
            lines.append(f"| {corpus} / {language} | {statuses} | {_percent(row['semantic_precision'])} | "
                         f"{_percent(row['valid_coverage'])} |")
    lines.extend(["", "## Cost", ""])
    if cost["recorded_attempts"] == 0:
        lines.append("No generation attempts are recorded. No measured pilot cost or quality result is available.")
    elif cost["recorded_usage_complete"]:
        label = "Pilot cost" if cost["complete"] else "Recorded pilot cost"
        lines.append(f"{label}: **${cost['recorded_total_usd']:.6f} USD**, from "
                     f"{cost['recorded_attempts']} attempts, {cost['input_tokens']:,} input tokens and "
                     f"{cost['output_tokens']:,} output tokens. {cost['cost_basis']} "
                     f"Token-rate calculation alone: ${cost['rate_calculated_usd']:.6f}. "
                     "This is API usage accounting, not an invoice.")
    else:
        lines.append(f"Known pilot usage cost: **at least ${cost['known_usd']:.6f} USD**. "
                     f"{len(cost['attempts_without_usage'])} attempts have no usable token record; "
                     "their cost is unknown.")
    if cost["billing_incidents"]:
        lines.extend(["", (f"An interrupted run may have up to {cost['unrecorded_requests_upper_bound']} "
                      "additional requests without saved responses; final pilot billing is unknown.")])
        if cost["unrecorded_cost_allowance_usd"] is not None:
            lines.extend(["", (f"Allowance for these requests: ${cost['unrecorded_cost_allowance_usd']:.6f}; "
                          f"recorded spending plus allowance: ${cost['recorded_plus_allowance_usd']:.6f}. "
                          f"{cost['allowance_assumption']}")])
    if cost["rejected_before_generation"]:
        lines.extend(["", (f"{len(cost['rejected_before_generation'])} authentication/access requests "
                      "were rejected before generation and are excluded from generated-token costs.")])
    estimate = summary["full_workload_estimate"]
    if estimate["status"] != "estimated":
        lines.extend(["", f"Full-workload estimate: **{estimate['status']}**. "
                      f"{estimate.get('reason', '')}".rstrip()])
    else:
        if estimate.get("quality_gate") != "passed":
            lines.extend(["", "The requested cost estimate is for budgeting only; the pilot has not passed all quality criteria."])
        lines.extend(["", "| Corpus | Standard estimate | Conservative | Batch estimate |",
                      "|---|---:|---:|---:|"])
        for corpus, row in estimate["by_corpus"].items():
            lines.append(f"| {corpus} | ${row['standard_usd']:.4f} | ${row['conservative_usd']:.4f} | "
                         f"${row['batch_usd']:.4f} |")
        lines.extend([(f"| **Total** | **${estimate['standard_usd']:.4f}** | "
                      f"**${estimate['conservative_usd']:.4f}** | **${estimate['batch_usd']:.4f}** |"),
                      "", estimate["method"]])
    lines.extend(["", ("Standard rates per million tokens: $0.10 input, $0.01 cached input, "
                  "$0.125 cache writes, $0.50 output. Requests above 272,000 input tokens use "
                  "2× input/cache and 1.5× output rates. Batch estimates use half those rates. "
                  f"[Official pricing]({PRICING_URL})"), "",
                  "No retrieval chunks or Hugging Face updates are produced by this pilot.", ""])
    return "\n".join(lines)


def report(run_dir: Path, output_dir: Path, *, estimate_without_quality_gate: bool = False) -> dict:
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    tasks = read_jsonl(run_dir / "tasks.jsonl")
    if not tasks:
        raise ValueError("pilot tasks.jsonl is missing or empty; prepare the sample first")
    results = read_jsonl(run_dir / "results.jsonl")
    reviews = read_jsonl(run_dir / "reviews.jsonl")
    attempts = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((run_dir / "attempts").glob("*.json"))]
    metadata_path = run_dir / "run.json"
    run = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"status": "not_run"}
    assessment, queue = assess(tasks, results, reviews)
    reviewed = {row["alignment_id"]: row for row in reviews}
    target_texts = {task["task_id"]: task["target_text"] for task in tasks}
    accepted = []
    for row in results:
        decision = reviewed.get(row["alignment_id"], {})
        quote = row.get("target_text") or ""
        if (row.get("status") == "matched" and quote.strip()
                and quote in target_texts[row["task_id"]]
                and decision.get("result_hash") == result_hash(row)
                and decision.get("reviewer")
                and all(decision.get(field) is True for field in REVIEW_FIELDS)):
            accepted.append({**row, "review": decision})
    if len(accepted) != assessment["overall"]["valid_matches"]:
        raise AssertionError("accepted artifact disagrees with reviewed coverage")
    (run_dir / "accepted_alignments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    counts = read_jsonl(run_dir / "token_counts.jsonl")
    max_input = max([row.get("input_tokens", 0) for row in counts] + [0])
    cost = actual_cost(attempts, read_jsonl(run_dir / "incidents.jsonl"), max_input_tokens=max_input)
    if (run.get("status") != "completed"
            or (assessment["overall"]["gate"] != "passed" and not estimate_without_quality_gate)):
        estimate = {"status": "withheld", "reason": "Requires a completed pilot passing every corpus quality gate."}
    elif not cost["recorded_usage_complete"]:
        estimate = {"status": "pending_usage", "reason": "Some attempts have unknown usage/cost."}
    else:
        estimate = estimate_cost(read_jsonl(run_dir / "full_tasks.jsonl"), tasks, attempts,
                                 counts)
        estimate["quality_gate"] = assessment["overall"]["gate"]
        estimate["rollout_approved"] = assessment["overall"]["gate"] == "passed"
    summary = {
        "model": MODEL, "run": run, "assessment": assessment, "pilot_cost": cost,
        "full_workload_estimate": estimate,
        "rates_usd_per_million": RATES, "pricing_source": PRICING_URL,
        "criteria": {"semantic_precision": 0.95, "valid_coverage_per_corpus": 0.90,
                     "retained_quotes_exact_and_reviewed": True},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(_markdown(summary), encoding="utf-8")
    (run_dir / "review_queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--estimate-cost", action="store_true",
                        help="Estimate budgeting costs even if quality fails; never starts generation")
    args = parser.parse_args()
    summary = report(args.run_dir, args.output,
                     estimate_without_quality_gate=args.estimate_cost)
    print(json.dumps({"quality_gate": summary["assessment"]["overall"]["gate"],
                      "cost_status": summary["full_workload_estimate"]["status"]}))


if __name__ == "__main__":
    main()
