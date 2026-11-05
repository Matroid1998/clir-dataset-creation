"""Approximate input counts when the provider has no token-count endpoint.

``o200k_base`` is a local proxy, not a verified GPT-6 tokenizer. Estimates are
calibrated against provider-reported input usage for the exact current pilot
request shape. Observed requests retain their actual provider token counts.
This module never calls a generation or token-count endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .runner import (
    API_MODEL,
    build_request,
    digest,
    now,
    read_jsonl,
    request_hash,
    write_json,
    write_jsonl,
)

TOKENIZER = "o200k_base"
MODEL_PATTERN = re.compile(r"(?:openai/)?gpt-6-luna(?:-\d{4}-\d{2}-\d{2})?")
# Prefer stable local-language comparisons. Sparse strata explicitly fall back
# instead of claiming a one-observation length/genre fit has been validated.
STRATA = (
    (("corpus", "source_language", "target_language", "genre", "local_length_bucket"), 3),
    (("corpus", "source_language", "target_language", "genre"), 3),
    (("corpus", "source_language", "target_language"), 1),
    (("corpus", "target_language", "genre"), 3),
    (("corpus", "target_language"), 1),
    (("corpus", "source_language"), 1),
    (("corpus",), 1),
)


def render_local_input(request: dict) -> str:
    """Count all supplied instruction/input/schema text, not response settings.

    Provider-side message framing and schema transformations are unknown; the
    calibration ratio absorbs their average effect and their residuals remain
    visible. Documents and passages are preserved exactly as rendered by runner.
    """
    return "\n".join((request["instructions"], request["input"],
                      json.dumps(request["text"], ensure_ascii=False, sort_keys=True)))


def local_tokens(task: dict, encoding: Any) -> int:
    count = len(encoding.encode(render_local_input(build_request(task)), disallowed_special=()))
    if count <= 0:
        raise ValueError("Local tokenization returned an empty input")
    return count


def length_bucket(tokens: int) -> str:
    for upper in (2_000, 8_000, 32_000, 128_000):
        if tokens <= upper:
            return f"<= {upper}"
    return "> 128000"


def _features(task: dict, tokens: int) -> dict:
    return {
        "corpus": task["corpus"], "source_language": task["source_language"],
        "target_language": task["target_language"], "genre": task.get("genre", "unknown"),
        "local_length_bucket": length_bucket(tokens),
    }


def valid_input_usage(usage: dict) -> bool:
    """Cache partitions cannot be negative or exceed total provider input."""
    tokens = usage.get("input_tokens")
    if type(tokens) is not int or tokens <= 0:
        return False
    details = usage.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        return False
    cached = details.get("cached_tokens")
    written = details.get("cache_write_tokens")
    cached = 0 if cached is None else cached
    written = 0 if written is None else written
    return (type(cached) is int and type(written) is int and cached >= 0 and written >= 0
            and cached + written <= tokens)


def _observations(run_dir: Path, tasks: list[dict], local: dict[str, int]) -> tuple[list[dict], dict]:
    task_map = {task["task_id"]: task for task in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("Duplicate full-workload task identifiers")
    identities = {key: request_hash(task) for key, task in task_map.items()}
    candidates: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    excluded = Counter()
    for path in sorted((run_dir / "attempts").glob("*.json")):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        task_id = envelope.get("task_id")
        if task_id not in task_map:
            excluded["unknown_task"] += 1
            continue
        if envelope.get("request_hash") != identities[task_id]:
            excluded["stale_request_hash"] += 1
            continue
        response = envelope.get("response")
        if not isinstance(response, dict):
            excluded["no_response"] += 1
            continue
        if not MODEL_PATTERN.fullmatch(str(response.get("model", ""))):
            excluded["different_model"] += 1
            continue
        usage = response.get("usage")
        if not isinstance(usage, dict) or not valid_input_usage(usage):
            excluded["invalid_input_usage"] += 1
            continue
        candidates[task_id].append((path.name, envelope))
    observations = []
    variable_usage = []
    for task_id in sorted(candidates):
        entries = sorted(candidates[task_id], key=lambda pair: (pair[1].get("attempt", 0), pair[0]))
        filename, first = entries[0]
        native = first["response"]["usage"]["input_tokens"]
        variants = sorted({entry["response"]["usage"]["input_tokens"] for _, entry in entries})
        if len(variants) > 1:
            variable_usage.append({"task_id": task_id, "input_token_counts": variants})
        observations.append({
            "task_id": task_id, "request_hash": identities[task_id],
            "native_input_tokens": native, "local_input_tokens": local[task_id],
            "ratio": native / local[task_id], "attempt_file": filename,
            "provider_response_id": first["response"].get("id"),
            **_features(task_map[task_id], local[task_id]),
        })
    return observations, {"excluded_attempts": dict(excluded),
                          "varying_input_usage_for_identical_request": variable_usage}


def _fit(features: dict, observations: list[dict]) -> tuple[dict, list[dict]]:
    for fields, minimum in STRATA:
        members = [row for row in observations if all(row[field] == features[field] for field in fields)]
        if len(members) >= minimum:
            ratio = mean(row["ratio"] for row in members)
            values = {field: features[field] for field in fields}
            return {
                "stratum": "/".join(fields), "values": values, "samples": len(members),
                "ratio": ratio, "fit": "mean_provider_to_local_ratio",
                "calibration_id": digest({"fields": fields, "values": values}),
            }, members
    raise ValueError(f"No current provider token observations for corpus {features['corpus']!r}")


def _residuals(pairs: list[tuple[float, int]]) -> dict:
    if not pairs:
        return {"samples": 0, "mean_absolute_tokens": None, "max_absolute_tokens": None,
                "mean_absolute_relative_error": None, "max_absolute_relative_error": None}
    absolute = [abs(predicted - native) for predicted, native in pairs]
    relative = [abs(predicted - native) / native for predicted, native in pairs]
    return {
        "samples": len(pairs), "mean_absolute_tokens": mean(absolute), "max_absolute_tokens": max(absolute),
        "mean_absolute_relative_error": mean(relative), "max_absolute_relative_error": max(relative),
    }


def estimate(run_dir: Path, *, encoding: Any = None) -> dict:
    """Write approximate full-request input counts calibrated on recorded usage."""
    run_dir = Path(run_dir)
    tasks = read_jsonl(run_dir / "full_tasks.jsonl")
    if not tasks:
        raise ValueError("Prepare full_tasks.jsonl before estimating tokens")
    if encoding is None:
        import tiktoken

        encoding = tiktoken.get_encoding(TOKENIZER)
    local = {task["task_id"]: local_tokens(task, encoding) for task in tasks}
    observations, diagnostics = _observations(run_dir, tasks, local)
    if not observations:
        raise ValueError("No current provider input-token observations; calibration is unavailable")
    observed = {row["task_id"]: row for row in observations}
    groups = {}
    counts = []
    for task in tasks:
        task_id = task["task_id"]
        fit, members = _fit(_features(task, local[task_id]), observations)
        group_id = fit["calibration_id"]
        if group_id not in groups:
            groups[group_id] = {
                **fit, "sample_task_ids": [row["task_id"] for row in members],
                "fitted_residuals": _residuals([(row["local_input_tokens"] * fit["ratio"], row["native_input_tokens"]) for row in members]),
            }
        if task_id in observed:
            observation = observed[task_id]
            tokens = observation["native_input_tokens"]
            method = "observed_provider_usage"
            provenance = {"attempt_file": observation["attempt_file"], "provider_response_id": observation["provider_response_id"]}
        else:
            tokens = max(1, math.ceil(local[task_id] * fit["ratio"]))
            method = "calibrated_local_tokenizer"
            provenance = {}
        counts.append({
            "task_id": task_id, "request_hash": request_hash(task), "model": API_MODEL,
            "input_tokens": tokens, "method": method, "approximate": method != "observed_provider_usage",
            "tokenizer": TOKENIZER, "local_input_tokens": local[task_id], "calibration": fit,
            **provenance,
        })
    fitted_pairs, loo_pairs = [], []
    loo_unavailable = []
    for row in observations:
        fit, _ = _fit(row, observations)
        fitted_pairs.append((row["local_input_tokens"] * fit["ratio"], row["native_input_tokens"]))
        others = [other for other in observations if other["task_id"] != row["task_id"]]
        try:
            loo_fit, _ = _fit(row, others)
        except ValueError:
            loo_unavailable.append(row["task_id"])
            continue
        loo_pairs.append((row["local_input_tokens"] * loo_fit["ratio"], row["native_input_tokens"]))
    calibration = {
        "method": "calibrated_local_tokenizer", "approximate": True, "tokenizer": TOKENIZER,
        "model": API_MODEL, "created_at": now(),
        "reason": "The OpenRouter input-token-count endpoint is unavailable. o200k_base is a proxy, not a verified GPT-6 tokenizer.",
        "rendering": "instructions + newline + unchanged rendered input + newline + JSON text/schema configuration; provider framing is absorbed by calibration",
        "full_tasks_sha256": digest(tasks), "groups": list(groups.values()),
        "observations": observations, "observed_requests": len(observations),
        "estimated_requests": sum(row["approximate"] for row in counts),
        "total_requests": len(counts), "input_tokens": sum(row["input_tokens"] for row in counts),
        "fitted_residuals": _residuals(fitted_pairs),
        "leave_one_out_residuals": _residuals(loo_pairs),
        "leave_one_out_unavailable_task_ids": loo_unavailable,
        "limitations": [
            "Residuals describe this pilot; they are not a statistical error bound for the full corpus.",
            "Single-observation language strata have no independent within-stratum validation.",
            "Observed requests retain exact input usage; all other requests remain approximate.",
        ],
        **diagnostics,
    }
    write_json(run_dir / "calibration.json", calibration)
    write_jsonl(run_dir / "token_counts.jsonl", sorted(counts, key=lambda row: row["task_id"]))
    return {key: calibration[key] for key in (
        "method", "approximate", "tokenizer", "observed_requests", "estimated_requests",
        "total_requests", "input_tokens", "fitted_residuals", "leave_one_out_residuals",
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
