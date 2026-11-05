"""Provider usage calibrates an explicitly approximate local tokenizer fallback."""

import json

import pytest

from clir_bench.alignment import token_estimates as estimates
from clir_bench.alignment.runner import API_MODEL, request_hash, write_json, write_jsonl


class CharacterEncoding:
    """Deterministic test proxy; never downloads or invokes a real model."""

    def encode(self, text, *, disallowed_special):
        assert disallowed_special == ()
        return list(text)


def task(identity, *, corpus="eurlex", source="en", target="fr", genre="article"):
    return {
        "task_id": identity, "corpus": corpus, "genre": genre,
        "source_unit_id": identity + ":" + source, "target_unit_id": identity + ":" + target,
        "source_language": source, "target_language": target,
        "source_text": "Source condition " + identity, "target_text": "Condition cible " + identity,
        "anchors": [{"anchor_id": identity + "-a", "text": "condition", "qa_ids": ["q"]}],
    }


def save_attempt(run_dir, item, input_tokens, *, attempt=1, identity=None, model=API_MODEL, cached=0):
    path = run_dir / "attempts" / f"{item['task_id']}-{attempt}.json"
    write_json(path, {
        "task_id": item["task_id"], "attempt": attempt,
        "request_hash": identity if identity is not None else request_hash(item),
        "response": {
            "id": f"resp-{item['task_id']}-{attempt}", "model": model,
            "usage": {"input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": cached}},
        },
    })


def load_counts(run_dir):
    return {row["task_id"]: row for row in (
        json.loads(line) for line in (run_dir / "token_counts.jsonl").read_text().splitlines()
    )}


def test_calibrates_remaining_requests_but_preserves_exact_observed_usage(tmp_path):
    encoder = CharacterEncoding()
    tasks = [task(f"t{number}") for number in range(4)]
    write_jsonl(tmp_path / "full_tasks.jsonl", tasks)
    for item in tasks[:3]:
        save_attempt(tmp_path, item, 2 * estimates.local_tokens(item, encoder))
    summary = estimates.estimate(tmp_path, encoding=encoder)
    assert summary["approximate"] is True
    assert summary["observed_requests"] == 3
    assert summary["estimated_requests"] == 1
    assert summary["tokenizer"] == "o200k_base"
    counts = load_counts(tmp_path)
    assert counts["t0"]["method"] == "observed_provider_usage"
    assert counts["t0"]["approximate"] is False
    assert counts["t0"]["input_tokens"] == 2 * estimates.local_tokens(tasks[0], encoder)
    assert counts["t3"]["method"] == "calibrated_local_tokenizer"
    assert counts["t3"]["input_tokens"] == 2 * estimates.local_tokens(tasks[3], encoder)
    assert counts["t3"]["calibration"]["samples"] == 3
    assert counts["t3"]["calibration"]["stratum"].endswith("genre/local_length_bucket")
    assert summary["leave_one_out_residuals"]["max_absolute_relative_error"] == 0


def test_first_native_usage_per_current_request_is_used_once(tmp_path):
    encoder = CharacterEncoding()
    item = task("first")
    write_jsonl(tmp_path / "full_tasks.jsonl", [item])
    save_attempt(tmp_path, item, 1200, attempt=1)
    save_attempt(tmp_path, item, 1800, attempt=2)
    estimates.estimate(tmp_path, encoding=encoder)
    counts = load_counts(tmp_path)
    assert counts["first"]["input_tokens"] == 1200
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert calibration["observed_requests"] == 1
    assert calibration["varying_input_usage_for_identical_request"] == [
        {"task_id": "first", "input_token_counts": [1200, 1800]}
    ]
    assert calibration["leave_one_out_residuals"]["samples"] == 0
    assert calibration["leave_one_out_residuals"]["mean_absolute_tokens"] is None


def test_stale_request_hash_and_other_models_do_not_calibrate(tmp_path):
    item = task("stale")
    write_jsonl(tmp_path / "full_tasks.jsonl", [item])
    save_attempt(tmp_path, item, 1000, identity="previous-prompt")
    save_attempt(tmp_path, item, 1000, attempt=2, model="gpt-5.6-luna")
    with pytest.raises(ValueError, match="No current provider"):
        estimates.estimate(tmp_path, encoding=CharacterEncoding())
    assert not (tmp_path / "token_counts.jsonl").exists()


@pytest.mark.parametrize("cached", [-1, 1001, True, "100"])
def test_invalid_cache_partition_is_excluded_from_calibration(tmp_path, cached):
    good, bad = task("good"), task("bad")
    write_jsonl(tmp_path / "full_tasks.jsonl", [good, bad])
    save_attempt(tmp_path, good, 1000)
    save_attempt(tmp_path, bad, 1000, cached=cached)
    estimates.estimate(tmp_path, encoding=CharacterEncoding())
    counts = load_counts(tmp_path)
    assert counts["good"]["method"] == "observed_provider_usage"
    assert counts["bad"]["method"] == "calibrated_local_tokenizer"
    calibration = json.loads((tmp_path / "calibration.json").read_text())
    assert calibration["excluded_attempts"]["invalid_input_usage"] == 1


def test_sparse_language_pair_uses_broader_corpus_target_calibration(tmp_path):
    observed = task("observed", source="de", target="fr")
    missing = task("missing", source="en", target="fr")
    write_jsonl(tmp_path / "full_tasks.jsonl", [observed, missing])
    save_attempt(tmp_path, observed, 1400)
    estimates.estimate(tmp_path, encoding=CharacterEncoding())
    row = load_counts(tmp_path)["missing"]
    assert row["calibration"]["stratum"] == "corpus/target_language"
    assert row["calibration"]["samples"] == 1


def test_does_not_apply_one_corpuss_calibration_to_an_unobserved_corpus(tmp_path):
    observed = task("observed")
    missing = task("missing", corpus="un", target="zh", genre="meeting")
    write_jsonl(tmp_path / "full_tasks.jsonl", [observed, missing])
    save_attempt(tmp_path, observed, 1400)
    with pytest.raises(ValueError, match="No current provider token observations for corpus 'un'"):
        estimates.estimate(tmp_path, encoding=CharacterEncoding())
    assert not (tmp_path / "token_counts.jsonl").exists()


def test_rendering_contains_instructions_full_input_and_schema_only():
    rendered = estimates.render_local_input({
        "instructions": "Exact instructions", "input": "Exact input <|endoftext|>",
        "text": {"format": {"type": "json_schema", "schema": {"description": "café"}}},
        "max_output_tokens": 32768,
    })
    assert rendered.startswith("Exact instructions\nExact input <|endoftext|>\n")
    assert '"description": "café"' in rendered
    assert "32768" not in rendered
