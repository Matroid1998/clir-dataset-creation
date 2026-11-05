"""Evidence alignment must preserve literal quotes and fail closed on incomplete output."""

import copy
import json
from types import SimpleNamespace

import pytest

from clir_bench.alignment.runner import (
    AccessBlocked,
    _client,
    build_request,
    count_inputs,
    request_hash,
    run_pilot,
    run_task,
    validate_response,
)


@pytest.fixture
def task():
    return {
        "task_id": "eurlex-pair-1",
        "corpus": "eurlex",
        "document_id": "article-1",
        "source_unit_id": "eurlex:article-1:en",
        "target_unit_id": "eurlex:article-1:fr",
        "source_language": "en",
        "target_language": "fr",
        "source_text": "Heading. The rule applies only after approval. No further conditions.",
        "target_text": "Titre. La règle s’applique  après approbation. Aucune autre condition.",
        "source_title": "Article 1",
        "target_title": "Article premier",
        "genre": "article",
        "anchors": [
            {
                "anchor_id": "a1",
                "text": "The rule applies only after approval.",
                "qa_ids": ["q1", "q2"],
                "origins": ["evidence", "answer"],
                "question_languages": ["en"],
                "modes": ["lookup"],
            },
            {
                "anchor_id": "a2",
                "text": "No further conditions.",
                "qa_ids": ["q3"],
                "origins": ["answer"],
                "question_languages": ["en"],
                "modes": ["lookup"],
            },
        ],
    }


def response_for(results=None, **payload_updates):
    payload = {
        "source_language_matches": True,
        "target_language_matches": True,
        "results": results
        if results is not None
        else [
            {
                "anchor_id": "a1",
                "status": "matched",
                "text": "La règle s’applique  après approbation.",
                "reason": "The condition is retained.",
            },
            {
                "anchor_id": "a2",
                "status": "matched",
                "text": "Aucune autre condition.",
                "reason": "Equivalent support.",
            },
        ],
    }
    payload.update(payload_updates)
    return {
        "id": "resp_test",
        "model": "gpt-6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps(payload)}],
            }
        ],
        "usage": {
            "input_tokens": 123,
            "output_tokens": 45,
            "input_tokens_details": {"cached_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 20},
        },
    }


def payload_of(response):
    return json.loads(response["output"][0]["content"][0]["text"])


def replace_payload(response, payload):
    response["output"][0]["content"][0]["text"] = json.dumps(payload)
    return response


def test_request_contains_full_documents_and_all_distinct_spans(task):
    request = build_request(task)
    assert request["model"] == "openai/gpt-6-luna"
    assert request["reasoning"]["effort"] == "medium"
    assert request["max_output_tokens"] == 32768
    rendered = json.dumps(request, ensure_ascii=False)
    assert task["source_text"] in rendered
    assert task["target_text"] in rendered
    assert all(anchor["text"] in rendered for anchor in task["anchors"])
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True


def test_valid_unicode_quotes_remain_exact(task):
    rows, issues = validate_response(task, response_for())
    assert issues == []
    assert len(rows) == 2
    assert all(row["exact_match"] and row["status"] == "matched" for row in rows)
    by_id = {row["anchor_id"]: row for row in rows}
    assert by_id["a1"]["text"] == "La règle s’applique  après approbation."


def test_openrouter_reasoning_item_with_null_content_preserves_text_response(task):
    response = response_for()
    response["output"].insert(0, {"type": "reasoning", "content": None, "summary": []})
    rows, issues = validate_response(task, response)
    assert issues == []
    assert len(rows) == len(task["anchors"])
    assert all(row["status"] == "matched" for row in rows)


@pytest.mark.parametrize(
    "quotation",
    [
        "La règle s’applique après approbation.",  # collapsed whitespace
        "La re\u0300gle s’applique  apre\u0300s approbation.",  # Unicode normalization
        "LA RÈGLE S’APPLIQUE  APRÈS APPROBATION.",
        "  La règle s’applique  après approbation.  ",
        "",
        " ",  # Present in the target, but never supporting evidence.
        "  ",
    ],
)
def test_quote_is_not_repaired_or_normalized(task, quotation):
    response = response_for()
    payload = payload_of(response)
    payload["results"][0]["text"] = quotation
    rows, issues = validate_response(task, replace_payload(response, payload))
    assert issues
    row = next(row for row in rows if row["anchor_id"] == "a1")
    assert row["status"] != "matched"
    assert row["exact_match"] is False


@pytest.mark.parametrize("status", ["not_found", "uncertain"])
def test_explicit_absence_is_a_valid_response_but_not_a_match(task, status):
    response = response_for()
    payload = payload_of(response)
    payload["results"][0].update(status=status, text="", reason="No reliable equivalent.")
    rows, issues = validate_response(task, replace_payload(response, payload))
    assert issues == []
    assert rows[0]["status"] == status
    assert rows[0]["exact_match"] is False


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra"])
def test_exact_anchor_coverage_is_required(task, change):
    response = response_for()
    payload = payload_of(response)
    if change == "missing":
        payload["results"].pop()
    elif change == "duplicate":
        payload["results"].append(copy.deepcopy(payload["results"][0]))
    else:
        extra = copy.deepcopy(payload["results"][0])
        extra["anchor_id"] = "invented"
        payload["results"].append(extra)
    _, issues = validate_response(task, replace_payload(response, payload))
    assert issues


@pytest.mark.parametrize("field", ["source_language_matches", "target_language_matches"])
def test_language_disagreement_is_visible_and_cannot_be_retained(task, field):
    rows, issues = validate_response(task, response_for(**{field: False}))
    assert issues
    assert rows
    assert all(row["status"] == "language_mismatch" for row in rows)


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
def test_incomplete_response_is_not_success_even_with_parseable_output(task, status):
    response = response_for()
    response["status"] = status
    rows, issues = validate_response(task, response)
    assert issues
    assert not any(row["status"] == "matched" for row in rows)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.6-luna",
        "gpt-6-luna-pro",
        "openai/gpt-6-luna-pro",
        "gpt-6-luna-other",
        "other-provider/gpt-6-luna",
    ],
)
def test_unrequested_model_cannot_pass_validation(task, model):
    response = response_for()
    response["model"] = model
    rows, issues = validate_response(task, response)
    assert issues
    assert not any(row["status"] == "matched" for row in rows)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-luna",
        "openai/gpt-6-luna",
        "gpt-6-luna-2026-11-05",
        "openai/gpt-6-luna-2026-11-05",
    ],
)
def test_same_requested_model_accepts_short_and_canonical_provider_identifiers(task, model):
    response = response_for()
    response["model"] = model
    rows, issues = validate_response(task, response)
    assert issues == []
    assert all(row["status"] == "matched" for row in rows)


@pytest.mark.parametrize("output", [[], [{"type": "message", "content": []}]])
def test_missing_output_is_not_an_empty_success(task, output):
    response = response_for()
    response["output"] = output
    rows, issues = validate_response(task, response)
    assert issues
    assert not any(row["status"] == "matched" for row in rows)


def test_refusal_is_not_an_empty_success(task):
    response = response_for()
    response["output"][0]["content"] = [{"type": "refusal", "refusal": "Cannot comply."}]
    rows, issues = validate_response(task, response)
    assert issues
    assert not any(row["status"] == "matched" for row in rows)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, **_kwargs):
        return copy.deepcopy(self.payload)


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return FakeResponse(output)


class FakeClient:
    def __init__(self, *outputs):
        self.responses = FakeResponses(outputs)


def fake_catalog(*identifiers):
    return SimpleNamespace(
        data=[
            SimpleNamespace(
                id=identifier,
                model_dump=lambda _identifier=identifier, **_kwargs: {"id": _identifier},
            )
            for identifier in identifiers
        ]
    )


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("clir_bench.alignment.runner.time.sleep", lambda _delay: None)


@pytest.mark.parametrize("field", ["source_text", "target_text", "source_language"])
def test_request_identity_includes_document_contents_and_language(task, field):
    before = request_hash(task)
    changed = copy.deepcopy(task)
    changed[field] += " changed"
    assert request_hash(changed) != before


def test_request_identity_includes_evidence_not_only_the_document(task):
    changed = copy.deepcopy(task)
    changed["anchors"][0]["text"] = "The rule applies only after approval"
    assert request_hash(changed) != request_hash(task)


def test_resume_uses_persisted_response_without_duplicate_paid_calls(task, tmp_path):
    client = FakeClient(response_for())
    first = run_task(client, task, tmp_path)
    assert len(client.responses.requests) == 1
    replay_client = FakeClient()
    replay = run_task(replay_client, task, tmp_path)
    assert replay == first
    assert replay_client.responses.requests == []
    assert first[0]["qa_ids"] == ["q1", "q2"]
    assert first[0]["attempt_count"] == 1
    attempts = list((tmp_path / "attempts").glob("*.json"))
    assert len(attempts) == 1
    persisted = json.loads(attempts[0].read_text())
    assert persisted["response"]["usage"] == response_for()["usage"]


def test_saved_null_reasoning_response_replays_without_a_paid_retry(task, tmp_path):
    response = response_for()
    response["output"].insert(0, {"type": "reasoning", "content": None, "summary": []})
    identity = request_hash(task)
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / f"{task['task_id']}-{identity[:12]}-1.json").write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "attempt": 1,
                "request_hash": identity,
                "response": response,
            }
        )
    )
    client = FakeClient()
    rows = run_task(client, task, tmp_path)
    assert client.responses.requests == []
    assert all(row["status"] == "matched" and row["attempt_count"] == 1 for row in rows)


def test_changed_evidence_cannot_inherit_a_cached_response(task, tmp_path):
    client = FakeClient(response_for(), response_for())
    run_task(client, task, tmp_path)
    changed = copy.deepcopy(task)
    changed["anchors"][0]["text"] = "The rule applies only after approval"
    run_task(client, changed, tmp_path)
    assert len(client.responses.requests) == 2
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2


def test_invalid_quote_retries_and_preserves_first_attempt_quality(task, tmp_path):
    bad = response_for()
    payload = payload_of(bad)
    payload["results"][0]["text"] = "La règle s’applique après approbation."
    replace_payload(bad, payload)
    client = FakeClient(bad, response_for())
    rows = run_task(client, task, tmp_path)
    assert len(client.responses.requests) == 2
    first = next(row for row in rows if row["anchor_id"] == "a1")
    assert first["status"] == "matched"
    assert first["first_status"] == "invalid"
    assert first["first_exact_match"] is False
    assert first["attempt_count"] == 2
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2


def test_exhausted_invalid_responses_stay_visible_and_do_not_restart(task, tmp_path):
    invalid = response_for()
    invalid["output"] = []
    client = FakeClient(invalid, invalid, invalid)
    rows = run_task(client, task, tmp_path)
    assert len(client.responses.requests) == 3
    assert len(rows) == len(task["anchors"])
    assert all(row["status"] != "matched" and row["attempt_count"] == 3 for row in rows)
    replay_client = FakeClient()
    assert run_task(replay_client, task, tmp_path) == rows
    assert replay_client.responses.requests == []


def test_transient_error_is_retried_and_logged(task, tmp_path):
    client = FakeClient(ConnectionError("Temporary network failure"), response_for())
    rows = run_task(client, task, tmp_path)
    assert len(client.responses.requests) == 2
    assert all(row["status"] == "matched" for row in rows)
    attempts = [json.loads(path.read_text()) for path in (tmp_path / "attempts").glob("*.json")]
    assert len(attempts) == 2
    assert sum("error" in attempt for attempt in attempts) == 1


@pytest.mark.parametrize("status", [401, 403])
def test_account_access_errors_fail_fast(task, tmp_path, status):
    error = RuntimeError("Account cannot access this model")
    error.status_code = status
    client = FakeClient(error)
    with pytest.raises(AccessBlocked):
        run_task(client, task, tmp_path)
    assert len(client.responses.requests) == 1
    attempts = list((tmp_path / "attempts").glob("*.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text())["error"]["status"] == status


def test_fixed_credentials_can_resume_a_task_after_an_access_failure(task, tmp_path):
    error = RuntimeError("Organization access denied")
    error.status_code = 401
    with pytest.raises(AccessBlocked):
        run_task(FakeClient(error), task, tmp_path)
    repaired = FakeClient(response_for())
    rows = run_task(repaired, task, tmp_path)
    assert len(repaired.responses.requests) == 1
    assert all(row["status"] == "matched" for row in rows)
    saved = [json.loads(path.read_text()) for path in (tmp_path / "attempts").glob("*.json")]
    assert any(row.get("error", {}).get("status") == 401 for row in saved)
    assert any("response" in row for row in saved)


def write_tasks(tmp_path, task, filename="tasks.jsonl"):
    (tmp_path / filename).write_text(json.dumps(task) + "\n")


def test_preflight_failure_is_saved_without_any_generation(task, tmp_path):
    write_tasks(tmp_path, task)
    client = FakeClient()
    error = RuntimeError("Organization access denied")
    error.status_code = 401

    def denied():
        raise error

    client.models = SimpleNamespace(list=denied)
    result = run_pilot(tmp_path, client=client)
    saved = json.loads((tmp_path / "run.json").read_text())
    assert result["status"] == saved["status"] == "blocked"
    assert saved["error"]["status"] == 401
    assert client.responses.requests == []
    assert not (tmp_path / "attempts").exists()


def test_run_pilot_completes_all_alignment_rows(task, tmp_path):
    write_tasks(tmp_path, task)
    client = FakeClient(response_for())
    client.models = SimpleNamespace(
        list=lambda: fake_catalog("openai/gpt-6-luna")
    )
    manifest = run_pilot(tmp_path, workers=1, client=client)
    assert manifest["status"] == "completed"
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == len(task["anchors"])
    assert {row["alignment_id"] for row in rows} == {
        task["task_id"] + ":" + anchor["anchor_id"] for anchor in task["anchors"]
    }


def test_openrouter_catalog_does_not_allow_a_model_substitution(task, tmp_path):
    write_tasks(tmp_path, task)
    client = FakeClient()
    client.models = SimpleNamespace(
        list=lambda: fake_catalog("openai/gpt-6-luna-pro", "openai/gpt-5.6-luna")
    )
    manifest = run_pilot(tmp_path, workers=1, client=client)
    assert manifest["status"] == "blocked"
    assert client.responses.requests == []


def test_client_uses_explicit_openrouter_credentials_and_endpoint(monkeypatch):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-direct-key")
    monkeypatch.setattr("openai.OpenAI", constructor)
    _client()
    assert captured["api_key"] == "test-router-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["max_retries"] == 0


def test_unexpected_worker_failure_updates_run_status(task, tmp_path, monkeypatch):
    write_tasks(tmp_path, task)
    client = FakeClient()
    client.models = SimpleNamespace(
        list=lambda: fake_catalog("openai/gpt-6-luna")
    )

    def broken_worker(*_args, **_kwargs):
        raise ValueError("Checkpoint identity mismatch")

    monkeypatch.setattr("clir_bench.alignment.runner.run_task", broken_worker)
    try:
        run_pilot(tmp_path, workers=1, client=client)
    except ValueError:
        pass  # A propagated programming error is fine if the on-disk state is truthful.
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["status"] == "failed"


def test_missing_client_configuration_is_recorded_as_a_block(task, tmp_path, monkeypatch):
    write_tasks(tmp_path, task)

    def missing_key():
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr("clir_bench.alignment.runner._client", missing_key)
    try:
        run_pilot(tmp_path)
    except RuntimeError:
        pass
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["status"] in {"blocked", "failed"}


def test_token_count_uses_full_request_and_resumes_without_recount(task, tmp_path):
    write_tasks(tmp_path, task, "full_tasks.jsonl")
    received = []

    def count(**payload):
        received.append(copy.deepcopy(payload))
        return SimpleNamespace(input_tokens=456)

    client = FakeClient()
    client.responses.input_tokens = SimpleNamespace(count=count)
    first = count_inputs(tmp_path, client=client)
    assert first["groups"] == 1 and first["input_tokens"] == 456
    assert len(received) == 1
    request = build_request(task)
    for key in ["model", "instructions", "input", "reasoning", "text"]:
        assert received[0][key] == request[key]
    assert "max_output_tokens" not in received[0]
    assert count_inputs(tmp_path, client=client) == first
    assert len(received) == 1


def test_changed_request_forces_fresh_input_token_count(task, tmp_path):
    write_tasks(tmp_path, task, "full_tasks.jsonl")
    received = []

    def count(**payload):
        received.append(copy.deepcopy(payload))
        return SimpleNamespace(input_tokens=100 * len(received))

    client = FakeClient()
    client.responses.input_tokens = SimpleNamespace(count=count)
    count_inputs(tmp_path, client=client)
    changed = copy.deepcopy(task)
    changed["target_text"] += " Additional text."
    write_tasks(tmp_path, changed, "full_tasks.jsonl")
    result = count_inputs(tmp_path, client=client)
    assert len(received) == 2
    assert result == {"groups": 1, "input_tokens": 200}


def test_missing_openrouter_count_endpoint_uses_explicit_calibration(task, tmp_path, monkeypatch):
    write_tasks(tmp_path, task, "full_tasks.jsonl")
    error = RuntimeError("Token-count endpoint is not available")
    error.status_code = 404

    def unsupported(**_kwargs):
        raise error

    calls = []

    def calibrated(run_dir):
        calls.append(run_dir)
        return {"method": "calibrated_local_tokenizer", "approximate": True}

    monkeypatch.setattr("clir_bench.alignment.token_estimates.estimate", calibrated)
    client = FakeClient()
    client.responses.input_tokens = SimpleNamespace(count=unsupported)
    result = count_inputs(tmp_path, client=client)
    assert result["approximate"] is True
    assert calls == [tmp_path]
    assert client.responses.requests == []
