"""The full Terra run must keep its model, question context, and durable progress."""

import copy
import csv
import fcntl
import json
import signal
from threading import Event
from types import SimpleNamespace

import pytest

from clir_bench.alignment import terra


@pytest.fixture
def task():
    return {
        "task_id": "pair-terra-1",
        "corpus": "eurlex",
        "document_id": "article-1",
        "source_unit_id": "eurlex:article-1:en",
        "target_unit_id": "eurlex:article-1:fr",
        "source_language": "en",
        "target_language": "fr",
        "source_text": "No germline trials may be conducted. Notify the Commission immediately.",
        "target_text": "Aucun essai germinal ne peut être conduit. Informez  immédiatement la Commission.",
        "genre": "article",
        "anchors": [
            {
                "anchor_id": "a1",
                "text": "No germline trials may be conducted.",
                "qa_ids": ["q1", "q2"],
                "qa_links": [
                    {"qa_id": "q1", "question": "Are germline trials permitted?"},
                    {"qa_id": "q1", "question": "Are germline trials permitted?"},
                    {"qa_id": "q2", "question": "What clinical trials are prohibited?"},
                ],
            },
            {
                "anchor_id": "a2",
                "text": "Notify the Commission immediately.",
                "qa_ids": ["q3"],
                "qa_links": [{"qa_id": "q3", "question": "Who must be notified, and when?"}],
            },
        ],
    }


def response_for(*, model="gpt-6.1-terra", status="completed", first_quote=None):
    payload = {
        "source_language_matches": True,
        "target_language_matches": True,
        "results": [
            {
                "anchor_id": "a1", "status": "matched",
                "text": "Aucun essai germinal ne peut être conduit." if first_quote is None else first_quote,
                "reason": "The prohibition remains explicit.",
            },
            {
                "anchor_id": "a2", "status": "matched",
                "text": "Informez  immédiatement la Commission.", "reason": "Immediate notification.",
            },
        ],
    }
    return {
        "id": "resp-terra-test", "model": model, "status": status,
        "output": [
            {"type": "reasoning", "content": None},
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]},
        ],
        "usage": {
            "input_tokens": 400, "output_tokens": 90,
            "input_tokens_details": {"cached_tokens": 100},
            "output_tokens_details": {"reasoning_tokens": 30},
        },
    }


def test_request_uses_direct_terra_and_full_document_and_question_context(task):
    request = terra.build_request(task)
    assert request["model"] == "gpt-6.1-terra"
    text = json.dumps(request, ensure_ascii=False)
    assert task["source_text"] in text
    assert task["target_text"] in text
    for question in ["Are germline trials permitted?", "What clinical trials are prohibited?",
                     "Who must be notified, and when?"]:
        assert text.count(question) == 1
    assert request["text"]["format"]["strict"] is True


@pytest.mark.parametrize("field", ["source_language", "target_language"])
@pytest.mark.parametrize("language", ["ar", "ru", "xx"])
def test_unsupported_language_cannot_reach_generation(task, field, language):
    task[field] = language
    with pytest.raises(ValueError):
        terra.build_request(task)


@pytest.mark.parametrize("language", ["en", "de", "es", "fr", "zh"])
def test_five_supported_languages_can_be_used_as_targets(task, language):
    if language == "en":
        task["source_language"] = "fr"
    task["target_language"] = language
    assert terra.build_request(task)["model"] == "gpt-6.1-terra"


@pytest.mark.parametrize("model", ["gpt-6.1-terra", "gpt-6.1-terra-2026-11-05"])
def test_same_terra_model_and_dated_snapshot_validate(task, model):
    rows, issues = terra.validate_response(task, response_for(model=model))
    assert issues == []
    assert len(rows) == 2
    assert all(row["status"] == "matched" and row["exact_match"] for row in rows)


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6.1-terra-pro", "gpt-6x1-terra"])
def test_other_model_cannot_be_accepted_as_terra(task, model):
    rows, issues = terra.validate_response(task, response_for(model=model))
    assert issues
    assert not any(row["status"] == "matched" for row in rows)


@pytest.mark.parametrize("quote", ["", " ", "Aucun essai germinal ne peut être conduit. "])
def test_empty_or_modified_quotation_is_not_an_exact_match(task, quote):
    # The padded case uses an extra trailing newline so it cannot be a literal
    # occurrence of the surrounding inter-sentence space.
    if quote.endswith(" ") and quote.strip():
        quote += "\n"
    rows, issues = terra.validate_response(task, response_for(first_quote=quote))
    assert issues
    assert rows[0]["status"] != "matched"


def test_changing_question_changes_paid_request_identity(task):
    changed = copy.deepcopy(task)
    changed["anchors"][0]["qa_links"][0]["question"] = "Is the trial prohibited?"
    assert terra.request_hash(changed) != terra.request_hash(task)


def test_changing_evidence_changes_paid_request_identity(task):
    changed = copy.deepcopy(task)
    changed["anchors"][0]["text"] = "germline trials"
    assert terra.request_hash(changed) != terra.request_hash(task)


def test_direct_client_cannot_accidentally_use_openrouter_environment(monkeypatch):
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setenv("OPENAI_API_KEY", "direct-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr("openai.OpenAI", constructor)
    terra._client()
    assert captured.get("api_key") == "direct-test-key"
    assert captured.get("base_url", "").rstrip("/") == "https://api.openai.com/v1"
    assert captured["max_retries"] == 0


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, **_kwargs):
        return copy.deepcopy(self.payload)


class FakeResponses:
    def __init__(self, outputs, on_call=None):
        self.outputs = list(outputs)
        self.requests = []
        self.on_call = on_call

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        if self.on_call:
            self.on_call(len(self.requests))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return FakeResponse(output)


class FakeClient:
    def __init__(self, *outputs, on_call=None):
        self.responses = FakeResponses(outputs, on_call)


class FakeStore:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.events = []
        self.results = None

    def record_attempt(self, envelope, rows):
        # Durable raw responses must exist before the database/CSV journal is
        # updated, making an interrupted journal update recoverable locally.
        saved = [json.loads(path.read_text()) for path in (self.run_dir / "attempts").glob("*.json")]
        assert any(item.get("request_hash") == envelope["request_hash"] for item in saved)
        self.events.append(("attempt", copy.deepcopy(envelope), copy.deepcopy(rows)))

    def record_results(self, task, rows):
        self.events.append(("results", task["task_id"], copy.deepcopy(rows)))
        self.results = copy.deepcopy(rows)

    def export(self):
        self.events.append(("export",))

    def close(self):
        self.events.append(("close",))


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(terra.time, "sleep", lambda _delay: None)


def test_success_persists_raw_usage_and_final_qa_links_before_return(task, tmp_path):
    client = FakeClient(response_for())
    store = FakeStore(tmp_path)
    rows = terra.run_task(client, task, tmp_path, store)
    assert len(client.responses.requests) == 1
    assert store.events[0][0] == "attempt"
    assert store.events[0][1]["response"]["usage"] == response_for()["usage"]
    assert store.results == rows
    assert rows[0]["qa_ids"] == ["q1", "q2"]
    assert all(row["status"] == "matched" for row in rows)


def test_retry_cannot_start_until_previous_response_and_rows_are_persisted(task, tmp_path):
    store = FakeStore(tmp_path)

    def check_before_request(number):
        if number == 2:
            attempts = [event for event in store.events if event[0] == "attempt"]
            assert len(attempts) == 1
            assert attempts[0][2][0]["status"] == "invalid"

    client = FakeClient(response_for(first_quote="invented"), response_for(), on_call=check_before_request)
    rows = terra.run_task(client, task, tmp_path, store)
    assert len(client.responses.requests) == 2
    assert rows[0]["first_status"] == "invalid"
    assert rows[0]["status"] == "matched"
    assert rows[0]["attempt_count"] == 2


def test_saved_success_replays_without_a_second_paid_request(task, tmp_path):
    initial = terra.run_task(FakeClient(response_for()), task, tmp_path, FakeStore(tmp_path))
    replay_client = FakeClient()
    replay_store = FakeStore(tmp_path)
    replay = terra.run_task(replay_client, task, tmp_path, replay_store)
    assert replay_client.responses.requests == []
    assert replay == initial
    assert replay_store.results == initial


@pytest.mark.parametrize("failure_stage", ["attempt", "results"])
def test_journal_failure_after_paid_call_can_recover_from_raw_response(task, tmp_path, failure_stage):
    class BrokenStore(FakeStore):
        def record_attempt(self, envelope, rows):
            super().record_attempt(envelope, rows)
            if failure_stage == "attempt":
                raise OSError("Journal disk failure")

        def record_results(self, task, rows):
            super().record_results(task, rows)
            if failure_stage == "results":
                raise OSError("Result snapshot failure")

    first_client = FakeClient(response_for())
    with pytest.raises(OSError):
        terra.run_task(first_client, task, tmp_path, BrokenStore(tmp_path))
    assert len(first_client.responses.requests) == 1
    replay_client = FakeClient()
    rows = terra.run_task(replay_client, task, tmp_path, FakeStore(tmp_path))
    assert replay_client.responses.requests == []
    assert all(row["status"] == "matched" for row in rows)


def test_exhausted_invalid_output_is_visible_after_three_attempts(task, tmp_path):
    invalid = response_for(first_quote="not in the document")
    client = FakeClient(invalid, invalid, invalid)
    store = FakeStore(tmp_path)
    rows = terra.run_task(client, task, tmp_path, store)
    assert len(client.responses.requests) == 3
    assert len([event for event in store.events if event[0] == "attempt"]) == 3
    assert rows[0]["status"] == "invalid"
    assert rows[0]["attempt_count"] == 3
    assert store.results == rows


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_nonretryable_api_rejection_is_saved_and_stops_immediately(task, tmp_path, status):
    error = RuntimeError("Request or account rejected")
    error.status_code = status
    client = FakeClient(error)
    store = FakeStore(tmp_path)
    with pytest.raises(terra.legacy.AccessBlocked):
        terra.run_task(client, task, tmp_path, store)
    assert len(client.responses.requests) == 1
    attempts = [event for event in store.events if event[0] == "attempt"]
    assert len(attempts) == 1
    assert attempts[0][1]["error"]["status"] == status


def test_stop_before_scheduling_never_calls_provider(task, tmp_path):
    stop = Event()
    stop.set()
    client = FakeClient()
    with pytest.raises(InterruptedError):
        terra.run_task(client, task, tmp_path, FakeStore(tmp_path), stop_event=stop)
    assert client.responses.requests == []


def test_stop_during_request_keeps_completed_response_before_return(task, tmp_path):
    stop = Event()
    client = FakeClient(response_for(), on_call=lambda _number: stop.set())
    store = FakeStore(tmp_path)
    rows = terra.run_task(client, task, tmp_path, store, stop_event=stop)
    assert len(client.responses.requests) == 1
    assert all(row["status"] == "matched" for row in rows)
    assert store.results == rows


def test_stop_after_invalid_response_keeps_attempt_and_prevents_retry(task, tmp_path):
    stop = Event()
    client = FakeClient(response_for(first_quote="invented"), on_call=lambda _number: stop.set())
    store = FakeStore(tmp_path)
    with pytest.raises(InterruptedError):
        terra.run_task(client, task, tmp_path, store, stop_event=stop)
    assert len(client.responses.requests) == 1
    assert len([event for event in store.events if event[0] == "attempt"]) == 1


def qa_inventory(task):
    result = []
    for anchor in task["anchors"]:
        for qa_id in anchor["qa_ids"]:
            result.append({
                "qa_id": qa_id, "corpus": task["corpus"], "question": "Test question",
                "answer": anchor["text"], "source_unit_id": task["source_unit_id"],
                "source_language": task["source_language"], "question_language": task["source_language"],
                "anchor_id": anchor["anchor_id"], "source_evidence_text": anchor["text"],
                "source_evidence_origin": "answer", "eligibility_status": "eligible",
                "metadata_json": "{}", "evidence_all_languages": {
                    task["source_language"]: [{"text": anchor["text"], "source_unit_id": task["source_unit_id"]}]
                },
                "expected_targets": [{"language": task["target_language"],
                                      "source_unit_id": task["target_unit_id"], "task_id": task["task_id"]}],
            })
    return result


def test_repaired_credentials_resume_with_real_journal_without_reusing_attempt_identity(task, tmp_path):
    from clir_bench.alignment.terra_store import AlignmentStore

    error = RuntimeError("Organization access denied")
    error.status_code = 401
    store = AlignmentStore(tmp_path, qa_inventory(task))
    try:
        with pytest.raises(terra.legacy.AccessBlocked):
            terra.run_task(FakeClient(error), task, tmp_path, store)
        repaired = FakeClient(response_for())
        rows = terra.run_task(repaired, task, tmp_path, store)
        assert len(repaired.responses.requests) == 1
        assert all(row["status"] == "matched" for row in rows)
        with (tmp_path / "calls.csv").open(newline="") as stream:
            calls = list(csv.DictReader(stream))
        assert len(calls) == 2
        assert json.loads(calls[0]["error_json"])["status"] == 401
        assert calls[1]["response_id"] == "resp-terra-test"
    finally:
        store.close()


def test_real_calls_csv_contains_invalid_attempt_before_next_paid_request(task, tmp_path):
    from clir_bench.alignment.terra_store import AlignmentStore

    store = AlignmentStore(tmp_path, qa_inventory(task))

    def before_call(number):
        if number == 2:
            with (tmp_path / "calls.csv").open(newline="") as stream:
                calls = list(csv.DictReader(stream))
            assert len(calls) == 1
            assert json.loads(calls[0]["results_json"])[0]["status"] == "invalid"

    client = FakeClient(response_for(first_quote="fabricated"), response_for(), on_call=before_call)
    try:
        rows = terra.run_task(client, task, tmp_path, store)
        assert len(client.responses.requests) == 2
        assert all(row["status"] == "matched" for row in rows)
    finally:
        store.close()


def write_inventory(tmp_path, task):
    (tmp_path / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "qa_index.jsonl").write_text(
        "".join(json.dumps(qa) + "\n" for qa in qa_inventory(task))
    )


def direct_model_client(*responses):
    client = FakeClient(*responses)
    client.models = SimpleNamespace(retrieve=lambda model: SimpleNamespace(
        id=model, model_dump=lambda **_kwargs: {"id": model}
    ))
    return client


def test_full_run_saves_completed_status_and_real_csv_outputs(task, tmp_path):
    write_inventory(tmp_path, task)
    result = terra.run(tmp_path, workers=1, client=direct_model_client(response_for()))
    assert result["status"] == "completed"
    assert result["completed_this_session"] == 1
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "completed"
    with (tmp_path / "calls.csv").open(newline="") as stream:
        calls = list(csv.DictReader(stream))
    assert len(calls) == 1
    assert calls[0]["response_id"] == "resp-terra-test"


def test_another_process_lock_blocks_run_before_any_api_request(task, tmp_path):
    write_inventory(tmp_path, task)
    client = direct_model_client()
    with (tmp_path / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another alignment process"):
            terra.run(tmp_path, workers=1, client=client)
        assert client.responses.requests == []
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_export_error_still_closes_store_restores_signals_and_releases_lock(task, tmp_path, monkeypatch):
    write_inventory(tmp_path, task)

    class BrokenExportStore(FakeStore):
        def export(self):
            raise OSError("Snapshot destination unavailable")

    store = BrokenExportStore(tmp_path)
    monkeypatch.setattr("clir_bench.alignment.terra_store.AlignmentStore", lambda *_args: store)
    original_int = signal.getsignal(signal.SIGINT)
    original_term = signal.getsignal(signal.SIGTERM)
    with pytest.raises(OSError):
        terra.run(tmp_path, workers=1, client=direct_model_client(response_for()))
    assert ("close",) in store.events
    assert signal.getsignal(signal.SIGINT) is original_int
    assert signal.getsignal(signal.SIGTERM) is original_term
    with (tmp_path / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
