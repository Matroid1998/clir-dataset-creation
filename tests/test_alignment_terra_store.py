"""Per-call CSV checkpoints survive interruption without claiming semantic approval."""

import csv
import json
import sqlite3

import pytest

from clir_bench.alignment.terra_store import AlignmentStore


def fixtures():
    question = 'Which "rule", applies?\nPlease explain.'
    source = 'The rule applies, unless exempt.\nSecond line.'
    quote = 'La règle s’applique, sauf exemption.\nDeuxième ligne.'
    qa = {
        "qa_id": "q1", "corpus": "eurlex", "question": question, "answer": source,
        "question_language": "en", "source_unit_id": "doc:en", "source_language": "en",
        "anchor_id": "a1", "source_evidence_text": source, "source_evidence_origin": "answer",
        "eligibility_status": "eligible", "metadata_json": json.dumps({"persona": "practitioner"}),
        "evidence_all_languages": {"en": [{"source_unit_id": "doc:en", "text": source,
                                             "origin": "answer", "match_status": "exact"}]},
        "expected_targets": [{"language": "fr", "source_unit_id": "doc:fr", "task_id": "t1"}],
    }
    task = {
        "task_id": "t1", "corpus": "eurlex", "source_unit_id": "doc:en", "target_unit_id": "doc:fr",
        "source_language": "en", "target_language": "fr", "source_text": source,
        "target_text": "Préambule.\n" + quote,
        "anchors": [{"anchor_id": "a1", "text": source, "qa_ids": ["q1"]}],
    }
    row = {key: task[key] for key in ("task_id", "corpus", "source_unit_id", "target_unit_id",
                                     "source_language", "target_language")}
    row.update(alignment_id="t1:a1", anchor_id="a1", source_text=source, target_text=quote,
               status="matched", exact_match=True, qa_ids=["q1"], request_hash="h1",
               model="gpt-6.1-terra", provider="openai", attempt_count=1)
    envelope = {
        "task_id": "t1", "request_hash": "h1", "attempt": 1, "model": "gpt-6.1-terra",
        "provider": "openai", "source_language": "en", "target_language": "fr",
        "raw_response_path": "attempts/t1-h1-1.json",
        "response": {"id": "resp_1", "model": "gpt-6.1-terra", "status": "completed",
                     "usage": {"input_tokens": 100, "output_tokens": 20, "cost": .002},
                     "output": [{"type": "reasoning", "content": None},
                                {"type": "message", "content": [{"type": "output_text", "text": quote}]}]},
    }
    return qa, task, row, envelope


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_each_api_call_and_completed_task_have_immediate_csv_checkpoints(tmp_path):
    qa, task, row, envelope = fixtures()
    with AlignmentStore(tmp_path, [qa]) as store:
        initial = csv_rows(tmp_path / "qa_evidence.csv")
        assert json.loads(initial[0]["evidence_all_languages"]).keys() == {"en"}
        store.record_attempt(envelope, [row])
        calls = csv_rows(tmp_path / "calls.csv")
        assert len(calls) == 1
        assert calls[0]["response_id"] == "resp_1"
        assert json.loads(calls[0]["results_json"])[0]["target_text"] == row["target_text"]
        store.record_results(task, [row])
        updates = csv_rows(tmp_path / "qa_updates.csv")
        assert len(updates) == 1
        assert updates[0]["question"] == qa["question"]
        assert updates[0]["answer"] == qa["answer"]
        evidence = json.loads(updates[0]["evidence_all_languages"])
        assert evidence["fr"][0]["text"] == row["target_text"]
        assert evidence["fr"][0]["validation_status"] == "pending_review"
        assert evidence["en"][0]["validation_status"] == "source_exact"
        store.export()
        assert csv_rows(tmp_path / "qa_evidence.csv")[0]["evidence_all_languages"] == updates[0]["evidence_all_languages"]
        assert len(csv_rows(tmp_path / "results.csv")) == 1


def test_exact_flag_does_not_admit_modified_quote(tmp_path):
    qa, task, row, _ = fixtures()
    row["target_text"] = row["target_text"].replace("\n", " ")
    with AlignmentStore(tmp_path, [qa]) as store:
        store.record_results(task, [row])
        record = csv_rows(tmp_path / "qa_updates.csv")[0]
        assert "fr" not in json.loads(record["evidence_all_languages"])
        assert json.loads(record["evidence_alignment_status"])["fr"][0]["status"] == "invalid"


def test_failed_new_final_result_removes_previous_quote_but_preserves_original(tmp_path):
    qa, task, row, _ = fixtures()
    with AlignmentStore(tmp_path, [qa]) as store:
        store.record_results(task, [row])
        revised = {**row, "request_hash": "h2", "status": "not_found", "target_text": "", "exact_match": False}
        store.record_results(task, [revised])
        updates = csv_rows(tmp_path / "qa_updates.csv")
        assert len(updates) == 2
        assert int(updates[1]["revision"]) > int(updates[0]["revision"])
        assert set(json.loads(updates[-1]["evidence_all_languages"])) == {"en"}
        statuses = json.loads(updates[-1]["evidence_alignment_status"])
        assert len(statuses["fr"]) == 1
        assert statuses["fr"][0]["status"] == "not_found"
        assert store.load_results("t1", "h1") == []
        assert store.load_results("t1", "h2")[0]["status"] == "not_found"


def test_identical_resume_is_idempotent_and_corrupt_csvs_are_rebuilt(tmp_path):
    qa, task, row, envelope = fixtures()
    with AlignmentStore(tmp_path, [qa]) as store:
        store.record_attempt(envelope, [row])
        store.record_results(task, [row])
    for name in ("calls.csv", "qa_updates.csv", "qa_evidence.csv", "results.csv"):
        (tmp_path / name).write_text("interrupted,partial,garbage\n")
    with AlignmentStore(tmp_path, [qa]) as store:
        assert store.record_attempt(envelope, [row])["inserted"] is False
        assert store.record_results(task, [row])["inserted"] is False
        assert len(csv_rows(tmp_path / "calls.csv")) == 1
        assert len(csv_rows(tmp_path / "qa_updates.csv")) == 1
        assert len(csv_rows(tmp_path / "results.csv")) == 1


def test_commit_survives_csv_append_failure_and_is_repaired_on_restart(tmp_path, monkeypatch):
    qa, _, row, envelope = fixtures()
    store = AlignmentStore(tmp_path, [qa])
    original = store._append_csv

    def interrupted(*_args, **_kwargs):
        raise OSError("simulated disk interruption after SQLite commit")

    monkeypatch.setattr(store, "_append_csv", interrupted)
    with pytest.raises(OSError, match="interruption"):
        store.record_attempt(envelope, [row])
    # Simulate process death, not a graceful close/export.
    store._connection.close()
    store._closed = True
    monkeypatch.setattr(store, "_append_csv", original)
    with AlignmentStore(tmp_path, [qa]):
        assert len(csv_rows(tmp_path / "calls.csv")) == 1


def test_completed_qa_update_survives_crash_before_csv_flush(tmp_path, monkeypatch):
    qa, task, row, _ = fixtures()
    store = AlignmentStore(tmp_path, [qa])

    def interrupted(*_args, **_kwargs):
        raise OSError("interrupted after final task transaction")

    monkeypatch.setattr(store, "_append_csv", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        store.record_results(task, [row])
    store._connection.close()
    store._closed = True
    with AlignmentStore(tmp_path, [qa]) as restored:
        updates = csv_rows(tmp_path / "qa_updates.csv")
        assert len(updates) == 1
        assert json.loads(updates[0]["evidence_all_languages"])["fr"][0]["text"] == row["target_text"]
        assert restored.record_results(task, [row])["inserted"] is False


def test_multiple_evidence_entries_per_language_are_preserved_and_replaced_independently(tmp_path):
    qa, task, row, _ = fixtures()
    qa["expected_targets"].append({"language": "fr", "source_unit_id": "other:fr", "task_id": "t2"})
    second_task = {**task, "task_id": "t2", "target_unit_id": "other:fr"}
    second_row = {**row, "task_id": "t2", "alignment_id": "t2:a1", "target_unit_id": "other:fr"}
    with AlignmentStore(tmp_path, [qa]) as store:
        store.record_results(task, [row])
        store.record_results(second_task, [second_row])
        evidence = json.loads(csv_rows(tmp_path / "qa_updates.csv")[-1]["evidence_all_languages"])
        assert {entry["source_unit_id"] for entry in evidence["fr"]} == {"doc:fr", "other:fr"}
        store.record_results(task, [{**row, "request_hash": "h2", "status": "failed", "target_text": ""}])
        evidence = json.loads(csv_rows(tmp_path / "qa_updates.csv")[-1]["evidence_all_languages"])
        assert [entry["source_unit_id"] for entry in evidence["fr"]] == ["other:fr"]


def test_frozen_qa_and_schema_mismatch_refused(tmp_path):
    qa, _, _, _ = fixtures()
    with AlignmentStore(tmp_path, [qa]):
        pass
    changed = {**qa, "answer": "changed"}
    with pytest.raises(ValueError, match="inventory"):
        AlignmentStore(tmp_path, [changed])
    with sqlite3.connect(tmp_path / "alignment.sqlite") as connection:
        connection.execute("UPDATE metadata SET value='future_v2' WHERE key='schema_version'")
    with pytest.raises(ValueError, match="schema"):
        AlignmentStore(tmp_path, [qa])


def test_arabic_russian_and_changed_source_provenance_rejected(tmp_path):
    qa, task, row, envelope = fixtures()
    with AlignmentStore(tmp_path, [qa]) as store:
        with pytest.raises(ValueError, match="language"):
            store.record_results({**task, "target_language": "ru"}, [row])
        with pytest.raises(ValueError, match="language"):
            store.record_attempt({**envelope, "target_language": "ar"}, [row])
        with pytest.raises(ValueError, match="source anchor"):
            store.record_results({**task, "source_text": "different document"}, [row])
        with pytest.raises(ValueError, match="QA links"):
            store.record_results(task, [{**row, "qa_ids": ["q2"]}])


def test_ineligible_qas_remain_in_full_snapshot_without_invented_evidence(tmp_path):
    qa, _, _, _ = fixtures()
    excluded = {**qa, "qa_id": "q2", "anchor_id": None, "source_evidence_text": None,
                "eligibility_status": "no_exact_source_span", "evidence_all_languages": {},
                "expected_targets": []}
    with AlignmentStore(tmp_path, [qa, excluded]):
        records = csv_rows(tmp_path / "qa_evidence.csv")
        assert len(records) == 2
        row = next(row for row in records if row["qa_id"] == "q2")
        assert row["question"] == qa["question"]
        assert json.loads(row["evidence_all_languages"]) == {}


def test_attempts_remain_distinct_across_retries_but_conflicting_identity_fails(tmp_path):
    qa, _, row, envelope = fixtures()
    with AlignmentStore(tmp_path, [qa]) as store:
        store.record_attempt(envelope, [row])
        second = {**envelope, "attempt": 2, "response": {**envelope["response"], "id": "resp_2"}}
        store.record_attempt(second, [row])
        assert len(csv_rows(tmp_path / "calls.csv")) == 2
        with pytest.raises(ValueError, match="conflicting"):
            store.record_attempt({**envelope, "elapsed_seconds": 999}, [row])
