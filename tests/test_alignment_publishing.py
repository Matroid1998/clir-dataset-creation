"""Publishing adds evidence without changing original QA or corpus semantics."""

import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clir_bench.publishing.alignment_update import (
    CONTRACT,
    LANGUAGES,
    NEW_FIELDS,
    prepare_alignment_update,
)


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    old, run, output = tmp_path / "old", tmp_path / "run", tmp_path / "patch"
    old.mkdir()
    run.mkdir()
    schema = pa.schema([
        pa.field("_id", pa.string(), metadata={b"identity": b"unchanged"}),
        ("corpus", pa.string()), ("question", pa.string()), ("answer", pa.string()),
        ("question_language", pa.string()), ("source_unit_id", pa.string()),
        ("metadata_json", pa.string()), ("legacy_list", pa.list_(pa.string())),
        ("legacy_number", pa.int64()), ("legacy_flag", pa.bool_()),
    ], metadata={b"custom": b"preserve-schema-metadata"})
    qas = [
        {"_id": "q2", "corpus": "eurlex", "question": "Excluded?", "answer": "not verbatim",
         "question_language": "en", "source_unit_id": "doc:en", "metadata_json": '{"raw":"unchanged"}',
         "legacy_list": [], "legacy_number": None, "legacy_flag": False},
        {"_id": "q1", "corpus": "eurlex", "question": 'Which "rule", applies?\nWhy?',
         "answer": "The rule applies unless exempt.", "question_language": "en", "source_unit_id": "doc:en",
         "metadata_json": '{ "raw": [1, true, null] }', "legacy_list": ["a", "b"],
         "legacy_number": 7, "legacy_flag": True},
    ]
    for i, row in enumerate(qas):
        path = old / f"data/qas/train-{i:05}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([row], schema=schema), path)
    quote = "La règle s’applique,\nsauf exemption."
    configs = {
        "corpus": [{"_id": "doc:en", "language": "en", "text": qas[1]["answer"]},
                   {"_id": "doc:fr", "language": "fr", "text": "Préambule. " + quote}],
        "evidence": [{"query_id": "q1", "text": "legacy evidence"}],
        "source_qrels": [{"query_id": "q1", "corpus_id": "doc:fr", "score": 1}],
    }
    for config, rows in configs.items():
        path = old / f"data/{config}/train-00000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
    (old / "README.md").write_text("---\nconfigs: []\n---\n\n# Original title\n\nOriginal content.\n")
    dump(old / "report.json", {"original_report": {"preserve": True}})
    manifest = {"version": 1, "selected_corpora": ["eurlex"], "corpus_scope": "full",
                "configs": {"qas": 2, **{k: len(v) for k, v in configs.items()}},
                "historical_metadata": "preserved", "files": {}}
    for path in old.rglob("*"):
        if path.is_file():
            manifest["files"][path.relative_to(old).as_posix()] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    dump(old / "manifest.json", manifest)
    frozen = [{"qa_id": row["_id"], **{key: row[key] for key in (
        "corpus", "question", "answer", "question_language", "source_unit_id")},
        "eligibility_status": "eligible" if row["_id"] == "q1" else "no_exact_source_span",
        "anchor_id": "a1" if row["_id"] == "q1" else None,
        "source_evidence_text": qas[1]["answer"] if row["_id"] == "q1" else None,
        "expected_targets": [{"language": "fr", "task_id": "t1", "source_unit_id": "doc:fr"}]
        if row["_id"] == "q1" else []} for row in qas]
    (run / "qa_index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in frozen))
    dump(run / "run.json", {"status": "completed", "model": "gpt-6.1-terra", "provider": "openai"})
    original = {"source_unit_id": "doc:en", "language": "en", "text": qas[1]["answer"],
                "anchor_id": "a1", "origin": "answer", "alignment_id": "source:doc:en:a1",
                "validation_status": "source_exact", "match_status": "exact", "is_original": True,
                "alignment_status": "source_exact"}
    translated = {"source_unit_id": "doc:fr", "text": quote, "anchor_id": "a1",
                  "origin": "cross_language_alignment", "alignment_id": "t1:a1",
                  "validation_status": "pending_review", "match_status": "exact", "is_original": False,
                  "task_id": "t1", "request_hash": "hash1", "model": "gpt-6.1-terra",
                  "api_model": "gpt-6.1-terra", "provider": "openai"}
    status = {key: translated[key] for key in ("source_unit_id", "anchor_id", "alignment_id", "task_id",
                                              "request_hash", "model", "api_model", "provider", "validation_status")}
    status.update(status="matched", exact_match=True, reason="")
    csv_rows = []
    for row in reversed(frozen):
        csv_rows.append({key: row[key] for key in ("qa_id", "corpus", "question", "answer",
                                                  "question_language", "source_unit_id", "eligibility_status")})
        csv_rows[-1]["evidence_all_languages"] = {"en": [original], "fr": [translated]} if row["qa_id"] == "q1" else {}
        csv_rows[-1]["evidence_alignment_status"] = {"en": [{"source_unit_id": "doc:en", "anchor_id": "a1",
                                                            "status": "source_exact"}], "fr": [status]} if row["qa_id"] == "q1" else {}
    write_csv(run, csv_rows)
    return old, run, output, csv_rows


def write_csv(run, rows):
    with (run / "qa_evidence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def prepare(release):
    old, run, output, _ = release
    return prepare_alignment_update(old, run, output, corpus="eurlex", repo_id="user/eurlex-qa", revision="pinned-revision")


def test_native_roundtrip_preserves_old_cells_schema_shard_order_and_all_span_metadata(release):
    old, _, output, rows = release
    result = prepare(release)
    assert result["evidence_alignment"]["qa_count"] == 2
    assert len(list((output / "data/qas").glob("*.parquet"))) == 2
    for path in (old / "data/qas").glob("*.parquet"):
        before = pq.read_table(path)
        after = pq.read_table(output / path.relative_to(old))
        assert after.select(before.column_names).equals(before, check_metadata=True)
        assert set(after.column_names) - set(before.column_names) == {field.name for field in NEW_FIELDS}
    updated = pq.read_table(output / "data/qas/train-00001.parquet").to_pylist()[0]
    assert set(updated["evidence_all_languages"]) == set(LANGUAGES)
    assert updated["evidence_zh"] == []
    assert updated["evidence_fr"] == [rows[0]["evidence_all_languages"]["fr"][0]["text"]]
    for key, value in rows[0]["evidence_all_languages"]["fr"][0].items():
        assert updated["evidence_all_languages"]["fr"][0][key] == value
    assert updated["evidence_alignment_status"]["fr"][0]["exact_match"] is True
    assert updated["evidence_all_languages"]["fr"][0]["validation_status"] == "pending_review"
    assert updated["evidence_alignment_contract"] == CONTRACT
    excluded = pq.read_table(output / "data/qas/train-00000.parquet").to_pylist()[0]
    assert all(not spans for spans in excluded["evidence_all_languages"].values())


def test_untouched_manifest_entries_configs_and_report_are_preserved(release):
    old, _, output, _ = release
    prepare(release)
    before = json.loads((old / "manifest.json").read_text())
    after = json.loads((output / "manifest.json").read_text())
    assert after["configs"] == before["configs"]
    assert after["historical_metadata"] == before["historical_metadata"]
    for relative in before["files"]:
        if relative.startswith(("data/corpus/", "data/source_qrels/", "data/evidence/")):
            assert after["files"][relative] == before["files"][relative]
            assert not (output / relative).exists()
    assert json.loads((output / "report.json").read_text())["original_report"] == {"preserve": True}
    assert after["evidence_alignment"]["base_revision"] == "pinned-revision"
    assert "Original content." in (output / "README.md").read_text()
    assert '["evidence_all_languages"]["en"][0]["text"]' in (output / "README.md").read_text()
    with (output / "exports/qa_evidence_readable.csv").open(newline="") as stream:
        readable = list(csv.DictReader(stream))
    assert [row["qa_id"] for row in readable] == ["q2", "q1"]
    assert readable[0]["evidence_fr"] == ""
    assert readable[1]["evidence_fr"] == "La règle s’applique,\nsauf exemption."


@pytest.mark.parametrize("problem", ["missing_qa", "changed_qa", "unsupported_language", "pending", "unknown_metadata", "bad_quote"])
def test_invalid_or_incomplete_alignment_data_aborts_before_writing(release, problem):
    _, run, output, rows = release
    if problem == "missing_qa":
        rows.pop()
    elif problem == "changed_qa":
        rows[0]["question"] = "edited question"
    elif problem == "unsupported_language":
        rows[0]["evidence_all_languages"]["ar"] = []
    elif problem == "pending":
        rows[0]["evidence_alignment_status"]["fr"][0]["status"] = "pending"
    elif problem == "unknown_metadata":
        rows[0]["evidence_all_languages"]["fr"][0]["new_untyped_field"] = "must not disappear"
    elif problem == "bad_quote":
        rows[0]["evidence_all_languages"]["fr"][0]["text"] = "invented target quotation"
    write_csv(run, rows)
    with pytest.raises(ValueError):
        prepare(release)
    assert not output.exists()


def test_run_must_be_complete(release):
    _, run, _, _ = release
    dump(run / "run.json", {"status": "running"})
    with pytest.raises(ValueError, match="not completed"):
        prepare(release)
