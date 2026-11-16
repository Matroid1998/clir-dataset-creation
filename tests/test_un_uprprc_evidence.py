"""Prepared UN blocks retain exact original character locations through publication."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clir_bench.domains.legal.qac.evidence import CONTRACT, EvidenceSource, validate_spans
from clir_bench.domains.legal.un import UN_LANGUAGES
from clir_bench.publishing.evidence import SPAN_TYPE, align_spans, saved_spans
from clir_bench.publishing.generations import prepare
from clir_bench.publishing.records import normalize
from clir_bench.publishing.sources import SourceIndex, load_un, source_row


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def character_source():
    block = "same α support"
    document = f"Header\r\n{block}\n\nOutside accepted blocks.\n\n{block}\r\nFooter"
    start = document.rindex(block)
    return document, EvidenceSource(
        "target", "uprprc:revision:document#2", "en", block, block, "block",
        doc_id="uprprc:revision:document", location_kind="document_chars",
        block_document_start=start, block_document_end=start + len(block),
        document_text_sha256=digest(document),
    )


def canonical(source):
    return validate_spans(
        {"answer_is_translation": False, "evidence_spans": [
            {"source_ref": "target", "language": "en", "text": "α support"},
        ]},
        {("target", "en"): source}, ["target"],
    )


def test_generation_saves_original_document_location_and_block_hash(character_source):
    _, source = character_source
    span, = canonical(source)["evidence_spans"]
    assert span == {
        "source_ref": "target", "source_id": source.source_id, "language": "en",
        "text": "α support", "start": 5, "end": len(source.full_text),
        "source_text_sha256": digest(source.full_text), "doc_id": source.doc_id,
        "location_kind": "document_chars", "block_document_start": source.block_document_start,
        "block_document_end": source.block_document_end,
        "document_text_sha256": source.document_text_sha256,
    }
    assert "line_start" not in span and "line_end" not in span


@pytest.mark.parametrize("changes", [
    {"block_document_start": None}, {"block_document_start": True},
    {"block_document_start": -1}, {"block_document_end": 1},
    {"block_document_end": 1000}, {"document_text_sha256": ""},
    {"document_text_sha256": "z" * 64}, {"original_sha256": "0" * 64},
    {"doc_id": ""}, {"line_start": 1, "line_end": 1},
    {"location_kind": "unknown"}, {"location_kind": ""},
])
def test_generation_rejects_invalid_or_mixed_character_locations(character_source, changes):
    _, source = character_source
    with pytest.raises(ValueError):
        canonical(replace(source, **changes))


@pytest.fixture
def character_index(tmp_path, character_source):
    document, source = character_source
    index = SourceIndex(tmp_path / "index.sqlite")
    identity = {"manifest_path": "/prepared/manifest.json", "manifest_sha256": "a" * 64,
                "database_sha256": "b" * 64}
    index.add(source_row("un", source.doc_id, source.doc_id, "en", "document", "", document,
                         {"source_manifest": identity}))
    raw = {
        "doc_id": source.doc_id, "block_id": source.source_id, "target_id": source.source_id,
        "question_language": "en", "question": "Which support?", "answer": "α support",
        "mode": "lookup", "generator_model_id": "model", "evidence_contract": CONTRACT,
        "un_dataset_identity": json.dumps(identity),
        "source_locations": {"en": {
            "start": source.block_document_start, "end": source.block_document_end,
            "document_sha256": source.document_text_sha256,
        }},
        **canonical(source),
    }
    yield index, raw, document
    index.close()


def test_character_export_uses_saved_block_with_exact_full_original_text(character_index):
    index, raw, document = character_index
    row = normalize("un", raw, {}, index)
    span, = row["evidence_spans"]
    assert span["document_start"] == document.rindex("α support")
    assert span["document_start"] != document.index("α support")
    assert span["document_end"] == span["document_start"] + len("α support")
    assert row["document_text"] == document
    assert row["support_start"] == span["document_start"]
    assert document[row["support_start"]:row["support_end"]] == row["supporting_text"]
    assert pa.array([row["evidence_spans"]], type=pa.list_(SPAN_TYPE)).to_pylist()[0][0][
        "document_text_sha256"
    ] == digest(document)


@pytest.mark.parametrize("changes,error", [
    ({"document_text_sha256": "0" * 64}, "differs from saved"),
    ({"source_text_sha256": "0" * 64}, "source text changed"),
    ({"block_document_end": 1000}, "differs from saved"),
    ({"block_document_start": 0}, "differs from saved"),
    ({"block_document_start": True}, "must be an integer"),
    ({"start": 0, "end": 9}, "offsets do not reproduce"),
    ({"line_start": 1}, "cannot mix"),
    ({"location_kind": ""}, "incomplete"),
    ({"document_text_sha256": "bad"}, "invalid evidence"),
    ({"source_text_sha256": ""}, "invalid evidence"),
])
def test_export_rejects_stale_or_invalid_character_evidence(character_index, changes, error):
    index, raw, _ = character_index
    altered = {**raw, "evidence_spans": [{**raw["evidence_spans"][0], **changes}]}
    with pytest.raises(ValueError, match=error):
        align_spans("un", altered, index)


def test_full_document_hash_detects_changes_outside_the_block(character_index):
    index, raw, _ = character_index
    altered = {**raw, "source_locations": None}
    span = {**raw["evidence_spans"][0], "document_text_sha256": "0" * 64}
    with pytest.raises(ValueError, match="document text changed"):
        align_spans("un", {**altered, "evidence_spans": [span]}, index)


def test_outside_character_range_rejected_without_saved_raw_location(character_index):
    index, raw, _ = character_index
    span = {**raw["evidence_spans"][0], "block_document_end": 1000}
    with pytest.raises(ValueError, match="outside its document"):
        align_spans("un", {**raw, "source_locations": None, "evidence_spans": [span]}, index)


@pytest.mark.parametrize("changes", [
    {"line_start": 1}, {"block_document_start": True}, {"block_document_start": "5"},
    {"source_locations": {"en": {"start": 0}}}, {"source_locations": {"fr": {}}},
    {"source_locations": []}, {"source_locations": {"en": {"start": True}}},
])
def test_raw_target_character_location_is_checked(character_index, changes):
    index, raw, _ = character_index
    with pytest.raises(ValueError):
        align_spans("un", {**raw, **changes}, index)


def test_character_metadata_survives_csv_json_fields(character_index):
    index, raw, _ = character_index
    csv = {**raw, "evidence_spans": json.dumps(raw["evidence_spans"]),
           "source_locations": json.dumps(raw["source_locations"])}
    assert align_spans("un", csv, index) == align_spans("un", raw, index)
    assert saved_spans({"answer": "legacy"}) == []


def test_manifest_export_keeps_full_documents_and_six_language_families(tmp_path, monkeypatch):
    from clir_bench.domains.legal.un import uprprc_index

    docs = [
        {"doc_id": "uprprc:r:one", "symbol": "A/RES/80/1", "title": "Resolution"},
        {"doc_id": "uprprc:r:two", "symbol": "A/80/2", "title": "Annual report"},
    ]
    texts = {(doc["doc_id"], language): f"{language}\r\nFull source\n\nOutside blocks α."
             for doc in docs for language in UN_LANGUAGES}
    closed = []
    prepared = SimpleNamespace(
        iter_documents=lambda: iter(docs), document_text=lambda doc, language: texts[doc, language],
        identity={"manifest_sha256": "a" * 64, "database_sha256": "b" * 64},
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(uprprc_index, "UPRPRCIndex", lambda path: prepared)
    rows = []
    load_un(SimpleNamespace(add=rows.append), {
        "dataset_manifest": "prepared/manifest.json", "languages": list(UN_LANGUAGES),
        "document_genres": ["resolution"],
    }, tmp_path, None)
    assert closed == [True]
    assert len(rows) == 6
    assert {row["source_id"] for row in rows} == {docs[0]["doc_id"]}
    for row in rows:
        assert row["text"] == texts[row["source_id"], row["language"]]
        metadata = json.loads(row["metadata_json"])
        assert metadata["source_manifest"] == prepared.identity
        assert "line_start" not in metadata


def test_manifest_export_rejects_incomplete_multilingual_document(tmp_path, monkeypatch):
    from clir_bench.domains.legal.un import uprprc_index

    prepared = SimpleNamespace(
        iter_documents=lambda: iter([{"doc_id": "uprprc:r:one"}]),
        document_text=lambda doc, language: "Original full source" if language != "ru" else "",
        identity={}, close=lambda: None,
    )
    monkeypatch.setattr(uprprc_index, "UPRPRCIndex", lambda path: prepared)
    with pytest.raises(ValueError, match="all six languages"):
        load_un(SimpleNamespace(add=lambda row: None), {
            "dataset_manifest": "manifest.json", "languages": ["en"],
        }, tmp_path, None)


def test_prepared_document_and_character_evidence_export_end_to_end(tmp_path, character_source):
    document, source = character_source
    database = tmp_path / "sources.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE docs(doc_id TEXT PRIMARY KEY, payload TEXT);
            CREATE TABLE texts(doc_id TEXT, language TEXT, text TEXT, sha256 TEXT);
            CREATE TABLE blocks(doc_id TEXT, block_index INTEGER, block_id TEXT, payload TEXT);
        """)
        connection.execute("INSERT INTO docs VALUES (?,?)", (source.doc_id, json.dumps({
            "doc_id": source.doc_id, "symbol": "A/RES/80/1", "title": "Resolution",
            "year": 2025, "char_count": len(document), "n_blocks": 3,
            "n_targets": 1, "target_idxs": [2],
        })))
        for language in UN_LANGUAGES:
            text = document if language == "en" else f"{language}\n{document}"
            connection.execute("INSERT INTO texts VALUES (?,?,?,?)", (
                source.doc_id, language, text, digest(text),
            ))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "un-uprprc-v1", "version": 1, "database": database.name,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
    }))
    raw = {
        "doc_id": source.doc_id, "block_id": source.source_id, "target_id": source.source_id,
        "question_language": "en", "question": "Which support?", "answer": "α support",
        "mode": "lookup", "generator_model_id": "model", **canonical(source),
        "source_locations": {"en": {
            "start": source.block_document_start, "end": source.block_document_end,
            "document_sha256": source.document_text_sha256,
        }},
    }
    from clir_bench.domains.legal.un.uprprc_index import UPRPRCIndex

    raw["un_dataset_identity"] = json.dumps(UPRPRCIndex(tmp_path / "manifest.json").identity)
    with sqlite3.connect(tmp_path / "run.sqlite") as connection:
        connection.execute("CREATE TABLE outcomes(task,status,rows,error)")
        connection.execute("INSERT INTO outcomes VALUES (?,?,?,?)", (
            "one", "completed", json.dumps([raw]), "",
        ))
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"version": 1, "corpora": {"un": {
        "generation_sqlite": ["run.sqlite"], "dataset_manifest": "manifest.json",
        "languages": list(UN_LANGUAGES), "document_genres": ["resolution"],
    }}}))
    output = tmp_path / "export"
    report = prepare(input_path, output)
    assert report["configs"] == {"qas": 1, "evidence": 1, "source_qrels": 6, "corpus": 6}
    qa, = pq.read_table(output / "data/qas").to_pylist()
    evidence, = pq.read_table(output / "data/evidence").to_pylist()
    corpus = pq.read_table(output / "data/corpus").to_pylist()
    assert {row["language"] for row in corpus} == set(UN_LANGUAGES)
    assert qa["document_text"] == document
    assert evidence["start"] == document.rindex("α support")
    assert qa["evidence_spans"][0]["document_text_sha256"] == digest(document)
    # Changing an unused language must not silently create a different set of
    # translations for an otherwise identical English quotation.
    changed_text = "Changed Russian original"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE texts SET text=?,sha256=? WHERE doc_id=? AND language='ru'", (
            changed_text, digest(changed_text), source.doc_id,
        ))
    prepared_manifest = tmp_path / "manifest.json"
    changed_manifest = json.loads(prepared_manifest.read_text())
    changed_manifest["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    prepared_manifest.write_text(json.dumps(changed_manifest))
    with pytest.raises(ValueError, match="dataset identity differs"):
        prepare(input_path, tmp_path / "changed-export")
    assert not (tmp_path / "changed-export").exists()


@pytest.mark.parametrize("value", [
    None, "", "{}", "[]", json.dumps({"manifest_path": "different"}),
    json.dumps({"manifest_path": "/prepared/manifest.json", "manifest_sha256": "a" * 64,
                "database_sha256": "c" * 64}),
])
def test_publication_rejects_missing_or_different_multilingual_source_identity(character_index, value):
    index, raw, _ = character_index
    with pytest.raises(ValueError, match="(?i)(identity|dataset|manifest|source store)"):
        normalize("un", {**raw, "un_dataset_identity": value}, {}, index)


def test_publication_checks_dataset_identity_for_rows_without_evidence_contract(character_index):
    index, raw, _ = character_index
    legacy_contract = {key: value for key, value in raw.items()
                       if key not in {"evidence_contract", "evidence_spans", "un_dataset_identity"}}
    with pytest.raises(ValueError, match="missing its pinned UN dataset identity"):
        normalize("un", legacy_contract, {}, index)
