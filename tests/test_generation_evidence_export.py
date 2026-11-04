"""Future evidence spans keep their own sources and exact generation-to-corpus alignment."""

import hashlib
import json
import sqlite3

import pyarrow.parquet as pq
import pytest

from clir_bench.publishing.evidence import CONTRACT, align_spans, saved_spans
from clir_bench.publishing.generations import prepare
from clir_bench.publishing.records import normalize
from clir_bench.publishing.sources import SourceIndex, source_row


def span(source_ref, source_id, language, source_text, quote, **extra):
    start = source_text.index(quote)
    return {
        "source_ref": source_ref,
        "source_id": source_id,
        "language": language,
        "text": quote,
        "start": start,
        "end": start + len(quote),
        "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        **extra,
    }


def legal_qa(spans):
    return {
        "target_id": "eli/target",
        "document_id": "ACT1",
        "question_language": "en",
        "mode": "lookup",
        "generator_model_id": "model",
        "question": "What condition applies?",
        "answer": "Additional condition.",
        "answer_is_translation": True,
        "articles_involved": "1,ACT2:2",
        "articles_involved_eli": "eli/target,eli/reference",
        "multi_article": True,
        "cross_act": True,
        "evidence_contract": CONTRACT,
        "evidence_spans": spans,
    }


@pytest.fixture
def multi_article(tmp_path):
    texts = {
        "eli/target": "Before. Target rule. After.",
        "eli/reference": "Avant. Condition supplémentaire. Après.",
    }
    spans = [
        span("ACT2:2", "eli/reference", "fr", texts["eli/reference"], "Condition supplémentaire."),
        span("1", "eli/target", "en", texts["eli/target"], "Target rule."),
    ]
    articles = [
        {
            "eli_id": identifier,
            "celex_id": act,
            "language": language,
            "unit_type": "article",
            "article_number": number,
            "heading": "",
            "text": texts[identifier],
        }
        for identifier, act, language, number in (
            ("eli/target", "ACT1", "en", "1"),
            ("eli/reference", "ACT2", "fr", "2"),
        )
    ]
    (tmp_path / "articles.jsonl").write_text("".join(json.dumps(a) + "\n" for a in articles))
    raw = legal_qa(spans)
    with sqlite3.connect(tmp_path / "run.sqlite") as db:
        db.execute("CREATE TABLE outcomes(task, status, rows, error)")
        db.execute(
            "INSERT INTO outcomes VALUES (?,?,?,?)", ("one", "completed", json.dumps([raw]), "")
        )
    manifest = {
        "version": 1,
        "corpora": {
            "eurlex": {
                "generation_sqlite": ["run.sqlite"],
                "articles": "articles.jsonl",
                "languages": ["en"],
            }
        },
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(manifest))
    return path, raw, texts


def test_export_all_spans_and_load_their_foreign_language_sources(multi_article, tmp_path):
    manifest, raw, texts = multi_article
    output = tmp_path / "export"
    report = prepare(manifest, output)
    assert report["configs"] == {"qas": 1, "evidence": 2, "source_qrels": 2, "corpus": 2}
    (qa,) = pq.read_table(output / "data/qas").to_pylist()
    evidence = pq.read_table(output / "data/evidence").to_pylist()
    corpus = {r["_id"]: r for r in pq.read_table(output / "data/corpus").to_pylist()}
    assert qa["document_text"] == texts["eli/target"]
    assert qa["source_unit_id"] == "eurlex:eli/target:en"
    assert qa["support_source_unit_id"] == "eurlex:eli/reference:fr"
    assert qa["evidence_source_id"] == "eli/reference"
    assert qa["evidence_language"] == "fr"
    assert qa["evidence_is_translation"] is False
    assert qa["answer_is_translation"] is True
    assert qa["supporting_text_origin"] == "evidence_spans"
    assert qa["multi_article"] and qa["cross_act"]
    assert json.loads(qa["metadata_json"]) == raw
    assert [r["span_index"] for r in evidence] == [0, 1]
    assert {r["source_unit_id"] for r in evidence} == set(corpus)
    for saved, normalized, record in zip(
        raw["evidence_spans"], qa["evidence_spans"], evidence, strict=True
    ):
        assert {key: normalized[key] for key in saved} == saved
        assert record["text"] == saved["text"]
        assert (
            corpus[record["source_unit_id"]]["text"][record["start"] : record["end"]]
            == record["text"]
        )
    # The source language was not in the manifest; explicit evidence still requires it.
    assert "eurlex/fr" in report["corpus_units"]


def test_translated_question_can_use_target_evidence_language(tmp_path):
    index = SourceIndex(tmp_path / "index.sqlite")
    text = "Header. A useful rule. Footer."
    index.add(source_row("eurlex", "eli/target", "ACT1", "en", "article", "", text, {}))
    try:
        raw = {
            **legal_qa([span("1", "eli/target", "en", text, "A useful rule.")]),
            "question_language": "zh",
            "answer": "一条有用的规则。",
            "articles_involved": "1",
            "articles_involved_eli": "eli/target",
            "multi_article": False,
            "cross_act": False,
        }
        normalized = normalize("eurlex", raw, {}, index)
        assert normalized["context_language"] == "en"
        assert normalized["question_language"] == "zh"
        assert normalized["source_unit_id"] == "eurlex:eli/target:en"
        assert normalized["supporting_text"] == "A useful rule."
    finally:
        index.close()


@pytest.fixture
def un_source(tmp_path):
    index = SourceIndex(tmp_path / "index.sqlite")
    full_text = "Header\nsame support\nSeparator\nsame support\nFooter"
    index.add(
        source_row(
            "un",
            "doc",
            "doc",
            "en",
            "document",
            "",
            full_text,
            {"line_start": 101, "line_end": 105},
        )
    )
    raw = {
        "doc_id": "doc",
        "block_id": "doc#2",
        "target_id": "doc#2",
        "line_start": 104,
        "line_end": 104,
        "question_language": "en",
        "question": "Which support?",
        "answer": "support",
        "mode": "lookup",
        "generator_model_id": "model",
        "evidence_contract": CONTRACT,
        "evidence_spans": [
            span(
                "target",
                "doc#2",
                "en",
                "same support",
                "support",
                doc_id="doc",
                line_start=104,
                line_end=104,
            ),
        ],
    }
    yield index, raw, full_text
    index.close()


def test_un_maps_saved_block_offsets_instead_of_first_document_match(un_source):
    index, raw, full_text = un_source
    row = normalize("un", raw, {}, index)
    (first,) = row["evidence_spans"]
    assert first["start"] == 5
    assert first["end"] == 12
    assert first["document_start"] == full_text.rindex("support")
    assert first["document_start"] != full_text.index("support")
    assert row["support_start"] == first["document_start"]
    assert row["support_source_unit_id"] == "un:doc:en"
    assert full_text[row["support_start"] : row["support_end"]] == "support"


@pytest.mark.parametrize(
    "change,error",
    [
        ({"start": 0, "end": 7}, "offsets do not reproduce"),
        ({"end": 1000}, "offsets do not reproduce"),
        ({"source_text_sha256": "0" * 64}, "source text changed"),
        ({"line_start": 102, "line_end": 102}, "line range differs"),
        ({"line_start": 99}, "outside its document"),
        ({"source_id": "doc#1"}, "saved target block"),
        ({"doc_id": "other"}, "saved target block"),
        ({"language": "fr"}, "missing evidence source"),
    ],
)
def test_un_rejects_stale_or_mislocated_evidence(un_source, change, error):
    index, raw, _ = un_source
    altered = {**raw, "evidence_spans": [{**raw["evidence_spans"][0], **change}]}
    with pytest.raises(ValueError, match=error):
        align_spans("un", altered, index)


def test_csv_spans_round_trip_and_legacy_rows_stay_empty(un_source):
    index, raw, full_text = un_source
    csv_row = {**raw, "evidence_spans": json.dumps(raw["evidence_spans"])}
    assert (
        normalize("un", csv_row, {}, index)["evidence_spans"]
        == normalize("un", raw, {}, index)["evidence_spans"]
    )
    assert saved_spans({"answer": "support", "evidence": ""}) == []
    legacy = {k: v for k, v in raw.items() if k not in {"evidence_spans", "evidence_contract"}}
    assert normalize("un", legacy, {}, index)["support_start"] == full_text.index("support")
    assert normalize("un", legacy, {}, index)["evidence_spans"] == []


@pytest.mark.parametrize("change", [None, [], "[]", {}, "{}"])
def test_new_contract_never_falls_back_to_answer(change):
    with pytest.raises(ValueError, match="requires nonempty evidence_spans"):
        saved_spans({"evidence_contract": CONTRACT, "evidence_spans": change, "answer": "answer"})


def test_export_fails_on_incomplete_article_coverage(multi_article, tmp_path):
    manifest, raw, _ = multi_article
    raw["evidence_spans"] = raw["evidence_spans"][:1]
    with sqlite3.connect(tmp_path / "run.sqlite") as db:
        db.execute("UPDATE outcomes SET rows=?", (json.dumps([raw]),))
    with pytest.raises(ValueError, match="cover exactly all involved"):
        prepare(manifest, tmp_path / "export")
    assert not (tmp_path / "export").exists()
