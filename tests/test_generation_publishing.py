"""Generation exports preserve provenance without inventing retrieval judgments."""

import csv
import json
import sqlite3

import pyarrow.parquet as pq
import pytest

from clir_bench.publishing.generations import prepare, upload
from clir_bench.publishing.records import boolean, match_support, normalize
from clir_bench.publishing.sources import SourceIndex, load_un, source_row


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_config(output, config):
    return pq.read_table(output / "data" / config).to_pylist()


@pytest.fixture
def inputs(tmp_path):
    articles = []
    for number in (1, 2, 3):
        for language in ("en", "fr"):
            articles.append(
                {
                    "eli_id": f"eli/article-{number}",
                    "celex_id": "ACT",
                    "language": language,
                    "unit_type": "article",
                    "article_number": str(number),
                    "heading": "Heading",
                    "text": f"Article {number} support in {language}",
                }
            )
    write_jsonl(tmp_path / "articles.jsonl", articles)
    raw = {
        "target_id": "eli/article-1",
        "document_id": "ACT",
        "question_language": "en",
        "mode": "comparison",
        "question": "Compare requirements",
        "answer": "Article 1 support",
        "evidence": "",
        "generator_model_id": "model/a",
        "candidate_rank": 1,
        "articles_involved": "1,2",
        "multi_article": "true",
        "cross_act": "false",
        "reference_articles_supplied": "3",
        "is_best": "false",
        "custom_metadata": {"x": 3},
    }
    with sqlite3.connect(tmp_path / "run.sqlite") as connection:
        connection.execute("CREATE TABLE outcomes(task, status, rows, error)")
        connection.execute(
            "INSERT INTO outcomes VALUES (?,?,?,?)",
            (
                "task",
                "completed",
                json.dumps([raw, {**raw, "generator_model_id": "model/b"}]),
                "",
            ),
        )
    chemistry = []
    for identifier, partners in (
        ("A_en", ["B_fr"]),
        ("B_fr", ["A_en", "C_de"]),
        ("C_de", ["B_fr"]),
    ):
        chemistry.append(
            {
                "id": identifier,
                "language": identifier[-2:],
                "publication_number": "PATENT",
                "title": "",
                "abstract": "α supporting evidence",
                "context": "α supporting evidence",
                "paired_document_ids": json.dumps(partners),
            }
        )
    write_csv(tmp_path / "chemistry.csv", chemistry)
    write_csv(
        tmp_path / "questions.csv",
        [
            {
                "source_document_id": "A_en",
                "question_language": "en",
                "mode": "lookup",
                "question": "What is disclosed?",
                "answer": "supporting evidence",
                "evidence": "supporting evidence",
                "candidate_rank": "1",
                "is_best": "false",
            }
        ],
    )
    manifest = {
        "version": 1,
        "corpora": {
            "eurlex": {
                "generation_sqlite": ["run.sqlite"],
                "articles": "articles.jsonl",
                "languages": ["en", "fr"],
            },
            "chemistry": {
                "generation_csv": [{"path": "questions.csv", "model": "model/c"}],
                "corpus_csv": "chemistry.csv",
            },
        },
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest))
    return path, raw


def test_export_preserves_metadata_models_and_direct_relevance(inputs, tmp_path, monkeypatch):
    manifest, raw = inputs
    output = tmp_path / "export"
    report = prepare(manifest, output)
    assert report["queries"] == {"eurlex": 2, "chemistry": 1}
    qas = read_config(output, "qas")
    legal = [row for row in qas if row["corpus"] == "eurlex"]
    assert len({row["_id"] for row in legal}) == 2
    assert json.loads(legal[0]["metadata_json"]) == raw
    assert legal[0]["multi_article"] is True
    assert legal[0]["cross_act"] is False
    assert legal[0]["is_best"] is False
    assert legal[0]["supporting_text_origin"] == "answer"
    assert legal[0]["evidence"] == ""
    assert legal[0]["document_text"] == "Article 1 support in en"
    assert legal[0]["involved_article_ids"] == ["eli/article-1", "eli/article-2"]
    corpus = {row["_id"]: row for row in read_config(output, "corpus")}
    assert not any("article-3" in key for key in corpus)
    assert "C_de" not in corpus  # A-B-C is not an accepted A-C pair.
    qrels = read_config(output, "source_qrels")
    assert all(row["corpus_id"] in corpus for row in qrels)
    assert len([r for r in qrels if r["query_id"] == legal[0]["_id"]]) == 4
    chem = next(row for row in qas if row["corpus"] == "chemistry")
    assert {r["corpus_id"] for r in qrels if r["query_id"] == chem["_id"]} == {"A_en", "B_fr"}
    assert chem["document_text"][chem["support_start"] : chem["support_end"]] == chem["evidence"]
    assert chem["multi_article_declared"] is None
    assert all(
        len(list((output / "data" / config).glob("*.parquet"))) for config in report["configs"]
    )
    # Exercise the actual Hugging Face directory/config loader, entirely offline.
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    loaded = load_dataset(str(output), "qas", split="train", cache_dir=str(tmp_path / "cache"))
    assert len(loaded) == 3
    assert loaded.features["multi_article"].dtype == "bool"


def test_corpus_selection_and_existing_output_protection(inputs, tmp_path):
    manifest, _ = inputs
    output = tmp_path / "one"
    report = prepare(manifest, output, corpora=["chemistry"])
    assert report["queries"] == {"chemistry": 1}
    assert {r["corpus"] for r in read_config(output, "corpus")} == {"chemistry"}
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        prepare(manifest, output)
    assert (output / "manifest.json").read_bytes() == before


def test_un_uses_full_document_and_rejects_truncated_alignment(tmp_path):
    write_jsonl(
        tmp_path / "docs.jsonl",
        [
            {"doc_id": "doc-1", "line_start": 1, "line_end": 3, "n_lines": 3, "title": "UN"},
            {"doc_id": "doc-2", "line_start": 4, "line_end": 4, "n_lines": 1},
        ],
    )
    (tmp_path / "aligned.en").write_text("HEADER\nGeneration block\nFooter\nOther document\n")
    spec = {"documents": "docs.jsonl", "text_prefix": "aligned", "languages": ["en"]}
    index = SourceIndex(tmp_path / "index.sqlite")
    try:
        load_un(index, spec, tmp_path, {"doc-1"})
        doc = index.get("un:doc-1:en")
        assert doc["text"] == "HEADER\nGeneration block\nFooter"
        row = normalize(
            "un",
            {
                "document_id": "doc-1",
                "target_id": "doc-1#0",
                "question_language": "en",
                "question": "What?",
                "answer": "Footer",
                "generator_model_id": "model",
                "document_text": "Generation block",
                "references_supplied": "doc-2",
            },
            {},
            index,
        )
        assert row["support_match_status"] == "exact"
        assert row["generation_support_match_status"] == "unmatched"
        assert row["involved_source_unit_ids"] == ["un:doc-1:en"]
        assert row["generation_target_id"] == "doc-1#0"
    finally:
        index.close()
    (tmp_path / "aligned.en").write_text("HEADER\nGeneration block\nFooter\n")
    index = SourceIndex(tmp_path / "truncated.sqlite")
    try:
        with pytest.raises(ValueError, match="truncated"):
            load_un(index, spec, tmp_path, {"doc-2"})
    finally:
        index.close()


def test_cross_act_involvement_and_declaration_conflict(tmp_path):
    index = SourceIndex(tmp_path / "index.sqlite")
    try:
        for article, act in (("eli/one", "ACT1"), ("eli/two", "ACT2")):
            index.add(source_row("eurlex", article, act, "en", "article", "", "support", {}))
        row = normalize(
            "eurlex",
            {
                "target_id": "eli/one",
                "question_language": "en",
                "question": "Compare?",
                "answer": "support",
                "generator_model_id": "model",
                "multi_article": "false",
                "articles_involved_eli": "eli/one,eli/two",
                "cross_act": "false",
            },
            {},
            index,
        )
        assert row["multi_article"] is True
        assert row["multi_article_declared"] is False
        assert row["multi_article_conflict"] is True
        assert row["cross_act"] is True
        assert row["cross_act_declared"] is False
    finally:
        index.close()


@pytest.mark.parametrize("value,expected", [("false", False), ("True", True), ("", None)])
def test_boolean_strings(value, expected):
    assert boolean(value) is expected


def test_support_offsets_do_not_fabricate_spans():
    assert match_support("α β", "prefix α β suffix") == ("exact", 7, 10)
    assert match_support("α β", "α\n\tβ") == ("whitespace_normalized", None, None)
    assert match_support("α β", "α skipped β") == ("unmatched", None, None)
    assert match_support("", "abc") == ("missing", None, None)


class FakeHub:
    def __init__(self, files=()):
        self.files = files
        self.created = None
        self.uploaded = None

    def create_repo(self, repo_id, **kwargs):
        self.created = (repo_id, kwargs)

    def list_repo_files(self, repo_id, **kwargs):
        return self.files

    def upload_folder(self, **kwargs):
        self.uploaded = kwargs


def test_upload_exact_bundle_and_detect_edits(inputs, tmp_path):
    manifest, _ = inputs
    output = tmp_path / "export"
    prepare(manifest, output)
    api = FakeHub()
    assert upload(output, "user/dataset", private=False, api=api).endswith("/user/dataset")
    assert api.created[1]["private"] is False
    assert api.uploaded["repo_type"] == "dataset"
    assert "manifest.json" in api.uploaded["allow_patterns"]
    (output / "report.json").write_text("{}")
    api = FakeHub()
    with pytest.raises(ValueError, match="changed after preparation"):
        upload(output, "user/dataset", api=api)
    assert api.created is None


def test_upload_refuses_to_mix_releases(inputs, tmp_path):
    manifest, _ = inputs
    output = tmp_path / "export"
    prepare(manifest, output)
    api = FakeHub(["data/qas/old.parquet"])
    with pytest.raises(ValueError, match="already contains"):
        upload(output, "user/dataset", api=api)
    assert api.uploaded is None


def test_unresolved_articles_abort_without_leaving_partial_export(inputs, tmp_path):
    manifest, raw = inputs
    with sqlite3.connect(tmp_path / "run.sqlite") as connection:
        connection.execute(
            "UPDATE outcomes SET rows=?",
            (json.dumps([{**raw, "articles_involved_eli": "missing-article"}]),),
        )
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="unresolved involved articles"):
        prepare(manifest, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".generation-export-*"))
