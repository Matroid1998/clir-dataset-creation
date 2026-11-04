"""Full retrieval corpora can expand without rewriting published QA artifacts."""

import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clir_bench.publishing.corpus_expansion import expand_corpus
from clir_bench.publishing.generations import QRELS_SCHEMA, sha256
from clir_bench.publishing.records import dumps
from clir_bench.publishing.sources import (
    CORPUS_SCHEMA,
    SourceIndex,
    load_chemistry,
    load_eurlex,
    load_un,
)


def write_jsonl(path, rows):
    path.write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")


def source_fixture(root, corpus):
    if corpus == "eurlex":
        articles = [
            {
                "eli_id": f"eli/{number}",
                "celex_id": "ACT",
                "article_number": str(number),
                "language": language,
                "unit_type": "annex" if number == 3 else "article",
                "heading": "Heading",
                "text": f"Provision {number} in {language}",
            }
            for number in (1, 2, 3)
            for language in ("en", "fr")
        ]
        write_jsonl(root / "articles.jsonl", articles)
        return (
            {"articles": "articles.jsonl", "languages": ["en", "fr"]},
            load_eurlex,
            lambda row: row["source_id"] == "eli/1",
        )
    if corpus == "un":
        documents = [
            {"doc_id": f"doc-{number}", "line_start": number, "line_end": number, "n_lines": 1}
            for number in (1, 2)
        ]
        write_jsonl(root / "documents.jsonl", documents)
        languages = ["en", "fr", "es", "zh", "ar", "ru"]
        for language in languages:
            (root / f"aligned.{language}").write_text(
                f"Document one in {language}\nUnrelated document two in {language}\n",
                encoding="utf-8",
            )
        return (
            {"documents": "documents.jsonl", "text_prefix": "aligned", "languages": languages},
            load_un,
            lambda row: row["source_id"] == "doc-1" and row["language"] not in {"ar", "ru"},
        )
    documents = [
        {
            "id": identifier,
            "language": identifier[-2:],
            "publication_number": identifier[0],
            "title": "",
            "abstract": f"Patent abstract {identifier}",
            "paired_document_ids": dumps([partner]),
        }
        for identifier, partner in (
            ("A_en", "B_fr"),
            ("B_fr", "A_en"),
            ("C_de", "D_es"),
            ("D_es", "C_de"),
        )
    ]
    with (root / "chemistry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=documents[0])
        writer.writeheader()
        writer.writerows(documents)
    return (
        {"corpus_csv": "chemistry.csv"},
        load_chemistry,
        lambda row: row["source_id"] in {"A_en", "B_fr"},
    )


def published_bundle(root, corpus):
    spec, loader, selected = source_fixture(root, corpus)
    manifest = {"version": 1, "corpora": {corpus: spec}}
    manifest_path = root / "input.json"
    manifest_path.write_text(dumps(manifest), encoding="utf-8")
    index = SourceIndex(root / "fixture.sqlite")
    try:
        loader(index, spec, root, None)
        corpus_rows = [row for row in index.rows() if selected(row)]
    finally:
        index.close()
    target = next(row for row in corpus_rows if row["language"] == "en")
    old = root / "old"
    qas = [{"_id": "q1", "question": "Original?", "involved_source_unit_ids": [target["_id"]]}]
    evidence = [{"query_id": "q1", "text": "Original support"}]
    qrels = [
        {
            "query_id": "q1",
            "corpus_id": row["_id"],
            "score": 1,
            "relation": "generation_source" if row == target else "direct_translation",
            "corpus": corpus,
            "language": row["language"],
        }
        for row in corpus_rows
    ]
    configurations = {
        "qas": qas,
        "evidence": evidence,
        "source_qrels": qrels,
        "corpus": corpus_rows,
    }
    for config, rows in configurations.items():
        path = old / "data" / config
        path.mkdir(parents=True)
        schema = {"corpus": CORPUS_SCHEMA, "source_qrels": QRELS_SCHEMA}.get(config)
        filename = "train-00003.parquet" if config == "source_qrels" else "train-00000.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path / filename)
    counts = {name: len(rows) for name, rows in configurations.items()}
    report = {
        "corpus_scope": "referenced",
        "configs": counts,
        "queries": {corpus: 1},
        "models": {corpus: {"original-generator": 1}},
        "support_matches": {corpus: {"unmatched": 1}},
    }
    (old / "report.json").write_text(dumps(report), encoding="utf-8")
    (old / "README.md").write_text(
        "# Original card\n\n"
        "Contains documents associated with the published questions and their available "
        "translations. This QA release does not include the complete retrieval haystack.\n\n"
        + "\n".join(f"- `{name}`: {count:,} rows" for name, count in counts.items())
        + "\n\nunits are full articles (or explicitly involved annexes); patent units are title + abstract.\n"
        "Legacy evidence documentation stays untouched.\n",
        encoding="utf-8",
    )
    files = {
        str(path.relative_to(old)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in old.rglob("*")
        if path.is_file()
    }
    old_spec = json.loads(dumps(manifest))
    if corpus == "un":
        old_spec["corpora"]["un"]["languages"] = ["en", "fr", "es", "zh"]
    (old / "manifest.json").write_text(
        dumps(
            {
                "version": 1,
                "selected_corpora": [corpus],
                "corpus_scope": "referenced",
                "configs": counts,
                "input_manifest": old_spec,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return old, manifest_path, report


@pytest.mark.parametrize(
    "corpus,total,added", [("eurlex", 6, 0), ("un", 12, 2), ("chemistry", 4, 0)]
)
def test_expands_full_corpus_preserving_published_bytes(tmp_path, corpus, total, added):
    old, manifest_path, previous_report = published_bundle(tmp_path, corpus)
    output = tmp_path / "full"
    report = expand_corpus(old, manifest_path, output, corpus=corpus)
    assert report["corpus_scope"] == "full"
    assert report["configs"]["corpus"] == total
    assert report["configs"]["source_qrels"] == previous_report["configs"]["source_qrels"] + added
    for metric in ("models", "queries", "support_matches"):
        assert report[metric] == previous_report[metric]
    for config in ("qas", "evidence", "source_qrels"):
        for path in (old / "data" / config).glob("*.parquet"):
            assert path.read_bytes() == (output / path.relative_to(old)).read_bytes()
    assert pq.read_schema(output / "data/qas/train-00000.parquet") == pq.read_schema(
        old / "data/qas/train-00000.parquet"
    )
    corpus_rows = pq.read_table(output / "data/corpus").to_pylist()
    assert len(corpus_rows) == total
    qrels = pq.read_table(output / "data/source_qrels").to_pylist()
    assert len({(row["query_id"], row["corpus_id"]) for row in qrels}) == len(qrels)
    assert {row["corpus_id"] for row in qrels} <= {row["_id"] for row in corpus_rows}
    if corpus == "un":
        extra = pq.read_table(output / "data/source_qrels/train-00004.parquet").to_pylist()
        assert {row["language"] for row in extra} == {"ar", "ru"}
        assert all(row["relation"] == "direct_translation" for row in extra)
    else:
        assert len(list((output / "data/source_qrels").glob("*.parquet"))) == 1
    card = (output / "README.md").read_text()
    assert "Full local corpus for the configured languages" in card
    assert "Legacy evidence documentation stays untouched." in card
    if corpus == "eurlex":
        assert "all locally collected articles and annexes" in card
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["input_manifest"] == json.loads(manifest_path.read_text())
    assert (
        saved["generation_input_manifest"]
        == json.loads((old / "manifest.json").read_text())["input_manifest"]
    )
    for relative, details in saved["files"].items():
        assert details["sha256"] == sha256(output / relative)
    assert not list(output.glob("*.sqlite*"))
    assert not list(tmp_path.glob(".corpus-expansion-*"))


def test_reexpansion_does_not_duplicate_translation_links(tmp_path):
    old, manifest_path, _ = published_bundle(tmp_path, "un")
    first = tmp_path / "first"
    report = expand_corpus(old, manifest_path, first, corpus="un")
    second = tmp_path / "second"
    repeated = expand_corpus(first, manifest_path, second, corpus="un")
    assert repeated["configs"] == report["configs"]
    assert repeated["corpus_expansion"]["source_qrels_added"] == 0
    for path in (first / "data/source_qrels").glob("*.parquet"):
        assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()


def test_changed_published_source_aborts_and_cleans_stage(tmp_path):
    old, manifest_path, _ = published_bundle(tmp_path, "un")
    (tmp_path / "aligned.en").write_text("Changed document\nUnrelated document two in en\n")
    output = tmp_path / "full"
    with pytest.raises(ValueError, match="published source changed"):
        expand_corpus(old, manifest_path, output, corpus="un")
    assert not output.exists()
    assert not list(tmp_path.glob(".corpus-expansion-*"))


def test_missing_old_language_aborts_instead_of_dropping_sources(tmp_path):
    old, manifest_path, _ = published_bundle(tmp_path, "un")
    spec = json.loads(manifest_path.read_text())
    spec["corpora"]["un"]["languages"].remove("fr")
    manifest_path.write_text(dumps(spec))
    with pytest.raises(ValueError, match="published source changed or is missing"):
        expand_corpus(old, manifest_path, tmp_path / "full", corpus="un")


def test_modified_published_bundle_rejected_before_source_loading(tmp_path):
    old, manifest_path, _ = published_bundle(tmp_path, "chemistry")
    with (old / "data/qas/train-00000.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="bundle changed after preparation"):
        expand_corpus(old, manifest_path, tmp_path / "full", corpus="chemistry")
    assert not list(tmp_path.glob(".corpus-expansion-*"))


def test_unrecorded_shards_rejected(tmp_path):
    old, manifest_path, _ = published_bundle(tmp_path, "chemistry")
    (old / "data/qas/unrecorded.parquet").write_bytes(
        (old / "data/qas/train-00000.parquet").read_bytes()
    )
    with pytest.raises(ValueError, match="unrecorded or missing data"):
        expand_corpus(old, manifest_path, tmp_path / "full", corpus="chemistry")
