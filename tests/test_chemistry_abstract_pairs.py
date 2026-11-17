"""Offline accepted-pair import, provenance and exact-neighbor validation."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from clir_bench.core import corpus
from clir_bench.core.config import Settings
from clir_bench.core.context import AppContext
from clir_bench.domains.chemistry.domain import SPEC
from clir_bench.domains.chemistry.schema import SCHEMA
from clir_bench.domains.chemistry.sources import abstract_pairs as importer

REVISION = "a" * 40


def pair(publication="EP-1234567-A1", source="en", target="de", *, verdict="exact", accepted=True):
    return {
        "example_id": f"within-document:abstract:{source}-{target}:{publication}",
        "doc_id": publication, "corpus_id": "google-patents-chem", "source": "EP",
        "group_key": "patent-family-1", "pub_date": "20200101", "field": "abstract",
        "language_pair": f"{source}-{target}", "source_language": source, "target_language": target,
        "source_text": f"{source}:  Fe<sub>2</sub>O3\nα & β  ",
        "target_text": f"{target}:  Fe<sub>2</sub>O3\nα & β  ",
        "source_char_count": 20, "target_char_count": 20, "source_token_count": 8,
        "target_token_count": 8, "source_sentence_count": 1, "target_sentence_count": 1,
        "source_tokens_per_sentence": 8.0, "target_tokens_per_sentence": 8.0,
        "tokenizer": "o200k_base", "selection_rule": "archived-selection",
        "judge_model": "glm-flash-judge", "judge_verdict": verdict,
        "judge_reason": "The abstracts express the same disclosed invention.", "pair_ok": accepted,
    }


def build(rows, **kwargs):
    return importer.build_corpus(rows, schema=SCHEMA, revision=REVISION, **kwargs)


@pytest.fixture
def context(tmp_path):
    return AppContext.build(Settings(
        project_root=tmp_path, data_dir=tmp_path / "data", reports_dir=tmp_path / "reports",
        cluster_dir=tmp_path / "cluster",
    ), SPEC)


def test_filter_requires_acceptance_verdict_and_both_supported_languages():
    rows = [pair(), pair("EP-2", verdict="high"), pair("EP-3", accepted=False),
            pair("EP-4", verdict="medium"), pair("EP-5", source="ar", target="en"),
            pair("EP-6", source="en", target="ko")]
    documents, accepted, counts = build(rows)
    assert [row["doc_id"] for row in accepted] == ["EP-1234567-A1", "EP-2"]
    assert len(documents) == 4
    assert counts["source_pairs"] == 6
    assert counts["accepted_pairs_all_languages"] == 4
    assert counts["accepted_pairs_selected_languages"] == counts["selected_pairs"] == 2
    assert counts["pairs_by_verdict"] == {"exact": 1, "high": 1}
    assert all(row["source"] == importer.SOURCE_VALUE for row in documents)


def test_exact_endpoint_text_and_direct_edges_survive_deduplication():
    first, second = pair(), pair(source="de", target="fr", verdict="high")
    first["upstream_extra"] = {"retained": True}
    documents, accepted, counts = build([first, second])
    by_lang = {row["language"]: row for row in documents}
    assert len(documents) == 3 and counts["publications"] == 1
    assert by_lang["en"]["abstract"] == by_lang["en"]["context"] == first["source_text"]
    assert by_lang["en"]["title"] == by_lang["en"]["description"] == by_lang["en"]["first_claim"] == ""
    assert by_lang["en"]["country_code"] == "EP"
    assert by_lang["en"]["publication_date"] == "20200101"
    assert by_lang["en"]["publication_number"] == first["doc_id"]
    assert by_lang["en"]["upstream_group_key"] == "patent-family-1"
    assert by_lang["en"]["upstream_corpus_id"] == "google-patents-chem"
    assert json.loads(by_lang["en"]["paired_document_ids"]) == [by_lang["de"]["id"]]
    assert json.loads(by_lang["fr"]["paired_document_ids"]) == [by_lang["de"]["id"]]
    assert set(json.loads(by_lang["de"]["paired_document_ids"])) == {by_lang["en"]["id"], by_lang["fr"]["id"]}
    assert set(json.loads(by_lang["de"]["accepted_pair_ids"])) == {first["example_id"], second["example_id"]}
    assert accepted[0]["upstream_extra"] == {"retained": True}
    assert accepted[0]["source_document_id"] == by_lang["en"]["id"]
    assert accepted[0]["target_document_id"] == by_lang["de"]["id"]


def test_japanese_and_russian_are_imported_with_exact_text_and_direct_partners():
    first = pair(source="en", target="ja")
    second = pair(source="ja", target="ru", verdict="high")
    first["target_text"] = second["source_text"] = "日本語の要約：Fe<sub>2</sub>O3。\nα & β"
    second["target_text"] = "Русский реферат: Fe<sub>2</sub>O3.\nα & β"
    documents, accepted, counts = build([first, second])
    by_language = {row["language"]: row for row in documents}
    assert set(by_language) == {"en", "ja", "ru"}
    assert by_language["ja"]["abstract"] == first["target_text"]
    assert by_language["ru"]["abstract"] == second["target_text"]
    assert counts["selected_pairs"] == len(accepted) == 2
    assert json.loads(by_language["en"]["paired_document_ids"]) == [by_language["ja"]["id"]]
    assert json.loads(by_language["ru"]["paired_document_ids"]) == [by_language["ja"]["id"]]


def test_language_subset_and_limit_apply_to_pairs_not_individual_documents():
    rows = [pair("EP-1", source="en", target="zh"), pair("EP-2"), pair("EP-3")]
    documents, accepted, counts = build(rows, languages=["en", "de"], limit=1)
    assert len(documents) == 2
    assert [row["doc_id"] for row in accepted] == ["EP-2"]
    assert counts["accepted_pairs_selected_languages"] == 2
    assert counts["selected_pairs"] == 1


@pytest.mark.parametrize("kwargs", [{"languages": ["en", "ar"]}, {"languages": []},
                                    {"limit": 0}, {"limit": -1}, {"limit": True}])
def test_invalid_selection_arguments_fail(kwargs):
    with pytest.raises(ValueError):
        build([pair()], **kwargs)


@pytest.mark.parametrize("changes", [
    {"pair_ok": "true"}, {"pair_ok": 1}, {"judge_verdict": "maybe"},
    {"source_text": ""}, {"target_text": None}, {"doc_id": ""},
    {"field": "title"}, {"language_pair": "de-en"}, {"target_language": "en"},
])
def test_malformed_rows_fail_without_silent_coercion(changes):
    with pytest.raises(ValueError):
        build([dict(pair(), **changes)])


def test_duplicate_pair_ids_edges_and_conflicting_endpoint_text_are_rejected():
    with pytest.raises(ValueError, match="pair ID"):
        build([pair(), pair()])
    with pytest.raises(ValueError, match="publication/language edge"):
        build([pair(), dict(pair(), example_id="different-id")])
    second = pair(source="de", target="fr")
    second["source_text"] = "Different stored German abstract"
    with pytest.raises(ValueError, match="Conflicting text"):
        build([pair(), second])


def test_jsonl_and_parquet_readers_preserve_text(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    expected = [pair()]
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(json.dumps(expected[0], ensure_ascii=False) + "\n\n", encoding="utf-8")
    parquet = tmp_path / "pairs.parquet"
    pq.write_table(pa.Table.from_pylist(expected), parquet)
    assert importer.read_pair_rows(jsonl) == importer.read_pair_rows(parquet) == expected
    jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        importer.read_pair_rows(jsonl)


def test_hub_transport_resolves_revision_before_shard_download(monkeypatch, tmp_path):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "pairs.parquet"
    pq.write_table(pa.Table.from_pylist([pair()]), path)
    calls = []

    def dataset_info(repo_id, revision):
        assert repo_id == importer.DATASET_REPO and revision == "main"
        return SimpleNamespace(sha=REVISION, siblings=[
            SimpleNamespace(rfilename="README.md"),
            SimpleNamespace(rfilename="data/train-00000-of-00001.parquet"),
        ])

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == REVISION
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=dataset_info))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    rows, provenance = importer.load_hub_rows()
    assert rows == [pair()]
    assert provenance["resolved_revision"] == REVISION
    assert provenance["requested_revision"] == "main"
    assert provenance["files"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls == [{"repo_id": importer.DATASET_REPO, "repo_type": "dataset", "revision": REVISION,
                      "filename": "data/train-00000-of-00001.parquet"}]


def test_ingest_local_archive_writes_complete_provenance_and_pair_sidecar(context, tmp_path):
    raw = pair()
    raw["upstream_extra"] = {"retained": True}
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
    args = SimpleNamespace(input=path, revision="main", languages=None, limit=None)
    assert importer.ingest(context, args) == 0
    output = context.workspace.corpus_csv("gp")
    rows = corpus.read_rows(output)
    assert len(rows) == 2
    assert all(row["upstream_revision"].startswith("local-sha256:") for row in rows)
    assert rows[1]["context"] == raw["source_text"]
    importer.validate_corpus(corpus.group_by_family(rows, SCHEMA), SCHEMA)
    saved_pair, = corpus.read_rows(output.with_name(importer.PAIRS_FILENAME))
    assert saved_pair["source_text"] == raw["source_text"]
    assert saved_pair["pair_ok"] == "true"
    assert json.loads(saved_pair["upstream_extra"]) == {"retained": True}
    manifest = json.loads(output.with_name(importer.MANIFEST_FILENAME).read_text())
    assert manifest["upstream"]["remote_revision_verified"] is False
    assert manifest["counts"]["selected_pairs"] == 1
    assert manifest["counts"]["documents"] == 2
    for filename, metadata in manifest["artifacts"].items():
        assert metadata["sha256"] == hashlib.sha256(output.with_name(filename).read_bytes()).hexdigest()


def test_ingest_validation_failure_preserves_existing_artifacts(context, tmp_path):
    output = context.workspace.corpus_csv("gp")
    output.parent.mkdir(parents=True)
    output.write_text("old corpus", encoding="utf-8")
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(dict(pair(), source_text="")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_text"):
        importer.ingest(context, SimpleNamespace(input=path))
    assert output.read_text() == "old corpus"
    assert not output.with_name(importer.PAIRS_FILENAME).exists()


def test_import_rejects_a_different_configured_dataset(context):
    from dataclasses import replace

    changed = replace(context, domain_settings={"input_dataset_repo": "other/unchecked-dataset"})
    with pytest.raises(ValueError, match="input_dataset_repo must be"):
        importer.ingest(changed, SimpleNamespace())


def test_artifact_publish_failure_rolls_back_previous_bundle(tmp_path, monkeypatch):
    staged, output = tmp_path / "staged", tmp_path / "output"
    staged.mkdir()
    output.mkdir()
    names = (importer.CORPUS_FILENAME, importer.PAIRS_FILENAME, importer.MANIFEST_FILENAME)
    for name in names:
        (staged / name).write_text("new " + name)
        (output / name).write_text("old " + name)
    original_replace = importer.os.replace

    def replace(source, destination):
        if source == staged / importer.MANIFEST_FILENAME:
            raise OSError("simulated publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(importer.os, "replace", replace)
    with pytest.raises(OSError, match="publication failure"):
        importer._publish_files(staged, output, names)
    assert all((output / name).read_text() == "old " + name for name in names)


@pytest.mark.parametrize("changes", [
    {"source": "google_patents"}, {"source": "epo"}, {"upstream_dataset": "other/repo"},
    {"upstream_revision": ""}, {"language": "ar"}, {"id": ""}, {"id": "arbitrary-id"},
    {"abstract": ""}, {"context": "different text"}, {"abstract": None},
    {"accepted_pair_ids": "[]"}, {"accepted_pair_ids": '"pair-id"'},
    {"accepted_pair_ids": '["duplicate", "duplicate"]'},
    {"paired_document_ids": "not-json"}, {"paired_document_ids": "[]"},
    {"paired_document_ids": '["missing-document"]'},
])
def test_validator_rejects_legacy_missing_or_malformed_provenance(changes):
    documents, _, _ = build([pair()])
    documents[0].update(changes)
    with pytest.raises(ValueError):
        importer.validate_corpus(corpus.group_by_family(documents, SCHEMA), SCHEMA)


def test_validator_rejects_duplicate_incomplete_or_invented_edges():
    documents, _, _ = build([pair(), pair(source="de", target="fr")])
    with pytest.raises(ValueError, match="Duplicate"):
        importer.validate_corpus(corpus.group_by_family(documents + [documents[0]], SCHEMA), SCHEMA)
    with pytest.raises(ValueError, match="exactly two endpoints"):
        importer.validate_corpus(corpus.group_by_family(documents[1:], SCHEMA), SCHEMA)
    changed = copy.deepcopy(documents)
    by_lang = {row["language"]: row for row in changed}
    by_lang["en"]["paired_document_ids"] = json.dumps([by_lang["de"]["id"], by_lang["fr"]["id"]])
    with pytest.raises(ValueError, match="Direct neighbors disagree"):
        importer.validate_corpus(corpus.group_by_family(changed, SCHEMA), SCHEMA)


def test_validator_rejects_wrong_publication_and_same_language_neighbors():
    documents, _, _ = build([pair(), pair("EP-OTHER")])
    by_id = {row["id"]: row for row in documents}
    en_ids = [row["id"] for row in documents if row["language"] == "en"]
    first, second = en_ids
    for this, other in ((first, second), (second, first)):
        by_id[this]["accepted_pair_ids"] = '["false-edge"]'
        by_id[this]["paired_document_ids"] = json.dumps([other])
    with pytest.raises(ValueError, match="share a publication and differ in language"):
        importer.validate_corpus(corpus.group_by_family([by_id[first], by_id[second]], SCHEMA), SCHEMA)
