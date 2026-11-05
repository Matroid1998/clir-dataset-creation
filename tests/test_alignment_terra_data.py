"""Terra preparation preserves all QA rows without inventing multilingual support."""

import copy
import json
from pathlib import Path

import pytest

from clir_bench.alignment import terra_data
from clir_bench.alignment.data import digest_text, dumps


def fixture_rows():
    source, target, quarantined = "EP-GOOD_en", "EP-GOOD_fr", "EP-1732767-A1_es"
    documents = {
        source: {"_id": source, "text": "Only after approval. Another sentence.", "language": "en", "unit_type": "title_abstract"},
        target: {"_id": target, "text": "Seulement après approbation.", "language": "fr", "unit_type": "title_abstract"},
        quarantined: {"_id": quarantined, "text": "Ein deutscher Text.", "language": "es", "unit_type": "title_abstract"},
    }
    qas = []
    for identity, unit, evidence, answer in (
        ("q1", source, "Only after approval.", "approval"),
        ("q2", source, "Nonliteral supporting text", "Only after approval."),
        ("q3", source, "No matching evidence", "No matching answer"),
        ("q4", quarantined, "Ein deutscher Text.", "Ein deutscher Text."),
    ):
        qas.append({
            "_id": identity, "source_unit_id": unit, "question": "What is required?",
            "answer": answer, "evidence": evidence, "question_language": documents[unit]["language"],
            "context_language": documents[unit]["language"], "generator_model_id": "original-generator",
            "persona": "practitioner", "mode": "lookup", "provenance_json": '{"file":"original.sqlite"}',
            "document_text": "HEAVY ORIGINAL DOCUMENT", "generation_source_text": "HEAVY GENERATION TEXT",
            "metadata_json": '{"do_not_copy":"raw residual source fields"}',
        })
    anchor_id = "a_" + digest_text(dumps(["chemistry", source, "Only after approval."]))
    tasks = [{
        "task_id": "t1", "corpus": "chemistry", "document_id": "EP-GOOD",
        "source_unit_id": source, "target_unit_id": target,
        "source_language": "en", "target_language": "fr",
        "source_text": documents[source]["text"], "target_text": documents[target]["text"],
        "source_text_sha256": digest_text(documents[source]["text"]),
        "target_text_sha256": digest_text(documents[target]["text"]),
        "anchors": [{"anchor_id": anchor_id, "text": "Only after approval.", "qa_ids": ["q1", "q2"]}],
    }]
    qrels = [{"query_id": qa["_id"], "corpus_id": target, "relation": "direct_translation"} for qa in qas]
    return qas, qrels, documents, tasks


def test_all_qas_survive_with_exact_original_evidence_and_explicit_exclusions():
    qas, qrels, documents, tasks = fixture_rows()
    before = copy.deepcopy(qas)
    indexed = terra_data.build_qa_index("chemistry", qas, qrels, documents, tasks)
    assert len(indexed) == len(qas) == 4
    assert qas == before
    by_id = {row["qa_id"]: row for row in indexed}
    assert by_id["q1"]["source_evidence_origin"] == "evidence"
    assert by_id["q2"]["source_evidence_origin"] == "answer"
    assert by_id["q1"]["anchor_id"] == by_id["q2"]["anchor_id"]
    assert by_id["q1"]["expected_targets"] == [{"language": "fr", "source_unit_id": "EP-GOOD_fr", "task_id": "t1"}]
    original = by_id["q1"]["evidence_all_languages"]
    assert set(original) == {"en"}
    assert original["en"][0]["text"] == "Only after approval."
    assert original["en"][0]["match_status"] == "exact"
    assert original["en"][0]["is_original"] is True
    assert by_id["q3"]["eligibility_status"] == "no_exact_source_passage"
    assert by_id["q4"]["eligibility_status"] == "quarantined_source_language"
    for identity in ("q3", "q4"):
        assert by_id[identity]["anchor_id"] is None
        assert by_id[identity]["source_evidence_text"] is None
        assert by_id[identity]["evidence_all_languages"] == {}
        assert by_id[identity]["expected_targets"] == []


def test_compact_metadata_keeps_original_identity_and_provenance_but_not_documents():
    qas, qrels, documents, tasks = fixture_rows()
    row = terra_data.build_qa_index("chemistry", qas, qrels, documents, tasks)[0]
    metadata = json.loads(row["metadata_json"])
    assert metadata["generator_model_id"] == "original-generator"
    assert metadata["persona"] == "practitioner"
    assert metadata["provenance_json"] == '{"file":"original.sqlite"}'
    assert metadata["published_source"]["qa_id"] == "q1"
    assert metadata["published_source"]["revision"] == terra_data.RELEASES["chemistry"][2]
    assert "document_text" not in metadata and "generation_source_text" not in metadata
    assert "metadata_json" not in metadata and "HEAVY" not in row["metadata_json"]


def test_filter_rejects_unsupported_language_on_either_side_without_mutating_source():
    _, _, _, tasks = fixture_rows()
    variants = [copy.deepcopy(tasks[0]) for _ in range(3)]
    variants[1].update(task_id="t2", target_language="ar", target_unit_id="EP-GOOD_ar")
    variants[2].update(task_id="t3", source_language="ru", source_unit_id="EP-GOOD_ru")
    before = copy.deepcopy(variants)
    assert terra_data.filter_tasks(variants) == [variants[0]]
    assert variants == before


def test_missing_eligible_qa_mapping_is_not_silently_dropped():
    qas, qrels, documents, tasks = fixture_rows()
    tasks[0]["anchors"][0]["qa_ids"] = ["q1"]
    with pytest.raises(ValueError, match="Eligible published QA is missing"):
        terra_data.build_qa_index("chemistry", qas, qrels, documents, tasks)


def test_qrels_cannot_invent_a_target_missing_from_the_frozen_workload():
    qas, qrels, documents, tasks = fixture_rows()
    documents["EP-GOOD_de"] = {"language": "de"}
    qrels.append({"query_id": "q1", "corpus_id": "EP-GOOD_de", "relation": "direct_translation"})
    with pytest.raises(ValueError, match="Frozen targets differ"):
        terra_data.build_qa_index("chemistry", qas, qrels, documents, tasks)


def test_frozen_anchor_must_still_match_the_published_source_verbatim():
    qas, qrels, documents, tasks = fixture_rows()
    tasks[0]["anchors"][0]["text"] = "Only  after approval."
    with pytest.raises(ValueError, match="nonempty literal source passage"):
        terra_data.build_qa_index("chemistry", qas, qrels, documents, tasks)


def test_tampered_frozen_file_is_rejected_before_any_output(tmp_path):
    source = tmp_path / "luna"
    source.mkdir()
    (source / "full_tasks.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="SHA-256"):
        terra_data.prepare(tmp_path / "terra", tmp_path, source_dir=source)
    assert not (tmp_path / "terra").exists()


def test_preparation_cannot_overwrite_prior_luna_directory(tmp_path):
    with pytest.raises(ValueError, match="must not overwrite the Luna pilot"):
        terra_data.prepare(tmp_path, Path.cwd(), source_dir=tmp_path)
