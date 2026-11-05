"""Preparation keeps literal support, document identity, and fixed pilot inputs."""

from pathlib import Path

import pytest

from clir_bench.alignment import data


def test_exact_evidence_wins_over_an_also_exact_answer():
    assert data.select_anchor(
        {"evidence": "only after approval", "answer": "approval"},
        "The rule applies only after approval.",
    ) == ("only after approval", "evidence")


@pytest.mark.parametrize("evidence", [None, "", " ", "translated evidence", "only  after approval"])
def test_nonliteral_evidence_falls_back_to_literal_answer(evidence):
    assert data.select_anchor(
        {"evidence": evidence, "answer": "only after approval"},
        "The rule applies only after approval.",
    ) == ("only after approval", "answer")


@pytest.mark.parametrize(
    "passage,source",
    [
        ("only after approval", "only  after approval"),
        ("Rule", "rule"),
        ("café", "cafe\u0301"),
        ("A B", "A\nB"),
        ("Article 1", "Article\u00a01"),
        ("", "an arbitrary source"),
        (" ", "a space exists here"),
    ],
)
def test_source_support_is_never_normalized_or_invented(passage, source):
    assert data.select_anchor({"evidence": passage, "answer": passage}, source) is None


def candidate(unit, document, length):
    return {
        "source_unit_id": unit,
        "document_id": document,
        "source_language": "en",
        "source_text": "x" * length,
    }


def test_quantile_selection_is_order_independent_with_canonical_tie_break():
    rows = [candidate("z", "last", 30), candidate("b", "second", 10), candidate("a", "first", 10)]
    expected = data.choose_quantiles(rows, [0, .5, 1], set())
    observed = data.choose_quantiles(list(reversed(rows)), [0, .5, 1], set())
    assert observed == expected
    assert [row["source_unit_id"] for row in observed] == ["a", "b", "z"]


def test_quantile_selection_never_counts_two_language_versions_as_two_documents():
    rows = [
        candidate("same:en", "same", 10),
        candidate("same:fr", "same", 11),
        candidate("second:en", "second", 12),
        candidate("third:en", "third", 20),
    ]
    used = {"same"}
    selected = data.choose_quantiles(rows, [0, 1], used)
    assert [row["document_id"] for row in selected] == ["second", "third"]
    assert used == {"same", "second", "third"}


def test_quantile_selection_fails_when_distinct_documents_are_insufficient():
    with pytest.raises(ValueError, match="Insufficient distinct documents"):
        data.choose_quantiles([candidate("x:en", "x", 4)], [0, 1], set())


def test_immutable_artifact_allows_exact_replay_and_preserves_bytes_on_conflict(tmp_path):
    path = tmp_path / "nested" / "sample.json"
    content = '{"quote":"café\\nexact"}\n'.encode()
    data.write_immutable(path, content)
    data.write_immutable(path, content)
    with pytest.raises(ValueError, match="Refusing to replace"):
        data.write_immutable(path, b"different")
    assert path.read_bytes() == content


def test_known_language_error_is_quarantined_as_source_and_target(monkeypatch, tmp_path):
    bad, other = "EP-1732767-A1_es", "EP-1732767-A1_de"
    texts = {bad: "Die Erfindung betrifft ein Verfahren.", other: "Die Erfindung betrifft ein Verfahren."}
    neighbors = {bad: {other}, other: {bad}}
    documents = {
        unit: {
            "_id": unit, "text": texts[unit], "language": language,
            "source_id": unit, "parent_id": "EP-1732767-A1", "unit_type": "title_abstract",
            "title": "",
        }
        for unit, language in ((bad, "es"), (other, "de"))
    }
    qas = [
        {
            "_id": f"q-{unit}", "source_unit_id": unit, "context_language": row["language"],
            "question_language": row["language"], "evidence_language": row["language"],
            "evidence": row["text"], "answer": row["text"], "mode": "lookup",
        }
        for unit, row in documents.items()
    ]
    qrels = [
        {"query_id": f"q-{unit}", "corpus_id": next(iter(targets)), "relation": "direct_translation"}
        for unit, targets in neighbors.items()
    ]

    def fake_release(root, corpus):
        return (qas, qrels, documents, {}) if corpus == "chemistry" else ([], [], {}, {})

    class SelectionReached(Exception):
        pass

    def inspect_candidates(candidates):
        # The bad source is excluded despite literal evidence. Its counterpart is
        # also excluded because its sole accepted target is the quarantined source.
        assert candidates["chemistry"] == []
        raise SelectionReached

    monkeypatch.setattr(data, "_load_release", fake_release)
    monkeypatch.setattr(data, "_chemistry_snapshot", lambda: (texts, neighbors, {}))
    monkeypatch.setattr(data, "select_sample", inspect_candidates)
    with pytest.raises(SelectionReached):
        data.prepare(tmp_path / "pilot", Path.cwd())
    assert not (tmp_path / "pilot").exists()
