"""UN retrieval scope follows document types without applying generation limits."""

import json
from types import SimpleNamespace

import pytest

from clir_bench.publishing.sources import load_un


@pytest.fixture
def sources(tmp_path):
    descriptions = [
        ("2001/a/res/56_1", "Resolution", ["Resolution text", "Second original line"]),
        ("2001/a/dec/56_2", "Decision", ["Decision text"]),
        ("2001/s/pv_1234", "Meeting", ["Long meeting " * 4000, "Final meeting line"]),
        ("2001/a/56_3", "Letter dated 1 January", ["Letter text"]),
        ("2001/a/56_4", "Note verbale", ["Note text"]),
        ("2001/a/56_5", "Identical letters dated 2 January", ["Identical letters text"]),
        ("2001/a/56_6", "Annual report", ["Excluded report line", "Another report line"]),
        ("2001/a/res/corr_1", "Resolution corrigendum", ["Excluded correction"]),
        ("2001/a/agenda/sr_4", "Agenda", ["Excluded logistics"]),
        ("2001/journal/1", "Letter dated 3 January", ["Excluded journal"]),
        ("2001/a/res/56_7", "Final resolution", ["Correctly aligned final document"]),
    ]
    documents = []
    position = 1
    for identifier, title, lines in descriptions:
        documents.append(
            {
                "doc_id": identifier,
                "title": title,
                "line_start": position,
                "line_end": position + len(lines) - 1,
                "n_lines": len(lines),
                "char_count": len("\n".join(lines)),
                "n_targets": 0,
                "target_idxs": [],
            }
        )
        position += len(lines)
    (tmp_path / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8"
    )
    languages = ["en", "fr", "es", "zh", "ar", "ru"]
    for language in languages:
        text = "\r\n".join(
            f"{language} α {line}" for _, _, lines in descriptions for line in lines
        ) + "\r\n"
        (tmp_path / f"aligned.{language}").write_bytes(text.encode("utf-8"))
    spec = {
        "documents": "documents.jsonl",
        "text_prefix": "aligned",
        "languages": languages,
        "document_genres": ["resolution", "meeting", "letter"],
    }
    return spec, documents


def read_sources(root, spec, wanted=None):
    rows = []
    load_un(SimpleNamespace(add=rows.append), spec, root, wanted)
    return rows


def test_selected_genres_preserve_original_documents_in_all_languages(tmp_path, sources):
    spec, documents = sources
    selected = read_sources(tmp_path, spec)
    expected_ids = {row["doc_id"] for row in documents[:6]} | {documents[-1]["doc_id"]}
    assert len(selected) == len(expected_ids) * 6
    assert {row["source_id"] for row in selected} == expected_ids
    assert {row["language"] for row in selected} == set(spec["languages"])
    # The unfiltered legacy path provides an independent exact-record baseline.
    unfiltered_spec = {key: value for key, value in spec.items() if key != "document_genres"}
    baseline = {row["_id"]: row for row in read_sources(tmp_path, unfiltered_spec)}
    for row in selected:
        assert row == baseline[row["_id"]]
        assert row["unit_type"] == "document"
        assert json.loads(row["metadata_json"])["n_targets"] == 0
    meeting = next(
        row for row in selected if row["source_id"] == "2001/s/pv_1234" and row["language"] == "en"
    )
    assert len(meeting["text"]) > 30_000
    assert meeting["text"].endswith("\nen α Final meeting line")
    last = next(
        row for row in selected if row["source_id"] == documents[-1]["doc_id"] and row["language"] == "ru"
    )
    assert last["text"] == "ru α Correctly aligned final document"
    assert json.loads(last["metadata_json"])["line_start"] == documents[-1]["line_start"]


def test_wanted_sources_intersect_genre_scope(tmp_path, sources):
    spec, _ = sources
    rows = read_sources(
        tmp_path,
        {**spec, "document_genres": ["letter"]},
        {"2001/a/56_3", "2001/a/res/56_1", "2001/journal/1"},
    )
    assert len(rows) == 6
    assert {row["source_id"] for row in rows} == {"2001/a/56_3"}


@pytest.mark.parametrize("value", [None, [], "resolution", {}, ["report"], [1], [{}]])
def test_invalid_scope_rejected_before_reading_sources(tmp_path, value):
    with pytest.raises(ValueError, match="document_genres must be a nonempty list"):
        read_sources(tmp_path, {"document_genres": value})


def test_legacy_absent_scope_includes_every_document(tmp_path, sources):
    spec, documents = sources
    spec.pop("document_genres")
    rows = read_sources(tmp_path, spec)
    assert len(rows) == len(documents) * 6
    assert {row["source_id"] for row in rows} == {row["doc_id"] for row in documents}


def test_excluded_documents_still_checked_for_truncation(tmp_path, sources):
    spec, _ = sources
    path = tmp_path / "aligned.en"
    path.write_text("Only one original line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="truncated UN language file"):
        read_sources(tmp_path, {**spec, "document_genres": ["letter"]}, wanted=set())


def test_trailing_lines_rejected_with_genre_filter(tmp_path, sources):
    spec, _ = sources
    with (tmp_path / "aligned.en").open("a", encoding="utf-8") as handle:
        handle.write("Unexpected trailing line\n")
    with pytest.raises(ValueError, match="lines beyond document index"):
        read_sources(tmp_path, spec)


def test_gapped_excluded_document_ranges_still_rejected(tmp_path, sources):
    spec, documents = sources
    documents[6]["line_start"] += 1
    (tmp_path / "documents.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid/gapped UN document range"):
        read_sources(tmp_path, spec)
