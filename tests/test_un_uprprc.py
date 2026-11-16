"""Safety properties of deterministic UPRPRC preparation; no network or models."""

import copy
import json
import random
import sqlite3

import pytest

from clir_bench.domains.legal.un import uprprc as u


def _words(number, prefix="provision"):
    return " ".join(f"{prefix}{index}" for index in range(number))


def _source(english, *, identifier="source-1", year="2024-03-25", other=None):
    original = english if isinstance(english, list) else [english]
    by_language = {
        language: list(original)
        if language == "en"
        else list(other)
        if other is not None
        else [f"{language} translated paragraph {i}." for i in range(len(original))]
        for language in u.REQUIRED_LANGUAGES
    }
    row = {
        "id": identifier,
        "symbol": "S/RES/2728 (2024)",
        "publication_date": year,
        "title": "Peace and security",
        "symbols": ["", "", "S/RES/2728 (2024)"],
        "job_numbers": [f"job-{language}" for language in u.SOURCE_LANGUAGES],
        "ftxt": [
            "\n\n".join(by_language[language]) if language in by_language else ""
            for language in u.SOURCE_LANGUAGES
        ],
    }
    return row


def _link(source, language, english=(0,), other=None, *, reverse=False):
    other = english if other is None else other
    texts = dict(zip(u.SOURCE_LANGUAGES, source["ftxt"]))
    languages = (language, "en") if reverse else ("en", language)
    numbers = (other, english) if reverse else (english, other)
    row = {
        "id": source["id"],
        "src_lang": languages[0],
        "dst_lang": languages[1],
        "src_rate": 0.47,
        "dst_rate": 0.51,
        "clean_para_index_set_pair": "|".join(",".join(map(str, side)) for side in numbers),
    }
    for side, lang, indices in zip(("src", "dst"), languages, numbers):
        row[f"{side}_job_number"] = source["job_numbers"][u.SOURCE_LANGUAGES.index(lang)]
        row[f"{side}_text"] = "\n".join(texts[lang].split("\n\n")[i] for i in indices)
    return row


def _links(source, *, paragraphs=None):
    if paragraphs is None:
        paragraphs = range(len(source["ftxt"][2].split("\n\n")))
    return [
        _link(source, language, (index,), reverse=language in ("ar", "fr"))
        for index in paragraphs
        for language in u.OTHER_LANGUAGES
    ]


def _read(output):
    with sqlite3.connect(output / "sources.sqlite") as connection:
        docs = [
            json.loads(row[0])
            for row in connection.execute("SELECT payload FROM docs ORDER BY doc_id")
        ]
        blocks = [
            json.loads(row[0])
            for row in connection.execute("SELECT payload FROM blocks ORDER BY doc_id,block_index")
        ]
        texts = {
            (row[0], row[1]): row[2]
            for row in connection.execute("SELECT doc_id,language,text FROM texts")
        }
    return docs, blocks, texts


def test_roundtrip_preserves_all_six_originals_and_absolute_unicode_spans(tmp_path):
    source = _source(["United Nations", _words(180)])
    source["ftxt"][0] = "الأمم المتحدة\n\nالنص العربي الأصلي الدقيق."
    output = tmp_path / "prepared"
    manifest = u.prepare_records([source], _links(source, paragraphs=[1]), output)
    docs, blocks, texts = _read(output)
    assert docs[0]["doc_id"] == f"uprprc:{u.FILEWISE_REVISION}:source-1"
    assert docs[0]["target_idxs"] == [1]
    assert blocks[0]["texts"] == {"en": "United Nations"}
    assert not blocks[0]["usable"]
    target = blocks[1]
    assert target["line_start"] is target["line_end"] is None
    assert set(target["texts"]) == set(u.REQUIRED_LANGUAGES)
    for language in u.REQUIRED_LANGUAGES:
        original = source["ftxt"][u.SOURCE_LANGUAGES.index(language)]
        location = target["source_locations"][language]
        assert texts[(docs[0]["doc_id"], language)] == original
        assert original[location["start"] : location["end"]] == target["texts"][language]
        assert location["start"] == len(original.split("\n\n")[0]) + 2
        assert location["document_sha256"] == u.sha256_text(original)
    assert docs[0]["char_count"] == len(source["ftxt"][2])
    assert manifest["counts"]["targets"] == 1
    assert manifest["publication_ranges"]["targets"] == {
        "first": "2024-03-25",
        "last": "2024-03-25",
    }
    assert manifest["coverage"]["complete_upstream_snapshot"] is False
    assert u.audit(output / "manifest.json")["targets"] == 1


def test_every_english_paragraph_needs_every_pivot_pair(tmp_path):
    source = _source([_words(100), _words(100, "second")])
    # All six languages occur in one component. English paragraph 1 nonetheless
    # lacks four required mappings: language presence alone must never pass.
    links = [_link(source, "fr", (0, 1), (0,))]
    links += [
        _link(source, language, (0,), (0,)) for language in u.OTHER_LANGUAGES if language != "fr"
    ]
    manifest = u.prepare_records([source], links, tmp_path / "prepared")
    assert manifest["counts"]["targets"] == 0
    assert manifest["rejected"]["incomplete_multiway_coverage"] == 1
    assert all(not block["usable"] for block in _read(tmp_path / "prepared")[1])


def _many_to_many_source():
    source = _source([_words(90), _words(90, "second"), _words(90, "third")])
    links = [_link(source, "fr", (0, 1), (0, 1))]
    links += [
        _link(source, language, (index,), (index,))
        for language in u.OTHER_LANGUAGES
        if language != "fr"
        for index in (0, 1)
    ]
    links += [_link(source, language, (2,), (2,)) for language in u.OTHER_LANGUAGES]
    return source, links


def test_many_to_many_components_are_sorted_and_packed_as_raw_slices(tmp_path):
    source, links = _many_to_many_source()
    random.Random(123).shuffle(links)
    output = tmp_path / "prepared"
    u.prepare_records([source], links, output)
    _, blocks, _ = _read(output)
    assert len(blocks) == 1
    assert blocks[0]["alignment_components"] == 2
    assert blocks[0]["token_count"] == 270
    assert blocks[0]["text"] == source["ftxt"][2]
    assert blocks[0]["para_start"] == 0 and blocks[0]["para_end"] == 2
    for language in u.REQUIRED_LANGUAGES:
        assert blocks[0]["texts"][language] == source["ftxt"][u.SOURCE_LANGUAGES.index(language)]
    assert u.audit(output / "manifest.json")["targets"] == 1


@pytest.mark.parametrize("gap,expected_targets", [("", 1), ("unmapped substantive paragraph", 0)])
def test_only_whitespace_gaps_inside_a_component_are_allowed(tmp_path, gap, expected_targets):
    source = _source(
        [_words(90), gap, _words(90, "tail")], other=["translated first", "", "translated last"]
    )
    links = [_link(source, language, (0, 2), (0, 2)) for language in u.OTHER_LANGUAGES]
    output = tmp_path / "prepared"
    manifest = u.prepare_records([source], links, output)
    assert manifest["counts"]["targets"] == expected_targets
    if expected_targets:
        assert _read(output)[1][0]["text"] == source["ftxt"][2]
    else:
        assert manifest["rejected"]["noncontiguous_paragraph_set"] == 1


def test_crossed_document_order_rejects_both_components(tmp_path):
    source = _source([_words(180), _words(180, "tail")])
    links = [
        _link(source, language, (index,), (1 - index if language == "fr" else index,))
        for index in (0, 1)
        for language in u.OTHER_LANGUAGES
    ]
    manifest = u.prepare_records([source], links, tmp_path / "prepared")
    assert manifest["counts"]["targets"] == 0
    assert manifest["rejected"]["crossing_or_overlapping_components"] == 2


def test_english_spans_must_not_overlap_even_only_in_whitespace(tmp_path):
    source = _source([_words(180), "", "", _words(180, "tail")])
    links = [
        _link(source, language, english, other)
        for language in u.OTHER_LANGUAGES
        for english, other in (((0, 2), (0,)), ((1, 3), (1,)))
    ]
    manifest = u.prepare_records([source], links, tmp_path / "prepared")
    assert manifest["counts"]["targets"] == 0
    assert manifest["rejected"]["crossing_or_overlapping_components"] == 2


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("src_job_number", "another-version", "job_number_mismatch"),
        ("dst_job_number", "another-version", "job_number_mismatch"),
        ("src_text", "not the original text", "paragraph_text_mismatch"),
        ("clean_para_index_set_pair", "999|0", "paragraph_index_out_of_bounds"),
        ("clean_para_index_set_pair", "1,0|0", "unordered_or_duplicate_paragraph_indices"),
        ("clean_para_index_set_pair", "0,0|0", "unordered_or_duplicate_paragraph_indices"),
        ("clean_para_index_set_pair", "-1|0", "invalid_paragraph_indices"),
    ],
)
def test_a_bad_pivot_link_quarantines_the_document(tmp_path, field, value, reason):
    source = _source(_words(180))
    links = _links(source)
    malformed = dict(links[0], **{field: value})
    manifest = u.prepare_records([source], links + [malformed], tmp_path / "prepared")
    docs, blocks, texts = _read(tmp_path / "prepared")
    assert manifest["counts"]["targets"] == 0
    assert manifest["counts"]["quarantined_documents"] == 1
    assert docs[0]["invalid_reason"] == reason
    assert texts[(docs[0]["doc_id"], "en")] == source["ftxt"][2]
    assert all(not block["usable"] for block in blocks)


def test_identity_and_exact_join_are_verified():
    source = _source([_words(90), _words(90, "second")])
    paragraphs = {
        language: u.Paragraphs.of(source["ftxt"][index])
        for index, language in enumerate(u.REQUIRED_LANGUAGES)
    }
    link = _link(source, "fr", (0, 1), (0, 1))
    assert "\n\n" not in link["src_text"]
    assert u.verify_link(link, source, paragraphs).english == (0, 1)
    with pytest.raises(u.SourceValidationError, match="document_identity"):
        u.verify_link(dict(link, id="other-source"), source, paragraphs)
    with pytest.raises(u.SourceValidationError, match="paragraph_text"):
        u.verify_link(dict(link, src_text=source["ftxt"][2]), source, paragraphs)


def test_oversized_component_is_kept_intact_and_never_targeted(tmp_path):
    source = _source(_words(451))
    manifest = u.prepare_records([source], _links(source), tmp_path / "prepared")
    blocks = _read(tmp_path / "prepared")[1]
    assert len(blocks) == 1 and blocks[0]["token_count"] == 451
    assert set(blocks[0]["texts"]) == set(u.REQUIRED_LANGUAGES)
    assert blocks[0]["usable"] and not blocks[0]["in_range"]
    assert manifest["counts"]["targets"] == 0


def test_greedy_packing_never_exceeds_450_or_trims_a_tail(tmp_path):
    source = _source([_words(200), _words(200, "second"), _words(100, "tail")])
    manifest = u.prepare_records([source], _links(source), tmp_path / "prepared")
    blocks = _read(tmp_path / "prepared")[1]
    assert [block["token_count"] for block in blocks] == [400, 100]
    assert blocks[0]["text"] == "\n\n".join(source["ftxt"][2].split("\n\n")[:2])
    assert blocks[1]["text"] == source["ftxt"][2].split("\n\n")[2]
    assert manifest["counts"]["targets"] == 1


def test_unmapped_foreign_content_prevents_packing_across_a_gap(tmp_path):
    source = _source([_words(160), _words(160, "tail")])
    source["ftxt"][3] = "first translation\n\nunmapped foreign content\n\nsecond translation"
    links = [
        _link(source, language, (index,), (2 if index == 1 and language == "fr" else index,))
        for index in (0, 1)
        for language in u.OTHER_LANGUAGES
    ]
    manifest = u.prepare_records([source], links, tmp_path / "prepared")
    blocks = _read(tmp_path / "prepared")[1]
    assert len(blocks) == 2 and manifest["counts"]["targets"] == 2
    assert "unmapped" not in "".join(block["texts"]["fr"] for block in blocks)


def test_annex_boundary_and_mixed_furniture_are_never_hidden(tmp_path):
    source = _source([_words(160), "Annex I", _words(160, "annex")])
    u.prepare_records([source], _links(source), tmp_path / "annex")
    docs, blocks, _ = _read(tmp_path / "annex")
    assert [block["part"] for block in blocks] == ["body", "annex"]
    assert docs[0]["n_targets"] == 2
    assert docs[0]["annexes"] == [
        {"part_id": "anx_1", "kind": "annex", "label": "I", "block_start": 1, "block_end": 1}
    ]
    mixed = _source(["United Nations", _words(180)])
    links = [_link(mixed, language, (0, 1), (0, 1)) for language in u.OTHER_LANGUAGES]
    manifest = u.prepare_records([mixed], links, tmp_path / "mixed")
    assert manifest["counts"]["targets"] == 0
    assert "mixed_furniture_content" in _read(tmp_path / "mixed")[1][0]["junk_flags"]


@pytest.mark.parametrize("language", u.REQUIRED_LANGUAGES)
def test_empty_actual_language_text_cannot_be_replaced_by_metadata(tmp_path, language):
    source = _source(_words(180))
    source["ftxt"][u.SOURCE_LANGUAGES.index(language)] = " \n "
    manifest = u.prepare_records([source], [], tmp_path / "prepared")
    assert manifest["counts"]["documents"] == 0
    assert manifest["rejected"]["missing_required_language_text"] == 1


def test_older_newer_and_undated_originals_are_reference_only(tmp_path):
    sources = [
        _source(_words(180), identifier=f"source-{index}", year=year)
        for index, year in enumerate(("2014-12-31", "2024-03-25", "2026-01-01", None))
    ]
    manifest = u.prepare_records(
        sources, [link for source in sources for link in _links(source)], tmp_path / "prepared"
    )
    docs, blocks, _ = _read(tmp_path / "prepared")
    assert len(docs) == 4 and manifest["counts"]["targets"] == 1
    assert [doc["n_targets"] for doc in docs] == [0, 1, 0, 0]
    assert all(len(block["texts"]) == 6 for block in blocks)
    assert manifest["publication_ranges"]["documents"] == {
        "first": "2014-12-31",
        "last": "2026-01-01",
    }
    assert manifest["publication_ranges"]["targets"] == {
        "first": "2024-03-25",
        "last": "2024-03-25",
    }


def test_conflicting_duplicate_source_versions_cannot_generate(tmp_path):
    source = _source(_words(180))
    conflict = copy.deepcopy(source)
    conflict["ftxt"][2] += " modified"
    manifest = u.prepare_records([source, conflict], _links(source), tmp_path / "prepared")
    assert manifest["counts"]["targets"] == 0
    assert _read(tmp_path / "prepared")[0][0]["invalid_reason"] == "conflicting_source_versions"


def test_invalid_publication_dates_are_reference_only(tmp_path):
    source = _source(_words(180), year="2024-02-31")
    manifest = u.prepare_records([source], _links(source), tmp_path / "prepared")
    assert manifest["counts"]["documents"] == 1
    assert manifest["counts"]["targets"] == 0
    assert _read(tmp_path / "prepared")[0][0]["year"] == ""


def test_database_is_deterministic_under_record_and_link_reordering(tmp_path):
    first, links = _many_to_many_source()
    second = _source(_words(180), identifier="another-source")
    links += _links(second)
    first_manifest = u.prepare_records([first, second], links, tmp_path / "first")
    second_manifest = u.prepare_records([second, first], list(reversed(links)), tmp_path / "second")
    assert first_manifest == second_manifest
    assert (tmp_path / "first" / "sources.sqlite").read_bytes() == (
        tmp_path / "second" / "sources.sqlite"
    ).read_bytes()


def test_quarantine_is_deterministic_under_conflicting_input_reordering(tmp_path):
    source = _source(_words(180))
    conflict = copy.deepcopy(source)
    conflict["ftxt"][2] += " modified"
    links = _links(source)
    links.extend(
        [
            dict(links[0], src_job_number="another-version"),
            dict(links[0], src_text="incorrect source"),
        ]
    )
    first = u.prepare_records([source, conflict, source], links, tmp_path / "first")
    second = u.prepare_records([source, conflict, source][::-1], links[::-1], tmp_path / "second")
    assert first == second
    assert (tmp_path / "first" / "sources.sqlite").read_bytes() == (
        tmp_path / "second" / "sources.sqlite"
    ).read_bytes()


def test_existing_output_and_interrupted_build_never_publish_a_manifest(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        u.prepare_records([], [], existing)

    def broken():
        yield _source(_words(180))
        raise RuntimeError("input stream interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        u.prepare_records(broken(), [], tmp_path / "interrupted")
    assert not (tmp_path / "interrupted").exists()
    assert not list(tmp_path.glob(".interrupted.building-*"))


def _rehash_manifest(output):
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["database_sha256"] = u.sha256_file(output / "sources.sqlite")
    path.write_text(json.dumps(manifest))


def test_audit_rechecks_per_paragraph_multiway_coverage(tmp_path):
    source, links = _many_to_many_source()
    output = tmp_path / "prepared"
    u.prepare_records([source], links, output)
    with sqlite3.connect(output / "sources.sqlite") as connection:
        block = json.loads(connection.execute("SELECT payload FROM blocks").fetchone()[0])
        block["alignment_links"] = [
            link
            for link in block["alignment_links"]
            if not (link["language"] == "ru" and link["english"] == [1])
        ]
        connection.execute("UPDATE blocks SET payload=?", (json.dumps(block),))
    _rehash_manifest(output)
    with pytest.raises(u.SourceValidationError, match="incomplete_or_crossing_alignment_proof"):
        u.audit(output / "manifest.json")


def test_selected_block_verification_rechecks_proof_and_raw_sources(tmp_path):
    source, links = _many_to_many_source()
    output = tmp_path / "prepared"
    manifest = u.prepare_records([source], links, output)
    docs, blocks, stored = _read(output)
    texts = {language: stored[(docs[0]["doc_id"], language)] for language in u.REQUIRED_LANGUAGES}
    u.verify_block(blocks[0], docs[0], texts, manifest["config"])
    broken = copy.deepcopy(blocks[0])
    broken["alignment_links"] = [
        link
        for link in broken["alignment_links"]
        if not (link["language"] == "ru" and link["english"] == [1])
    ]
    with pytest.raises(u.SourceValidationError, match="incomplete_or_crossing_alignment_proof"):
        u.verify_block(broken, docs[0], texts, manifest["config"])
    broken = copy.deepcopy(blocks[0])
    broken["source_locations"]["zh"]["start"] += 1
    with pytest.raises(u.SourceValidationError, match="invalid_original_text_span"):
        u.verify_block(broken, docs[0], texts, manifest["config"])


def test_local_parquet_cli_hashes_actual_input_shards(tmp_path, capsys):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = _source(_words(180))
    filewise = tmp_path / "filewise"
    bilingual = tmp_path / "bilingual"
    filewise.mkdir()
    (bilingual / "train").mkdir(parents=True)
    source_path = filewise / "part.parquet"
    links_path = bilingual / "train" / "part.parquet"
    pq.write_table(pa.Table.from_pylist([source]), source_path)
    pq.write_table(pa.Table.from_pylist(_links(source)), links_path)
    output = tmp_path / "prepared"
    u.main(
        [
            "prepare",
            "--filewise",
            str(filewise),
            "--bilingual",
            str(bilingual),
            "--out-dir",
            str(output),
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sources"]["filewise"]["inputs"][0]["sha256"] == u.sha256_file(source_path)
    assert manifest["sources"]["bilingual"]["inputs"][0]["sha256"] == u.sha256_file(links_path)
    assert json.loads(capsys.readouterr().out)["targets"] == 1
    u.main(["audit", str(output / "manifest.json")])
    assert json.loads(capsys.readouterr().out)["targets"] == 1


def test_legacy_reference_paths_are_all_or_none(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        u.prepare_records([], [], tmp_path / "invalid", legacy_docs=tmp_path / "docs")
    docs, blocks, corpus = tmp_path / "docs", tmp_path / "blocks", tmp_path / "corpus"
    docs.write_text("")
    blocks.write_text("")
    corpus.mkdir()
    manifest = u.prepare_records(
        [], [], tmp_path / "valid", legacy_docs=docs, legacy_blocks=blocks, legacy_sixway_dir=corpus
    )
    assert manifest["legacy"] == {
        "docs": str(docs),
        "blocks": str(blocks),
        "sixway_dir": str(corpus),
    }
