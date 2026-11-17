"""Payload assembly: priority context slots and annex-anchored references."""

from __future__ import annotations

import json

import pytest

from clir_bench.domains.legal.qac import un_context as ctx


def _unit(i, text, part_id="", part_label=""):
    return ctx.BlockUnit(
        block_id=f"d#{i}", doc_id="d", symbol="S/RES/1(1999)", title="t",
        block_index=i, n_blocks=6, line_start=i, line_end=i, token_count=10,
        in_range=True, part_id=part_id, part_label=part_label,
        texts={"en": text})


def test_select_context_priority_beats_neighbours():
    blocks = [_unit(i, f"block {i} " + "x" * 90) for i in range(6)]
    # budget fits the opening plus ONE more block; the anchored sibling (4)
    # must win over the immediate neighbours of target 1.
    chosen = ctx._select_context(blocks, 1, budget=2 * 105, priority=[4])
    assert [b.block_index for b in chosen] == [0, 4]


def test_select_context_without_priority_keeps_neighbour_order():
    blocks = [_unit(i, f"block {i} " + "x" * 90) for i in range(6)]
    chosen = ctx._select_context(blocks, 1, budget=2 * 105)
    assert [b.block_index for b in chosen] == [0, 2]


class _MiniIndex(ctx.BlockIndex):
    """BlockIndex over in-memory rows, no files."""

    def __init__(self, docs, blocks_by_doc, tmp_path):
        docs_path = tmp_path / "docs.jsonl"
        docs_path.write_text("".join(json.dumps(d) + "\n" for d in docs.values()))
        self._rows = blocks_by_doc
        super().__init__(blocks_path=tmp_path / "missing_blocks.jsonl",
                         docs_path=docs_path,
                         status_path=tmp_path / "missing_status.jsonl")

    def blocks_for(self, doc_id):
        return [ctx.BlockUnit.from_row(r) for r in self._rows[doc_id]]


def _row(doc_id, i, text, part="body", part_id="", part_label=""):
    return {"block_id": f"{doc_id}#{i}", "doc_id": doc_id,
            "symbol": "S/RES/1265(1999)" if "1265" in doc_id else "S/RES/1(1999)",
            "title": "PROTECTION OF CIVILIANS", "block_index": i, "n_blocks": 3,
            "line_start": i, "line_end": i, "token_count": 10,
            "in_range": True, "usable": True, "part": part, "part_id": part_id,
            "part_label": part_label, "text": text}


def test_reference_renders_the_cited_annex_block(tmp_path):
    cited = "1999/s/res/1265_1999_"
    citing = "1999/s/res/1_1999_"
    docs = {
        cited: {"doc_id": cited, "symbol": "S/RES/1265(1999)",
                "title": "PROTECTION OF CIVILIANS", "n_blocks": 3,
                "n_targets": 3, "target_idxs": [0, 1, 2],
                "annexes": [{"part_id": "anx_1", "kind": "annex", "label": "I",
                             "block_start": 2, "block_end": 2}], "offset": 0},
        citing: {"doc_id": citing, "symbol": "S/RES/1(1999)", "title": "T",
                 "n_blocks": 1, "n_targets": 1, "target_idxs": [0],
                 "annexes": [], "offset": 0},
    }
    blocks = {
        cited: [_row(cited, 0, "1. Head block of the cited resolution;"),
                _row(cited, 1, "2. Second operative paragraph;"),
                _row(cited, 2, "Annex I\nTHE ANNEXED FRAMEWORK TEXT",
                     part="annex", part_id="anx_1", part_label="Annex I")],
        citing: [_row(citing, 0,
                      "Recalling annex I to resolution 1265 (1999), decides;")],
    }
    index = _MiniIndex(docs, blocks, tmp_path)
    payload = index.build(citing, 0)
    (ref,) = payload.references
    assert ref.part_label == "Annex I"
    assert "ANNEXED FRAMEWORK" in ref.text
    assert "(cited annex Annex I)" in payload.text
    assert index.incomplete is None    # status file absent -> gate unavailable


def test_priority_anchor_overflow_does_not_end_the_walk():
    blocks = [_unit(i, f"block {i} " + "x" * 90) for i in range(6)]
    blocks[4] = _unit(4, "y" * 5000)          # oversized anchor
    chosen = ctx._select_context(blocks, 1, budget=2 * 105, priority=[4, 3])
    # anchor 4 does not fit; anchor 3 and the opening still do
    assert [b.block_index for b in chosen] == [0, 3]


def test_custom_paths_do_not_gate_against_the_production_status(tmp_path):
    import json as _json
    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text(_json.dumps({
        "doc_id": "d", "symbol": "S/RES/1(1999)", "title": "t",
        "n_blocks": 0, "n_targets": 0, "target_idxs": [], "offset": 0}) + "\n")
    index = ctx.BlockIndex(blocks_path=tmp_path / "blocks.jsonl",
                           docs_path=docs_path)
    assert index.incomplete is None


@pytest.mark.parametrize("language,text", [
    ("fr", "La traduction."),
    ("ar", "الترجمة العربية."),
    ("ru", "Русский перевод."),
])
def test_translation_lookup_keeps_the_standalone_path_hook(tmp_path, monkeypatch, language, text):
    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text(json.dumps({
        "doc_id": "d", "symbol": "S/RES/1(1999)", "n_blocks": 0,
        "offset": 0, "line_start": 1, "line_end": 1,
    }) + "\n")
    translated = tmp_path / f"translation.{language}"
    translated.write_text(text + "\n")
    monkeypatch.setattr(ctx.paths, "text_file", lambda language: translated)
    index = ctx.BlockIndex(blocks_path=tmp_path / "blocks.jsonl", docs_path=docs_path)
    index.preload_translations([language], ["d"])
    assert index._translations[language]["d"] == [text]


@pytest.mark.parametrize("language,text,question", [
    ("ar", "يطلب المجلس تقديم تقارير سنوية.", "ماذا يطلب المجلس بشأن التقارير؟"),
    ("ru", "Совет запрашивает ежегодные доклады.", "Какие доклады запрашивает Совет?"),
])
def test_arabic_and_russian_payloads_preserve_original_evidence(
    tmp_path, monkeypatch, language, text, question,
):
    from clir_bench.domains.legal.qac import un_generate

    monkeypatch.delenv("CLIR_PROMPT_MANIFEST", raising=False)
    monkeypatch.setenv("CLIR_PROMPT_SOURCE", "local")
    docs = {"d": {"doc_id": "d", "symbol": "S/RES/1(1999)", "n_blocks": 1,
                  "title": "Reporting", "line_start": 1, "line_end": 1,
                  "target_idxs": [0], "annexes": [], "offset": 0}}
    row = _row("d", 0, "The Council requests annual reports.")
    row.update(line_start=1, line_end=1)
    index = _MiniIndex(docs, {"d": [row]}, tmp_path)
    index._translations[language] = {"d": [text]}
    payload = index.build("d", 0, languages=ctx.payload_languages(language))
    prompt, body = un_generate.build_messages(payload, "lookup", language)
    assert f"[{language.upper()}]" in body["content"]
    assert text in body["content"]
    assert "legal_evidence_spans_v1" in prompt["content"]
    candidates = un_generate.parse_candidates([{
        "question": question, "answer": text, "question_type": "operative_action",
        "question_cited": question, "anchor": "المجلس" if language == "ar" else "Совет",
        "answer_is_translation": False,
        "evidence_spans": [{"source_ref": "target", "language": language, "text": text}],
    }], "lookup", payload=payload, require_evidence=True)
    rows = un_generate.rows_for(payload, candidates, mode="lookup", language=language)
    assert rows[0]["question_language"] == language
    assert rows[0]["answer"] == rows[0]["evidence"] == text
    assert rows[0]["evidence_language"] == language


def test_default_un_modes_have_prompts_for_every_corpus_language(monkeypatch):
    from clir_bench.domains.legal.qac import un_batch
    from clir_bench.domains.legal.un import UN_LANGUAGES

    monkeypatch.delenv("CLIR_PROMPT_MANIFEST", raising=False)
    monkeypatch.setenv("CLIR_PROMPT_SOURCE", "local")
    for mode in un_batch.DEFAULT_MODES:
        for language in UN_LANGUAGES:
            assert un_batch.gen.PROMPTS.generation(mode, language)


def test_references_travel_in_every_payload_language(tmp_path):
    """A cited document is rendered in the question language, not only English.

    Blocks are contiguous line ranges, so the other-language rendering is the
    same ranges read from that language's 6-way file; the payload must show
    both, exactly as it does for the target block.
    """
    cited, citing = "1999/s/res/1265_1999_", "1999/s/res/1_1999_"
    docs = {
        cited: {"doc_id": cited, "symbol": "S/RES/1265(1999)", "n_blocks": 1,
                "title": "PROTECTION OF CIVILIANS", "char_count": 60,
                "line_start": 0, "line_end": 0, "target_idxs": [0], "annexes": []},
        citing: {"doc_id": citing, "symbol": "S/RES/1(1999)", "n_blocks": 1,
                 "title": "RESOLUTION 1", "char_count": 60,
                 "line_start": 0, "line_end": 0, "target_idxs": [0], "annexes": []},
    }
    blocks = {
        cited: [_row(cited, 0, "1. Demands the protection of civilians;")],
        citing: [_row(citing, 0, "Recalling resolution 1265 (1999), decides;")],
    }
    index = _MiniIndex(docs, blocks, tmp_path)
    # Stand in for preload_translations: the French lines of both documents.
    index._translations["fr"] = {
        cited: ["1. Exige la protection des civils;"],
        citing: ["Rappelant la résolution 1265 (1999), décide;"],
    }

    payload = index.build(citing, 0, languages=ctx.payload_languages("fr"))
    (ref,) = payload.references
    assert sorted(ref.texts) == ["en", "fr"]
    assert ref.texts["fr"] == "1. Exige la protection des civils;"
    assert ref.text == "1. Demands the protection of civilians;"   # pivot stays English
    # Both language versions of the reference reach the payload.
    assert "[FR] Reference: S/RES/1265(1999)" in payload.text
    assert "[EN] Reference: S/RES/1265(1999)" in payload.text
    assert "Exige la protection des civils" in payload.text


def test_reference_language_is_omitted_when_the_translation_is_missing(tmp_path):
    """No partial documents: a language with no lines falls back to English."""
    cited, citing = "1999/s/res/1265_1999_", "1999/s/res/1_1999_"
    docs = {
        cited: {"doc_id": cited, "symbol": "S/RES/1265(1999)", "n_blocks": 1,
                "title": "PROTECTION OF CIVILIANS", "char_count": 60,
                "line_start": 0, "line_end": 0, "target_idxs": [0], "annexes": []},
        citing: {"doc_id": citing, "symbol": "S/RES/1(1999)", "n_blocks": 1,
                 "title": "RESOLUTION 1", "char_count": 60,
                 "line_start": 0, "line_end": 0, "target_idxs": [0], "annexes": []},
    }
    blocks = {
        cited: [_row(cited, 0, "1. Demands the protection of civilians;")],
        citing: [_row(citing, 0, "Recalling resolution 1265 (1999), decides;")],
    }
    index = _MiniIndex(docs, blocks, tmp_path)
    index._translations["fr"] = {citing: ["Rappelant la résolution 1265 (1999), décide;"]}

    payload = index.build(citing, 0, languages=ctx.payload_languages("fr"))
    (ref,) = payload.references
    assert sorted(ref.texts) == ["en"]                 # cited doc has no FR lines
    assert "[EN] Reference: S/RES/1265(1999)" in payload.text
    assert "[FR] Reference:" not in payload.text
