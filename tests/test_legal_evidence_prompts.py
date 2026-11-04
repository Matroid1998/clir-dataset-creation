"""The versioned legal evidence contract is present in prompts and their worked outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clir_bench.core import grading, prompt_registry

ROOT = Path(__file__).resolve().parents[1] / "src/clir_bench/domains/legal/qac"
GENERATION = sorted(ROOT.glob("prompts_*/generation/*/*.txt"))
VERIFIERS = sorted(ROOT.glob("prompts_*/verifiers/*.txt"))
MARKER = "EVIDENCE_CONTRACT: legal_evidence_spans_v1"


def candidate_examples(text):
    """Includes inline worked objects and array entries, ignoring source JSON and skips."""
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            return
        try:
            row, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        position = start + length
        if isinstance(row, dict) and "question" in row and "answer" in row:
            yield start, row


@pytest.mark.parametrize("path", GENERATION, ids=lambda p: str(p.relative_to(ROOT)))
def test_generation_contract_and_all_output_examples(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith(MARKER + "\n")
    assert text.count(MARKER) == 1
    examples = list(candidate_examples(text))
    assert examples, f"no exact output shape in {path}"
    for position, row in examples:
        assert type(row.get("answer_is_translation")) is bool
        assert not {"evidence", "evidence_language", "evidence_is_translation"} & row.keys()
        spans = row["evidence_spans"]
        assert spans and all(set(span) == {"source_ref", "language", "text"} for span in spans)
        for span in spans:
            assert all(isinstance(value, str) and value for value in span.values())
            assert span["language"] in {"de", "en", "es", "fr", "zh"}
            if span["text"] != "...":
                assert span["text"] in text[:position], (
                    "A worked evidence quote must exist in its supplied teaching material", path, span
                )
        if "prompts_un" in path.parts:
            assert len(spans) == 1 and spans[0]["source_ref"] == "target"
        # Some legacy examples deliberately show incorrect article declarations, marked ✗.
        line_start = text.rfind("\n", 0, position) + 1
        negative = "✗" in text[line_start:position]
        if "articles_involved" in row and not negative:
            assert {span["source_ref"] for span in spans} == set(row["articles_involved"])
        if path.parent.name == "source_finding":
            assert row["answer_is_translation"] is False
        elif not row["answer_is_translation"]:
            assert row["answer"] == spans[0]["text"]


def test_registered_generation_variants_have_versioned_contract():
    inventory = prompt_registry.local_legal_prompts()
    selected = {key: value for key, value in inventory.items() if "/generation/" in key}
    assert selected
    assert all(value.startswith(MARKER + "\n") for value in selected.values())
    assert {key.split("/")[0] for key in selected} == {"eurlex", "un"}


@pytest.mark.parametrize("path", VERIFIERS, ids=lambda p: str(p.relative_to(ROOT)))
def test_verifiers_check_new_spans_without_changing_rubric_keys(path):
    text = path.read_text(encoding="utf-8")
    assert 'evidence_contract="legal_evidence_spans_v1"' in text
    assert "Legacy candidates without this contract" in text
    assert "no fixed token cap" in text
    assert "smallest sufficient continuous original-language excerpt" in text
    assert "wrong source/language" in text
    if path.name != "faithfulness_batch.txt":
        if path.name not in {"descriptive_batch.txt", "technical_batch.txt"}:
            assert text.startswith("# MODE: ")
        assert grading.rubric_keys(text)
