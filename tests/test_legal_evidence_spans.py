"""Future evidence is an exact, minimal source-bound excerpt, never inferred on replay."""

import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from clir_bench.domains.legal.qac import eurlex_context as ec
from clir_bench.domains.legal.qac import eurlex_generate as eg
from clir_bench.domains.legal.qac import evidence
from clir_bench.domains.legal.qac import generation_metadata as annotations
from clir_bench.domains.legal.qac import un_context as uc
from clir_bench.domains.legal.qac import un_generate as ug


@pytest.fixture
def legal_payload():
    target = ec.ArticleUnit(
        "eli:target", "ACT", "4", texts={
            "en": "Heading\nOfficers report annually unless exempt. Other unrelated duties apply.",
            "fr": "Titre\nLes agents rendent compte chaque année sauf exemption. Autres obligations.",
        },
    )
    reference = ec.ArticleUnit("eli:reference", "ACT", "3", texts={
        "en": "Auditors check annual reports. An unrelated paragraph follows.",
        "fr": "Les auditeurs vérifient les rapports annuels.",
    })
    external = ec.ArticleUnit("eli:external", "OTHER", "8", texts={
        "en": "Exemption requires written approval. Irrelevant text follows.",
    })
    sources = ec.visible_evidence_sources(
        target, [reference], external=[external], languages=("en", "fr"), reference_chars=40,
    )
    payload = ec.GenerationPayload(
        target, [reference], [],
        ec.render_payload(target, [reference], external=[external],
                          languages=("en", "fr"), reference_chars=40),
        external_references=[external], evidence_sources=sources,
    )
    return payload


def candidate(**changes):
    item = {
        "question": "When do officers report?", "answer": "Officers report annually unless exempt.",
        "articles_involved": ["4"], "answer_is_translation": False,
        "evidence_spans": [{"source_ref": "4", "language": "en",
                            "text": "Officers report annually unless exempt."}],
    }
    return {**item, **changes}


@pytest.mark.parametrize("mode", eg.MODES)
def test_every_eurlex_mode_preserves_exact_evidence_and_roundtrips(legal_payload, mode):
    item = candidate()
    if mode == "source_finding":
        item.update(answer="Article 4 of ACT", source_identifier="ACT", source_article="4")
    parsed, = eg.parse_candidates([item], legal_payload, mode, require_evidence=True)
    assert parsed.evidence_contract == evidence.CONTRACT
    span, = parsed.evidence_spans
    assert span["source_id"] == legal_payload.target.eli_id
    assert legal_payload.target.texts["en"][span["start"]:span["end"]] == parsed.evidence
    assert span["source_text_sha256"] == hashlib.sha256(
        legal_payload.target.texts["en"].encode()).hexdigest()
    assert parsed.evidence_is_translation is False
    assert eg.Candidate(**asdict(parsed)) == parsed
    row = annotations.row_metadata(parsed)
    assert json.loads(row["evidence_spans"]) == parsed.evidence_spans
    restored = annotations.restore_metadata(row, mode, eurlex=True)
    assert eg.Candidate(**{**asdict(parsed), **restored}) == parsed


def test_cross_act_coverage_and_primary_reference(legal_payload):
    item = candidate(articles_involved=["4", "OTHER:8"], answer="Exemption requires written approval.")
    item["evidence_spans"].insert(0, {
        "source_ref": "OTHER:8", "language": "en", "text": "Exemption requires written approval.",
    })
    parsed, = eg.parse_candidates([item], legal_payload, "comparison", require_evidence=True)
    assert parsed.multi_article and parsed.cross_act
    assert parsed.evidence_source_id == "eli:external"
    assert {span["source_id"] for span in parsed.evidence_spans} == {"eli:target", "eli:external"}
    assert parsed.evidence == item["answer"]  # no concatenation with the target excerpt
    with pytest.raises(ValueError, match="missing for involved"):
        eg.parse_candidates([{**item, "evidence_spans": item["evidence_spans"][:1]}],
                            legal_payload, "comparison", require_evidence=True)


def test_original_language_quote_survives_translated_answer(legal_payload):
    item = candidate(answer_is_translation=True, answer="Les agents rendent compte chaque année sauf exemption.")
    parsed, = eg.parse_candidates([item], legal_payload, "lookup", require_evidence=True)
    assert parsed.answer_is_translation is True
    assert parsed.evidence_language == "en" and parsed.evidence_is_translation is False
    assert parsed.evidence != parsed.answer


@pytest.mark.parametrize("change,match", [
    ({"evidence_spans": []}, "nonempty list"),
    ({"evidence_spans": [{"source_ref": "4", "language": "de", "text": "Officers"}]}, "not supplied"),
    ({"evidence_spans": [{"source_ref": "4", "language": "en", "text": "officers"}]}, "exact visible"),
    ({"evidence_spans": [{"source_ref": "missing", "language": "en", "text": "Officers"}]}, "not an involved"),
    ({"evidence_spans": [{"source_ref": "4", "language": "en", "text": ""}]}, "nonempty string"),
    ({"answer_is_translation": "false"}, "JSON boolean"),
    ({"articles_involved": ["4", "unknown"]}, "supplied source keys"),
])
def test_invalid_new_evidence_raises(legal_payload, change, match):
    with pytest.raises(ValueError, match=match):
        eg.parse_candidates([candidate(**change)], legal_payload, "lookup", require_evidence=True)


def test_source_finding_primary_must_be_target(legal_payload):
    item = candidate(articles_involved=["4", "3"])
    item["evidence_spans"].insert(0, {"source_ref": "3", "language": "en", "text": "Auditors"})
    with pytest.raises(ValueError, match="start with the target"):
        eg.parse_candidates([item], legal_payload, "source_finding", require_evidence=True)


def test_whole_article_and_duplicate_source_rejected(legal_payload):
    item = candidate(evidence_spans=[{
        "source_ref": "4", "language": "en", "text": legal_payload.target.texts["en"],
    }])
    with pytest.raises(ValueError, match="whole-article"):
        eg.parse_candidates([item], legal_payload, "lookup", require_evidence=True)
    item = candidate()
    item["evidence_spans"] *= 2
    with pytest.raises(ValueError, match="duplicate evidence"):
        eg.parse_candidates([item], legal_payload, "lookup", require_evidence=True)


def test_reference_truncation_does_not_admit_hidden_text_or_synthetic_marker(legal_payload):
    for quote in ("unrelated paragraph follows.", "[…truncated]"):
        item = candidate(articles_involved=["4", "3"])
        item["evidence_spans"].append({"source_ref": "3", "language": "en", "text": quote})
        with pytest.raises(ValueError, match="exact visible"):
            eg.parse_candidates([item], legal_payload, "lookup", require_evidence=True)


def test_real_eurlex_builder_records_languages_and_custom_limit(legal_payload):
    index = ec.ArticleIndex.__new__(ec.ArticleIndex)
    index.by_eli = {unit.eli_id: unit for unit in [legal_payload.target, *legal_payload.references]}
    index.references = {legal_payload.target.eli_id: [legal_payload.references[0].eli_id]}
    built = index.build(legal_payload.target.eli_id, languages=("en",), reference_chars=12)
    assert set(built.evidence_sources) == {("4", "en"), ("3", "en")}
    assert built.evidence_sources["3", "en"].text == "Auditors che"
    assert "Auditors che […truncated]" in built.text
    assert "Les agents" not in built.text


def test_capped_annex_has_original_hash_but_no_synthetic_evidence(tmp_path, legal_payload):
    original = "An essential safeguard applies. " + "Unrelated text. " * 1600
    row = {"eli_id": "http://example/eli/anx_1/oj", "celex_id": "ACT",
           "unit_type": "annex", "language": "en", "text": original}
    path = tmp_path / "articles.jsonl"
    path.write_text(json.dumps(row) + "\n")
    index = ec.ArticleIndex.__new__(ec.ArticleIndex)
    index.by_eli = {}
    index._load_articles(path, annex_whitelist={row["eli_id"]})
    annex = index.by_eli[row["eli_id"]]
    assert annex.texts["en"].endswith("[…truncated]")
    sources = ec.visible_evidence_sources(legal_payload.target, [], annexes=[annex],
                                         languages=("en",), reference_chars=None)
    source = sources["ACT:anx_1", "en"]
    assert "[…truncated]" not in source.text
    payload = replace(legal_payload, references=[], external_references=[], annexes=[annex],
                      evidence_sources=sources)
    item = candidate(articles_involved=["4", "ACT:anx_1"])
    item["evidence_spans"].append({"source_ref": "ACT:anx_1", "language": "en",
                                   "text": "An essential safeguard applies."})
    parsed, = eg.parse_candidates([item], payload, "lookup", require_evidence=True)
    span = parsed.evidence_spans[1]
    assert span["source_text_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert original[span["start"]:span["end"]] == span["text"]
    item["evidence_spans"][1]["text"] = "[…truncated]"
    with pytest.raises(ValueError, match="exact visible"):
        eg.parse_candidates([item], payload, "lookup", require_evidence=True)


def un_payload():
    block = uc.BlockUnit("doc#1", "doc", "S/1", "Title", 1, 3, 12, 13, 10, True,
                         texts={"en": "Heading\nThe Council called for dialogue. Other material.",
                                "fr": "Le Conseil a demandé le dialogue."})
    context = replace(block, block_id="doc#2", block_index=2,
                      texts={"en": "A neighbouring block alone contains sanctions."})
    return uc.GenerationPayload(block, [context], 0, [], [],
                                uc.render_payload(block, [context]),
                                uc.visible_evidence_sources(block, ("en",)))


@pytest.mark.parametrize("mode", ug.MODES)
def test_every_un_mode_preserves_target_bound_offsets(mode):
    payload = un_payload()
    item = candidate(answer="The Council called for dialogue.", evidence_spans=[{
        "source_ref": "target", "language": "en", "text": "The Council called for dialogue.",
    }])
    parsed, = ug.parse_candidates([item], mode, payload=payload, require_evidence=True)
    span, = parsed.evidence_spans
    assert span["source_id"] == "doc#1"
    assert (span["doc_id"], span["line_start"], span["line_end"]) == ("doc", 12, 13)
    assert span["start"] == len("Heading\n")
    row = annotations.row_metadata(parsed)
    assert annotations.restore_metadata(row, mode)["evidence_spans"] == [span]


@pytest.mark.parametrize("span", [
    {"source_ref": "target", "language": "en", "text": "contains sanctions"},
    {"source_ref": "doc#2", "language": "en", "text": "contains sanctions"},
    {"source_ref": "target", "language": "fr", "text": "Le Conseil"},
])
def test_un_cannot_quote_context_or_unsupplied_translation(span):
    with pytest.raises(ValueError):
        ug.parse_candidates([candidate(evidence_spans=[span])], "lookup",
                            payload=un_payload(), require_evidence=True)


def test_skip_is_not_malformed_output(legal_payload):
    assert eg.parse_candidates([{"skip_reason": "No sufficient short passage"}], legal_payload,
                               "lookup", require_evidence=True) == []
    for data in ([], {}, [candidate(), {"skip_reason": "other"}]):
        with pytest.raises(ValueError):
            eg.parse_candidates(data, legal_payload, "lookup", require_evidence=True)


def test_generation_enforces_resolved_prompt_contract_only(monkeypatch, legal_payload):
    from clir_bench.core import llm
    legacy = {"question": "Who?", "answer": "Officers", "articles_involved": ["4"]}
    monkeypatch.setattr(llm, "chat", lambda *a, **kw: json.dumps([legacy]))
    for generator, payload in ((eg, legal_payload), (ug, un_payload())):
        monkeypatch.setattr(generator, "PROMPTS", SimpleNamespace(generation=lambda *a: "Old prompt"))
        old, = generator.generate(payload, mode="lookup", language="en", model="test", client=None)
        assert old.evidence_spans == [] and old.evidence_contract == ""
        monkeypatch.setattr(generator, "PROMPTS", SimpleNamespace(generation=lambda *a: evidence.MARKER))
        with pytest.raises(ValueError):
            generator.generate(payload, mode="lookup", language="en", model="test", client=None)


def test_verbatim_whitespace_unicode_offsets_and_legacy_defaults(legal_payload):
    source = evidence.EvidenceSource("4", "eli:target", "en", "α\n  β  γ\nrest", "α\n  β  γ\nrest", "article")
    payload = replace(legal_payload, evidence_sources={("4", "en"): source})
    item = candidate(evidence_spans=[{"source_ref": "4", "language": "en", "text": "  β  γ"}])
    parsed, = eg.parse_candidates([item], payload, "lookup", require_evidence=True)
    assert parsed.evidence == "  β  γ"
    assert parsed.evidence_spans[0]["start"] == 2
    assert annotations.restore_metadata(annotations.row_metadata(parsed), "lookup")["evidence"] == "  β  γ"
    legacy, = eg.parse_candidates([{"question": "Who?", "answer": "Officers"}], legal_payload, "lookup")
    assert legacy.evidence_spans == [] and legacy.evidence_contract == ""
