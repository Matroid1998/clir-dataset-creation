"""Official operative article boundaries survive newer Formex divisions."""

import json
from xml.etree import ElementTree

import pytest

from clir_bench.domains.legal.structure import ACT_LANGUAGES, cellar, paths, segment


def article(number, content="Officials shall report."):
    return (f'<ARTICLE IDENTIFIER="{number:03d}"><TI.ART>Article {number}</TI.ART>'
            f"<STI.ART>Reporting</STI.ART><ALINEA>{content}</ALINEA></ARTICLE>")


def formex():
    quoted = '<GR.SEQ LEVEL="1">' + article(77) + '</GR.SEQ>'
    return ("<ACT><ENACTING.TERMS><TOC>" + article(99) + "</TOC>"
            '<GR.SEQ LEVEL="1"><TITLE><TI>CHAPTER I</TI></TITLE>'
            + article(1, "Amendment text: " + quoted)
            + '<GR.SEQ LEVEL="2">' + article(2) + "</GR.SEQ></GR.SEQ>"
            + "<ANNEX>" + article(88) + "</ANNEX></ENACTING.TERMS></ACT>").encode()


def test_normalization_only_changes_operative_divisions_and_keeps_quoted_articles():
    original = formex()
    xml, expected = segment._profile_article_xml(original)
    root = ElementTree.fromstring(xml)
    assert expected == ("1", "2")
    assert root.find("ENACTING.TERMS/DIVISION/DIVISION/ARTICLE").get("IDENTIFIER") == "002"
    assert root.find("ENACTING.TERMS/DIVISION/ARTICLE/ALINEA/GR.SEQ/ARTICLE").get("IDENTIFIER") == "077"
    assert root.find("ENACTING.TERMS/TOC/ARTICLE").get("IDENTIFIER") == "099"
    assert root.find("ENACTING.TERMS/ANNEX/ARTICLE").get("IDENTIFIER") == "088"
    assert ElementTree.fromstring(original).find("ENACTING.TERMS/GR.SEQ") is not None


def test_existing_division_source_does_not_need_serialization():
    original = b"<ACT><ENACTING.TERMS><DIVISION>" + article(1).encode() + b"</DIVISION></ENACTING.TERMS></ACT>"
    normalized, expected = segment._profile_article_xml(original)
    assert normalized is original and expected == ("1",)


@pytest.mark.parametrize("articles", [
    [{"article_number": "text"}],
    [{"article_number": "1"}],
    [{"article_number": "1"}, {"article_number": "1"}],
    [{"article_number": "1"}, {"article_number": "2"}, {"article_number": "77"}],
])
def test_missing_duplicated_or_promoted_article_inventory_is_rejected(tmp_path, articles):
    def parse(xml, destination):
        return {"articles": articles}

    destination = tmp_path / "parsed.json"
    with pytest.raises(segment.FormexArticleError, match="article-inventory-mismatch"):
        segment._profile_parse(formex(), destination, parse)
    assert not destination.exists()


def test_unknown_operative_wrapper_is_counted_for_inventory_validation(tmp_path):
    xml = ("<ACT><ENACTING.TERMS><UNSUPPORTED>" + article(1)
           + "</UNSUPPORTED></ENACTING.TERMS></ACT>").encode()
    normalized, expected = segment._profile_article_xml(xml)
    assert normalized is xml and expected == ("1",)
    with pytest.raises(segment.FormexArticleError, match="article-inventory-mismatch"):
        segment._profile_parse(xml, tmp_path / "parsed.json", lambda *args: {"articles": []})


def test_profile_parse_checks_original_boundaries_and_reuses_verified_cache(tmp_path):
    calls = []

    def parse(xml, destination):
        calls.append(xml)
        root = ElementTree.fromstring(xml)
        assert root.find("ENACTING.TERMS/DIVISION/DIVISION") is not None
        return {"articles": [{"article_number": "1"}, {"article_number": "2"}]}

    destination = tmp_path / "parsed.json"
    first = segment._profile_parse(formex(), destination, parse)
    assert segment._profile_parse(formex(), destination, parse) == first
    assert len(calls) == 1
    marker = destination.with_suffix(".source.json")
    metadata = json.loads(marker.read_text())
    assert metadata["segmentation_sha256"] == segment._file_hash(segment.__file__)
    metadata["segmentation_sha256"] = "0" * 64
    marker.write_text(json.dumps(metadata))
    assert segment._profile_parse(formex(), destination, parse) == first
    assert len(calls) == 2


@pytest.mark.parametrize("xml, reason", [
    (("<ACT><ENACTING.TERMS>" + article(1) + article(1) + "</ENACTING.TERMS></ACT>").encode(),
     "duplicate-formex-article-number"),
    (b'<ACT><ENACTING.TERMS><ARTICLE IDENTIFIER="001"><TI.ART>Article 2</TI.ART></ARTICLE></ENACTING.TERMS></ACT>',
     "conflicting-formex-article-number"),
    (b'<ACT><ENACTING.TERMS><ARTICLE><TI.ART>Number absent</TI.ART></ARTICLE></ENACTING.TERMS></ACT>',
     "missing-formex-article-number"),
])
def test_invalid_source_article_identity_fails_closed(xml, reason):
    with pytest.raises(segment.FormexArticleError, match=reason):
        segment._profile_article_xml(xml)


def test_inventory_failure_is_recorded_without_publishing_partial_language(tmp_path, monkeypatch):
    profile = paths.StructurePaths(tmp_path)
    profile.ensure_dirs()
    monkeypatch.setattr(cellar, "read_documents", lambda celex, language, **kwargs:
                        cellar.ActDocuments(celex, language, formex(), []))
    rows, summary = segment.segment_act(
        "32021R1059", {"eli": "http://data.europa.eu/eli/reg/2021/1059/oj"}, {},
        profile=profile, parser=lambda *args: {"articles": [{"article_number": "text"}]},
    )
    assert not rows
    assert summary["articles"] == {}
    assert summary["errors"] == {lg: "formex-article-inventory-mismatch" for lg in ACT_LANGUAGES}
