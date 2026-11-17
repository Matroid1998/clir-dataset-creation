"""Adaptive article budgets count validated multilingual targets, not raw headings."""

import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from clir_bench.domains.legal.structure import ACT_LANGUAGES, cellar
from clir_bench.domains.legal.structure import eurlex_resources as er
from clir_bench.domains.legal.structure.profile import ProfileArticleIndex

ACTS = ["32019R0904", "32020R0905", "32021L0906"]


def make_inputs(tmp_path, n=3):
    inventory = tmp_path / "inventory"
    er.inventory_records([
        {"celex": act, "date": f"{act[1:5]}-06-01", "title": "Reporting duties",
         "text": "Article 1\nScope.\nArticle 2\nDuties."}
        for act in ACTS[:n]
    ], inventory)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "articles.jsonl").write_text("")
    (baseline / "quarantine.jsonl").write_text("")
    return inventory, baseline


def transport(calls):
    reverse = {value: key for key, value in cellar.LANG_CODE.items()}

    def fetch(url, accept, *, language=None):
        celex = url.rsplit("/", 1)[-1]
        calls.append((celex, language))
        kind = "reg" if celex[5] == "R" else "dir"
        if language is None:
            return (f"<identifiers><uri>http://data.europa.eu/eli/{kind}/"
                    f"{celex[1:5]}/{int(celex[6:])}/oj</uri></identifiers>").encode()
        xml = (f"<ACT><BIB.INSTANCE><LG.DOC>{reverse[language]}</LG.DOC><NO.DOC>"
               f"<YEAR>{celex[1:5]}</YEAR><NO.CURRENT>{int(celex[6:])}</NO.CURRENT>"
               "</NO.DOC></BIB.INSTANCE><TITLE><TI><P>Reporting regulation</P></TI></TITLE>"
               "<BODY><ARTICLE><TI.ART>Article 1</TI.ART><PARAG><P>Scope.</P></PARAG></ARTICLE>"
               "<ARTICLE><TI.ART>Article 2</TI.ART><PARAG><P>Duties.</P></PARAG></ARTICLE>"
               "</BODY></ACT>")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("act.xml", xml)
        return buffer.getvalue()

    return fetch


def parser(calls):
    def parse(xml, out):
        root = ET.fromstring(xml)
        year = root.findtext("BIB.INSTANCE/NO.DOC/YEAR")
        language = root.findtext("BIB.INSTANCE/LG.DOC")
        calls.append((year, language))
        long = "Operators shall submit an annual report describing all relevant activities. " * 15
        return {"articles": [
            {"article_number": "1", "article_title": "Scope",
             "article_html": f"<p>{'Scope only.' if year == '2019' else long}</p>"},
            {"article_number": "2", "article_title": "Reporting",
             "article_html": f"<p>{long}</p>"},
        ]}

    return parse


def test_budget_orders_initial_year_type_coverage_then_article_rich_acts():
    rows = [
        {"celex_id": "32019R0001", "year": 2019, "doc_type": "regulation", "proxy_articles": 10, "chars_en": 100},
        {"celex_id": "32019R0002", "year": 2019, "doc_type": "regulation", "proxy_articles": 20, "chars_en": 100},
        {"celex_id": "32020L0001", "year": 2020, "doc_type": "directive", "proxy_articles": 2, "chars_en": 100},
    ]
    ordered = er.article_budget_order(rows)
    assert [row["celex_id"] for row in ordered] == ["32019R0002", "32020L0001", "32019R0001"]
    assert ordered == er.article_budget_order(reversed(rows))


def test_budget_caps_eligible_articles_and_retains_full_source_acts(tmp_path):
    inventory, baseline = make_inputs(tmp_path)
    fetched, parsed = [], []
    out = tmp_path / "prepared"
    manifest = er.prepare(inventory, baseline, out, target_articles=2, batch_size=1,
                          workers=1, transport=transport(fetched), parser=parser(parsed))
    assert manifest["counts"]["selected_new_acts"] == 2
    assert manifest["counts"]["target_articles"] == 2
    assert manifest["counts"]["eligible_articles_before_cap"] == 3
    assert {act for act, _ in fetched} == set(ACTS[:2])
    assert len(parsed) == 8
    targets = json.loads((out / "target_articles.json").read_text())
    assert len(targets) == 2 and targets == sorted(set(targets))
    assert "target_articles.json" in manifest["files"]
    index = ProfileArticleIndex(out / "manifest.json")
    assert len(index.by_eli) == 4
    assert all(index.is_target_eligible(identifier) for identifier in targets)
    assert sum(index.is_target_eligible(identifier) for identifier in index.by_eli) == 2
    assert manifest["counts"]["new_unit_rows"] == 16
    assert er.audit(out / "manifest.json")["target_articles"] == 2


def test_exhausted_budget_never_publishes_partial_and_resume_reuses_successes(tmp_path, monkeypatch):
    inventory, baseline = make_inputs(tmp_path, n=2)
    fetched, parsed = [], []
    fetch, parse = transport(fetched), parser(parsed)
    out = tmp_path / "prepared"
    kwargs = dict(target_articles=4, batch_size=1, workers=1, transport=fetch, parser=parse)
    with pytest.raises(er.ValidationError, match="Only 3 eligible articles"):
        er.prepare(inventory, baseline, out, **kwargs)
    assert not out.exists()
    first_fetches, first_parses = len(fetched), len(parsed)
    with pytest.raises(er.ValidationError, match="Only 3 eligible articles"):
        er.prepare(inventory, baseline, out, **kwargs)
    assert len(fetched) == first_fetches and len(parsed) == first_parses
    with pytest.raises(er.ValidationError, match="different inputs/configuration"):
        er.prepare(inventory, baseline, out, **dict(kwargs, target_articles=3))
    original_provenance = er._parser_provenance
    monkeypatch.setattr(er, "_parser_provenance", lambda injected: dict(
        original_provenance(injected), tool_revision="changed-external-parser"))
    with pytest.raises(er.ValidationError, match="different inputs/configuration"):
        er.prepare(inventory, baseline, out, **kwargs)


def test_excluded_profile_acts_supply_no_additional_question_targets(tmp_path):
    inventory, baseline = make_inputs(tmp_path)
    first = tmp_path / "first"
    er.prepare(inventory, baseline, first, target_articles=1, batch_size=1,
               transport=transport([]), parser=parser([]), workers=1)
    second = tmp_path / "second"
    manifest = er.prepare(inventory, baseline, second, target_articles=2, batch_size=1,
                          exclude_profiles=[first / "manifest.json"],
                          cache_from=[tmp_path / ".first.work/new"],
                          transport=transport([]), parser=parser([]), workers=1)
    assert json.loads((second / "target_acts.json").read_text()) == [ACTS[1]]
    assert manifest["source"]["excluded_profiles"][0]["manifest_path"] == str(first / "manifest.json")


def test_cache_seed_validates_raw_identity_and_metadata_before_reuse(tmp_path):
    source, destination = tmp_path / "seed", tmp_path / "new"
    cache = source / "cache" / "2019"
    cache.mkdir(parents=True)
    calls = []
    raw = cache / f"{ACTS[0]}.en.zip"
    raw.write_bytes(transport(calls)(f"https://example/{ACTS[0]}", "zip", language="eng"))
    metadata = raw.with_suffix(".zip.meta.json")
    metadata.write_text(json.dumps({"sha256": er.sha256_file(raw), "retrieved_at": None}))
    er._seed_official_cache(source, destination, {ACTS[0]})
    assert (destination / "2019" / raw.name).read_bytes() == raw.read_bytes()
    raw.write_bytes(b"tampered")
    with pytest.raises(er.ValidationError, match="checksum mismatch"):
        er._seed_official_cache(source, tmp_path / "other", {ACTS[0]})


@pytest.mark.parametrize("budget,batch", [(0, 1), (-1, 1), (True, 1), (1, 0)])
def test_budget_requires_positive_integer_limits(tmp_path, budget, batch):
    with pytest.raises(ValueError, match="positive integer"):
        er.prepare(tmp_path / "unused", tmp_path / "unused", tmp_path / "out",
                   target_articles=budget, batch_size=batch)
