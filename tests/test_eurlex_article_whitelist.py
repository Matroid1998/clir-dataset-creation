"""An article budget restricts QA targets without hiding their references."""

import hashlib
import json

import pytest

from clir_bench.domains.legal.qac import eurlex_batch
from clir_bench.domains.legal.structure import ACT_LANGUAGES
from clir_bench.domains.legal.structure.profile import ProfileArticleIndex, load_profile


ACT = "32019R0001"
FIRST = "http://data.europa.eu/eli/reg/2019/1/art_1/oj"
SECOND = "http://data.europa.eu/eli/reg/2019/1/art_2/oj"


def freeze(directory, *, targets=None, budget=None, listed=True):
    manifest = {"format": "eurlex-resources-v1", "version": 1, "config": {}}
    names = ["articles.jsonl", "internal_edges.jsonl", "external_edges.jsonl",
             "reference_status.jsonl", "quarantine.jsonl", "target_acts.json"]
    if targets is not None:
        (directory / "target_articles.json").write_text(json.dumps(targets))
        if listed:
            names.append("target_articles.json")
    if budget is not None:
        manifest["config"]["target_articles"] = budget
    manifest["files"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in names}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.fixture
def bundle(tmp_path):
    def write(name, rows):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    write("articles.jsonl", [
        {"eli_id": eli, "celex_id": ACT, "article_number": number,
         "unit_type": "article", "language": language, "heading": "Reporting",
         "text": f"{language} Authorities shall publish complete annual reports. " * 20}
        for eli, number in [(FIRST, "1"), (SECOND, "2")]
        for language in ACT_LANGUAGES
    ])
    write("internal_edges.jsonl", [{"source_article_id": SECOND, "target_article_id": FIRST,
                                    "source_unit_type": "article", "char_start": 1}])
    write("external_edges.jsonl", [])
    write("reference_status.jsonl", [{"eli_id": eli, "complete": True} for eli in [FIRST, SECOND]])
    write("quarantine.jsonl", [])
    (tmp_path / "target_acts.json").write_text(json.dumps([ACT]))
    return tmp_path


def test_article_budget_restricts_selection_and_replay_but_retains_reference(bundle):
    path = freeze(bundle, targets=[SECOND], budget=1)
    index = ProfileArticleIndex(path)
    assert index.profile.target_articles == {SECOND}
    assert not index.is_target_eligible(FIRST)
    with pytest.raises(ValueError, match="outside the prepared article budget"):
        index.build(FIRST)
    payload = index.build(SECOND, languages=ACT_LANGUAGES)
    assert payload.target.eli_id == SECOND
    assert [unit.eli_id for unit in payload.references] == [FIRST]
    assert all(payload.references[0].texts[language] for language in ACT_LANGUAGES)
    targets = eurlex_batch.select(index, n=1, seed=42, languages=["en"], modes=["lookup"],
                                 strata=(("one_ref", 1, 1, 1.0),))
    assert [target.eli_id for target in targets] == [SECOND]


def test_legacy_profile_has_no_article_budget(bundle):
    index = ProfileArticleIndex(freeze(bundle))
    assert index.profile.target_articles is None
    assert index.is_target_eligible(FIRST) and index.is_target_eligible(SECOND)


@pytest.mark.parametrize("targets,budget,listed", [
    ([FIRST, FIRST], 2, True),
    ([SECOND, FIRST], 2, True),
    ([FIRST], 2, True),
    ([FIRST], True, True),
    ([FIRST], None, True),
    ([], 0, True),
    ([3], 1, True),
    (["http://data.europa.eu/eli/reg/2019/1/anx_1/oj"], 1, True),
    ([FIRST], 1, False),
    (None, 1, False),
])
def test_malformed_or_unpinned_article_budgets_are_rejected(bundle, targets, budget, listed):
    path = freeze(bundle, targets=targets, budget=budget, listed=listed)
    with pytest.raises(ValueError, match="target article|article budget"):
        load_profile(path)


def test_whitelist_cannot_name_absent_article_in_same_act(bundle):
    missing = "http://data.europa.eu/eli/reg/2019/1/art_3/oj"
    path = freeze(bundle, targets=[missing], budget=1)
    with pytest.raises(ValueError, match="absent or ineligible unit"):
        ProfileArticleIndex(path)


def test_changing_allowlist_invalidates_frozen_identity(bundle):
    path = freeze(bundle, targets=[SECOND], budget=1)
    (bundle / "target_articles.json").write_text(json.dumps([FIRST]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_profile(path)
