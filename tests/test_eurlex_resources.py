"""Discovery and isolated official-source preparation without network/model calls."""

import io
import json
import lzma
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import pytest

from clir_bench.domains.legal.structure import ACT_LANGUAGES, cellar, segment
from clir_bench.domains.legal.structure import eurlex_resources as er
from clir_bench.domains.legal.structure.profile import ProfileArticleIndex, load_profile

NEW = "32019R0904"
OLD = "32015R0100"
NEW_ELI = "http://data.europa.eu/eli/reg_impl/2019/904/oj"
OLD_ELI = "http://data.europa.eu/eli/reg/2015/100/oj"


def discovery(celex=NEW, *, date="2019-06-01", text=None, title="A regulation"):
    return {
        "celex": celex,
        "date": date,
        "title": title,
        "text": text
        if text is not None
        else "Article 1\nOperators must report.\nArticle 2\nApplication.",
    }


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def baseline(directory, *, quarantined=False, annex=False):
    directory.mkdir()
    rows = []
    for language in ACT_LANGUAGES:
        parsed = {
            "articles": [
                {
                    "article_number": "1",
                    "article_title": "Existing duties",
                    "article_html": f"<p>{language} An existing duty applies.</p>",
                }
            ]
        }
        rows += segment.records_for(
            OLD,
            language,
            OLD_ELI,
            {"doc_type": "regulation", "year": 2015},
            parsed,
            [],
            None,
            act_title_text="Existing regulation",
        )
    if annex:
        rows += segment.records_for(
            OLD,
            "en",
            OLD_ELI,
            {"doc_type": "regulation", "year": 2015},
            {
                "articles": [],
                "annexes": [
                    {
                        "annex_id": "1",
                        "annex_title": "ANNEX I",
                        "annex_html": "<p>Existing technical specifications.</p>",
                    }
                ],
            },
            [],
            None,
        )
    # Deliberately preserve original formatting, Unicode, and CRLF byte layout.
    raw = b"".join(
        (json.dumps(row, ensure_ascii=False, indent=None) + "\r\n").encode() for row in rows
    )
    (directory / "articles.jsonl").write_bytes(raw)
    quarantines = (
        [{"celex_id": OLD, "reason": "inherited-manual-quarantine", "note": "keep me"}]
        if quarantined
        else []
    )
    (directory / "quarantine.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in quarantines)
    )
    return raw


def formex_zip(celex, language):
    xml = (
        f"<ACT><BIB.INSTANCE><LG.DOC>{language.upper()}</LG.DOC>"
        f"<NO.DOC><YEAR>{celex[1:5]}</YEAR><NO.CURRENT>{int(celex[6:])}</NO.CURRENT></NO.DOC>"
        "</BIB.INSTANCE><TITLE><TI><P>Official reporting regulation</P></TI></TITLE>"
        "<BODY><P>Authentic source text.</P></BODY></ACT>"
    )
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr(zipfile.ZipInfo("act.xml", date_time=(2020, 1, 1, 0, 0, 0)), xml)
    return result.getvalue()


def fake_transport(calls, failures=None):
    failures = failures if failures is not None else {}
    reverse = {value: key for key, value in cellar.LANG_CODE.items()}

    def fetch(url, accept, *, language=None):
        celex = url.rsplit("/", 1)[-1]
        calls.append((celex, language or "eli"))
        if language in failures:
            raise failures[language]
        if language is None:
            return f"<identifiers><uri>{NEW_ELI}</uri></identifiers>".encode()
        return formex_zip(celex, reverse[language])

    return fetch


def fake_parser(calls, *, omit=None, crash=None, annex_reference=False, empty=None):
    omit = omit if omit is not None else set()
    crash = crash if crash is not None else set()
    empty = empty if empty is not None else set()

    def parse(xml, out):
        language = ElementTree.fromstring(xml).findtext("BIB.INSTANCE/LG.DOC").lower()
        calls.append(language)
        if language in crash:
            raise RuntimeError("fixture interrupted")
        reference = (
            "Annex I of Regulation (EU) 2015/100"
            if annex_reference
            else "Article 1 of Regulation (EU) 2015/100"
        )
        first = (
            f"Operators shall submit complete annual reports. Article 2 applies. {reference} "
            "sets the existing requirements. " + " ".join(f"provision{i}" for i in range(100))
        )
        articles = [
            {
                "article_number": "1",
                "article_title": "Reporting duties",
                "article_html": ""
                if language in empty
                else f"<p>{first if language == 'en' else language + ' Authentic translated duties apply.'}</p>",
            }
        ]
        if language not in omit:
            articles.append(
                {
                    "article_number": "2",
                    "article_title": "Scope",
                    "article_html": f"<p>{language} This regulation applies to operators.</p>",
                }
            )
        return {"articles": articles}

    return parse


def prepare_fixture(
    tmp_path,
    *,
    fetch_failures=None,
    omit=None,
    crash=None,
    quarantined=False,
    annex=False,
    empty=None,
    out_name="prepared",
):
    inv = tmp_path / "inventory"
    er.inventory_records([discovery()], inv)
    old = tmp_path / "baseline"
    original = baseline(old, quarantined=quarantined, annex=annex)
    fetched, parsed = [], []
    fetch = fake_transport(fetched, fetch_failures)
    parser = fake_parser(parsed, omit=omit, crash=crash, annex_reference=annex, empty=empty)
    out = tmp_path / out_name
    return inv, old, original, fetched, parsed, fetch, parser, out


def test_inventory_filters_versions_and_reports_conflicts_without_guessing(tmp_path):
    rows = [
        discovery(),
        discovery(),
        discovery("32019L0001"),
        discovery("32019L0001", title="conflicting version"),
        discovery("02019R0904-20200101"),
        discovery("32019R0904R(01)"),
        discovery("32019D0001"),
        discovery("32016R0001"),
        discovery("32023R0001"),
        dict(discovery("32020R0001", date="2020-01-01"), language="fr"),
    ]
    manifest = er.inventory_records(rows, tmp_path / "inventory")
    assert manifest["counts"]["canonical_acts"] == 1
    assert manifest["counts"]["duplicate_records"] == 1
    assert manifest["counts"]["conflicting_acts"] == 1
    (kept,) = read_rows(tmp_path / "inventory/inventory.jsonl")
    assert kept["celex_id"] == NEW
    assert kept["text_sha256"] == er._text_hash(discovery()["text"])
    (conflict,) = read_rows(tmp_path / "inventory/conflicts.jsonl")
    assert conflict["celex_id"] == "32019L0001" and len(conflict["variants"]) == 2
    assert len(read_rows(tmp_path / "inventory/rejected.jsonl")) == 6
    assert manifest["source"]["date_field"] == "date"
    assert "not verified as OJ publication" in manifest["source"]["date_semantics"]
    assert er.audit(tmp_path / "inventory/manifest.json")["canonical_acts"] == 1


def test_stratified_30_is_deterministic_covers_year_and_type_and_preserves_prefix():
    records = [
        {
            "celex_id": f"3{year}{kind}{i:04d}",
            "year": year,
            "doc_type": "regulation" if kind == "R" else "directive",
            "size_bin": size,
        }
        for year in range(2017, 2023)
        for kind in ("R", "L")
        for i, size in enumerate(("tiny", "small", "medium", "large"), 1)
    ]
    selected = er.stratified_sample(records, 30, 42)
    assert len(selected) == 30 and len({row["celex_id"] for row in selected}) == 30
    assert {(row["year"], row["doc_type"]) for row in selected} == {
        (year, kind) for year in range(2017, 2023) for kind in ("regulation", "directive")
    }
    assert selected == er.stratified_sample(reversed(records), 30, 42)
    assert selected == er.stratified_sample(records, 31, 42)[:30]
    assert er.stratified_sample(records, 0, 42) == []


def test_inventory_metadata_is_deterministic_under_valid_input_reordering(tmp_path):
    rows = [discovery(), discovery("32020L0001", date="2020-02-02")]
    first = er.inventory_records(rows, tmp_path / "first")
    second = er.inventory_records(reversed(rows), tmp_path / "second")
    assert first == second
    assert (tmp_path / "first/inventory.jsonl").read_bytes() == (
        tmp_path / "second/inventory.jsonl"
    ).read_bytes()


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"date": "nonsense"}, "invalid-source-date"),
        ({"date": "2016-01-01"}, "source-date-outside-2017-2022"),
        ({"text": " \n "}, "empty-english-text"),
        ({"celex": "32019R00001"}, "not-original"),
    ],
)
def test_invalid_discovery_records_never_enter_candidates(tmp_path, changes, reason):
    er.inventory_records([dict(discovery(), **changes)], tmp_path / "inventory")
    assert not read_rows(tmp_path / "inventory/inventory.jsonl")
    assert reason in read_rows(tmp_path / "inventory/rejected.jsonl")[0]["reason"]


def test_inventory_download_resolves_revision_once_and_keeps_only_two_english_files(
    tmp_path, monkeypatch
):
    import huggingface_hub

    calls = []
    pinned = "a" * 40

    class API:
        def dataset_info(self, repository, *, revision):
            calls.append(("resolve", repository, revision))
            return type("Info", (), {"sha": pinned})()

    def download(*, repo_id, repo_type, revision, filename, local_dir):
        calls.append((repo_id, repo_type, revision, filename))
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with lzma.open(path, "wt") as out:
            out.write(
                json.dumps(discovery("32019L0001" if "directive" in filename else NEW)) + "\n"
            )
        return str(path)

    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    manifest = er.inventory(tmp_path / "inventory", revision="main")
    assert manifest["source"]["revision"] == pinned
    assert calls[0] == ("resolve", er.REPOSITORY, "main")
    assert [entry[-1] for entry in calls[1:]] == list(er.SOURCE_FILES)
    assert all(entry[2] == pinned for entry in calls[1:])
    assert all(
        (tmp_path / "inventory" / entry["path"]).is_file() for entry in manifest["source"]["inputs"]
    )
    assert manifest["counts"]["canonical_acts"] == 2


def test_prepare_preserves_baseline_bytes_and_recomputes_union_references(tmp_path):
    inv, old, original, fetched, parsed, fetch, parser, out = prepare_fixture(tmp_path)
    manifest = er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    assert (old / "articles.jsonl").read_bytes() == original
    assert (out / "articles.jsonl").read_bytes().startswith(original)
    assert json.loads((out / "target_acts.json").read_text()) == [NEW]
    assert manifest["counts"]["accepted_new_acts"] == 1
    assert manifest["counts"]["new_unit_rows"] == 8
    assert len(fetched) == 5 and Counter(parsed) == Counter(ACT_LANGUAGES)
    assert manifest["source"]["article_source"] == "official_CELLAR_Formex"
    assert manifest["source"]["baseline"]["articles_sha256"] == er._text_hash(original.decode())
    edges = read_rows(out / "external_edges.jsonl")
    assert len(edges) == 1 and edges[0]["source_celex"] == NEW and edges[0]["target_celex"] == OLD
    assert edges[0]["available_languages"] == list(ACT_LANGUAGES)
    index = ProfileArticleIndex(out / "manifest.json")
    assert index.by_eli[f"{OLD_ELI[:-3]}/art_1/oj"].celex_id == OLD
    payload = index.build(f"{NEW_ELI[:-3]}/art_1/oj", languages=ACT_LANGUAGES)
    assert payload and len(payload.references) == 1 and len(payload.external_references) == 1
    assert er.audit(out / "manifest.json")["accepted_new_acts"] == 1
    assert load_profile(out / "manifest.json").target_acts == {NEW}


def test_baseline_quarantine_is_carried_and_blocks_new_external_reference(tmp_path):
    inv, old, original, _, _, fetch, parser, out = prepare_fixture(tmp_path, quarantined=True)
    er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    (inherited,) = [row for row in read_rows(out / "quarantine.jsonl") if row["celex_id"] == OLD]
    assert inherited == {
        "celex_id": OLD,
        "reason": "inherited-manual-quarantine",
        "note": "keep me",
    }
    status = {
        row["article_number"]: row
        for row in read_rows(out / "reference_status.jsonl")
        if row["celex_id"] == NEW
    }
    assert not status["1"]["complete"]
    assert status["1"]["unresolved_reasons"]["target_act_quarantined"] == 1
    assert (old / "articles.jsonl").read_bytes() == original


def test_unit_conflict_scan_ignores_identical_duplicates_and_known_quarantines(tmp_path):
    old = tmp_path / "old"
    baseline(old)
    rows = read_rows(old / "articles.jsonl")
    extra = dict(rows[0], text=rows[0]["text"] + " A conflicting version.")
    path = tmp_path / "union.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows + [rows[0]]))
    assert er.find_unit_conflicts(path) == []
    with path.open("a") as out:
        out.write(json.dumps(extra) + "\n")
    conflicts = er.find_unit_conflicts(path)
    assert len(conflicts) == 1 and conflicts[0]["celex_id"] == OLD
    assert conflicts[0]["conflicts"][0]["kind"] == "source-language-record-conflict"
    assert er.find_unit_conflicts(path, {OLD}) == []
    reversed_path = tmp_path / "reversed.jsonl"
    reversed_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows + [rows[0], extra]))
    )
    assert er.find_unit_conflicts(reversed_path) == conflicts


def test_unit_identity_collision_quarantines_both_acts_even_across_languages(tmp_path):
    old = tmp_path / "old"
    baseline(old)
    original = read_rows(old / "articles.jsonl")[0]
    conflicting = dict(
        original,
        celex_id="32016R0001",
        language="fr",
        unit_type="annex",
        article_number="",
        annex_id="1",
    )
    union = tmp_path / "union.jsonl"
    union.write_text(json.dumps(original) + "\n" + json.dumps(conflicting) + "\n")
    conflicts = er.find_unit_conflicts(union)
    assert {row["celex_id"] for row in conflicts} == {OLD, "32016R0001"}
    assert all(row["conflicts"][0]["kind"] == "unit-identity-conflict" for row in conflicts)


@pytest.mark.parametrize("known", [False, True])
def test_union_conflicts_are_quarantined_before_references_without_altering_raw_baseline(
    tmp_path, known
):
    inv, old, original, _, _, fetch, parser, out = prepare_fixture(tmp_path, quarantined=known)
    conflict = read_rows(old / "articles.jsonl")[0]
    conflict["text"] += " A different source version."
    original += (json.dumps(conflict) + "\n").encode()
    (old / "articles.jsonl").write_bytes(original)
    manifest = er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    assert (out / "articles.jsonl").read_bytes().startswith(original)
    assert (old / "articles.jsonl").read_bytes() == original
    assert manifest["counts"]["identity_conflict_acts"] == (0 if known else 1)
    conflicts = read_rows(out / "unit_conflicts.jsonl")
    assert len(conflicts) == (0 if known else 1)
    assert OLD in {row["celex_id"] for row in read_rows(out / "quarantine.jsonl")}
    status = next(
        row
        for row in read_rows(out / "reference_status.jsonl")
        if row["celex_id"] == NEW and row["article_number"] == "1"
    )
    assert not status["complete"]
    assert status["unresolved_reasons"]["target_act_quarantined"] == 1
    index = ProfileArticleIndex(out / "manifest.json")
    assert OLD not in index.by_act
    assert er.audit(out / "manifest.json")["accepted_new_acts"] == 1


def test_final_audit_rejects_unquarantined_conflicts_after_checksum_refresh(tmp_path):
    inv, old, _, _, _, fetch, parser, out = prepare_fixture(tmp_path)
    conflict = read_rows(old / "articles.jsonl")[0]
    conflict["heading"] = "Conflicting metadata"
    with (old / "articles.jsonl").open("a") as source:
        source.write(json.dumps(conflict) + "\n")
    er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    (out / "quarantine.jsonl").write_text("")
    rehash(out, "quarantine.jsonl")
    with pytest.raises(er.ValidationError, match="Unquarantined conflicting"):
        er.audit(out / "manifest.json")


def test_audit_detects_removed_inherited_quarantine_after_checksum_refresh(tmp_path):
    inv, old, _, _, _, fetch, parser, out = prepare_fixture(tmp_path, quarantined=True)
    er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    (out / "quarantine.jsonl").write_text("")
    rehash(out, "quarantine.jsonl")
    with pytest.raises(er.ValidationError, match="not carried"):
        er.audit(out / "manifest.json")


def test_acts_already_in_baseline_remain_reference_only_without_refetch(tmp_path):
    inv, old, _, fetched, _, fetch, parser, out = prepare_fixture(tmp_path)
    content = (old / "articles.jsonl").read_text().replace(OLD, NEW).replace(OLD_ELI, NEW_ELI)
    (old / "articles.jsonl").write_text(content)
    manifest = er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    assert manifest["counts"]["selected_new_acts"] == 0
    assert json.loads((out / "target_acts.json").read_text()) == []
    assert fetched == []


@pytest.mark.parametrize("failure", ["missing-language", "article-set", "empty-text"])
def test_incomplete_new_acts_are_quarantined_without_losing_baseline(tmp_path, failure):
    kwargs = (
        {"fetch_failures": {"fra": cellar.FetchError("404 no French Formex")}}
        if failure == "missing-language"
        else ({"omit": {"fr"}} if failure == "article-set" else {"empty": {"fr"}})
    )
    inv, old, original, _, _, fetch, parser, out = prepare_fixture(tmp_path, **kwargs)
    manifest = er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    assert manifest["counts"]["accepted_new_acts"] == 0
    assert json.loads((out / "target_acts.json").read_text()) == []
    assert NEW in {row["celex_id"] for row in read_rows(out / "quarantine.jsonl")}
    assert (out / "articles.jsonl").read_bytes().startswith(original)


def test_legacy_english_completeness_is_preserved_while_translation_gap_is_reported(tmp_path):
    inv, old, _, _, _, fetch, parser, out = prepare_fixture(tmp_path, annex=True)
    manifest = er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    status = next(
        row
        for row in read_rows(out / "reference_status.jsonl")
        if row["celex_id"] == NEW and row["article_number"] == "1"
    )
    assert status["complete"]
    assert manifest["counts"]["language_incomplete_articles"] == 1
    with pytest.raises(ValueError, match="(?i)(language|version|translation|reference)"):
        ProfileArticleIndex(out / "manifest.json").build(
            f"{NEW_ELI[:-3]}/art_1/oj", languages=ACT_LANGUAGES
        )


def test_interrupted_prepare_resumes_raw_and_successful_parse_caches_atomically(tmp_path):
    crash = {"fr"}
    inv, old, original, fetched, parsed, fetch, parser, out = prepare_fixture(tmp_path, crash=crash)
    with pytest.raises(RuntimeError, match="interrupted"):
        er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    assert not out.exists()
    assert out.with_name(".prepared.work").is_dir()
    assert len(fetched) == 5
    before = len(parsed)
    crash.clear()
    er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    assert len(fetched) == 5
    assert parsed[before:] == ["fr", "de", "es"]
    assert (old / "articles.jsonl").read_bytes() == original
    assert er.audit(out / "manifest.json")["accepted_new_acts"] == 1


def test_resume_rejects_changed_baseline_or_configuration(tmp_path):
    crash = {"fr"}
    inv, old, _, _, _, fetch, parser, out = prepare_fixture(tmp_path, crash=crash)
    with pytest.raises(RuntimeError):
        er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    with (old / "articles.jsonl").open("ab") as handle:
        handle.write(b"\n")
    crash.clear()
    with pytest.raises(er.ValidationError, match="different inputs"):
        er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    assert not out.exists()


def test_repeat_preparation_has_identical_content_and_distinct_truthful_retrieval_records(tmp_path):
    inv, old, _, _, _, fetch, parser, first = prepare_fixture(tmp_path)
    one = er.prepare(inv, old, first, transport=fetch, parser=parser, workers=1)
    two = er.prepare(inv, old, tmp_path / "second", transport=fetch, parser=parser, workers=1)
    assert {
        name: digest for name, digest in one["files"].items() if name != "official_sources.json"
    } == {name: digest for name, digest in two["files"].items() if name != "official_sources.json"}
    assert one["source"] == two["source"]
    original = json.loads((first / "official_sources.json").read_text())
    later = json.loads((tmp_path / "second/official_sources.json").read_text())
    for before, after in zip(original[0]["files"], later[0]["files"]):
        assert before["sha256"] == after["sha256"]
        assert before["request"] == after["request"]
        assert before["retrieved_at"] and after["retrieved_at"]


def test_official_provenance_binds_retrieval_metadata_requests_and_raw_bytes(tmp_path):
    inv, old, _, _, _, fetch, parser, out = prepare_fixture(tmp_path)
    manifest = er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    (act,) = json.loads((out / "official_sources.json").read_text())
    assert act["celex_id"] == NEW and len(act["files"]) == 5
    for entry in act["files"]:
        assert entry["cache_metadata"]["sha256"] == entry["sha256"]
        assert entry["retrieved_at"] == entry["cache_metadata"]["retrieved_at"]
        assert entry["retrieved_at"] and len(entry["cache_metadata_sha256"]) == 64
        assert entry["request"]["url"] == f"https://publications.europa.eu/resource/celex/{NEW}"
        if entry["name"].endswith(".zip"):
            assert entry["request"]["accept_language"] in cellar.LANG_CODE.values()
            assert entry["request"]["accept"] == "application/zip;mtype=fmx4"
    assert manifest["source"]["parser"]["kind"] == "injected_python_parser"
    assert manifest["source"]["parser"]["python_version"]
    assert len(manifest["source"]["preparation_code"][er.__name__]) == 64


def test_default_parser_provenance_uses_pinned_helper_fields(monkeypatch, tmp_path):
    expected = {
        "entrypoint_sha256": "a" * 64,
        "node_version": "v22.0.0",
        "git_head": "b" * 40,
        "git_dirty": True,
        "tracked_backend_diff_sha256": "c" * 64,
        "package_lock_sha256": "d" * 64,
    }
    monkeypatch.setattr(segment, "parser_provenance", lambda: expected)
    monkeypatch.setattr(segment, "parse_fmx_path", lambda: tmp_path / "parse-fmx.js")
    assert er._parser_provenance() == {
        "kind": "parse_fmx",
        "entrypoint_name": "parse-fmx.js",
        **expected,
    }


@pytest.mark.parametrize("change", ["raw", "metadata"])
def test_audit_checks_original_cache_when_available_and_bundle_remains_portable(tmp_path, change):
    inv, old, _, _, _, fetch, parser, out = prepare_fixture(tmp_path)
    er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    portable = tmp_path / "portable" / "bundle"
    shutil.copytree(out, portable)
    cache = out.with_name(".prepared.work") / "new" / "cache"
    raw = cache / "2019" / f"{NEW}.en.zip"
    if change == "raw":
        raw.write_bytes(raw.read_bytes() + b"changed")
    else:
        metadata = raw.with_suffix(".zip.meta.json")
        value = json.loads(metadata.read_text())
        value["retrieved_at"] = "2000-01-01T00:00:00+00:00"
        metadata.write_text(json.dumps(value))
    with pytest.raises(er.ValidationError, match="cache"):
        er.audit(out / "manifest.json")
    assert er.audit(portable / "manifest.json")["accepted_new_acts"] == 1
    with pytest.raises(er.ValidationError, match="cache"):
        er.audit(portable / "manifest.json", cache_dir=cache)


def rehash(bundle, name):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][name] = er.sha256_file(bundle / name)
    path.write_text(json.dumps(manifest))


def test_audit_independently_checks_preserved_baseline_even_after_rehash(tmp_path):
    inv, old, _, _, _, fetch, parser, out = prepare_fixture(tmp_path)
    er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    content = (out / "articles.jsonl").read_text().replace("An existing duty", "A changed old duty")
    (out / "articles.jsonl").write_text(content)
    rehash(out, "articles.jsonl")
    with pytest.raises(er.ValidationError, match="Baseline article bytes"):
        er.audit(out / "manifest.json")


def test_audit_rejects_foreign_article_identity_drift_after_checksum_refresh(tmp_path):
    inv, old, _, _, _, fetch, parser, out = prepare_fixture(tmp_path)
    er.prepare(inv, old, out, transport=fetch, parser=parser, workers=1)
    rows = read_rows(out / "articles.jsonl")
    for row in rows:
        if row["celex_id"] == NEW and row["language"] == "fr":
            row["act_eli"] = row["act_eli"].replace("/reg_impl/", "/reg/")
            row["eli_id"] = row["eli_id"].replace("/reg_impl/", "/reg/")
    original = (old / "articles.jsonl").read_bytes()
    additions = [row for row in rows if row["celex_id"] == NEW]
    (out / "articles.jsonl").write_bytes(
        original + b"".join((json.dumps(row) + "\n").encode() for row in additions)
    )
    rehash(out, "articles.jsonl")
    with pytest.raises(er.ValidationError, match="four-language article identities"):
        er.audit(out / "manifest.json")


def test_existing_output_and_unsafe_work_paths_fail_without_writing_inputs(tmp_path):
    inv, old, original, fetched, _, fetch, parser, out = prepare_fixture(tmp_path)
    out.mkdir()
    with pytest.raises(FileExistsError):
        er.prepare(inv, old, out, transport=fetch, parser=parser)
    with pytest.raises(ValueError, match="separate"):
        er.prepare(
            inv, old, tmp_path / "other", work_dir=old / "work", transport=fetch, parser=parser
        )
    assert fetched == [] and (old / "articles.jsonl").read_bytes() == original


def test_cli_audit_and_invalid_revision_are_explicit(tmp_path, capsys):
    er.inventory_records([discovery()], tmp_path / "inventory")
    er.main(["audit", str(tmp_path / "inventory/manifest.json")])
    assert json.loads(capsys.readouterr().out)["canonical_acts"] == 1
    with pytest.raises(ValueError, match="immutable"):
        er.inventory_records([discovery()], tmp_path / "bad", revision="main")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad.building-*"))
