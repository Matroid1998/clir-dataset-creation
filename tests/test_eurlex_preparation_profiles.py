"""Isolated official-source preparation remains resumable and fail closed."""

import io
import json
import subprocess
import urllib.error
import zipfile
from dataclasses import FrozenInstanceError
from xml.etree import ElementTree

import pytest

from clir_bench.domains.legal.structure import (
    ACT_LANGUAGES,
    audit,
    cellar,
    paths,
    project,
    references,
    segment,
)
from clir_bench.domains.legal.structure import resolve_external as resolver

CELEX = "32019R0904"
ELI = "http://data.europa.eu/eli/reg_impl/2019/904/oj"


def formex_zip(language="en", *, celex=CELEX, annex_language=None):
    bibliography = (f"<BIB.INSTANCE><LG.DOC>{language.upper()}</LG.DOC>"
                    f"<NO.DOC><YEAR>{celex[1:5]}</YEAR>"
                    f"<NO.CURRENT>{int(celex[6:])}</NO.CURRENT></NO.DOC></BIB.INSTANCE>")
    text = (f"<ACT>{bibliography}<TITLE><TI><P>Example regulation</P></TI></TITLE>"
            "<BODY><P>Officials shall submit annual reports.</P></BODY></ACT>")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("act.xml", text)
        if annex_language:
            archive.writestr("annex.xml", f"<ANNEX><BIB.INSTANCE><LG.DOC>{annex_language.upper()}"
                              "</LG.DOC></BIB.INSTANCE><TI>ANNEX I</TI><P>Body</P></ANNEX>")
    return output.getvalue()


def transport(calls, failures=None):
    failures = failures if failures is not None else {}
    reverse = {value: key for key, value in cellar.LANG_CODE.items()}

    def fetch(url, accept, *, language=None):
        calls.append(language or "eli")
        if language in failures:
            raise failures[language]
        if language is None:
            return f"<identifiers><uri>{ELI}</uri></identifiers>".encode()
        return formex_zip(reverse[language])

    return fetch


def parser(calls, failures=None):
    failures = failures if failures is not None else set()

    def parse(xml, out):
        language = ElementTree.fromstring(xml).findtext("BIB.INSTANCE/LG.DOC").lower()
        calls.append(language)
        if language in failures:
            return None
        return {"articles": [{"article_number": "1", "article_title": "Reporting duties",
                               "article_html": f"<p>{language} Officials must submit annual reports.</p>"}]}

    return parse


def rows(profile):
    return [json.loads(line) for line in profile.ARTICLES_JSONL.read_text().splitlines()]


def test_profile_paths_are_immutable_and_leave_default_paths_untouched(tmp_path):
    original = paths.ARTICLES_JSONL
    profile = paths.StructurePaths(tmp_path / "isolated")
    profile.ensure_dirs()
    assert profile.ROOT == paths.ROOT
    assert profile.ARTICLES_JSONL == tmp_path / "isolated/articles.jsonl"
    assert profile.REPORTS_DIR == tmp_path / "isolated/reports"
    assert paths.ARTICLES_JSONL == original
    with pytest.raises(FrozenInstanceError):
        profile.structure_dir = tmp_path / "other"


def test_partial_fetch_manifest_and_transient_miss_are_retried(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    profile.ensure_dirs()
    calls, failures = [], {"fra": urllib.error.URLError("temporary DNS failure")}
    fetch = transport(calls, failures)
    first = cellar.fetch_all([CELEX], profile=profile, transport=fetch, workers=1)
    assert first[CELEX]["languages"]["fr"]["zip"] is None
    miss = cellar.cache_file(CELEX, "fr.miss", profile=profile)
    assert not json.loads(miss.read_text())["permanent"]
    failures.clear()
    before = len(calls)
    second = cellar.fetch_all([CELEX], profile=profile, transport=fetch, workers=1)
    assert second[CELEX]["eli"] == ELI
    assert all(item["zip"] for item in second[CELEX]["languages"].values())
    assert calls[before:] == ["fra"]
    assert not miss.exists()
    before = len(calls)
    cellar.fetch_all([CELEX], profile=profile, transport=fetch, workers=1)
    assert len(calls) == before


def test_old_text_dns_miss_does_not_become_permanent(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    miss = cellar.cache_file(CELEX, "identifiers.miss", profile=profile)
    miss.write_text("Temporary failure in name resolution")
    calls = []
    assert cellar.eli_for(CELEX, profile=profile, transport=transport(calls)) == ELI
    assert calls == ["eli"] and not miss.exists()


def test_actual_404_is_cached_without_repeated_downloads(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    calls = []
    fetch = transport(calls, {"fra": cellar.FetchError("404 official Formex absent")})
    cellar.fetch_all([CELEX], profile=profile, transport=fetch, workers=1)
    count = len(calls)
    cellar.fetch_all([CELEX], profile=profile, transport=fetch, workers=1)
    assert len(calls) == count
    assert json.loads(cellar.cache_file(CELEX, "fr.miss", profile=profile).read_text())["permanent"]


@pytest.mark.parametrize(("payload", "reason"), [
    (b"not a zip", "invalid-formex-archive"),
    (formex_zip("fr"), "language-mismatch"),
    (formex_zip("en", celex="32019R0999"), "act-identity-mismatch"),
    (formex_zip("en", annex_language="fr"), "language-mismatch"),
])
def test_invalid_formex_is_not_installed_as_a_successful_cache(tmp_path, payload, reason):
    profile = paths.StructurePaths(tmp_path)
    result, error = cellar.ensure_zip(CELEX, "en", profile=profile,
                                     transport=lambda *args, **kwargs: payload)
    assert result is None and reason in error
    assert not cellar.cache_file(CELEX, "en.zip", profile=profile).exists()
    assert not json.loads(cellar.cache_file(CELEX, "en.miss", profile=profile).read_text())["permanent"]


def test_wrong_or_ambiguous_authoritative_eli_is_not_guessed(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    raw = (b"<notice><uri>http://data.europa.eu/eli/reg/2019/904/oj</uri>"
           b"<uri>http://data.europa.eu/eli/reg_impl/2019/904/oj</uri></notice>")
    assert cellar.eli_for(CELEX, profile=profile, transport=lambda *args, **kwargs: raw) is None
    assert not cellar.cache_file(CELEX, "identifiers.xml", profile=profile).exists()


def test_absent_formex_number_is_reported_as_missing_proof_not_wrong_identity(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("act.xml", "<ACT><BIB.INSTANCE><LG.DOC>EN</LG.DOC>"
                          "<DATE ISO='20181023'>20181023</DATE></BIB.INSTANCE>"
                          "<TITLE><TI>Regulation (EU) 2018/1725</TI></TITLE></ACT>")
        archive.writestr("act.doc.xml", "<DOC><BIB.DOC><PROD.ID>20181018004</PROD.ID>"
                          "<FIN.ID>217615</FIN.ID></BIB.DOC></DOC>")
    result, error = cellar.ensure_zip("32018R1725", "en", profile=profile,
                                     transport=lambda *args, **kwargs: output.getvalue())
    assert result is None and error == "missing-formex-act-identity"
    assert not cellar.cache_file("32018R1725", "en.zip", profile=profile).exists()


def test_cached_raw_mutation_is_refused(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    result, error = cellar.ensure_zip(CELEX, "en", profile=profile, transport=transport([]))
    assert not error
    result.write_bytes(formex_zip("en", celex="32019R0999"))
    _, error = cellar.ensure_zip(CELEX, "en", profile=profile,
                                 transport=lambda *args, **kwargs: pytest.fail("mutated cache redownloaded"))
    assert "checksum-mismatch" in error


def test_incomplete_aggregate_is_rebuilt_from_atomic_per_act_records(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    fetch = transport([])
    cellar.fetch_all([CELEX], profile=profile, transport=fetch, workers=1)
    profile.ARTICLES_JSONL.write_text(json.dumps({"celex_id": CELEX, "text": "partial"}) + "\n")
    calls = []
    parse = parser(calls)
    segment.segment([CELEX], profile=profile, transport=fetch, parser=parse, workers=1)
    assert set(calls) == set(ACT_LANGUAGES)
    assert {row["language"] for row in rows(profile)} == set(ACT_LANGUAGES)
    before = profile.ARTICLES_JSONL.read_bytes()
    calls.clear()
    segment.segment([CELEX], profile=profile, transport=fetch, parser=parse, workers=1)
    assert not calls and profile.ARTICLES_JSONL.read_bytes() == before


def test_partial_language_parse_is_revisited_without_reparsing_good_languages(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    fetch = transport([])
    cellar.fetch_all([CELEX], profile=profile, transport=fetch, workers=1)
    calls, failures = [], {"fr"}
    parse = parser(calls, failures)
    segment.segment([CELEX], profile=profile, transport=fetch, parser=parse, workers=1)
    assert {row["language"] for row in rows(profile)} == set(ACT_LANGUAGES) - {"fr"}
    state = profile.PARSED_DIR / "acts/2019" / f"{CELEX}.json"
    assert not json.loads(state.read_text())["complete"]
    failures.clear()
    calls.clear()
    segment.segment([CELEX], profile=profile, transport=fetch, parser=parse, workers=1)
    assert calls == ["fr"]
    assert len(rows(profile)) == 4 and json.loads(state.read_text())["complete"]


def test_corrupt_per_act_checkpoint_is_rebuilt_without_appending_duplicates(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    fetch = transport([])
    cellar.fetch_all([CELEX], profile=profile, transport=fetch, workers=1)
    parse = parser([])
    segment.segment([CELEX], profile=profile, transport=fetch, parser=parse, workers=1)
    state = profile.PARSED_DIR / "acts/2019" / f"{CELEX}.json"
    state.write_text("{interrupted")
    segment.segment([CELEX], profile=profile, transport=fetch, parser=parse, workers=1)
    assert len(rows(profile)) == 4
    assert len({(row["eli_id"], row["language"]) for row in rows(profile)}) == 4


def article_rows(celex, text):
    act_eli = f"http://data.europa.eu/eli/reg/{celex[1:5]}/{int(celex[6:])}/oj"
    return [segment.records_for(celex, language, act_eli, {}, {
        "articles": [{"article_number": "1", "article_title": "Duties", "article_html": f"<p>{text}</p>"}]
    }, [], None)[0] for language in ACT_LANGUAGES]


def test_reference_stages_and_audits_use_only_explicit_profile(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    profile.ensure_dirs()
    originals = {name: getattr(paths, name) for name in ("ARTICLES_JSONL", "EXTERNAL_EDGES_JSONL", "REPORTS_DIR")}
    data = article_rows("32004R0021", "Officials shall report annually.")
    data += article_rows(CELEX, "Article 1 of Regulation (EC) No 21/2004 applies.")
    profile.ARTICLES_JSONL.write_text("".join(json.dumps(row) + "\n" for row in data))
    references.extract(profile=profile)
    good, bad = project.run(profile=profile)
    assert good == {"32004R0021", CELEX} and not bad
    summary = resolver.resolve(profile=profile)
    assert summary["edges"] == 1
    edge = json.loads(profile.EXTERNAL_EDGES_JSONL.read_text())
    assert edge["target_celex"] == "32004R0021"
    assert edge["available_languages"] == list(ACT_LANGUAGES)
    audit.sample(10, 42, profile=profile)
    audit.recall(profile=profile)
    resolver.audit(profile=profile)
    assert (profile.REPORTS_DIR / "audit_sample.csv").exists()
    assert (profile.REPORTS_DIR / "recall_probe.json").exists()
    assert (profile.REPORTS_DIR / "external_resolution_audit.json").exists()
    assert all(getattr(paths, name) == value for name, value in originals.items())


def test_baseline_quarantine_is_carried_into_an_otherwise_matching_join(tmp_path):
    profile = paths.StructurePaths(tmp_path)
    profile.ensure_dirs()
    data = article_rows(CELEX, "Officials shall report annually.")
    profile.ARTICLES_JSONL.write_text("".join(json.dumps(row) + "\n" for row in data))
    profile.INTERNAL_EDGES_JSONL.write_text("")
    profile.EXTERNAL_REFS_JSONL.write_text("")
    inherited = [{"celex_id": CELEX, "reason": "previously-rejected", "languages_present": []}]
    good, bad = project.run(profile=profile, baseline_quarantine=inherited)
    assert not good and bad == inherited
    assert json.loads(profile.QUARANTINE_JSONL.read_text()) == inherited[0]


def test_parser_identity_changes_when_imported_tracked_backend_code_changes(tmp_path, monkeypatch):
    tool = tmp_path / "parser-tool"
    backend = tool / "backend"
    entrypoint = backend / "bin/parse-fmx.js"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("require('../library.js');\n")
    library = backend / "library.js"
    library.write_text("module.exports = 'original';\n")
    (backend / "package.json").write_text('{"name":"fixture"}')
    (backend / "package-lock.json").write_text('{"lockfileVersion":3}')
    for args in (("init", "-q"), ("add", "backend"),
                 ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")):
        subprocess.run(["git", "-C", str(tool), *args], check=True, capture_output=True)
    monkeypatch.setenv("EURLEX_PARSE_FMX", str(entrypoint))
    monkeypatch.setattr(segment, "_node_version", lambda: "v20.fixture")
    original_entrypoint = entrypoint.read_bytes()
    before = segment._parser_identity(segment.run_parse_fmx)
    provenance = segment.parser_provenance()
    assert provenance["git_head"] and provenance["git_dirty"] is False
    library.write_text("module.exports = 'changed dependency';\n")
    after = segment._parser_identity(segment.run_parse_fmx)
    changed = segment.parser_provenance()
    assert before != after and entrypoint.read_bytes() == original_entrypoint
    assert changed["git_head"] == provenance["git_head"]
    assert changed["git_dirty"] is True
    assert changed["tracked_backend_diff_sha256"] != provenance["tracked_backend_diff_sha256"]


def test_custom_parser_fingerprints_adjacent_lockfile_and_runtime_without_parent_git(tmp_path, monkeypatch):
    backend = tmp_path / "outer-repository/custom-tool/backend"
    entrypoint = backend / "bin/parse-fmx.js"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("// custom parser\n")
    # There is deliberately no .git at custom-tool; an ancestor's repository
    # must never become this parser's provenance source.
    (tmp_path / "outer-repository/.git").mkdir()
    lockfile = backend / "package-lock.json"
    lockfile.write_text('{"lockfileVersion":2}')
    monkeypatch.setenv("EURLEX_PARSE_FMX", str(entrypoint))
    monkeypatch.setattr(segment, "_node_version", lambda: "v20.fixture")
    monkeypatch.setattr(segment, "_tool_git", lambda *args: pytest.fail("searched parent repository"))
    before = segment._parser_identity(segment.run_parse_fmx)
    lockfile.write_text('{"lockfileVersion":3}')
    assert segment._parser_identity(segment.run_parse_fmx) != before
    after_lockfile = segment._parser_identity(segment.run_parse_fmx)
    monkeypatch.setattr(segment, "_node_version", lambda: "v22.fixture")
    assert segment._parser_identity(segment.run_parse_fmx) != after_lockfile
    assert segment.parser_provenance()["git_head"] is None
