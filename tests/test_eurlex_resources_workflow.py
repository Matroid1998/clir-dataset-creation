"""Pinned newer EUR-Lex articles use the existing QAC workflow and full source export."""

import csv
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from clir_bench.cli import build_parser
from clir_bench.core import grading, llm, prompt_registry
from clir_bench.core.config import Settings
from clir_bench.core.context import AppContext
from clir_bench.domains.legal import SPEC, qac
from clir_bench.domains.legal.qac import eurlex_batch as batch
from clir_bench.domains.legal.qac import eurlex_generate as generator
from clir_bench.domains.legal.qac.batch_recording import read_run_metadata
from clir_bench.domains.legal.structure import ACT_LANGUAGES
from clir_bench.publishing.generations import prepare
from clir_bench.publishing.records import normalize
from clir_bench.publishing.sources import SourceIndex, load_eurlex

NEW_ACT = "32020R9001"
OLD_ACT = "32009R9000"
NEW_ELI = "http://data.europa.eu/eli/reg/2020/9001/art_1/oj"
SECOND_ELI = "http://data.europa.eu/eli/reg/2020/9001/art_2/oj"
OLD_ELI = "http://data.europa.eu/eli/reg/2009/9000/art_5/oj"
QUOTE = "Officers must report annually."
GENERATOR = "test/eurlex-generator"
VERIFIER = "test/eurlex-verifier"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def freeze_manifest(directory):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {
        "format": "eurlex-resources-v1", "version": 1,
    }
    files = ["articles.jsonl", "internal_edges.jsonl", "external_references.jsonl",
             "external_edges.jsonl", "reference_status.jsonl", "quarantine.jsonl", "target_acts.json"]
    manifest["files"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in files}
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    rows = []
    for eli, act, number in ((NEW_ELI, NEW_ACT, "1"), (SECOND_ELI, NEW_ACT, "2"),
                             (OLD_ELI, OLD_ACT, "5")):
        for language in ACT_LANGUAGES:
            text = (QUOTE + " " + "The authority supports institutions and publishes reports. " * 20
                    if language == "en" else
                    f"{language} α Original translated article {number}. " * 20)
            rows.append({"eli_id": eli, "celex_id": act, "article_number": number,
                         "language": language, "unit_type": "article", "heading": "Reporting",
                         "act_title": f"Regulation {act}", "text": text})
    write_jsonl(directory / "articles.jsonl", rows)
    for name in ("internal_edges.jsonl", "external_references.jsonl", "quarantine.jsonl"):
        write_jsonl(directory / name, [])
    write_jsonl(directory / "external_edges.jsonl", [{
        "source_article_id": SECOND_ELI, "target_article_id": OLD_ELI,
        "source_unit_type": "article", "target_unit_type": "article", "char_start": 12,
    }])
    write_jsonl(directory / "reference_status.jsonl", [
        {"eli_id": eli, "complete": True, "cites_annex": False}
        for eli in (NEW_ELI, SECOND_ELI, OLD_ELI)
    ])
    (directory / "target_acts.json").write_text(json.dumps([NEW_ACT]))
    return freeze_manifest(directory), rows


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_registry, "selection", lambda: None)
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data",
                        reports_dir=tmp_path / "reports", cluster_dir=tmp_path / "cluster")
    return AppContext.build(settings, SPEC)


def arguments(context, operation, run, *extra):
    return build_parser(context).parse_args([
        "qac", operation, "--source", "eurlex", "--run-dir", str(run),
        "--workers", "1", *extra,
    ])


def generation_arguments(context, manifest, run, *extra):
    return arguments(context, "generate", run, "--eurlex-dataset", str(manifest),
                     "--questions", "1", "--langs", "en", "--modes", "lookup",
                     "--generation-model", GENERATOR, "--verifier-model", VERIFIER, *extra)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fake_models(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "client_for", lambda _: object())

    def generate(payload, *, mode, language, **kwargs):
        calls.append("generation")
        assert payload.target.celex_id == NEW_ACT
        assert set(payload.target.texts) == set(ACT_LANGUAGES)
        return generator.parse_candidates([{
            "question": "How often must officers report?", "answer": "Annually.",
            "question_type": "monitoring_or_reporting", "articles_involved": ["1"],
            "question_cited": "Under Regulation 9001/2020, how often must officers report?",
            "anchor": "officers", "answer_is_translation": False,
            "evidence_spans": [{"source_ref": "1", "language": "en", "text": QUOTE}],
        }], payload, mode, require_evidence=True)

    def faithfulness(*args, **kwargs):
        calls.append("faithfulness")
        return [dict.fromkeys(grading.FAITHFULNESS_KEYS, 5) | {"overall": 15} for _ in args[4]]

    def quality(*args, **kwargs):
        calls.append("quality")
        keys = grading.rubric_keys(args[2])
        return [dict.fromkeys(keys, 4) | {"overall": 4 * len(keys), "_keys": keys} for _ in args[4]]

    monkeypatch.setattr(generator, "generate", generate)
    monkeypatch.setattr(grading, "grade_faithfulness", faithfulness)
    monkeypatch.setattr(grading, "grade_quality", quality)
    return calls


def test_profile_index_keeps_all_translations_and_baseline_reference(prepared):
    from clir_bench.domains.legal.structure.profile import ProfileArticleIndex

    manifest, original = prepared
    index = ProfileArticleIndex(manifest)
    targets = batch.select(index, n=1, seed=42, languages=["en"], modes=["lookup"])
    target, = targets
    assert target.eli_id == NEW_ELI and target.celex_id == NEW_ACT
    assert OLD_ELI in index.by_eli
    payload = index.build(SECOND_ELI, languages=("fr", "en"))
    assert [unit.eli_id for unit in payload.external_references] == [OLD_ELI]
    for row in original:
        assert index.by_eli[row["eli_id"]].texts[row["language"]] == row["text"]


def test_new_profile_dry_run_performs_no_model_calls_or_output(prepared, context, tmp_path, monkeypatch):
    manifest, _ = prepared
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("dry run contacted a model"))
    run = tmp_path / "dry-run"
    assert qac.generate(generation_arguments(context, manifest, run, "--dry-run"), context) == 0
    assert not run.exists()


def test_generate_resume_bare_csv_regrade_and_full_publication(prepared, context, tmp_path, monkeypatch):
    manifest, original = prepared
    calls = fake_models(monkeypatch)
    run = tmp_path / "run"
    assert qac.generate(generation_arguments(context, manifest, run), context) == 0
    row, = read_rows(run / "results.csv")
    assert calls == ["generation", "faithfulness", "quality"]
    identity = json.loads(row["eurlex_dataset_identity"])
    assert identity == read_run_metadata(run)["config"]["eurlex_dataset_identity"]
    assert row["grading_status"] == "completed" and row["is_best"] == "True"
    assert json.loads(row["evidence_spans"])[0]["text"] == QUOTE
    calls.clear()
    assert qac.generate(arguments(context, "generate", run, "--resume"), context) == 0
    assert calls == [] and read_rows(run / "results.csv") == [row]

    # A standalone CSV retains the same immutable profile without a parent run.sqlite.
    standalone = tmp_path / "standalone.csv"
    with standalone.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        # The supported single-source hint supplies an empty corpus cell; it
        # must still recover the original profile before loading any sources.
        writer.writerow({**row, "corpus": ""})
    revised_run = tmp_path / "regrade"
    assert qac.regrade(arguments(context, "regrade", revised_run, "--input", str(standalone),
                                "--verifier-model", VERIFIER), context) == 0
    revised, = read_rows(revised_run / "results.csv")
    assert calls == ["faithfulness", "quality"]
    for key in ("question", "answer", "candidate_id", "target_id", "source_payload_sha256",
                "eurlex_dataset_identity", "evidence_spans"):
        assert revised[key] == row[key]

    calls.clear()
    replay_run = tmp_path / "target-replay"
    assert qac.generate(arguments(context, "generate", replay_run, "--targets-from", str(standalone),
                                  "--generation-model", GENERATOR, "--verifier-model", VERIFIER), context) == 0
    replayed, = read_rows(replay_run / "results.csv")
    assert replayed["eurlex_dataset_identity"] == row["eurlex_dataset_identity"]
    assert replayed["source_payload_sha256"] == row["source_payload_sha256"]
    assert calls == ["generation", "faithfulness", "quality"]

    # A row marked as originating in this profile cannot silently lose its pin
    # merely because its parent run.sqlite is no longer beside the CSV.
    unpinned = tmp_path / "unpinned.csv"
    with unpinned.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({**row, "eurlex_dataset_identity": ""})
    calls.clear()
    with pytest.raises(ValueError, match="(?i)(identity|profile|dataset)"):
        qac.regrade(arguments(context, "regrade", tmp_path / "unpinned-regrade", "--input", str(unpinned),
                              "--verifier-model", VERIFIER), context)
    assert calls == []

    spec = {"version": 1, "corpora": {"eurlex": {
        "generation_sqlite": [str(revised_run / "run.sqlite")], "dataset_manifest": str(manifest),
        "languages": list(ACT_LANGUAGES), "unit_types": ["article", "annex"],
    }}}
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(spec))
    output = tmp_path / "export"
    report = prepare(input_path, output, scope="full")
    assert report["configs"] == {"qas": 1, "evidence": 1, "source_qrels": 4, "corpus": 12}
    corpus = pq.read_table(output / "data/corpus").to_pylist()
    expected = {(row["eli_id"], row["language"]): row["text"] for row in original}
    for source in corpus:
        assert source["text"] == expected[source["source_id"], source["language"]]
        assert json.loads(source["metadata_json"])["dataset_identity"] == identity
    qa, = pq.read_table(output / "data/qas").to_pylist()
    assert qa["document_text"] == expected[NEW_ELI, "en"]
    assert qa["evidence_spans"][0]["text"] == QUOTE

    # A valid rehashed profile with changed FR text is a different source family,
    # even though the quoted English article and its evidence hash are unchanged.
    modified = [dict(article) for article in original]
    next(article for article in modified if article["eli_id"] == NEW_ELI and article["language"] == "fr")[
        "text"
    ] += " Nouveau texte."
    write_jsonl(manifest.parent / "articles.jsonl", modified)
    freeze_manifest(manifest.parent)
    with pytest.raises(ValueError, match="dataset identity differs"):
        prepare(input_path, tmp_path / "changed-export", scope="full")
    assert not (tmp_path / "changed-export").exists()
    calls.clear()
    with pytest.raises(ValueError, match="(?i)(dataset|profile|changed|identity)"):
        qac.generate(arguments(context, "generate", run, "--resume"), context)
    assert calls == []


def test_profile_article_spec_is_exclusive_before_loading(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_eurlex(None, {"dataset_manifest": "manifest.json", "articles": "articles.jsonl"},
                    tmp_path, None)


def test_profile_conflicts_are_detected_outside_selected_language_and_scope(prepared, tmp_path):
    manifest, original = prepared
    duplicate = next(dict(row) for row in original if row["eli_id"] == OLD_ELI and row["language"] == "fr")
    duplicate["text"] = "A conflicting French reference version"
    write_jsonl(manifest.parent / "articles.jsonl", [*original, duplicate])
    freeze_manifest(manifest.parent)
    index = SourceIndex(tmp_path / "index.sqlite")
    try:
        with pytest.raises(ValueError, match="conflict"):
            load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en"]},
                        tmp_path, {NEW_ELI})
    finally:
        index.close()


def test_cross_language_article_identity_cannot_form_a_false_translation_family(prepared, tmp_path):
    manifest, original = prepared
    modified = [dict(row) for row in original]
    next(row for row in modified if row["eli_id"] == OLD_ELI and row["language"] == "fr")[
        "celex_id"
    ] = NEW_ACT
    write_jsonl(manifest.parent / "articles.jsonl", modified)
    freeze_manifest(manifest.parent)
    index = SourceIndex(tmp_path / "index.sqlite")
    try:
        with pytest.raises(ValueError, match="conflicting EUR-Lex article identity"):
            load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en"]},
                        tmp_path, {NEW_ELI})
    finally:
        index.close()


def test_legacy_baseline_rows_stay_usable_but_new_targets_require_identity(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, _ = prepared
    index = SourceIndex(tmp_path / "index.sqlite")
    try:
        load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en"]}, tmp_path, None)
        legacy = {"target_id": OLD_ELI, "document_id": OLD_ACT, "question_language": "en",
                  "question": "How often?", "answer": "Annually.", "mode": "lookup",
                  "generator_model_id": GENERATOR, "articles_involved_eli": OLD_ELI}
        assert normalize("eurlex", legacy, {}, index)["source_document_id"] == OLD_ELI
        with pytest.raises(ValueError, match="saved dataset identity"):
            normalize("eurlex", {**legacy, "eurlex_dataset_format": "eurlex-resources-v1"}, {}, index)
        with pytest.raises(ValueError, match="saved dataset identity"):
            normalize("eurlex", {**legacy, "target_id": NEW_ELI, "document_id": NEW_ACT,
                                 "articles_involved_eli": NEW_ELI}, {}, index)
        identity = load_profile(manifest).identity
        with pytest.raises(ValueError, match="dataset identity differs"):
            normalize("eurlex", {**legacy, "eurlex_dataset_identity": {
                **identity, "manifest_sha256": "0" * 64,
            }}, {}, index)
    finally:
        index.close()


def test_pinned_profile_qa_cannot_use_baseline_reference_as_its_primary_target(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, _ = prepared
    index = SourceIndex(tmp_path / "index.sqlite")
    try:
        load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en"]}, tmp_path, None)
        row = {"target_id": OLD_ELI, "document_id": OLD_ACT, "question_language": "en",
               "question": "How often?", "answer": "Annually.", "mode": "lookup",
               "generator_model_id": GENERATOR, "articles_involved_eli": OLD_ELI,
               "eurlex_dataset_format": "eurlex-resources-v1",
               "eurlex_dataset_identity": load_profile(manifest).identity}
        with pytest.raises(ValueError, match="primary source must be a newer target article"):
            normalize("eurlex", row, {}, index)
    finally:
        index.close()


def test_pinned_profile_qa_cannot_use_annex_of_a_new_act_as_its_primary_target(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, original = prepared
    annex_eli = "http://data.europa.eu/eli/reg/2020/9001/anx_1/oj"
    annex = {"eli_id": annex_eli, "celex_id": NEW_ACT, "article_number": "", "language": "en",
             "unit_type": "annex", "heading": "Annex I", "text": "Original annex table."}
    write_jsonl(manifest.parent / "articles.jsonl", [*original, annex])
    freeze_manifest(manifest.parent)
    index = SourceIndex(tmp_path / "index.sqlite")
    try:
        load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en"]}, tmp_path, None)
        row = {"target_id": annex_eli, "document_id": NEW_ACT, "question_language": "en",
               "question": "Which value?", "answer": "The table value.", "mode": "lookup",
               "generator_model_id": GENERATOR, "articles_involved_eli": annex_eli,
               "eurlex_dataset_identity": load_profile(manifest).identity}
        with pytest.raises(ValueError, match="primary source must be a newer target article"):
            normalize("eurlex", row, {}, index)
    finally:
        index.close()


def _limit_publication_targets(manifest, targets):
    path = manifest.parent / "target_articles.json"
    path.write_text(json.dumps(sorted(targets)), encoding="utf-8")
    frozen = json.loads(manifest.read_text())
    frozen["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen.setdefault("config", {})["target_articles"] = len(targets)
    frozen.setdefault("counts", {})["target_articles"] = len(targets)
    manifest.write_text(json.dumps(frozen), encoding="utf-8")


def test_article_limited_publication_keeps_unlisted_articles_as_references(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, _ = prepared
    _limit_publication_targets(manifest, [NEW_ELI])
    index = SourceIndex(tmp_path / "publication.sqlite")
    try:
        load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en", "fr"]},
                    tmp_path, None)
        sources = {row["source_id"]: row for row in index.rows() if row["language"] == "en"}
        assert set(sources) == {NEW_ELI, SECOND_ELI, OLD_ELI}
        assert json.loads(sources[NEW_ELI]["metadata_json"])["dataset_target_article"] is True
        assert json.loads(sources[SECOND_ELI]["metadata_json"])["dataset_target_article"] is False
        assert json.loads(sources[OLD_ELI]["metadata_json"])["dataset_target_article"] is False
        raw = {"target_id": NEW_ELI, "document_id": NEW_ACT, "question_language": "en",
               "question": "How often?", "answer": "Annually.", "mode": "lookup",
               "generator_model_id": GENERATOR,
               "articles_involved_eli": [NEW_ELI, SECOND_ELI],
               "eurlex_dataset_identity": load_profile(manifest).identity}
        normalized = normalize("eurlex", raw, {}, index)
        assert f"eurlex:{SECOND_ELI}:en" in normalized["involved_source_unit_ids"]
        with pytest.raises(ValueError, match="outside its article target allowlist"):
            normalize("eurlex", {**raw, "target_id": SECOND_ELI}, {}, index)
    finally:
        index.close()


def test_article_limited_publication_requires_explicit_target_metadata(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import load_profile
    from clir_bench.publishing.evidence import validate_eurlex_dataset_identity

    manifest, _ = prepared
    _limit_publication_targets(manifest, [NEW_ELI])
    index = SourceIndex(tmp_path / "publication.sqlite")
    try:
        load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en"]}, tmp_path, None)
        document = index.resolve_article(NEW_ELI, "en")
        metadata = json.loads(document["metadata_json"])
        metadata.pop("dataset_target_article")
        document["metadata_json"] = json.dumps(metadata)
        with pytest.raises(ValueError, match="outside its article target allowlist"):
            validate_eurlex_dataset_identity(
                {"eurlex_dataset_identity": load_profile(manifest).identity}, document,
            )
    finally:
        index.close()


def test_old_profile_publication_preserves_act_level_article_scope(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import load_profile
    from clir_bench.publishing.evidence import validate_eurlex_dataset_identity

    manifest, _ = prepared
    index = SourceIndex(tmp_path / "publication.sqlite")
    try:
        load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en"]}, tmp_path, None)
        for eli in (NEW_ELI, SECOND_ELI):
            document = index.resolve_article(eli, "en")
            metadata = json.loads(document["metadata_json"])
            assert metadata.pop("dataset_target_article") is True
            # Existing exported source metadata predates article allowlists.
            document["metadata_json"] = json.dumps(metadata)
            validate_eurlex_dataset_identity(
                {"eurlex_dataset_identity": load_profile(manifest).identity}, document,
            )
    finally:
        index.close()


@pytest.mark.parametrize("incomplete_eli", [NEW_ELI, OLD_ELI])
def test_english_question_still_requires_all_four_original_target_and_reference_languages(
        prepared, incomplete_eli):
    from clir_bench.domains.legal.structure.profile import ProfileArticleIndex

    manifest, original = prepared
    write_jsonl(manifest.parent / "articles.jsonl", [
        row for row in original if not (row["eli_id"] == incomplete_eli and row["language"] == "fr")
    ])
    freeze_manifest(manifest.parent)
    index = ProfileArticleIndex(manifest)
    target = NEW_ELI if incomplete_eli == NEW_ELI else SECOND_ELI
    with pytest.raises(ValueError, match="four original languages"):
        index.build(target, languages=("en",))


def test_missing_french_annex_is_checked_only_when_the_annex_is_supplied(prepared):
    from clir_bench.domains.legal.structure.profile import ProfileArticleIndex

    manifest, original = prepared
    annex_eli = "http://data.europa.eu/eli/reg/2020/9001/anx_1/oj"
    annexes = [{"eli_id": annex_eli, "celex_id": NEW_ACT, "article_number": "",
                "language": language, "unit_type": "annex", "heading": "Annex I",
                "text": f"{language} Original annex table."} for language in ACT_LANGUAGES
               if language != "fr"]
    write_jsonl(manifest.parent / "articles.jsonl", [*original, *annexes])
    write_jsonl(manifest.parent / "internal_edges.jsonl", [{
        "source_article_id": SECOND_ELI, "target_article_id": annex_eli,
        "source_unit_type": "article", "target_unit_type": "annex", "char_start": 0,
    }])
    freeze_manifest(manifest.parent)
    index = ProfileArticleIndex(manifest)
    capped = index.build(SECOND_ELI, languages=("en",), max_references=0)
    assert capped.annexes == [] and capped.external_references == []
    assert capped.dropped_annex_references
    with pytest.raises(ValueError, match="four original languages"):
        index.build(SECOND_ELI, languages=("en",), max_references=6)


@pytest.mark.parametrize("changes", [
    {"eli_id": OLD_ELI, "celex_id": OLD_ACT, "article_number": "5"},
    {"celex_id": OLD_ACT}, {"article_number": "2"},
])
def test_fixed_targets_cannot_bypass_new_act_or_article_identity_checks(
        prepared, context, tmp_path, monkeypatch, changes):
    manifest, _ = prepared
    target = batch.Target(NEW_ELI, NEW_ACT, "1", 0, "no_refs", "lookup", "en")
    selection = {**asdict(target), **changes}
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("invalid replay reached a model"))
    run = tmp_path / "invalid-replay"
    with pytest.raises(ValueError, match="(?i)(target|article|identity)"):
        qac.generate(generation_arguments(context, manifest, run), context,
                     selections=[{"corpus": "eurlex", "target": selection}])
    assert not run.exists()


def test_mixed_source_execution_keeps_eurlex_pins_out_of_un_run_config(prepared, tmp_path, monkeypatch):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, _ = prepared
    identity = load_profile(manifest).identity
    options = {"seed": 42, "eurlex_dataset": str(manifest), "eurlex_dataset_identity": identity}
    selections = [{"corpus": source, "target": {}} for source in ("un", "eurlex")]
    locations = {source: (tmp_path / source, tmp_path / source / "results.csv", {})
                 for source in ("un", "eurlex")}
    captured = []

    def execute(*args, **kwargs):
        captured.append((args[7][0]["corpus"], dict(args[6]), kwargs.get("validate_only", False)))
        return 0

    monkeypatch.setattr(qac, "_execute", execute)
    assert qac._execute_sources(None, None, "generate", locations, options, selections, {},
                                {"un": object(), "eurlex": object()}, []) == 0
    assert len(captured) == 4  # Preflight and execution for both sources.
    for source, saved, _ in captured:
        assert saved == ({"seed": 42} if source == "un" else options)
    assert options["eurlex_dataset_identity"] == identity


def test_joined_metadata_retains_eurlex_profile_in_either_source_order(prepared):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, _ = prepared
    identity = load_profile(manifest).identity
    un = {"config": {"seed": 42}, "targets": [{"corpus": "un", "target": {"doc_id": "legacy"}}]}
    eurlex = {"config": {"eurlex_dataset": str(manifest), "eurlex_dataset_identity": identity},
              "targets": [{"corpus": "eurlex", "target": {"eli_id": NEW_ELI}}]}
    for records in ([un, eurlex], [eurlex, un]):
        merged = qac._joined_metadata(records)
        assert merged["config"]["eurlex_dataset_identity"] == identity
        assert merged["config"]["eurlex_dataset"] == str(manifest)
        assert {entry["corpus"] for entry in merged["targets"]} == {"un", "eurlex"}
    changed = {"config": {**eurlex["config"], "eurlex_dataset_identity": {
        **identity, "manifest_sha256": "0" * 64,
    }}}
    with pytest.raises(ValueError, match="disagree on the EUR-Lex dataset identity"):
        qac._joined_metadata([un, eurlex, changed])


def test_un_only_replay_discards_inherited_eurlex_options_without_loading_profile(prepared, monkeypatch):
    from clir_bench.domains.legal.structure import profile

    manifest, _ = prepared
    un_index = object()
    options = {"seed": 42, "eurlex_dataset": str(manifest), "eurlex_dataset_identity": {"old": "pin"}}
    monkeypatch.setattr(profile, "ProfileArticleIndex", lambda *a, **kw: pytest.fail("UN replay loaded EUR-Lex"))
    assert qac._profile_indexes(["un"], {"un": un_index}, None, options) == {"un": un_index}
    assert options == {"seed": 42}


@pytest.mark.parametrize("operation", ["generate", "regrade"])
def test_explicit_eurlex_profile_on_un_only_work_is_rejected_before_loading_sources(
        prepared, context, tmp_path, monkeypatch, operation):
    manifest, _ = prepared
    monkeypatch.setattr(qac, "_run_locations", lambda *a, **kw: pytest.fail("invalid flag reached source setup"))
    args = SimpleNamespace(source=["un"], eurlex_dataset=manifest)
    if operation == "regrade":
        input_path = tmp_path / "un.csv"
        with input_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["corpus", "question"])
            writer.writeheader()
            writer.writerow({"corpus": "un", "question": "An old UN question"})
        args.input = input_path
    with pytest.raises(ValueError, match="requires.*(eurlex|EUR-Lex)"):
        getattr(qac, operation)(args, context)


def test_nested_diagnostic_hash_is_part_of_the_profile_and_publication_pin(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, _ = prepared
    diagnostic = manifest.parent / "reports" / "audit.json"
    diagnostic.parent.mkdir()
    diagnostic.write_text('{"accepted":true}')
    data = json.loads(manifest.read_text())
    data["files"]["reports/audit.json"] = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    original = load_profile(manifest)
    assert original.file("reports/audit.json") == diagnostic
    diagnostic.write_text('{"accepted":false}')
    with pytest.raises(ValueError, match="changed during use"):
        original.assert_unchanged()
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_profile(manifest)
    data["files"]["reports/audit.json"] = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    changed = load_profile(manifest)
    assert changed.identity != original.identity
    index = SourceIndex(tmp_path / "changed-index.sqlite")
    try:
        load_eurlex(index, {"dataset_manifest": str(manifest), "languages": ["en"]}, tmp_path, None)
        row = {"target_id": NEW_ELI, "document_id": NEW_ACT, "question_language": "en",
               "question": "How often?", "answer": "Annually.", "mode": "lookup",
               "generator_model_id": GENERATOR, "articles_involved_eli": NEW_ELI,
               "eurlex_dataset_identity": original.identity}
        with pytest.raises(ValueError, match="dataset identity differs"):
            normalize("eurlex", row, {}, index)
    finally:
        index.close()


@pytest.mark.parametrize("name", ["../outside.json", "/tmp/outside.json", "reports/../articles.jsonl"])
def test_profile_rejects_absolute_or_traversing_artifact_names(prepared, name):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, _ = prepared
    data = json.loads(manifest.read_text())
    data["files"][name] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="artifact name/checksum"):
        load_profile(manifest)


def test_profile_rejects_artifact_symlink_escaping_bundle(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import load_profile

    manifest, _ = prepared
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (manifest.parent / "escape.json").symlink_to(outside)
    data = json.loads(manifest.read_text())
    data["files"]["escape.json"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes its bundle"):
        load_profile(manifest)


def test_quarantined_baseline_conflicts_remain_raw_but_cannot_supply_sources(prepared, tmp_path):
    from clir_bench.domains.legal.structure.profile import ProfileArticleIndex

    manifest, original = prepared
    baseline = [row for row in original if row["celex_id"] == OLD_ACT]
    duplicate = {**next(row for row in baseline if row["language"] == "en"),
                 "text": "A conflicting baseline version already covered by quarantine."}
    prefix = "".join(json.dumps(row) + "\n" for row in [*baseline, duplicate]).encode()
    additions = "".join(json.dumps(row) + "\n" for row in original if row["celex_id"] != OLD_ACT).encode()
    articles = manifest.parent / "articles.jsonl"
    articles.write_bytes(prefix + additions)
    write_jsonl(manifest.parent / "quarantine.jsonl", [
        {"celex_id": OLD_ACT, "reason": "baseline-article-identity-conflict"},
    ])
    freeze_manifest(manifest.parent)
    index = ProfileArticleIndex(manifest)
    assert OLD_ELI not in index.by_eli and OLD_ACT not in index.by_act
    assert index.build(NEW_ELI, languages=("en",)).target.eli_id == NEW_ELI
    with pytest.raises(ValueError, match="reference endpoint is absent"):
        index.build(SECOND_ELI, languages=("en",))

    exported = SourceIndex(tmp_path / "profile-index.sqlite")
    try:
        load_eurlex(exported, {"dataset_manifest": str(manifest), "languages": list(ACT_LANGUAGES)},
                    tmp_path, None)
        rows = list(exported.rows())
        assert len(rows) == 8 and {row["parent_id"] for row in rows} == {NEW_ACT}
        assert exported.resolve_article(OLD_ELI, "en") is None
        assert exported.resolve_article(f"{OLD_ACT}:5", "en") is None
    finally:
        exported.close()
    assert articles.read_bytes() == prefix + additions
    assert articles.read_bytes().startswith(prefix)

    # The legacy explicit-article adapter retains its historical last-row rule.
    legacy = SourceIndex(tmp_path / "legacy-index.sqlite")
    try:
        load_eurlex(legacy, {"articles": str(articles), "languages": ["en"]}, tmp_path, None)
        assert legacy.resolve_article(OLD_ELI, "en")["text"] == duplicate["text"]
        assert legacy.duplicates == 1
    finally:
        legacy.close()
