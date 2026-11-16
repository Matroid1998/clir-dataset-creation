"""New UN sources pass through the existing question workflow without model alignment."""

import csv
import hashlib
import json
import sqlite3
from dataclasses import asdict

import pytest

from clir_bench.cli import build_parser
from clir_bench.core import grading, llm, prompt_registry
from clir_bench.core.config import Settings
from clir_bench.core.context import AppContext
from clir_bench.domains.legal import SPEC, qac
from clir_bench.domains.legal.qac import un_batch as batch
from clir_bench.domains.legal.qac import un_generate as generator
from clir_bench.domains.legal.qac.batch_recording import read_run_metadata
from clir_bench.domains.legal.un.uprprc import prepare_records
from clir_bench.domains.legal.un.uprprc_index import UPRPRCIndex

LANGUAGES = ("ar", "zh", "en", "fr", "ru", "es", "de")
GENERATOR = "test/un-generator"
VERIFIER = "test/un-verifier"
QUOTE = "Officers must report annually."


def source_record(identifier, symbol, bodies, *, date="2024-03-25"):
    return {
        "id": identifier,
        "symbol": symbol,
        "publication_date": date,
        "title": "Exampleland resolution",
        "ftxt": [f"{symbol}\n\n{bodies[language]}" if language in bodies else ""
                 for language in LANGUAGES],
        "job_numbers": [f"{identifier}-{language}" if language in bodies else ""
                        for language in LANGUAGES],
    }


def links(record, bodies):
    return [{
        "id": record["id"],
        "src_job_number": f"{record['id']}-en",
        "dst_job_number": f"{record['id']}-{language}",
        "clean_para_index_set_pair": "1|1",
        "src_lang": "en", "dst_lang": language,
        "src_text": bodies["en"], "dst_text": body,
        "src_rate": 1.0, "dst_rate": 1.0,
    } for language, body in bodies.items() if language != "en"]


@pytest.fixture
def prepared(tmp_path):
    bodies = {
        "en": (QUOTE + " " + "The mission supports local institutions and protects civilians. " * 21
               + "Recalling Human Rights Council resolution 51/1."),
        "fr": "Les agents font rapport chaque année. La mission protège les civils.",
        "es": "Los agentes informan anualmente. La misión protege a los civiles.",
        "ar": "يقدم المسؤولون تقارير سنوية. وتحمي البعثة المدنيين.",
        "ru": "Сотрудники представляют ежегодные доклады. Миссия защищает гражданских лиц.",
        "zh": "官员必须每年提交报告。特派团保护平民。",
    }
    reference_bodies = {language: f"{body} {body}" for language, body in bodies.items()}
    reference_bodies["en"] = "The Council requests annual reports on protection of civilians."
    target = source_record("workflow-target", "S/RES/9000 (2024)", bodies)
    reference = source_record("workflow-reference", "A/HRC/RES/51/1", reference_bodies,
                              date="2022-10-06")
    out = tmp_path / "prepared"
    prepare_records([target, reference], links(target, bodies) + links(reference, reference_bodies),
                    out)
    return out / "manifest.json", target, reference


@pytest.fixture
def context(tmp_path, monkeypatch):
    # Tests use the checked-in prompts and cannot contact a registry or model endpoint.
    monkeypatch.setattr(prompt_registry, "selection", lambda: None)
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data",
                        reports_dir=tmp_path / "reports", cluster_dir=tmp_path / "cluster")
    return AppContext.build(settings, SPEC)


def arguments(context, operation, run_dir, *extra):
    return build_parser(context).parse_args([
        "qac", operation, "--source", "un", "--run-dir", str(run_dir),
        "--workers", "1", *extra,
    ])


def generation_arguments(context, manifest, run_dir, *extra):
    return arguments(context, "generate", run_dir,
                     "--un-dataset", str(manifest), "--questions", "1",
                     "--langs", "fr", "--modes", "lookup",
                     "--generation-model", GENERATOR, "--verifier-model", VERIFIER, *extra)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_models(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "client_for", lambda _: object())

    def generate(payload, *, mode, language, **kwargs):
        calls.append(("generation", language))
        assert language == "fr"
        assert ("target", "fr") in payload.evidence_sources
        assert ("target", "en") in payload.evidence_sources
        return generator.parse_candidates([{
            "question": "À quelle fréquence les agents font-ils rapport ?",
            "answer": "Les agents doivent faire rapport chaque année.",
            "question_type": "monitoring_or_reporting",
            "anchor": "agents",
            "question_cited": "Selon S/RES/9000 (2024), à quelle fréquence les agents font-ils rapport ?",
            "answer_is_translation": True,
            "evidence_spans": [{"source_ref": "target", "language": "en", "text": QUOTE}],
        }], mode, payload=payload, require_evidence=True)

    def faithfulness(*args, **kwargs):
        calls.append(("faithfulness", len(args[4])))
        return [dict.fromkeys(grading.FAITHFULNESS_KEYS, 5) | {"overall": 15}
                for _ in args[4]]

    def quality(*args, **kwargs):
        calls.append(("quality", len(args[4])))
        keys = grading.rubric_keys(args[2])
        assert kwargs["policies"]["reference_policy"] == "target_only"
        return [dict.fromkeys(keys, 4) | {"overall": 4 * len(keys), "_keys": keys}
                for _ in args[4]]

    monkeypatch.setattr(generator, "generate", generate)
    monkeypatch.setattr(grading, "grade_faithfulness", faithfulness)
    monkeypatch.setattr(grading, "grade_quality", quality)
    return calls


def test_prepared_index_preserves_six_languages_context_and_references(prepared):
    manifest, original, reference_original = prepared
    index = UPRPRCIndex(manifest)
    targets = batch.select(index, n=1, seed=42, languages=["fr"], modes=["technical"])
    assert len(targets) == 1
    target, = targets
    assert target.stratum == "resolution" and target.reference_complete
    assert target.doc_id.startswith("uprprc:")
    block = index.blocks_for(target.doc_id)[target.block_index]
    assert set(block.texts) == set(LANGUAGES) - {"de"}
    assert block.line_start is None and block.line_end is None
    for language, location in block.source_locations.items():
        full_text = index.document_text(target.doc_id, language)
        assert full_text == original["ftxt"][LANGUAGES.index(language)]
        assert full_text[location["start"]:location["end"]] == block.texts[language]
        assert hashlib.sha256(full_text.encode()).hexdigest() == location["document_sha256"]
    # A short document has no generation target, but remains available as cited context.
    reference_id, = [identifier for identifier, doc in index.docs.items()
                     if doc["symbol"] == reference_original["symbol"]]
    assert not index.docs[reference_id]["target_idxs"]
    payload = batch.prepare_payload(target, index, context_chars=30000)
    assert payload.target.block_id == target.block_id
    assert {reference.symbol for reference in payload.references} == {"A/HRC/RES/51/1"}
    assert original["symbol"] in payload.text
    assert "Les agents font rapport" in payload.text
    assert "The Council requests annual reports" in payload.text
    # Only the exact target bodies are answer-bearing, even with full raw context/references.
    assert set(payload.evidence_sources) == {("target", "en"), ("target", "fr")}


@pytest.mark.parametrize("language", ["en", "fr", "es", "zh"])
def test_new_cli_dry_run_makes_no_calls_or_run_artifacts(
        prepared, context, tmp_path, monkeypatch, language):
    manifest, _, _ = prepared
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("dry-run contacted a model"))
    run = tmp_path / "dry-run"
    args = generation_arguments(context, manifest, run, "--dry-run")
    args.langs = [language]
    assert qac.generate(args, context) == 0
    index = UPRPRCIndex(manifest)
    target, = batch.select(index, n=1, seed=42, languages=[language], modes=["lookup"])
    payload = batch.prepare_payload(target, index, context_chars=30000)
    question, answer = {
        "en": ("How often must officers report?", QUOTE),
        "fr": ("À quelle fréquence les agents font-ils rapport ?", "Chaque année."),
        "es": ("¿Con qué frecuencia deben informar los agentes?", "Cada año."),
        "zh": ("官员必须多久报告一次？", "每年。"),
    }[language]
    candidate, = generator.parse_candidates([{
        "question": question, "answer": answer, "answer_is_translation": language != "en",
        "evidence_spans": [{"source_ref": "target", "language": "en", "text": QUOTE}],
    }], "lookup", payload=payload, require_evidence=True)
    assert candidate.evidence == QUOTE and candidate.evidence_language == "en"
    assert not run.exists()


def test_new_source_default_modes_have_existing_generation_prompts(
        prepared, context, tmp_path, monkeypatch):
    manifest, _, _ = prepared
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("dry-run contacted a model"))
    args = arguments(context, "generate", tmp_path / "defaults", "--un-dataset", str(manifest),
                     "--questions", "1", "--dry-run")
    assert qac.generate(args, context) == 0


def test_optional_legacy_union_keeps_raw_sources_and_reference_only_membership(prepared, tmp_path):
    _, original, _ = prepared
    bodies = {language: original["ftxt"][LANGUAGES.index(language)].split("\n\n", 1)[1]
              for language in LANGUAGES[:-1]}
    bodies["en"] = bodies["en"].replace("Human Rights Council resolution 51/1",
                                        "Security Council resolution 1 (1999)")
    original = source_record("legacy-referencing-target", original["symbol"], bodies)
    output = tmp_path / "with-legacy"
    legacy = tmp_path / "legacy"
    sixway = legacy / "sixway"
    sixway.mkdir(parents=True)
    legacy_ids = ["1999/s/res/1_1999_", "1999/s/res/2_1999_"]
    texts = {language: f"{language} protection of civilians\n{language} annual reports"
             for language in LANGUAGES[:-1]}
    for language, text in texts.items():
        (sixway / f"UNv1.0.6way.{language}").write_text(
            text + f"\n{language} second document\n", encoding="utf-8")
    docs = []
    with (legacy / "blocks.jsonl").open("wb") as stream:
        for position, identifier in enumerate(legacy_ids):
            first, last = (1, 2) if position == 0 else (3, 3)
            text = texts["en"] if position == 0 else "en second document"
            symbol = f"S/RES/{position + 1}(1999)"
            docs.append({
                "doc_id": identifier, "symbol": symbol, "title": "Legacy resolution",
                "n_blocks": 1, "n_targets": 1, "target_idxs": [0], "annexes": [],
                "line_start": first, "line_end": last, "n_lines": last - first + 1,
                "char_count": len(text), "offset": stream.tell(),
            })
            block = {"doc_id": identifier, "block_id": f"{identifier}#0", "symbol": symbol,
                     "block_index": 0, "n_blocks": 1, "line_start": first, "line_end": last,
                     "token_count": len(text.split()), "text": text, "in_range": True}
            stream.write((json.dumps(block) + "\n").encode())
    (legacy / "docs.jsonl").write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    prepare_records([original], links(original, bodies), output,
                    legacy_docs=legacy / "docs.jsonl", legacy_blocks=legacy / "blocks.jsonl",
                    legacy_sixway_dir=sixway)
    manifest = output / "manifest.json"
    index = UPRPRCIndex(manifest)
    assert index.identity["legacy_sha256"]
    assert all(not index.docs[identifier]["target_idxs"] for identifier in legacy_ids)
    for language, text in texts.items():
        assert index.document_text(legacy_ids[0], language) == text
        assert index.document_text(legacy_ids[1], language) == f"{language} second document"
    target, = batch.select(index, n=1, seed=42, languages=["fr"], modes=["lookup"])
    assert target.doc_id not in legacy_ids
    payload = batch.prepare_payload(target, index, context_chars=30000)
    reference, = payload.references
    assert reference.doc_id == legacy_ids[0]
    assert reference.texts == {"en": texts["en"], "fr": texts["fr"]}
    with pytest.raises(ValueError, match="reference-only"):
        index.build(legacy_ids[0], 0)


def test_generated_source_and_evidence_export_with_all_six_originals(
        prepared, context, tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    from clir_bench.publishing.generations import prepare

    manifest, original, _ = prepared
    fake_models(monkeypatch)
    run = tmp_path / "run"
    assert qac.generate(generation_arguments(context, manifest, run), context) == 0
    export_manifest = tmp_path / "export.json"
    export_manifest.write_text(json.dumps({
        "version": 1, "corpora": {"un": {
            "generation_sqlite": [str(run / "run.sqlite")],
            "dataset_manifest": str(manifest),
            "languages": list(LANGUAGES[:-1]),
        }},
    }))
    output = tmp_path / "export"
    report = prepare(export_manifest, output, scope="full")
    assert report["configs"]["qas"] == 1
    assert report["configs"]["corpus"] == 12
    qa, = pq.read_table(output / "data/qas").to_pylist()
    evidence, = pq.read_table(output / "data/evidence").to_pylist()
    corpus = {row["_id"]: row for row in pq.read_table(output / "data/corpus").to_pylist()}
    source = corpus[evidence["source_unit_id"]]
    assert source["text"] == original["ftxt"][LANGUAGES.index("en")]
    assert source["text"][evidence["start"]:evidence["end"]] == QUOTE
    assert evidence["start"] == source["text"].index(QUOTE)
    assert qa["question_language"] == "fr" and qa["evidence_language"] == "en"
    assert qa["answer_is_translation"] is True
    assert {row["language"] for row in corpus.values()} == set(LANGUAGES[:-1])
    raw = json.loads(qa["metadata_json"])
    locations = json.loads(raw["source_locations"])
    assert set(locations) == set(LANGUAGES[:-1])
    for language, location in locations.items():
        full_text = corpus[f"un:{qa['source_document_id']}:{language}"]["text"]
        expected_body = original["ftxt"][LANGUAGES.index(language)].split("\n\n", 1)[1]
        assert full_text[location["start"]:location["end"]] == expected_body
        assert hashlib.sha256(full_text.encode()).hexdigest() == location["document_sha256"]


def test_generate_resume_regrade_preserve_source_and_candidate_identity(
        prepared, context, tmp_path, monkeypatch):
    manifest, _, _ = prepared
    calls = fake_models(monkeypatch)
    run = tmp_path / "run"
    args = generation_arguments(context, manifest, run)
    assert qac.generate(args, context) == 0
    original, = read_rows(run / "results.csv")
    assert original["grading_status"] == "completed" and original["is_best"] == "True"
    assert original["question_language"] == "fr" and original["answer_is_translation"] == "True"
    saved_span, = json.loads(original["evidence_spans"])
    assert saved_span["location_kind"] == "document_chars"
    assert saved_span.get("line_start") is None and saved_span.get("line_end") is None
    assert saved_span["text"] == QUOTE
    assert set(json.loads(original["source_locations"])) == set(LANGUAGES[:-1])
    saved = read_run_metadata(run)
    assert saved["config"]["un_dataset"] == str(manifest.resolve())
    assert saved["config"]["un_dataset_identity"]["database_sha256"]
    assert [stage for stage, _ in calls] == ["generation", "faithfulness", "quality"]

    calls.clear()
    # Omitted --un-dataset must use the original pinned dataset on resume.
    resume = arguments(context, "generate", run, "--resume")
    assert qac.generate(resume, context) == 0
    assert calls == []
    assert read_rows(run / "results.csv") == [original]

    regrade_run = tmp_path / "regrade"
    regrade = arguments(context, "regrade", regrade_run, "--input", str(run / "results.csv"),
                        "--verifier-model", VERIFIER)
    assert qac.regrade(regrade, context) == 0
    revised, = read_rows(regrade_run / "results.csv")
    for field in ("candidate_id", "question", "answer", "target_id", "source_payload_sha256",
                  "evidence_spans", "generator_model_id"):
        assert revised[field] == original[field]
    assert revised["regrade_status"] == "completed"
    assert [stage for stage, _ in calls] == ["faithfulness", "quality"]
    assert read_run_metadata(regrade_run)["config"]["un_dataset_identity"] == saved["config"]["un_dataset_identity"]


@pytest.mark.parametrize("reverse", [False, True])
def test_joined_run_metadata_retains_un_identity_in_both_source_orders(prepared, reverse):
    manifest, _, _ = prepared
    identity = UPRPRCIndex(manifest).identity
    records = [
        {"config": {"sources": ["eurlex"]}, "targets": []},
        {"config": {"sources": ["un"], "un_dataset": str(manifest),
                    "un_dataset_identity": identity}, "targets": []},
    ]
    if reverse:
        records.reverse()
    joined = qac._joined_metadata(records)
    assert joined["config"]["un_dataset"] == str(manifest)
    assert joined["config"]["un_dataset_identity"] == identity
    conflicting = {"config": {"un_dataset_identity": identity | {"manifest_sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="disagree.*UN dataset identity"):
        qac._joined_metadata([*records, conflicting])


@pytest.mark.parametrize("operation", ["generate", "regrade"])
def test_bare_csv_replay_inherits_pinned_source_without_parent_database(
        prepared, context, tmp_path, monkeypatch, operation):
    manifest, _, _ = prepared
    calls = fake_models(monkeypatch)
    original_run = tmp_path / "original"
    assert qac.generate(generation_arguments(context, manifest, original_run), context) == 0
    original, = read_rows(original_run / "results.csv")
    bare = tmp_path / "bare" / "candidates.csv"
    write_rows(bare, [original])
    assert not (bare.parent / "run.sqlite").exists()
    calls.clear()
    run = tmp_path / operation
    if operation == "generate":
        args = arguments(context, operation, run, "--targets-from", str(bare),
                         "--generation-model", GENERATOR, "--verifier-model", VERIFIER)
        assert qac.generate(args, context) == 0
        expected_stages = ["generation", "faithfulness", "quality"]
    else:
        args = arguments(context, operation, run, "--input", str(bare),
                         "--verifier-model", VERIFIER)
        assert qac.regrade(args, context) == 0
        expected_stages = ["faithfulness", "quality"]
    regenerated, = read_rows(run / "results.csv")
    for field in ("candidate_id", "question", "answer", "target_id", "source_payload_sha256",
                  "un_dataset_identity", "evidence_spans"):
        assert regenerated[field] == original[field]
    assert read_run_metadata(run)["config"]["un_dataset_identity"] == json.loads(original["un_dataset_identity"])
    assert [stage for stage, _ in calls] == expected_stages


@pytest.mark.parametrize("operation", ["generate", "regrade"])
def test_bare_uprprc_csv_requires_source_identity_on_every_row(
        prepared, context, tmp_path, monkeypatch, operation):
    manifest, _, _ = prepared
    fake_models(monkeypatch)
    original_run = tmp_path / "original"
    assert qac.generate(generation_arguments(context, manifest, original_run), context) == 0
    original, = read_rows(original_run / "results.csv")
    bare = tmp_path / "bare" / "candidates.csv"
    write_rows(bare, [original | {"un_dataset_identity": ""}])
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("unpinned CSV reached a model"))
    input_flag = "--targets-from" if operation == "generate" else "--input"
    args = arguments(context, operation, tmp_path / operation, input_flag, str(bare), "--dry-run")
    with pytest.raises(ValueError, match="require.*pinned UN dataset identity"):
        getattr(qac, operation)(args, context)


def test_failed_grading_rows_retain_source_identity_for_bare_csv_recovery(
        prepared, context, tmp_path, monkeypatch):
    manifest, _, _ = prepared
    calls = fake_models(monkeypatch)
    original_run = tmp_path / "original"
    assert qac.generate(generation_arguments(context, manifest, original_run), context) == 0
    original, = read_rows(original_run / "results.csv")
    run_one = batch.run_one

    def interrupted_grading(*args, **kwargs):
        raise RuntimeError("grading service unavailable")

    monkeypatch.setattr(batch, "run_one", interrupted_grading)
    failed_run = tmp_path / "failed"
    args = arguments(context, "regrade", failed_run,
                     "--input", str(original_run / "results.csv"), "--verifier-model", VERIFIER)
    assert qac.regrade(args, context) == 1
    failed, = read_rows(failed_run / "results.csv")
    assert failed["grading_status"] == "failed"
    assert "grading service unavailable" in failed["grading_error"]
    assert failed["is_best"] == "False"
    for field in ("candidate_id", "question", "answer", "doc_id", "block_id", "block_index",
                  "n_blocks", "target_id", "source_payload_sha256", "un_dataset_identity",
                  "source_locations", "evidence_spans"):
        assert failed[field] == original[field]

    bare = tmp_path / "bare" / "failed.csv"
    write_rows(bare, [failed])
    monkeypatch.setattr(batch, "run_one", run_one)
    calls.clear()
    recovered_run = tmp_path / "recovered"
    recovered_args = arguments(context, "regrade", recovered_run, "--input", str(bare),
                               "--verifier-model", VERIFIER)
    assert qac.regrade(recovered_args, context) == 0
    recovered, = read_rows(recovered_run / "results.csv")
    assert recovered["grading_status"] == "completed"
    assert recovered["candidate_id"] == original["candidate_id"]
    assert recovered["un_dataset_identity"] == original["un_dataset_identity"]
    assert [stage for stage, _ in calls] == ["faithfulness", "quality"]


def test_resume_refuses_a_changed_manifest_before_any_model_stage(
        prepared, context, tmp_path, monkeypatch):
    manifest, _, _ = prepared
    fake_models(monkeypatch)
    run = tmp_path / "run"
    assert qac.generate(generation_arguments(context, manifest, run), context) == 0
    manifest.write_text(manifest.read_text() + "\n")
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("changed sources reached a model"))
    with pytest.raises(ValueError, match="(?i)(dataset|source|changed|hash)"):
        qac.generate(arguments(context, "generate", run, "--resume"), context)


def test_changed_database_is_refused(prepared):
    manifest, _, _ = prepared
    configuration = json.loads(manifest.read_text())
    database = manifest.parent / configuration["database"]
    with database.open("ab") as stream:
        stream.write(b"changed input")
    with pytest.raises(ValueError, match="(?i)(database|hash|changed)"):
        UPRPRCIndex(manifest)


def test_replay_cannot_turn_context_into_a_target(prepared, context, tmp_path, monkeypatch):
    manifest, _, _ = prepared
    index = UPRPRCIndex(manifest)
    target, = batch.select(index, n=1, seed=42, languages=["fr"], modes=["technical"])
    assert target.block_index > 0  # The unmatched header is retained before the target.
    forged = asdict(target) | {"block_index": 0, "block_id": f"{target.doc_id}#0"}
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("context reached a model"))
    run = tmp_path / "forged"
    with pytest.raises(ValueError, match="(?i)(target|eligible|reconstruct)"):
        qac.generate(generation_arguments(context, manifest, run), context,
                     selections=[{"corpus": "un", "target": forged}])
    assert not run.exists()


def test_rehashed_artifact_still_cannot_generate_without_all_six_sources(
        prepared, context, tmp_path, monkeypatch):
    manifest, _, _ = prepared
    configuration = json.loads(manifest.read_text())
    database = manifest.parent / configuration["database"]
    with sqlite3.connect(database) as db:
        for doc_id, block_index, raw in db.execute("SELECT doc_id,block_index,payload FROM blocks"):
            payload = json.loads(raw)
            if len(payload.get("texts", {})) != 6:
                continue
            del payload["texts"]["ru"]
            db.execute("UPDATE blocks SET payload=? WHERE doc_id=? AND block_index=?",
                       (json.dumps(payload), doc_id, block_index))
    configuration["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(configuration))
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("incomplete target reached a model"))
    run = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="(?i)(language|six|target|source|translation)"):
        qac.generate(generation_arguments(context, manifest, run, "--dry-run"), context)
    assert not run.exists()


def test_explicit_replay_requires_all_six_languages_of_references(
        prepared, context, tmp_path, monkeypatch):
    manifest, _, _ = prepared
    index = UPRPRCIndex(manifest)
    target, = batch.select(index, n=1, seed=42, languages=["fr"], modes=["lookup"])
    reference_id, = [identifier for identifier, doc in index.docs.items()
                     if doc["symbol"] == "A/HRC/RES/51/1"]
    configuration = json.loads(manifest.read_text())
    database = manifest.parent / configuration["database"]
    with sqlite3.connect(database) as db:
        db.execute("DELETE FROM texts WHERE doc_id=? AND language='ru'", (reference_id,))
    configuration["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(configuration))
    monkeypatch.setattr(llm, "client_for", lambda _: pytest.fail("incomplete reference reached a model"))
    run = tmp_path / "incomplete-reference"
    with pytest.raises(ValueError, match="(?i)(reference|six|language)"):
        qac.generate(generation_arguments(context, manifest, run, "--dry-run"), context,
                     selections=[{"corpus": "un", "target": asdict(target)}])
    assert not run.exists()


@pytest.mark.parametrize("change", ["empty", "altered"])
def test_generation_preflight_rechecks_alignment_proof_after_artifact_is_rehashed(
        prepared, change):
    manifest, _, _ = prepared
    original = UPRPRCIndex(manifest)
    target, = batch.select(original, n=1, seed=42, languages=["fr"], modes=["lookup"])
    configuration = json.loads(manifest.read_text())
    database = manifest.parent / configuration["database"]
    with sqlite3.connect(database) as db:
        raw, = db.execute("SELECT payload FROM blocks WHERE block_id=?",
                          (target.block_id,)).fetchone()
        block = json.loads(raw)
        assert block["alignment_links"]
        if change == "empty":
            block["alignment_links"] = []
        else:
            block["alignment_links"][0]["english_text_sha256"] = "0" * 64
        db.execute("UPDATE blocks SET payload=? WHERE block_id=?",
                   (json.dumps(block), target.block_id))
    configuration["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="(?i)(alignment|proof)"):
        UPRPRCIndex(manifest).build(target.doc_id, target.block_index, languages=("fr", "en"))
