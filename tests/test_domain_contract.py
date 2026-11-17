"""Domain discovery, chemistry contracts, and legal run CLI integration."""

from __future__ import annotations

import pytest

from clir_bench import domains
from clir_bench.core.domain import DomainSpec


@pytest.fixture(scope="module")
def spec() -> DomainSpec:
    return domains.load("chemistry")


def test_domains_are_discoverable_without_importing_them() -> None:
    assert "chemistry" in domains.available()
    assert "legal" in domains.available()


def test_unknown_domain_reports_what_is_available() -> None:
    with pytest.raises(domains.DomainNotFound) as excinfo:
        domains.load("no_such_domain")
    assert "chemistry" in str(excinfo.value)


def test_every_source_declares_an_attribution(spec: DomainSpec) -> None:
    """Publishing data without naming its provider is a licensing failure."""
    for source in spec.sources:
        assert spec.attribution_for(source.name).strip()


def test_only_ai4chem_source_is_active_with_its_own_attribution(spec: DomainSpec) -> None:
    gp = spec.attribution_for("gp")
    assert spec.source_names == ("gp",)
    assert spec.source("gp").source_value == "ai4chem_abstract_pairs"
    assert "BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs" in gp
    assert "CC BY 4.0" not in gp
    assert "does not establish a verified license" in gp
    with pytest.raises(KeyError, match="unknown source"):
        spec.source("epo")


def test_chemistry_input_source_and_supported_languages_are_explicit(spec: DomainSpec) -> None:
    assert set(spec.languages.inventory) == {"en", "de", "fr", "es", "zh", "ja", "ru"}
    assert spec.languages.inventory == spec.languages.working == spec.source("gp").languages
    assert spec.defaults["input_dataset_repo"] == (
        "BASF-AI/ai4chem-clir-google-patents-within-document-abstract-pairs"
    )
    assert spec.defaults["corpus_repo"] != spec.defaults["input_dataset_repo"]
    assert spec.qac_validate_corpus is not None
    assert spec.schema.parallel_ids_field == "paired_document_ids"
    assert {"upstream_dataset", "upstream_revision", "upstream_corpus_id",
            "upstream_group_key", "accepted_pair_ids", "paired_document_ids"} <= set(spec.schema.fields)


def test_working_languages_have_names_and_prompts(spec: DomainSpec) -> None:
    from clir_bench.core.prompts import PromptPack

    pack = PromptPack(package=spec.prompts_package)
    for language in spec.languages.working:
        assert spec.languages.name_of(language) != language, f"no display name for {language}"
    for mode in spec.analysis.modes:
        available = pack.available_languages(mode)
        missing = set(spec.languages.working) - set(available)
        assert not missing, f"mode {mode} has no generation prompt for {sorted(missing)}"


def test_schema_round_trips_document_ids(spec: DomainSpec) -> None:
    schema = spec.schema
    doc_id = schema.make_doc_id("EP-3686982-A1", "de")
    assert doc_id == "EP-3686982-A1_de"
    assert schema.language_from_doc_id(doc_id) == "de"
    assert schema.family_from_doc_id(doc_id) == "EP-3686982-A1"


def test_language_inference_ignores_ids_without_a_language_suffix(spec: DomainSpec) -> None:
    """Shared-haystack documents may not encode a language; that must not crash."""
    assert spec.schema.language_from_doc_id("EP-3686982-A1") == ""
    assert spec.schema.family_from_doc_id("EP-3686982-A1") == "EP-3686982-A1"


def test_text_falls_through_an_empty_field(spec: DomainSpec) -> None:
    """A present-but-empty context must fall through to the abstract.

    The old English-first pipeline used dict-default chaining here, so an empty
    context skipped a non-empty abstract and landed on the title.
    """
    row = {"context": "", "abstract": "An abstract.", "title": "A title"}
    assert spec.schema.text_of(row) == "An abstract."


def test_dedup_key_normalizes_across_source_formats(spec: DomainSpec) -> None:
    """The two sources format publication numbers differently.

    This normalization is what keeps the corpora disjoint; without it the same
    patent appears twice under two spellings.
    """
    from_gp = {"publication_number": "EP-3686982-A1", "country_code": "EP"}
    from_epo = {"publication_number": "3686982", "country_code": "EP"}
    assert spec.schema.dedup_key(from_gp) == spec.schema.dedup_key(from_epo)


def test_data_layout_resolves_under_the_data_root(spec: DomainSpec) -> None:
    from pathlib import Path

    from clir_bench.core.paths import Workspace

    workspace = Workspace.build(
        data_dir=Path("/tmp/data"), reports_dir=Path("/tmp/reports"), domain=spec
    )
    # Each domain owns a subtree named after it, so data/chemistry and a future
    # data/legal never collide.
    assert workspace.domain_data_dir == Path("/tmp/data") / spec.name
    assert workspace.data("gp_corpus") == Path(
        f"/tmp/data/{spec.name}/ai4chem_abstract_pairs/multilingual_corpus.csv"
    )
    assert workspace.corpus_csv("gp") == Path(
        f"/tmp/data/{spec.name}/ai4chem_abstract_pairs/multilingual_corpus.csv"
    )
    assert workspace.data("gp_pairs").name == "accepted_pairs.csv"
    assert workspace.data("gp_manifest").name == "manifest.json"
    with pytest.raises(KeyError):
        workspace.data("no_such_key")


def test_declared_plans_are_callable(spec: DomainSpec) -> None:
    assert spec.qac_plans, "a domain with no generation plan cannot build a benchmark"
    for name, builder in spec.qac_plans.items():
        assert callable(builder), f"plan {name} is not callable"


def _context(tmp_path, domain):
    from clir_bench.core.config import Settings
    from clir_bench.core.context import AppContext

    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        cluster_dir=tmp_path / "cluster",
    )
    return AppContext.build(settings, domains.load(domain))


def test_legal_sources_keep_runs_and_structured_inputs_in_their_own_directories(tmp_path):
    context = _context(tmp_path, "legal")
    assert context.domain.source_names == ("eurlex", "un")
    assert context.workspace.qac_dir("eurlex") == tmp_path / "data/legal/eurlex/qac"
    assert context.workspace.qac_dir("un") == tmp_path / "data/legal/un_parallel/qac"
    assert context.workspace.data("eurlex_structure") == tmp_path / "data/legal/eurlex/structure"
    assert context.workspace.data("un_blocks") == tmp_path / "data/legal/un_parallel/blocks"
    assert context.setting("generation_model") == "gpt-5.6-luna"
    assert context.setting("verifier_model") == "google/gemini-3.8-flash"
    for source in context.domain.sources:
        assert context.domain.attribution_for(source.name)
        assert context.workspace.corpus_csv(source).suffix == ".csv"


@pytest.mark.parametrize(
    "command,options",
    [
        (
            "generate",
            [
                "--source",
                "eurlex",
                "un",
                "--generation-model",
                "provider/first",
                "--generation-model",
                "provider/second",
                "--questions",
                "30",
            ],
        ),
        ("regrade", ["--input", "previous.csv"]),
        ("best", ["--input", "previous.csv"]),
    ],
)
def test_legal_qac_dispatches_to_existing_domain_pipeline(tmp_path, monkeypatch, command, options):
    from clir_bench.cli import build_parser
    from clir_bench.domains.legal import qac

    context = _context(tmp_path, "legal")
    captured = []

    def run(args, received_context):
        captured.append((args, received_context))
        return 7

    monkeypatch.setattr(qac, command, run, raising=False)
    parser = build_parser(context, domains.load_module("legal"))
    args = parser.parse_args(["qac", command, *options])
    assert args.handler(args, context) == 7
    assert captured == [(args, context)]
    if command == "generate":
        assert args.source == ["eurlex", "un"]
        assert args.generation_model == ["provider/first", "provider/second"]
        assert args.questions == 30
        assert args.workers == 6
        assert args.retries is None
        assert args.keep is None
    elif command == "regrade":
        assert args.source is None
        assert args.workers == 6
        assert args.retries is None


@pytest.mark.parametrize(
    "options,error",
    [
        (["--append"], "--append is not supported"),
        (["--resume"], "--resume requires --run-dir"),
    ],
)
def test_legal_qac_rejects_ambiguous_run_writes(tmp_path, options, error):
    from clir_bench.cli import build_parser

    context = _context(tmp_path, "legal")
    parser = build_parser(context, domains.load_module("legal"))
    args = parser.parse_args(["qac", "generate", "--source", "un", *options])
    with pytest.raises(ValueError, match=error):
        args.handler(args, context)


def test_legal_qac_accepts_run_and_replay_arguments(tmp_path):
    from pathlib import Path

    from clir_bench.cli import build_parser

    context = _context(tmp_path, "legal")
    parser = build_parser(context, domains.load_module("legal"))
    args = parser.parse_args(
        [
            "qac",
            "generate",
            "--source",
            "un",
            "--run-dir",
            "runs/example",
            "--targets-from",
            "runs/previous",
            "--output",
            "comparison.csv",
            "--resume",
            "--modes",
            "semantic",
            "--langs",
            "en",
            "fr",
            "--dry-run",
        ]
    )
    assert args.run_dir == Path("runs/example")
    assert args.targets_from == Path("runs/previous")
    assert args.output == Path("comparison.csv")
    assert args.modes == ["semantic"]
    assert args.langs == ["en", "fr"]
    assert args.resume and args.dry_run


def test_chemistry_qac_keeps_existing_argument_types_and_defaults(tmp_path):
    from clir_bench.cli import build_parser
    from clir_bench.cli.qac import _generate

    context = _context(tmp_path, "chemistry")
    parser = build_parser(context, domains.load_module("chemistry"))
    args = parser.parse_args(
        [
            "qac",
            "generate",
            "--source",
            "gp",
            "--generation-model",
            "provider/generator",
        ]
    )
    assert args.source == "gp"
    assert args.generation_model == "provider/generator"
    assert args.workers == 1
    assert args.plan == "balanced"
    assert args.handler is _generate
    assert not hasattr(args, "run_dir")


@pytest.mark.parametrize("domain", ["legal", "chemistry"])
def test_qac_help_only_advertises_supported_selection_flags(tmp_path, capsys, domain):
    from clir_bench.cli import build_parser

    context = _context(tmp_path, domain)
    parser = build_parser(context, domains.load_module(domain))
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["qac", "generate", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--plan", "--pool", "--priority-langs", "--append", "--exclude-from"):
        assert (flag in help_text) == (domain == "chemistry")
    assert "--generation-model" in help_text


@pytest.fixture
def legal_run(tmp_path, monkeypatch):
    """Exercise real orchestration and SQLite checkpoints with local batch doubles."""
    from collections import Counter
    from dataclasses import dataclass
    from types import SimpleNamespace

    from clir_bench.cli import build_parser
    from clir_bench.domains.legal import qac

    @dataclass
    class Target:
        language: str
        mode: str
        eli_id: str = ""
        celex_id: str = ""
        article_number: str = "1"
        doc_id: str = ""
        block_id: str = ""

    state = SimpleNamespace(
        selections=[],
        payloads=[],
        invocations=[],
        stages=Counter(),
        prompt_version="one",
        rewrite_questions=False,
        fail_batch=False,
    )
    context = _context(tmp_path, "legal")
    parser = build_parser(context, domains.load_module("legal"))
    indexes = {source: object() for source in ("eurlex", "un")}

    class Prompts:
        def faithfulness(self, arity):
            return "Faithfulness " + state.prompt_version

        def quality(self, mode, arity):
            return "Quality " + mode + " " + state.prompt_version

        def generation(self, mode, language):
            return "Generation " + mode + " " + language

    def batch(source):
        modes = ("lookup", "fact_pattern") if source == "eurlex" else ("technical", "semantic")

        def select(index, *, n, seed, languages, modes):
            assert index is indexes[source]
            state.selections.append((source, n, tuple(languages), tuple(modes)))
            return [
                Target(
                    language=languages[position % len(languages)],
                    mode=modes[position % len(modes)],
                    eli_id=f"eli/{position}" if source == "eurlex" else "",
                    celex_id=f"act/{position}" if source == "eurlex" else "",
                    doc_id=f"doc/{position}" if source == "un" else "",
                    block_id=f"doc/{position}#0" if source == "un" else "",
                )
                for position in range(n)
            ]

        def prepare_payload(target, index, **limits):
            assert index is indexes[source]
            return SimpleNamespace(
                text=f"{source}:{target.eli_id or target.block_id}:{target.language}:{limits}"
            )

        def run_one(
            target,
            index,
            *,
            gen_model,
            grade_model,
            existing_candidates,
            checkpoint,
            payload,
            keep,
            **kwargs,
        ):
            assert index is indexes[source]
            state.invocations.append((source, gen_model, existing_candidates is not None))
            state.payloads.append((source, gen_model, target, payload))
            if state.fail_batch:
                raise RuntimeError("synthetic provider failure")

            def stage(name, value):
                def execute():
                    state.stages[name] += 1
                    return value

                return checkpoint.run(name, execute)

            target_id = target.eli_id or target.block_id
            candidates = existing_candidates
            if candidates is None:
                candidates = stage(
                    "generation",
                    [
                        {
                            "candidate_id": f"{source}:{target_id}:{gen_model}:{position}",
                            "question": f"What does clause {position} require?",
                            "answer": f"Rule {position}.",
                        }
                        for position in range(2)
                    ],
                )
            stage("faithfulness", "faithful")
            stage("quality", "good")
            return [
                {
                    "corpus": source,
                    "document_id": target.celex_id or target.doc_id,
                    "target_id": target_id,
                    "question_language": target.language,
                    "mode": target.mode,
                    "candidate_id": row["candidate_id"],
                    "generator_model_id": gen_model,
                    "question": "unexpected rewrite"
                    if state.rewrite_questions
                    else row["question"],
                    "answer": "unexpected rewrite" if state.rewrite_questions else row["answer"],
                    "document_text": payload.text,
                    "total_score": 37 - position,
                    "faith_grounding": 5,
                    "qual_current": 4,
                    "grading_status": "completed",
                    "grading_error": "",
                    "faithfulness_verifier_model": grade_model,
                    "quality_verifier_model": grade_model,
                }
                for position, row in enumerate(candidates[:keep])
            ]

        def target_from_rows(rows):
            row = rows[0]
            return Target(
                language=row["question_language"],
                mode=row["mode"],
                eli_id=row["target_id"] if source == "eurlex" else "",
                celex_id=row["document_id"] if source == "eurlex" else "",
                doc_id=row["document_id"] if source == "un" else "",
                block_id=row["target_id"] if source == "un" else "",
            )

        return SimpleNamespace(
            Target=Target,
            gen=SimpleNamespace(MODES=modes, DEFAULT_MODES=modes, PROMPTS=Prompts()),
            DEFAULT_MODES=modes,
            SUPPORTED_MODES=modes,
            ACT_LANGUAGES=("en", "de"),
            UN_LANGUAGES=("en", "zh"),
            select=select,
            prepare_payload=prepare_payload,
            run_one=run_one,
            target_from_rows=target_from_rows,
        )

    batches = {source: batch(source) for source in indexes}
    monkeypatch.setattr(qac, "_batches", lambda: batches)
    monkeypatch.setattr(
        qac,
        "_indexes",
        lambda sources, supplied=None, context=None: {s: indexes[s] for s in sources},
    )

    def parse(command, *options):
        return parser.parse_args(["qac", command, *map(str, options)])

    def generate(directory, *options):
        args = parse(
            "generate",
            "--source",
            "eurlex",
            "--questions",
            "1",
            "--langs",
            "en",
            "--generation-model",
            "provider/first",
            "--generation-model",
            "provider/second",
            "--run-dir",
            directory,
            *options,
        )
        return args.handler(args, context)

    return SimpleNamespace(
        qac=qac,
        context=context,
        state=state,
        parse=parse,
        generate=generate,
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("source,subdirectory", [("eurlex", "eurlex"), ("un", "un_parallel")])
def test_legal_generation_defaults_to_its_source_run_directory(legal_run, monkeypatch, source, subdirectory):
    from clir_bench.core import runs

    run = legal_run
    monkeypatch.setattr(runs, "make_run_id", lambda: "generation-run")
    args = run.parse("generate", "--source", source, "--questions", 1, "--langs", "en")

    assert args.handler(args, run.context) == 0

    directory = run.tmp_path / "data/legal" / subdirectory / "qac/generation-run"
    assert {path.name for path in directory.iterdir()} == {"results.csv", "run.sqlite"}
    rows, _ = run.qac._read_csv(directory / "results.csv")
    assert {row["corpus"] for row in rows} == {source}
    assert not (run.tmp_path / "data/legal/qac").exists()


def test_un_generation_defaults_include_arabic_and_russian(legal_run, monkeypatch):
    from clir_bench.domains.legal.un import UN_LANGUAGES

    run = legal_run
    monkeypatch.setattr(run.qac._batches()["un"], "UN_LANGUAGES", UN_LANGUAGES)
    selections = run.qac._select(["un"], run.qac._indexes(["un"]), {
        "langs": None, "modes": None, "questions": 6, "seed": 42,
    })
    assert run.state.selections[0][2] == UN_LANGUAGES
    assert {entry["target"]["language"] for entry in selections} == set(UN_LANGUAGES)


@pytest.mark.parametrize("source,subdirectory", [("eurlex", "eurlex"), ("un", "un_parallel")])
def test_legal_regrade_infers_its_source_run_directory_from_csv(legal_run, monkeypatch, source, subdirectory):
    from clir_bench.core import runs

    run = legal_run
    original = run.tmp_path / "original"
    args = run.parse(
        "generate", "--source", source, "--questions", 1, "--langs", "en", "--run-dir", original
    )
    assert args.handler(args, run.context) == 0
    previous_generations = run.state.stages["generation"]
    monkeypatch.setattr(runs, "make_run_id", lambda: "regrade-run")
    args = run.parse("regrade", "--input", original / "results.csv")
    assert args.source is None

    assert args.handler(args, run.context) == 0

    directory = run.tmp_path / "data/legal" / subdirectory / "qac/regrade-run"
    assert {path.name for path in directory.iterdir()} == {"results.csv", "run.sqlite"}
    rows, _ = run.qac._read_csv(directory / "results.csv")
    assert {row["corpus"] for row in rows} == {source}
    assert run.state.stages["generation"] == previous_generations
    assert not (run.tmp_path / "data/legal/qac").exists()


@pytest.mark.parametrize("source", ["eurlex", "un"])
def test_legal_source_runs_honor_explicit_run_directories(legal_run, source):
    run = legal_run
    original = run.tmp_path / "custom-generation"
    args = run.parse(
        "generate", "--source", source, "--questions", 1, "--langs", "en", "--run-dir", original,
        "--output", "questions.csv",
    )
    assert args.handler(args, run.context) == 0
    assert {path.name for path in original.iterdir()} == {"questions.csv", "run.sqlite"}

    regraded = run.tmp_path / "custom-regrade"
    args = run.parse("regrade", "--input", original / "questions.csv", "--run-dir", regraded)
    assert args.handler(args, run.context) == 0
    assert {path.name for path in regraded.iterdir()} == {"results.csv", "run.sqlite"}
    assert not (run.tmp_path / "data/legal").exists()


@pytest.mark.parametrize("custom_parent", [False, True])
def test_legal_multi_source_runs_share_inputs_and_resume_independently(
    legal_run, monkeypatch, custom_parent
):
    from collections import Counter, defaultdict

    from clir_bench.core import runs
    from clir_bench.domains.legal.qac.batch_recording import read_run_metadata

    run = legal_run
    monkeypatch.setattr(runs, "make_run_id", lambda: "shared-run")
    parent = run.tmp_path / "comparison"
    options = ("--run-dir", parent) if custom_parent else ()
    args = run.parse(
        "generate", "--source", "eurlex", "un", "--questions", 2, "--langs", "en",
        "--generation-model", "provider/first", "--generation-model", "provider/second",
        *options,
    )
    assert args.handler(args, run.context) == 0
    directories = {
        source: parent / subdirectory if custom_parent else (
            run.tmp_path / "data/legal" / subdirectory / "qac/shared-run"
        )
        for source, subdirectory in (("eurlex", "eurlex"), ("un", "un_parallel"))
    }
    candidates = []
    originals = {}
    for source, directory in directories.items():
        assert {path.name for path in directory.iterdir()} == {"results.csv", "run.sqlite"}
        rows, _ = run.qac._read_csv(directory / "results.csv")
        assert len(rows) == 4
        assert {row["corpus"] for row in rows} == {source}
        assert Counter(row["generator_model_id"] for row in rows) == {
            "provider/first": 2, "provider/second": 2,
        }
        candidates.extend(rows)
        metadata = read_run_metadata(directory)
        assert metadata["status"] == "completed"
        assert metadata["summary"]["candidates"] == 4
        assert {entry["corpus"] for entry in metadata["targets"]} == {source}
        originals[source] = (directory / "results.csv").read_bytes()
    assert len({row["candidate_id"] for row in candidates}) == 8
    assert not (run.tmp_path / "data/legal/qac").exists()
    if custom_parent:
        assert {path.name for path in parent.iterdir()} == {"eurlex", "un_parallel"}
    else:
        assert len({directory.name for directory in directories.values()}) == 1
    by_target = defaultdict(list)
    for source, model, target, payload in run.state.payloads:
        by_target[source, target.eli_id or target.block_id].append((model, payload))
    assert len(by_target) == 2
    for paired in by_target.values():
        assert len(paired) == 2
        assert paired[0][1] is paired[1][1]
    assert run.state.stages == {"generation": 4, "faithfulness": 4, "quality": 4}
    for source, directory in directories.items():
        args = run.parse("generate", "--source", source, "--run-dir", directory, "--resume")
        assert args.handler(args, run.context) == 0
        assert (directory / "results.csv").read_bytes() == originals[source]
    assert run.state.stages == {"generation": 4, "faithfulness": 4, "quality": 4}
    assert len(run.state.selections) == 2
    if custom_parent:
        args = run.parse("generate", "--source", "eurlex", "un", "--run-dir", parent, "--resume")
        assert args.handler(args, run.context) == 0
        assert run.state.stages == {"generation": 4, "faithfulness": 4, "quality": 4}
        for source, directory in directories.items():
            assert (directory / "results.csv").read_bytes() == originals[source]

    reused = run.tmp_path / "reused-targets"
    args = run.parse(
        "generate", "--source", "eurlex", "un", "--targets-from", directories["eurlex"],
        "--run-dir", reused, "--generation-model", "provider/third",
    )
    assert args.handler(args, run.context) == 0
    assert len(run.state.selections) == 2
    for source, subdirectory in (("eurlex", "eurlex"), ("un", "un_parallel")):
        rows, _ = run.qac._read_csv(reused / subdirectory / "results.csv")
        assert len(rows) == 2
        assert {row["corpus"] for row in rows} == {source}
        assert {row["generator_model_id"] for row in rows} == {"provider/third"}
        assert {row["source_payload_sha256"] for row in rows} == {
            row["source_payload_sha256"] for row in candidates if row["corpus"] == source
        }


@pytest.mark.parametrize("custom_parent", [False, True])
def test_legal_regrade_splits_combined_csv_into_source_runs(
    legal_run, monkeypatch, custom_parent
):
    from clir_bench.core import runs

    run = legal_run
    original_rows = []
    for source in ("eurlex", "un"):
        original = run.tmp_path / ("original-" + source)
        args = run.parse(
            "generate", "--source", source, "--questions", 1, "--langs", "en",
            "--generation-model", "provider/first", "--generation-model", "provider/second",
            "--run-dir", original,
        )
        assert args.handler(args, run.context) == 0
        rows, _ = run.qac._read_csv(original / "results.csv")
        original_rows.extend(rows)
    combined = run.tmp_path / "legacy-combined.csv"
    run.qac._write_csv(combined, original_rows)
    input_bytes = combined.read_bytes()
    previous_generations = run.state.stages["generation"]
    monkeypatch.setattr(runs, "make_run_id", lambda: "regraded-run")
    parent = run.tmp_path / "regraded"
    options = ("--run-dir", parent) if custom_parent else ()
    args = run.parse(
        "regrade", "--input", combined, "--verifier-model", "provider/new-judge", *options
    )
    assert args.handler(args, run.context) == 0
    for source, subdirectory in (("eurlex", "eurlex"), ("un", "un_parallel")):
        directory = parent / subdirectory if custom_parent else (
            run.tmp_path / "data/legal" / subdirectory / "qac/regraded-run"
        )
        assert {path.name for path in directory.iterdir()} == {"results.csv", "run.sqlite"}
        rows, _ = run.qac._read_csv(directory / "results.csv")
        before = [row for row in original_rows if row["corpus"] == source]
        assert len(rows) == len(before) == 4
        assert {row["corpus"] for row in rows} == {source}
        assert {row["generator_model_id"] for row in rows} == {"provider/first", "provider/second"}
        for expected, actual in zip(before, rows):
            for field in ("candidate_id", "question", "answer", "document_text", "source_payload_sha256"):
                assert actual[field] == expected[field]
            assert actual["faithfulness_verifier_model"] == "provider/new-judge"
            assert actual["quality_verifier_model"] == "provider/new-judge"
            assert actual["regrade_status"] == "completed"
    assert all(invocation[2] for invocation in run.state.invocations[-4:])
    assert run.state.stages["generation"] == previous_generations
    assert combined.read_bytes() == input_bytes
    assert not (run.tmp_path / "data/legal/qac").exists()


@pytest.mark.parametrize("source", ["eurlex", "un"])
def test_legal_new_regrade_uses_current_judge_default_and_resume_keeps_it(legal_run, source):
    from dataclasses import replace

    from clir_bench.domains.legal.qac.batch_recording import read_run_metadata

    run = legal_run
    original = run.tmp_path / "old-judge"
    args = run.parse(
        "generate", "--source", source, "--questions", 1, "--langs", "en",
        "--run-dir", original, "--verifier-model", "provider/old-judge",
        "--max-references", 2, "--context-chars", 1234,
    )
    assert args.handler(args, run.context) == 0
    previous_generations = run.state.stages["generation"]
    output = run.tmp_path / "current-judge"
    args = run.parse("regrade", "--input", original / "results.csv", "--run-dir", output)
    assert args.handler(args, run.context) == 0
    current_judge = run.context.setting("verifier_model")
    metadata = read_run_metadata(output)
    assert metadata["config"]["verifier_model"] == current_judge
    assert metadata["config"]["max_references"] == 2
    assert metadata["config"]["context_chars"] == 1234
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert {row["faithfulness_verifier_model"] for row in rows} == {current_judge}
    assert {row["quality_verifier_model"] for row in rows} == {current_judge}
    assert run.state.stages["generation"] == previous_generations

    changed_context = replace(
        run.context, domain_settings=dict(run.context.domain_settings, verifier_model="provider/future-judge")
    )
    stages_before_resume = dict(run.state.stages)
    args = run.parse("regrade", "--input", original / "results.csv", "--run-dir", output, "--resume")
    assert args.handler(args, changed_context) == 0
    assert read_run_metadata(output)["config"]["verifier_model"] == current_judge
    assert dict(run.state.stages) == stages_before_resume


def test_legal_regrade_preserves_candidates_and_clears_obsolete_grades(legal_run):
    run = legal_run
    original = run.tmp_path / "original"
    assert run.generate(original) == 0
    rows, fields = run.qac._read_csv(original / "results.csv")
    for row in rows:
        row.update(qual_obsolete="5", quality_audit_status="old", collection_note="preserve this")
    rows[0]["question"] = 'Which rule applies to "équipe"?\nUse the supplied text.'
    run.qac._write_csv(original / "results.csv", rows, fields)
    input_bytes = (original / "results.csv").read_bytes()
    previous_generations = run.state.stages["generation"]
    run.state.rewrite_questions = True
    output = run.tmp_path / "regraded"
    args = run.parse(
        "regrade",
        "--input",
        original / "results.csv",
        "--run-dir",
        output,
        "--verifier-model",
        "provider/new-judge",
    )
    assert args.handler(args, run.context) == 0
    regraded, _ = run.qac._read_csv(output / "results.csv")
    assert len(regraded) == len(rows)
    assert run.state.stages["generation"] == previous_generations
    assert all(invocation[2] for invocation in run.state.invocations[-2:])
    for before, after in zip(rows, regraded):
        for key in (
            "candidate_id",
            "question",
            "answer",
            "generator_model_id",
            "generator_model_name",
            "target_id",
            "document_text",
            "collection_note",
        ):
            assert after[key] == before[key]
        assert after["qual_obsolete"] == ""
        assert after["quality_audit_status"] == ""
        assert after["qual_current"] == "4"
        assert after["faithfulness_verifier_model"] == "provider/new-judge"
        assert after["quality_verifier_model"] == "provider/new-judge"
        assert after["regrade_status"] == "completed"
    assert (original / "results.csv").read_bytes() == input_bytes
    assert {path.name for path in output.iterdir()} == {"results.csv", "run.sqlite"}


def test_legal_regrade_preserves_rows_when_a_batch_fails(legal_run):
    run = legal_run
    original = run.tmp_path / "original"
    assert run.generate(original) == 0
    rows, _ = run.qac._read_csv(original / "results.csv")
    run.state.fail_batch = True
    output = run.tmp_path / "failed-regrade"
    args = run.parse("regrade", "--input", original / "results.csv", "--run-dir", output)
    assert args.handler(args, run.context) == 1
    failed, _ = run.qac._read_csv(output / "results.csv")
    assert [row["candidate_id"] for row in failed] == [row["candidate_id"] for row in rows]
    assert all(row["grading_status"] == "failed" and row["total_score"] == "" for row in failed)
    assert all(row["faith_grounding"] == "" and row["qual_current"] == "" for row in failed)


def test_legal_resume_refuses_changed_prompts_without_touching_results(legal_run):
    run = legal_run
    directory = run.tmp_path / "comparison"
    assert run.generate(directory) == 0
    original = (directory / "results.csv").read_bytes()
    before_calls = run.state.stages.copy()
    run.state.prompt_version = "two"
    args = run.parse("generate", "--source", "eurlex", "--run-dir", directory, "--resume")
    with pytest.raises(ValueError, match="prompts.*changed"):
        args.handler(args, run.context)
    assert run.state.stages == before_calls
    assert (directory / "results.csv").read_bytes() == original


def test_legal_regrade_rejects_missing_generator_identity_before_calls(legal_run):
    run = legal_run
    original = run.tmp_path / "original"
    assert run.generate(original) == 0
    rows, fields = run.qac._read_csv(original / "results.csv")
    rows[0]["generator_model_id"] = ""
    run.qac._write_csv(original / "results.csv", rows, fields)
    calls = run.state.stages.copy()
    output = run.tmp_path / "invalid"
    args = run.parse("regrade", "--input", original / "results.csv", "--run-dir", output)
    with pytest.raises(ValueError, match="generator identity"):
        args.handler(args, run.context)
    assert run.state.stages == calls
    assert not output.exists()


def test_legal_dry_run_does_not_call_models_or_create_files(legal_run):
    directory = legal_run.tmp_path / "preview"
    assert legal_run.generate(directory, "--dry-run") == 0
    assert not directory.exists()
    assert not legal_run.state.stages


def test_legal_selection_rejects_unknown_language_instead_of_dropping_it(legal_run):
    directory = legal_run.tmp_path / "invalid"
    with pytest.raises(ValueError, match="unsupported language"):
        legal_run.generate(directory, "--langs", "en", "xx")
    assert not legal_run.state.stages
    assert not directory.exists()


def test_legal_targets_from_reuses_context_limits_and_avoids_resampling(legal_run):
    run = legal_run
    original = run.tmp_path / "original"
    assert run.generate(original, "--context-chars", "12345", "--max-references", "2") == 0
    calls = len(run.state.selections)
    reused = run.tmp_path / "reused"
    args = run.parse(
        "generate",
        "--source",
        "eurlex",
        "--targets-from",
        original,
        "--run-dir",
        reused,
        "--generation-model",
        "provider/third",
    )
    assert args.handler(args, run.context) == 0
    rows, _ = run.qac._read_csv(reused / "results.csv")
    original_rows, _ = run.qac._read_csv(original / "results.csv")
    assert len(run.state.selections) == calls
    assert len(rows) == 2
    assert {row["generator_model_id"] for row in rows} == {"provider/third"}
    assert {row["source_payload_sha256"] for row in rows} == {
        row["source_payload_sha256"] for row in original_rows
    }


def test_legal_mode_quotas_advance_short_batches_and_resume_without_extra_calls(legal_run, monkeypatch):
    from collections import Counter

    from clir_bench.domains.legal.qac.batch_recording import RunState, read_run_metadata

    run = legal_run
    original, output = run.tmp_path / "pool", run.tmp_path / "quota"
    assert run.generate(original, "--questions", 8) == 0
    prior_selections = len(run.state.selections)
    batch = run.qac._batches()["eurlex"]
    run_one, visited = batch.run_one, []

    def variable_batch(target, *args, **kwargs):
        visited.append((target.mode, target.eli_id, kwargs["gen_model"]))
        if target.eli_id == "eli/0":
            return []
        rows = run_one(target, *args, **kwargs)
        return rows[:1] if target.eli_id == "eli/2" else rows[::-1]

    monkeypatch.setattr(batch, "run_one", variable_batch)
    args = run.parse(
        "generate", "--source", "eurlex", "--questions-per-mode", 3,
        "--targets-from", original, "--run-dir", output, "--workers", 4,
        "--generation-model", "provider/first", "--generation-model", "provider/second",
    )
    assert args.handler(args, run.context) == 0
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert Counter((row["mode"], row["generator_model_id"]) for row in rows) == {
        (mode, model): 3 for mode in ("lookup", "fact_pattern")
        for model in ("provider/first", "provider/second")
    }
    assert {target for _, target, _ in visited} == {f"eli/{i}" for i in range(5)}
    assert len(visited) == 10
    assert len(run.state.selections) == prior_selections
    assert all(row["total_score"] == "37" for row in rows if row["target_id"] == "eli/3")
    metadata = read_run_metadata(output)
    assert metadata["summary"]["generated_candidates"] == 14
    assert metadata["summary"]["candidates"] == 12
    assert all(item["shortage"] == 0 for item in metadata["summary"]["quotas"])
    with RunState(output / "run.sqlite") as state:
        assert sum(len(record["rows"]) for record in state.outcomes()) == 14
    before = (output / "results.csv").read_bytes()
    stage_counts = run.state.stages.copy()
    args = run.parse("generate", "--source", "eurlex", "--run-dir", output, "--resume")
    assert args.handler(args, run.context) == 0
    assert len(visited) == 10
    assert run.state.stages == stage_counts
    assert (output / "results.csv").read_bytes() == before


def test_legal_quota_records_shortages_and_retains_questions_with_failed_grades(legal_run, monkeypatch):
    from clir_bench.domains.legal.qac.batch_recording import read_run_metadata

    run = legal_run
    original, output = run.tmp_path / "pool", run.tmp_path / "quota"
    assert run.generate(original, "--questions", 2, "--modes", "lookup") == 0
    batch = run.qac._batches()["eurlex"]
    run_one = batch.run_one

    def failing_batch(target, *args, **kwargs):
        if target.eli_id == "eli/0":
            raise RuntimeError("generation unavailable")
        rows = run_one(target, *args, **kwargs)
        for row in rows:
            row.update(grading_status="failed", grading_error="judge unavailable", total_score=None)
        return rows

    monkeypatch.setattr(batch, "run_one", failing_batch)
    args = run.parse("generate", "--source", "eurlex", "--questions-per-mode", 5,
                     "--targets-from", original, "--run-dir", output)
    assert args.handler(args, run.context) == 1
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert len(rows) == 2
    assert all(row["grading_status"] == "failed" and row["total_score"] == "" for row in rows)
    metadata = read_run_metadata(output)
    assert metadata["status"] == "completed_with_errors"
    assert metadata["summary"]["failed"] == 2
    assert metadata["summary"]["quotas"][0]["shortage"] == 3


def test_legal_quota_resumes_partial_group_without_regenerating_finished_targets(legal_run, monkeypatch):
    run = legal_run
    original, output = run.tmp_path / "pool", run.tmp_path / "quota"
    assert run.generate(original, "--questions", 4, "--modes", "lookup") == 0
    batch = run.qac._batches()["eurlex"]
    run_one, visited = batch.run_one, []
    interrupt = True

    def interrupted_batch(target, *args, **kwargs):
        visited.append(target.eli_id)
        if interrupt and target.eli_id == "eli/1":
            raise KeyboardInterrupt()
        return run_one(target, *args, **kwargs)

    monkeypatch.setattr(batch, "run_one", interrupted_batch)
    args = run.parse("generate", "--source", "eurlex", "--questions-per-mode", 5,
                     "--targets-from", original, "--run-dir", output, "--workers", 1)
    with pytest.raises(KeyboardInterrupt):
        args.handler(args, run.context)
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert len(rows) == 2
    interrupt = False
    args = run.parse("generate", "--source", "eurlex", "--run-dir", output, "--resume")
    assert args.handler(args, run.context) == 0
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert len(rows) == 5
    assert visited == ["eli/0", "eli/1", "eli/1", "eli/2"]


def test_legal_quotas_share_rounds_across_models_and_finish_interrupted_round(legal_run, monkeypatch):
    from collections import Counter

    from clir_bench.domains.legal.qac.batch_recording import RunState, read_run_metadata

    run = legal_run
    original, output = run.tmp_path / "pool", run.tmp_path / "quota"
    assert run.generate(original, "--questions", 4, "--modes", "lookup") == 0
    batch = run.qac._batches()["eurlex"]
    run_one, visited = batch.run_one, []
    interrupt = True

    def uneven_batch(target, *args, **kwargs):
        model = kwargs["gen_model"]
        visited.append((target.eli_id, model))
        if interrupt and model == "provider/second":
            raise KeyboardInterrupt()
        rows = run_one(target, *args, **kwargs)
        return rows[:1] if model == "provider/second" else rows

    monkeypatch.setattr(batch, "run_one", uneven_batch)
    args = run.parse("generate", "--source", "eurlex", "--questions-per-mode", 2,
                     "--targets-from", original, "--run-dir", output, "--workers", 1,
                     "--generation-model", "provider/first", "--generation-model", "provider/second")
    with pytest.raises(KeyboardInterrupt):
        args.handler(args, run.context)
    interrupt = False
    args = run.parse("generate", "--source", "eurlex", "--run-dir", output, "--resume", "--workers", 2)
    assert args.handler(args, run.context) == 0
    assert Counter(visited) == {
        ("eli/0", "provider/first"): 1, ("eli/0", "provider/second"): 2,
        ("eli/1", "provider/first"): 1, ("eli/1", "provider/second"): 1,
    }
    with RunState(output / "run.sqlite") as state:
        all_rows = [row for outcome in state.outcomes() for row in outcome["rows"]]
    assert {row["target_id"] for row in all_rows if row["generator_model_id"] == "provider/first"} == {
        row["target_id"] for row in all_rows if row["generator_model_id"] == "provider/second"
    } == {"eli/0", "eli/1"}
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert Counter(row["generator_model_id"] for row in rows) == {"provider/first": 2, "provider/second": 2}
    summary = read_run_metadata(output)["summary"]
    assert summary["generated_candidates"] == 6
    assert summary["candidates"] == 4


def test_legal_quota_batches_necessary_targets_concurrently(legal_run, monkeypatch):
    from collections import Counter
    from threading import Event

    run = legal_run
    original, output = run.tmp_path / "pool", run.tmp_path / "quota"
    assert run.generate(original, "--questions", 4, "--modes", "lookup") == 0
    batch = run.qac._batches()["eurlex"]
    run_one, visited, later_started = batch.run_one, [], Event()

    def slow_first_target(target, *args, **kwargs):
        visited.append((target.eli_id, kwargs["gen_model"]))
        if target.eli_id == "eli/0" and kwargs["gen_model"] == "provider/first":
            assert later_started.wait(timeout=2), "later necessary targets were blocked by the first target"
        if target.eli_id == "eli/1":
            later_started.set()
        return run_one(target, *args, **kwargs)

    monkeypatch.setattr(batch, "run_one", slow_first_target)
    args = run.parse("generate", "--source", "eurlex", "--questions-per-mode", 5,
                     "--targets-from", original, "--run-dir", output, "--workers", 4,
                     "--generation-model", "provider/first", "--generation-model", "provider/second")
    assert args.handler(args, run.context) == 0
    assert Counter(visited) == {
        (f"eli/{position}", model): 1 for position in range(3)
        for model in ("provider/first", "provider/second")
    }
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert Counter(row["generator_model_id"] for row in rows) == {"provider/first": 5, "provider/second": 5}


def test_legal_quota_resume_finishes_started_targets_beyond_filled_quota(legal_run):
    from clir_bench.domains.legal.qac.batch_recording import RunState, read_run_metadata

    run = legal_run
    original, output = run.tmp_path / "pool", run.tmp_path / "quota"
    assert run.generate(original, "--questions", 4, "--modes", "lookup") == 0
    args = run.parse("generate", "--source", "eurlex", "--questions-per-mode", 2,
                     "--targets-from", original, "--run-dir", output,
                     "--generation-model", "provider/first", "--generation-model", "provider/second")
    assert args.handler(args, run.context) == 0
    target = read_run_metadata(output)["targets"][2]["target"]
    task = run.qac._digest(["eurlex", target, "provider/first", []])
    with RunState(output / "run.sqlite") as state:
        state.checkpoint(task).run("generation", lambda: [
            {"candidate_id": f"partial-{position}", "question": f"Question {position}?", "answer": "Answer"}
            for position in range(2)
        ])
    run.state.payloads.clear()
    generations = run.state.stages["generation"]
    args = run.parse("generate", "--source", "eurlex", "--run-dir", output, "--resume")
    assert args.handler(args, run.context) == 0
    assert {(target.eli_id, model) for _, model, target, _ in run.state.payloads} == {
        (f"eli/{position}", model) for position in (1, 2)
        for model in ("provider/first", "provider/second")
    }
    assert run.state.stages["generation"] - generations == 3  # The partial generation was replayed.
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert len(rows) == 4


@pytest.mark.parametrize("csv_input", [False, True])
def test_legal_quota_reconstructs_imported_targets_and_filters_pool(legal_run, csv_input):
    from clir_bench.domains.legal.qac.batch_recording import RunState, read_run_metadata

    run = legal_run
    original, output = run.tmp_path / "imported", run.tmp_path / "quota"
    assert run.generate(original, "--questions", 8, "--langs", "en", "de") == 0
    with RunState(original / "run.sqlite") as state:
        state.put("targets", [])
        state.put("imported", True)
    previous_selections = len(run.state.selections)
    args = run.parse("generate", "--source", "eurlex", "--questions-per-mode", 3,
                     "--targets-from", original / "results.csv" if csv_input else original,
                     "--langs", "en", "--modes", "lookup", "--run-dir", output)
    assert args.handler(args, run.context) == 0
    rows, _ = run.qac._read_csv(output / "results.csv")
    assert len(rows) == 3
    assert all(row["mode"] == "lookup" and row["question_language"] == "en" for row in rows)
    assert len(run.state.selections) == previous_selections
    metadata = read_run_metadata(output)
    assert len(metadata["targets"]) == 4
    original_rows, _ = run.qac._read_csv(original / "results.csv")
    assert {row["source_payload_sha256"] for row in rows} <= {
        row["source_payload_sha256"] for row in original_rows
    }


@pytest.mark.parametrize("options,error", [
    (["--questions-per-mode", "0"], "must be positive"),
    (["--questions-per-mode", "3", "--questions", "3"], "cannot be combined"),
])
def test_legal_quota_validation_precedes_calls(legal_run, options, error):
    run = legal_run
    args = run.parse("generate", "--source", "eurlex", "--run-dir", run.tmp_path / "bad", *options)
    with pytest.raises(ValueError, match=error):
        args.handler(args, run.context)
    assert not run.state.invocations


def test_legal_regrade_rejects_conflicting_payload_hashes_across_models(legal_run):
    run = legal_run
    original = run.tmp_path / "original"
    assert run.generate(original) == 0
    rows, fields = run.qac._read_csv(original / "results.csv")
    first_model = rows[0]["generator_model_id"]
    source = rows[0]["corpus"]
    for row in rows:
        if row["corpus"] == source and row["generator_model_id"] != first_model:
            row["source_payload_sha256"] = "0" * 64
    run.qac._write_csv(original / "results.csv", rows, fields)
    args = run.parse(
        "regrade", "--input", original / "results.csv", "--run-dir", run.tmp_path / "invalid"
    )
    with pytest.raises(ValueError, match="source payload"):
        args.handler(args, run.context)


def test_legal_resume_accepts_unchanged_language_options_with_no_sample_in_one_language(legal_run):
    run = legal_run
    directory = run.tmp_path / "sparse-languages"
    assert run.generate(directory, "--langs", "en", "de") == 0
    rows, _ = run.qac._read_csv(directory / "results.csv")
    assert {row["question_language"] for row in rows} == {"en"}
    calls = run.state.stages.copy()
    assert run.generate(directory, "--langs", "en", "de", "--resume") == 0
    assert run.state.stages == calls


def test_legal_indexes_resolve_workspace_overrides_and_translation_inputs(tmp_path):
    import json
    from dataclasses import replace

    from clir_bench.core.context import AppContext
    from clir_bench.domains.legal import qac

    context = _context(tmp_path, "legal")
    settings = replace(
        context.settings,
        domain_overrides={
            "legal": {
                "paths": {
                    "eurlex_structure": "custom/eu",
                    "un_blocks": "custom/blocks",
                    "un_parallel": str(tmp_path / "external-translations"),
                }
            }
        },
    )
    context = AppContext.build(settings, context.domain)
    eurlex = context.workspace.data("eurlex_structure")
    blocks = context.workspace.data("un_blocks")
    translations = context.workspace.data("un_parallel")
    for directory in (eurlex, blocks, translations):
        directory.mkdir(parents=True)
    (eurlex / "articles.jsonl").write_text(
        json.dumps(
            {
                "eli_id": "eli/1",
                "celex_id": "act/1",
                "article_number": "1",
                "unit_type": "article",
                "language": "en",
                "text": "Custom EUR-Lex input.",
            }
        )
        + "\n"
    )
    (eurlex / "internal_edges.jsonl").write_text("")
    (blocks / "docs_en.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "d",
                "symbol": "S/RES/1(1999)",
                "title": "Test",
                "n_blocks": 1,
                "offset": 0,
                "line_start": 1,
                "line_end": 1,
            }
        )
        + "\n"
    )
    (blocks / "blocks_en.jsonl").write_text(
        json.dumps(
            {
                "block_id": "d#0",
                "doc_id": "d",
                "symbol": "S/RES/1(1999)",
                "title": "Test",
                "block_index": 0,
                "n_blocks": 1,
                "line_start": 1,
                "line_end": 1,
                "token_count": 4,
                "text": "Custom UN input.",
            }
        )
        + "\n"
    )
    (translations / "UNv1.0.6way.fr").write_text("Texte français personnalisé.\n")
    indexes = qac._indexes(["eurlex", "un"], context=context)
    assert indexes["eurlex"].articles_path == eurlex / "articles.jsonl"
    assert indexes["eurlex"].quarantine_path == eurlex / "quarantine.jsonl"
    assert indexes["eurlex"].by_eli["eli/1"].texts["en"] == "Custom EUR-Lex input."
    assert indexes["un"].blocks_path == blocks / "blocks_en.jsonl"
    indexes["un"].preload_translations(["fr"], ["d"])
    payload = indexes["un"].build("d", 0, languages=("fr", "en"))
    assert payload.target.texts["en"] == "Custom UN input."
    assert payload.target.texts["fr"] == "Texte français personnalisé."
