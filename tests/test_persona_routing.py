"""Offline coverage of weighted eligibility, shared assignments and safe replay."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Event
from types import SimpleNamespace

import pytest

from clir_bench import domains
from clir_bench.cli import build_parser
from clir_bench.core import llm
from clir_bench.domains.legal import qac
from clir_bench.domains.legal.qac import decider
from clir_bench.domains.legal.qac import persona_routing as routing
from clir_bench.domains.legal.qac.batch_recording import RunState, read_run_metadata
from test_domain_contract import _context
from test_legal_batch_replay_integration import case


def eligibility(source, modes):
    return decider.parse_eligibility({"answers": {
        mode: {"type": "noul", "noul": float(mode in modes)} for mode in decider.MODES[source]
    }}, source)


@pytest.mark.parametrize("source", ["eurlex", "un"])
def test_mix_and_constrained_eligibility(source):
    weights, counts = routing.resolve_weights(), Counter()
    all_yes = eligibility(source, decider.MODES[source])
    for i in range(240):
        decision = routing.choose(source, all_yes, counts, weights, seed=42, identity=str(i))
        counts[decision["mode"]] += 1
        if i == 23:
            assert counts == {mode: weights[mode] for mode in decider.MODES[source]}
    assert counts == {mode: 10 * weights[mode] for mode in decider.MODES[source]}

    counts = Counter(lookup=120)
    decision = routing.choose(source, eligibility(source, ["lookup"]), counts, weights,
                              seed=42, identity="lookup-only")
    assert decision["mode"] == "lookup"
    decision = routing.choose(source, eligibility(source, ["lookup", "comparison"]), counts,
                              weights, seed=42, identity="rare-opportunity")
    assert decision["mode"] == "comparison"
    decision = routing.choose(source, eligibility(source, []), counts, weights,
                              seed=42, identity="none")
    assert decision["mode"] == "skip"


@pytest.mark.parametrize("invalid", [[], {"lookpu": 5}, {"lookup": 0}, {"comparison": -1},
                                     {"lookup": True}, {"lookup": float("inf")},
                                     {"lookup": float("nan")}, {"lookup": "5"}])
def test_invalid_weights(invalid):
    with pytest.raises(ValueError):
        routing.resolve_weights(invalid)


def test_config_weights_are_used_and_frozen_on_resume(tmp_path):
    context = replace(_context(tmp_path, "legal"), domain_settings={"persona_weights": {"comparison": 9}})
    args = SimpleNamespace(decider_model="jev")
    options = qac._options(args, context)
    assert options["persona_weights"]["comparison"] == 9
    context = replace(context, domain_settings={"persona_weights": {"comparison": 1}})
    saved = {"fingerprint": "old", "config": options}
    assert qac._options(args, context, saved)["persona_weights"]["comparison"] == 9
    assert "persona_selection_policy" not in qac._options(
        args, context, {"fingerprint": "legacy", "config": {"decider_model": "jev"}})


def records(source, n):
    batch, _, base, payload, _ = case(source)
    targets = []
    for i in range(n):
        target = (replace(base, eli_id=f"eli:{i}", celex_id=f"act:{i}") if source == "eurlex"
                  else replace(base, doc_id=f"doc:{i}", block_id=f"doc:{i}#0"))
        entry = {"corpus": source, "target": asdict(target)}
        targets.append((entry, replace(payload, text=f"packet:{i}")))
    index = (SimpleNamespace(status={f"eli:{i}": {"complete": True} for i in range(n)})
             if source == "eurlex" else SimpleNamespace(incomplete={}))
    return batch, targets, index


@pytest.mark.parametrize("source", ["eurlex", "un"])
def test_safety_uses_authoritative_index(source):
    _, targets, index = records(source, 1)
    entries = [targets[0][0]]
    routing.require_safe_targets(entries, {source: index})
    if source == "eurlex":
        index.status = {"eli:0": {"complete": False}}
    else:
        index.incomplete = {"doc:0#0": {"complete": False}}
    with pytest.raises(ValueError, match="reference-complete"):
        routing.require_safe_targets(entries, {source: index})
    if source == "eurlex":
        index.status = {}
    else:
        index.incomplete = None
    with pytest.raises(ValueError, match="missing reference status"):
        routing.require_safe_targets(entries, {source: index})


@pytest.mark.parametrize("source", ["eurlex", "un"])
def test_parallel_eligibility_order_and_interrupted_selection_resume(tmp_path, monkeypatch, source):
    _, targets, _ = records(source, 4)
    original_choose = routing.choose
    calls = []
    second_finished = Event()

    def assess(corpus, text, *, checkpoint):
        def call():
            calls.append(text)
            # The second response becomes available before the first.
            if text == "packet:0":
                assert second_finished.wait(timeout=5)
            if text == "packet:1":
                second_finished.set()
            return eligibility(corpus, decider.MODES[corpus])
        return checkpoint.run("eligibility", call)

    def interrupt(*a, **kw):
        if sum(a[2].values()) == 2:
            raise KeyboardInterrupt("stop before third selection commit")
        return original_choose(*a, **kw)

    monkeypatch.setattr(decider, "decide_eligibility", assess)
    monkeypatch.setattr(routing, "choose", interrupt)
    path = tmp_path / "run.sqlite"
    with RunState(path) as state, ThreadPoolExecutor(max_workers=4) as executor:
        with pytest.raises(KeyboardInterrupt):
            list(routing.route_targets(state, targets, weights=routing.DEFAULT_WEIGHTS,
                                       seed=42, retries=1, executor=executor))
        with state.transaction() as db:
            assert db.execute("SELECT count(*) FROM stages WHERE stage='persona_selection'").fetchone()[0] == 2
    monkeypatch.setattr(routing, "choose", original_choose)
    with RunState(path) as state, ThreadPoolExecutor(max_workers=1) as executor:
        resumed = list(routing.route_targets(state, targets, weights=routing.DEFAULT_WEIGHTS,
                                            seed=42, retries=1, executor=executor))
    assert len(calls) == 4  # Eligibility and completed choices were reused.
    with RunState(tmp_path / "fresh.sqlite") as state, ThreadPoolExecutor(max_workers=1) as executor:
        fresh = list(routing.route_targets(state, targets, weights=routing.DEFAULT_WEIGHTS,
                                          seed=42, retries=1, executor=executor))
    assert resumed == fresh
    assert [sum(d["counts_before"].values()) for _, d, _ in resumed] == [0, 1, 2, 3]


def run_fixture(tmp_path, monkeypatch, source, n, *, eligible=None, fail_generation=False,
                unsafe=False):
    batch, targets, index = records(source, n)
    context = _context(tmp_path, "legal")
    parser = build_parser(context, domains.load_module("legal"))
    argv = ["qac", "generate", "--source", source, "--decider-model", "jev",
            "--generation-model", "provider/a", "--generation-model", "provider/b",
            "--verifier-model", "judge", "--langs", "en", "--workers", "4",
            "--retries", "1", "--trace", "--run-dir", str(tmp_path / "run")]
    if unsafe:
        if source == "eurlex":
            index.status["eli:0"]["complete"] = False
        else:
            index.incomplete["doc:0#0"] = {"complete": False}
    monkeypatch.setattr(qac, "_indexes", lambda *a, **kw: {source: index})
    payloads = {entry["target"].get("eli_id") or entry["target"].get("block_id"): payload
                for entry, payload in targets}
    monkeypatch.setattr(batch, "prepare_payload", lambda target, *a, **kw:
                        payloads[getattr(target, "eli_id", None) or target.block_id])
    native_calls, generation_calls = [], []

    def native(request):
        native_calls.append(request)
        modes = eligible if eligible is not None else decider.MODES[source]
        return {"answers": {mode: {"type": "noul", "noul": float(mode in modes)}
                            for mode in decider.MODES[source]}}

    def generate(target, index, *, gen_model, **kw):
        identity = getattr(target, "eli_id", None) or target.block_id
        generation_calls.append((identity, gen_model, target.mode))
        if fail_generation and gen_model == "provider/a":
            raise ValueError("mock generation failure")
        return [{"corpus": source, "document_id": identity, "target_id": identity,
                 "question_language": target.language, "mode": target.mode,
                 "generator_model_id": gen_model, "candidate_id": f"{identity}/{gen_model}/{i}",
                 "question": "Mock question", "answer": "Mock answer", "total_score": 35,
                 "grading_status": "completed", "faith_grounding": 5} for i in range(3)]

    monkeypatch.setattr(llm, "decisions", native)
    monkeypatch.setattr(batch, "run_one", generate)
    return context, parser, argv, [entry for entry, _ in targets], native_calls, generation_calls


@pytest.mark.parametrize("backend", ["jev", "generator"])
def test_automatic_generator_is_selected_after_the_persona_and_resumes(tmp_path, monkeypatch, backend):
    from clir_bench.core.generation_models import HAIKU_MODEL, QWEN_MODEL

    context, parser, argv, entries, calls, generated = run_fixture(
        tmp_path, monkeypatch, "un", 2, eligible=["lookup"])
    first = argv.index("--generation-model")
    del argv[first:first + 4]  # Exercise the default rather than explicit model comparisons.
    argv[argv.index("--decider-model") + 1] = backend
    if backend == "generator":
        def decide(source, payload, *, backend, model, checkpoint):
            assert model == HAIKU_MODEL
            calls.append(source)
            return {"mode": "lookup", "generation_mode": "lookup", "model": model,
                    "reason": "The source states a fact", "confidence": None,
                    "probabilities": None}
        monkeypatch.setattr(decider, "decide", decide)
    assert qac.generate(parser.parse_args(argv), context, selections=entries) == 0
    assert len(calls) == 2
    assert {(model, mode) for _, model, mode in generated} == {(QWEN_MODEL, "lookup")}
    rows, _ = qac._read_csv(tmp_path / "run/results.csv")
    assert len(rows) == 6
    assert all(row["generator_model_id"] == QWEN_MODEL for row in rows)
    assert all(row["generator_selection_basis"] == "paired_target_mean" for row in rows)
    before = len(generated)
    assert qac.generate(parser.parse_args(argv + ["--resume"]), context) == 0
    assert len(calls) == 2 and len(generated) == before


@pytest.mark.parametrize("source", ["eurlex", "un"])
def test_pipeline_shares_choices_counts_once_and_reuses_resume(tmp_path, monkeypatch, source):
    context, parser, argv, entries, calls, generated = run_fixture(tmp_path, monkeypatch, source, 24)
    assert qac.generate(parser.parse_args(argv), context, selections=entries) == 0
    assert len(calls) == 24 and len(generated) == 48
    assert len({(identity, mode) for identity, _, mode in generated}) == 24
    metadata = read_run_metadata(tmp_path / "run")
    balance = metadata["summary"]["persona_balance"]
    assert balance["assigned"] == 24 and metadata["summary"]["candidates"] == 144
    assert {m: r["assigned"] for m, r in balance["personas"].items()} == {
        m: routing.DEFAULT_WEIGHTS[m] for m in decider.MODES[source]}
    rows, _ = qac._read_csv(tmp_path / "run" / "results.csv")
    assert all(row["decider_selection_policy"] == routing.POLICY for row in rows)
    assert len({row["decider_routing_task"] for row in rows}) == 24
    trace = json.loads((tmp_path / "run" / "llm_calls.json").read_text())
    selections = [s for s in trace["stages"] if s["stage"] == "persona_selection"]
    assert len(selections) == 24
    assert len({s["parsed_output"]["mode"] for s in selections}) == 6
    assert qac.generate(parser.parse_args(argv + ["--resume"]), context) == 0
    assert len(calls) == 24 and len(generated) == 48
    assert read_run_metadata(tmp_path / "run")["summary"]["persona_balance"] == balance


@pytest.mark.parametrize("eligible,expected_calls", [([], 0), (["lookup"], 8)])
def test_all_no_and_oversupplied_only_yes(tmp_path, monkeypatch, eligible, expected_calls):
    context, parser, argv, entries, calls, generated = run_fixture(
        tmp_path, monkeypatch, "eurlex", 4, eligible=eligible)
    assert qac.generate(parser.parse_args(argv), context, selections=entries) == 0
    assert len(calls) == 4 and len(generated) == expected_calls
    balance = read_run_metadata(tmp_path / "run")["summary"]["persona_balance"]
    assert balance["assigned"] == (4 if eligible else 0)
    assert balance["no_eligible_persona"] == (0 if eligible else 4)
    assert all(mode == "lookup" for _, _, mode in generated)


def test_generation_failure_does_not_undo_or_duplicate_assignment(tmp_path, monkeypatch):
    context, parser, argv, entries, calls, generated = run_fixture(
        tmp_path, monkeypatch, "eurlex", 2, fail_generation=True)
    assert qac.generate(parser.parse_args(argv), context, selections=entries) == 1
    first = read_run_metadata(tmp_path / "run")["summary"]["persona_balance"]
    assert first["assigned"] == 2
    assert qac.generate(parser.parse_args(argv + ["--resume"]), context) == 1
    assert len(calls) == 2
    assert len(generated) == 6  # Only the two failed generator cases run again.
    assert read_run_metadata(tmp_path / "run")["summary"]["persona_balance"] == first


@pytest.mark.parametrize("source", ["eurlex", "un"])
def test_unsafe_imported_plan_fails_before_calls(tmp_path, monkeypatch, source):
    context, parser, argv, entries, calls, generated = run_fixture(
        tmp_path, monkeypatch, source, 2, unsafe=True)
    with pytest.raises(ValueError, match="reference-complete"):
        qac.generate(parser.parse_args(argv), context, selections=entries)
    assert calls == generated == []
    assert not (tmp_path / "run" / "run.sqlite").exists()


def test_invalid_eligibility_remains_failure_and_retry_budget_survives_resume(tmp_path, monkeypatch):
    context, parser, argv, entries, calls, generated = run_fixture(tmp_path, monkeypatch, "un", 2)
    def invalid(request):
        calls.append(request)
        return {"answers": {}}
    monkeypatch.setattr(llm, "decisions", invalid)
    assert qac.generate(parser.parse_args(argv), context, selections=entries) == 1
    assert generated == [] and len(calls) == 2
    balance = read_run_metadata(tmp_path / "run")["summary"]["persona_balance"]
    assert balance["eligibility_failures"] == 2 and balance["assigned"] == 0
    assert balance["no_eligible_persona"] == 0
    assert qac.generate(parser.parse_args(argv + ["--resume"]), context) == 1
    assert generated == [] and len(calls) == 2


def test_successful_eligibility_retry_counts_one_assignment(tmp_path, monkeypatch):
    context, parser, argv, entries, calls, generated = run_fixture(tmp_path, monkeypatch, "eurlex", 1)
    argv[argv.index("--retries") + 1] = "2"
    def retry(request):
        calls.append(request)
        if len(calls) == 1:
            return {"answers": {}}
        return {"answers": {mode: {"type": "noul", "noul": 1}
                            for mode in decider.MODES["eurlex"]}}
    monkeypatch.setattr(llm, "decisions", retry)
    assert qac.generate(parser.parse_args(argv), context, selections=entries) == 0
    assert len(calls) == 2 and len(generated) == 2
    assert read_run_metadata(tmp_path / "run")["summary"]["persona_balance"]["assigned"] == 1
    assert qac.generate(parser.parse_args(argv + ["--resume"]), context) == 0
    assert len(calls) == 2 and len(generated) == 2


def test_interruption_during_generation_reuses_all_assignments(tmp_path, monkeypatch):
    context, parser, argv, entries, calls, generated = run_fixture(tmp_path, monkeypatch, "eurlex", 3)
    batch = qac._batches()["eurlex"]
    original = batch.run_one
    def interrupt(*a, **kw):
        raise KeyboardInterrupt("generation interrupted")
    monkeypatch.setattr(batch, "run_one", interrupt)
    with pytest.raises(KeyboardInterrupt):
        qac.generate(parser.parse_args(argv), context, selections=entries)
    assert read_run_metadata(tmp_path / "run")["status"] == "interrupted"
    before = json.loads((tmp_path / "run" / "llm_calls.json").read_text())
    choices_before = [s for s in before["stages"] if s["stage"] == "persona_selection"]
    assert len(choices_before) == 3
    monkeypatch.setattr(batch, "run_one", original)
    assert qac.generate(parser.parse_args(argv + ["--resume"]), context) == 0
    assert len(calls) == 3 and len(generated) == 6
    after = json.loads((tmp_path / "run" / "llm_calls.json").read_text())
    assert [s for s in after["stages"] if s["stage"] == "persona_selection"] == choices_before
    assert read_run_metadata(tmp_path / "run")["summary"]["persona_balance"]["assigned"] == 3


def test_mixed_corpora_keep_separate_assignment_counts(tmp_path, monkeypatch):
    targets = []
    for i in range(2):
        for source in ("eurlex", "un"):
            targets.append(records(source, 2)[1][i])
    monkeypatch.setattr(decider, "decide_eligibility", lambda source, text, **kw:
                        eligibility(source, ["lookup"]))
    with RunState(tmp_path / "run.sqlite") as state, ThreadPoolExecutor(max_workers=2) as executor:
        results = list(routing.route_targets(state, targets, weights=routing.DEFAULT_WEIGHTS,
                                            seed=42, retries=1, executor=executor))
    assert [d["counts_before"]["lookup"] for _, d, _ in results] == [0, 0, 1, 1]
