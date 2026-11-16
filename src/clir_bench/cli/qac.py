"""The ``qac`` command group: generate and re-grade question/answer/context data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clir_bench import domains
from clir_bench.core import corpus as corpus_io
from clir_bench.core import qagen
from clir_bench.core.context import AppContext
from clir_bench.core.grading import GraderConfig
from clir_bench.core.llm import client_for
from clir_bench.core.parallel import run_tasks
from clir_bench.core.prompts import PromptPack


def register(subparsers: argparse._SubParsersAction, context: AppContext | None) -> None:
    parser = subparsers.add_parser("qac", help="Generate and grade question/answer data")
    sub = parser.add_subparsers(dest="qac_command", metavar="<subcommand>")

    run_workflow = bool(context and context.setting("qac_runs", False))
    eligibility_routing = bool(context and context.domain.qac_route_plan)
    workers = 6 if run_workflow else 1
    plans = (
        ", ".join(context.domain.qac_plans)
        if context and context.domain.qac_plans
        else "none declared"
    )
    generate = sub.add_parser(
        "generate",
        help="Generate graded questions from a corpus",
        description=(
            "Generate and grade questions using the same selected targets for every "
            "generator. Each source gets one run folder and results CSV containing all models."
        )
        if run_workflow
        else (
            "Generates candidate questions per document, grades them for "
            "faithfulness and quality, and keeps the best per document, language and "
            f"mode. Plans available for this domain: {plans}."
        ),
    )
    generate.add_argument(
        "--source",
        required=True,
        **(
            {
                "nargs": "+",
                "choices": context.domain.source_names,
                "help": "Source corpora to include in the same run",
            }
            if run_workflow
            else {"help": "Source corpus to generate from"}
        ),
    )
    generate.add_argument(
        "--plan",
        default="balanced",
        help=(argparse.SUPPRESS if run_workflow else f"Generation plan ({plans})"),
    )
    generate.add_argument(
        "--questions",
        type=int,
        default=None if run_workflow else 100,
        help=(
            "Target/language/persona cases per model, split across sources (default: 100); "
            "each case can produce up to --keep candidates"
            if run_workflow
            else ("Questions per mode, or target/language cases with --decider-model"
                  if eligibility_routing else "Questions per mode")
        ),
    )
    generate.add_argument(
        "--pool",
        type=int,
        default=None,
        help=(argparse.SUPPRESS if run_workflow else "Documents to sample from"),
    )
    generate.add_argument("--langs", nargs="+", help="Restrict question languages")
    generate.add_argument(
        "--priority-langs",
        nargs="+",
        help=(
            argparse.SUPPRESS
            if run_workflow
            else "Used by the `coverage` plan: prefer documents that exist in these "
            "languages, so questions land where coverage is thin"
        ),
    )
    generate.add_argument("--modes", nargs="+", help="Question modes (default: the domain's)")
    generate.add_argument(
        "--output",
        type=Path,
        help=(
            "CSV filename within the run folder (default: results.csv)"
            if run_workflow
            else "Output CSV (default: the source's qac dir)"
        ),
    )
    generate.add_argument(
        "--append",
        action="store_true",
        help=(argparse.SUPPRESS if run_workflow else "Append to an existing dataset"),
    )
    generate.add_argument(
        "--exclude-from",
        type=Path,
        help=(argparse.SUPPRESS if run_workflow else "Skip documents already covered by this CSV"),
    )
    generate.add_argument(
        "--generation-model",
        **(
            {"action": "append", "help": "Generator model ID; repeat to compare models"}
            if run_workflow
            else {"help": "Override the generation model"}
        ),
    )
    generate.add_argument("--verifier-model", help="Override the grading model")
    generate.add_argument(
        "--workers", type=int, default=workers, help=f"Parallel documents (default: {workers})"
    )
    generate.add_argument(
        "--seed",
        type=int,
        default=None if run_workflow else 42,
        help="Selection seed (default: 42; reused on resume)",
    )
    generate.add_argument("--limit", type=int, help="Stop after N plan items (smoke tests)")
    generate.add_argument(
        "--dry-run", action="store_true", help="Show the plan without calling any model"
    )
    if run_workflow:
        _run_arguments(generate)
        generate.add_argument(
            "--decider-model", choices=("generator", "jev"),
            help="Select a mode with its generator, or balance eligible personas using Jev yes/no checks; "
                 "omit to keep explicit modes",
        )
        generate.add_argument(
            "--questions-per-mode", type=int,
            help="Save this many candidates per source, mode and generator; advance through targets until filled",
        )
        generate.add_argument(
            "--targets-from",
            type=Path,
            help="Reuse targets and contexts from a run folder or results CSV; --questions-per-mode allows filtering its pool",
        )
        generate.add_argument(
            "--generation-cache", type=Path,
            help="Replay an existing response cache for matching model IDs and inputs",
        )
        generate.add_argument(
            "--keep",
            type=int,
            default=None,
            choices=(1, 2, 3),
            help="Maximum candidates per target (default: 3)",
        )
        _context_arguments(generate)
    elif eligibility_routing:
        generate.add_argument(
            "--decider-model", choices=("jev",),
            help="Assess persona eligibility with Jev and balance assignments among eligible personas; "
                 "omit to generate explicit modes",
        )
    generate.set_defaults(handler=_domain_command if run_workflow else _generate)

    regrade = sub.add_parser(
        "regrade",
        help="Re-grade existing candidates with a different judge",
        description=(
            "Re-runs the verifiers over already-generated candidates and rewrites "
            "the scores. Generation is untouched, so this isolates judge changes."
        ),
    )
    regrade.add_argument("--input", type=Path, required=True, help="CSV of generated candidates")
    regrade.add_argument(
        "--source",
        required=not run_workflow,
        **(
            {
                "nargs": "+",
                "choices": context.domain.source_names,
                "help": "Source corpora present in the input (default: infer from its rows)",
            }
            if run_workflow
            else {"help": "Source corpus the rows came from"}
        ),
    )
    regrade.add_argument(
        "--output",
        type=Path,
        help=(
            "CSV filename within the new run folder (default: results.csv)"
            if run_workflow
            else "Output CSV (default: <input>_regraded.csv)"
        ),
    )
    regrade.add_argument("--verifier-model", help="Override the grading model")
    regrade.add_argument("--workers", type=int, default=workers)
    if run_workflow:
        _run_arguments(regrade)
        _context_arguments(regrade)
        regrade.add_argument(
            "--dry-run",
            action="store_true",
            help="Inspect the input without calling models or writing files",
        )
    regrade.set_defaults(handler=_domain_command if run_workflow else _regrade)

    best = sub.add_parser(
        "best",
        help="Select the highest-scoring question per document, language and mode",
    )
    best.add_argument("--input", type=Path, required=True)
    best.add_argument("--output", type=Path, help="Default: <input>_best.csv")
    best.set_defaults(handler=_domain_command if run_workflow else _best)


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trace", action="store_true",
        help="Export all LLM inputs and outputs as llm_calls.json and trace.md in each run folder",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        help="Run folder (default: each source's qac/<timestamp>); with multiple sources, a parent for corpus subfolders",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing --run-dir without repeating completed stages",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Total attempts per generation or grading stage (default: 3)",
    )


def _context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-references",
        type=int,
        default=None,
        help="Maximum EUR-Lex reference articles (default: 6)",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=None,
        help="UN context character budget (default: 30000)",
    )


def _domain_command(args: argparse.Namespace, context: AppContext) -> int:
    if getattr(args, "append", False):
        raise ValueError("--append is not supported for runs; use a new run or --resume --run-dir")
    if getattr(args, "resume", False) and not getattr(args, "run_dir", None):
        raise ValueError("--resume requires --run-dir")
    module = domains.load_module(context.domain.name)
    return module.qac_command(args.qac_command, args, context)


def _generation_config(args: argparse.Namespace, context: AppContext) -> qagen.GenerationConfig:
    llm = context.settings.llm
    return qagen.GenerationConfig(
        generation_model=getattr(args, "generation_model", None) or llm.generation_model,
        grader=GraderConfig(
            model=getattr(args, "verifier_model", None) or llm.verifier_model,
            reasoning_effort=llm.grading_reasoning_effort,
            thinking_budget_tokens=llm.thinking_budget_tokens,
            thinking_max_tokens=llm.thinking_max_tokens,
        ),
        generation_reasoning_effort=llm.generation_reasoning_effort,
        retries=llm.retries,
        candidate_parser=context.domain.qac_candidate_parser,
    )


def _generate(args: argparse.Namespace, context: AppContext) -> int:
    domain = context.domain
    source = domain.source(args.source)
    corpus_path = context.workspace.corpus_csv(source)
    if not corpus_path.exists():
        raise FileNotFoundError(f"no corpus for source {source.name!r} at {corpus_path}")

    plan_builder = domain.qac_plans.get(args.plan)
    if plan_builder is None:
        available = ", ".join(domain.qac_plans) or "none"
        raise KeyError(
            f"unknown plan {args.plan!r} for domain {domain.name!r}; available: {available}"
        )

    schema = context.schema
    grouped = corpus_io.load_grouped(corpus_path, schema)
    if domain.qac_validate_corpus:
        domain.qac_validate_corpus(grouped, schema)
    plan = plan_builder(context=context, source=source, grouped=grouped, args=args)
    if args.limit:
        plan = plan[: args.limit]

    output = args.output or (context.workspace.qac_dir(source) / f"qac_{source.name}.csv")
    prompts = PromptPack(package=domain.prompts_package)
    fieldnames = qagen.output_fieldnames(schema, prompts=prompts, modes=domain.analysis.modes)

    print(f"plan: {len(plan)} question(s) from {len(grouped)} documents in {corpus_path.name}")
    if args.dry_run:
        if getattr(args, "decider_model", None):
            print("persona: pending Jev eligibility; dry-run makes no model calls")
        for item in plan[:20]:
            mode = "pending" if getattr(args, "decider_model", None) else item.mode
            print(f"  {item.family:<24} {item.language}  {mode:<10} {item.strategy_name}")
        if len(plan) > 20:
            print(f"  ... and {len(plan) - 20} more")
        print(f"would write -> {output}")
        return 0
    if not plan:
        print("nothing to generate")
        return 0

    if args.append:
        corpus_io.ensure_header(output, fieldnames)
    if domain.qac_route_plan:
        plan = domain.qac_route_plan(
            context=context, source=source, args=args, grouped=grouped, plan=plan, output=output,
        )
        if not plan:
            if not args.append:
                corpus_io.write_rows(output, [], fieldnames)
                corpus_io.write_rows(_suffixed(output, "_best"), [], fieldnames)
            print("no eligible targets to generate")
            return 0

    config = _generation_config(args, context)
    generation_client = client_for(config.generation_model)
    grading_client = client_for(config.grader.model)
    order = domain.languages.priority or domain.languages.working

    failures: list[dict[str, str]] = []

    def work(item: qagen.PlanItem):
        try:
            return qagen.generate_for_document(
                item,
                grouped.get(item.family, []),
                schema=schema,
                prompts=prompts,
                config=config,
                generation_client=generation_client,
                grading_client=grading_client,
                language_order=order,
            )
        except RuntimeError as exc:
            failure = {
                "source_document_id": item.document_id,
                "publication_number": item.family,
                "question_language": item.language,
                "mode": item.mode,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"warning: omitted {item.document_id}/{item.language}/{item.mode}: {exc}",
                  file=sys.stderr)
            return []

    generated: list[dict] = []
    for rows in run_tasks(plan, work, workers=args.workers, description="generate"):
        generated.extend(rows)

    rows = [qagen.normalize_row(row, fieldnames) for row in generated]
    corpus_io.write_rows(output, rows, fieldnames, append=args.append)
    if failures:
        failure_path = _suffixed(output, "_failures")
        corpus_io.write_rows(
            failure_path, sorted(failures, key=lambda row: row["source_document_id"]),
            ("source_document_id", "publication_number", "question_language", "mode", "error"),
            append=args.append,
        )
        print(f"omitted {len(failures)} target(s) after retry exhaustion -> {failure_path}")
    best_rows = qagen.select_best(rows, schema)
    best_path = _suffixed(output, "_best")
    corpus_io.write_rows(
        best_path, [qagen.normalize_row(r, fieldnames) for r in best_rows], fieldnames
    )

    print(f"wrote {len(rows)} candidate row(s) -> {output}")
    print(f"wrote {len(best_rows)} best row(s)      -> {best_path}")
    return 0


def _regrade(args: argparse.Namespace, context: AppContext) -> int:
    from clir_bench.core.grading import grade_columns, grade_one

    domain = context.domain
    schema = context.schema
    source = domain.source(args.source)
    grouped = corpus_io.load_grouped(context.workspace.corpus_csv(source), schema)
    if domain.qac_validate_corpus:
        domain.qac_validate_corpus(grouped, schema)
    rows = corpus_io.read_rows(args.input)
    if not rows:
        print(f"{args.input} is empty")
        return 0

    config = _generation_config(args, context)
    prompts = PromptPack(package=domain.prompts_package)
    fieldnames = qagen.output_fieldnames(schema, prompts=prompts, modes=domain.analysis.modes)
    empty_grades = dict.fromkeys(
        qagen.grading_fieldnames(prompts=prompts, modes=domain.analysis.modes), ""
    )
    client = client_for(config.grader.model)
    order = domain.languages.priority or domain.languages.working

    def work(row: dict) -> dict:
        family = str(row.get(schema.family_field, ""))
        mode = str(row.get("mode") or "technical")
        item = qagen.PlanItem(
            family=family, language=str(row.get("question_language", "")), mode=mode,
            document_id=str(row.get("source_document_id") or (
                row.get("corpus_id") if domain.qac_validate_corpus else ""
            ) or ""),
        )
        if domain.qac_validate_corpus and not item.document_id:
            raise ValueError("Regrading requires a selected source document ID")
        passages = corpus_io.build_passages_text(
            qagen.select_plan_rows(item, grouped.get(family, []), schema), schema,
            order=order, include_identifiers=True,
        )
        faith, quality = grade_one(
            client,
            config.grader,
            prompts.faithfulness("batch"),
            prompts.quality(mode, "batch"),
            passages,
            qagen.candidate_from_row(row),
            mode,
            strict=True,
        )
        return {
            **row, **empty_grades, **grade_columns(faith, quality, mode),
            "source_document_id": item.document_id,
        }

    regraded = list(run_tasks(rows, work, workers=args.workers, description="regrade"))
    output = args.output or _suffixed(args.input, "_regraded")
    corpus_io.write_rows(output, [qagen.normalize_row(r, fieldnames) for r in regraded], fieldnames)

    best_rows = qagen.select_best(regraded, schema)
    best_path = _suffixed(output, "_best")
    corpus_io.write_rows(
        best_path, [qagen.normalize_row(r, fieldnames) for r in best_rows], fieldnames
    )
    print(f"regraded {len(regraded)} row(s) -> {output}")
    print(f"best {len(best_rows)} row(s)     -> {best_path}")
    return 0


def _best(args: argparse.Namespace, context: AppContext) -> int:
    schema = context.schema
    rows = corpus_io.read_rows(args.input)
    best_rows = qagen.select_best(rows, schema)
    output = args.output or _suffixed(args.input, "_best")
    fieldnames = qagen.output_fieldnames(
        schema, prompts=PromptPack(package=context.domain.prompts_package),
        modes=context.domain.analysis.modes,
    )
    corpus_io.write_rows(
        output, [qagen.normalize_row(r, fieldnames) for r in best_rows], fieldnames
    )
    print(f"selected {len(best_rows)} of {len(rows)} row(s) -> {output}")
    return 0


def _suffixed(path: Path, suffix: str) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")
