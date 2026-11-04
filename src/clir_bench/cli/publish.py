"""The ``publish`` command group: push corpora, benchmarks and results to the Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from clir_bench.core import corpus as corpus_io
from clir_bench.core import publish as publish_io
from clir_bench.core.context import AppContext


def register(subparsers: argparse._SubParsersAction, context: AppContext | None) -> None:
    parser = subparsers.add_parser("publish", help="Publish datasets to the Hugging Face Hub")
    sub = parser.add_subparsers(dest="publish_command", metavar="<subcommand>")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", help="Target dataset repo (default: from clir.toml)")
    common.add_argument(
        "--dry-run", action="store_true", help="Write parquet locally instead of uploading"
    )
    common.add_argument("--private", action="store_true", help="Create the repo as private")

    corpus_parser = sub.add_parser(
        "corpus",
        parents=[common],
        help="Publish a corpus as the shared retrieval haystack",
        description=(
            "Publishes the corpus alone. Every evaluation retrieves against this "
            "shared haystack, so scores stay comparable across benchmarks."
        ),
    )
    corpus_parser.add_argument("--source", help="Source to publish (default: every source, merged)")
    corpus_parser.set_defaults(handler=_corpus)

    benchmark = sub.add_parser(
        "benchmark",
        parents=[common],
        help="Publish a corpus + queries + qrels retrieval benchmark",
    )
    benchmark.add_argument("--source", required=True, help="Source the benchmark is built from")
    benchmark.add_argument("--qac", type=Path, help="QAC CSV (default: the source's best file)")
    benchmark.add_argument("--only-configs", nargs="+", help="Re-push only these configs")
    benchmark.set_defaults(handler=_benchmark)

    results = sub.add_parser(
        "results", parents=[common], help="Upload a run's tables to a dataset repo"
    )
    results.add_argument("--run", help="Run id or path (default: latest)")
    results.add_argument(
        "--path-in-repo", default="benchmark_outputs", help="Destination folder in the repo"
    )
    results.set_defaults(handler=_results)

    generations = sub.add_parser(
        "generations",
        help="Export all generated QAs, metadata and their related documents",
    )
    inputs = generations.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path, help="Explicit generation/source input manifest")
    inputs.add_argument("--upload-from", type=Path, help="Upload an already prepared export")
    generations.add_argument("--output", type=Path, help="New local export directory")
    generations.add_argument(
        "--corpora",
        nargs="+",
        choices=["eurlex", "un", "chemistry"],
        help="Select corpora from the input manifest (default: all)",
    )
    generations.add_argument(
        "--corpus-scope",
        choices=["referenced", "full"],
        default="referenced",
        help="Related documents only (default), or the full local corpora",
    )
    generations.add_argument("--push", action="store_true", help="Upload after preparing")
    generations.add_argument("--repo", help="Destination Hugging Face dataset repository")
    generations.add_argument("--public", action="store_true", help="Create a public repository")
    generations.set_defaults(handler=_generations, needs_domain=False)

    expansion = sub.add_parser(
        "expand-corpus",
        help="Expand a published QA dataset to its full local source corpus",
    )
    expansion.add_argument(
        "--from-export", type=Path, required=True,
        help="Original published export, whose QA and evidence files are preserved",
    )
    expansion_inputs = expansion.add_mutually_exclusive_group(required=True)
    expansion_inputs.add_argument("--manifest", type=Path, help="Full source input manifest")
    expansion_inputs.add_argument("--upload-from", type=Path, help="Prepared expanded export")
    expansion.add_argument("--corpus", choices=["eurlex", "un", "chemistry"])
    expansion.add_argument("--output", type=Path, help="New local export directory")
    expansion.add_argument("--push", action="store_true", help="Upload after preparing")
    expansion.add_argument("--repo", help="Existing Hugging Face dataset repository")
    expansion.set_defaults(handler=_expand_corpus, needs_domain=False)


def _generations(args: argparse.Namespace, context: AppContext | None) -> int:
    import json

    from clir_bench.publishing.generations import prepare, upload

    push = args.push or args.upload_from is not None
    if push and not args.repo:
        raise ValueError("upload requires --repo OWNER/DATASET")
    if args.repo and not push:
        raise ValueError("use --push to upload, or omit --repo to prepare locally")
    if args.upload_from and args.output:
        raise ValueError("--upload-from cannot be combined with --output")
    if args.manifest and not args.output:
        raise ValueError("preparation requires --output")
    output = args.upload_from or args.output
    if args.manifest:
        report = prepare(args.manifest, output, scope=args.corpus_scope, corpora=args.corpora)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Prepared dataset: {output}")
    if push:
        print(upload(output, args.repo, private=not args.public))
    return 0


def _expand_corpus(args: argparse.Namespace, context: AppContext | None) -> int:
    import json

    from clir_bench.publishing.corpus_expansion import expand_corpus
    from clir_bench.publishing.corpus_update import upload_corpus_expansion

    push = args.push or args.upload_from is not None
    if push and not args.repo:
        raise ValueError("upload requires --repo OWNER/DATASET")
    if args.repo and not push:
        raise ValueError("use --push to upload, or omit --repo to prepare locally")
    if args.upload_from and (args.output or args.corpus):
        raise ValueError("--upload-from cannot be combined with --output or --corpus")
    if args.manifest and (not args.output or not args.corpus):
        raise ValueError("preparation requires --output and --corpus")
    output = args.upload_from or args.output
    if args.manifest:
        report = expand_corpus(args.from_export, args.manifest, output, corpus=args.corpus)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Prepared expanded dataset: {output}")
    if push:
        commit = upload_corpus_expansion(args.from_export, output, args.repo)
        print(f"Published expanded corpus: https://huggingface.co/datasets/{args.repo}")
        print(f"Revision: {commit.oid}")
    return 0


def _corpus(args: argparse.Namespace, context: AppContext) -> int:
    domain = context.domain
    sources = [domain.source(args.source)] if args.source else list(domain.sources)

    rows: list[dict] = []
    used: list[str] = []
    for source in sources:
        path = context.workspace.corpus_csv(source)
        if not path.exists():
            print(f"skip {source.name}: no corpus at {path}")
            continue
        rows.extend(corpus_io.read_rows(path))
        used.append(source.name)
    if not rows:
        raise FileNotFoundError("no corpus rows to publish")
    if domain.qac_validate_corpus:
        domain.qac_validate_corpus(corpus_io.group_by_family(rows, context.schema), context.schema)

    repo = context.hf_repo("corpus_repo", args.repo)
    if not repo:
        raise ValueError("no target repo: pass --repo or set corpus_repo in clir.toml")

    bundle = publish_io.DatasetBundle()
    bundle.add("corpus", publish_io.corpus_rows(rows, context.schema))

    attribution = "\n\n".join(domain.attribution_for(name) for name in used)
    card = publish_io.build_card(
        title=f"{domain.title} — shared corpus",
        description=(
            f"{domain.description}\n\nThe shared retrieval haystack for every "
            f"{domain.name} benchmark. Sources: {', '.join(used)}."
        ),
        attribution=attribution,
        bundle=bundle,
    )
    url = publish_io.publish_bundle(
        bundle,
        repo,
        card=card,
        private=args.private,
        dry_run=args.dry_run,
        dry_run_dir=context.settings.data_dir / "hf_export" / "corpus",
    )
    print(url)
    return 0


def _benchmark(args: argparse.Namespace, context: AppContext) -> int:
    domain = context.domain
    source = domain.source(args.source)
    corpus_path = context.workspace.corpus_csv(source)
    qac_path = args.qac or (context.workspace.qac_dir(source) / f"qac_{source.name}_best.csv")
    for path in (corpus_path, qac_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"required input missing: {path}")

    repo = context.hf_repo(f"{source.name}_benchmark_repo", args.repo) or context.hf_repo(
        "benchmark_repo", args.repo
    )
    if not repo:
        raise ValueError(
            "no target repo: pass --repo or set "
            f"{source.name}_benchmark_repo in clir.toml under [domains.{domain.name}]"
        )

    corpus_rows = corpus_io.read_rows(corpus_path)
    if domain.qac_validate_corpus:
        domain.qac_validate_corpus(
            corpus_io.group_by_family(corpus_rows, context.schema),
            context.schema,
        )
    configs = publish_io.build_retrieval_configs(
        corpus_rows,
        corpus_io.read_rows(qac_path),
        context.schema,
    )
    bundle = publish_io.bundle_from_configs(configs)
    relevance = (
        "its selected source document and that document's directly accepted translation partners"
        if context.schema.parallel_ids_field
        else "every language version of its source document"
    )
    card = publish_io.build_card(
        title=f"{domain.title} — {source.name} retrieval benchmark",
        description=(
            f"{domain.description}\n\nQueries are graded questions about documents "
            f"from {source.name}. A query is relevant to {relevance}; "
            "the `cross_language-*` configs keep only documents "
            "in a language other than the query's."
        ),
        attribution=domain.attribution_for(source.name),
        bundle=bundle,
    )
    url = publish_io.publish_bundle(
        bundle,
        repo,
        card=card,
        private=args.private,
        dry_run=args.dry_run,
        dry_run_dir=context.settings.data_dir / "hf_export" / f"{source.name}_benchmark",
        only_configs=args.only_configs,
    )
    print(url)
    return 0


def _results(args: argparse.Namespace, context: AppContext) -> int:
    from clir_bench.core.runs import resolve_run_dir

    run_dir = resolve_run_dir(context.workspace.runs_dir, args.run)
    tables = run_dir / "mteb_tables"
    source_dir = tables if tables.exists() else run_dir
    repo = context.hf_repo("benchmark_repo", args.repo)
    if not repo:
        raise ValueError("no target repo: pass --repo")
    if args.dry_run:
        print(f"[dry run] would upload {source_dir} -> {repo}/{args.path_in_repo}")
        return 0
    url = publish_io.upload_directory(source_dir, repo, path_in_repo=args.path_in_repo)
    print(url)
    return 0
