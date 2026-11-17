"""
Commands this domain adds.

Two kinds: ingestion of the accepted BASF-AI abstract pairs, and the
chemistry-specific benchmark groups, which have no generic equivalent.

Handlers import their implementation lazily so that ``clir --help`` does not
pull in BigQuery, networkx or the model stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clir_bench.core.context import AppContext
from clir_bench.domains.chemistry.vocabulary import WORKING_LANGUAGES


def register_cli(registrar, context: AppContext) -> None:
    _register_ingest(registrar)
    _register_alias_graph(registrar)
    _register_code_switch(registrar)
    _register_progressive(registrar)


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #

def _register_ingest(registrar) -> None:
    gp = registrar.ingest_source(
        "gp", help="Import accepted exact/high BASF-AI Google Patents abstract pairs",
    )
    gp.add_argument("--revision", default="main", help="Hugging Face revision to resolve and pin")
    gp.add_argument("--limit", type=int, help="Limit accepted pairs for a small local import")
    gp.add_argument(
        "--languages", nargs="+", choices=WORKING_LANGUAGES,
        help=f"Keep pairs with both sides in these languages (default: {' '.join(WORKING_LANGUAGES)})",
    )
    gp.add_argument("--input", type=Path, help="Read a local export in Parquet or JSONL format")
    gp.set_defaults(handler=_ingest_gp)


def _ingest_gp(args: argparse.Namespace, context: AppContext) -> int:
    from clir_bench.domains.chemistry.sources.abstract_pairs import ingest

    return ingest(context, args)


# --------------------------------------------------------------------------- #
# Alias-graph benchmark
# --------------------------------------------------------------------------- #

def _register_alias_graph(registrar) -> None:
    group = registrar.group(
        "alias-graph",
        help="Concept-retrieval benchmark: ChEBI concepts with look-alike hard negatives",
    )

    build = group.add_parser("build", help="Select concepts and assemble gold and hard-negative sets")
    build.add_argument("--source", default="gp", help="Corpus source to search (default: gp)")
    build.add_argument("--min-gold", type=int, default=2, help="Minimum gold documents per concept")
    build.add_argument("--min-negatives", type=int, default=3, help="Minimum hard negatives per concept")
    build.add_argument("--max-concepts", type=int, help="Cap the number of concepts kept")
    build.add_argument("--max-df", type=float, default=0.02, help="Drop names above this document frequency")
    build.add_argument("--langs", nargs="+", help="Languages to fetch concept names for")
    build.add_argument("--chebi-variant", choices=["full", "core", "lite"], help="ChEBI release to use")
    build.add_argument("--no-wikipedia", action="store_true", help="Use ontology names only")
    build.add_argument("--include-classes", action="store_true", help="Allow broad class concepts")
    build.add_argument("--include-non-molecular", action="store_true", help="Allow role and group concepts")
    build.set_defaults(handler=_alias_build)

    qa = group.add_parser("qa", help="Generate concept-centric queries")
    qa.add_argument("--strategy", type=int, default=1, choices=[1, 2, 3, 4])
    qa.add_argument("--limit", type=int, help="Documents to generate for")
    qa.add_argument("--workers", type=int, default=1)
    qa.add_argument("--seed", type=int, default=42)
    qa.add_argument("--model", help="Override the generation model")
    qa.set_defaults(handler=_alias_qa)

    publish = group.add_parser("publish", help="Publish the alias-graph benchmark")
    publish.add_argument("--repo", help="Target dataset repo")
    publish.add_argument("--only-configs", nargs="+", help="Re-push only these configs")
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--private", action="store_true")
    publish.set_defaults(handler=_alias_publish)

    export = group.add_parser("export-concept", help="Write one concept's documents to a CSV")
    export.add_argument("concept", help="Concept id, e.g. CHEBI:15365")
    export.add_argument("--source", default="gp")
    export.set_defaults(handler=_alias_export)

    names = group.add_parser("names", help="Fetch or validate multilingual concept names")
    names.add_argument("--cache-all", action="store_true", help="Populate the cache for every concept")
    names.add_argument("--check", action="store_true", help="Check names against parallel translations")
    names.add_argument("--langs", nargs="+")
    names.set_defaults(handler=_alias_names)


def _alias_build(args, context):
    from clir_bench.domains.chemistry.aliasgraph.builder import build_alias_graph

    return build_alias_graph(context, args)


def _alias_qa(args, context):
    from clir_bench.domains.chemistry.aliasgraph.concept_qa import generate_concept_qa

    return generate_concept_qa(context, args)


def _alias_publish(args, context):
    from clir_bench.domains.chemistry.aliasgraph.publish import publish_alias_graph

    return publish_alias_graph(context, args)


def _alias_export(args, context):
    from clir_bench.domains.chemistry.aliasgraph.builder import export_concept

    return export_concept(context, args)


def _alias_names(args, context):
    from clir_bench.domains.chemistry.aliasgraph.names import run_names_command

    return run_names_command(context, args)


# --------------------------------------------------------------------------- #
# Code-switching benchmarks
# --------------------------------------------------------------------------- #

def _register_code_switch(registrar) -> None:
    group = registrar.group(
        "code-switch",
        help="Perturbation benchmark: swap a term and see whether retrieval survives",
    )

    build = group.add_parser("build", help="Build the A-F variant corpus")
    build.add_argument(
        "--variants",
        default="A,B,C,D,F",
        help="Variants to emit: A baseline, B in-set language, C out-of-set, "
        "D spelling noise, E non-chemistry control (LLM), F ontology form",
    )
    build.add_argument("--source", default="gp")
    build.add_argument("--limit", type=int, help="Concepts to process")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--model", help="Model for the LLM-generated variant E")
    build.set_defaults(handler=_cs_build)

    qa = group.add_parser("qa", help="Generate queries for the variant corpus")
    qa.add_argument("--source", default="gp", help="Corpus source the variants were built from")
    qa.add_argument("--limit", type=int)
    qa.add_argument("--workers", type=int, default=1)
    qa.add_argument("--seed", type=int, default=42)
    qa.add_argument("--model")
    qa.set_defaults(handler=_cs_qa)


def _cs_build(args, context):
    from clir_bench.domains.chemistry.codeswitch.builder import build_code_switched

    return build_code_switched(context, args)


def _cs_qa(args, context):
    from clir_bench.domains.chemistry.codeswitch.variant_qa import generate_variant_qa

    return generate_variant_qa(context, args)


def _register_progressive(registrar) -> None:
    group = registrar.group(
        "progressive",
        help="Dose-response benchmark: swap one more term per rung and measure decay",
    )

    build = group.add_parser("build", help="Build the cumulative ladder corpus")
    build.add_argument("--steps", type=int, default=5, help="Ladder depth (default: 5)")
    build.add_argument("--modes", default="B,C,D,F", help="Swap modes to draw from")
    build.add_argument("--source", default="gp")
    build.add_argument("--limit", type=int)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(handler=_pcs_build)

    qa = group.add_parser("qa", help="Generate one fixed query per base document")
    qa.add_argument("--source", default="gp", help="Corpus source the ladder was built from")
    qa.add_argument("--strategy", type=int, default=4, choices=[1, 2, 3, 4])
    qa.add_argument("--limit", type=int)
    qa.add_argument("--workers", type=int, default=1)
    qa.add_argument("--seed", type=int, default=42)
    qa.add_argument("--model")
    qa.add_argument("--grader-model")
    qa.set_defaults(handler=_pcs_qa)

    publish = group.add_parser("publish", help="Publish the progressive benchmark")
    publish.add_argument("--repo")
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--private", action="store_true")
    publish.set_defaults(handler=_pcs_publish)

    build_all = group.add_parser("all", help="Build, generate queries and publish in one go")
    build_all.add_argument("--steps", type=int, default=5)
    build_all.add_argument("--modes", default="B,C,D,F")
    build_all.add_argument("--source", default="gp")
    build_all.add_argument("--limit", type=int)
    build_all.add_argument("--seed", type=int, default=42)
    build_all.add_argument("--strategy", type=int, default=4, choices=[1, 2, 3, 4])
    build_all.add_argument("--workers", type=int, default=1)
    build_all.add_argument("--model")
    build_all.add_argument("--grader-model")
    build_all.add_argument("--repo")
    build_all.add_argument("--dry-run", action="store_true")
    build_all.add_argument("--private", action="store_true")
    build_all.set_defaults(handler=_pcs_all)

    evaluate = group.add_parser(
        "eval",
        help="Measure retrieval decay by ladder depth (loads models; use on a compute node)",
    )
    evaluate.add_argument("models", nargs="*")
    evaluate.add_argument("--repo", help="Published progressive dataset")
    evaluate.add_argument("--corpus-repo", help="Shared haystack repo")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--allow-local", action="store_true", help="Confirm local model loading")
    evaluate.set_defaults(handler=_pcs_eval)


def _pcs_build(args, context):
    from clir_bench.domains.chemistry.codeswitch.progressive import build_progressive

    return build_progressive(context, args)


def _pcs_qa(args, context):
    from clir_bench.domains.chemistry.codeswitch.progressive_qa import generate_progressive_qa

    return generate_progressive_qa(context, args)


def _pcs_publish(args, context):
    from clir_bench.domains.chemistry.codeswitch.progressive_publish import publish_progressive

    return publish_progressive(context, args)


def _pcs_all(args, context):
    for step in (_pcs_build, _pcs_qa, _pcs_publish):
        code = step(args, context)
        if code:
            return code
    return 0


def _pcs_eval(args, context):
    from clir_bench.domains.chemistry.codeswitch.progressive_eval import evaluate_progressive

    return evaluate_progressive(context, args)


__all__ = ["register_cli"]
