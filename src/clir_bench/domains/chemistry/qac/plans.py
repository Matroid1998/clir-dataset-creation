"""Select chemistry source documents and ask in each selected document's language.

The existing plan names control sampling order and pool size. Every plan keeps
an explicit source document identity and restricts generation to the supported
languages with chemistry prompts; no plan invents a missing-language target.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from clir_bench.core import corpus as corpus_io
from clir_bench.core.context import AppContext
from clir_bench.core.domain import SourceSpec
from clir_bench.core.qagen import STRATEGY_NAMES, STRATEGY_RANDOM_EXISTING, PlanItem
from clir_bench.domains.chemistry import vocabulary as vocab

STRATEGIES = (STRATEGY_RANDOM_EXISTING,)


def _languages(args: argparse.Namespace, context: AppContext, source: SourceSpec) -> list[str]:
    """Validate language filters; the selected document supplies the actual language."""
    supported = set(vocab.WORKING_LANGUAGES)
    for option in ("langs", "priority_langs"):
        unknown = set(getattr(args, option, None) or ()) - supported
        if unknown:
            raise ValueError(
                f"Unsupported chemistry --{option.replace('_', '-')}: {', '.join(sorted(unknown))}; "
                f"choose from {', '.join(vocab.WORKING_LANGUAGES)}"
            )
    requested = getattr(args, "langs", None)
    return list(dict.fromkeys(requested or [
        language for language in (source.languages or context.languages.working)
        if language in supported
    ]))


def _modes(args: argparse.Namespace) -> list[str]:
    requested = getattr(args, "modes", None)
    if getattr(args, "decider_model", None):
        if requested is not None:
            raise ValueError("--decider-model cannot be combined with --modes")
        # Sample each document once; eligibility replaces this placeholder.
        return [vocab.MODES[0]]
    modes = list(dict.fromkeys(requested or vocab.MODES))
    unknown = set(modes) - set(vocab.MODES)
    if unknown:
        raise ValueError(
            f"Unknown chemistry modes: {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(vocab.MODES)}"
        )
    return modes


def _eligible_documents(grouped, context, args, languages):
    """Return distinct usable endpoints; ambiguous identities cannot be resolved safely."""
    schema = context.schema
    excluded_documents, excluded_families = set(), set()
    path = getattr(args, "exclude_from", None)
    if path and Path(path).exists():
        for row in corpus_io.read_rows(path):
            identity = (row.get("source_document_id") or row.get("corpus_id")
                        or schema.id_of(row))
            if identity:
                excluded_documents.add(str(identity).strip())
            elif schema.family_of(row):
                # Old outputs without endpoint identity retain their family-wide exclusion.
                excluded_families.add(schema.family_of(row))

    documents = {}
    for family, rows in grouped.items():
        if not family or family in excluded_families:
            continue
        for row in rows:
            if schema.language_of(row) not in languages or not schema.text_of(row):
                continue
            identity = schema.id_of(row)
            if not identity:
                raise ValueError(f"Chemistry source document in {family!r} has no {schema.id_field!r}")
            if identity in excluded_documents:
                continue
            if schema.family_of(row) != family:
                raise ValueError(f"Chemistry document {identity!r} has a mismatched publication")
            if identity in documents:
                raise ValueError(f"Duplicate chemistry source document identity: {identity!r}")
            documents[identity] = row
    return [documents[identity] for identity in sorted(documents)]


def _expand(documents, context, modes, args, *, rotate=True):
    if not documents:
        return []
    count = min(args.questions, len(documents))
    schema = context.schema
    items = []
    for index, mode in enumerate(modes):
        start = (index * count) % len(documents) if rotate else 0
        selected = (documents[start:] + documents[:start])[:count]
        for row in selected:
            items.append(PlanItem(
                family=schema.family_of(row), language=schema.language_of(row), mode=mode,
                document_id=schema.id_of(row), strategy=STRATEGY_RANDOM_EXISTING,
                metadata={"strategy_name": STRATEGY_NAMES[STRATEGY_RANDOM_EXISTING]},
            ))
    return items


def _prepare(context, source, grouped, args):
    if args.questions <= 0:
        raise ValueError("total questions must be positive")
    languages = _languages(args, context, source)
    modes = _modes(args)
    documents = _eligible_documents(grouped, context, args, languages)
    return modes, documents, random.Random(args.seed)


def balanced_plan(*, context: AppContext, source: SourceSpec,
                  grouped: Mapping[str, Sequence[Mapping[str, str]]],
                  args: argparse.Namespace) -> list[PlanItem]:
    """Sample a document pool, then assign up to the requested count per persona."""
    modes, documents, rng = _prepare(context, source, grouped, args)
    pool_size = getattr(args, "pool", None)
    if pool_size is not None and pool_size <= 0:
        raise ValueError("document pool must be positive")
    pool_size = pool_size or max(args.questions * len(modes), 1)
    pool = rng.sample(documents, min(pool_size, len(documents)))
    return _expand(pool, context, modes, args)


def uniform_plan(*, context: AppContext, source: SourceSpec,
                 grouped: Mapping[str, Sequence[Mapping[str, str]]],
                 args: argparse.Namespace) -> list[PlanItem]:
    """Sample distinct source documents uniformly for each persona."""
    modes, documents, rng = _prepare(context, source, grouped, args)
    rng.shuffle(documents)
    return _expand(documents, context, modes, args)


def coverage_plan(*, context: AppContext, source: SourceSpec,
                  grouped: Mapping[str, Sequence[Mapping[str, str]]],
                  args: argparse.Namespace) -> list[PlanItem]:
    """Prefer documents actually written in the requested priority languages."""
    modes, documents, rng = _prepare(context, source, grouped, args)
    rng.shuffle(documents)
    priority = list(dict.fromkeys(getattr(args, "priority_langs", None) or ()))
    documents.sort(key=lambda row: (
        priority.index(context.schema.language_of(row))
        if context.schema.language_of(row) in priority else len(priority)
    ))
    return _expand(documents, context, modes, args, rotate=False)


PLANS: Mapping[str, Callable[..., list[PlanItem]]] = {
    "balanced": balanced_plan,
    "uniform": uniform_plan,
    "coverage": coverage_plan,
}


__all__ = ["PLANS", "STRATEGIES", "balanced_plan", "coverage_plan", "uniform_plan"]
