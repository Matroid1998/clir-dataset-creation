"""Enumerate eligible source targets and render the existing corpus payloads."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from fractions import Fraction
from pathlib import Path

from clir_bench.core import corpus
from clir_bench.domains.chemistry.qac import decider as patent_decider
from clir_bench.domains.legal.qac import decider as legal_decider
from clir_bench.domains.legal.qac import persona_routing
from clir_bench.domains.legal.structure.profile import ProfileArticleIndex

from .qac_batch_content import LEGAL, digest, freeze_payload, patent_payload


@dataclass(frozen=True)
class SourceTarget:
    key: str
    parent: str
    published: str
    languages: tuple[str, ...]
    value: object


def round_order(pool, config):
    """Spread recent EUR-Lex targets across acts and concentrate the older draw."""
    cutoff = config["recent_since"]
    seeded = lambda item: digest(f"{config['seed']}:{item.key}")
    if config["source"] != "eurlex":
        return sorted(pool, key=lambda item: (len(item.languages), seeded(item)))

    acts = defaultdict(list)
    for target in sorted(pool, key=seeded):
        acts[target.parent].append(target)
    recent, older = [], []
    for group in acts.values():
        if cutoff and group[0].published >= cutoff:
            recent.append(group)
        else:
            older.append(group)
    # First reach separate newer acts when they are represented by few articles.
    interleaved = [group[offset] for offset in range(max(map(len, recent), default=0))
                   for group in recent if offset < len(group)]
    return interleaved + [item for group in sorted(older, key=len, reverse=True) for item in group]


def _eurlex_targets(path, config, used):
    module = LEGAL["eurlex"]
    index = ProfileArticleIndex(path)
    with index.profile.file("selected_acts.json").open(encoding="utf-8") as stream:
        dates = {row["celex_id"]: date.fromisoformat(row["date"][:10]).isoformat()
                 for row in json.load(stream)}
    languages = tuple(config["languages"])
    targets = module.select(
        index, n=len(index.by_eli), seed=config["seed"], languages=languages, modes=("auto",),
        strata=(("all", 0, float("inf"), 1.0),),
        max_per_act=None if config["no_act_cap"] else 2,
    )
    pool = []
    for target in targets:
        # A normalized ELI identifies the same article across profile and HTTP variants.
        key = target.eli_id.removeprefix("https://").removeprefix("http://")
        if key not in used:
            pool.append(SourceTarget(key, target.celex_id, dates[target.celex_id], languages, target))
    return round_order(pool, config), index


def groups(context, config, profiles, used, scan_limit):
    source = config["source"]
    languages = tuple(config["languages"])
    if source == "eurlex":
        for path in profiles:
            yield _eurlex_targets(Path(path), config, used)
        return
    if source == "un":
        module = LEGAL["un"]
        path = context.workspace.data("un_blocks")
        index = module.ctx.BlockIndex(
            blocks_path=path / "blocks_en.jsonl", docs_path=path / "docs_en.jsonl",
            status_path=path / "reference_status_en.jsonl",
            sixway_dir=context.workspace.data("un_parallel"),
        )
        chosen = module.select(
            index, n=scan_limit, seed=config["seed"], languages=("en",), modes=("auto",),
            excluded_blocks=frozenset(used),
        )
        yield [SourceTarget(item.block_id, item.doc_id, "", languages, item)
               for item in chosen], index
        return

    specification = context.domain.source(source)
    grouped = corpus.load_grouped(context.workspace.corpus_csv(specification), context.schema)
    context.domain.qac_validate_corpus(grouped, context.schema)
    pool = []
    for publication, endpoints in grouped.items():
        if publication in used:
            continue
        dates = {row["publication_date"] for row in endpoints}
        if len(dates) != 1:
            continue
        raw = next(iter(dates))
        if len(raw) != 8:
            continue
        try:
            published = date.fromisoformat(f"{raw[:4]}-{raw[4:6]}-{raw[6:]}").isoformat()
        except ValueError:
            continue
        versions = {row["language"]: row for row in endpoints if row["language"] in languages}
        if versions:
            pool.append(SourceTarget(publication, publication, published, tuple(versions), versions))
    yield round_order(pool, config), None


def assess(context, source, item, language, index, *, retries):
    if source == "gp":
        packet = patent_payload(context, item.value[language])
        return patent_decider.decide_eligibility(packet["text"], retries=retries)
    module = LEGAL[source]
    payload = module.prepare_payload(replace(item.value, language="en"), index)
    if payload is None:
        return None
    return legal_decider.decide_eligibility(source, payload.text, retries=retries)


def load_un_languages(index, assigned):
    module = LEGAL["un"]
    needed = defaultdict(set)
    for item, language, _, _ in assigned:
        if language != "en":
            needed[language].add(item.parent)
            needed[language].update(module._cited_doc_ids(index, item.value))
    for language, documents in needed.items():
        # Preload once per language for the whole selected group.
        index.preload_translations([language], documents)


def prepare(context, source, item, index, language, mode):
    if source == "gp":
        return {}, patent_payload(context, item.value[language]), {}
    target = replace(item.value, language=language, mode=mode)
    payload = LEGAL[source].prepare_payload(target, index)
    if payload is None or not payload.target.texts.get(language, "").strip():
        return None
    complete = (target.reference_complete and not target.n_unresolved
                if source == "un" else target.complete)
    if not complete:
        return None
    metadata = {}
    if source == "eurlex":
        metadata = {
            "eurlex_dataset_identity": json.dumps(index.identity, sort_keys=True),
            "eurlex_dataset_format": "eurlex-resources-v1",
        }
    return asdict(target), freeze_payload(source, payload, language), metadata


def persona(context, config, eligibility, counts, key):
    source = config["source"]
    configured = context.setting("persona_weights")
    if source == "gp":
        result = patent_decider.choose(
            eligibility, counts, patent_decider.resolve_weights(configured),
            seed=config["seed"], identity=key,
        )
        return result["mode"], patent_decider.row_metadata(
            result, digest(patent_decider.eligibility_prompt_text())
        )
    result = persona_routing.choose(
        source, eligibility, counts, persona_routing.resolve_weights(configured),
        seed=config["seed"], identity=key,
    )
    return result["generation_mode"], legal_decider.row_metadata(result)


def winner_selection(winners, config):
    """Fill language quotas while preserving the two corpus-level recency checks."""
    quotas = Counter(config["quotas"])
    cutoff = config["recent_since"]
    ordered = sorted(winners, key=lambda row: (-float(row["total_score"]), row["target_id"]))
    if not cutoff:
        selected = []
        for row in ordered:
            language = row["question_language"]
            if quotas[language]:
                selected.append(row)
                quotas[language] -= 1
        return selected

    recent = [row for row in ordered if row["source_date"] >= cutoff]
    older = [row for row in ordered if row["source_date"] < cutoff]
    selected, recent_acts = [], set()
    eurlex = config["source"] == "eurlex"
    first_by_act, remainder = [], []
    if eurlex:
        # Assign distinct recent acts to available language slots before taking
        # extra articles from those acts. Augmenting paths avoid losing a rare
        # language slot to an act that can be represented in another language.
        by_act = defaultdict(dict)
        for row in recent:
            by_act[row["document_id"]].setdefault(row["question_language"], row)
        assigned = defaultdict(list)

        def match(act, visited):
            for lang in by_act[act]:
                if lang in visited:
                    continue
                visited.add(lang)
                if len(assigned[lang]) < quotas[lang]:
                    assigned[lang].append(act)
                    return True
                for position, other in enumerate(assigned[lang]):
                    if match(other, visited):
                        assigned[lang][position] = act
                        return True
            return False

        for act in by_act:
            match(act, set())
        first_by_act = [by_act[act][lang] for lang, acts in assigned.items() for act in acts]
        keys = {row["target_id"] for row in first_by_act}
        remainder = [row for row in recent if row["target_id"] not in keys]
    for row in (first_by_act + remainder if eurlex else recent):
        language = row["question_language"]
        if quotas[language]:
            selected.append(row)
            quotas[language] -= 1
            recent_acts.add(row["document_id"])

    older_budget = config["questions"] - config["min_recent"]
    if eurlex:
        by_act = defaultdict(list)
        for row in older:
            by_act[row["document_id"]].append(row)

        def capacity(rows):
            counts = Counter(row["question_language"] for row in rows)
            return sum(min(quotas[language], count) for language, count in counts.items())

        share = Fraction(str(config["min_recent_share"]))
        max_older_acts = int(len(recent_acts) * (1 - share) / share) if share else len(by_act)
        cohorts = []
        for _ in range(max_older_acts):
            if not by_act:
                break
            act = max(by_act, key=lambda key: (capacity(by_act[key]), key))
            rows = by_act.pop(act)
            if capacity(rows) == 0:
                break
            picked = []
            for row in rows:
                language = row["question_language"]
                if quotas[language] and older_budget:
                    picked.append(row)
                    quotas[language] -= 1
                    older_budget -= 1
            cohorts.extend(picked)
        return selected + cohorts

    for row in older:
        language = row["question_language"]
        if quotas[language] and older_budget:
            selected.append(row)
            quotas[language] -= 1
            older_budget -= 1
    return selected
