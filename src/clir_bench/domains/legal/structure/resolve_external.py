"""
Stage 5 -- resolve cross-act references to articles that are in the corpus.

``references.py`` records every "Article N of Regulation (EC) No 2792/1999"
citation in ``external_references.jsonl`` with the act's surface form and
nothing else: the act it names may or may not be in the corpus, and its
identifier may or may not be readable. This stage reads those rows once and
sorts them into two files:

    external_edges.jsonl        (source article) -> (article of ANOTHER act),
                                only when the cited act is in the corpus, its
                                identifier parsed unambiguously and the cited
                                article number exists in that act
    unresolved_external.jsonl   every other row, each with the reason it could
                                not be resolved -- anaphoric, act not in the
                                corpus, unnumbered instrument, and so on

and derives a third:

    reference_status.jsonl      one row per English article: how many article
                                citations it makes, how many are resolved
                                (same act or other act), how many are not, and
                                a single ``complete`` verdict

``complete`` is what question generation keys on. A question written about an
article whose citations cannot all be followed is a question about text the
generator only half saw; ``eurlex_batch.select`` therefore draws targets from
complete articles only. Annex citations count the same way: resolved annexes
are supplied to the generator (the REFERENCED ANNEXES payload block), and an
annex citation that cannot be pinned to one annex in the corpus -- label not
in the cited act's inventory, a bare "the Annex" with zero or several
candidates, or an act outside the corpus -- makes the article incomplete.

Resolution is deliberately conservative -- see ``act_designation`` for the
rules and why the year/number order is never guessed. Anaphoric citations
("of that Regulation", "thereof", a bare "the Directive") are recorded as
unresolved: nothing in the surface identifies the act, and inventing an edge
would be indistinguishable from a real one downstream.

``external_references.jsonl`` is left exactly as extracted, so this stage can
be re-run on its own after a parser change without re-extracting.

Usage:
    python -m clir_bench.domains.legal.structure.resolve_external
    python -m clir_bench.domains.legal.structure.resolve_external --audit
    python -m clir_bench.domains.legal.structure.resolve_external --sample 300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from clir_bench.domains.legal.structure import ACT_LANGUAGES, cellar, paths, references
from clir_bench.domains.legal.structure.act_designation import (
    is_bare_designator,
    parse_act_designation,
    resolve_celex,
)

STATS_JSON = paths.STRUCTURE_DIR / "external_resolution_stats.json"
AUDIT_JSON = paths.REPORTS_DIR / "external_resolution_audit.json"
REVIEW_MD = paths.STRUCTURE_DIR / "external_review_sample.md"

# Reasons a row lands in unresolved_external.jsonl. Kept as one tuple so the
# stats and the review file can enumerate them in a stable order.
REASONS = (
    "anaphoric", "truncated_surface", "designator_unsupported", "no_identifier",
    "malformed_surface", "compound_surface", "doc_type_not_in_corpus", "self_reference",
    "out_of_corpus", "target_act_quarantined", "target_article_not_in_inventory",
    "unparseable_enumeration", "target_annex_not_in_inventory", "target_annex_ambiguous",
    "target_annex_bare_no_annex",
)


# -- corpus lookup ---------------------------------------------------------- #

@dataclass
class Corpus:
    """What resolution needs to know about the acts on disk.

    ``act_eli`` is looked up from the article records, never derived from the
    CELEX id (see ``structure/__init__``: 32014R0680 is ``reg_impl``, not
    ``reg``). ``inventories`` are per language because the join in
    ``project.py`` is per language too, and an edge into an article that a
    language version lacks would be a broken link for that language.
    """

    act_eli: dict[str, str] = field(default_factory=dict)
    inventories: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    # celex -> [(article_number, eli_id)] in English, file order.
    articles_en: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list))
    # Annexes of each act: labelled ones by label (what "Annex I to X" can
    # name), every one per language (for availability), and the full English
    # list (a bare "the Annex to X" resolves only against exactly one).
    annex_eli: dict[str, dict[str, str]] = field(
        default_factory=lambda: defaultdict(dict))
    annex_langs: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    annexes_en: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: defaultdict(list))
    quarantined: set[str] = field(default_factory=set)

    def __contains__(self, celex: object) -> bool:
        return celex in self.act_eli

    def target_eli(self, celex: str, number: str) -> str:
        return cellar.article_eli(self.act_eli[celex], number)


def load_corpus(articles_path: Path | None = None,
                quarantine_path: Path | None = None, *, profile=None) -> Corpus:
    """One streaming pass over the article records; nothing else is loaded."""
    profile_paths = profile or paths
    corpus = Corpus()
    with (articles_path or profile_paths.ARTICLES_JSONL).open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            unit_type = row.get("unit_type")
            if unit_type not in ("article", "annex") or not row.get("eli_id"):
                continue
            celex, language = row["celex_id"], row["language"]
            if row.get("act_eli"):
                corpus.act_eli.setdefault(celex, row["act_eli"])
            if unit_type == "annex":
                annex_id = row.get("annex_id") or ""
                corpus.annex_langs[celex][language].add(annex_id)
                if language == "en":
                    if row.get("annex_labelled"):
                        # A label seen twice in one act identifies neither
                        # record; resolving against it would serve whichever
                        # happened to load last. Collisions are dropped.
                        if annex_id in corpus.annex_eli[celex]:
                            corpus.annex_eli[celex][annex_id] = None
                        else:
                            corpus.annex_eli[celex][annex_id] = row["eli_id"]
                    corpus.annexes_en[celex].append((annex_id, row["eli_id"]))
                continue
            corpus.inventories[celex][language].add(row["article_number"])
            if language == "en":
                corpus.articles_en[celex].append((row["article_number"], row["eli_id"]))
    quarantine = quarantine_path or profile_paths.QUARANTINE_JSONL
    if quarantine.exists():
        with quarantine.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                celex = row.get("celex_id") or row.get("celex")
                if celex:
                    corpus.quarantined.add(celex)
    return corpus


# -- one row ---------------------------------------------------------------- #

def resolve_row(row: dict, corpus: Corpus) -> tuple[list[dict], list[dict]]:
    """(edges, unresolved) for one external_references.jsonl row.

    A row is one enumeration ("Articles 3, 5 and 7") governed by one act
    surface, so it can yield several edges, and a mix of edges and unresolved
    entries when some cited numbers do not exist in the target act.
    """
    surface = row.get("target_act_surface", "") or ""
    base = {k: v for k, v in row.items() if k != "available_languages"}

    def unresolved(reason: str, **extra) -> dict:
        return {**base, "reason": reason, **extra}

    anaphoric_rules = ("external_thereof", "external_annex_thereto")
    if row.get("target_act_anaphoric") or row.get("rule") in anaphoric_rules \
            or is_bare_designator(surface):
        return [], [unresolved("anaphoric")]

    celex, reason = resolve_celex(surface, corpus, source_celex=row.get("source_celex"))
    if celex is None:
        parsed = parse_act_designation(surface)
        extra = {"parsed_celex": parsed.celex} if parsed else {}
        return [], [unresolved(reason, **extra)]
    if celex in corpus.quarantined:
        return [], [unresolved("target_act_quarantined", parsed_celex=celex)]

    if row.get("rule") == "external_annex_of_act":
        return _resolve_annex_row(row, corpus, celex, unresolved)

    enumerations = references.find_enumerations(row["surface_form"])
    if not enumerations:
        return [], [unresolved("unparseable_enumeration", parsed_celex=celex)]

    parsed = parse_act_designation(surface)
    source_languages = row.get("available_languages") or list(ACT_LANGUAGES)
    inventory = corpus.inventories[celex]
    edges: list[dict] = []
    misses: list[dict] = []
    seen: set[str] = set()
    range_log: list[dict] = []
    for target in references.expand(enumerations[0], log=range_log,
                                    source=row["source_article_id"]):
        number = target["number"]
        if number in seen:
            continue
        seen.add(number)
        if number not in inventory.get("en", set()):
            misses.append(unresolved("target_article_not_in_inventory",
                                     target_celex=celex, target_article_number=number))
            continue
        edges.append({
            "source_article_id": row["source_article_id"],
            "source_celex": row["source_celex"],
            "source_unit_type": row.get("source_unit_type", "article"),
            "source_article_number": row.get("source_article_number", ""),
            "target_article_id": corpus.target_eli(celex, number),
            "target_celex": celex,
            "target_article_number": number,
            "target_unit_type": "article",
            "target_act_surface": surface,
            "target_act_eli": corpus.act_eli[celex],
            "shape": parsed.shape if parsed else "",
            "surface_form": row["surface_form"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "target_paragraph_surface": target["sub"],
            "rule": "external_of_act",
            "expansion": target["rule"],
            "available_languages": [lg for lg in source_languages
                                    if number in inventory.get(lg, set())],
            "surface_language": "en",
        })
    return edges, misses


def _annex_edge(row: dict, corpus: "Corpus", celex: str, annex_id: str,
                target_eli: str) -> dict:
    source_languages = row.get("available_languages") or list(ACT_LANGUAGES)
    parsed = parse_act_designation(row.get("target_act_surface", ""))
    return {
        "source_article_id": row["source_article_id"],
        "source_celex": row["source_celex"],
        "source_unit_type": row.get("source_unit_type", "article"),
        "source_article_number": row.get("source_article_number", ""),
        "target_article_id": target_eli,
        "target_celex": celex,
        "target_article_number": "",
        "target_annex_id": annex_id,
        "target_unit_type": "annex",
        "target_act_surface": row.get("target_act_surface", ""),
        "target_act_eli": corpus.act_eli[celex],
        "shape": parsed.shape if parsed else "",
        "surface_form": row["surface_form"],
        "char_start": row["char_start"],
        "char_end": row["char_end"],
        "target_paragraph_surface": "",
        "rule": "external_annex_of_act",
        "expansion": "single",
        "available_languages": [lg for lg in source_languages
                                if annex_id in corpus.annex_langs[celex].get(lg, set())],
        "surface_language": "en",
    }


def _resolve_annex_row(row: dict, corpus: "Corpus", celex: str,
                       unresolved) -> tuple[list[dict], list[dict]]:
    """Edges for "Annex I to Regulation X" / "the Annex to Regulation X".

    Labelled citations are validated against the target act's labelled annex
    inventory; a bare "the Annex" resolves only when the target act has exactly
    one annex. Anything else is a miss, never a guess.
    """
    labelled = references.find_annex_references(row["surface_form"])
    if not labelled:
        annexes = corpus.annexes_en.get(celex, [])
        if len(annexes) == 1:
            annex_id, target_eli = annexes[0]
            return [_annex_edge(row, corpus, celex, annex_id, target_eli)], []
        reason = "target_annex_ambiguous" if annexes else "target_annex_bare_no_annex"
        return [], [unresolved(reason, target_celex=celex)]
    edges: list[dict] = []
    misses: list[dict] = []
    inventory = corpus.annex_eli.get(celex, {})
    for label in references.expand_annexes(labelled[0]):
        target_eli = inventory.get(label)
        if target_eli is None:   # absent, or a colliding label stored as None
            misses.append(unresolved("target_annex_not_in_inventory",
                                     target_celex=celex, target_annex_id=label))
            continue
        edges.append(_annex_edge(row, corpus, celex, label, target_eli))
    return edges, misses


# -- per-article status ----------------------------------------------------- #

@dataclass
class _Tally:
    internal: set[str] = field(default_factory=set)
    internal_not_in_inventory: int = 0
    external: set[str] = field(default_factory=set)
    unresolved: Counter = field(default_factory=Counter)
    annex_targets: set[str] = field(default_factory=set)
    annex_external: set[str] = field(default_factory=set)
    # Unresolved citations that were ANNEX citations, whatever the reason --
    # so cites_annex is true even when the miss is act-level (out_of_corpus).
    annex_unresolved: int = 0


def status_row(eli_id: str, celex: str, number: str, tally: _Tally | None) -> dict:
    t = tally or _Tally()
    n_unresolved = sum(t.unresolved.values())
    return {
        "eli_id": eli_id,
        "celex_id": celex,
        "article_number": number,
        "n_internal": len(t.internal),
        "n_internal_not_in_inventory": t.internal_not_in_inventory,
        "n_external_resolved": len(t.external),
        "n_external_unresolved": n_unresolved,
        "unresolved_reasons": dict(t.unresolved),
        "cites_annex": bool(t.annex_targets or t.annex_external or t.annex_unresolved),
        "n_annex_targets": len(t.annex_targets),
        "n_annex_external_resolved": len(t.annex_external),
        # Unresolved annex citations count in ``unresolved`` alongside article
        # ones, so one condition covers both: nothing the article cites may be
        # left unresolved.
        "complete": t.internal_not_in_inventory == 0 and n_unresolved == 0,
    }


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


# -- driver ----------------------------------------------------------------- #

def resolve(*, external_path: Path | None = None, internal_path: Path | None = None,
            dropped_path: Path | None = None, corpus: Corpus | None = None,
            edges_out: Path | None = None, unresolved_out: Path | None = None,
            status_out: Path | None = None, stats_out: Path | None = None,
            profile=None) -> dict:
    profile_paths = profile or paths
    external_path = external_path or profile_paths.EXTERNAL_REFS_JSONL
    internal_path = internal_path or profile_paths.INTERNAL_EDGES_JSONL
    dropped_path = dropped_path or profile_paths.DROPPED_JSONL
    edges_out = edges_out or profile_paths.EXTERNAL_EDGES_JSONL
    unresolved_out = unresolved_out or profile_paths.UNRESOLVED_EXTERNAL_JSONL
    status_out = status_out or profile_paths.REFERENCE_STATUS_JSONL
    stats_out = stats_out or (profile_paths.STRUCTURE_DIR / "external_resolution_stats.json" if profile else STATS_JSON)

    corpus = corpus or (load_corpus(profile=profile) if profile else load_corpus())
    stats: Counter = Counter()
    tallies: dict[str, _Tally] = defaultdict(_Tally)
    target_acts: set[str] = set()
    source_articles: set[str] = set()
    reversed_hits = 0

    edges_sink = references._Sink(edges_out)
    unresolved_sink = references._Sink(unresolved_out)
    emitted: set[tuple] = set()
    for row in _iter_jsonl(external_path):
        stats["rows"] += 1
        edges, misses = resolve_row(row, corpus)
        is_article = row.get("source_unit_type") == "article"
        is_annex_citation = str(row.get("rule", "")).startswith("external_annex")
        for entry in misses:
            unresolved_sink.write(entry)
            stats[f"unresolved:{entry['reason']}"] += 1
            if is_article:
                tally = tallies[row["source_article_id"]]
                tally.unresolved[entry["reason"]] += 1
                if is_annex_citation:
                    tally.annex_unresolved += 1
        for edge in edges:
            key = (edge["source_article_id"], edge["target_article_id"], edge["char_start"])
            if key in emitted:
                stats["duplicate"] += 1
                continue
            emitted.add(key)
            edges_sink.write(edge)
            stats[f"edges:{edge['source_unit_type']}->{edge['target_unit_type']}"] += 1
            target_acts.add(edge["target_celex"])
            if is_article:
                source_articles.add(edge["source_article_id"])
                tally = tallies[edge["source_article_id"]]
                if edge["target_unit_type"] == "annex":
                    tally.annex_external.add(edge["target_article_id"])
                else:
                    tally.external.add(edge["target_article_id"])
        # Audit only: would reading the identifier the other way round ALSO
        # have named an act in the corpus? The parser never does this; the
        # count is here to show how often the guess would have been available.
        if edges and not row.get("target_act_anaphoric"):
            parsed = parse_act_designation(row["target_act_surface"])
            if parsed and 1952 <= parsed.number <= 2099 and parsed.year <= 9999:
                flipped = f"3{parsed.number}{parsed.letter}{parsed.year % 10000:04d}"
                if flipped != parsed.celex and flipped in corpus:
                    reversed_hits += 1
    edges_sink.close()
    unresolved_sink.close()

    # Same-act edges and inventory drops feed the completeness verdict.
    for edge in _iter_jsonl(internal_path):
        if edge.get("source_unit_type") != "article":
            continue
        tally = tallies[edge["source_article_id"]]
        if edge.get("target_unit_type") == "article":
            if edge["target_article_id"] != edge["source_article_id"]:
                tally.internal.add(edge["target_article_id"])
        elif edge.get("target_unit_type") == "annex":
            tally.annex_targets.add(edge["target_article_id"])
    if dropped_path.exists():
        annex_drop_reasons = {"annex-not-in-inventory": "annex_not_in_inventory",
                              "annex-bare-ambiguous": "annex_bare_ambiguous",
                              "annex-bare-no-annex": "annex_bare_no_annex"}
        for row in _iter_jsonl(dropped_path):
            if row.get("source_unit_type") != "article":
                continue
            if row.get("reason") == "not-in-inventory":
                tallies[row["source_article_id"]].internal_not_in_inventory += 1
            elif row.get("reason") in annex_drop_reasons:
                # A citation of an annex that cannot be pinned down blocks the
                # article exactly as an unresolvable article citation does.
                tally = tallies[row["source_article_id"]]
                tally.unresolved[annex_drop_reasons[row["reason"]]] += 1
                tally.annex_unresolved += 1

    status_sink = references._Sink(status_out)
    for celex, articles in corpus.articles_en.items():
        for number, eli_id in articles:
            row = status_row(eli_id, celex, number, tallies.get(eli_id))
            status_sink.write(row)
            stats["status:complete" if row["complete"] else "status:incomplete"] += 1
            if row["n_external_resolved"]:
                stats["status:with_external"] += 1
    status_sink.close()

    article_rows = sum(v for k, v in stats.items()
                       if k.startswith("edges:article->"))
    summary = {
        "external_rows": stats["rows"],
        "edges": edges_sink.count,
        "edges_from_articles": article_rows,
        "unresolved": unresolved_sink.count,
        "distinct_target_acts": len(target_acts),
        "distinct_source_articles": len(source_articles),
        "reversed_order_would_also_hit_corpus": reversed_hits,
        "corpus_acts": len(corpus.act_eli),
        "status_rows": status_sink.count,
        **{k: v for k, v in sorted(stats.items()) if k != "rows"},
    }
    stats_out.write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(summary: dict) -> None:
    print(f"external rows read      {summary['external_rows']:>8,}")
    print(f"resolved edges          {summary['edges']:>8,}  -> {paths.EXTERNAL_EDGES_JSONL.name}")
    print(f"unresolved rows         {summary['unresolved']:>8,}  -> {paths.UNRESOLVED_EXTERNAL_JSONL.name}")
    print(f"distinct target acts    {summary['distinct_target_acts']:>8,}")
    print(f"citing articles gained  {summary['distinct_source_articles']:>8,}")
    print(f"reversed order would also have hit an act in {summary['reversed_order_would_also_hit_corpus']} cases (never used)")
    print(f"status rows             {summary['status_rows']:>8,}  -> {paths.REFERENCE_STATUS_JSONL.name}")
    print("\nby outcome:")
    for key, value in summary.items():
        if key.startswith(("edges:", "unresolved:", "status:", "duplicate")):
            print(f"  {key:<42} {value:>8,}")


# -- audit against an independent parser ------------------------------------ #

def _parsed_path(celex: str, *, profile=None) -> Path:
    profile_paths = profile or paths
    return profile_paths.PARSED_DIR / celex[1:5] / f"{celex}.en.json"


def audit(*, edges_path: Path | None = None, unresolved_path: Path | None = None,
          corpus_celexes: set[str] | None = None, profile=None) -> dict:
    """Compare our (source article, target act, target article) triples with the
    ``crossReferences`` block of the parse-fmx JSON, produced by a different
    tool. Read-only use of that tool's output; agreement is a sanity check on
    the parser, not ground truth."""
    profile_paths = profile or paths
    edges_path = edges_path or profile_paths.EXTERNAL_EDGES_JSONL
    unresolved_path = unresolved_path or profile_paths.UNRESOLVED_EXTERNAL_JSONL
    if corpus_celexes is None:
        # "In the corpus" means the act has article records -- the same test the
        # resolver applies -- not merely a CELEX id in the candidate list.
        corpus_celexes = {row["celex_id"] for row in _iter_jsonl(profile_paths.REFERENCE_STATUS_JSONL)} \
            if profile_paths.REFERENCE_STATUS_JSONL.exists() else set()

    ours: dict[str, set[tuple[str, str, str]]] = defaultdict(set)      # source celex -> triples
    # (source celex, article) -> {act celex we parsed: surface it came from},
    # for resolved and unresolved rows alike: the parser is what is audited.
    ours_acts: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for edge in _iter_jsonl(edges_path):
        if edge.get("source_unit_type") != "article":
            continue
        if edge.get("target_unit_type") == "annex":
            continue  # the second-opinion parser records article targets only
        key = edge["source_celex"]
        ours[key].add((edge["source_article_number"], edge["target_celex"], edge["target_article_number"]))
        ours_acts[(key, edge["source_article_number"])].setdefault(
            edge["target_celex"], edge["target_act_surface"])
    if unresolved_path.exists():
        for row in _iter_jsonl(unresolved_path):
            if row.get("source_unit_type") == "article" and row.get("parsed_celex"):
                ours_acts[(row["source_celex"], row["source_article_number"])].setdefault(
                    row["parsed_celex"], row.get("target_act_surface", ""))

    agree = only_ours = only_theirs_in = only_theirs_out = 0
    act_agree = act_disagree = 0
    only_ours_samples: list[dict] = []
    only_theirs_samples: list[dict] = []
    act_disagree_samples: list[dict] = []
    acts_compared = 0
    celexes = {c for c, _ in ours_acts}
    for celex in sorted(celexes):
        path = _parsed_path(celex, profile=profile) if profile else _parsed_path(celex)
        if not path.exists():
            continue
        try:
            cross = json.loads(path.read_text(encoding="utf-8")).get("crossReferences") or {}
        except (OSError, ValueError):
            continue
        acts_compared += 1
        theirs: set[tuple[str, str, str]] = set()
        their_acts: dict[str, set[str]] = defaultdict(set)
        for article, refs in (cross.items() if isinstance(cross, dict) else []):
            # Only article-sourced citations of OTHER acts are comparable: our
            # edges here are article-sourced, and an act citing itself by
            # identifier is an internal edge on our side.
            if not str(article)[:1].isdigit():
                continue
            for ref in refs:
                if ref.get("type") != "external" or not ref.get("actCelex"):
                    continue
                if ref["actCelex"] == celex:
                    continue
                their_acts[str(article)].add(ref["actCelex"])
                number = ref.get("articleNumber")
                if number:
                    theirs.add((str(article), ref["actCelex"], str(number)))
        mine = ours.get(celex, set())
        agree += len(mine & theirs)
        for triple in mine - theirs:
            only_ours += 1
            if len(only_ours_samples) < 40:
                only_ours_samples.append({"source_celex": celex, "article": triple[0],
                                          "target": f"{triple[1]}:{triple[2]}"})
        for triple in theirs - mine:
            if triple[1] in corpus_celexes:
                only_theirs_in += 1
                if len(only_theirs_samples) < 40:
                    only_theirs_samples.append({"source_celex": celex, "article": triple[0],
                                                "target": f"{triple[1]}:{triple[2]}"})
            else:
                only_theirs_out += 1
        # Act-level parser agreement, including acts we could not attach.
        for (c, article), acts in ours_acts.items():
            if c != celex:
                continue
            for act, surface in acts.items():
                if act in their_acts.get(article, set()):
                    act_agree += 1
                elif their_acts.get(article):
                    act_disagree += 1
                    if len(act_disagree_samples) < 60:
                        act_disagree_samples.append({
                            "source_celex": celex, "article": article, "surface": surface,
                            "ours": act, "theirs": sorted(their_acts[article])})
    result = {
        "acts_compared": acts_compared,
        "triples_agree": agree,
        "triples_only_ours": only_ours,
        "triples_only_theirs_target_in_corpus": only_theirs_in,
        "triples_only_theirs_target_out_of_corpus": only_theirs_out,
        "act_level_agree": act_agree,
        "act_level_disagree": act_disagree,
        "only_ours_samples": only_ours_samples,
        "only_theirs_in_corpus_samples": only_theirs_samples,
        "act_level_disagree_samples": act_disagree_samples,
    }
    destination = profile_paths.REPORTS_DIR / "external_resolution_audit.json" if profile else AUDIT_JSON
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    return result


# -- manual review sample --------------------------------------------------- #

def _context(text: str, start: int, end: int, width: int = 90) -> str:
    before = text[max(0, start - width):start].replace("\n", " ")
    after = text[end:end + width].replace("\n", " ")
    return f"…{before}«{text[start:end]}»{after}…"


def write_review_sample(n: int, seed: int, out: Path, *, profile=None) -> None:
    profile_paths = profile or paths
    random.seed(seed)
    by_source: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for edge in _iter_jsonl(profile_paths.EXTERNAL_EDGES_JSONL):
        if edge.get("source_unit_type") == "article":
            by_source[edge["source_article_id"]].append(("edge", edge))
    for row in _iter_jsonl(profile_paths.UNRESOLVED_EXTERNAL_JSONL):
        if row.get("source_unit_type") == "article":
            by_source[row["source_article_id"]].append(("unresolved", row))
    sources = sorted(by_source)
    random.shuffle(sources)
    chosen = set(sources[:n])

    texts: dict[str, str] = {}
    with profile_paths.ARTICLES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("language") == "en" and row.get("eli_id") in chosen:
                texts[row["eli_id"]] = row["text"]

    outcomes: Counter = Counter()
    sections: list[str] = []
    for source in sorted(chosen):
        text = texts.get(source, "")
        lines = [f"## {source}", ""]
        for kind, row in sorted(by_source[source], key=lambda kv: kv[1]["char_start"]):
            if kind == "edge":
                outcomes["resolved"] += 1
                status = (f"→ **{row['target_celex']}** Article {row['target_article_number']}"
                          f"  ({row['shape']}, langs {','.join(row['available_languages'])})")
            else:
                outcomes[row["reason"]] += 1
                status = f"→ UNRESOLVED ({row['reason']}"
                if row.get("parsed_celex"):
                    status += f": {row['parsed_celex']}"
                status += ")"
            lines.append(f"- {_context(text, row['char_start'], row['char_end'])}"
                         if text else f"- {row['surface_form']} of {row.get('target_act_surface')}")
            lines.append(f"  {status}")
        lines.append("")
        sections.append("\n".join(lines))

    total = sum(outcomes.values())
    header = ["# EUR-Lex cross-act reference resolution — manual review sample", "",
              f"{len(chosen)} citing articles (seed {seed}); {total} citation sites, "
              f"{outcomes['resolved']} resolved ({outcomes['resolved'] / max(total, 1):.0%}).", "",
              "| outcome | n |", "|---|---|"]
    header += [f"| {k} | {v} |" for k, v in outcomes.most_common()]
    out.write_text("\n".join(header) + "\n\n" + "\n".join(sections), encoding="utf-8")
    print(f"wrote {len(chosen)} citing articles -> {out}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", action="store_true",
                        help="compare existing edges with the parse-fmx crossReferences block")
    parser.add_argument("--sample", type=int, default=0,
                        help="write a manual-review markdown file with this many citing articles")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    paths.ensure_dirs()

    if args.audit:
        result = audit()
        for key, value in result.items():
            if not key.endswith("samples"):
                print(f"{key:<44} {value:>8,}")
        print(f"-> {AUDIT_JSON}")
        return
    if args.sample:
        write_review_sample(args.sample, args.seed, REVIEW_MD)
        return
    print_summary(resolve())


if __name__ == "__main__":
    main()
