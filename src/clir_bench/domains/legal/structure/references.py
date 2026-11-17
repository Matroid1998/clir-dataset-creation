"""
Stage 5 -- extract article-to-article references from English article text.

One pass, English only. The other language versions are reached by the
structural join in ``project.py``, not by re-running this: article numbering is
identical across language versions by drafting rule, while re-extracting per
language measurably drifts (the same act yields 362 English edges and 240 German
ones through the same parser).

The rules fire in the order the spec fixes, and the order is load-bearing:

  1. match ``Article(s) N``, keeping the surface span
  2. expand ranges and lists
  3. whitelist ``of this Regulation/Directive/Decision`` as internal
  4. only then exclude ``of`` + an act designator as external
  5. validate surviving targets against the act's own article inventory
  6. keep surface form and span so paragraph granularity can be added later

Step 3 must precede step 4. Reversed, every explicit ``Article 5 of this
Regulation`` -- the most unambiguous internal reference the drafting style
produces -- is thrown out as external, and that single ordering mistake is the
dominant false-negative source.

Targets are resolved at article granularity only. Paragraph and point surface
forms are recorded unresolved so that granularity can be added later without
re-extracting.

Usage:
    python -m clir_bench.domains.legal.structure.references
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from clir_bench.domains.legal.structure import cellar, paths
from clir_bench.domains.legal.structure.act_designation import parse_act_designation

# How far right of an enumeration an act designator may sit and still govern it.
BRIDGE_WINDOW = 80

# Ranges wider than this are almost always a parse artefact rather than a real
# citation. The VAT Directive genuinely cites "Articles 380 to 390", so the cap
# has to be generous; anything beyond it is logged, never silently dropped.
MAX_RANGE_SIZE = 200

# -- step 1: find enumerations --------------------------------------------- #

_HEAD_RE = re.compile(r"\bArticles?\b\s+(?=\d)", re.I)
# "12", "7a" -- optionally followed by paragraph/point parentheses, "5(1)(a)".
# The gap before a parenthesis is [ \t]* rather than \s*: a citation never breaks
# across lines, but a list item on the next line does start "(1)", and allowing
# newlines let "ARTICLE 13(1)\n\n(1) Telecommunications..." swallow the list
# marker as if it were a further sub-reference.
_ITEM_RE = re.compile(r"(\d+[a-z]?)((?:[ \t]*\([0-9a-z]{1,4}\))*)", re.I)
# Connectors are named rather than numbered: the group indices shifted twice
# while this pattern grew, and reading the wrong one silently turned every range
# into a list.
#
# Two shapes the drafting style produces that a naive pattern misses:
#   ", and Articles 380 to 390"    -- comma *and* conjunction together
#   "Articles 2(2) and (3), 4"     -- list resuming after a parenthesised
#                                     sub-reference of the previous item
# The parenthesised run after the connector is captured as ``subs`` because it
# changes what a range means: in "Articles 41(2) to (5), 45" the "to" spans
# paragraphs 2-5 of Article 41, not Articles 41-45, so a range word followed by
# a sub-reference must not turn the next article number into a range end.
_CONNECTOR_RE = re.compile(
    r"\s*(?:(?P<comma>,)\s*)?"
    r"(?:\b(?P<list>and|or)\b|\b(?P<range>to|through|until)\b)?\s*"
    r"(?P<subs>(?:\(\s*[0-9a-z]{1,4}\s*\)\s*(?:,|\band\b|\bor\b)?\s*)*)"
    r"(?:Articles?\s+)?(?=\d)", re.I)

# -- steps 3 and 4: what governs the enumeration ---------------------------- #

_THIS_ACT_RE = re.compile(
    r"^[\s,;:]*of\s+this\s+(Regulation|Directive|Decision|Act)\b", re.I)

_ACT_DESIGNATOR = (
    r"(?:Framework\s+Decision|Regulations?|Directives?|Decisions?|Treaty|Treaties"
    r"|Protocol|Convention|Charter|Recommendation|Agreement|Act\s+of\s+Accession)")

# Determiners that point at an act named earlier in the sentence. "Article 4 of
# that Regulation" is external exactly as "Article 4 of Regulation (EC) No 847/96"
# is; only "this" means the act being read, and step 3 has already claimed it.
_ANAPHORIC_DET = r"(?:that|the\s+said|the\s+same|those|such)"

# Proper-noun modifiers sit between the determiner and the designator in real
# citations -- "the Seventh Council Directive 83/349/EEC", "the OECD Model Tax
# Convention". The capital is matched case-sensitively via (?-i:...) even though
# the pattern as a whole is case-insensitive: under re.I a bare [A-Z] also
# matches lowercase, which would let ordinary prose bridge the gap and turn
# every trailing "of the ... Regulation" clause into a false external.
_ACT_MODIFIER = r"(?:(?-i:[A-Z])[\w./-]*\s+)"

# "(EC) No 847/96", "83/349/EEC", "2016/679". Optional: an anaphoric citation
# ("of that Regulation") names no identifier at all, which is precisely why those
# are never resolved. Both components allow a single digit -- "Directive 98/8/EC",
# "Directive 2014/6/EU": 170 acts in the corpus are numbered below 10 and were
# unreachable while the second component demanded two digits -- and whitespace
# is tolerated around the slash ("No 1303 /2013" used to truncate the surface
# to "Regulation (EU) No", losing the identifier entirely).
_ACT_IDENTIFIER = (
    r"(?:\s*\((?:EU|EC|EEC|Euratom|CE|CEE)[^)]{0,24}\))?"
    r"(?:\s*No\s*)?"
    r"(?:\s*\d{1,4}\s*/\s*\d{1,4}(?:/[A-Z]{2,8})?)?")

_EXTERNAL_RE = re.compile(
    rf"\bof\s+(?:(?P<anaphoric>{_ANAPHORIC_DET})\s+|the\s+)?"
    rf"(?P<act>{_ACT_MODIFIER}{{0,5}}{_ACT_DESIGNATOR}\b{_ACT_IDENTIFIER})", re.I)

# Words allowed between the enumeration and a governing "of ...". Anything else
# means the "of" belongs to a different clause and does not govern this citation.
_BRIDGE_RE = re.compile(
    r"^(?:[\s,;:()\-‐-―]|\b(?:paragraphs?|points?|subparagraphs?|indents?"
    r"|sentences?|first|second|third|fourth|last|thereto|and|or|to|the|of\s+which)\b"
    r"|\(\s*[0-9a-z]{1,4}\s*\))*$", re.I)

_THEREOF_RE = re.compile(r"^[\s,;:]*thereof\b", re.I)

# -- annex targets ---------------------------------------------------------- #

# "Annex I", "Annexes II and III", "Annex VIIa", "Annex 2". Annexes are cited by
# label, and the label -- not the document's position in the Formex zip -- is
# what `segment.annex_label` uses as the identifier.
# Same-line whitespace only, like _ITEM_RE: a citation never breaks across
# lines, but a flattened Formex table puts "Annex" and "II to IV" in adjacent
# cells, and \s+ glued them into a citation the drafter never wrote.
_ANNEX_HEAD_RE = re.compile(r"\bAnnexe?s?[ \t]+(?=[IVXLC\d])", re.I)

# Annexes are governed by "to" where articles are governed by "of": "Annex I to
# Regulation (EC) No 376/2008" is that regulation's annex, not this act's, and
# 2,549 of the 20,751 article->annex edges in the first extraction were exactly
# this mistake. The whitelist-before-external ordering of the module docstring
# applies here unchanged.
_ANNEX_THIS_ACT_RE = re.compile(
    r"^[\s,;:]*(?:to|of)\s+this\s+(Regulation|Directive|Decision|Act)\b", re.I)
_ANNEX_GOVERNS_RE = re.compile(
    rf"\b(?:to|of)\s+(?:(?P<anaphoric>{_ANAPHORIC_DET})\s+|the\s+)?"
    rf"(?P<act>{_ACT_MODIFIER}{{0,5}}{_ACT_DESIGNATOR}\b{_ACT_IDENTIFIER})", re.I)
# "the Annex thereto" points at an act named earlier; "the Annex hereto" is this
# act's own annex by the same suffix logic.
_ANNEX_THERETO_RE = re.compile(r"^[\s,;:]*(?:thereto|thereof)\b", re.I)
_ANNEX_HERETO_RE = re.compile(r"^[\s,;:]*hereto\b", re.I)
# Structural units that may sit between an annex label and its governing "to":
# "Annex I, part A, to Regulation ...". Kept separate from _BRIDGE_RE so the
# article rules are untouched.
_ANNEX_BRIDGE_RE = re.compile(
    r"^(?:[\s,;:()\-‐-―]|\b(?:parts?|points?|paragraphs?|sections?|tables?"
    r"|appendix|appendices|chapters?|first|second|third|fourth|last|and|or|to|the)\b"
    # A part/point designator: "part A", "point 3", "Section IV". A single
    # capital only -- lower-case single letters are prose, not designators.
    r"|\b(?:(?-i:[A-Z])|[IVXLC]{1,4}|\d{1,3})\b"
    r"|\(\s*[0-9a-zIVXLC]{1,6}\s*\))*$", re.I)

# "Annex I to the standard IOC/T 20/Doc. No 15", "to the international
# standard ISO 3166-1": governed by something that is not an EU act at all.
# Not this act's annex, and not resolvable -- routed external so it blocks
# completeness instead of minting a wrong internal edge. The governor must
# carry an identifier-like token (digits, or a run of capitals), which keeps
# datives ("to the competent authority") and infinitives out.
_ANNEX_NONACT_GOVERNOR_RE = re.compile(
    r"^[\s,;:]*(?:to|of)\s+the\s+(?:(?-i:[a-z])[\w-]*\s+){0,4}"
    r"(?:(?-i:[A-Z]{2,})[\w./-]*|[\w./-]*\d[\w./-]*)")

# "the Annex" with no label: the drafting style for an act with a single annex.
# The lookahead keeps labelled citations ("the Annex I" never occurs, but "the
# Annexes ..." must stay with the labelled path via its own head pattern).
# The lookahead's roman class is case-sensitive: under re.I it also matched
# the "i" of "is" and the "c" of "contains", silently skipping bare citations
# in ordinary prose ("the Annex is amended ...").
_BARE_ANNEX_RE = re.compile(r"\bthe\s+Annex\b(?!\s+(?:(?-i:[IVXLC])|\d))", re.I)
# The optional suffix consumes preceding space only when a letter actually
# follows, so a bare "Annex III to ..." does not trail whitespace into the
# recorded surface form.
# The trailing lookahead stops a roman core from swallowing the first letter
# of the next word: without it, "Annex I to Council Regulation ..." read the
# "C" of "Council" as annex 100, the citation span ended mid-word, the
# governing act became invisible, and the citation range-exploded into false
# internal edges. A letter may not follow the item (the optional single-letter
# suffix has already been consumed).
_ANNEX_ITEM_RE = re.compile(r"([IVXLC]+|\d+)(?:[ \t]*([a-z])\b)?(?![A-Za-z])")
_ANNEX_CONNECTOR_RE = re.compile(
    r"[ \t]*(?:(?P<comma>,)[ \t]*)?(?:\b(?P<list>and|or)\b|\b(?P<range>to)\b)?[ \t]*"
    r"(?:Annexe?s?[ \t]+)?(?=[IVXLC\d])", re.I)

# A correlation table is a flat two-column mapping between a repealed act's
# articles and this one's. Every cell looks like a citation, so extracting from
# one manufactures hundreds of edges that the drafter never wrote.
_CORRELATION_RE = re.compile(
    r"correlation table|table of equivalences|corresponding\s+(?:article|provision)", re.I)


def looks_like_correlation_table(text: str) -> bool:
    if _CORRELATION_RE.search(text or ""):
        return True
    # Or: dense back-to-back article citations with almost no prose between them.
    citations = list(re.finditer(r"\bArticles?\s+\d+[a-z]?", text or ""))
    if len(citations) < 20:
        return False
    gaps = [citations[i + 1].start() - citations[i].end() for i in range(len(citations) - 1)]
    tight = sum(1 for g in gaps if g <= 12)
    return tight / len(gaps) > 0.6


def classify_annex(text: str, start: int, end: int, *,
                   own_celex: str | None = None) -> tuple[str, str, str, bool]:
    """(kind, rule, target_act_surface, anaphoric) for one annex citation.

    Mirrors ``classify`` for articles, with "to" accepted alongside "of" as the
    governing preposition because that is how annexes are cited.
    """
    window = text[end:end + BRIDGE_WINDOW]

    if _ANNEX_THIS_ACT_RE.match(window) or _ANNEX_HERETO_RE.match(window):
        return "internal", "annex_this_act", "", False

    governs = _ANNEX_GOVERNS_RE.search(window)
    if governs and _ANNEX_BRIDGE_RE.match(window[:governs.start()]):
        surface = re.sub(r"\s+", " ", governs.group("act")).strip(" ,;:")
        anaphoric = bool(governs.group("anaphoric"))
        if own_celex and not anaphoric:
            parsed = parse_act_designation(surface)
            if parsed is not None and parsed.celex == own_celex:
                return "internal", "annex_own_act_identifier", "", False
        return "external", "external_annex_of_act", surface, anaphoric

    if _ANNEX_THERETO_RE.match(window):
        return "external", "external_annex_thereto", "", True

    nonact = _ANNEX_NONACT_GOVERNOR_RE.match(window)
    if nonact:
        surface = re.sub(r"\s+", " ", nonact.group(0)).strip(" ,;:")
        return "external", "external_annex_nonact", surface, False

    return "internal", "annex_reference", "", False


def find_bare_annex_references(text: str) -> list[dict]:
    """"the Annex" citations that carry no label at all.

    Resolvable only when the act has exactly one annex -- which is precisely
    the drafting situation that produces this form. The labelled path never
    sees these because ``_ANNEX_HEAD_RE`` requires a label.
    """
    return [{"start": m.start(), "end": m.end(), "raw": m.group(0)}
            for m in _BARE_ANNEX_RE.finditer(text)]


def find_annex_references(text: str) -> list[dict]:
    """Every ``Annex ...`` citation, mirroring `find_enumerations` for articles."""
    results: list[dict] = []
    consumed_to = -1
    for head in _ANNEX_HEAD_RE.finditer(text):
        if head.start() < consumed_to:
            continue
        cursor = head.end()
        items: list[dict] = []
        pending_range = False
        while True:
            item = _ANNEX_ITEM_RE.match(text, cursor)
            if not item:
                break
            core, suffix = item.group(1), (item.group(2) or "").lower()
            number = int(core) if core.isdigit() else _roman(core)
            if number is None:
                break
            items.append({"label": f"{number}{suffix}", "number": number,
                          "suffix": suffix, "end": item.end(),
                          "via_range": pending_range})
            cursor = item.end()
            connector = _ANNEX_CONNECTOR_RE.match(text, cursor)
            if not connector or not any(connector.group(n)
                                        for n in ("comma", "list", "range")):
                break
            pending_range = bool(connector.group("range"))
            cursor = connector.end()
        if not items:
            continue
        consumed_to = items[-1]["end"]
        results.append({"start": head.start(), "end": items[-1]["end"],
                        "raw": text[head.start():items[-1]["end"]], "items": items})
    return results


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman(value: str) -> int | None:
    total = previous = 0
    for char in reversed(value.upper()):
        digit = _ROMAN_VALUES.get(char)
        if digit is None:
            return None
        total = total - digit if digit < previous else total + digit
        previous = max(previous, digit)
    return total or None


def expand_annexes(reference: dict) -> list[str]:
    """Annex labels a citation resolves to, expanding ranges and lists.

    Ranges wider than ``MAX_RANGE_SIZE`` are parse artefacts, not citations --
    no act has hundreds of annexes -- and yield only their endpoints.
    """
    items = reference["items"]
    out: list[str] = []
    for index, item in enumerate(items):
        if item["via_range"] and index > 0:
            low = items[index - 1]
            if (not low["suffix"] and not item["suffix"]
                    and low["number"] < item["number"]
                    and item["number"] - low["number"] <= MAX_RANGE_SIZE):
                out += [str(n) for n in range(low["number"] + 1, item["number"] + 1)]
                continue
        out.append(item["label"])
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def find_enumerations(text: str) -> list[dict]:
    """Every ``Article(s) ...`` enumeration, with its items and overall span.

    A long citation chain repeats the word "Article" -- "Articles 110 and 111,
    Article 125(1), Article 127, ... and Articles 380 to 390". The first head
    consumes the whole chain, so every later head inside it would re-parse a
    suffix of the same citation and emit the same targets again. Heads that fall
    inside an enumeration already consumed are therefore skipped; without this,
    one VAT Directive article alone produced five overlapping enumerations and
    the edge file carried 32% redundant rows.
    """
    results: list[dict] = []
    consumed_to = -1
    for head in _HEAD_RE.finditer(text):
        if head.start() < consumed_to:
            continue
        cursor = head.end()
        items: list[dict] = []
        pending_range = False
        while True:
            item = _ITEM_RE.match(text, cursor)
            if not item:
                break
            items.append({
                "number": item.group(1),
                "sub": (item.group(2) or "").strip(),
                "start": item.start(1),
                "end": item.end(),
                "via_range": pending_range,
            })
            cursor = item.end()
            connector = _CONNECTOR_RE.match(text, cursor)
            # Every part of the connector is optional, so an empty match would
            # happily glue "Article 5 6" together. Require a real connector.
            if not connector or not any(connector.group(name)
                                        for name in ("comma", "list", "range")):
                break
            pending_range = bool(connector.group("range")) and not connector.group("subs")
            cursor = connector.end()
        if not items:
            continue
        consumed_to = items[-1]["end"]
        results.append({
            "start": head.start(),
            "end": items[-1]["end"],
            "raw": text[head.start():items[-1]["end"]],
            "items": items,
        })
    return results


# -- step 2: expand ranges and lists ---------------------------------------- #

def expand(enumeration: dict, *, log: list[dict], source: str) -> list[dict]:
    """Turn enumeration items into concrete target numbers."""
    items = enumeration["items"]
    out: list[dict] = []
    multi = len(items) > 1
    for index, item in enumerate(items):
        if item["via_range"] and index > 0:
            lo_raw, hi_raw = items[index - 1]["number"], item["number"]
            if lo_raw.isdigit() and hi_raw.isdigit():
                lo, hi = int(lo_raw), int(hi_raw)
                if lo < hi and (hi - lo) <= MAX_RANGE_SIZE:
                    for n in range(lo + 1, hi + 1):
                        out.append({"number": str(n), "sub": "",
                                    "rule": "range_expansion"})
                    continue
                log.append({"source": source, "reason": "range-not-expanded",
                            "detail": f"{lo_raw} to {hi_raw}",
                            "raw": enumeration["raw"]})
            else:
                # Suffixed endpoints ("5a to 5c") have no reliable ordering.
                log.append({"source": source, "reason": "range-endpoint-suffixed",
                            "detail": f"{lo_raw} to {hi_raw}",
                            "raw": enumeration["raw"]})
            out.append({"number": hi_raw, "sub": item["sub"], "rule": "range_endpoint"})
            continue
        out.append({
            "number": item["number"], "sub": item["sub"],
            "rule": "list_expansion" if multi else "single",
        })
    # A range's own lower endpoint is already present from the previous item.
    seen: set[tuple[str, str]] = set()
    unique = []
    for entry in out:
        key = (entry["number"], entry["sub"])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


# -- steps 3 and 4: classify ------------------------------------------------ #

def classify(text: str, enumeration: dict, *,
             own_celex: str | None = None) -> tuple[str, str, str, bool]:
    """(kind, rule, target_act_surface, anaphoric) for one enumeration.

    kind is "internal" or "external". ``own_celex`` is the CELEX id of the act
    being read: an act occasionally cites itself by full identifier ("Article 3
    of Regulation (EC) No 1334/2008" inside that very regulation), which is an
    internal reference wearing external clothes.
    """
    window = text[enumeration["end"]:enumeration["end"] + BRIDGE_WINDOW]

    # Step 3 first -- see the module docstring.
    this_act = _THIS_ACT_RE.match(window)
    if this_act:
        return "internal", "this_act_whitelist", "", False

    # Step 4. The designator must actually govern the enumeration, so only
    # connective material may sit between them.
    external = _EXTERNAL_RE.search(window)
    if external and _BRIDGE_RE.match(window[:external.start()]):
        surface = re.sub(r"\s+", " ", external.group("act")).strip(" ,;:")
        anaphoric = bool(external.group("anaphoric"))
        if own_celex and not anaphoric:
            parsed = parse_act_designation(surface)
            if parsed is not None and parsed.celex == own_celex:
                return "internal", "own_act_identifier", "", False
        return "external", "external_of_act", surface, anaphoric

    # "Article 5 thereof" points at an act named earlier in the sentence, which
    # in an enacting article is nearly always a different act. Routed to the
    # external file rather than guessed at: an internal edge invented here would
    # be indistinguishable from a real one downstream.
    if _THEREOF_RE.match(window):
        return "external", "external_thereof", "", True

    return "internal", "bare_article", "", False


# -- driver ----------------------------------------------------------------- #

def load_articles(unit_types=("article",), *, profile=None) -> dict[str, list[dict]]:
    """English records of the given unit types, grouped by act.

    Only English is loaded. References are extracted once from English and
    reach the other languages through the structural join, so holding the other
    three language versions here would triple the memory for nothing -- which
    matters at corpus scale, where the file runs to millions of records.
    """
    profile_paths = profile or paths
    english: dict[str, list[dict]] = defaultdict(list)
    with profile_paths.ARTICLES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["unit_type"] in unit_types and row["language"] == "en":
                english[row["celex_id"]].append(row)
    return english


class _Sink:
    """Append-only JSONL writer that keeps a count. Nothing is held in memory."""

    def __init__(self, path):
        self.path = path
        self.handle = path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, row: dict) -> None:
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1

    # `expand` logs skipped ranges through a list-like interface.
    def append(self, row: dict) -> None:
        self.write(row)

    def close(self) -> None:
        self.handle.close()


def extract(*, profile=None) -> None:
    profile_paths = profile or paths
    profile_paths.ensure_dirs()
    units = load_articles(("article", "recital", "annex"), profile=profile)

    internal = _Sink(profile_paths.INTERNAL_EDGES_JSONL)
    external = _Sink(profile_paths.EXTERNAL_REFS_JSONL)
    dropped = _Sink(profile_paths.DROPPED_JSONL)
    stats = Counter()

    emitted: set[tuple] = set()
    for celex, all_rows in sorted(units.items()):
        rows = [r for r in all_rows if r["unit_type"] == "article"]
        inventory = {r["article_number"] for r in rows}
        # Annexes are addressable targets too, but only those that carried a real
        # "ANNEX <n>" heading -- a positional fallback id is not something a
        # citation in the text can legitimately resolve to.
        # A label two annex records share cannot be resolved to either --
        # "ANNEX I" and "ANNEX 1" normalise to the same id in a handful of
        # acts -- so colliding labels leave the inventory entirely.
        label_counts = Counter(r["annex_id"] for r in all_rows
                               if r["unit_type"] == "annex" and r.get("annex_labelled"))
        annex_inventory = {label for label, n in label_counts.items() if n == 1}
        annex_eli = {r["annex_id"]: r["eli_id"] for r in all_rows
                     if r["unit_type"] == "annex" and r.get("annex_labelled")
                     and label_counts[r["annex_id"]] == 1}
        # Every annex, labelled or not: "the Annex" can only be resolved when
        # there is exactly one, and that one may well be unlabelled.
        all_annexes = [(r["annex_id"], r["eli_id"]) for r in all_rows
                       if r["unit_type"] == "annex"]

        for row in all_rows:
            text = row["text"]
            source_id = row["eli_id"]
            source_kind = row["unit_type"]

            # Correlation tables map a repealed act's articles onto this one's.
            # Every cell reads as a citation, so one table would manufacture
            # hundreds of edges the drafter never wrote.
            if source_kind == "annex" and looks_like_correlation_table(text):
                stats["skipped:correlation-table"] += 1
                dropped.write({"source_article_id": source_id, "source_celex": celex,
                               "reason": "correlation-table", "unit_type": source_kind})
                continue

            # -- annex targets ------------------------------------------------ #
            def write_annex_edge(target_eli: str, label: str, reference: dict,
                                 rule: str) -> None:
                key = (source_id, "anx", label, reference["start"])
                if key in emitted:
                    return
                emitted.add(key)
                internal.write({
                    "source_article_id": source_id,
                    "target_article_id": target_eli,
                    "source_celex": celex,
                    "source_unit_type": source_kind,
                    "target_unit_type": "annex",
                    "source_article_number": row.get("article_number", ""),
                    "target_article_number": "",
                    "target_annex_id": label,
                    "surface_form": reference["raw"],
                    "char_start": reference["start"],
                    "char_end": reference["end"],
                    "target_paragraph_surface": "",
                    "rule": rule,
                    "expansion": "single",
                    "source_is_amending": bool(row.get("is_amending")),
                })
                stats[f"internal:{source_kind}->annex"] += 1

            def write_annex_external(reference: dict, rule: str,
                                     act_surface: str, anaphoric: bool) -> None:
                external.write({
                    "source_article_id": source_id,
                    "source_celex": celex,
                    "source_unit_type": source_kind,
                    "source_article_number": row.get("article_number", ""),
                    "surface_form": reference["raw"],
                    "char_start": reference["start"],
                    "char_end": reference["end"],
                    "target_act_surface": act_surface,
                    "target_act_anaphoric": anaphoric,
                    "rule": rule,
                    "resolved": False,
                })
                stats["external-annex"] += 1

            for reference in find_annex_references(text):
                kind, rule, act_surface, anaphoric = classify_annex(
                    text, reference["start"], reference["end"], own_celex=celex)
                stats[f"annex:{rule}"] += 1
                if kind == "external":
                    write_annex_external(reference, rule, act_surface, anaphoric)
                    continue
                for label in expand_annexes(reference):
                    if label not in annex_inventory:
                        stats["dropped:annex-not-in-inventory"] += 1
                        dropped.write({
                            "source_article_id": source_id, "source_celex": celex,
                            "source_unit_type": source_kind,
                            "source_article_number": row.get("article_number", ""),
                            "surface_form": reference["raw"],
                            "char_start": reference["start"],
                            "target_annex_id": label,
                            "reason": "annex-not-in-inventory",
                        })
                        continue
                    if annex_eli[label] == source_id:
                        continue
                    write_annex_edge(annex_eli[label], label, reference, rule)

            # "the Annex" -- no label, resolvable only against a single annex.
            for reference in find_bare_annex_references(text):
                kind, rule, act_surface, anaphoric = classify_annex(
                    text, reference["start"], reference["end"], own_celex=celex)
                if kind == "external":
                    stats[f"annex:{rule}:bare"] += 1
                    write_annex_external(reference, rule, act_surface, anaphoric)
                    continue
                if len(all_annexes) == 1:
                    label, target_eli = all_annexes[0]
                    if target_eli != source_id:
                        stats["annex:annex_single"] += 1
                        write_annex_edge(target_eli, label, reference, "annex_single")
                elif all_annexes:
                    stats["dropped:annex-bare-ambiguous"] += 1
                    dropped.write({
                        "source_article_id": source_id, "source_celex": celex,
                        "source_unit_type": source_kind,
                        "source_article_number": row.get("article_number", ""),
                        "surface_form": reference["raw"],
                        "char_start": reference["start"],
                        "reason": "annex-bare-ambiguous",
                    })
                else:
                    stats["dropped:annex-bare-no-annex"] += 1
                    dropped.write({
                        "source_article_id": source_id, "source_celex": celex,
                        "source_unit_type": source_kind,
                        "source_article_number": row.get("article_number", ""),
                        "surface_form": reference["raw"],
                        "char_start": reference["start"],
                        "reason": "annex-bare-no-annex",
                    })
            for enumeration in find_enumerations(text):
                kind, rule, act_surface, anaphoric = classify(
                    text, enumeration, own_celex=celex)
                stats[f"enumeration:{rule}"] += 1

                if kind == "external":
                    external.write({
                        "source_article_id": source_id,
                        "source_celex": celex,
                        "source_unit_type": source_kind,
                        "source_article_number": row.get("article_number", ""),
                        "surface_form": enumeration["raw"],
                        "char_start": enumeration["start"],
                        "char_end": enumeration["end"],
                        "target_act_surface": act_surface,
                        # True when the act is named only by back-reference
                        # ("of that Regulation", "thereof"), so the surface
                        # alone cannot identify it. Recorded, never resolved.
                        # Named acts are resolved by ``resolve_external`` when
                        # the cited act is itself in the corpus; this file is
                        # left as extracted, and ``resolved`` stays False here.
                        "target_act_anaphoric": anaphoric,
                        "rule": rule,
                        "resolved": False,
                    })
                    stats["external"] += 1
                    continue

                for target in expand(enumeration, log=dropped, source=source_id):
                    number = target["number"]
                    record = {
                        "source_article_id": source_id,
                        "target_article_id": cellar.article_eli(row["act_eli"], number),
                        "source_celex": celex,
                        "source_unit_type": source_kind,
                        "target_unit_type": "article",
                        "source_article_number": row.get("article_number", ""),
                        "target_article_number": number,
                        "surface_form": enumeration["raw"],
                        "char_start": enumeration["start"],
                        "char_end": enumeration["end"],
                        "target_paragraph_surface": target["sub"],
                        "rule": rule,
                        "expansion": target["rule"],
                        "source_is_amending": bool(row.get("is_amending")),
                    }

                    # Drop anaphoric self-reference (only meaningful when the
                    # citing unit is itself an article).
                    if source_kind == "article" and number == row["article_number"]:
                        dropped.write({**record, "reason": "self-reference"})
                        stats["dropped:self"] += 1
                        continue

                    # Step 5: the free precision filter.
                    if number not in inventory:
                        dropped.write({**record, "reason": "not-in-inventory"})
                        stats["dropped:not-in-inventory"] += 1
                        continue

                    # Belt and braces after the overlapping-head fix: one
                    # citation site may not yield the same target twice.
                    key = (source_id, number, record["char_start"])
                    if key in emitted:
                        stats["dropped:duplicate"] += 1
                        continue
                    emitted.add(key)

                    internal.write(record)
                    stats[f"internal:{source_kind}->article"] += 1

    for sink in (internal, external, dropped):
        sink.close()

    print(f"internal edges      {internal.count:>7,}  -> {profile_paths.INTERNAL_EDGES_JSONL.name}")
    print(f"external references {external.count:>7,}  -> {profile_paths.EXTERNAL_REFS_JSONL.name}")
    print(f"dropped / logged    {dropped.count:>7,}  -> {profile_paths.DROPPED_JSONL.name}")
    print("\nrules fired:")
    for key, value in sorted(stats.items()):
        print(f"  {key:<34} {value:>6,}")
    (profile_paths.STRUCTURE_DIR / "reference_stats.json").write_text(
        json.dumps(dict(stats), indent=2))


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    extract()


if __name__ == "__main__":
    main()
