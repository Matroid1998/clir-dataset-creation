"""
Assemble the model input for EUR-Lex question generation.

A question is generated *about one article*, but EU legislation cross-references
constantly, so an article read alone is often not answerable: "processing is
lawful only if ... Article 9(2) applies" means nothing without Article 9. The
extracted reference graph is what lets those articles travel together.

Two things this module is careful about, because both change what the model can
do:

**The target and the context are separated in the payload.** They are not
concatenated into one blob. The prompt tells the model to write the question
about the TARGET block; the REFERENCED block exists so the answer can be
complete. Without the separation the model drifts and asks about whichever
article happened to be most interesting.

**References are capped.** One article in the corpus cites 345 others -- the
delegated-powers articles enumerate half the act. Sending all of them would
bury the target, blow the context window, and make "which articles are involved"
meaningless. References are taken in order of first mention (which tracks how
central they are to the article's argument) and capped; whatever is dropped is
recorded, never silently discarded.

**Annexes travel too.** "the amounts fixed in Annex IV" is unanswerable
without Annex IV, so annexes the target cites -- of its own act or of another
act in the corpus -- are supplied under their own header, clipped hard because
annex tables run long. The model cites an annex by its ``Cite as:`` key,
``CELEX:anx_<id>``, never by a bare number.

**Articles of other acts travel too, in their own block.** "the conditions of
Article 15 of Regulation (EC) No 2792/1999" is as unanswerable alone as a
same-act citation is. When ``structure/resolve_external`` could pin the cited
act down to one in the corpus, its article is supplied under a third header
and identified to the model by a ``Cite as:`` key of the form ``CELEX:number``
(``32004R0021:5``), so that a bare number in ``articles_involved`` always means
the target's own act and nothing can collide. Same-act and cross-act references
share one cap, ranked by first mention.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from clir_bench.domains.legal.qac.evidence import EvidenceSource
from clir_bench.domains.legal.structure import ACT_LANGUAGES, paths

# Beyond this the target stops being the subject of the payload. Measured over
# the 16,701 citing articles in the corpus: median 1 reference, p90 = 5,
# p95 = 7, max 256. A cap of 6 carries 95.0% of citing articles whole; raising
# it to 10 buys 3 points more and lets a single delegated-powers article drag in
# an entire act.
DEFAULT_MAX_REFERENCES = 6

# Per-reference character budget. The target is never truncated.
DEFAULT_REFERENCE_CHARS = 4000

TARGET_HEADER = "### TARGET ARTICLE — write the questions about THIS article"
CONTEXT_HEADER = ("### REFERENCED ARTICLES — supporting context only, cited by the "
                  "target article. Do not write questions *about* these.")
CONTEXT_NONE = ("### REFERENCED ARTICLES — none. "
                "The target article cites no other article of this act.")
EXTERNAL_HEADER = ("### REFERENCED ARTICLES FROM OTHER ACTS — supporting context only, "
                   "cited by the target article. Do not write questions *about* these.")
EXTERNAL_NONE = ("### REFERENCED ARTICLES FROM OTHER ACTS — none. "
                 "The target article cites no article of another act in the corpus.")
ANNEX_HEADER = ("### REFERENCED ANNEXES — annexes cited by the target article, of this "
                "act or of another act (see each block's Act line). Supporting context "
                "only. Do not write questions *about* these.")
ANNEX_NONE = ("### REFERENCED ANNEXES — none. "
              "The target article cites no resolvable annex.")

# Annex bodies can run to megabytes of tables; they are clipped at load so the
# index stays in memory, and clipped again per payload like any reference.
ANNEX_LOAD_CHARS = 20_000


def external_key(unit: ArticleUnit) -> str:
    """How the model names anything that is not an article of the target's act.

    ``CELEX:number`` for another act's article (``32004R0021:5``);
    ``CELEX:anx_<id>`` for an annex of any act (``32009R1122:anx_1``). Bare
    numbers are reserved for the target's own act's articles, so these tokens
    cannot be mistaken for one. The CELEX id is short, is already printed in
    the block's ``Cite as:`` line, and maps back to the ELI through the payload
    -- no parsing of act titles required.
    """
    if unit.unit_type == "annex":
        subdivision = unit.eli_id.rstrip("/").rsplit("/", 2)[-2]
        return f"{unit.celex_id}:{subdivision}"
    return f"{unit.celex_id}:{unit.article_number}"


def payload_languages(question_language: str) -> tuple[str, ...]:
    """Which language versions to send, given the language the question is in.

    English only when asking in English; otherwise the question language plus
    English. Sending all four versions was wasteful and, worse, misleading: the
    generator is writing in one language, and three additional versions mostly
    added tokens and invited it to blend terminology across languages.

    English is kept as the second version because it is the pivot the whole
    pipeline is built on -- references are extracted from English, so the English
    wording is the one the reference metadata actually describes. The question
    language comes first so its terminology is what the generator sees first.
    """
    if question_language == "en":
        return ("en",)
    return (question_language, "en")


@dataclass
class ArticleUnit:
    """One article, in every language version that exists.

    Carries its own document metadata -- act title and structural position --
    because an article shown to a model without them is a page torn out of a
    book: the generator cannot tell a cosmetics regulation from a customs one,
    and writes questions that are unanswerable once the document is out of view.
    Every field here is per-language except the article number, which is
    language-independent by drafting rule.
    """

    eli_id: str
    celex_id: str
    article_number: str
    headings: dict[str, str] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)
    act_titles: dict[str, str] = field(default_factory=dict)
    locations: dict[str, str] = field(default_factory=dict)
    # "article" or "annex". An annex has no article_number; its heading
    # ("ANNEX I") is its label and its ELI subdivision is its citation key.
    unit_type: str = "article"
    original_hashes: dict[str, str] = field(default_factory=dict)
    original_lengths: dict[str, int] = field(default_factory=dict)
    loaded_body_lengths: dict[str, int] = field(default_factory=dict)

    def languages(self) -> list[str]:
        return [lg for lg in ACT_LANGUAGES if self.texts.get(lg)]


@dataclass
class GenerationPayload:
    """What gets sent, plus the bookkeeping needed to interpret what comes back.

    ``references`` are same-act articles (named by bare number),
    ``external_references`` are articles of other acts (named by
    ``external_key``); both dropped lists hold the tokens that were cut by the
    cap, in the same vocabulary.
    """

    target: ArticleUnit
    references: list[ArticleUnit]
    dropped_references: list[str]
    text: str
    external_references: list[ArticleUnit] = field(default_factory=list)
    dropped_external_references: list[str] = field(default_factory=list)
    annexes: list[ArticleUnit] = field(default_factory=list)
    dropped_annex_references: list[str] = field(default_factory=list)
    evidence_sources: dict[tuple[str, str], EvidenceSource] = field(default_factory=dict)

    @property
    def involved_universe(self) -> list[str]:
        """Tokens the model is allowed to name as involved: bare numbers for the
        target's act's articles, ``CELEX:number`` keys for articles of other
        acts, ``CELEX:anx_<id>`` keys for annexes of any act."""
        return ([self.target.article_number]
                + [r.article_number for r in self.references]
                + [external_key(u) for u in self.external_references]
                + [external_key(u) for u in self.annexes])


class ArticleIndex:
    """Articles and their reference edges -- same-act and cross-act -- keyed for lookup.

    ``references`` holds intra-act article targets, ``external_references`` the
    resolved cross-act ones (both in first-mention order), ``first_mention``
    the character offset at which each target is first cited so the two kinds
    can be ranked together, and ``status`` the per-article completeness verdict
    from ``reference_status.jsonl``. The cross-act and status files are optional
    inputs: an index built before stage 5 ran simply has none.
    """

    def __init__(self, articles_path=None, edges_path=None, *,
                 external_edges_path=None, status_path=None, quarantine_path=None) -> None:
        self.articles_path = Path(articles_path or paths.ARTICLES_JSONL).expanduser().resolve()
        default_quarantine = (self.articles_path.with_name("quarantine.jsonl")
                              if articles_path is not None else paths.QUARANTINE_JSONL)
        self.quarantine_path = Path(quarantine_path or default_quarantine).expanduser().resolve()
        self.by_eli: dict[str, ArticleUnit] = {}
        self.by_act: dict[str, list[str]] = defaultdict(list)
        self.references: dict[str, list[str]] = defaultdict(list)
        self.annex_references: dict[str, list[str]] = defaultdict(list)
        self.external_references: dict[str, list[str]] = defaultdict(list)
        self.first_mention: dict[tuple[str, str], int] = {}
        self.status: dict[str, dict] = {}

        # The corpus-level cross-act and status files are only assumed when the
        # corpus-level edge file is: an index built on a custom edge set must not
        # be gated by verdicts computed against a different one.
        if edges_path is None:
            external_edges_path = external_edges_path or paths.EXTERNAL_EDGES_JSONL
            status_path = status_path or paths.REFERENCE_STATUS_JSONL

        # Edges are read before articles: annex bodies are loaded only for the
        # annexes something actually cites, because a handful of them run to
        # megabytes and the corpus holds nineteen thousand.
        internal_rows = self._article_edges(edges_path or paths.INTERNAL_EDGES_JSONL,
                                            targets=("article", "annex"))
        external_rows = []
        if external_edges_path and Path(external_edges_path).exists():
            external_rows = self._article_edges(external_edges_path,
                                                targets=("article", "annex"))
        wanted_annexes = {e["target_article_id"]
                          for e in internal_rows + external_rows
                          if e.get("target_unit_type") == "annex"}
        self._load_articles(self.articles_path, annex_whitelist=wanted_annexes)

        for edge in internal_rows:
            self._record_edge(edge, self.annex_references
                              if edge.get("target_unit_type") == "annex"
                              else self.references)
        for edge in external_rows:
            self._record_edge(edge, self.annex_references
                              if edge.get("target_unit_type") == "annex"
                              else self.external_references)
        if status_path and Path(status_path).exists():
            self._load_status(status_path)

    def _load_articles(self, path, annex_whitelist: set[str] | None = None) -> None:
        annex_whitelist = annex_whitelist or set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                unit_type = row.get("unit_type")
                if not row.get("eli_id"):
                    continue
                if unit_type == "annex":
                    if row["eli_id"] not in annex_whitelist:
                        continue
                elif unit_type != "article":
                    continue
                unit = self.by_eli.get(row["eli_id"])
                if unit is None:
                    unit = ArticleUnit(row["eli_id"], row["celex_id"],
                                       row.get("article_number", ""),
                                       unit_type=unit_type)
                    self.by_eli[row["eli_id"]] = unit
                    if unit_type == "article":
                        self.by_act[row["celex_id"]].append(row["eli_id"])
                text = row["text"]
                if unit_type == "annex":
                    language = row["language"]
                    unit.original_hashes[language] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    unit.original_lengths[language] = len(text)
                    if len(text) > ANNEX_LOAD_CHARS:
                        body = text[:ANNEX_LOAD_CHARS].rstrip()
                        text = body + " […truncated]"
                        unit.loaded_body_lengths[language] = len(body)
                    else:
                        unit.loaded_body_lengths[language] = len(text)
                unit.texts[row["language"]] = text
                unit.headings[row["language"]] = row.get("heading", "")
                unit.act_titles[row["language"]] = row.get("act_title", "")
                unit.locations[row["language"]] = structural_location(row)

    def _article_edges(self, path, *, targets=("article",)) -> list[dict]:
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                edge = json.loads(line)
                if edge.get("source_unit_type") != "article":
                    continue
                if edge.get("target_unit_type", "article") not in targets:
                    continue
                rows.append(edge)
        # Ordered by first mention: char_start tracks how central a reference is
        # to the article's argument better than any ranking we could invent.
        rows.sort(key=lambda e: (e["source_article_id"], e["char_start"]))
        return rows

    def _record_edge(self, edge: dict, store: dict[str, list[str]]) -> None:
        # A target that is not indexed (its act failed the article load) is
        # skipped in build(), not invented here; recording it is harmless and
        # keeps this loader ignorant of load order.
        source, target = edge["source_article_id"], edge["target_article_id"]
        targets = store[source]
        if target not in targets:
            targets.append(target)
            self.first_mention.setdefault((source, target), edge["char_start"])

    def _load_status(self, path) -> None:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                self.status[row["eli_id"]] = row

    def build(self, target_eli: str, *,
              max_references: int = DEFAULT_MAX_REFERENCES,
              reference_chars: int = DEFAULT_REFERENCE_CHARS,
              languages: Sequence[str] = ACT_LANGUAGES) -> GenerationPayload | None:
        target = self.by_eli.get(target_eli)
        if target is None or target.unit_type != "article" or not target.languages():
            return None

        # One cap over every kind of reference, ranked by first mention.
        # Same-act articles win ties, then annexes, then other acts' articles;
        # with no offsets known each kind keeps its stored order.
        internal = [t for t in self.references.get(target_eli, [])
                    if t != target_eli and t in self.by_eli]
        annexes = [t for t in getattr(self, "annex_references", {}).get(target_eli, [])
                   if t in self.by_eli]
        external = [t for t in getattr(self, "external_references", {}).get(target_eli, [])
                    if t in self.by_eli]
        mention = getattr(self, "first_mention", {})
        order = {"internal": 0, "annex": 1, "external": 2}
        ranked = sorted([(t, "internal") for t in internal]
                        + [(t, "annex") for t in annexes]
                        + [(t, "external") for t in external],
                        key=lambda item: (mention.get((target_eli, item[0]), 10 ** 9),
                                          order[item[1]]))
        kept, dropped = ranked[:max_references], ranked[max_references:]
        references = [self.by_eli[t] for t, kind in kept if kind == "internal"]
        kept_annexes = [self.by_eli[t] for t, kind in kept if kind == "annex"]
        externals = [self.by_eli[t] for t, kind in kept if kind == "external"]

        text = render_payload(target, references, annexes=kept_annexes,
                              external=externals, languages=languages,
                              reference_chars=reference_chars)
        return GenerationPayload(
            target, references,
            [self.by_eli[t].article_number for t, kind in dropped if kind == "internal"],
            text,
            external_references=externals,
            dropped_external_references=[external_key(self.by_eli[t])
                                         for t, kind in dropped if kind == "external"],
            annexes=kept_annexes,
            dropped_annex_references=[external_key(self.by_eli[t])
                                      for t, kind in dropped if kind == "annex"],
            evidence_sources=visible_evidence_sources(
                target, references, external=externals, annexes=kept_annexes,
                languages=languages, reference_chars=reference_chars))


def structural_location(row: dict) -> str:
    """"PART TWO › TITLE I › CHAPTER 2 › SECTION 3" -- whichever levels exist.

    All four grouping levels are optional in EU acts and most acts have none, so
    this is built from what is present rather than from a fixed template.
    """
    parts = []
    for number_key, title_key in (("", "part"), ("", "title_division"),
                                  ("chapter_number", "chapter"),
                                  ("section_number", "section")):
        number = (row.get(number_key) or "").strip() if number_key else ""
        title = (row.get(title_key) or "").strip()
        if number and title:
            parts.append(f"{number} {title}")
        elif number or title:
            parts.append(number or title)
    return " › ".join(parts)


def visible_body(unit: ArticleUnit, language: str, limit: int | None) -> str:
    """The exact body prefix sent to the model, without its truncation marker."""
    body = unit.texts.get(language, "")
    body = body[:unit.loaded_body_lengths.get(language, len(body))]
    return body[:limit].rstrip() if limit and len(body) > limit else body


def visible_evidence_sources(target, references, *, external=(), annexes=(),
                             languages=ACT_LANGUAGES, reference_chars=DEFAULT_REFERENCE_CHARS):
    sources = {}
    units = ([(target, target.article_number, None)]
             + [(unit, unit.article_number, reference_chars) for unit in references]
             + [(unit, external_key(unit), reference_chars) for unit in (*external, *annexes)])
    for unit, reference, limit in units:
        for language in languages:
            if unit.texts.get(language):
                sources[reference, language] = EvidenceSource(
                    reference, unit.eli_id, language, visible_body(unit, language, limit),
                    visible_body(unit, language, None), unit.unit_type,
                    original_sha256=unit.original_hashes.get(language, ""),
                    original_length=unit.original_lengths.get(language))
    return sources


def _block(unit: ArticleUnit, language: str, *, limit: int | None,
           key: str | None = None) -> str:
    heading = unit.headings.get(language) or ""
    body = unit.texts.get(language, "")
    if limit and len(body) > limit:
        body = body[:limit].rstrip() + " […truncated]"
    if unit.unit_type == "annex":
        # An annex's heading IS its label ("ANNEX I"); there is no number.
        label = heading or "Annex"
    else:
        label = f"Article {unit.article_number}"
        if heading:
            label += f" — {heading}"
    lines = [f"[{language.upper()}] {label}"]
    act = unit.act_titles.get(language) or ""
    if act:
        lines.append(f"  Act: {act}")
    if key:
        lines.append(f"  Cite as: {key}")
    location = unit.locations.get(language) or ""
    if location:
        lines.append(f"  Location: {location}")
    lines.append(body)
    return "\n".join(lines)


def _reference_blocks(units: Iterable[ArticleUnit], *, languages: Sequence[str],
                      reference_chars: int | None, keyed: bool) -> list[str]:
    blocks = []
    for unit in units:
        unit_langs = [lg for lg in languages if unit.texts.get(lg)]
        key = external_key(unit) if keyed else None
        blocks.append("\n\n".join(
            _block(unit, lg, limit=reference_chars, key=key) for lg in unit_langs))
    return blocks


def render_payload(target: ArticleUnit, references: Iterable[ArticleUnit], *,
                   external: Iterable[ArticleUnit] = (),
                   annexes: Iterable[ArticleUnit] = (),
                   languages: Sequence[str] = ACT_LANGUAGES,
                   reference_chars: int | None = DEFAULT_REFERENCE_CHARS) -> str:
    """The user message: target, same-act articles, other acts' articles, annexes.

    Four headers, always in this order, each present even when empty so the
    model never has to infer which block is missing. Every language version of
    each unit is included, preserving the pipeline's cross-lingual grounding
    property -- the generator and the graders see the same act in all its
    language versions rather than one translation.
    """
    present = [lg for lg in languages if target.texts.get(lg)]
    parts = [TARGET_HEADER,
             "\n\n".join(_block(target, lg, limit=None) for lg in present)]

    references = list(references)
    if references:
        parts.append(CONTEXT_HEADER)
        parts.extend(_reference_blocks(references, languages=languages,
                                       reference_chars=reference_chars, keyed=False))
    else:
        parts.append(CONTEXT_NONE)

    external = list(external)
    if external:
        parts.append(EXTERNAL_HEADER)
        parts.extend(_reference_blocks(external, languages=languages,
                                       reference_chars=reference_chars, keyed=True))
    else:
        parts.append(EXTERNAL_NONE)

    annexes = list(annexes)
    if annexes:
        parts.append(ANNEX_HEADER)
        parts.extend(_reference_blocks(annexes, languages=languages,
                                       reference_chars=reference_chars, keyed=True))
    else:
        parts.append(ANNEX_NONE)
    return "\n\n".join(parts)


def _canonical_token(item: str, payload: GenerationPayload) -> str:
    """One declared article, in the vocabulary of ``involved_universe``.

    ``"Article 6"`` -> ``"6"``; ``"32004R0021: Article 5"`` -> ``"32004R0021:5"``;
    a full ELI id of a supplied article -> its universe token. Anything else is
    returned stripped so it can be reported as rejected.
    """
    text = str(item).strip()
    by_eli = {payload.target.eli_id: payload.target.article_number}
    by_eli.update({r.eli_id: r.article_number for r in payload.references})
    by_eli.update({u.eli_id: external_key(u) for u in payload.external_references})
    by_eli.update({u.eli_id: external_key(u) for u in payload.annexes})
    if text in by_eli:
        return by_eli[text]
    # "Article 6", "Art. 6", "article 32004R0021:5" -- the word is noise either way.
    bare = re.sub(r"^\s*(?:articles?|art\.?)\s*", "", text, flags=re.IGNORECASE).strip(" .()")
    if ":" in bare and not bare.lower().startswith("http"):
        celex, number = bare.split(":", 1)
        celex = celex.strip().upper()
        number = re.sub(r"^\s*(?:articles?|art\.?)\s*", "", number, flags=re.IGNORECASE).strip(" .()").lower()
        # "anx 1" / "ANNEX_1" -> "anx_1": the key is the ELI subdivision.
        number = re.sub(r"^annex", "anx", re.sub(r"[\s]+", "_", number))
        # A key naming the target's own act's ARTICLE is just a same-act
        # number; its annexes keep the key (bare numbers never mean annexes).
        if celex == payload.target.celex_id.upper() and not number.startswith("anx"):
            return number
        return f"{celex}:{number}"
    return bare.lower()


def normalise_involved(raw, payload: GenerationPayload) -> tuple[list[str], list[str]]:
    """(accepted tokens, rejected tokens) from the model's answer.

    The model names plain article numbers for the target's act and
    ``CELEX:number`` keys for articles of other acts, because that is what it
    can see in the text. They are validated against the articles actually
    supplied -- a token the model never received cannot be an honest citation
    -- and the target is always included, since the question is by construction
    about it.
    """
    universe = set(payload.involved_universe)
    accepted: list[str] = []
    rejected: list[str] = []
    items: list[str] = []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.replace(" and ", ",").split(",")]
    elif isinstance(raw, Iterable):
        items = [str(p).strip() for p in raw]
    for item in items:
        token = _canonical_token(item, payload)
        if not token:
            continue
        if token in universe:
            if token not in accepted:
                accepted.append(token)
        else:
            rejected.append(item)
    if payload.target.article_number not in accepted:
        accepted.insert(0, payload.target.article_number)
    order = {n: i for i, n in enumerate(payload.involved_universe)}
    accepted.sort(key=lambda n: order.get(n, 999))
    return accepted, rejected


def involved_elis(tokens: Sequence[str], payload: GenerationPayload) -> list[str]:
    """Map involved tokens back to ELI ids, for downstream qrels."""
    lookup = {payload.target.article_number: payload.target.eli_id}
    lookup.update({r.article_number: r.eli_id for r in payload.references})
    lookup.update({external_key(u): u.eli_id for u in payload.external_references})
    lookup.update({external_key(u): u.eli_id for u in payload.annexes})
    return [lookup[n] for n in tokens if n in lookup]


SOURCE_SEP = "\n\n----------\n\n"


def _involved_lookup(payload: GenerationPayload) -> dict[str, ArticleUnit]:
    """Declared token -> unit, over everything the model was allowed to cite."""
    lookup: dict[str, ArticleUnit] = {payload.target.article_number: payload.target}
    lookup.update({r.article_number: r for r in payload.references})
    lookup.update({external_key(u): u for u in payload.external_references})
    lookup.update({external_key(u): u for u in payload.annexes})
    return lookup


def involved_units(tokens: Sequence[str], payload: GenerationPayload) -> list[ArticleUnit]:
    """The units behind ``articles_involved``, in the order the model declared.

    The same lookup :func:`involved_elis` performs, returning the unit rather
    than its id so callers can render the text a question was drawn from.
    """
    lookup = _involved_lookup(payload)
    return [lookup[token] for token in tokens if token in lookup]


def unit_source(unit: ArticleUnit, language: str, *, token: str = "") -> str:
    """One unit rendered for a CSV cell: its label, then its text.

    The label follows :func:`_block`'s convention so a reviewer reads the same
    heading the generator saw. ``token`` prefixes the declared citation key so a
    cell can be matched back to ``articles_involved``. The language falls back
    to English, which is what makes this work for a zh question: the corpus has
    no zh act versions, so a zh row still shows the source it was built on.
    """
    lang = language if unit.texts.get(language) else "en"
    heading = unit.headings.get(lang) or ""
    if unit.unit_type == "annex":
        label = heading or "Annex"
    else:
        label = f"Article {unit.article_number}"
        if heading:
            label += f" \u2014 {heading}"
    if token:
        label = f"[{token}] {label}"
    return f"{label}\n{unit.texts.get(lang, '')}".strip()


def referenced_sources(tokens: Sequence[str], payload: GenerationPayload,
                       language: str) -> str:
    """Text of every involved article EXCEPT the target, in declared order.

    The target is excluded because it has its own column and appears in
    ``articles_involved`` on every row; repeating it would roughly double the
    file for the single-article case, which is most of the corpus. An empty cell
    therefore means exactly "the target article alone answers this".
    """
    lookup = _involved_lookup(payload)
    parts = []
    for token in tokens:
        unit = lookup.get(token)
        if unit is None or unit.eli_id == payload.target.eli_id:
            continue
        parts.append(unit_source(unit, language, token=token))
    return SOURCE_SEP.join(parts)


__all__ = [
    "ANNEX_HEADER",
    "CONTEXT_HEADER",
    "DEFAULT_MAX_REFERENCES",
    "DEFAULT_REFERENCE_CHARS",
    "EXTERNAL_HEADER",
    "SOURCE_SEP",
    "TARGET_HEADER",
    "ArticleIndex",
    "ArticleUnit",
    "GenerationPayload",
    "external_key",
    "involved_elis",
    "involved_units",
    "normalise_involved",
    "referenced_sources",
    "render_payload",
    "unit_source",
]
