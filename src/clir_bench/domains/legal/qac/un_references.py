"""
Extract, classify and resolve cross-references from UN block text.

UN documents cite each other in rigidly formulaic ways -- "resolution 918
(1994)", "resolution 51/210", "(S/PRST/1994/16)", "paragraph 11 (j) of
Security Council resolution 1244 (1999)" -- which makes regex extraction
reliable. This module finds those citations in a block's English text,
classifies each one (external document, internal part of the same document,
or a family this corpus cannot model), and resolves it only when the surface
pins exactly one target:

* **External documents** resolve against the corpus symbol index. An explicitly
  named body restricts slash-form resolutions to that body's symbols. Without
  a named body, overlapping numbers across bodies are ambiguous rather than
  resolved by candidate order. Literal document symbols remain authoritative.
  A citation with a section letter
  ("resolution 51/153 B") names ONLY the lettered document; when the corpus
  holds just the plain twin the citation stays unresolved, because 118 of the
  169 lettered symbols coexist with a plain sibling and resolving to it was a
  measured wrong-document channel.
* **Internal parts** ("paragraph 5 above", "annex II to the present
  resolution") anchor against the citing document's own blocks and annex
  inventory; an anchor that is missing or matches more than one block is a
  named failure, never a guess. A bare "paragraph 5" is anchored only inside
  resolutions and decisions -- in a meeting record it usually names the report
  under discussion, exactly the mis-anchor this module exists to avoid.
* **Unmodelled families** (treaty articles, rules of procedure, general
  comments) are detected so the completeness gate can see them; agenda items
  and coarse structural pointers are recorded without blocking.

Every citation ends with either a resolved target or a ``reason`` from
``REASONS``; ``is_blocking`` says whether that reason disqualifies the citing
block from question generation. The corpus-wide pass that persists this lives
in ``domains/legal/un/references_status.py``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

# References must stay visibly smaller than the same-document context
# (30k chars): cap 4 carries 92% of citing blocks whole (the references-per-
# block distribution's p92, the same place EUR-Lex put its cap); one UN block
# is ~1.0-1.4k chars, so 1500 fits it untruncated.
DEFAULT_MAX_REFERENCES = 4
DEFAULT_REFERENCE_CHARS = 1500

KIND_SC = "sc_resolution"
KIND_SLASH = "slash_resolution"
KIND_DECISION = "decision"
KIND_SYMBOL = "symbol"
KIND_ROMAN = "roman_resolution"
KIND_ES = "es_resolution"
KIND_INTERNAL_PARA = "internal_paragraph"
KIND_INTERNAL_ANNEX = "internal_annex"
KIND_TREATY = "treaty_article"
KIND_RULE = "rules_of_procedure"
KIND_GC = "general_comment"
KIND_AGENDA = "agenda_item"
KIND_STRUCT = "structural_part"
KIND_PARA_OTHER = "paragraph_of_other"

SCOPE_EXTERNAL = "external_doc"
SCOPE_INTERNAL = "internal"
SCOPE_UNMODELLED = "unmodelled"

# Why a citation did not resolve cleanly. ``*_recorded`` reasons are
# observations, not failures; everything else disqualifies the citing block.
REASONS = (
    "out_of_corpus", "pre_corpus", "ambiguous_candidates", "self_reference",
    "external_annex_not_found", "external_annex_ambiguous",
    "internal_paragraph_not_found", "internal_paragraph_ambiguous",
    "internal_annex_not_found", "internal_annex_ambiguous",
    "treaty_article_unmodelled", "rules_of_procedure_unmodelled",
    "general_comment_unmodelled",
    "agenda_item_recorded", "structural_part_recorded",
    "paragraph_of_other_recorded", "annex_of_other_recorded",
    "bare_paragraph_nonres_recorded",
)
NON_BLOCKING_REASONS = frozenset({
    "self_reference", "agenda_item_recorded", "structural_part_recorded",
    "paragraph_of_other_recorded", "annex_of_other_recorded",
    "bare_paragraph_nonres_recorded",
})

_NUM_YR = r"\d{1,4}\s*\(\s*\d{4}\s*\)"
SC_RUN_RE = re.compile(
    rf"\bresolutions?\s+((?:{_NUM_YR})(?:(?:\s*,\s*|\s*,?\s+and\s+)(?:{_NUM_YR}))*)",
    re.IGNORECASE)
SC_ITEM_RE = re.compile(r"(\d{1,4})\s*\(\s*(\d{4})\s*\)")

_SLASH_ITEM = r"\d{1,4}/\d{1,4}(?:\s+[A-Z](?![\w/]))?"
SLASH_RUN_RE = re.compile(
    rf"\bresolutions?\s+((?:{_SLASH_ITEM})(?:(?:\s*,\s*|\s*,?\s+and\s+)(?:{_SLASH_ITEM}))*)")
SLASH_ITEM_RE = re.compile(r"(\d{1,4})/(\d{1,4})(?:\s+([A-Z])(?![\w/]))?")

DEC_RUN_RE = re.compile(
    r"\bdecisions?\s+((?:\d{1,4}/\d{1,4})(?:(?:\s*,\s*|\s*,?\s+and\s+)(?:\d{1,4}/\d{1,4}))*)")

SYMBOL_RE = re.compile(r"\b([A-Z]{1,6}(?:/[A-Za-z0-9.\-]+){2,7})(\s?\(\d{4}\))?")
SYMBOL_BODIES = frozenset({
    "A", "S", "E", "ST", "CEDAW", "CERD", "CAT", "CRC", "CCPR", "CMW", "CRPD",
    "CED", "HRI", "DP", "TD", "CD", "FCCC", "NPT", "ISBA", "SPLOS", "PCNICC",
    "CTOC", "CAC", "CCW", "APLC", "BWC", "UNEP", "HSP", "PBC", "ICCD", "JIU",
    "UNW", "SAICM", "CLCS",
})

ROMAN_RES_RE = re.compile(r"\bresolutions?\s+\d{1,4}\s*(?:[A-Z]\s+)?\(\s*[IVXLC]+\s*\)")
# Special sessions: "resolution ES-10/2" (GA emergency, always A/RES/ES.N/M)
# and "resolution S-20/1" (GA special OR HRC special -- the organ surface, or
# a session number that exists in exactly one space, must pin it).
ES_RES_RE = re.compile(
    r"\b(?:(General\s+Assembly|Human\s+Rights\s+Council)\s+)?"
    r"resolutions?\s+((ES|S)-(\d{1,2})/(\d{1,3}))")

PARA_OF_RE = re.compile(
    r"\b(?:operative\s+)?paragraphs?\s+(\d+(?:\s*\([a-z]\))?)"
    r"(?:(?:\s*,\s*|\s*,?\s+and\s+)\d+(?:\s*\([a-z]\))?)*"
    r"\s*,?\s+(?:of|in)\s+"
    r"((?:its\s+|General\s+Assembly\s+|Security\s+Council\s+|Council\s+|Assembly\s+)?"
    r"resolutions?\s+[^,;.]{0,40})", re.IGNORECASE)

# -- unmodelled families ---------------------------------------------------- #

# The comma before "of" is standard drafting ("article 5, paragraph 4, of the
# Optional Protocol"); without it the whole family fell through to the
# paragraph classifier, ~20k times corpus-wide.
TREATY_ARTICLE_RE = re.compile(
    r"\b(?:articles?|paragraphs?)\s+\d{1,3}(?:\s*\([a-z0-9]+\))?"
    r"(?:(?:\s*,\s*|\s*,?\s+and\s+)\d{1,3}(?:\s*\([a-z0-9]+\))?)*"
    r"(?:\s*,?\s+paragraphs?\s+\d{1,2}(?:\s*\([a-z0-9]+\))?)?"
    r"\s*,?\s+of\s+(?:the\s+)?[^.;:]{0,70}?"
    r"\b(Charter|Covenant|Convention|Protocol|Statute|Treaty)\b", re.IGNORECASE)
RULES_OF_PROCEDURE_RE = re.compile(
    r"\brules?\s+\d{1,3}(?:\s*\([a-z0-9]+\))?"
    r"(?:(?:\s*,\s*|\s*,?\s+and\s+)\d{1,3}(?:\s*\([a-z0-9]+\))?)*"
    r"\s+of\s+(?:its|the|their)\s+(?:provisional\s+)?rules\s+of\s+procedure",
    re.IGNORECASE)
GENERAL_COMMENT_RE = re.compile(
    r"\bgeneral\s+(?:comment|recommendation)s?\s+No\.?\s*\d{1,3}", re.IGNORECASE)
AGENDA_ITEM_RE = re.compile(
    r"\bagenda\s+items?\s+\d{1,3}(?:\s*\([a-z]\))?", re.IGNORECASE)
STRUCTURAL_PART_RE = re.compile(
    r"\b(?i:section|chapter|part|table|figure)s?\s+"
    r"(?:[IVXLC]{1,6}|\d{1,3}|[A-H])(?![\w/])")

# -- internal parts --------------------------------------------------------- #

PARA_RUN_RE = re.compile(
    r"\b(?:operative\s+)?paragraphs?\s+(\d{1,3})(?:\s*\([a-z]\))?"
    r"((?:(?:\s*,\s*|\s*,?\s+and\s+)\d{1,3}(?:\s*\([a-z]\))?)*)", re.IGNORECASE)
INTERNAL_TAIL_RE = re.compile(
    r"^\s*,?\s*(?:above|below)\b"
    r"|^\s*,?\s*(?:of|in)\s+the\s+present\s+(?:resolution|decision|statement|report"
    r"|note|letter|document|declaration)\b"
    r"|^\s*,?\s*(?:of|in)\s+this\s+(?:resolution|decision)\b", re.IGNORECASE)
OF_TAIL_RE = re.compile(r"^\s*,?\s*(?:of|in)\s+", re.IGNORECASE)
PARA_OF_ANNEX_TAIL_RE = re.compile(
    r"^\s*,?\s*(?:of|in)\s+(?:the\s+annex(?:es)?|annex\s+([IVXLC]{1,6}|\d{1,2}))(?![\w-])",
    re.IGNORECASE)

_ANNEX_LABEL = r"(?:[IVXLC]{1,6}|\d{1,2})"
ANNEX_REF_RE = re.compile(
    rf"\b(?:(the|its)\s+)?(annex(?:es)?|appendix|appendices)"
    rf"(?:\s+({_ANNEX_LABEL}))?(?![\w-])", re.IGNORECASE)
ANNEX_INTERNAL_TAIL_RE = re.compile(
    r"^\s+to\s+(?:the\s+present|this)\s+\w+", re.IGNORECASE)
ANNEX_TO_TAIL_RE = re.compile(r"^\s+to\s+", re.IGNORECASE)
ANNEXED_RE = re.compile(
    r"\bannexed\s+(?:hereto|to\s+(?:the\s+present|this)\s+\w+)", re.IGNORECASE)

_ORGANS = ("General Assembly", "Security Council", "Human Rights Council",
           "Economic and Social Council", "Commission on Human Rights",
           "Sub-Commission")
_ORGAN_NAMES = "|".join(r"\s+".join(map(re.escape, organ.split())) for organ in _ORGANS)
_GOVERNING_ORGAN_RE = re.compile(
    rf"\b({_ORGAN_NAMES})(?:['’]s)?\s*(?P<bridge>,?\s*(?:by|in|under)\s+its\s+)?$",
    re.IGNORECASE)
_COORDINATED_ORGAN_RE = re.compile(
    rf"\b(?:{_ORGAN_NAMES})\s+(?:and|or)\s+(?:the\s+)?$", re.IGNORECASE)
_CLAUSE_SUBJECT_RE = re.compile(r"(?:^|[.;!?]\s+)(?:the\s+)?$", re.IGNORECASE)
_RESOLUTION_PREFIXES = {
    "General Assembly": "A/RES/",
    "Human Rights Council": "A/HRC/RES/",
    "Economic and Social Council": "E/RES/",
}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_or_int(label: str) -> int | None:
    if label.isdigit():
        return int(label)
    total, previous = 0, 0
    for char in reversed(label.upper()):
        digit = _ROMAN_VALUES.get(char)
        if digit is None:
            return None
        total = total - digit if digit < previous else total + digit
        previous = max(previous, digit)
    return total or None


@dataclass
class Citation:
    """One cross-reference found in a block's English text."""

    start: int
    end: int
    raw: str
    kind: str
    symbol: str | None = None           # constructed + normalized, when known
    doc_id: str | None = None           # in-corpus document, filled by resolve
    paragraph: int | None = None        # cited operative paragraph, if anchored
    section_letter: str | None = None   # "B" in "resolution 57/270 B"
    organ_hint: str | None = None       # from the 60-char pre-window; metadata
    candidates: tuple[str, ...] = field(default=())
    scope: str = SCOPE_EXTERNAL         # external_doc | internal | unmodelled
    reason: str | None = None           # from REASONS; None once cleanly resolved
    numbers: tuple[int, ...] = ()       # internal paragraph run
    annex_label: str | None = None      # "" = bare "the annex"; "I"/"2" = labelled
    annex_plural: bool = False
    part_kind: str | None = None        # "annex" | "appendix"
    part_id: str | None = None          # resolved annex part
    anchor_blocks: tuple[int, ...] = () # resolved internal target block indices


def is_blocking(citation: Citation) -> bool:
    """Does this citation disqualify its block from question generation?"""
    return citation.reason is not None and citation.reason not in NON_BLOCKING_REASONS


def normalise_symbol(text: str) -> str:
    """Canonical symbol form: uppercase, no whitespace, no trailing punctuation.

    "S/RES/1234 (2005)" and "S/RES/1234(2005)" collapse to one key. ADD./CORR./
    REV. suffixes are kept -- they name their own documents in the index.
    """
    s = text.upper().strip()
    s = re.sub(r"[.,;:\-]+$", "", s)
    return "".join(s.split())


# A handful of stored symbols carry a stray dot from double-underscore doc ids:
# "S/RES/1173(.1998)". The document exists; only its key was unreachable.
_DIRTY_PAREN_RE = re.compile(r"\(\.(?=\d)")


@dataclass(frozen=True)
class SymbolIndex:
    """The corpus symbol inventory, plus what safe resolution needs to know.

    Built in exactly one place so every consumer -- resolver, payload builder,
    status stage -- sees the same keys, the same hygiene, and the same
    special-session facts. Colliding keys are dropped as ambiguous rather than
    silently letting the last document win.
    """

    map: dict[str, str]
    ga_special_sessions: frozenset[int] = frozenset()
    hrc_special_sessions: frozenset[int] = frozenset()
    n_collisions: int = 0
    n_dirty_keys: int = 0

    def __contains__(self, key: object) -> bool:
        return key in self.map

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]]) -> SymbolIndex:
        mapping: dict[str, str] = {}
        dropped: set[str] = set()
        dirty = 0
        for doc_id, symbol in pairs:
            key = normalise_symbol(symbol)
            cleaned = _DIRTY_PAREN_RE.sub("(", key)
            if cleaned != key:
                dirty += 1
            key = cleaned
            if key in dropped:
                continue
            if key in mapping and mapping[key] != doc_id:
                dropped.add(key)
                del mapping[key]
                continue
            mapping[key] = doc_id
        ga = frozenset(int(m.group(1)) for k in mapping
                       if (m := re.match(r"A/RES/S\.(\d+)/", k)))
        hrc = frozenset(int(m.group(1)) for k in mapping
                        if (m := re.match(r"A/HRC/RES/S\.(\d+)/", k)))
        return cls(map=mapping, ga_special_sessions=ga, hrc_special_sessions=hrc,
                   n_collisions=len(dropped), n_dirty_keys=dirty)

    @classmethod
    def from_docs(cls, docs: Mapping[str, Mapping]) -> SymbolIndex:
        return cls.from_pairs((doc_id, row["symbol"]) for doc_id, row in docs.items())

    @classmethod
    def from_map(cls, mapping: Mapping[str, str]) -> SymbolIndex:
        return cls.from_pairs((doc_id, symbol) for symbol, doc_id in mapping.items())


def build_symbol_map(pairs: Iterable[tuple[str, str]]) -> SymbolIndex:
    """(doc_id, symbol) pairs -> SymbolIndex. See SymbolIndex.from_pairs."""
    return SymbolIndex.from_pairs(pairs)


def is_resolution_doc(doc_id: str | None) -> bool:
    """Resolutions and decisions carry their own operative numbering; a bare
    "paragraph 5" inside one names that numbering. Anywhere else it usually
    names the document under discussion."""
    key = (doc_id or "").lower()
    return "/res/" in key or "/dec/" in key


def _organ_hint(text: str, start: int) -> str | None:
    window = text[max(0, start - 60):start]
    found = None
    for organ in _ORGANS:
        pos = window.rfind(organ)
        if pos >= 0 and (found is None or pos > found[0]):
            found = (pos, organ)
    return found[1] if found else None


def _governing_organ(text: str, start: int) -> str | None:
    """Body directly governing a citation, rather than merely mentioned nearby."""
    match = _GOVERNING_ORGAN_RE.search(text[:start])
    if match is None:
        return None
    prefix = text[:match.start()]
    if _COORDINATED_ORGAN_RE.search(prefix):
        # "General Assembly and Security Council resolutions ..." cannot
        # bind every number in the run to the final body alone.
        return None
    if match.group("bridge") and not _CLAUSE_SUBJECT_RE.search(prefix):
        # In "the Assembly established the Human Rights Council by its
        # resolution", "its" refers to the Assembly, not the nearest body.
        # Accept this indirect form only with a body at the clause's start.
        return None
    name = " ".join(match.group(1).split()).casefold()
    return next(organ for organ in _ORGANS if organ.casefold() == name)


def _own_heading_line(text: str, start: int, end: int) -> bool:
    """True when [start, end) is an entire line of the block -- the block's own
    "Annex II" heading, not a citation of it."""
    line_start = text.rfind("\n", 0, start) + 1
    if line_start != start:
        return False
    line_end = text.find("\n", end)
    tail = text[end:] if line_end < 0 else text[end:line_end]
    return not tail.strip()


def extract_citations(text: str, *, doc_id: str | None = None,
                      symbol: str | None = None) -> list[Citation]:
    """Pure regex pass over one block's text. ``doc_id`` (the citing document)
    only decides whether a bare "paragraph N" may be treated as internal."""
    citations: list[Citation] = []
    symbol_spans: list[tuple[int, int]] = []

    # Literal symbols first: an explicit "A/RES/51/210" must not double as a
    # slash-form item of a surrounding "resolution ..." phrase.
    for m in SYMBOL_RE.finditer(text):
        raw = m.group(1)
        if raw.split("/")[0] not in SYMBOL_BODIES:
            continue
        symbol = raw
        if m.group(2) and "/RES/" in raw.upper():
            symbol = raw + m.group(2)           # prose form "S/RES/918 (1994)"
        key = normalise_symbol(symbol)
        citations.append(Citation(
            start=m.start(), end=m.end(), raw=m.group(0).strip(),
            kind=KIND_SYMBOL, symbol=key,
            organ_hint=_organ_hint(text, m.start()),
            # The (SUPP) twin is the same document published as an Official
            # Records supplement; no other suffix extension is safe.
            candidates=(key, key + "(SUPP)")))
        symbol_spans.append((m.start(), m.end()))

    def covered(a: int, b: int) -> bool:
        return any(s <= a and b <= e for s, e in symbol_spans)

    for m in SC_RUN_RE.finditer(text):
        for item in SC_ITEM_RE.finditer(m.group(1)):
            a = m.start(1) + item.start()
            b = m.start(1) + item.end()
            if covered(a, b):
                continue
            symbol = normalise_symbol(f"S/RES/{item.group(1)}({item.group(2)})")
            citations.append(Citation(
                start=a, end=b, raw=item.group(0), kind=KIND_SC, symbol=symbol,
                organ_hint=_organ_hint(text, m.start()), candidates=(symbol,)))

    es_spans: list[tuple[int, int]] = []
    for m in ES_RES_RE.finditer(text):
        if covered(m.start(2), m.end(2)):
            continue
        organ, tag, session, number = (m.group(1), m.group(3),
                                       int(m.group(4)), m.group(5))
        if tag == "ES":
            candidates = (normalise_symbol(f"A/RES/ES.{session}/{number}"),)
        elif organ and organ.startswith("General"):
            candidates = (normalise_symbol(f"A/RES/S.{session}/{number}"),)
        elif organ:
            candidates = (normalise_symbol(f"A/HRC/RES/S.{session}/{number}"),)
        else:
            # Un-pinned "resolution S-20/1": both spaces are candidates and
            # resolve_citations decides from the corpus session inventory.
            candidates = (normalise_symbol(f"A/RES/S.{session}/{number}"),
                          normalise_symbol(f"A/HRC/RES/S.{session}/{number}"))
        citations.append(Citation(
            start=m.start(2), end=m.end(2), raw=m.group(2), kind=KIND_ES,
            organ_hint=_organ_hint(text, m.start()), candidates=candidates))
        es_spans.append((m.start(2), m.end(2)))

    def in_es(a: int, b: int) -> bool:
        return any(s <= a and b <= e for s, e in es_spans)

    for m in SLASH_RUN_RE.finditer(text):
        organ = _governing_organ(text, m.start())
        for item in SLASH_ITEM_RE.finditer(m.group(1)):
            a = m.start(1) + item.start()
            b = m.start(1) + item.end()
            if covered(a, b) or in_es(a, b):
                continue
            x, y, letter = item.group(1), item.group(2), item.group(3)
            if letter:
                # "resolution 51/153 B" names ONLY the lettered document.
                # Falling back to the plain twin was a measured wrong-document
                # channel (118 of 169 lettered symbols have a plain sibling).
                chain = (normalise_symbol(f"A/RES/{x}/{y}{letter}"),)
            else:
                # Expanded corpora can contain the same numbers in several
                # bodies. Resolution must check all matches for ambiguity.
                chain = (normalise_symbol(f"A/RES/{x}/{y}"),
                         normalise_symbol(f"A/HRC/RES/{x}/{y}"),
                         normalise_symbol(f"E/RES/{x}/{y}"))
            if organ is not None:
                prefix = _RESOLUTION_PREFIXES.get(organ)
                # An unsupported named body must not fall back to a different
                # body, including when the cited number has a section letter.
                chain = tuple(candidate for candidate in chain
                              if prefix is not None and candidate.startswith(prefix))
            citations.append(Citation(
                start=a, end=b, raw=item.group(0), kind=KIND_SLASH,
                section_letter=letter,
                organ_hint=organ or _organ_hint(text, m.start()),
                candidates=chain))

    for m in DEC_RUN_RE.finditer(text):
        for item in SLASH_ITEM_RE.finditer(m.group(1)):
            a = m.start(1) + item.start()
            b = m.start(1) + item.end()
            if covered(a, b):
                continue
            # The only */DEC/ space in the corpus is A/HRC/DEC (sessions 13-27,
            # disjoint from GA sessions 48-69 and ECOSOC year forms). GA and
            # ECOSOC decisions have no documents here and stay unresolved.
            citations.append(Citation(
                start=a, end=b, raw=item.group(0), kind=KIND_DECISION,
                organ_hint=_organ_hint(text, m.start()),
                candidates=(normalise_symbol(
                    f"A/HRC/DEC/{item.group(1)}/{item.group(2)}"),)))

    for m in ROMAN_RES_RE.finditer(text):
        if covered(m.start(), m.end()):
            continue
        citations.append(Citation(
            start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_ROMAN,
            organ_hint=_organ_hint(text, m.start())))

    doc_spans = [(c.start, c.end) for c in citations]

    def overlaps_doc(a: int, b: int) -> bool:
        return any(not (b <= s or e <= a) for s, e in doc_spans)

    def doc_citation_starting_in(a: int, b: int) -> Citation | None:
        best = None
        for c in citations:
            if (c.scope == SCOPE_EXTERNAL and a <= c.start < b
                    and (best is None or c.start < best.start)):
                best = c
        return best

    # Unmodelled but detected: the completeness gate must see these.
    unmodelled_spans: list[tuple[int, int]] = []
    for regex, kind, reason in (
            (TREATY_ARTICLE_RE, KIND_TREATY, "treaty_article_unmodelled"),
            (RULES_OF_PROCEDURE_RE, KIND_RULE, "rules_of_procedure_unmodelled"),
            (GENERAL_COMMENT_RE, KIND_GC, "general_comment_unmodelled")):
        for m in regex.finditer(text):
            if overlaps_doc(m.start(), m.end()):
                continue  # "paragraph 11 of resolution 1244 (1999)" is a
                          # document citation, not a treaty article
            citations.append(Citation(
                start=m.start(), end=m.end(), raw=m.group(0).strip(), kind=kind,
                scope=SCOPE_UNMODELLED, reason=reason))
            unmodelled_spans.append((m.start(), m.end()))

    def in_unmodelled(a: int) -> bool:
        return any(s <= a < e for s, e in unmodelled_spans)

    # Paragraph anchors: "paragraph 11 (j) of ... resolution 1244 (1999)"
    # annotates the first citation inside the phrase's resolution part.
    para_of_spans: list[tuple[int, int]] = []
    for m in PARA_OF_RE.finditer(text):
        number = int(re.match(r"\d+", m.group(1)).group(0))
        lo, hi = m.start(2), m.end(2)
        for c in citations:
            if c.scope == SCOPE_EXTERNAL and lo <= c.start < hi and c.paragraph is None:
                c.paragraph = number
                para_of_spans.append((m.start(), m.end()))
                break

    def in_para_of(a: int) -> bool:
        return any(s <= a < e for s, e in para_of_spans)

    # Internal / other paragraph references.
    for m in PARA_RUN_RE.finditer(text):
        if in_para_of(m.start()) or in_unmodelled(m.start()):
            continue
        numbers = tuple(int(n) for n in re.findall(r"\d{1,3}", m.group(0)))
        tail = text[m.end():m.end() + 90]
        annex_tail = PARA_OF_ANNEX_TAIL_RE.match(tail)
        if INTERNAL_TAIL_RE.match(tail):
            citations.append(Citation(
                start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_INTERNAL_PARA,
                scope=SCOPE_INTERNAL, numbers=numbers))
        elif annex_tail:
            # "paragraph 5 of the annex" -- an internal annex citation that
            # also needs the paragraph anchored inside the resolved part.
            citations.append(Citation(
                start=m.start(), end=m.end() + annex_tail.end(),
                raw=m.group(0) + annex_tail.group(0), kind=KIND_INTERNAL_ANNEX,
                scope=SCOPE_INTERNAL, numbers=numbers,
                annex_label=annex_tail.group(1) or "", part_kind="annex"))
        elif OF_TAIL_RE.match(tail):
            target = doc_citation_starting_in(m.end(), m.end() + 60)
            if target is not None:
                if target.paragraph is None:
                    target.paragraph = numbers[0]
                continue
            citations.append(Citation(
                start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_PARA_OTHER,
                scope=SCOPE_UNMODELLED, numbers=numbers,
                reason="paragraph_of_other_recorded"))
        elif is_resolution_doc(symbol or doc_id):
            citations.append(Citation(
                start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_INTERNAL_PARA,
                scope=SCOPE_INTERNAL, numbers=numbers))
        else:
            citations.append(Citation(
                start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_INTERNAL_PARA,
                scope=SCOPE_UNMODELLED, numbers=numbers,
                reason="bare_paragraph_nonres_recorded"))

    # Annex / appendix references. A mention already folded into a
    # "paragraph N of the annex" citation must not fire twice.
    folded_spans = [(c.start, c.end) for c in citations
                    if c.kind == KIND_INTERNAL_ANNEX]
    for m in ANNEX_REF_RE.finditer(text):
        if any(s <= m.start() < e for s, e in folded_spans):
            continue
        determiner, word, label = m.group(1), m.group(2), m.group(3)
        if label is None and determiner is None:
            continue                     # "an annex", the verb, prose
        if _own_heading_line(text, m.start(), m.end()):
            continue                     # the block's own "Annex II" heading
        if overlaps_doc(m.start(), m.end()) or in_unmodelled(m.start()):
            continue
        part_kind = "appendix" if word.lower().startswith("append") else "annex"
        plural = word.lower() in ("annexes", "appendices")
        tail = text[m.end():m.end() + 90]
        if ANNEX_INTERNAL_TAIL_RE.match(tail) or not ANNEX_TO_TAIL_RE.match(tail):
            citations.append(Citation(
                start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_INTERNAL_ANNEX,
                scope=SCOPE_INTERNAL, annex_label=label or "",
                annex_plural=plural, part_kind=part_kind))
        else:
            target = doc_citation_starting_in(m.end(), m.end() + 80)
            if target is not None:
                if target.annex_label is None:
                    target.annex_label = label or ""
                    target.annex_plural = plural
                    target.part_kind = part_kind
            else:
                citations.append(Citation(
                    start=m.start(), end=m.end(), raw=m.group(0),
                    kind=KIND_INTERNAL_ANNEX, scope=SCOPE_UNMODELLED,
                    annex_label=label or "", annex_plural=plural,
                    part_kind=part_kind, reason="annex_of_other_recorded"))

    for m in ANNEXED_RE.finditer(text):
        citations.append(Citation(
            start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_INTERNAL_ANNEX,
            scope=SCOPE_INTERNAL, annex_label="", part_kind="annex"))

    # Recorded, never blocking.
    for m in AGENDA_ITEM_RE.finditer(text):
        citations.append(Citation(
            start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_AGENDA,
            scope=SCOPE_UNMODELLED, reason="agenda_item_recorded"))
    for m in STRUCTURAL_PART_RE.finditer(text):
        if overlaps_doc(m.start(), m.end()) or in_unmodelled(m.start()):
            continue
        if _own_heading_line(text, m.start(), m.end()):
            continue
        citations.append(Citation(
            start=m.start(), end=m.end(), raw=m.group(0), kind=KIND_STRUCT,
            scope=SCOPE_UNMODELLED, reason="structural_part_recorded"))

    citations.sort(key=lambda c: c.start)
    return citations


def resolve_citations(citations: list[Citation],
                      symbols: SymbolIndex | Mapping[str, str], *,
                      citing_doc_id: str | None = None) -> list[Citation]:
    """Fill ``symbol``/``doc_id``/``reason`` for external-document citations.

    Invariant on return: for every ``scope == "external_doc"`` citation,
    ``reason is None`` exactly when ``doc_id`` is set. Self-references keep
    ``doc_id=None`` (their material already travels as document context) with
    the benign reason ``self_reference``.
    """
    index = symbols if isinstance(symbols, SymbolIndex) else SymbolIndex.from_map(symbols)
    hint_prefix = {"Human Rights Council": "A/HRC/RES/",
                   "Economic and Social Council": "E/RES/"}
    for c in citations:
        if c.scope != SCOPE_EXTERNAL:
            continue
        if c.kind == KIND_ROMAN:
            c.reason = "pre_corpus"
            continue
        if c.kind == KIND_SLASH:
            matches = [candidate for candidate in c.candidates if candidate in index.map]
            if len(matches) > 1:
                c.symbol = None
                c.doc_id = None
                c.reason = "ambiguous_candidates"
                continue
        if c.kind == KIND_ES and len(c.candidates) == 2:
            session = int(re.search(r"S\.(\d+)/", c.candidates[0]).group(1))
            in_ga = session in index.ga_special_sessions
            in_hrc = session in index.hrc_special_sessions
            if in_ga and in_hrc:
                # "resolution S-20/1" with no organ surface: both spaces hold
                # session 20. Nothing in the text pins one; never guess.
                c.reason = "ambiguous_candidates"
                c.symbol = c.candidates[0]
                continue
            if in_ga:
                c.candidates = (c.candidates[0],)
            elif in_hrc:
                c.candidates = (c.candidates[1],)
        for candidate in c.candidates:
            if candidate in index.map:
                c.symbol = candidate
                c.doc_id = index.map[candidate]
                c.reason = None
                break
        else:
            if c.symbol is None and c.candidates:
                # Unresolved: record the organ-appropriate symbol when the
                # 60-char window names one, else the first candidate.
                prefix = hint_prefix.get(c.organ_hint or "")
                c.symbol = next(
                    (x for x in c.candidates if prefix and x.startswith(prefix)),
                    c.candidates[0])
            if c.reason is None:
                c.reason = "out_of_corpus"
        if citing_doc_id is not None and c.doc_id == citing_doc_id:
            c.doc_id = None
            c.reason = "self_reference"
    return citations


def resolve_external_annexes(citations: Iterable[Citation],
                             annexes_for: Callable[[str], Sequence[Mapping]]) -> None:
    """Resolve "annex I to resolution X" annotations against the cited
    document's annex inventory. Unique or nothing."""
    for c in citations:
        if c.scope != SCOPE_EXTERNAL or c.annex_label is None:
            continue
        if c.doc_id is None:
            continue        # the document-level reason already covers it
        inventory = [a for a in annexes_for(c.doc_id)
                     if a.get("kind", "annex") == (c.part_kind or "annex")]
        if c.annex_plural:
            if not inventory:
                c.reason = "external_annex_not_found"
            continue        # every annex travels; whole doc is the target
        if c.annex_label:
            wanted = _roman_or_int(c.annex_label)
            matches = [a for a in inventory
                       if a.get("label") and _roman_or_int(str(a["label"])) == wanted]
        else:
            matches = inventory
        if len(matches) == 1:
            c.part_id = matches[0]["part_id"]
        elif not matches:
            c.reason = "external_annex_not_found"
        else:
            c.reason = "external_annex_ambiguous"


def anchored_block_indices(block_texts: Sequence[str], paragraph: int) -> list[int]:
    """Every block containing a line that starts the numbered paragraph."""
    pattern = re.compile(rf"(?m)^\s*{paragraph}\.\s")
    return [i for i, text in enumerate(block_texts) if pattern.search(text)]


def anchored_block_index(block_texts: Sequence[str], paragraph: int) -> int | None:
    """First block (document order) containing a line that starts the cited
    operative paragraph. Measured: 92% unique, 95% found; first-match wins
    because operative numbering precedes annex/section restarts."""
    hits = anchored_block_indices(block_texts, paragraph)
    return hits[0] if hits else None


def resolve_internal(citations: Iterable[Citation], *, block_index: int,
                     block_texts: Sequence[str], block_parts: Sequence[str],
                     annexes: Sequence[Mapping]) -> None:
    """Anchor internal citations against the citing document's own structure.

    A paragraph number anchors only among blocks of the SAME part (operative
    numbering restarts inside annexes); the citing block itself satisfies its
    own numbers. An annex label must match exactly one inventory entry.
    Anything else is a named miss -- never the closest block.
    """
    own_part = block_parts[block_index] if block_index < len(block_parts) else ""
    for c in citations:
        if c.scope != SCOPE_INTERNAL:
            continue
        if c.kind == KIND_INTERNAL_PARA:
            anchors: list[int] = []
            for number in c.numbers:
                hits = [i for i in anchored_block_indices(block_texts, number)
                        if block_parts[i] == own_part]
                if block_index in hits:
                    continue            # the block cites its own paragraph
                if len(hits) == 1:
                    anchors.append(hits[0])
                elif not hits:
                    c.reason = c.reason or "internal_paragraph_not_found"
                else:
                    c.reason = c.reason or "internal_paragraph_ambiguous"
            c.anchor_blocks = tuple(dict.fromkeys(anchors))
            continue
        if c.kind == KIND_INTERNAL_ANNEX:
            inventory = [a for a in annexes
                         if a.get("kind", "annex") == (c.part_kind or "annex")]
            if c.annex_plural:
                if inventory:
                    c.anchor_blocks = tuple(a["block_start"] for a in inventory)
                else:
                    c.reason = "internal_annex_not_found"
                continue
            if c.annex_label:
                wanted = _roman_or_int(c.annex_label)
                matches = [a for a in inventory
                           if a.get("label") and _roman_or_int(str(a["label"])) == wanted]
            else:
                matches = inventory
            if len(matches) == 1:
                part = matches[0]
                c.part_id = part["part_id"]
                c.anchor_blocks = (part["block_start"],)
                if c.numbers:
                    # "paragraph 5 of the annex": the number must anchor
                    # inside the resolved part's block range.
                    lo, hi = part["block_start"], part.get("block_end", part["block_start"])
                    for number in c.numbers:
                        hits = [i for i in anchored_block_indices(block_texts, number)
                                if lo <= i <= hi]
                        if len(hits) == 1:
                            c.anchor_blocks = tuple(dict.fromkeys(
                                list(c.anchor_blocks) + hits))
                        elif not hits:
                            c.reason = c.reason or "internal_paragraph_not_found"
                        else:
                            c.reason = c.reason or "internal_paragraph_ambiguous"
            elif not matches:
                c.reason = "internal_annex_not_found"
            else:
                c.reason = "internal_annex_ambiguous"


def citation_status(citations: Sequence[Citation]) -> dict:
    """The per-block verdict the completeness gate consumes.

    A citation whose document resolved but whose cited annex did not counts as
    unresolved, not resolved: the material the text points at is still
    missing.
    """
    from collections import Counter
    reasons: Counter = Counter(c.reason for c in citations if c.reason)
    external = [c for c in citations if c.scope == SCOPE_EXTERNAL]
    internal = [c for c in citations if c.scope == SCOPE_INTERNAL]
    unmodelled = [c for c in citations if c.scope == SCOPE_UNMODELLED]
    return {
        "n_citations": len(citations),
        "n_external_resolved": sum(1 for c in external
                                   if c.doc_id and not is_blocking(c)),
        "n_external_unresolved": sum(1 for c in external if is_blocking(c)),
        "n_internal_resolved": sum(1 for c in internal if c.reason is None),
        "n_internal_unresolved": sum(1 for c in internal if is_blocking(c)),
        "n_unmodelled_blocking": sum(1 for c in unmodelled if is_blocking(c)),
        "n_recorded": sum(1 for c in citations
                          if c.reason in NON_BLOCKING_REASONS),
        "reasons": dict(reasons),
        "complete": not any(is_blocking(c) for c in citations),
    }


def referenced_docs(citations: Iterable[Citation],
                    max_references: int = DEFAULT_MAX_REFERENCES,
                    ) -> tuple[list[Citation], list[str]]:
    """(kept citations, dropped symbols): one per resolved document,
    first-mention order -- the order of mention tracks how central a cited
    instrument is, same rationale as eurlex_context."""
    kept: list[Citation] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for c in citations:
        if c.doc_id is None or c.doc_id in seen:
            continue
        seen.add(c.doc_id)
        if len(kept) < max_references:
            kept.append(c)
        else:
            dropped.append(c.symbol or c.doc_id)
    return kept, dropped


def main() -> None:  # pragma: no cover
    raise SystemExit(
        "the review harness moved: run "
        "`python -m clir_bench.domains.legal.un.references_status --sample 400`")


if __name__ == "__main__":
    main()
