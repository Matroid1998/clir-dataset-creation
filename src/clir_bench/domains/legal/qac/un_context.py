"""
Assemble the model input for UN question generation.

A question is generated *about one block* -- 150-220 tokens of one UN document
-- but a block cut from the middle of a report is often uninterpretable alone:
"the Mission", "the present report", "the reporting period" all point outside
it. So the payload carries the rest of the document as context.

The contract differs from ``eurlex_context`` in one deliberate way: there the
referenced articles may *complete* an answer; here the DOCUMENT CONTEXT is
disambiguation-only, and the prompts (``prompts_un``) instruct the model that
every answer must come from the target block itself. The faithfulness verifier
enforces it. What travels in the context is therefore a coverage decision, not
a grounding one: send the whole document when it fits the budget, otherwise the
document opening (which resolves document-level references) plus the passages
nearest the target (which resolve local ones).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from clir_bench.domains.legal.qac import un_references as refs
from clir_bench.domains.legal.qac.evidence import EvidenceSource
from clir_bench.domains.legal.un import paths

# Whole-document context fits this budget for roughly three quarters of the
# corpus (median document is ~8k chars, p75 ~29k). Beyond it, context becomes
# the opening plus a window around the target.
DEFAULT_CONTEXT_CHARS = 30_000

TARGET_HEADER = "### TARGET BLOCK — write the questions about THIS text"
REFERENCES_HEADER = ("### REFERENCED DOCUMENTS — other documents CITED by the target "
                     "block, supplied so you can UNDERSTAND those citations. Context "
                     "only: never a source of answers, never the subject of a question.")
REFERENCES_NONE = ("### REFERENCED DOCUMENTS — none. "
                   "The target block cites no other document available in the corpus.")
CONTEXT_HEADER = ("### DOCUMENT CONTEXT — surrounding text of the SAME document, "
                  "supporting context only. It resolves what the target block "
                  "leaves implicit; it is never a source of answers.")
CONTEXT_NONE = ("### DOCUMENT CONTEXT — none. "
                "The target block is the whole document.")
TARGET_PLACEHOLDER = "[... the TARGET BLOCK appears here ...]"
GAP = "[... passages omitted ...]"


def payload_languages(question_language: str) -> tuple[str, ...]:
    """Which language versions to send, given the question's language.

    Same policy as EUR-Lex: English only when asking in English, otherwise the
    question language first plus English as the pivot reading. Only English
    blocks are extracted today, so non-English versions join once their line
    ranges are read from the other 6-way files.
    """
    if question_language == "en":
        return ("en",)
    return (question_language, "en")


@dataclass
class BlockUnit:
    """One block, with the document metadata the prompts expect.

    Junk lines (mastheads, TOC rows, vote rosters) live in the gaps between
    blocks, so they are silently absent from any rendered context -- a block's
    text is always exactly the corpus lines of its range.
    """

    block_id: str
    doc_id: str
    symbol: str
    title: str
    block_index: int
    n_blocks: int
    line_start: int
    line_end: int
    token_count: int
    in_range: bool
    usable: bool = True
    heading: str = ""
    part: str = "body"          # body | annex | appendix
    part_id: str = ""
    part_label: str = ""
    texts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict) -> BlockUnit:
        return cls(
            block_id=row["block_id"], doc_id=row["doc_id"],
            symbol=row.get("symbol", row["doc_id"]), title=row.get("title", ""),
            block_index=row["block_index"], n_blocks=row["n_blocks"],
            line_start=row["line_start"], line_end=row["line_end"],
            token_count=row["token_count"], in_range=row.get("in_range", True),
            usable=row.get("usable", True), heading=row.get("heading", ""),
            part=row.get("part", "body"), part_id=row.get("part_id", ""),
            part_label=row.get("part_label", ""),
            texts={"en": row["text"]},
        )


@dataclass
class ReferencedDoc:
    """A document cited by the target block, rendered for the payload."""

    symbol: str
    doc_id: str
    title: str
    texts: dict[str, str]       # per language: anchored block, cited part, or whole doc
    paragraph: int | None       # set only when the anchor was actually located
    part_label: str = ""        # "Annex I" when the citation named that annex

    @property
    def text(self) -> str:
        """The English rendering -- always present, and the pivot reading."""
        return self.texts.get("en", "")


@dataclass
class GenerationPayload:
    """What gets sent, plus the bookkeeping to interpret what comes back."""

    target: BlockUnit
    context_blocks: list[BlockUnit]     # in document order, target excluded
    n_context_dropped: int              # blocks that did not fit the budget
    references: list[ReferencedDoc]     # cited documents, first-mention order
    dropped_references: list[str]       # resolved but beyond the cap
    text: str
    evidence_sources: dict[tuple[str, str], EvidenceSource] = field(default_factory=dict)


class BlockIndex:
    """Documents in memory, blocks on disk behind per-document seeks.

    ``docs_en.jsonl`` (86k rows) is small enough to hold; ``blocks_en.jsonl``
    (the whole English corpus plus overhead) is not, so each document's block
    rows are read on demand at the byte offset the builder recorded. A fresh
    file handle per read keeps this safe under the batch driver's threads.
    """

    def __init__(self, blocks_path: Path | None = None,
                 docs_path: Path | None = None, *,
                 status_path: Path | None = None,
                 sixway_dir: Path | None = None) -> None:
        self.blocks_path = Path(blocks_path or paths.BLOCKS_JSONL)
        self.sixway_dir = Path(sixway_dir).expanduser() if sixway_dir is not None else None
        self.docs: dict[str, dict] = {}
        with open(docs_path or paths.DOCS_JSONL, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                self.docs[row["doc_id"]] = row
        # One builder for every consumer: hygiene, collision handling and the
        # special-session facts all live in SymbolIndex.
        self.symbols = refs.SymbolIndex.from_docs(self.docs)
        # The corpus-level status file is only assumed for the corpus-level
        # blocks/docs: an index over custom files must not be gated by
        # verdicts computed against different data.
        if status_path is None and blocks_path is None and docs_path is None:
            status_path = paths.REFERENCE_STATUS_JSONL
        self._status_path = Path(status_path) if status_path else None
        self._incomplete: dict[str, dict] | None = None
        self._status_loaded = False
        # language -> doc_id -> the document's corpus lines in that language,
        # filled by preload_translations. Read-only after preloading, so safe
        # under the batch driver's threads.
        self._translations: dict[str, dict[str, list[str]]] = {}

    @property
    def symbol_map(self) -> dict[str, str]:
        return self.symbols.map

    @property
    def incomplete(self) -> dict[str, dict] | None:
        """block_id -> status row, for blocks whose citations are NOT all
        resolved. ``None`` means the status stage has not been run; a block
        absent from the mapping is reference-complete (the stage writes rows
        only for citing blocks, and completeness is the default)."""
        if not self._status_loaded:
            if self._status_path is not None and self._status_path.exists():
                rows: dict[str, dict] = {}
                with open(self._status_path, encoding="utf-8") as fh:
                    for line in fh:
                        row = json.loads(line)
                        if not row.get("complete"):
                            rows[row["block_id"]] = row
                # Assign the finished mapping before the flag, so a concurrent
                # reader can never observe "loaded" with a half-built result.
                self._incomplete = rows
            self._status_loaded = True
        return self._incomplete

    def blocks_for(self, doc_id: str) -> list[BlockUnit]:
        doc = self.docs[doc_id]
        units: list[BlockUnit] = []
        with open(self.blocks_path, "rb") as fh:
            fh.seek(doc["offset"])
            for _ in range(doc["n_blocks"]):
                units.append(BlockUnit.from_row(json.loads(fh.readline())))
        return units

    def preload_translations(self, languages: Sequence[str],
                             doc_ids: Sequence[str] | set[str]) -> None:
        """Load the given documents' text in the given languages.

        The 6-way corpus is line-aligned, so a document's language version is
        exactly the document's line range read from that language's file. One
        sequential pass per language collects every requested range; ``build``
        then attaches each block's slice alongside the English text, giving
        non-English question runs the same "question language first, English
        pivot" payload the EUR-Lex flow sends. A language with no corpus file
        (``de`` is not a UN language) is skipped, and the payload stays
        English-only for it.
        """
        text_files = {
            language: (self.sixway_dir / f"UNv1.0.6way.{language}"
                       if self.sixway_dir is not None else paths.text_file(language))
            for language in dict.fromkeys(languages) if language != "en"
        }
        wanted = [language for language, path in text_files.items() if path.exists()]
        ranges = sorted((self.docs[d]["line_start"], self.docs[d]["line_end"], d)
                        for d in set(doc_ids) if d in self.docs)
        for language in wanted:
            store = self._translations.setdefault(language, {})
            todo = [r for r in ranges if r[2] not in store]
            if not todo:
                continue
            pointer = 0
            with open(text_files[language], encoding="utf-8") as fh:
                for number, line in enumerate(fh, start=1):
                    while pointer < len(todo) and number > todo[pointer][1]:
                        pointer += 1
                    if pointer >= len(todo):
                        break
                    start, _, doc_id = todo[pointer]
                    if number >= start:
                        store.setdefault(doc_id, []).append(line.rstrip("\n"))

    def _attach_translation(self, unit: BlockUnit, language: str) -> None:
        """Set ``unit.texts[language]`` from preloaded document lines.

        Blocks are contiguous line ranges (asserted at build time in
        ``un.blocks``), so the slice is the block's exact text in the other
        language -- same lines, same ``\\n`` join as the English builder.
        """
        if language in unit.texts:
            return
        lines = self._translations.get(language, {}).get(unit.doc_id)
        if lines is None:
            return
        doc_start = self.docs[unit.doc_id]["line_start"]
        start, end = unit.line_start - doc_start, unit.line_end - doc_start + 1
        if 0 <= start < end <= len(lines):
            text = "\n".join(lines[start:end])
            if text.strip():
                unit.texts[language] = text

    def build(self, doc_id: str, block_index: int, *,
              context_chars: int = DEFAULT_CONTEXT_CHARS,
              max_references: int = refs.DEFAULT_MAX_REFERENCES,
              reference_chars: int | None = refs.DEFAULT_REFERENCE_CHARS,
              languages: tuple[str, ...] = ("en",)) -> GenerationPayload | None:
        blocks = self.blocks_for(doc_id)
        if not 0 <= block_index < len(blocks):
            return None
        target = blocks[block_index]
        if not target.texts.get("en"):
            return None

        citations = refs.resolve_citations(
            refs.extract_citations(target.texts["en"], doc_id=doc_id),
            self.symbols, citing_doc_id=doc_id)
        refs.resolve_external_annexes(
            citations, lambda d: self.docs.get(d, {}).get("annexes", []))
        refs.resolve_internal(
            citations, block_index=block_index,
            block_texts=[b.texts["en"] for b in blocks],
            block_parts=[b.part_id for b in blocks],
            annexes=self.docs[doc_id].get("annexes", []))
        kept, dropped = refs.referenced_docs(citations, max_references)
        references = [self._render_reference(c, reference_chars, languages)
                      for c in kept]

        # An internally cited sibling block is guaranteed a context slot: it is
        # what the citation points at, so it may not be squeezed out by
        # generic neighbours.
        priority = sorted({i for c in citations
                           if c.scope == refs.SCOPE_INTERNAL
                           for i in c.anchor_blocks if i != block_index})
        chosen = _select_context(blocks, block_index, context_chars,
                                 priority=priority)
        for language in languages:
            if language != "en":
                self._attach_translation(target, language)
                for unit in chosen:
                    self._attach_translation(unit, language)
        text = render_payload(target, chosen, references=references,
                              languages=languages)
        return GenerationPayload(target, chosen, len(blocks) - 1 - len(chosen),
                                 references, dropped, text,
                                 evidence_sources=visible_evidence_sources(target, languages))

    def _render_reference(self, citation: refs.Citation, limit: int | None,
                          languages: tuple[str, ...] = ("en",)) -> ReferencedDoc:
        """The cited material that travels.

        With a character limit (the windowed pipeline): one excerpt -- the
        paragraph-anchored block when the citation names a paragraph we can
        locate (and is not a lettered section, where the anchor is
        unreliable), else the document's opening block.

        With ``limit=None`` (the whole-fit pipeline): the ENTIRE cited
        document, blocks joined in order -- the fit filter has already
        guaranteed it fits the budget.
        """
        doc = self.docs[citation.doc_id]
        blocks = self.blocks_for(citation.doc_id)
        hit = None
        part_label = ""
        if citation.part_id:
            # The citation named an annex of the cited document and resolution
            # pinned it to exactly one part: that part's opening block is the
            # cited material, not the document's head block.
            part = next((a for a in doc.get("annexes", [])
                         if a["part_id"] == citation.part_id), None)
            if part is not None and part["block_start"] < len(blocks):
                hit = part["block_start"]
                part_label = blocks[hit].part_label or f"Annex {part.get('label', '')}".strip()
        if (hit is None and citation.paragraph is not None
                and citation.section_letter is None and blocks):
            hit = refs.anchored_block_index(
                [b.texts["en"] for b in blocks], citation.paragraph)
        paragraph = citation.paragraph if hit is not None and not part_label else None

        # The cited document travels in every payload language. Blocks are
        # contiguous line ranges, so the other-language rendering is the same
        # ranges read from that language's 6-way file -- the anchor itself is
        # still located on English, where paragraph numbering was validated.
        texts: dict[str, str] = {}
        for language in languages:
            if language != "en":
                for unit in blocks:
                    self._attach_translation(unit, language)
            if limit is None:
                # A partial document would be misleading, so a language is
                # rendered only when every block of it is available.
                if blocks and all(b.texts.get(language) for b in blocks):
                    text = "\n\n".join(b.texts[language] for b in blocks)
                else:
                    text = ""
            else:
                body = blocks[hit] if hit is not None else (blocks[0] if blocks else None)
                text = body.texts.get(language, "") if body else ""
                if len(text) > limit:
                    text = text[:limit].rstrip() + " […truncated]"
            if text.strip():
                texts[language] = text
        return ReferencedDoc(symbol=citation.symbol or doc["symbol"],
                             doc_id=citation.doc_id, title=doc["title"],
                             texts=texts, paragraph=paragraph,
                             part_label=part_label)


def visible_evidence_sources(target, languages):
    return {("target", language): EvidenceSource(
        "target", target.block_id, language, target.texts[language], target.texts[language],
        "block", doc_id=target.doc_id, line_start=target.line_start, line_end=target.line_end,
    ) for language in languages if target.texts.get(language)}


def _select_context(blocks: list[BlockUnit], target_index: int,
                    budget: int, *, priority: Sequence[int] = ()) -> list[BlockUnit]:
    """Context blocks within the character budget, in document order.

    Priority: the document opening first (it resolves "the present report" and
    names the mission), then any ``priority`` blocks -- siblings an internal
    citation anchored to, which the payload must not squeeze out -- then
    neighbours expanding outward from the target (they resolve local
    anaphora). For most documents the budget admits everything and the context
    is simply the whole rest of the document.
    """
    priority_set: set[int] = set()
    candidates: list[int] = []
    if target_index != 0:
        candidates.append(0)
    for index in priority:
        if 0 <= index < len(blocks) and index != target_index \
                and index not in candidates:
            candidates.append(index)
            priority_set.add(index)
    step = 1
    while len(candidates) < len(blocks) - 1:
        for index in (target_index - step, target_index + step):
            if 0 <= index < len(blocks) and index != target_index and index not in candidates:
                candidates.append(index)
        step += 1

    chosen: set[int] = set()
    remaining = budget
    for index in candidates:
        cost = len(blocks[index].texts.get("en", ""))
        if cost > remaining:
            # An oversized anchor must not end the walk -- the guaranteed slot
            # for the NEXT anchor, and the neighbours after it, still stand.
            if index in priority_set:
                continue
            break
        chosen.add(index)
        remaining -= cost
    return [blocks[i] for i in sorted(chosen)]


def _metadata(unit: BlockUnit, language: str) -> str:
    # No block numbering in the payload: the model has no use for the block's
    # position beyond the placeholder in the context, and a number invites
    # questions about it. Position stays in the data (block_index), not the prompt.
    lines = [f"[{language.upper()}] Document: {unit.symbol}"]
    if unit.title:
        lines.append(f"  Title: {unit.title}")
    return "\n".join(lines)


def unit_source(unit: BlockUnit, language: str) -> str:
    """One block rendered for a CSV cell: its metadata line, then its text.

    The label follows :func:`_metadata`'s convention so a reviewer reads the
    same heading the generator saw. The language falls back to English, which is
    what makes this work for a zh question whose block has no zh version: the
    row still shows the source the question was built on.
    """
    lang = language if unit.texts.get(language) else "en"
    text = unit.texts.get(lang, "")
    return f"{_metadata(unit, lang)}\n{text}" if text else ""


def render_payload(target: BlockUnit, context_blocks: list[BlockUnit], *,
                   references: list[ReferencedDoc] = (),
                   languages: tuple[str, ...] = ("en",)) -> str:
    """The user message: target block, referenced documents, then the document
    context, clearly separated -- the literal ``###`` markers are the contract
    pinned by the ``prompts_un`` pack.

    The context is rendered in document order with a placeholder where the
    target sits and gap markers where blocks were dropped for budget, so the
    model can see the passage's position without receiving it twice.
    """
    present = [lg for lg in languages if target.texts.get(lg)]
    parts = [TARGET_HEADER]
    for language in present:
        parts.append(f"{_metadata(target, language)}\n{target.texts[language]}")

    if references:
        parts.append(REFERENCES_HEADER)
        for r in references:
            if r.part_label:
                anchor = f" (cited annex {r.part_label})"
            elif r.paragraph:
                anchor = f" (cited paragraph {r.paragraph})"
            else:
                anchor = ""
            for language in (present or ["en"]):
                body = r.texts.get(language)
                if body:
                    parts.append(f"[{language.upper()}] Reference: {r.symbol} — "
                                 f"{r.title}{anchor}\n{body}")
    else:
        parts.append(REFERENCES_NONE)

    if not context_blocks:
        parts.append(CONTEXT_NONE)
        return "\n\n".join(parts)

    parts.append(CONTEXT_HEADER)
    for language in present:
        pieces: list[str] = []
        previous = -1
        placed = False
        for unit in context_blocks:
            if not placed and unit.block_index > target.block_index:
                if target.block_index - previous > 1:
                    pieces.append(GAP)
                pieces.append(TARGET_PLACEHOLDER)
                previous = target.block_index
                placed = True
            if unit.block_index - previous > 1:
                pieces.append(GAP)
            pieces.append(unit.texts.get(language, ""))
            previous = unit.block_index
        if not placed:
            if target.block_index - previous > 1:
                pieces.append(GAP)
            pieces.append(TARGET_PLACEHOLDER)
            if target.block_index < target.n_blocks - 1:
                pieces.append(GAP)
        elif previous < target.n_blocks - 1:
            pieces.append(GAP)
        body = "\n\n".join(pieces)
        pieces_header = f"[{language.upper()}] Document {target.symbol} — surrounding passages"
        parts.append(f"{pieces_header}\n{body}")
    return "\n\n".join(parts)


__all__ = [
    "CONTEXT_HEADER",
    "DEFAULT_CONTEXT_CHARS",
    "REFERENCES_HEADER",
    "TARGET_HEADER",
    "BlockIndex",
    "BlockUnit",
    "GenerationPayload",
    "ReferencedDoc",
    "payload_languages",
    "render_payload",
    "unit_source",
]
