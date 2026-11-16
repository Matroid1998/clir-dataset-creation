"""Build an auditable six-language UN source from pinned UPRPRC alignments.

Only published bilingual links are used: no translation, model call, sentence
splitting, text normalization, or inference of an absent language is performed.
Paragraph indices refer to ``original.split("\\n\\n")``. Each link must round-trip
exactly, and the output always contains contiguous slices of the original texts.

Production input is local Parquet (one file or a directory of shards)::

    python -m clir_bench.domains.legal.un.uprprc prepare \\
        --filewise /data/filewise --bilingual /data/bilingual --out-dir /data/prepared
    python -m clir_bench.domains.legal.un.uprprc audit /data/prepared/manifest.json

The large input join is staged on disk. Only one source document and its links
are assembled at a time; the original full documents remain separate from blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from clir_bench.domains.legal.un.blocks import (
    ANNEX_HEADING_RE,
    CLS_HEADING,
    JUNK_CLASSES,
    Line,
    assign_parts,
    classify_lines,
    junk_flags,
)

FILEWISE_REPOSITORY = "bot-yaya/UPRPRC_FTXT_FILEWISE"
FILEWISE_REVISION = "67ae6c42e9008777fea18be9a0461e628f4d5ab1"
BILINGUAL_REPOSITORY = "bot-yaya/UPRPRC_BILINGUAL"
BILINGUAL_REVISION = "536e033bf67872ba1f06ea44b6a582fb85772760"
SOURCE_LANGUAGES = ("ar", "zh", "en", "fr", "ru", "es", "de")
REQUIRED_LANGUAGES = SOURCE_LANGUAGES[:-1]
OTHER_LANGUAGES = tuple(language for language in REQUIRED_LANGUAGES if language != "en")
FORMAT = "un-uprprc-v1"
VERSION = 1


class SourceValidationError(ValueError):
    """A source/link cannot safely identify original translated text."""


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize source value of type {type(value).__name__}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_id(source_id: str) -> str:
    return f"uprprc:{FILEWISE_REVISION}:{source_id}"


def _year(value: Any) -> int | None:
    if isinstance(value, (datetime, date)):
        return value.year
    try:
        return date.fromisoformat(str(value or "")[:10]).year
    except ValueError:
        return None


@dataclass(frozen=True)
class Paragraphs:
    text: str
    values: tuple[str, ...]
    starts: tuple[int, ...]

    @classmethod
    def of(cls, text: str) -> Paragraphs:
        values = tuple(text.split("\n\n"))
        starts = []
        offset = 0
        for value in values:
            starts.append(offset)
            offset += len(value) + 2
        return cls(text, values, tuple(starts))

    def span(self, indices: Iterable[int]) -> tuple[int, int]:
        numbers = sorted(indices)
        if not numbers:
            raise SourceValidationError("empty_paragraph_set")
        selected = set(numbers)
        for number in range(numbers[0], numbers[-1] + 1):
            if number not in selected and self.values[number].strip():
                raise SourceValidationError("noncontiguous_paragraph_set")
        start = self.starts[numbers[0]]
        end = self.starts[numbers[-1]] + len(self.values[numbers[-1]])
        if not self.text[start:end].strip():
            raise SourceValidationError("empty_aligned_span")
        return start, end


def parse_indices(value: str) -> tuple[int, ...]:
    """Read original paragraph positions without silently sorting or deduplicating."""
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:,\d+)*", value):
        raise SourceValidationError("invalid_paragraph_indices")
    indices = tuple(int(number) for number in value.split(","))
    if tuple(sorted(set(indices))) != indices:
        raise SourceValidationError("unordered_or_duplicate_paragraph_indices")
    return indices


@dataclass(frozen=True)
class VerifiedLink:
    language: str
    english: tuple[int, ...]
    other: tuple[int, ...]


def verify_link(row: dict, source: dict, paragraphs: dict[str, Paragraphs]) -> VerifiedLink:
    """Verify identity, original paragraph coordinates, and exact bilingual text."""
    src, dst = row.get("src_lang"), row.get("dst_lang")
    if src == "en" and dst in OTHER_LANGUAGES:
        language = dst
    elif dst == "en" and src in OTHER_LANGUAGES:
        language = src
    else:
        raise SourceValidationError("not_required_pivot_pair")
    if str(row.get("id", "")) != source["id"]:
        raise SourceValidationError("document_identity_mismatch")
    pair = row.get("clean_para_index_set_pair")
    if not isinstance(pair, str) or pair.count("|") != 1:
        raise SourceValidationError("invalid_paragraph_pair")
    indices = [parse_indices(side) for side in pair.split("|")]
    jobs = source["job_numbers"]
    for side, lang, numbers in zip(("src", "dst"), (src, dst), indices):
        if row.get(f"{side}_job_number") != jobs[SOURCE_LANGUAGES.index(lang)]:
            raise SourceValidationError("job_number_mismatch")
        original = paragraphs[lang].values
        if numbers[-1] >= len(original):
            raise SourceValidationError("paragraph_index_out_of_bounds")
        if "\n".join(original[number] for number in numbers) != row.get(f"{side}_text"):
            raise SourceValidationError("paragraph_text_mismatch")
    return VerifiedLink(
        language,
        indices[0] if src == "en" else indices[1],
        indices[1] if src == "en" else indices[0],
    )


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def root(self, node: tuple[str, int]) -> tuple[str, int]:
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def join(self, first: tuple[str, int], second: tuple[str, int]) -> None:
        left, right = self.root(first), self.root(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


@dataclass(frozen=True)
class Component:
    indices: dict[str, tuple[int, ...]]
    spans: dict[str, tuple[int, int]]
    links: tuple[VerifiedLink, ...] = ()
    component_count: int = 1


def aligned_components(
    paragraphs: dict[str, Paragraphs],
    links: Iterable[VerifiedLink],
    rejected: Counter | None = None,
) -> list[Component]:
    """Build complete EN-pivot components, rejecting holes and crossing intervals."""
    rejected = rejected if rejected is not None else Counter()
    groups = _DisjointSet()
    coverage: dict[int, set[str]] = defaultdict(set)
    verified = []
    for link in links:
        verified.append(link)
        nodes = [("en", number) for number in link.english]
        nodes.extend((link.language, number) for number in link.other)
        for node in nodes:
            groups.join(nodes[0], node)
        for number in link.english:
            coverage[number].add(link.language)
    members: dict[tuple[str, int], dict[str, set[int]]] = {}
    for node in sorted(groups.parent):
        members.setdefault(groups.root(node), defaultdict(set))[node[0]].add(node[1])
    links_by_group: dict[tuple[str, int], list[VerifiedLink]] = defaultdict(list)
    for link in verified:
        links_by_group[groups.root(("en", link.english[0]))].append(link)
    candidates = []
    for root, by_language in members.items():
        if set(by_language) != set(REQUIRED_LANGUAGES) or any(
            coverage[number] != set(OTHER_LANGUAGES) for number in by_language.get("en", ())
        ):
            rejected["incomplete_multiway_coverage"] += 1
            continue
        indices = {
            language: tuple(sorted(by_language[language])) for language in REQUIRED_LANGUAGES
        }
        try:
            spans = {
                language: paragraphs[language].span(indices[language])
                for language in REQUIRED_LANGUAGES
            }
        except SourceValidationError as error:
            rejected[str(error)] += 1
            continue
        proof = tuple(
            sorted(
                set(links_by_group[root]),
                key=lambda link: (link.language, link.english, link.other),
            )
        )
        candidates.append(Component(indices, spans, proof))
    candidates.sort(key=lambda component: component.spans["en"])
    # A translation must have the same document order in every language. Mark
    # both sides of every inversion, including non-adjacent nested intervals.
    unsafe: set[int] = set()
    for language in REQUIRED_LANGUAGES:
        maximum = -1
        maximum_index = -1
        for index, component in enumerate(candidates):
            start, end = component.spans[language]
            if start < maximum:
                unsafe.update((maximum_index, index))
            if end > maximum:
                maximum, maximum_index = end, index
        minimum = len(paragraphs[language].text) + 1
        minimum_index = -1
        for index in range(len(candidates) - 1, -1, -1):
            start, end = candidates[index].spans[language]
            if end > minimum:
                unsafe.update((minimum_index, index))
            if start < minimum:
                minimum, minimum_index = start, index
    rejected["crossing_or_overlapping_components"] += len(unsafe)
    return [component for index, component in enumerate(candidates) if index not in unsafe]


@dataclass(frozen=True)
class _Profile:
    part: tuple[str, str, str]
    classes: frozenset[str]
    mixed_parts: bool = False


def _profiles(english: Paragraphs) -> list[_Profile]:
    lines = []
    for paragraph, text in enumerate(english.values):
        for line in text.splitlines():
            if line.strip():
                lines.append(Line(len(lines) + 1, paragraph, line, len(line.split())))
    classes = classify_lines(lines)
    parts = assign_parts(lines, classes)
    by_paragraph: dict[int, list[int]] = defaultdict(list)
    for index, line in enumerate(lines):
        by_paragraph[line.paragraph].append(index)
    profiles = []
    current = ("body", "", "")
    for number in range(len(english.values)):
        positions = by_paragraph[number]
        if positions:
            labels = {parts[index] for index in positions}
            current = parts[positions[-1]]
            profiles.append(
                _Profile(
                    parts[positions[0]],
                    frozenset(classes[index] for index in positions),
                    len(labels) > 1,
                )
            )
        else:
            profiles.append(_Profile(current, frozenset()))
    return profiles


def _component_flags(component: Component, profiles: list[_Profile]) -> list[str]:
    first, last = component.indices["en"][0], component.indices["en"][-1]
    selected = profiles[first : last + 1]
    classes = set().union(*(profile.classes for profile in selected))
    flags = []
    if classes & JUNK_CLASSES:
        flags.append("mixed_furniture_content" if classes - JUNK_CLASSES else "furniture")
    if (
        any(profile.mixed_parts for profile in selected)
        or len({profile.part for profile in selected}) != 1
    ):
        flags.append("annex_boundary")
    return flags


def _adjacent(left: Component, right: Component, paragraphs: dict[str, Paragraphs]) -> bool:
    return all(
        left.spans[language][1] <= right.spans[language][0]
        and not paragraphs[language]
        .text[left.spans[language][1] : right.spans[language][0]]
        .strip()
        for language in REQUIRED_LANGUAGES
    )


def _merge(left: Component, right: Component) -> Component:
    return Component(
        {
            language: left.indices[language] + right.indices[language]
            for language in REQUIRED_LANGUAGES
        },
        {
            language: (left.spans[language][0], right.spans[language][1])
            for language in REQUIRED_LANGUAGES
        },
        left.links + right.links,
        left.component_count + right.component_count,
    )


def _pack(
    components: list[Component],
    paragraphs: dict[str, Paragraphs],
    profiles: list[_Profile],
    min_words: int,
    max_words: int,
) -> list[Component]:
    packed: list[Component] = []
    current: Component | None = None
    for component in components:
        if current is not None:
            combined = _merge(current, component)
            first_part = profiles[current.indices["en"][0]].part
            next_part = profiles[component.indices["en"][0]].part
            # Furniture and incomplete gaps are hard boundaries. A heading is
            # a soft boundary once the preceding block has reached minimum.
            next_classes = profiles[component.indices["en"][0]].classes
            current_words = len(paragraphs["en"].text[slice(*current.spans["en"])].split())
            can_join = (
                not _component_flags(current, profiles)
                and not _component_flags(component, profiles)
                and first_part == next_part
                and _adjacent(current, component, paragraphs)
                and len(paragraphs["en"].text[slice(*combined.spans["en"])].split()) <= max_words
                and not (CLS_HEADING in next_classes and current_words >= min_words)
            )
            if can_join:
                current = combined
                continue
            packed.append(current)
        current = component
    if current is not None:
        packed.append(current)
    return packed


def _context_spans(
    start: int, end: int, english: Paragraphs, profiles: list[_Profile]
) -> Iterator[tuple[int, int, int, int]]:
    """Split unmatched English context at annex boundaries, never invent translations."""
    current = None
    first = max(0, bisect_right(english.starts, start) - 1)
    for number in range(first, len(english.starts)):
        paragraph_start = english.starts[number]
        if paragraph_start >= end:
            break
        paragraph_end = paragraph_start + len(english.values[number])
        left, right = max(start, paragraph_start), min(end, paragraph_end)
        if left >= right or not english.text[left:right].strip():
            continue
        part = profiles[number].part
        if current is not None and current[4] == part:
            current = (current[0], right, current[2], number, part)
        else:
            if current is not None:
                yield current[:4]
            current = (left, right, number, number, part)
    if current is not None:
        yield current[:4]


def build_document(
    source: dict,
    texts: dict[str, str],
    links: Iterable[VerifiedLink],
    *,
    min_words: int = 150,
    max_words: int = 450,
    from_year: int = 2015,
    to_year: int = 2025,
    invalid_reason: str = "",
    rejected: Counter | None = None,
) -> tuple[dict, list[dict]]:
    """Assemble one source; full originals are supplied/stored separately from blocks."""
    doc_id = document_id(source["id"])
    paragraphs = {language: Paragraphs.of(texts[language]) for language in REQUIRED_LANGUAGES}
    english = paragraphs["en"]
    profiles = _profiles(english)
    components = [] if invalid_reason else aligned_components(paragraphs, links, rejected)
    packed = _pack(components, paragraphs, profiles, min_words, max_words)
    units: list[tuple[int, int, int, int, Component | None]] = []
    cursor = 0
    for component in packed:
        start, end = component.spans["en"]
        units.extend((*span, None) for span in _context_spans(cursor, start, english, profiles))
        units.append(
            (start, end, component.indices["en"][0], component.indices["en"][-1], component)
        )
        cursor = end
    units.extend(
        (*span, None) for span in _context_spans(cursor, len(english.text), english, profiles)
    )
    hashes = {language: sha256_text(texts[language]) for language in REQUIRED_LANGUAGES}
    symbol = source.get("symbol") or source["id"]
    title = str(source.get("title") or "")
    year = _year(source.get("publication_date"))
    target_eligible = year is not None and from_year <= year <= to_year
    blocks = []
    targets = []
    for index, (start, end, first, last, component) in enumerate(units):
        text = english.text[start:end]
        part, part_id, part_label = profiles[first].part
        flags = (
            _component_flags(component, profiles)
            if component
            else ["unverified_alignment" if not invalid_reason else "invalid_source_links"]
        )
        if not target_eligible:
            flags.append("outside_target_publication_window")
        flags.extend(flag for flag in junk_flags(text) if flag not in flags)
        token_count = len(text.split())
        in_range = min_words <= token_count <= max_words
        usable = component is not None and not flags
        if usable and in_range:
            targets.append(index)
        locations = {}
        block_texts = {}
        spans = component.spans if component else {"en": (start, end)}
        for language, (left, right) in spans.items():
            block_texts[language] = texts[language][left:right]
            locations[language] = {"start": left, "end": right, "document_sha256": hashes[language]}
        heading = []
        for number in range(first, last + 1):
            if CLS_HEADING not in profiles[number].classes:
                break
            heading.append(english.values[number].strip())
        blocks.append(
            {
                "block_id": f"{doc_id}#{index}",
                "doc_id": doc_id,
                "symbol": symbol,
                "body": str(symbol).split("/", 1)[0].lower(),
                "year": str(year or ""),
                "title": title,
                "block_index": index,
                "n_blocks": len(units),
                "line_start": None,
                "line_end": None,
                "n_lines": None,
                "para_start": first,
                "para_end": last,
                "token_count": token_count,
                "char_count": len(text),
                "in_range": in_range,
                "usable": usable,
                "junk_flags": flags,
                "splits_paragraph": False,
                "part": part,
                "part_id": part_id,
                "part_label": part_label,
                "heading": " / ".join(heading)[:200],
                "text": text,
                "texts": block_texts,
                "source_locations": locations,
                "alignment_components": 0 if component is None else component.component_count,
                "alignment_links": []
                if component is None
                else [
                    {
                        "language": link.language,
                        "english": list(link.english),
                        "other": list(link.other),
                        "english_job_number": source["job_numbers"][SOURCE_LANGUAGES.index("en")],
                        "other_job_number": source["job_numbers"][
                            SOURCE_LANGUAGES.index(link.language)
                        ],
                        "english_text_sha256": sha256_text(
                            "\n".join(english.values[i] for i in link.english)
                        ),
                        "other_text_sha256": sha256_text(
                            "\n".join(paragraphs[link.language].values[i] for i in link.other)
                        ),
                    }
                    for link in component.links
                ],
            }
        )
    annexes = []
    for block in blocks:
        if block["part"] == "body":
            continue
        index = block["block_index"]
        if (
            annexes
            and annexes[-1]["part_id"] == block["part_id"]
            and annexes[-1]["block_end"] == index - 1
        ):
            annexes[-1]["block_end"] = index
            continue
        match = ANNEX_HEADING_RE.match(block["part_label"])
        annexes.append(
            {
                "part_id": block["part_id"],
                "kind": block["part"],
                "label": (match.group(2) or "") if match else "",
                "block_start": index,
                "block_end": index,
            }
        )
    metadata = {
        "doc_id": doc_id,
        "upstream_id": source["id"],
        "symbol": symbol,
        "title": title,
        "body": str(symbol).split("/", 1)[0].lower(),
        "year": str(year or ""),
        "target_eligible": target_eligible,
        "publication_date": source.get("publication_date"),
        "n_blocks": len(blocks),
        "n_targets": len(targets),
        "target_idxs": targets,
        "n_in_range": sum(block["in_range"] for block in blocks),
        "token_count": len(english.text.split()),
        "char_count": len(english.text),
        "line_start": None,
        "line_end": None,
        "annexes": annexes,
        "source_revision": FILEWISE_REVISION,
        "upstream": source,
        "invalid_reason": invalid_reason,
    }
    return metadata, blocks


def parquet_files(path: Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    files = (
        [path] if path.is_file() and path.suffix == ".parquet" else sorted(path.rglob("*.parquet"))
    )
    if not files:
        raise ValueError(f"No local Parquet files found at {path}")
    return files


def parquet_records(files: Iterable[Path]) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Reading UPRPRC input requires pyarrow (installed by datasets)."
        ) from error
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def _validate_source(row: dict) -> tuple[dict, dict[str, str]]:
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise SourceValidationError("missing_source_id")
    values, jobs = row.get("ftxt"), row.get("job_numbers")
    if not isinstance(values, (tuple, list)) or len(values) != len(SOURCE_LANGUAGES):
        raise SourceValidationError("invalid_fulltext_language_slots")
    if not isinstance(jobs, (tuple, list)) or len(jobs) != len(SOURCE_LANGUAGES):
        raise SourceValidationError("invalid_job_language_slots")
    texts = dict(zip(SOURCE_LANGUAGES, values))
    if any(
        not isinstance(texts[lang], str) or not texts[lang].strip() for lang in REQUIRED_LANGUAGES
    ):
        raise SourceValidationError("missing_required_language_text")
    if any(
        not isinstance(jobs[index], str) or not jobs[index].strip()
        for index in range(len(REQUIRED_LANGUAGES))
    ):
        raise SourceValidationError("missing_required_language_job")
    source = {key: value for key, value in row.items() if key != "ftxt"}
    return source, {language: texts[language] for language in REQUIRED_LANGUAGES}


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def prepare_records(
    filewise_rows: Iterable[dict],
    bilingual_rows: Iterable[dict],
    out_dir: Path,
    *,
    min_words: int = 150,
    max_words: int = 450,
    from_year: int = 2015,
    to_year: int = 2025,
    input_sources: dict | None = None,
    legacy_docs: Path | None = None,
    legacy_blocks: Path | None = None,
    legacy_sixway_dir: Path | None = None,
) -> dict:
    """Prepare local records with bounded memory and an atomic completed manifest.

    ``input_sources`` records optional local-file hashes; the declared upstream
    revisions are fixed. Local data provenance must be retained by its downloader.
    A malformed pivot link quarantines its document instead of silently selecting
    another version. Quarantined originals remain available for reference context.
    """
    if (
        any(type(value) is not int for value in (min_words, max_words, from_year, to_year))
        or not 0 < min_words <= max_words
        or from_year > to_year
    ):
        raise ValueError("Require 0 < min_words <= max_words and from_year <= to_year")
    legacy_values = (legacy_docs, legacy_blocks, legacy_sixway_dir)
    if any(value is not None for value in legacy_values) and not all(
        value is not None for value in legacy_values
    ):
        raise ValueError("Legacy docs, blocks and sixway directory must be supplied together")
    legacy = None
    if legacy_docs is not None:
        legacy = {
            key: str(Path(value).expanduser().resolve())
            for key, value in zip(("docs", "blocks", "sixway_dir"), legacy_values)
        }
        if (
            not Path(legacy["docs"]).is_file()
            or not Path(legacy["blocks"]).is_file()
            or not Path(legacy["sixway_dir"]).is_dir()
        ):
            raise ValueError(
                "Legacy docs/blocks must be existing files and sixway_dir an existing directory"
            )
    out_dir = Path(out_dir).expanduser().absolute()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.building-", dir=out_dir.parent))
    stage = final = None
    counts: Counter = Counter()
    rejected: Counter = Counter()
    publication_ranges = {kind: {"first": None, "last": None} for kind in ("documents", "targets")}
    try:
        stage = _connect(working / "staging.sqlite")
        stage.executescript("""
            CREATE TABLE source_docs (source_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                                      fingerprint TEXT NOT NULL,
                                      invalid_reason TEXT NOT NULL DEFAULT '');
            CREATE TABLE source_versions (source_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                                          PRIMARY KEY (source_id, fingerprint));
            CREATE TABLE source_texts (source_id TEXT NOT NULL, language TEXT NOT NULL,
                text TEXT NOT NULL, PRIMARY KEY (source_id, language));
            CREATE TABLE links (source_id TEXT NOT NULL, language TEXT NOT NULL,
                english TEXT NOT NULL, other TEXT NOT NULL,
                PRIMARY KEY (source_id, language, english, other));
        """)
        for row in filewise_rows:
            counts["filewise_rows"] += 1
            try:
                source, texts = _validate_source(row)
            except SourceValidationError as error:
                rejected[str(error)] += 1
                continue
            encoded = _json(source)
            fingerprint = sha256_text(_json({"source": source, "texts": texts}))
            changes = stage.total_changes
            stage.execute(
                "INSERT OR IGNORE INTO source_versions VALUES (?,?)", (source["id"], fingerprint)
            )
            duplicate = stage.total_changes == changes
            previous = stage.execute(
                "SELECT fingerprint FROM source_docs WHERE source_id=?", (source["id"],)
            ).fetchone()
            if previous is not None:
                if not duplicate:
                    stage.execute(
                        "UPDATE source_docs SET invalid_reason='conflicting_source_versions' "
                        "WHERE source_id=?",
                        (source["id"],),
                    )
                    rejected["conflicting_source_versions"] += 1
                    # Preserve a deterministic representative for inspection;
                    # its ambiguous identity remains excluded from generation.
                    if fingerprint < previous[0]:
                        stage.execute(
                            "UPDATE source_docs SET payload=?,fingerprint=? WHERE source_id=?",
                            (encoded, fingerprint, source["id"]),
                        )
                        stage.execute("DELETE FROM source_texts WHERE source_id=?", (source["id"],))
                        stage.executemany(
                            "INSERT INTO source_texts VALUES (?,?,?)",
                            ((source["id"], lang, text) for lang, text in texts.items()),
                        )
                else:
                    counts["duplicate_filewise_rows"] += 1
                continue
            stage.execute(
                "INSERT INTO source_docs(source_id,payload,fingerprint) VALUES (?,?,?)",
                (source["id"], encoded, fingerprint),
            )
            stage.executemany(
                "INSERT INTO source_texts VALUES (?,?,?)",
                ((source["id"], lang, text) for lang, text in texts.items()),
            )
            counts["admitted_documents"] += 1
            if counts["filewise_rows"] % 1000 == 0:
                stage.commit()
        stage.commit()
        # A small LRU avoids reloading a document for adjacent bilingual rows;
        # its size never depends on corpus size. No full-corpus text dictionary.
        cache: OrderedDict[str, tuple[dict, dict[str, Paragraphs]]] = OrderedDict()
        for row in bilingual_rows:
            counts["bilingual_rows"] += 1
            src, dst = row.get("src_lang"), row.get("dst_lang")
            if not (
                (src == "en" and dst in OTHER_LANGUAGES) or (dst == "en" and src in OTHER_LANGUAGES)
            ):
                continue
            source_id = row.get("id")
            if not isinstance(source_id, str):
                rejected["missing_bilingual_source_id"] += 1
                continue
            entry = cache.get(source_id)
            if entry is None:
                record = stage.execute(
                    "SELECT payload FROM source_docs WHERE source_id=?", (source_id,)
                ).fetchone()
                if record is None:
                    continue
                source = json.loads(record[0])
                texts = dict(
                    stage.execute(
                        "SELECT language,text FROM source_texts WHERE source_id=?", (source_id,)
                    )
                )
                entry = source, {lang: Paragraphs.of(text) for lang, text in texts.items()}
                cache[source_id] = entry
                if len(cache) > 4:
                    cache.popitem(last=False)
            else:
                cache.move_to_end(source_id)
            try:
                link = verify_link(row, *entry)
            except SourceValidationError as error:
                rejected[str(error)] += 1
                stage.execute(
                    "UPDATE source_docs SET invalid_reason=CASE WHEN invalid_reason='' OR "
                    "(invalid_reason<>'conflicting_source_versions' AND ?<invalid_reason) "
                    "THEN ? ELSE invalid_reason END WHERE source_id=?",
                    (str(error), str(error), source_id),
                )
                continue
            changes = stage.total_changes
            stage.execute(
                "INSERT OR IGNORE INTO links VALUES (?,?,?,?)",
                (source_id, link.language, _json(link.english), _json(link.other)),
            )
            counts["verified_links" if stage.total_changes > changes else "duplicate_links"] += 1
            if counts["bilingual_rows"] % 10000 == 0:
                stage.commit()
        stage.commit()
        cache.clear()
        final = _connect(working / "sources.sqlite")
        final.executescript("""
            CREATE TABLE docs (doc_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE texts (doc_id TEXT NOT NULL, language TEXT NOT NULL,
                text TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY (doc_id, language));
            CREATE TABLE blocks (doc_id TEXT NOT NULL, block_index INTEGER NOT NULL,
                block_id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL,
                PRIMARY KEY (doc_id, block_index));
        """)
        for source_id, payload, invalid in stage.execute(
            "SELECT source_id,payload,invalid_reason FROM source_docs ORDER BY source_id"
        ):
            source = json.loads(payload)
            texts = dict(
                stage.execute(
                    "SELECT language,text FROM source_texts WHERE source_id=? ORDER BY language",
                    (source_id,),
                )
            )
            links = (
                VerifiedLink(lang, tuple(json.loads(english)), tuple(json.loads(other)))
                for lang, english, other in stage.execute(
                    "SELECT language,english,other FROM links WHERE source_id=? ORDER BY language,english,other",
                    (source_id,),
                )
            )
            metadata, blocks = build_document(
                source,
                texts,
                links,
                min_words=min_words,
                max_words=max_words,
                from_year=from_year,
                to_year=to_year,
                invalid_reason=invalid,
                rejected=rejected,
            )
            try:
                publication = date.fromisoformat(
                    str(metadata.get("publication_date", ""))[:10]
                ).isoformat()
            except ValueError:
                counts["documents_without_valid_publication_date"] += 1
            else:
                for kind in ("documents", "targets"):
                    if kind == "targets" and not metadata["n_targets"]:
                        continue
                    bounds = publication_ranges[kind]
                    bounds["first"] = min(bounds["first"] or publication, publication)
                    bounds["last"] = max(bounds["last"] or publication, publication)
            final.execute("INSERT INTO docs VALUES (?,?)", (metadata["doc_id"], _json(metadata)))
            final.executemany(
                "INSERT INTO texts VALUES (?,?,?,?)",
                (
                    (metadata["doc_id"], lang, texts[lang], sha256_text(texts[lang]))
                    for lang in sorted(texts)
                ),
            )
            final.executemany(
                "INSERT INTO blocks VALUES (?,?,?,?)",
                (
                    (metadata["doc_id"], block["block_index"], block["block_id"], _json(block))
                    for block in blocks
                ),
            )
            counts["documents"] += 1
            counts["blocks"] += len(blocks)
            counts["targets"] += metadata["n_targets"]
            counts["documents_with_targets"] += bool(metadata["n_targets"])
            counts["quarantined_documents"] += bool(invalid)
            counts["reference_only_documents"] += not bool(metadata["n_targets"])
            if counts["documents"] % 1000 == 0:
                final.commit()
        final.commit()
        final.close()
        final = None
        stage.close()
        stage = None
        (working / "staging.sqlite").unlink()
        # Keep zero-valued core counters available to consumers too.
        for key in (
            "documents",
            "blocks",
            "targets",
            "documents_with_targets",
            "quarantined_documents",
            "filewise_rows",
            "bilingual_rows",
            "admitted_documents",
            "verified_links",
        ):
            counts.setdefault(key, 0)
        inputs = input_sources or {}
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "database": "sources.sqlite",
            "database_sha256": sha256_file(working / "sources.sqlite"),
            "sources": {
                "filewise": {
                    "repository": FILEWISE_REPOSITORY,
                    "revision": FILEWISE_REVISION,
                    "inputs": inputs.get("filewise", []),
                },
                "bilingual": {
                    "repository": BILINGUAL_REPOSITORY,
                    "revision": BILINGUAL_REVISION,
                    "inputs": inputs.get("bilingual", []),
                },
            },
            "config": {
                "min_words": min_words,
                "max_words": max_words,
                "from_year": from_year,
                "to_year": to_year,
                "required_languages": list(REQUIRED_LANGUAGES),
                "pivot_language": "en",
                "word_count": "whitespace",
                "paragraph_separator": "\n\n",
                "alignment": "verified_en_pivot_components",
                "allow_model_alignment": False,
            },
            "counts": dict(sorted(counts.items())),
            "rejected": dict(sorted(rejected.items())),
            "publication_ranges": publication_ranges,
            "coverage": {
                "scope": "supplied_local_records",
                "complete_upstream_snapshot": False,
                "revision_provenance": "declared_by_input_provider",
            },
        }
        if legacy is not None:
            manifest["legacy"] = legacy
        (working / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        audit(working / "manifest.json")
        if out_dir.exists():
            raise FileExistsError(f"Output directory already exists: {out_dir}")
        os.rename(working, out_dir)
        return manifest
    except BaseException:
        if final is not None:
            final.close()
        if stage is not None:
            stage.close()
        shutil.rmtree(working, ignore_errors=True)
        raise


def prepare(filewise: Path, bilingual: Path, out_dir: Path, **kwargs: Any) -> dict:
    """Hash and stream local Parquet shards; never download a corpus implicitly."""
    if Path(out_dir).expanduser().exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    files = {"filewise": parquet_files(filewise), "bilingual": parquet_files(bilingual)}
    inputs = {
        kind: [
            {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ]
        for kind, paths in files.items()
    }
    return prepare_records(
        parquet_records(files["filewise"]),
        parquet_records(files["bilingual"]),
        out_dir,
        input_sources=inputs,
        **kwargs,
    )


def _audit_alignment(
    block: dict,
    metadata: dict,
    paragraphs: dict[str, Paragraphs],
    config: dict,
    profiles: list[_Profile],
) -> None:
    proof = block.get("alignment_links", [])
    if set(block["texts"]) != set(REQUIRED_LANGUAGES):
        if proof or block.get("usable") or set(block["texts"]) != {"en"}:
            raise SourceValidationError("unsafe_unaligned_context")
        return
    jobs = metadata.get("upstream", {}).get("job_numbers", [])
    if len(jobs) != len(SOURCE_LANGUAGES) or not isinstance(proof, list) or not proof:
        raise SourceValidationError("missing_alignment_proof")
    links = []
    for item in proof:
        language = item.get("language")
        if language not in OTHER_LANGUAGES:
            raise SourceValidationError("invalid_alignment_proof_language")
        indices = []
        for field, lang in (("english", "en"), ("other", language)):
            numbers = item.get(field)
            if (
                not isinstance(numbers, list)
                or not numbers
                or any(type(number) is not int or number < 0 for number in numbers)
                or numbers != sorted(set(numbers))
                or numbers[-1] >= len(paragraphs[lang].values)
            ):
                raise SourceValidationError("invalid_alignment_proof_indices")
            joined = "\n".join(paragraphs[lang].values[number] for number in numbers)
            if sha256_text(joined) != item.get(f"{field}_text_sha256") or jobs[
                SOURCE_LANGUAGES.index(lang)
            ] != item.get(f"{field}_job_number"):
                raise SourceValidationError("alignment_proof_source_mismatch")
            indices.append(tuple(numbers))
        links.append(VerifiedLink(language, *indices))
    rejected: Counter = Counter()
    components = aligned_components(paragraphs, links, rejected)
    if any(rejected.values()) or not components:
        raise SourceValidationError("incomplete_or_crossing_alignment_proof")
    packed = _pack(components, paragraphs, profiles, config["min_words"], config["max_words"])
    if len(packed) != 1 or packed[0].component_count != block.get("alignment_components"):
        raise SourceValidationError("noncontiguous_alignment_proof")
    component = packed[0]
    if any(
        component.spans[language] != (location["start"], location["end"])
        for language, location in block["source_locations"].items()
    ):
        raise SourceValidationError("alignment_proof_span_mismatch")
    if (
        block.get("para_start") != component.indices["en"][0]
        or block.get("para_end") != component.indices["en"][-1]
    ):
        raise SourceValidationError("alignment_proof_paragraph_mismatch")
    flags = _component_flags(component, profiles)
    if not metadata["target_eligible"]:
        flags.append("outside_target_publication_window")
    flags.extend(flag for flag in junk_flags(block["text"]) if flag not in flags)
    if block.get("junk_flags") != flags or block.get("usable") != (not flags):
        raise SourceValidationError("block_hygiene_mismatch")


def verify_block(block: dict, metadata: dict, texts: dict[str, str], config: dict) -> None:
    """Verify one selected block against its full originals and alignment proof.

    Generation can call this bounded document-level check without scanning the
    prepared database. A self-consistent file checksum alone does not substitute
    for complete per-paragraph multiway coverage.
    """
    if set(texts) != set(REQUIRED_LANGUAGES) or any(not text.strip() for text in texts.values()):
        raise SourceValidationError("missing_required_language_text")
    locations = block.get("source_locations", {})
    if set(locations) != set(block.get("texts", {})) or "en" not in locations:
        raise SourceValidationError("block_language_location_mismatch")
    for language, location in locations.items():
        start, end = location.get("start"), location.get("end")
        if (
            language not in texts
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(texts[language])
            or location.get("document_sha256") != sha256_text(texts[language])
            or texts[language][start:end] != block["texts"][language]
        ):
            raise SourceValidationError("invalid_original_text_span")
    if block.get("text") != block["texts"]["en"]:
        raise SourceValidationError("english_block_text_mismatch")
    words = len(block["text"].split())
    if block.get("token_count") != words or block.get("in_range") != (
        config["min_words"] <= words <= config["max_words"]
    ):
        raise SourceValidationError("block_word_count_mismatch")
    year = _year(metadata.get("publication_date"))
    target_eligible = year is not None and config["from_year"] <= year <= config["to_year"]
    if metadata.get("target_eligible") != target_eligible:
        raise SourceValidationError("document_publication_metadata_mismatch")
    if (
        block.get("usable")
        and block.get("in_range")
        and (not target_eligible or metadata.get("invalid_reason"))
    ):
        raise SourceValidationError("unsafe_target_block")
    paragraphs = {language: Paragraphs.of(text) for language, text in texts.items()}
    _audit_alignment(block, metadata, paragraphs, config, _profiles(paragraphs["en"]))


def audit(manifest_path: Path) -> dict:
    """Verify the manifest, original document hashes, target membership and slices."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise SourceValidationError("unsupported_manifest_format")
    if manifest.get("database") != "sources.sqlite":
        raise SourceValidationError("unexpected_database_path")
    for kind, revision in (("filewise", FILEWISE_REVISION), ("bilingual", BILINGUAL_REVISION)):
        if manifest.get("sources", {}).get(kind, {}).get("revision") != revision:
            raise SourceValidationError("unexpected_source_revision")
    database = manifest_path.parent / manifest["database"]
    if sha256_file(database) != manifest.get("database_sha256"):
        raise SourceValidationError("database_sha256_mismatch")
    config = manifest["config"]
    if config.get("required_languages") != list(REQUIRED_LANGUAGES):
        raise SourceValidationError("unexpected_required_languages")
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    counts: Counter = Counter()
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise SourceValidationError("invalid_sqlite_database")
        for doc_id, payload in connection.execute(
            "SELECT doc_id,payload FROM docs ORDER BY doc_id"
        ):
            metadata = json.loads(payload)
            if metadata.get("doc_id") != doc_id or doc_id != document_id(metadata["upstream_id"]):
                raise SourceValidationError("document_identity_mismatch")
            year = _year(metadata.get("publication_date"))
            if metadata.get("year") != str(year or "") or metadata.get("target_eligible") != (
                year is not None and config["from_year"] <= year <= config["to_year"]
            ):
                raise SourceValidationError("document_publication_metadata_mismatch")
            texts, hashes = {}, {}
            for language, text, digest in connection.execute(
                "SELECT language,text,sha256 FROM texts WHERE doc_id=?", (doc_id,)
            ):
                if sha256_text(text) != digest:
                    raise SourceValidationError("document_text_sha256_mismatch")
                texts[language], hashes[language] = text, digest
            if set(texts) != set(REQUIRED_LANGUAGES) or any(
                not text.strip() for text in texts.values()
            ):
                raise SourceValidationError("missing_required_language_text")
            if metadata["char_count"] != len(texts["en"]):
                raise SourceValidationError("document_character_count_mismatch")
            paragraphs = {language: Paragraphs.of(text) for language, text in texts.items()}
            profiles = _profiles(paragraphs["en"])
            targets = []
            previous_end = {language: -1 for language in REQUIRED_LANGUAGES}
            n_blocks = 0
            for index, block_id, block_payload in connection.execute(
                "SELECT block_index,block_id,payload FROM blocks WHERE doc_id=? ORDER BY block_index",
                (doc_id,),
            ):
                block = json.loads(block_payload)
                if (
                    index != n_blocks
                    or block.get("block_id") != block_id
                    or block_id != f"{doc_id}#{index}"
                    or block.get("doc_id") != doc_id
                    or block.get("block_index") != index
                    or block.get("n_blocks") != metadata["n_blocks"]
                ):
                    raise SourceValidationError("block_identity_mismatch")
                if block.get("line_start") is not None or block.get("line_end") is not None:
                    raise SourceValidationError("unexpected_legacy_line_coordinates")
                locations = block.get("source_locations", {})
                if set(locations) != set(block.get("texts", {})) or "en" not in locations:
                    raise SourceValidationError("block_language_location_mismatch")
                for language, location in locations.items():
                    start, end = location.get("start"), location.get("end")
                    if (
                        language not in texts
                        or type(start) is not int
                        or type(end) is not int
                        or not 0 <= start < end <= len(texts[language])
                        or start < previous_end[language]
                        or location.get("document_sha256") != hashes[language]
                        or texts[language][start:end] != block["texts"][language]
                    ):
                        raise SourceValidationError("invalid_original_text_span")
                    previous_end[language] = end
                if block.get("text") != block["texts"]["en"]:
                    raise SourceValidationError("english_block_text_mismatch")
                _audit_alignment(block, metadata, paragraphs, config, profiles)
                words = len(block["text"].split())
                in_range = config["min_words"] <= words <= config["max_words"]
                if block.get("token_count") != words or block.get("in_range") != in_range:
                    raise SourceValidationError("block_word_count_mismatch")
                if block.get("usable") and in_range:
                    if (
                        set(locations) != set(REQUIRED_LANGUAGES)
                        or any(not text.strip() for text in block["texts"].values())
                        or block.get("junk_flags")
                        or metadata.get("invalid_reason")
                        or not metadata["target_eligible"]
                    ):
                        raise SourceValidationError("unsafe_target_block")
                    targets.append(index)
                n_blocks += 1
            if (
                n_blocks != metadata["n_blocks"]
                or targets != metadata["target_idxs"]
                or len(targets) != metadata["n_targets"]
            ):
                raise SourceValidationError("document_target_membership_mismatch")
            counts["documents"] += 1
            counts["blocks"] += n_blocks
            counts["targets"] += len(targets)
            counts["documents_with_targets"] += bool(targets)
            counts["quarantined_documents"] += bool(metadata.get("invalid_reason"))
        for table in ("texts", "blocks"):
            if connection.execute(
                f"SELECT 1 FROM {table} t LEFT JOIN docs d ON t.doc_id=d.doc_id "
                "WHERE d.doc_id IS NULL LIMIT 1"
            ).fetchone():
                raise SourceValidationError("orphan_source_rows")
        for key in (
            "documents",
            "blocks",
            "targets",
            "documents_with_targets",
            "quarantined_documents",
        ):
            if counts[key] != manifest["counts"].get(key):
                raise SourceValidationError("manifest_count_mismatch")
        return {
            key: counts[key]
            for key in (
                "documents",
                "blocks",
                "targets",
                "documents_with_targets",
                "quarantined_documents",
            )
        }
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "prepare", help="Prepare pinned local fulltext and bilingual Parquet"
    )
    build.add_argument("--filewise", type=Path, required=True)
    build.add_argument("--bilingual", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--min-words", type=int, default=150)
    build.add_argument("--max-words", type=int, default=450)
    build.add_argument("--from-year", type=int, default=2015)
    build.add_argument("--to-year", type=int, default=2025)
    build.add_argument("--legacy-docs", type=Path)
    build.add_argument("--legacy-blocks", type=Path)
    build.add_argument("--legacy-sixway-dir", type=Path)
    check = commands.add_parser("audit", help="Check every prepared document and block")
    check.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit(args.manifest), sort_keys=True))
    else:
        manifest = prepare(
            args.filewise,
            args.bilingual,
            args.out_dir,
            min_words=args.min_words,
            max_words=args.max_words,
            from_year=args.from_year,
            to_year=args.to_year,
            legacy_docs=args.legacy_docs,
            legacy_blocks=args.legacy_blocks,
            legacy_sixway_dir=args.legacy_sixway_dir,
        )
        print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
