"""Source-unit adapters. UN generation blocks are never used as corpus documents."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pyarrow as pa

from .evidence import saved_spans
from .records import article_references, dumps, identifiers, source_identity, unit_id

CORPUS_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "_id",
            "corpus",
            "source_id",
            "parent_id",
            "language",
            "unit_type",
            "title",
            "text",
            "metadata_json",
        )
    ]
    + [("paired_document_ids", pa.list_(pa.string()))]
)


class SourceIndex:
    """Disk-backed index keeps full legal corpora out of Python memory."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            CREATE TABLE units (
                id TEXT PRIMARY KEY, corpus TEXT, source TEXT, parent TEXT,
                language TEXT, article_key TEXT, payload TEXT
            );
            CREATE INDEX by_source ON units(corpus, source);
            CREATE INDEX by_article ON units(article_key, language);
        """)
        self.duplicates = 0

    def close(self):
        self.connection.close()

    def add(self, row: dict, article_key: str = ""):
        # Match the generation article loader: the last record for an ELI/language wins.
        existed = self.connection.execute("SELECT 1 FROM units WHERE id=?", (row["_id"],))
        if existed.fetchone():
            if row["corpus"] != "eurlex":
                raise ValueError(f"duplicate corpus ID: {row['_id']}")
            self.duplicates += 1
        self.connection.execute(
            "INSERT OR REPLACE INTO units VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["_id"],
                row["corpus"],
                row["source_id"],
                row["parent_id"],
                row["language"],
                article_key,
                dumps(row),
            ),
        )

    def get(self, identifier: str) -> dict | None:
        result = self.connection.execute("SELECT payload FROM units WHERE id=?", (identifier,))
        found = result.fetchone()
        return json.loads(found[0]) if found else None

    def resolve_article(self, reference: str, language: str) -> dict | None:
        result = self.connection.execute(
            "SELECT payload FROM units WHERE corpus='eurlex' AND language=? "
            "AND (source=? OR article_key=?)",
            (language, reference, reference),
        ).fetchall()
        if len(result) > 1:
            raise ValueError(f"ambiguous article reference: {reference} ({language})")
        return json.loads(result[0][0]) if result else None

    def translations(self, corpus: str, source: str):
        for (payload,) in self.connection.execute(
            "SELECT payload FROM units WHERE corpus=? AND source=? ORDER BY id", (corpus, source)
        ):
            yield json.loads(payload)

    def rows(self):
        for (payload,) in self.connection.execute("SELECT payload FROM units ORDER BY id"):
            yield json.loads(payload)


def targets(
    generations, *, evidence_languages: dict[str, set[str]] | None = None
) -> dict[str, set[str]]:
    wanted: dict[str, set[str]] = {}
    for corpus, row, _ in generations:
        selected = wanted.setdefault(corpus, set())
        selected.add(source_identity(corpus, row)[0])
        if corpus == "eurlex":
            selected.update(article_references(row))
        for span in saved_spans(row):
            source = span.get("doc_id") if corpus == "un" else span["source_id"]
            if not source:
                raise ValueError("UN evidence is missing its original document ID")
            selected.add(source)
            if evidence_languages is not None:
                evidence_languages.setdefault(corpus, set()).add(span["language"])
    return wanted


def source_row(corpus, source, parent, language, kind, title, text, metadata, partners=()):
    return {
        "_id": unit_id(corpus, source, language),
        "corpus": corpus,
        "source_id": source,
        "parent_id": parent,
        "language": language,
        "unit_type": kind,
        "title": title,
        "text": text,
        "metadata_json": dumps(metadata),
        "paired_document_ids": list(partners),
    }


def load_eurlex(index: SourceIndex, spec: dict, root: Path, wanted: set[str] | None):
    languages = set(spec["languages"])
    unit_types = set(spec.get("unit_types", []))
    with (root / spec["articles"]).open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            source = raw.get("eli_id")
            key = f"{raw['celex_id']}:{raw['article_number']}"
            if not source or raw["language"] not in languages:
                continue
            if unit_types and raw["unit_type"] not in unit_types:
                continue
            if wanted is not None and source not in wanted and key not in wanted:
                continue
            index.add(
                source_row(
                    "eurlex",
                    source,
                    raw["celex_id"],
                    raw["language"],
                    raw["unit_type"],
                    raw.get("heading", ""),
                    raw["text"],
                    {k: v for k, v in raw.items() if k != "text"},
                ),
                article_key=key,
            )


def load_chemistry(index: SourceIndex, spec: dict, root: Path, wanted: set[str] | None):
    with (root / spec["corpus_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    selected = None if wanted is None else set(wanted)
    if selected is not None:
        for source in wanted:
            if source not in rows:
                raise ValueError(f"missing chemistry source: {source}")
            # Direct accepted edges only: never compute the transitive family closure.
            selected.update(identifiers(rows[source].get("paired_document_ids")))
    for source, raw in rows.items():
        if selected is not None and source not in selected:
            continue
        partners = identifiers(raw.get("paired_document_ids"))
        for other in partners:
            if other not in rows or source not in identifiers(
                rows[other].get("paired_document_ids")
            ):
                raise ValueError(f"missing or asymmetric chemistry pair: {source} -> {other}")
        text = "\n\n".join(v for v in (raw.get("title"), raw.get("abstract")) if v)
        if not text:
            raise ValueError(f"empty chemistry title/abstract: {source}")
        index.add(
            source_row(
                "chemistry",
                source,
                raw["publication_number"],
                raw["language"],
                "title_abstract",
                raw.get("title", ""),
                text,
                raw,
                partners,
            )
        )


def load_un(index: SourceIndex, spec: dict, root: Path, wanted: set[str] | None):
    with (root / spec["documents"]).open(encoding="utf-8") as handle:
        documents = sorted((json.loads(line) for line in handle), key=lambda d: d["line_start"])
    previous = 0
    for doc in documents:
        start, end = int(doc["line_start"]), int(doc["line_end"])
        if start != previous + 1 or end < start or end - start + 1 != int(doc["n_lines"]):
            raise ValueError(f"invalid/gapped UN document range: {doc['doc_id']}")
        previous = end
    for language in spec["languages"]:
        path = root / f"{spec['text_prefix']}.{language}"
        with path.open(encoding="utf-8") as handle:
            for doc in documents:
                include = wanted is None or doc["doc_id"] in wanted
                lines = []
                for _ in range(int(doc["n_lines"])):
                    line = handle.readline()
                    if not line:
                        raise ValueError(f"truncated UN language file: {path}, {doc['doc_id']}")
                    if include:
                        lines.append(line.rstrip("\r\n"))
                if include:
                    metadata = {**doc, "title_language": "en", "line_numbers": "1-based inclusive"}
                    index.add(
                        source_row(
                            "un",
                            doc["doc_id"],
                            doc["doc_id"],
                            language,
                            "document",
                            doc.get("title", "") if language == "en" else "",
                            "\n".join(lines),
                            metadata,
                        )
                    )
            if handle.readline():
                raise ValueError(f"UN file has lines beyond document index: {path}")
