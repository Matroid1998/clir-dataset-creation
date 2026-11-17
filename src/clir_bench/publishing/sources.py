"""Source-unit adapters. UN generation blocks are never used as corpus documents."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
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
    if "dataset_manifest" in spec:
        if "articles" in spec:
            raise ValueError("EUR-Lex dataset_manifest and articles are mutually exclusive")
        _load_eurlex_manifest(index, spec, root, wanted)
        return
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


def _load_eurlex_manifest(index, spec, root, wanted):
    """Read complete source articles from a pinned profile, including baseline references."""
    from clir_bench.domains.legal.structure import ACT_LANGUAGES
    from clir_bench.domains.legal.structure.profile import load_profile

    languages = spec["languages"]
    if (not isinstance(languages, list) or not languages
            or not all(isinstance(language, str) for language in languages)
            or len(set(languages)) != len(languages)
            or not set(languages) <= set(ACT_LANGUAGES)):
        raise ValueError("EUR-Lex languages must be distinct supported language codes")
    profile = load_profile(root / spec["dataset_manifest"])
    with profile.file("quarantine.jsonl").open(encoding="utf-8") as handle:
        quarantined = {json.loads(line)["celex_id"] for line in handle if line.strip()}
    unit_types = set(spec.get("unit_types", []))
    # Conflicts must be detected even outside the publication's language/scope
    # filters. Keep their fingerprints on disk when exporting a large corpus.
    with tempfile.TemporaryDirectory(prefix="eurlex-source-identities-") as directory:
        connection = sqlite3.connect(Path(directory) / "identities.sqlite")
        try:
            connection.execute("CREATE TABLE seen(source TEXT, language TEXT, digest TEXT, "
                               "PRIMARY KEY(source,language))")
            connection.execute("CREATE TABLE identities(source TEXT PRIMARY KEY, signature TEXT)")
            with profile.file("articles.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    raw = json.loads(line)
                    if raw.get("celex_id") in quarantined:
                        continue
                    source, language = raw.get("eli_id"), raw.get("language")
                    if not isinstance(source, str) or not source.strip():
                        raise ValueError("profile article requires an ELI identifier")
                    if language not in ACT_LANGUAGES:
                        raise ValueError("profile article has an unsupported language")
                    signature = dumps([raw["celex_id"], raw["unit_type"], raw.get("article_number", "")])
                    prior_identity = connection.execute(
                        "SELECT signature FROM identities WHERE source=?", (source,)
                    ).fetchone()
                    if prior_identity and prior_identity[0] != signature:
                        raise ValueError(f"conflicting EUR-Lex article identity: {source}")
                    connection.execute("INSERT OR IGNORE INTO identities VALUES (?,?)", (source, signature))
                    digest = hashlib.sha256(dumps(raw).encode("utf-8")).hexdigest()
                    prior = connection.execute(
                        "SELECT digest FROM seen WHERE source=? AND language=?", (source, language)
                    ).fetchone()
                    if prior:
                        if prior[0] != digest:
                            raise ValueError(f"conflicting EUR-Lex article versions: {source} ({language})")
                        continue
                    connection.execute("INSERT INTO seen VALUES (?,?,?)", (source, language, digest))
                    key = f"{raw['celex_id']}:{raw['article_number']}"
                    if language not in languages or (unit_types and raw["unit_type"] not in unit_types):
                        continue
                    if wanted is not None and source not in wanted and key not in wanted:
                        continue
                    if not isinstance(raw.get("text"), str) or not raw["text"].strip():
                        raise ValueError(f"profile article has no original text: {source} ({language})")
                    metadata = {key: value for key, value in raw.items() if key != "text"}
                    metadata.update(dataset_identity=profile.identity,
                                    dataset_target_act=raw["celex_id"] in profile.target_acts)
                    index.add(source_row(
                        "eurlex", source, raw["celex_id"], language, raw["unit_type"],
                        raw.get("heading", ""), raw["text"], metadata,
                    ), article_key=key)
            profile.assert_unchanged()
        finally:
            connection.close()


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
    genres = None
    if "document_genres" in spec:
        configured = spec["document_genres"]
        if (
            not isinstance(configured, list)
            or not configured
            or not all(isinstance(value, str) for value in configured)
            or not set(configured) <= {"resolution", "meeting", "letter"}
        ):
            raise ValueError(
                "UN document_genres must be a nonempty list of resolution, meeting, or letter"
            )
        genres = set(configured)
    with (root / spec["documents"]).open(encoding="utf-8") as handle:
        documents = sorted((json.loads(line) for line in handle), key=lambda d: d["line_start"])
    previous = 0
    for doc in documents:
        start, end = int(doc["line_start"]), int(doc["line_end"])
        if start != previous + 1 or end < start or end - start + 1 != int(doc["n_lines"]):
            raise ValueError(f"invalid/gapped UN document range: {doc['doc_id']}")
        previous = end
    eligible = None
    if genres is not None:
        from clir_bench.domains.legal.qac.un_batch import _excluded_doc, genre_for

        # Only document classes constrain retrieval. Generation passage, context
        # length and reference-completeness gates must not remove corpus documents.
        eligible = {
            doc["doc_id"]
            for doc in documents
            if not _excluded_doc(doc["doc_id"])
            and genre_for(doc["doc_id"], doc.get("title", "")) in genres
        }
    for language in spec["languages"]:
        path = root / f"{spec['text_prefix']}.{language}"
        with path.open(encoding="utf-8") as handle:
            for doc in documents:
                include = (wanted is None or doc["doc_id"] in wanted) and (
                    eligible is None or doc["doc_id"] in eligible
                )
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
