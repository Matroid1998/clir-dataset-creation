"""Read-only, manifest-pinned UN originals and deterministic multiway blocks.

The legacy line-aligned corpus retains its existing reader. This adapter never
infers translated positions from line numbers or searches for a repeated quote.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from clir_bench.domains.legal.qac import un_context as ctx
from clir_bench.domains.legal.qac import un_references as refs

LANGUAGES = ("ar", "en", "es", "fr", "ru", "zh")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UPRPRCIndex(ctx.BlockIndex):
    """Generation adapter over an immutable ``un-uprprc-v1`` manifest.

    Optional ``legacy`` paths add reference documents only; their bytes are also
    pinned in ``identity`` so a resumed run cannot acquire different references.
    Connections are short lived and read only, including in generation threads.
    """

    def __init__(self, manifest_path: str | Path):
        path = Path(manifest_path).expanduser().resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if manifest.get("format") != "un-uprprc-v1" or manifest.get("version") != 1:
            raise ValueError("unsupported UN dataset manifest format/version")
        self.manifest_path = path
        self.manifest = manifest
        self.database = self._path(manifest["database"])
        database_hash = sha256_file(self.database)
        if database_hash != manifest.get("database_sha256"):
            raise ValueError("UN dataset database checksum differs from its manifest")
        self.identity = {"manifest_path": str(path),
                         "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                         "database_sha256": database_hash}
        self._database_stat = self._stat()
        with self._connect() as connection:
            self.docs = {doc_id: json.loads(payload) for doc_id, payload in
                         connection.execute("SELECT doc_id,payload FROM docs ORDER BY doc_id")}
        for doc_id, doc in self.docs.items():
            if doc.get("doc_id") != doc_id:
                raise ValueError("UN document metadata identity mismatch")
        self._new_ids = frozenset(self.docs)
        self._legacy = None
        self._legacy_offsets: dict[str, dict[str, tuple[int, int]]] = {}
        self._legacy_stats: dict[str, tuple] = {}
        self._legacy_lock = RLock()
        legacy = manifest.get("legacy")
        if legacy:
            files = {key: self._path(legacy[key]) for key in ("docs", "blocks", "sixway_dir")}
            self._legacy = ctx.BlockIndex(docs_path=files["docs"], blocks_path=files["blocks"],
                                          sixway_dir=files["sixway_dir"])
            hashes = {key: sha256_file(files[key]) for key in ("docs", "blocks")}
            for language in LANGUAGES:
                text_path = files["sixway_dir"] / f"UNv1.0.6way.{language}"
                hashes[language] = sha256_file(text_path)
                self._legacy_stats[language] = self._file_stat(text_path)
            self.identity["legacy_sha256"] = hashes
            for doc_id, doc in self._legacy.docs.items():
                if doc_id in self.docs:
                    raise ValueError("legacy and new UN document IDs collide")
                self.docs[doc_id] = dict(doc, target_idxs=[], n_targets=0)
        self.symbols = refs.SymbolIndex.from_docs({d: row for d, row in self.docs.items()
            if row.get("invalid_reason") != "conflicting_source_versions"})
        self._incomplete = None
        self._status_loaded = False

    def _path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.manifest_path.parent / path).resolve()

    def _stat(self):
        return self._file_stat(self.database)

    @staticmethod
    def _file_stat(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    @contextmanager
    def _connect(self):
        if self._stat() != self._database_stat:
            raise ValueError("UN dataset database changed during use")
        connection = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)
        try:
            yield connection
        finally:
            connection.close()

    def close(self):
        """Compatibility with source-export readers; no persistent handles."""

    def iter_documents(self):
        return iter(self.docs.values())

    def document_text(self, doc_id: str, language: str) -> str:
        if doc_id not in self._new_ids:
            if self._legacy is None or doc_id not in self._legacy.docs:
                return ""
            self.preload_translations((language,), (doc_id,))
            path = self._legacy.sixway_dir / f"UNv1.0.6way.{language}"
            self._check_legacy(language, path)
            start, end = self._legacy_offsets[language][doc_id]
            with path.open("rb") as stream:
                stream.seek(start)
                raw = stream.read(end - start)
            lines = raw.split(b"\n")
            if raw.endswith(b"\n"):
                lines.pop()
            return "\n".join(line.rstrip(b"\r").decode("utf-8") for line in lines)
        with self._connect() as connection:
            row = connection.execute("SELECT text,sha256 FROM texts WHERE doc_id=? AND language=?",
                                     (doc_id, language)).fetchone()
        if row is None:
            return ""
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise ValueError("UN original text checksum mismatch")
        return row[0]

    def preload_translations(self, languages, doc_ids):
        # Build byte offsets once per language, retaining no corpus-sized text
        # cache. Full corpus export and scattered references then use bounded seeks.
        if self._legacy is None:
            return
        wanted = set(doc_ids) & self._legacy.docs.keys()
        if not wanted:
            return
        for language in dict.fromkeys(languages):
            path = self._legacy.sixway_dir / f"UNv1.0.6way.{language}"
            with self._legacy_lock:
                self._check_legacy(language, path)
                if language in self._legacy_offsets:
                    continue
                ranges = sorted((d["line_start"], d["line_end"], identifier)
                                for identifier, d in self._legacy.docs.items())
                offsets, number, position = {}, 0, 0
                with path.open("rb") as stream:
                    for start, end, identifier in ranges:
                        if start != number + 1 or end < start:
                            raise ValueError("gapped or invalid legacy UN document ranges")
                        first = position
                        for _ in range(start, end + 1):
                            line = stream.readline()
                            if not line:
                                raise ValueError("truncated legacy UN original")
                            position += len(line)
                        number = end
                        offsets[identifier] = (first, position)
                    if stream.read(1):
                        raise ValueError("legacy UN original exceeds its document inventory")
                self._check_legacy(language, path)
                self._legacy_offsets[language] = offsets

    def _check_legacy(self, language, path):
        if self._file_stat(path) != self._legacy_stats.get(language):
            raise ValueError("legacy UN original changed during use")

    def blocks_for(self, doc_id):
        if doc_id not in self._new_ids:
            return self._legacy.blocks_for(doc_id)
        with self._connect() as connection:
            rows = [json.loads(row[0]) for row in connection.execute(
                "SELECT payload FROM blocks WHERE doc_id=? ORDER BY block_index", (doc_id,))]
        blocks = [ctx.BlockUnit.from_row(row) for row in rows]
        if (len(blocks) != self.docs[doc_id]["n_blocks"] or any(
                b.doc_id != doc_id or b.block_index != i or b.block_id != f"{doc_id}#{i}"
                or b.n_blocks != len(blocks) for i, b in enumerate(blocks))):
            raise ValueError("UN block inventory identity mismatch")
        return blocks

    def _attach_translation(self, unit, language):
        if unit.doc_id not in self._new_ids:
            # Used only for internal anchors; actual references render the raw document.
            raw = self.document_text(unit.doc_id, language)
            start = unit.line_start - self.docs[unit.doc_id]["line_start"]
            end = unit.line_end - self.docs[unit.doc_id]["line_start"] + 1
            unit.texts[language] = "\n".join(raw.split("\n")[start:end])

    def reference_languages_complete(self, doc_ids):
        for doc_id in doc_ids:
            if doc_id in self._new_ids:
                with self._connect() as connection:
                    langs = {row[0] for row in connection.execute(
                        "SELECT language FROM texts WHERE doc_id=? AND length(trim(text))>0", (doc_id,))}
                if not set(LANGUAGES) <= langs:
                    return False
        return True

    @property
    def incomplete(self):
        if not self._status_loaded:
            from clir_bench.domains.legal.un.references_status import _doc_citations
            bad = {}
            for doc_id in sorted(self._new_ids):
                doc = self.docs[doc_id]
                if not doc.get("target_idxs"):
                    continue
                blocks = self.blocks_for(doc_id)
                rows = [{"text": b.texts["en"], "part_id": b.part_id} for b in blocks]
                for idx, citations in _doc_citations(doc_id, doc, rows, self.symbols, self.docs).items():
                    blocked = [c for c in citations if refs.is_blocking(c)]
                    if blocked:
                        reasons = Counter(c.reason for c in blocked)
                        key = blocks[idx].block_id
                        bad[key] = {"block_id": key, "complete": False,
                                    "n_unresolved": len(blocked), "reasons": dict(reasons)}
            self._incomplete = bad
            self._status_loaded = True
        return self._incomplete

    def _validate_target(self, target):
        from clir_bench.domains.legal.un.uprprc import verify_block

        if (target.block_index not in self.docs[target.doc_id].get("target_idxs", [])
                or not target.usable or not target.in_range
                or set(target.texts) != set(LANGUAGES)
                or set(target.source_locations) != set(LANGUAGES)):
            raise ValueError("UN target lacks validated six-language alignment")
        config = self.manifest.get("config", {})
        year = str(self.docs[target.doc_id].get("year", ""))
        if not year.isdigit() or not config.get("from_year", 2015) <= int(year) <= config.get("to_year", 2025):
            raise ValueError("UN target outside configured publication window")
        if not config.get("min_words", 150) <= len(target.texts["en"].split()) <= config.get("max_words", 450):
            raise ValueError("UN target outside configured English word window")
        originals = {}
        for language in LANGUAGES:
            raw = self.document_text(target.doc_id, language)
            originals[language] = raw
            loc = target.source_locations[language]
            start, end = loc.get("start"), loc.get("end")
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw)
                    or loc.get("document_sha256") != hashlib.sha256(raw.encode()).hexdigest()
                    or raw[start:end] != target.texts[language] or not target.texts[language].strip()):
                raise ValueError("UN target source span does not match its original document")
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM blocks WHERE doc_id=? AND block_index=?",
                                     (target.doc_id, target.block_index)).fetchone()
        verify_block(json.loads(row[0]), self.docs[target.doc_id], originals, config)

    def build(self, doc_id, block_index, *, context_chars=ctx.DEFAULT_CONTEXT_CHARS,
              max_references=refs.DEFAULT_MAX_REFERENCES, reference_chars=None, languages=("en",)):
        if doc_id not in self._new_ids:
            raise ValueError("legacy UN documents are reference-only in this dataset")
        blocks = self.blocks_for(doc_id)
        if not 0 <= block_index < len(blocks):
            raise ValueError("UN target block index outside document")
        target = blocks[block_index]
        self._validate_target(target)
        if target.block_id in self.incomplete:
            raise ValueError("UN target has unresolved references")
        payload = super().build(doc_id, block_index, context_chars=context_chars,
                                max_references=max_references, reference_chars=reference_chars,
                                languages=languages)
        if payload.dropped_references:
            raise ValueError("UN target references would be dropped")
        if not self.reference_languages_complete([r.doc_id for r in payload.references]):
            raise ValueError("UN reference lacks all six original language versions")
        for reference in payload.references:
            if any(not self.document_text(reference.doc_id, language).strip() for language in LANGUAGES):
                raise ValueError("UN reference lacks all six original language versions")
        for language in languages:
            if language not in LANGUAGES:
                raise ValueError(f"unsupported UN source language: {language}")
            if any(not r.texts.get(language) for r in payload.references):
                raise ValueError("UN reference missing requested original language")
        # Preserve the default whole-document+references English fit contract.
        total = len(self.document_text(doc_id, "en")) + sum(
            len(self.document_text(r.doc_id, "en")) for r in payload.references)
        if total > min(context_chars, ctx.DEFAULT_CONTEXT_CHARS):
            raise ValueError("UN full document and references exceed context budget")
        contexts = {}
        for language in languages:
            raw = self.document_text(doc_id, language)
            loc = target.source_locations[language]
            contexts[language] = raw[:loc["start"]] + ctx.TARGET_PLACEHOLDER + raw[loc["end"]:]
        payload.document_contexts = contexts
        payload.context_blocks = []
        payload.n_context_dropped = 0
        payload.text = ctx.render_payload(payload.target, [], references=payload.references,
                                          languages=languages, document_contexts=contexts)
        return payload

    def _render_reference(self, citation, limit, languages=("en",)):
        if limit is not None:
            raise ValueError("manifest-backed UN references must use full original documents")
        doc = self.docs[citation.doc_id]
        return ctx.ReferencedDoc(citation.symbol or doc["symbol"], citation.doc_id,
                                 doc.get("title", ""),
                                 {lg: self.document_text(citation.doc_id, lg) for lg in languages},
                                 None)
