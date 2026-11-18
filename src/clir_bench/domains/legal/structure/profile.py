"""Immutable prepared EUR-Lex collections and their generation boundary."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from clir_bench.domains.legal.qac.eurlex_context import ArticleIndex
from clir_bench.domains.legal.structure import ACT_LANGUAGES
from clir_bench.domains.legal.structure.eurlex_resources import (
    FROM_YEAR,
    _canonical,
    discovery_year,
)

FORMAT = "eurlex-resources-v1"
REQUIRED_FILES = frozenset({"articles.jsonl", "internal_edges.jsonl", "external_edges.jsonl",
                            "reference_status.jsonl", "quarantine.jsonl", "target_acts.json"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedProfile:
    path: Path
    manifest: dict
    identity: dict
    target_acts: frozenset[str]
    _stats: dict
    target_articles: frozenset[str] | None = None

    def file(self, name: str) -> Path:
        if name not in self.manifest["files"]:
            raise ValueError(f"file is not part of the frozen EUR-Lex profile: {name}")
        result = self.path.parent / name
        stat = result.stat()
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if current != self._stats[name]:
            raise ValueError(f"EUR-Lex dataset artifact changed during use: {name}")
        return result

    def assert_unchanged(self) -> None:
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.identity["manifest_sha256"]:
            raise ValueError("EUR-Lex dataset manifest changed during use")
        for name in self.manifest["files"]:
            self.file(name)


def load_profile(manifest_path: str | Path) -> PreparedProfile:
    path = Path(manifest_path).expanduser().resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != FORMAT or manifest.get("version") != 1:
        raise ValueError("unsupported EUR-Lex dataset manifest format/version")
    files = manifest.get("files")
    if not isinstance(files, dict) or not REQUIRED_FILES <= files.keys():
        raise ValueError("EUR-Lex dataset manifest lacks required artifacts")
    stats = {}
    for name, expected in files.items():
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (relative is None or relative.is_absolute() or ".." in relative.parts
                or "\\" in name or relative.as_posix() != name or name in (".", "..")
                or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)):
            raise ValueError("invalid EUR-Lex dataset artifact name/checksum")
        artifact = path.parent / name
        if not artifact.resolve().is_relative_to(path.parent):
            raise ValueError("EUR-Lex dataset artifact escapes its bundle")
        before = artifact.stat()
        if sha256_file(artifact) != expected:
            raise ValueError(f"EUR-Lex dataset artifact checksum mismatch: {name}")
        after = artifact.stat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if stamp(before) != stamp(after):
            raise ValueError(f"EUR-Lex dataset artifact changed while reading: {name}")
        stats[name] = stamp(after)
    targets = json.loads((path.parent / "target_acts.json").read_text(encoding="utf-8"))
    to_year = discovery_year(manifest)
    if (not isinstance(targets, list) or any(not _canonical(c, to_year=to_year) for c in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError("invalid newer EUR-Lex target act whitelist")
    article_targets = None
    article_file = "target_articles.json"
    requested = manifest.get("config", {}).get("target_articles")
    if article_file in files:
        entries = json.loads((path.parent / article_file).read_text(encoding="utf-8"))
        years = "|".join(map(str, range(FROM_YEAR, to_year + 1)))
        if (not isinstance(entries, list) or not entries
                or any(not isinstance(eli, str) or not re.fullmatch(
                    r"https?://data\.europa\.eu/eli/(?:reg|dir)(?:_impl|_del)?/"
                    rf"(?:{years})/[1-9]\d*/art_[^/?#\s]+/oj", eli)
                    for eli in entries)
                or entries != sorted(set(entries))):
            raise ValueError("invalid newer EUR-Lex target article whitelist")
        if type(requested) is not int or requested <= 0 or len(entries) != requested:
            raise ValueError("EUR-Lex target article count differs from its declared budget")
        article_targets = frozenset(entries)
    elif requested is not None or (path.parent / article_file).exists():
        raise ValueError("EUR-Lex article budget lacks its hashlisted target article whitelist")
    identity = {"manifest_path": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "files_sha256": files}
    return PreparedProfile(path, manifest, identity, frozenset(targets), stats, article_targets)


class ProfileArticleIndex(ArticleIndex):
    """The existing article format, with strict opt-in source and target checks.

    Baseline acts remain addressable for references; only the explicit newer
    whitelist can supply question targets, including fixed-target replay.
    """

    def __init__(self, manifest_path: str | Path):
        self.profile = load_profile(manifest_path)
        self.identity = self.profile.identity
        self.target_acts = self.profile.target_acts
        self.target_articles = self.profile.target_articles
        self._amending = set()
        self._quarantined = {json.loads(line)["celex_id"] for line in
                             self.profile.file("quarantine.jsonl").read_text().splitlines() if line.strip()}
        self._excluded_acts = self._quarantined
        super().__init__(articles_path=self.profile.file("articles.jsonl"),
                         edges_path=self.profile.file("internal_edges.jsonl"),
                         external_edges_path=self.profile.file("external_edges.jsonl"),
                         status_path=self.profile.file("reference_status.jsonl"),
                         quarantine_path=self.profile.file("quarantine.jsonl"))
        if self.target_acts - self.by_act.keys():
            raise ValueError("EUR-Lex target whitelist names acts with no indexed articles")
        if self.target_articles is not None:
            for eli in self.target_articles:
                unit = self.by_eli.get(eli)
                if unit is None or unit.unit_type != "article" or unit.celex_id not in self.target_acts:
                    raise ValueError("EUR-Lex target article whitelist names an absent or ineligible unit")

    def _load_articles(self, path, annex_whitelist=None):
        # Validate identities with bounded disk staging before the legacy loader's
        # last-record-wins behavior could hide a conflicting source/language row.
        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory(prefix="eurlex-profile-") as directory:
            connection = sqlite3.connect(Path(directory) / "identities.sqlite")
            try:
                connection.execute("CREATE TABLE units (eli TEXT, language TEXT, digest TEXT, "
                                   "PRIMARY KEY(eli,language))")
                connection.execute("CREATE TABLE identities (eli TEXT PRIMARY KEY, signature TEXT)")
                with Path(path).open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row.get("celex_id") in self._quarantined:
                            continue
                        if not row.get("eli_id"):
                            continue
                        signature = json.dumps([row["celex_id"], row["unit_type"], row.get("article_number", "")])
                        old = connection.execute("SELECT signature FROM identities WHERE eli=?",
                                                 (row["eli_id"],)).fetchone()
                        if old and old[0] != signature:
                            raise ValueError(f"conflicting EUR-Lex article identity: {row['eli_id']}")
                        connection.execute("INSERT OR IGNORE INTO identities VALUES (?,?)",
                                           (row["eli_id"], signature))
                        digest = hashlib.sha256(json.dumps(row, ensure_ascii=False,
                                                          sort_keys=True).encode()).hexdigest()
                        key = row["eli_id"], row["language"]
                        old = connection.execute("SELECT digest FROM units WHERE eli=? AND language=?", key).fetchone()
                        if old and old[0] != digest:
                            raise ValueError(f"conflicting EUR-Lex source/language records: {key}")
                        connection.execute("INSERT OR IGNORE INTO units VALUES (?,?,?)", (*key, digest))
                        if row["language"] == "en" and row.get("is_amending"):
                            self._amending.add(row["eli_id"])
                connection.commit()
            finally:
                connection.close()
        super()._load_articles(path, annex_whitelist=annex_whitelist)

    def validate_target(self, target_eli: str) -> None:
        from clir_bench.domains.legal.qac.eurlex_batch import MAX_CHARS, MIN_CHARS

        unit = self.by_eli.get(target_eli)
        if unit is None or unit.unit_type != "article" or unit.celex_id not in self.target_acts:
            raise ValueError("EUR-Lex target is not a newer target-eligible article")
        if self.target_articles is not None and target_eli not in self.target_articles:
            raise ValueError("EUR-Lex target is outside the prepared article budget")
        if unit.celex_id in self._quarantined or target_eli in self._amending:
            raise ValueError("EUR-Lex target is quarantined or amending")
        if not MIN_CHARS <= len(unit.texts.get("en", "")) <= MAX_CHARS:
            raise ValueError("EUR-Lex target outside current article length limits")
        if not self.status.get(target_eli, {}).get("complete"):
            raise ValueError("EUR-Lex target has unresolved references")
        for refs in (self.references, self.external_references, self.annex_references):
            if any(identifier not in self.by_eli for identifier in refs.get(target_eli, ())):
                raise ValueError("EUR-Lex reference endpoint is absent from the source index")

    def build(self, target_eli, *, max_references=6, reference_chars=4000, languages=ACT_LANGUAGES):
        self.profile.assert_unchanged()
        if type(max_references) is not int or max_references < 0:
            raise ValueError("EUR-Lex reference limit must be nonnegative")
        if set(languages) - set(ACT_LANGUAGES):
            raise ValueError("unsupported EUR-Lex profile language")
        self.validate_target(target_eli)
        payload = super().build(target_eli, max_references=max_references,
                                reference_chars=reference_chars, languages=languages)
        for unit in [payload.target, *payload.references, *payload.external_references, *payload.annexes]:
            if any(not unit.texts.get(lg, "").strip() for lg in ACT_LANGUAGES):
                raise ValueError("EUR-Lex supplied article/annex lacks all four original languages")
            if unit.celex_id in self._quarantined:
                raise ValueError("EUR-Lex supplied article/annex belongs to a quarantined act")
        return payload

    def is_target_eligible(self, target_eli: str, *, max_references=6) -> bool:
        try:
            self.build(target_eli, max_references=max_references, languages=())
        except ValueError:
            return False
        return True
