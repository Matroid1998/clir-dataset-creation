"""Where this stage reads and writes.

Standalone source-preparation commands use these repository-relative paths.
The legal ``DomainSpec`` also declares the structured-input directories so the
main question-generation workflow can resolve them through its workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def repo_root() -> Path:
    """Repository root, found by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


ROOT = repo_root()

EURLEX_DIR = ROOT / "data/legal/eurlex"
DOCUMENTS_DIR = EURLEX_DIR / "documents"

STRUCTURE_DIR = EURLEX_DIR / "structure"
CACHE_DIR = STRUCTURE_DIR / "cache"          # raw CELLAR responses, never re-fetched
PARSED_DIR = STRUCTURE_DIR / "parsed"        # parse-fmx JSON, one per (celex, lang)

CANDIDATES_CSV = STRUCTURE_DIR / "candidates.csv"
PILOT_CSV = STRUCTURE_DIR / "pilot.csv"

ARTICLES_JSONL = STRUCTURE_DIR / "articles.jsonl"
INTERNAL_EDGES_JSONL = STRUCTURE_DIR / "internal_edges.jsonl"
EXTERNAL_REFS_JSONL = STRUCTURE_DIR / "external_references.jsonl"
QUARANTINE_JSONL = STRUCTURE_DIR / "quarantine.jsonl"
DROPPED_JSONL = STRUCTURE_DIR / "dropped_references.jsonl"
FETCH_MANIFEST_JSON = STRUCTURE_DIR / "fetch_manifest.json"

# Stage 5 (resolve_external): cross-act references resolved to articles that
# are themselves in the corpus, everything that could not be resolved with its
# reason, and the per-article verdict on whether the citation graph is complete.
EXTERNAL_EDGES_JSONL = STRUCTURE_DIR / "external_edges.jsonl"
UNRESOLVED_EXTERNAL_JSONL = STRUCTURE_DIR / "unresolved_external.jsonl"
REFERENCE_STATUS_JSONL = STRUCTURE_DIR / "reference_status.jsonl"

REPORTS_DIR = ROOT / "reports/eurlex_structure"


@dataclass(frozen=True)
class StructurePaths:
    """Explicit, isolated source-preparation paths; never changes legacy globals."""

    structure_dir: Path
    reports_dir: Path | None = None
    ROOT: Path = field(default_factory=repo_root)

    def __post_init__(self):
        object.__setattr__(self, "structure_dir", Path(self.structure_dir).expanduser().resolve())
        report = self.reports_dir or self.structure_dir / "reports"
        object.__setattr__(self, "reports_dir", Path(report).expanduser().resolve())

    @property
    def STRUCTURE_DIR(self):
        return self.structure_dir

    @property
    def REPORTS_DIR(self):
        return self.reports_dir

    @property
    def CACHE_DIR(self):
        return self.structure_dir / "cache"

    @property
    def PARSED_DIR(self):
        return self.structure_dir / "parsed"

    def __getattr__(self, name):
        files = {
            "CANDIDATES_CSV": "candidates.csv", "PILOT_CSV": "pilot.csv",
            "ARTICLES_JSONL": "articles.jsonl", "INTERNAL_EDGES_JSONL": "internal_edges.jsonl",
            "EXTERNAL_REFS_JSONL": "external_references.jsonl", "QUARANTINE_JSONL": "quarantine.jsonl",
            "DROPPED_JSONL": "dropped_references.jsonl", "FETCH_MANIFEST_JSON": "fetch_manifest.json",
            "EXTERNAL_EDGES_JSONL": "external_edges.jsonl",
            "UNRESOLVED_EXTERNAL_JSONL": "unresolved_external.jsonl",
            "REFERENCE_STATUS_JSONL": "reference_status.jsonl",
        }
        if name in files:
            return self.structure_dir / files[name]
        raise AttributeError(name)

    def ensure_dirs(self):
        for path in (self.STRUCTURE_DIR, self.CACHE_DIR, self.PARSED_DIR, self.REPORTS_DIR):
            path.mkdir(parents=True, exist_ok=True)


def ensure_dirs() -> None:
    for path in (STRUCTURE_DIR, CACHE_DIR, PARSED_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ARTICLES_JSONL",
    "CACHE_DIR",
    "CANDIDATES_CSV",
    "DOCUMENTS_DIR",
    "DROPPED_JSONL",
    "EURLEX_DIR",
    "EXTERNAL_EDGES_JSONL",
    "EXTERNAL_REFS_JSONL",
    "FETCH_MANIFEST_JSON",
    "INTERNAL_EDGES_JSONL",
    "PARSED_DIR",
    "PILOT_CSV",
    "QUARANTINE_JSONL",
    "REFERENCE_STATUS_JSONL",
    "REPORTS_DIR",
    "ROOT",
    "STRUCTURE_DIR",
    "UNRESOLVED_EXTERNAL_JSONL",
    "StructurePaths",
    "ensure_dirs",
]
