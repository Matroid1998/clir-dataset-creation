"""Prepare isolated EUR-Lex resources from a pinned English discovery inventory.

The Hugging Face text is used only to discover and stratify original 2017–2022
regulations/directives. Official CELLAR identifiers and four-language Formex
supply the prepared text and article identities. No model or translation is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import lzma
import os
import platform
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

from clir_bench.domains.legal.structure import (
    ACT_LANGUAGES,
    cellar,
    project,
    references,
    resolve_external,
    segment,
)
from clir_bench.domains.legal.structure.paths import StructurePaths
from clir_bench.domains.legal.structure.select import bin_for, proxy_article_count

REPOSITORY = "joelniklaus/eurlex_resources"
REVISION = "4d7946ef7f0c5ff5e261e384db8015dfe8e417cb"
SOURCE_FILES = ("data/en/regulation.jsonl.xz", "data/en/directive.jsonl.xz")
INVENTORY_FORMAT = "eurlex-resources-inventory-v1"
FORMAT = "eurlex-resources-v1"
VERSION = 1
SEED = 42
FROM_YEAR, TO_YEAR = 2017, 2022
CELLAR_SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"
CELLAR_DISCOVERY = "cellar_sparql"
CELEX_RE = re.compile(r"^3(?P<year>\d{4})(?P<kind>[RL])(?P<number>\d{4,5})$")
REQUIRED_FILES = (
    "articles.jsonl",
    "internal_edges.jsonl",
    "external_edges.jsonl",
    "reference_status.jsonl",
    "quarantine.jsonl",
    "target_acts.json",
)


class ValidationError(ValueError):
    """An inventory or prepared bundle cannot satisfy its declared identity."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rows(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValidationError(f"Invalid JSON at {path.name}:{number}") from error
            if not isinstance(row, dict):
                raise ValidationError(f"Expected an object at {path.name}:{number}")
            yield row


def _manifest_path(value: Path) -> Path:
    value = Path(value).expanduser().resolve()
    return value / "manifest.json" if value.is_dir() else value


def _safe_file(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValidationError("Manifest paths must be relative to the bundle")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValidationError(f"Missing or escaping bundle file: {name}")
    return result


def _hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256_file(_safe_file(root, name)) for name in sorted(set(names))}


def _check_files(path: Path, manifest: dict) -> None:
    if manifest.get("version") != VERSION or not isinstance(manifest.get("files"), dict):
        raise ValidationError("Unsupported or malformed manifest")
    for name, digest in manifest["files"].items():
        if sha256_file(_safe_file(path.parent, name)) != digest:
            raise ValidationError(f"Bundle checksum mismatch: {name}")


def _canonical(celex: Any, *, to_year: int = TO_YEAR) -> re.Match | None:
    if not isinstance(celex, str):
        return None
    match = CELEX_RE.fullmatch(celex)
    if (
        not match
        or not FROM_YEAR <= int(match["year"]) <= to_year
        or not int(match["number"])
        or f"{int(match['number']):04d}" != match["number"]
    ):
        return None
    return match


def discovery_year(manifest: dict) -> int:
    """The pinned HF inventory keeps its historical scope; official snapshots declare theirs."""
    source = manifest.get("source", {})
    if source.get("mode") != CELLAR_DISCOVERY:
        return TO_YEAR
    year = manifest.get("config", {}).get("to_year")
    if (source.get("repository") != CELLAR_SPARQL or type(year) is not int
            or not TO_YEAR <= year <= 2099
            or not re.fullmatch(r"[0-9a-f]{64}", source.get("revision", ""))):
        raise ValidationError("Invalid official EUR-Lex discovery scope")
    return year


def _inventory_row(row: dict) -> dict:
    celex = row.get("celex")
    match = _canonical(celex)
    if match is None:
        raise ValidationError("not-original-2017-2022-regulation-or-directive")
    if row.get("language") and str(row["language"]).casefold() not in ("en", "eng", "english"):
        raise ValidationError("not-english")
    value = row.get("date")
    try:
        publication = (
            value.date()
            if isinstance(value, datetime)
            else (value if isinstance(value, date) else date.fromisoformat(str(value)[:10]))
        )
    except ValueError as error:
        raise ValidationError("invalid-source-date") from error
    if not FROM_YEAR <= publication.year <= TO_YEAR:
        raise ValidationError("source-date-outside-2017-2022")
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("empty-english-text")
    title = row.get("title")
    if title is not None and not isinstance(title, str):
        raise ValidationError("invalid-title")
    n_articles = proxy_article_count(text)
    return {
        "celex_id": celex,
        "year": int(match["year"]),
        "doc_type": "regulation" if match["kind"] == "R" else "directive",
        "date": publication.isoformat(),
        "title": title or "",
        "proxy_articles": n_articles,
        "size_bin": bin_for(n_articles) or "unknown",
        "chars_en": len(text),
        "text_sha256": _text_hash(text),
        "eligible": True,
    }


def stratified_sample(rows: Iterable[dict], n: int = 30, seed: int = SEED) -> list[dict]:
    """Stable round-robin across year/type and, within them, article-count bins."""
    if type(n) is not int or n < 0:
        raise ValueError("Pilot size must be a nonnegative integer")
    strata: dict[tuple[int, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("eligible", True):
            strata[(int(row["year"]), row["doc_type"])][row["size_bin"]].append(row)
    order = []
    for base in sorted(strata):
        bins = strata[base]
        for bucket in bins.values():
            bucket.sort(key=lambda row: (_text_hash(f"{seed}:{row['celex_id']}"), row["celex_id"]))
        flattened = []
        for depth in range(max(map(len, bins.values()), default=0)):
            for name in sorted(bins):
                if depth < len(bins[name]):
                    flattened.append(bins[name][depth])
        order.append(flattened)
    result = []
    for depth in range(max(map(len, order), default=0)):
        for bucket in order:
            if len(result) >= n:
                return result
            if depth < len(bucket):
                result.append(bucket[depth])
    return result


def _inventory_build(
    records: Iterable[dict],
    root: Path,
    *,
    revision: str,
    pilot_size: int,
    seed: int,
    sources: list[dict],
    mode: str,
) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Inventory revision must be an immutable 40-character Hugging Face commit")
    counts: Counter = Counter()
    rejects: Counter = Counter()
    database = sqlite3.connect(root / "inventory-staging.sqlite")
    try:
        database.execute(
            "CREATE TABLE records (celex TEXT, fingerprint TEXT, payload TEXT, occurrences INTEGER, "
            "PRIMARY KEY(celex,fingerprint))"
        )
        with (root / "rejected.jsonl").open("w", encoding="utf-8") as rejected:
            for row in records:
                counts["source_records"] += 1
                try:
                    metadata = _inventory_row(row)
                except ValidationError as error:
                    rejects[str(error)] += 1
                    rejected.write(_json({"celex": row.get("celex"), "reason": str(error)}) + "\n")
                    continue
                encoded = _json(metadata)
                fingerprint = _text_hash(encoded)
                database.execute(
                    "INSERT INTO records VALUES (?,?,?,1) ON CONFLICT(celex,fingerprint) "
                    "DO UPDATE SET occurrences=occurrences+1",
                    (metadata["celex_id"], fingerprint, encoded),
                )
                counts["accepted_source_records"] += 1
                if counts["source_records"] % 1000 == 0:
                    database.commit()
        database.commit()
        candidates = []
        by_year: Counter = Counter()
        by_type: Counter = Counter()
        with (
            (root / "inventory.jsonl").open("w", encoding="utf-8") as out,
            (root / "conflicts.jsonl").open("w", encoding="utf-8") as conflicts,
        ):
            for celex, n in database.execute(
                "SELECT celex,COUNT(*) FROM records GROUP BY celex ORDER BY celex"
            ):
                variants = list(
                    database.execute(
                        "SELECT fingerprint,payload,occurrences FROM records WHERE celex=? "
                        "ORDER BY fingerprint",
                        (celex,),
                    )
                )
                counts["duplicate_records"] += sum(row[2] - 1 for row in variants)
                if n > 1:
                    counts["conflicting_acts"] += 1
                    conflicts.write(
                        _json(
                            {
                                "celex_id": celex,
                                "reason": "conflicting-source-records",
                                "variants": [
                                    {
                                        "fingerprint": fingerprint,
                                        "record": json.loads(payload),
                                        "occurrences": occurrences,
                                    }
                                    for fingerprint, payload, occurrences in variants
                                ],
                            }
                        )
                        + "\n"
                    )
                    continue
                record = json.loads(variants[0][1])
                out.write(_json(record) + "\n")
                candidates.append(record)
                by_year[str(record["year"])] += 1
                by_type[record["doc_type"]] += 1
        counts["canonical_acts"] = len(candidates)
        selected = stratified_sample(candidates, pilot_size, seed)
        counts["pilot_acts"] = len(selected)
        _write_json(root / "pilot.json", selected)
        for name in (
            "source_records",
            "accepted_source_records",
            "duplicate_records",
            "conflicting_acts",
        ):
            counts.setdefault(name, 0)
        report = {
            "counts": dict(sorted(counts.items())),
            "rejected": dict(sorted(rejects.items())),
            "by_year": dict(sorted(by_year.items())),
            "by_type": dict(sorted(by_type.items())),
            "source_date_range": {
                "first": min((row["date"] for row in candidates), default=None),
                "last": max((row["date"] for row in candidates), default=None),
            },
            "pilot_celex": [row["celex_id"] for row in selected],
        }
        _write_json(root / "inventory_report.json", report)
    finally:
        database.close()
        (root / "inventory-staging.sqlite").unlink(missing_ok=True)
    names = [
        "inventory.jsonl",
        "rejected.jsonl",
        "conflicts.jsonl",
        "pilot.json",
        "inventory_report.json",
    ]
    names += [source["path"] for source in sources if source.get("path")]
    manifest = {
        "format": INVENTORY_FORMAT,
        "version": VERSION,
        "source": {
            "repository": REPOSITORY,
            "revision": revision,
            "language": "en",
            "date_field": "date",
            "date_semantics": "upstream dataset date; not verified as OJ publication",
            "filenames": list(SOURCE_FILES),
            "inputs": sources,
            "mode": mode,
        },
        "config": {
            "from_year": FROM_YEAR,
            "to_year": TO_YEAR,
            "seed": seed,
            "pilot_size": pilot_size,
            "original_sector": "3",
            "types": ["R", "L"],
        },
        "counts": report["counts"],
        "files": _hashes(root, names),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _fresh_work(out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir).expanduser().absolute()
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.building-", dir=out_dir.parent))
    return out_dir, temporary


def inventory_records(
    records: Iterable[dict],
    out_dir: Path,
    *,
    revision: str = REVISION,
    pilot_size: int = 30,
    seed: int = SEED,
) -> dict:
    """Prepare an offline discovery fixture; text hashes preserve exact raw identity."""
    out_dir, temporary = _fresh_work(out_dir)
    try:
        manifest = _inventory_build(
            records,
            temporary,
            revision=revision,
            pilot_size=pilot_size,
            seed=seed,
            sources=[],
            mode="provided_records",
        )
        audit(temporary / "manifest.json")
        if out_dir.exists():
            raise FileExistsError(f"Output already exists: {out_dir}")
        temporary.rename(out_dir)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _xz_records(files: Iterable[Path]) -> Iterator[dict]:
    for path in files:
        with lzma.open(path, "rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValidationError(f"Expected JSON object in {path.name}:{number}")
                    yield row


def inventory(
    out_dir: Path, *, revision: str = REVISION, pilot_size: int = 30, seed: int = SEED
) -> dict:
    """Download only English regulation/directive XZ files at one resolved commit."""
    from huggingface_hub import HfApi, hf_hub_download

    out_dir, temporary = _fresh_work(out_dir)
    try:
        pinned = (
            revision
            if re.fullmatch(r"[0-9a-f]{40}", revision)
            else HfApi().dataset_info(REPOSITORY, revision=revision).sha
        )
        if not pinned or not re.fullmatch(r"[0-9a-f]{40}", pinned):
            raise ValidationError("Hugging Face did not return an immutable revision")
        downloaded = [
            Path(
                hf_hub_download(
                    repo_id=REPOSITORY,
                    repo_type="dataset",
                    revision=pinned,
                    filename=name,
                    local_dir=temporary / "source",
                )
            )
            for name in SOURCE_FILES
        ]
        sources = [
            {
                "path": str(path.relative_to(temporary)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in downloaded
        ]
        manifest = _inventory_build(
            _xz_records(downloaded),
            temporary,
            revision=pinned,
            pilot_size=pilot_size,
            seed=seed,
            sources=sources,
            mode="downloaded",
        )
        audit(temporary / "manifest.json")
        if out_dir.exists():
            raise FileExistsError(f"Output already exists: {out_dir}")
        temporary.rename(out_dir)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _baseline_info(directory: Path) -> tuple[dict, set[str], list[dict]]:
    articles = directory / "articles.jsonl"
    if not articles.is_file():
        raise FileNotFoundError(f"Missing baseline articles: {articles}")
    acts = {row["celex_id"] for row in _rows(articles)}
    quarantine = list(_rows(directory / "quarantine.jsonl"))
    return (
        {
            "articles_sha256": sha256_file(articles),
            "articles_bytes": articles.stat().st_size,
            "quarantine_sha256": sha256_file(directory / "quarantine.jsonl")
            if (directory / "quarantine.jsonl").is_file()
            else None,
            "acts": len(acts),
        },
        acts,
        quarantine,
    )


def _callable_identity(function: Any) -> str:
    if function is None:
        return "default"
    try:
        return _text_hash(inspect.getsource(function))
    except (OSError, TypeError):
        return f"{getattr(function, '__module__', '')}:{getattr(function, '__qualname__', type(function).__name__)}"


def _csv(path: Path, rows: list[dict]) -> None:
    keys = ["celex_id", "doc_type", "year", "proxy_articles", "size_bin", "chars_en"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _new_validation(path: Path, selected: set[str], *, ignore_other: bool = False,
                    to_year: int = TO_YEAR) -> list[dict]:
    per_act: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    bad: dict[str, set[str]] = defaultdict(set)
    for row in _rows(path):
        celex, language = row.get("celex_id"), row.get("language")
        if celex not in selected:
            if ignore_other:
                continue
            raise ValidationError("New-stage output contains an unselected act")
        if language not in ACT_LANGUAGES:
            bad[celex].add("unexpected-language")
        if row.get("unit_type") != "article":
            continue
        number = row.get("article_number")
        if not number or number in per_act[celex][language]:
            bad[celex].add("duplicate-or-missing-article-number")
        per_act[celex][language][number] = row
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            bad[celex].add("empty-article-text")
        elif row.get("char_count") != len(row["text"]) or row.get("token_count") != len(
            row["text"].split()
        ):
            bad[celex].add("article-text-count-mismatch")
        act_eli = row.get("act_eli", "")
        match = re.fullmatch(r"https?://data\.europa\.eu/eli/([a-z_]+)/(\d{4})/(\d+)/oj", act_eli)
        canonical = _canonical(celex, to_year=to_year)
        if (
            not match
            or not canonical
            or int(match[2]) != int(canonical["year"])
            or int(match[3]) != int(canonical["number"])
            or not match[1].startswith("reg" if canonical["kind"] == "R" else "dir")
            or row.get("eli_id") != cellar.article_eli(act_eli, number)
        ):
            bad[celex].add("article-eli-identity-mismatch")
    for celex in sorted(selected):
        versions = per_act.get(celex, {})
        if not all(versions.get(language) for language in ACT_LANGUAGES):
            bad[celex].add("missing-language-or-no-articles")
        elif any(set(versions[language]) != set(versions["en"]) for language in ACT_LANGUAGES):
            bad[celex].add("article-set-mismatch")
        elif any(
            versions[language][number].get("eli_id") != versions["en"][number].get("eli_id")
            for language in ACT_LANGUAGES
            for number in versions["en"]
        ):
            bad[celex].add("article-eli-language-mismatch")
    return [
        {"celex_id": celex, "reason": reason, "stage": "eurlex-resources-preparation"}
        for celex in sorted(bad)
        for reason in sorted(bad[celex])
    ]


def _copy_union(baseline: Path, additions: Path, target: Path) -> None:
    shutil.copyfile(baseline, target)
    with target.open("ab") as out:
        if baseline.stat().st_size:
            with baseline.open("rb") as source:
                source.seek(-1, os.SEEK_END)
                if source.read(1) != b"\n":
                    out.write(b"\n")
        if additions.exists():
            with additions.open("rb") as source:
                shutil.copyfileobj(source, out)


def find_unit_conflicts(articles_path: Path, quarantined: Iterable[str] = ()) -> list[dict]:
    """Find ambiguous logical unit identities without loading the corpus in RAM.

    Repeated identical rows are harmless. An ELI naming different logical units,
    or different records for one ELI/language, quarantines every involved act.
    Previously quarantined acts do not participate in the logical source join.
    """
    ignored = set(quarantined)
    details: dict[str, list[dict]] = defaultdict(list)
    totals: Counter = Counter()
    with tempfile.TemporaryDirectory(prefix="eurlex-unit-identities-") as directory:
        connection = sqlite3.connect(Path(directory) / "identities.sqlite")
        try:
            connection.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA temp_store=FILE;
                PRAGMA cache_size=-8192;
                CREATE TABLE identities (eli TEXT, signature TEXT, celex TEXT,
                    PRIMARY KEY(eli,signature,celex));
                CREATE TABLE versions (eli TEXT, language TEXT, digest TEXT, celex TEXT,
                    PRIMARY KEY(eli,language,digest,celex));
            """)
            for number, row in enumerate(_rows(Path(articles_path)), 1):
                celex, eli = row.get("celex_id"), row.get("eli_id")
                if celex in ignored or not eli:
                    continue
                signature = _json(
                    [
                        celex,
                        row.get("unit_type"),
                        row.get("article_number", ""),
                        row.get("annex_id", ""),
                        row.get("recital_number", ""),
                    ]
                )
                connection.execute(
                    "INSERT OR IGNORE INTO identities VALUES (?,?,?)", (eli, signature, celex)
                )
                connection.execute(
                    "INSERT OR IGNORE INTO versions VALUES (?,?,?,?)",
                    (eli, row.get("language", ""), _text_hash(_json(row)), celex),
                )
                if number % 10000 == 0:
                    connection.commit()
            connection.commit()
            for (eli,) in connection.execute(
                "SELECT eli FROM identities GROUP BY eli "
                "HAVING COUNT(DISTINCT signature)>1 ORDER BY eli"
            ):
                for (celex,) in connection.execute(
                    "SELECT DISTINCT celex FROM identities WHERE eli=? ORDER BY celex", (eli,)
                ):
                    totals[celex] += 1
                    if len(details[celex]) < 40:
                        details[celex].append({"eli_id": eli, "kind": "unit-identity-conflict"})
            for eli, language in connection.execute(
                "SELECT eli,language FROM versions GROUP BY eli,language "
                "HAVING COUNT(DISTINCT digest)>1 ORDER BY eli,language"
            ):
                for (celex,) in connection.execute(
                    "SELECT DISTINCT celex FROM versions WHERE eli=? AND language=? ORDER BY celex",
                    (eli, language),
                ):
                    totals[celex] += 1
                    if len(details[celex]) < 40:
                        details[celex].append(
                            {
                                "eli_id": eli,
                                "language": language,
                                "kind": "source-language-record-conflict",
                            }
                        )
        finally:
            connection.close()
    return [
        {
            "celex_id": celex,
            "reason": "conflicting-unit-records",
            "stage": "eurlex-resources-union-identity-audit",
            "n_conflicts": totals[celex],
            "conflicts": sorted(details[celex], key=_json),
        }
        for celex in sorted(totals)
    ]


def _language_incomplete(profile: StructurePaths) -> int:
    """Measure translated endpoint gaps without changing legacy English status."""
    languages: dict[str, set[str]] = defaultdict(set)
    for row in _rows(profile.ARTICLES_JSONL):
        if isinstance(row.get("text"), str) and row["text"].strip():
            languages[row.get("eli_id", "")].add(row.get("language"))
    affected: Counter = Counter()
    for path in (profile.INTERNAL_EDGES_JSONL, profile.EXTERNAL_EDGES_JSONL):
        for edge in _rows(path):
            if edge.get("source_unit_type") != "article":
                continue
            if any(
                not set(ACT_LANGUAGES) <= values
                for values in (
                    set(edge.get("available_languages", [])),
                    languages[edge["source_article_id"]],
                    languages[edge["target_article_id"]],
                )
            ):
                affected[edge["source_article_id"]] += 1
    return len(affected)


def _official_provenance(profile: StructurePaths, selected: Iterable[str]) -> list[dict]:
    result = []
    for celex in sorted(selected):
        files = []
        for path in sorted((profile.CACHE_DIR / celex[1:5]).glob(f"{celex}.*")):
            if path.suffix in (".xml", ".zip"):
                digest = sha256_file(path)
                metadata_path = path.with_suffix(path.suffix + ".meta.json")
                metadata = (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.exists()
                    else None
                )
                if metadata is not None and metadata.get("sha256") != digest:
                    raise ValidationError(f"Official raw metadata checksum mismatch: {path.name}")
                language = path.name[len(celex) + 1 : -4] if path.suffix == ".zip" else None
                files.append(
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": digest,
                        "request": {
                            "url": cellar.CELLAR.format(celex=celex).replace("http:", "https:", 1),
                            "accept": "application/zip;mtype=fmx4"
                            if language
                            else "application/xml;notice=identifiers",
                            "accept_language": cellar.LANG_CODE[language] if language else None,
                            "provenance": "profile request configuration",
                        },
                        "retrieved_at": metadata.get("retrieved_at")
                        if metadata is not None
                        else None,
                        "cache_metadata": metadata,
                        "cache_metadata_sha256": sha256_file(metadata_path)
                        if metadata is not None
                        else None,
                    }
                )
        result.append({"celex_id": celex, "files": files})
    return result


def _parser_provenance(parser=None) -> dict:
    if parser is None or parser is segment.run_parse_fmx:
        return {
            "kind": "parse_fmx",
            "entrypoint_name": segment.parse_fmx_path().name,
            **segment.parser_provenance(),
        }
    code = getattr(parser, "__code__", None)
    source = Path(code.co_filename) if code is not None else None
    return {
        "kind": "injected_python_parser",
        "python_version": platform.python_version(),
        "callable": f"{getattr(parser, '__module__', '')}:{getattr(parser, '__qualname__', type(parser).__name__)}",
        "callable_sha256": _callable_identity(parser),
        "entrypoint": {"name": source.name, "sha256": sha256_file(source)}
        if source is not None and source.is_file()
        else None,
    }


def _audit_official_provenance(root: Path, *, cache_dir: Path | None = None,
                               to_year: int = TO_YEAR) -> None:
    """Verify recorded retrieval metadata, and its raw bytes when cache remains."""
    if cache_dir is None:
        candidates = (
            root.parent / "new" / "cache",
            root.with_name(f".{root.name}.work") / "new" / "cache",
        )
        cache_dir = next((path for path in candidates if path.is_dir()), None)
    elif not Path(cache_dir).is_dir():
        raise ValidationError("Explicit official-source cache directory does not exist")
    for act in json.loads((root / "official_sources.json").read_text(encoding="utf-8")):
        celex = act.get("celex_id")
        if _canonical(celex, to_year=to_year) is None:
            raise ValidationError("Invalid official-source act identity")
        for entry in act.get("files", []):
            name, digest = entry.get("name"), entry.get("sha256")
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.startswith(celex + ".")
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValidationError("Invalid official-source raw file identity")
            language = name[len(celex) + 1 : -4] if name.endswith(".zip") else None
            if (language is not None and language not in ACT_LANGUAGES) or (
                language is None and name != f"{celex}.identifiers.xml"
            ):
                raise ValidationError("Invalid official-source language")
            request = entry.get("request", {})
            if request != {
                "url": cellar.CELLAR.format(celex=celex).replace("http:", "https:", 1),
                "accept": "application/zip;mtype=fmx4"
                if language
                else "application/xml;notice=identifiers",
                "accept_language": cellar.LANG_CODE[language] if language else None,
                "provenance": "profile request configuration",
            }:
                raise ValidationError("Official-source request identity mismatch")
            metadata = entry.get("cache_metadata")
            if metadata is not None and (
                metadata.get("sha256") != digest
                or entry.get("retrieved_at") != metadata.get("retrieved_at")
            ):
                raise ValidationError("Official-source retrieval metadata checksum mismatch")
            timestamp = entry.get("retrieved_at")
            if timestamp is not None:
                try:
                    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except (ValueError, AttributeError) as error:
                    raise ValidationError("Invalid official-source retrieval timestamp") from error
            if cache_dir is not None:
                path = Path(cache_dir) / celex[1:5] / name
                if (
                    not path.is_file()
                    or sha256_file(path) != digest
                    or path.stat().st_size != entry.get("bytes")
                ):
                    raise ValidationError(f"Official raw cache checksum mismatch: {name}")
                if metadata is not None:
                    metadata_path = path.with_suffix(path.suffix + ".meta.json")
                    if (
                        not metadata_path.is_file()
                        or sha256_file(metadata_path) != entry.get("cache_metadata_sha256")
                        or json.loads(metadata_path.read_text(encoding="utf-8")) != metadata
                    ):
                        raise ValidationError(f"Official raw cache metadata changed: {name}")


def article_budget_order(rows: Iterable[dict], seed: int = SEED) -> list[dict]:
    """Prioritize article-rich acts, initially covering every year/type stratum."""
    def rank(row):
        return (-int(row["proxy_articles"]), int(row["chars_en"]),
                _text_hash(f"{seed}:{row['celex_id']}"), row["celex_id"])

    ranked = sorted((row for row in rows if row.get("eligible", True)), key=rank)
    first = {}
    for row in ranked:
        first.setdefault((int(row["year"]), row["doc_type"]), row)
    coverage = [first[key] for key in sorted(first)]
    covered = {row["celex_id"] for row in coverage}
    return coverage + [row for row in ranked if row["celex_id"] not in covered]


def _seed_official_cache(source: Path, destination: Path, selected: set[str]) -> None:
    """Copy only identity-validated, checksum-bound original responses."""
    source = source.expanduser().resolve()
    if (source / "cache").is_dir():
        source = source / "cache"
    if not source.is_dir():
        raise ValueError(f"Official cache source does not exist: {source}")
    if source == destination.resolve():
        return
    for celex in sorted(selected):
        for suffix in ["identifiers.xml", *(f"{language}.zip" for language in ACT_LANGUAGES)]:
            raw = source / celex[1:5] / f"{celex}.{suffix}"
            metadata = raw.with_suffix(raw.suffix + ".meta.json")
            target = destination / celex[1:5] / raw.name
            if target.exists() or not raw.is_file() or not metadata.is_file():
                continue
            payload = raw.read_bytes()
            recorded = json.loads(metadata.read_text(encoding="utf-8"))
            if recorded.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise ValidationError(f"Seed cache raw checksum mismatch: {raw.name}")
            try:
                if suffix == "identifiers.xml":
                    cellar._checked_eli(payload, celex)
                else:
                    cellar.validate_formex(payload, celex, suffix[:-4])
            except cellar.FetchError:
                # Unusable original responses never become successful cache slots.
                continue
            cellar._atomic_write(target, payload)
            cellar._atomic_json(target.with_suffix(target.suffix + ".meta.json"), recorded)


def prepare(
    inventory_path: Path,
    baseline_structure: Path,
    out_dir: Path,
    *,
    pilot_size: int = 30,
    seed: int = SEED,
    workers: int = 4,
    work_dir: Path | None = None,
    target_articles: int | None = None,
    batch_size: int = 50,
    exclude_profiles: Iterable[Path] = (),
    cache_from: Iterable[Path] = (),
    transport=None,
    parser=None,
) -> dict:
    """Prepare an act pilot or stop once a distinct eligible-article budget is met."""
    if target_articles is None:
        if tuple(exclude_profiles):
            raise ValueError("Excluded profiles require an eligible article budget")
        return _prepare_acts(
            inventory_path, baseline_structure, out_dir, pilot_size=pilot_size,
            seed=seed, workers=workers, work_dir=work_dir, transport=transport,
            parser=parser, cache_from=cache_from,
        )
    if type(target_articles) is not int or target_articles < 1:
        raise ValueError("Target article count must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Article preparation batch size must be a positive integer")
    from clir_bench.domains.legal.structure.profile import ProfileArticleIndex, load_profile

    inventory_path = _manifest_path(inventory_path)
    audit(inventory_path)
    baseline_structure = Path(baseline_structure).expanduser().resolve()
    baseline, baseline_acts, _ = _baseline_info(baseline_structure)
    out_dir = Path(out_dir).expanduser().absolute()
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}")
    excluded_acts = set()
    exclusions = []
    for path in exclude_profiles:
        prior = load_profile(_manifest_path(Path(path)))
        excluded_acts.update(prior.target_acts)
        exclusions.append(prior.identity)
    exclusions.sort(key=lambda entry: entry["manifest_path"])
    rows = [row for row in _rows(inventory_path.parent / "inventory.jsonl")
            if row["celex_id"] not in baseline_acts | excluded_acts]
    ordered = article_budget_order(rows, seed)
    if not ordered:
        raise ValidationError("No new acts remain after excluding prior prepared profiles")
    work = (Path(work_dir).expanduser().absolute() if work_dir
            else out_dir.with_name(f".{out_dir.name}.work"))
    if (work == out_dir or work.is_relative_to(baseline_structure)
            or work.is_relative_to(inventory_path.parent)
            or out_dir.is_relative_to(baseline_structure)
            or out_dir.is_relative_to(inventory_path.parent)):
        raise ValueError("Preparation work must be separate from the output and baseline")
    identity = {
        "mode": "eligible-article-budget-v1",
        "inventory_sha256": sha256_file(inventory_path),
        "baseline": baseline,
        "candidate_order": [row["celex_id"] for row in ordered],
        "target_articles": target_articles,
        "batch_size": batch_size,
        "seed": seed,
        "excluded_profiles": exclusions,
        "transport": _callable_identity(transport),
        "parser": _parser_provenance(parser),
        "implementation": {
            "preparation": sha256_file(Path(__file__)),
            "segmentation": sha256_file(Path(segment.__file__)),
            "retrieval": sha256_file(Path(cellar.__file__)),
            "references": sha256_file(Path(references.__file__)),
            "projection": sha256_file(Path(project.__file__)),
            "resolution": sha256_file(Path(resolve_external.__file__)),
            "qa_eligibility": _text_hash(inspect.getsource(ProfileArticleIndex)),
        },
    }
    state = work / "work_manifest.json"
    if work.exists():
        if not state.is_file() or json.loads(state.read_text(encoding="utf-8")) != identity:
            raise ValidationError("Resumable workspace belongs to different inputs/configuration")
    else:
        work.mkdir(parents=True)
        _write_json(state, identity)
    progress_path = work / "article_budget_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"batches": []}
    batches = progress["batches"]
    start = batches[-1]["processed_acts"] if batches else min(batch_size, len(ordered))
    started = time.monotonic()
    previous_attempt = None
    for end in range(start, len(ordered) + batch_size, batch_size):
        end = min(end, len(ordered))
        attempt = work / f"attempt-{end:06d}"
        print(f"[article-budget] Preparing {end:,} acts; target {target_articles:,} eligible articles", flush=True)
        if attempt.exists():
            audit(attempt / "manifest.json", cache_dir=work / "new" / "cache")
            manifest = json.loads((attempt / "manifest.json").read_text())
        else:
            manifest = _prepare_acts(
                inventory_path, baseline_structure, attempt, pilot_size=end, seed=seed,
                workers=workers, work_dir=work, transport=transport, parser=parser,
                cache_from=cache_from, _selected_rows=ordered[:end], _resume_identity=identity,
            )
        index = ProfileArticleIndex(attempt / "manifest.json")
        eligible = sorted(
            identifier for identifier, unit in index.by_eli.items()
            if unit.celex_id in index.target_acts and unit.unit_type == "article"
            and index.is_target_eligible(identifier)
        )
        del index
        count = len(eligible)
        record = {"processed_acts": end, "eligible_articles": count,
                  "elapsed_seconds": round(time.monotonic() - started, 3)}
        if not batches or batches[-1]["processed_acts"] != end:
            batches.append(record)
        else:
            batches[-1] = record
        _write_json(progress_path, {"batches": batches})
        print(f"[article-budget] {count:,}/{target_articles:,} eligible articles after {end:,} acts", flush=True)
        if count >= target_articles:
            chosen = sorted(sorted(eligible, key=lambda identifier: (
                _text_hash(f"{seed}:{identifier}"), identifier))[:target_articles])
            _write_json(attempt / "target_articles.json", chosen)
            report_path = attempt / "preparation_report.json"
            report = json.loads(report_path.read_text())
            report.update({"requested_target_articles": target_articles,
                           "eligible_articles_before_cap": count,
                           "batches": batches,
                           "selection_strategy": "article-rich with initial year/type coverage",
                           "excluded_profiles": exclusions})
            report["counts"].update({"target_articles": target_articles,
                                     "eligible_articles_before_cap": count})
            _write_json(report_path, report)
            manifest["config"].update({"target_articles": target_articles,
                                       "batch_size": batch_size,
                                       "selection_strategy": "article-rich with initial year/type coverage"})
            manifest["counts"].update({"target_articles": target_articles,
                                      "eligible_articles_before_cap": count})
            manifest["source"]["excluded_profiles"] = exclusions
            for name in ("target_articles.json", "preparation_report.json"):
                manifest["files"][name] = sha256_file(attempt / name)
            _write_json(attempt / "manifest.json", manifest)
            audit(attempt / "manifest.json", cache_dir=work / "new" / "cache")
            if (sha256_file(inventory_path) != identity["inventory_sha256"]
                    or _baseline_info(baseline_structure)[0] != baseline):
                raise ValidationError("Inventory or baseline changed during preparation")
            if out_dir.exists():
                raise FileExistsError(f"Output already exists: {out_dir}")
            attempt.rename(out_dir)
            if previous_attempt and previous_attempt.exists():
                shutil.rmtree(previous_attempt)
            return manifest
        if previous_attempt and previous_attempt != attempt and previous_attempt.exists():
            shutil.rmtree(previous_attempt)
        previous_attempt = attempt
        if end == len(ordered):
            break
    raise ValidationError(
        f"Only {count:,} eligible articles found; requested {target_articles:,}. "
        f"No partial collection was published; resumable work is retained at {work}."
    )


def _prepare_acts(
    inventory_path: Path,
    baseline_structure: Path,
    out_dir: Path,
    *,
    pilot_size: int = 30,
    seed: int = SEED,
    workers: int = 4,
    work_dir: Path | None = None,
    transport=None,
    parser=None,
    cache_from: Iterable[Path] = (),
    _selected_rows: list[dict] | None = None,
    _resume_identity: dict | None = None,
) -> dict:
    """Build a fresh union bundle, resuming official-source work after interruption.

    The default resumable workspace is ``.<output-name>.work`` beside the output.
    It is bound to exact inventory/baseline/configuration hashes and never mutates
    the baseline or repository-default structure files.
    """
    out_dir = Path(out_dir).expanduser().absolute()
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}")
    inventory_path = _manifest_path(inventory_path)
    inventory_manifest = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory_manifest.get("format") != INVENTORY_FORMAT:
        raise ValidationError("Expected a EUR-Lex resources inventory manifest")
    audit(inventory_path)
    to_year = discovery_year(inventory_manifest)
    baseline_structure = Path(baseline_structure).expanduser().resolve()
    baseline, baseline_acts, baseline_quarantine = _baseline_info(baseline_structure)
    candidates = [
        row
        for row in _rows(inventory_path.parent / "inventory.jsonl")
        if row["celex_id"] not in baseline_acts
    ]
    selected = (_selected_rows if _selected_rows is not None
                else stratified_sample(candidates, pilot_size, seed))
    if any(row not in candidates for row in selected):
        raise ValidationError("Selected preparation act is outside the input inventory")
    chosen = {row["celex_id"] for row in selected}
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    work = (
        Path(work_dir).expanduser().absolute()
        if work_dir
        else out_dir.with_name(f".{out_dir.name}.work")
    )
    if (
        work == out_dir
        or work.is_relative_to(baseline_structure)
        or work.is_relative_to(inventory_path.parent)
        or out_dir.is_relative_to(baseline_structure)
        or out_dir.is_relative_to(inventory_path.parent)
    ):
        raise ValueError("Preparation work must be separate from the output and baseline")
    identity = _resume_identity or {
        "inventory_sha256": sha256_file(inventory_path),
        "baseline": baseline,
        "selected": sorted(chosen),
        "pilot_size": pilot_size,
        "seed": seed,
        "transport": _callable_identity(transport),
        "parser": _callable_identity(parser),
    }
    state = work / "work_manifest.json"
    if work.exists():
        if not state.is_file() or json.loads(state.read_text(encoding="utf-8")) != identity:
            raise ValidationError("Resumable workspace belongs to different inputs/configuration")
    else:
        work.mkdir(parents=True)
        _write_json(state, identity)
    new_profile = StructurePaths(work / "new")
    new_profile.ensure_dirs()
    for source_cache in cache_from:
        _seed_official_cache(Path(source_cache), new_profile.CACHE_DIR, chosen)
    _csv(new_profile.CANDIDATES_CSV, selected)
    _csv(new_profile.PILOT_CSV, selected)
    if chosen:
        cellar.fetch_all(sorted(chosen), workers=workers, profile=new_profile, transport=transport)
        segment.segment(
            sorted(chosen),
            workers=workers,
            resume=True,
            profile=new_profile,
            transport=transport,
            parser=parser,
        )
    else:
        new_profile.ARTICLES_JSONL.touch()
    strict_quarantine = _new_validation(new_profile.ARTICLES_JSONL, chosen, to_year=to_year)
    bundle = work / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir()
    profile = StructurePaths(bundle)
    profile.ensure_dirs()
    if (baseline_structure / "quarantine.jsonl").exists():
        shutil.copyfile(
            baseline_structure / "quarantine.jsonl", bundle / "baseline_quarantine.jsonl"
        )
    else:
        (bundle / "baseline_quarantine.jsonl").write_text("", encoding="utf-8")
    _copy_union(
        baseline_structure / "articles.jsonl", new_profile.ARTICLES_JSONL, profile.ARTICLES_JSONL
    )
    inherited_quarantine = baseline_quarantine + strict_quarantine
    known_bad = {row.get("celex_id") or row.get("celex") for row in inherited_quarantine}
    identity_quarantine = find_unit_conflicts(profile.ARTICLES_JSONL, known_bad)
    with (bundle / "unit_conflicts.jsonl").open("w", encoding="utf-8") as out:
        for row in identity_quarantine:
            out.write(_json(row) + "\n")
    inherited_quarantine += identity_quarantine
    with profile.QUARANTINE_JSONL.open("w", encoding="utf-8") as out:
        for row in inherited_quarantine:
            out.write(_json(row) + "\n")
    references.extract(profile=profile)
    good, quarantine = project.run(profile=profile, baseline_quarantine=inherited_quarantine)
    resolution = resolve_external.resolve(profile=profile)
    n_language_incomplete = _language_incomplete(profile)
    targets = sorted(chosen & good)
    _write_json(bundle / "target_acts.json", targets)
    _write_json(bundle / "selected_acts.json", selected)
    _write_json(bundle / "official_sources.json", _official_provenance(new_profile, chosen))
    summaries = list(_rows(new_profile.STRUCTURE_DIR / "segment_summary.jsonl"))
    with (bundle / "segment_summary.jsonl").open("w", encoding="utf-8") as out:
        for row in summaries:
            out.write(_json(row) + "\n")
    article_rows = list(_rows(new_profile.ARTICLES_JSONL))
    complete = Counter()
    for row in _rows(profile.REFERENCE_STATUS_JSONL):
        if row["celex_id"] in targets:
            complete["complete_new_articles" if row["complete"] else "incomplete_new_articles"] += 1
    counts = {
        "baseline_acts": len(baseline_acts),
        "selected_new_acts": len(chosen),
        "segmented_new_acts": len({row["celex_id"] for row in article_rows}),
        "accepted_new_acts": len(targets),
        "quarantined_selected_acts": len(chosen - set(targets)),
        "new_unit_rows": len(article_rows),
        "identity_conflict_acts": len(identity_quarantine),
        "language_incomplete_articles": n_language_incomplete,
        "complete_new_articles": complete["complete_new_articles"],
        "incomplete_new_articles": complete["incomplete_new_articles"],
    }
    report = {
        "counts": counts,
        "requested_pilot_size": pilot_size,
        "selected_celex": sorted(chosen),
        "target_celex": targets,
        "quarantine": [
            row for row in quarantine if (row.get("celex_id") or row.get("celex")) in chosen
        ],
        "reference_resolution": resolution,
        "identity_conflicts": identity_quarantine,
        "source_date_range": {
            "first": min(
                (row["date"] for row in selected if row["celex_id"] in targets), default=None
            ),
            "last": max(
                (row["date"] for row in selected if row["celex_id"] in targets), default=None
            ),
        },
        "coverage_note": "A stratified pilot of supplied original acts; dates between endpoints are not a completeness claim.",
    }
    _write_json(bundle / "preparation_report.json", report)
    names = list(REQUIRED_FILES) + [
        "external_references.jsonl",
        "dropped_references.jsonl",
        "reference_stats.json",
        "baseline_quarantine.jsonl",
        "unit_conflicts.jsonl",
        "unresolved_external.jsonl",
        "external_resolution_stats.json",
        "selected_acts.json",
        "official_sources.json",
        "segment_summary.jsonl",
        "preparation_report.json",
    ]
    names += [
        str(path.relative_to(bundle))
        for path in sorted(profile.REPORTS_DIR.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "files": _hashes(bundle, names),
        "source": {
            "repository": inventory_manifest["source"]["repository"],
            "revision": inventory_manifest["source"]["revision"],
            "date_field": "date",
            "date_semantics": inventory_manifest["source"]["date_semantics"],
            "inventory_sha256": identity["inventory_sha256"],
            "baseline": baseline,
            "article_source": "official_CELLAR_Formex",
            "identifiers_source": "official_CELLAR_identifiers",
            "parser": _parser_provenance(parser),
            "preparation_code": {
                "clir_bench.domains.legal.structure.eurlex_resources": sha256_file(Path(__file__)),
                **{
                    module.__name__: sha256_file(Path(module.__file__))
                    for module in (cellar, segment, references, project, resolve_external)
                },
            },
        },
        "config": {
            "from_year": FROM_YEAR,
            "to_year": to_year,
            "pilot_size": pilot_size,
            "seed": seed,
            "required_languages": list(ACT_LANGUAGES),
            "model_alignment": False,
            "baseline_reference_only": True,
        },
        "counts": counts,
    }
    if inventory_manifest["source"].get("mode") == CELLAR_DISCOVERY:
        manifest["source"]["mode"] = CELLAR_DISCOVERY
    _write_json(bundle / "manifest.json", manifest)
    audit(bundle / "manifest.json")
    # Detect concurrent edits to either immutable input before publication.
    if (
        sha256_file(inventory_path) != identity["inventory_sha256"]
        or _baseline_info(baseline_structure)[0] != baseline
    ):
        raise ValidationError("Inventory or baseline changed during preparation")
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}")
    bundle.rename(out_dir)
    return manifest


def audit(manifest_path: Path, *, cache_dir: Path | None = None) -> dict:
    """Check all declared hashes and measured structural/target invariants."""
    manifest_path = _manifest_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _check_files(manifest_path, manifest)
    to_year = discovery_year(manifest)
    source = manifest.get("source", {})
    official = source.get("mode") == CELLAR_DISCOVERY
    if not official and not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", "")):
        raise ValidationError("Source revision must be an immutable commit")
    root = manifest_path.parent
    if manifest.get("format") == INVENTORY_FORMAT:
        if official and manifest["files"].get("cellar-discovery.json") != source["revision"]:
            raise ValidationError("Official EUR-Lex discovery snapshot checksum mismatch")
        records = list(_rows(root / "inventory.jsonl"))
        if any(not _canonical(row.get("celex_id"), to_year=to_year) or not row.get("eligible")
               for row in records):
            raise ValidationError("Invalid canonical inventory act")
        if len({row["celex_id"] for row in records}) != len(records):
            raise ValidationError("Duplicate inventory act")
        config = manifest["config"]
        if json.loads((root / "pilot.json").read_text()) != stratified_sample(
            records, config["pilot_size"], config["seed"]
        ):
            raise ValidationError("Inventory pilot differs from deterministic selection")
        if len(records) != manifest["counts"]["canonical_acts"]:
            raise ValidationError("Inventory act count mismatch")
        return manifest["counts"]
    if manifest.get("format") != FORMAT or any(
        name not in manifest["files"] for name in REQUIRED_FILES
    ):
        raise ValidationError("Unsupported or incomplete EUR-Lex resources bundle")
    _audit_official_provenance(root, cache_dir=cache_dir, to_year=to_year)
    parser = manifest.get("source", {}).get("parser")
    if not isinstance(parser, dict) or parser.get("kind") not in (
        "parse_fmx",
        "injected_python_parser",
    ):
        raise ValidationError("Missing parser provenance")
    if parser["kind"] == "parse_fmx" and not re.fullmatch(
        r"[0-9a-f]{64}", parser.get("entrypoint_sha256", "")
    ):
        raise ValidationError("Invalid parser entrypoint provenance")
    targets = json.loads((root / "target_acts.json").read_text())
    selected = json.loads((root / "selected_acts.json").read_text())
    selected_ids = {row["celex_id"] for row in selected}
    if (
        targets != sorted(set(targets))
        or any(not _canonical(celex, to_year=to_year) for celex in targets)
        or not set(targets) <= selected_ids
    ):
        raise ValidationError("Invalid new-act target allowlist")
    quarantined = {
        row.get("celex_id") or row.get("celex") for row in _rows(root / "quarantine.jsonl")
    }
    if set(targets) & quarantined:
        raise ValidationError("A quarantined act is allowed as a target")
    if find_unit_conflicts(root / "articles.jsonl", quarantined):
        raise ValidationError(
            "Unquarantined conflicting unit identities or source/language records"
        )
    prior_quarantine = _safe_file(root, "baseline_quarantine.jsonl")
    prior_hash = manifest["source"]["baseline"]["quarantine_sha256"]
    if prior_hash is not None and sha256_file(prior_quarantine) != prior_hash:
        raise ValidationError("Baseline quarantine bytes were changed")
    if not Counter(map(_json, _rows(prior_quarantine))) <= Counter(
        map(_json, _rows(root / "quarantine.jsonl"))
    ):
        raise ValidationError("Baseline quarantine records were not carried into the union")
    strict = _new_validation(root / "articles.jsonl", selected_ids, ignore_other=True,
                             to_year=to_year)
    if any(row["celex_id"] in targets for row in strict):
        raise ValidationError("Target act lacks clean four-language article identities")
    units: dict[str, dict[str, dict]] = defaultdict(dict)
    baseline_prefix = hashlib.sha256()
    prefix_remaining = manifest["source"]["baseline"]["articles_bytes"]
    with (root / "articles.jsonl").open("rb") as handle:
        while prefix_remaining:
            chunk = handle.read(min(prefix_remaining, 1024 * 1024))
            if not chunk:
                raise ValidationError("Baseline article prefix is truncated")
            baseline_prefix.update(chunk)
            prefix_remaining -= len(chunk)
    if baseline_prefix.hexdigest() != manifest["source"]["baseline"]["articles_sha256"]:
        raise ValidationError("Baseline article bytes were changed")
    for row in _rows(root / "articles.jsonl"):
        if row.get("eli_id"):
            units[row["eli_id"]][row["language"]] = row
    statuses = {row["eli_id"]: row for row in _rows(root / "reference_status.jsonl")}
    for versions in units.values():
        english = versions.get("en")
        if (
            english
            and english.get("celex_id") in targets
            and english.get("unit_type") == "article"
            and english["eli_id"] not in statuses
        ):
            raise ValidationError("Target article lacks reference completeness metadata")
    for name in ("internal_edges.jsonl", "external_edges.jsonl"):
        for edge in _rows(root / name):
            source = edge.get("source_article_id")
            if units.get(source, {}).get("en", {}).get("celex_id") not in targets:
                continue
            verdict = statuses.get(source, {})
            for endpoint in (source, edge.get("target_article_id")):
                if (
                    verdict.get("complete")
                    and not units.get(endpoint, {}).get("en", {}).get("text", "").strip()
                ):
                    raise ValidationError("Complete target reference lacks its English endpoint")
            raw = units[source]["en"]["text"]
            start, end = edge.get("char_start"), edge.get("char_end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw):
                raise ValidationError("Reference character span is outside its English source")
    if len(targets) != manifest["counts"]["accepted_new_acts"]:
        raise ValidationError("Accepted target act count mismatch")
    if "target_articles.json" in manifest["files"] or "target_articles" in manifest["config"]:
        from clir_bench.domains.legal.structure.profile import ProfileArticleIndex

        expected = manifest["config"].get("target_articles")
        if type(expected) is not int or expected < 1 or "target_articles.json" not in manifest["files"]:
            raise ValidationError("Missing or invalid eligible article budget")
        article_targets = json.loads((root / "target_articles.json").read_text())
        if (not isinstance(article_targets, list) or article_targets != sorted(set(article_targets))
                or len(article_targets) != expected
                or manifest["counts"].get("target_articles") != expected):
            raise ValidationError("Eligible article target count mismatch")
        # Release the full raw-unit audit map before constructing the QA index.
        del units, statuses
        index = ProfileArticleIndex(manifest_path)
        for identifier in article_targets:
            if not index.is_target_eligible(identifier):
                raise ValidationError(f"Article budget contains an ineligible QA target: {identifier}")
    return manifest["counts"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discovery = commands.add_parser(
        "inventory", help="Download and inventory pinned English discovery files"
    )
    discovery.add_argument("--out-dir", required=True, type=Path)
    discovery.add_argument("--revision", default=REVISION)
    discovery.add_argument("--pilot-size", type=int, default=30)
    discovery.add_argument("--seed", type=int, default=SEED)
    build = commands.add_parser(
        "prepare", help="Prepare an isolated four-language official-source pilot"
    )
    build.add_argument("--inventory", required=True, type=Path)
    build.add_argument("--baseline-structure", required=True, type=Path)
    build.add_argument("--out-dir", required=True, type=Path)
    build.add_argument("--pilot-size", type=int, default=30)
    build.add_argument("--seed", type=int, default=SEED)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument("--work-dir", type=Path)
    build.add_argument("--target-articles", type=int, help="Stop after this many distinct QA-eligible articles")
    build.add_argument("--batch-size", type=int, default=50)
    build.add_argument("--exclude-profile", type=Path, action="append", default=[])
    build.add_argument("--cache-from", type=Path, action="append", default=[])
    check = commands.add_parser("audit")
    check.add_argument("manifest", type=Path)
    check.add_argument(
        "--cache-dir",
        type=Path,
        help="Verify recorded original raw bytes and retrieval metadata against this cache",
    )
    args = parser.parse_args(argv)
    if args.command == "inventory":
        result = inventory(
            args.out_dir, revision=args.revision, pilot_size=args.pilot_size, seed=args.seed
        )
    elif args.command == "prepare":
        result = prepare(
            args.inventory,
            args.baseline_structure,
            args.out_dir,
            pilot_size=args.pilot_size,
            seed=args.seed,
            workers=args.workers,
            work_dir=args.work_dir,
            target_articles=args.target_articles,
            batch_size=args.batch_size,
            exclude_profiles=args.exclude_profile,
            cache_from=args.cache_from,
        )
    else:
        print(json.dumps(audit(args.manifest, cache_dir=args.cache_dir), sort_keys=True))
        return
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
