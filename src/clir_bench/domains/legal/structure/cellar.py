"""
Stage 2 -- fetch acts from CELLAR and cache them.

Everything CELLAR returns is written to disk once and read from disk forever
after. Iterating on the parser must never re-hit the Publications Office.

Two kinds of request per act, plus one per language:

    identifiers notice   919 bytes, carries the authoritative ELI URI
    Formex zip           one per language, contains the ACT plus its annexes

The ELI lookup is not optional bookkeeping. ``32014R0680`` is
``eli/reg_impl/2014/680``, not ``eli/reg/2014/680`` -- implementing and delegated
acts take distinct ELI subtypes that the CELEX letter does not encode, so
building the URI from the CELEX id would mint wrong identifiers for a large
share of post-2004 regulations.

Scale notes, from running the full corpus rather than a pilot:

* **The cache is sharded by year.** A flat directory would hold ~140k files for
  the whole corpus. ``cache/2014/32014R0110.en.zip`` keeps each directory near a
  thousand entries.
* **Zips are never extracted.** ``parse-fmx`` reads XML on stdin, so the act
  document is handed over in memory. Extracting would have created another
  ~120k files for no benefit.
* **Requests are concurrent but globally rate limited.** Threads share one token
  bucket, so raising the worker count never raises the request rate seen by the
  Publications Office.

Usage:
    python -m clir_bench.domains.legal.structure.cellar --pilot
    python -m clir_bench.domains.legal.structure.cellar --all --workers 6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from clir_bench.domains.legal.structure import ACT_LANGUAGES, paths

CELLAR = "http://publications.europa.eu/resource/celex/{celex}"

# CELLAR negotiates language by ISO 639-2/B three-letter code.
LANG_CODE = {"en": "eng", "fr": "fra", "de": "deu", "es": "spa"}

USER_AGENT = "clir-benchmark/0.1 (research; EUR-Lex article segmentation)"
_ELI_RE = re.compile(r"eli/([a-z_]+)/(\d{4})/(\d+)")


class RateLimiter:
    """A shared floor on the interval between requests, across all threads.

    The Publications Office documents no hard limit, so the polite thing is to
    keep the aggregate rate modest regardless of how much parallelism is used
    locally. Concurrency here hides latency; it does not raise the request rate.
    """

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            due = max(now, self._next)
            self._next = due + self._interval
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)


_limiter = RateLimiter(6.0)


class FetchError(RuntimeError):
    pass


def set_rate(per_second: float) -> None:
    global _limiter
    _limiter = RateLimiter(per_second)


def _cache_dir(celex: str, *, profile=None) -> Path:
    """Shard by the year embedded in the CELEX id (``32014R0110`` -> ``2014``)."""
    profile_paths = profile or paths
    year = celex[1:5] if len(celex) >= 5 and celex[1:5].isdigit() else "other"
    path = profile_paths.CACHE_DIR / year
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_file(celex: str, name: str, *, profile=None) -> Path:
    return _cache_dir(celex, profile=profile) / f"{celex}.{name}"


def _get(url: str, accept: str, *, language: str | None = None,
         retries: int = 4, timeout: int = 180) -> bytes:
    headers = {"Accept": accept, "User-Agent": USER_AGENT}
    if language:
        headers["Accept-Language"] = language

    last: Exception | None = None
    for attempt in range(retries):
        _limiter.wait()
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            # A 404 is a fact about the act, not a transient failure: CELLAR has
            # no Formex for it. Retrying cannot change that.
            if error.code == 404:
                raise FetchError(f"404 {url} ({accept})") from error
            # Back off hard if we are being throttled.
            if error.code in (429, 503):
                time.sleep(5 * (attempt + 1))
            last = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last = error
        time.sleep(2 ** attempt)
    raise FetchError(f"{url} ({accept}): {last}")


# -- ELI ------------------------------------------------------------------- #

def eli_for(celex: str, *, profile=None, transport=None) -> str | None:
    """Authoritative ELI act URI, e.g. ``eli/reg_impl/2014/680``. Cached."""
    if profile is not None:
        return _profile_eli_for(celex, profile, transport or _get)
    cache = cache_file(celex, "identifiers.xml")
    miss = cache_file(celex, "identifiers.miss")
    if miss.exists():
        return None
    if not cache.exists():
        try:
            cache.write_bytes(_get(CELLAR.format(celex=celex),
                                   "application/xml;notice=identifiers"))
        except FetchError as error:
            miss.write_text(str(error))
            return None
    match = _ELI_RE.search(cache.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    kind, year, number = match.groups()
    return f"http://data.europa.eu/eli/{kind}/{year}/{int(number)}/oj"


def article_eli(act_eli: str, article_number: str) -> str:
    """ELI subdivision URI for one article.

    Constructed, not looked up. CELLAR mints subdivision URIs only for the few
    articles that serve as a legal basis elsewhere (two of the GDPR's ninety-nine),
    and eur-lex.europa.eu returns HTTP 200 for `art_banana`, so neither the
    registry nor the web front end can confirm an article id. The subdivision
    pattern is the official ELI shape and is language-independent, which is why
    it is used, but every id here is only as correct as the Formex inventory it
    came from.
    """
    stem = act_eli[: -len("/oj")] if act_eli.endswith("/oj") else act_eli
    token = re.sub(r"\s+", "", str(article_number)).lower()
    return f"{stem}/art_{token}/oj"


# -- Formex ---------------------------------------------------------------- #

@dataclass
class ActDocuments:
    """The XML documents CELLAR returned for one (act, language), in memory."""

    celex: str
    language: str
    act_xml: bytes | None
    annex_xml: list[bytes]
    error: str = ""


def ensure_zip(celex: str, language: str, *, profile=None, transport=None) -> tuple[Path | None, str]:
    """Formex zip for one (act, language), fetching only on a cache miss."""
    if profile is not None:
        return _profile_ensure_zip(celex, language, profile, transport or _get)
    zip_path = cache_file(celex, f"{language}.zip")
    miss_path = cache_file(celex, f"{language}.miss")
    if zip_path.exists():
        return zip_path, ""
    if miss_path.exists():
        return None, miss_path.read_text().strip()
    try:
        data = _get(CELLAR.format(celex=celex), "application/zip;mtype=fmx4",
                    language=LANG_CODE[language])
    except FetchError as error:
        miss_path.write_text(str(error))
        return None, str(error)
    zip_path.write_bytes(data)
    return zip_path, ""


def read_documents(celex: str, language: str, *, profile=None, transport=None) -> ActDocuments:
    """Act and annex XML for one (act, language), read straight out of the zip."""
    zip_path, error = (ensure_zip(celex, language, profile=profile, transport=transport)
                       if profile is not None else ensure_zip(celex, language))
    if zip_path is None:
        return ActDocuments(celex, language, None, [], error)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = [n for n in archive.namelist()
                     if n.endswith(".xml")
                     and not n.endswith(".doc.xml")   # metadata sidecar
                     and not n.endswith(".toc.xml")]  # OJ table of contents
            act: bytes | None = None
            annexes: list[bytes] = []
            for name in sorted(names):
                payload = archive.read(name)
                head = payload[:600].decode("utf-8", "replace")
                # Route by root element. A Formex zip holds one XML per OJ
                # document, so annexes usually arrive as their own files and the
                # largest file is very often an annex rather than the act.
                if re.search(r"<ACT[\s>]", head):
                    if act is None:
                        act = payload
                elif re.search(r"<ANNEX[\s>]", head):
                    annexes.append(payload)
            if act is None:
                return ActDocuments(celex, language, None, annexes,
                                    "no-ACT-root-document")
            return ActDocuments(celex, language, act, annexes)
    except (zipfile.BadZipFile, OSError) as error:
        return ActDocuments(celex, language, None, [], f"bad-zip: {error}")


# -- bulk fetch ------------------------------------------------------------- #

def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value) -> None:
    _atomic_write(path, (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode())


def _permanent(error) -> bool:
    return (isinstance(error, urllib.error.HTTPError) and error.code == 404
            or isinstance(error, FetchError) and str(error).startswith("404 "))


def _fetch_error(path: Path, error) -> None:
    _atomic_json(path, {"error": str(error), "permanent": _permanent(error),
                        "recorded_at": datetime.now(timezone.utc).isoformat()})


def _cached_missing(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("permanent") is True
    except (ValueError, AttributeError):
        # Old text misses may have recorded temporary DNS/HTTP errors forever.
        return path.read_text().startswith("404 ")


def _record_raw(path: Path, payload: bytes) -> None:
    _atomic_json(path.with_suffix(path.suffix + ".meta.json"), {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    })


def _read_raw(path: Path) -> bytes | None:
    if not path.exists():
        return None
    payload = path.read_bytes()
    metadata = path.with_suffix(path.suffix + ".meta.json")
    if metadata.exists():
        try:
            expected = json.loads(metadata.read_text())["sha256"]
        except (ValueError, KeyError, TypeError) as error:
            raise FetchError(f"invalid-raw-cache-metadata: {path.name}") from error
        if hashlib.sha256(payload).hexdigest() != expected:
            raise FetchError(f"cached-raw-checksum-mismatch: {path.name}")
    if not metadata.exists():
        _atomic_json(metadata, {"sha256": hashlib.sha256(payload).hexdigest(),
                                "retrieved_at": None, "provenance": "recovered-cache"})
    return payload


def _checked_eli(payload: bytes, celex: str) -> str:
    try:
        ET.fromstring(payload)
    except ET.ParseError as error:
        raise FetchError("invalid-identifiers-xml") from error
    expected_year, expected_number = celex[1:5], int(celex[6:])
    prefix = "reg" if celex[5] == "R" else "dir"
    found = {(kind, year, int(number)) for kind, year, number in
             _ELI_RE.findall(payload.decode("utf-8"))
             if kind.startswith(prefix) and year == expected_year and int(number) == expected_number}
    if len(found) != 1:
        raise FetchError("missing-or-ambiguous-authoritative-eli")
    kind, year, number = next(iter(found))
    return f"http://data.europa.eu/eli/{kind}/{year}/{number}/oj"


def _profile_eli_for(celex, profile, transport):
    cache = cache_file(celex, "identifiers.xml", profile=profile)
    miss = cache_file(celex, "identifiers.miss", profile=profile)
    if _cached_missing(miss):
        return None
    try:
        payload = _read_raw(cache)
        if payload is None:
            payload = transport(CELLAR.format(celex=celex).replace("http:", "https:", 1),
                                "application/xml;notice=identifiers")
            eli = _checked_eli(payload, celex)
            _atomic_write(cache, payload)
            _record_raw(cache, payload)
        else:
            eli = _checked_eli(payload, celex)
        miss.unlink(missing_ok=True)
        return eli
    except (FetchError, urllib.error.URLError, OSError, ValueError) as error:
        _fetch_error(miss, error)
        return None


def _child(element, name):
    return next((node for node in element if node.tag.rsplit("}", 1)[-1] == name), None)


def _text(element, name):
    node = _child(element, name) if element is not None else None
    return (node.text or "").strip() if node is not None else ""


def validate_formex(payload: bytes, celex: str, language: str) -> None:
    """Verify archive/ACT identity using Formex's own bibliographic metadata."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.testzip() is not None:
                raise FetchError("corrupt-formex-member")
            acts = []
            for name in archive.namelist():
                if not name.endswith(".xml") or name.endswith((".doc.xml", ".toc.xml")):
                    continue
                root = ET.fromstring(archive.read(name))
                kind = root.tag.rsplit("}", 1)[-1]
                if kind not in {"ACT", "ANNEX"}:
                    continue
                bibliography = _child(root, "BIB.INSTANCE")
                lang = _text(bibliography, "LG.DOC")
                if lang and lang.lower() != language:
                    raise FetchError("formex-language-mismatch")
                if kind == "ACT":
                    acts.append(root)
            if len(acts) != 1:
                raise FetchError("formex-requires-one-ACT")
            bibliography = _child(acts[0], "BIB.INSTANCE")
            if _text(bibliography, "LG.DOC").lower() != language:
                raise FetchError("missing-or-wrong-formex-language")
            number = _child(bibliography, "NO.DOC") if bibliography is not None else None
            year, current = _text(number, "YEAR"), _text(number, "NO.CURRENT")
            if not year or not current.isdigit():
                raise FetchError("missing-formex-act-identity")
            if year != celex[1:5] or int(current) != int(celex[6:]):
                raise FetchError("formex-act-identity-mismatch")
    except FetchError:
        raise
    except (zipfile.BadZipFile, ET.ParseError, KeyError, RuntimeError) as error:
        raise FetchError(f"invalid-formex-archive: {error}") from error


def _profile_ensure_zip(celex, language, profile, transport):
    cache = cache_file(celex, f"{language}.zip", profile=profile)
    miss = cache_file(celex, f"{language}.miss", profile=profile)
    if _cached_missing(miss):
        return None, miss.read_text().strip()
    try:
        payload = _read_raw(cache)
        if payload is None:
            payload = transport(CELLAR.format(celex=celex).replace("http:", "https:", 1),
                                "application/zip;mtype=fmx4", language=LANG_CODE[language])
            validate_formex(payload, celex, language)
            _atomic_write(cache, payload)
            _record_raw(cache, payload)
        else:
            validate_formex(payload, celex, language)
        miss.unlink(missing_ok=True)
        return cache, ""
    except (FetchError, urllib.error.URLError, OSError, ValueError) as error:
        _fetch_error(miss, error)
        return None, str(error)


def _profile_fetch_all(celexes, workers, profile, transport):
    profile.ensure_dirs()
    manifest = profile.FETCH_MANIFEST_JSON
    try:
        done = json.loads(manifest.read_text()) if manifest.exists() else {}
    except ValueError:
        done = {}

    def one(celex):
        entry = {"eli": eli_for(celex, profile=profile, transport=transport), "languages": {}}
        for language in ACT_LANGUAGES:
            path, error = ensure_zip(celex, language, profile=profile, transport=transport)
            entry["languages"][language] = {"zip": str(path) if path else None, "error": error}
        return celex, entry

    # Revisit every selected act: successful slots are cache-only, while an old
    # partial/transient manifest entry must never suppress its missing slots.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for celex, entry in pool.map(one, sorted(set(celexes))):
            done[celex] = entry
            _atomic_json(manifest, done)
    return done


def fetch_all(celexes: list[str], *, workers: int = 6, profile=None, transport=None) -> dict:
    """Warm the cache for every act, concurrently. Safe to re-run: cache-first."""
    if profile is not None:
        return _profile_fetch_all(celexes, workers, profile, transport or _get)
    paths.ensure_dirs()
    manifest_path = paths.STRUCTURE_DIR / "fetch_manifest.json"
    done: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            done = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            done = {}

    todo = [c for c in celexes if c not in done]
    print(f"[fetch] {len(celexes):,} acts, {len(done):,} already recorded, "
          f"{len(todo):,} to do, {workers} workers", flush=True)

    lock = threading.Lock()
    counters = {"ok": 0, "partial": 0, "none": 0, "n": 0}
    started = time.monotonic()

    def one(celex: str) -> tuple[str, dict]:
        eli = eli_for(celex)
        entry: dict = {"eli": eli, "languages": {}}
        for language in ACT_LANGUAGES:
            zip_path, error = ensure_zip(celex, language)
            entry["languages"][language] = {
                "zip": str(zip_path) if zip_path else None, "error": error}
        return celex, entry

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, c): c for c in todo}
        for future in as_completed(futures):
            celex = futures[future]
            try:
                celex, entry = future.result()
            except Exception as error:  # never let one act kill the run
                entry = {"eli": None, "languages": {}, "error": str(error)[:200]}
            with lock:
                done[celex] = entry
                got = sum(1 for v in entry.get("languages", {}).values() if v["zip"])
                counters["n"] += 1
                counters["ok" if got == 4 else ("none" if got == 0 else "partial")] += 1
                if counters["n"] % 200 == 0 or counters["n"] == len(todo):
                    elapsed = time.monotonic() - started
                    rate = counters["n"] / elapsed if elapsed else 0
                    remaining = (len(todo) - counters["n"]) / rate if rate else 0
                    print(f"  {counters['n']:>6,}/{len(todo):,}  "
                          f"4-lang={counters['ok']:,} partial={counters['partial']:,} "
                          f"none={counters['none']:,}  {rate:.1f} acts/s  "
                          f"eta {remaining / 3600:.1f}h", flush=True)
                    manifest_path.write_text(json.dumps(done))

    manifest_path.write_text(json.dumps(done))
    print(f"[fetch] done -> {manifest_path}", flush=True)
    return done


def read_celexes(args) -> list[str]:
    celexes = list(args.celex)
    if args.pilot:
        with paths.PILOT_CSV.open(encoding="utf-8") as fh:
            celexes += [row["celex_id"] for row in csv.DictReader(fh)]
    if args.all:
        with paths.CANDIDATES_CSV.open(encoding="utf-8") as fh:
            celexes += [row["celex_id"] for row in csv.DictReader(fh)
                        if int(row["year"]) >= args.from_year]
    return celexes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="every act in candidates.csv")
    parser.add_argument("--from-year", type=int, default=2004)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rate", type=float, default=6.0,
                        help="global requests per second across all workers")
    parser.add_argument("celex", nargs="*")
    args = parser.parse_args()

    celexes = read_celexes(args)
    if not celexes:
        parser.error("give CELEX ids, --pilot or --all")
    set_rate(args.rate)
    fetch_all(celexes, workers=args.workers)


if __name__ == "__main__":
    main()
