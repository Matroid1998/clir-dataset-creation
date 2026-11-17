"""
Stage 3 -- segment cached Formex into article records.

Hybrid by design: segmentation comes from ``eur-lex-visualiser``'s ``parse-fmx``,
driven as a subprocess, because it already handles the awkward parts of Formex
(optional Part/Title/Chapter/Section nesting, quoted amending text, list
flattening). Reference extraction does *not* come from it -- that is stage 5,
written here against the spec so the rule that fired is recorded per edge.

That tool is GPL-3.0. It is cloned into ``tools/`` (gitignored) and invoked over
a pipe; no code from it is copied into this repository.

Annexes are handled separately. A Formex zip usually ships each annex as its own
XML document whose root element is ANNEX, and ``parse-fmx`` returns an empty body
for those, so annex text is extracted directly here.

Usage:
    python -m clir_bench.domains.legal.structure.segment --pilot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree

from clir_bench.domains.legal.structure import ACT_LANGUAGES, cellar, paths

DEFAULT_PARSE_FMX = paths.ROOT / "tools/eur-lex-visualiser/backend/bin/parse-fmx.js"


def parse_fmx_path() -> Path:
    path = Path(os.environ.get("EURLEX_PARSE_FMX", DEFAULT_PARSE_FMX))
    if not path.exists():
        raise SystemExit(
            f"parse-fmx not found at {path}.\n"
            "Clone the tool (GPL-3.0, used as a subprocess, never vendored):\n"
            "  git clone --depth 1 https://github.com/maastrichtlawtech/eur-lex-visualiser "
            "tools/eur-lex-visualiser\n"
            "  cd tools/eur-lex-visualiser/backend && npm install jsdom@25\n"
            "or set EURLEX_PARSE_FMX to an existing checkout."
        )
    return path


# -- HTML to text ---------------------------------------------------------- #

_BLOCK_TAGS = r"p|div|li|ol|ul|table|tr|h[1-6]|br|blockquote"
_BLOCK_RE = re.compile(rf"</?(?:{_BLOCK_TAGS})\b[^>]*>", re.I)

# Footnote markers are `<sup class="fmx-footnote-ref">...<a>1</a></sup>` sitting
# flush against the preceding word. Stripping tags naively glues the marker
# digit onto that word, which turned "Regulation (EEC) No 2913/92" into
# "...2913/921". Article numbers were not hit in the pilot, but "Article 5" plus
# a marker would read as "Article 51" and silently retarget an edge, so the
# markers are removed with their content before anything else runs.
_FOOTNOTE_REF_RE = re.compile(
    r"<sup\b[^>]*class=\"[^\"]*footnote-ref[^\"]*\"[^>]*>.*?</sup>", re.I | re.S)
# The trailing footnote list is apparatus (OJ citations), not enacting text.
_FOOTNOTE_BODY_RE = re.compile(
    r"<section\b[^>]*class=\"[^\"]*fmx-footnotes[^\"]*\"[^>]*>.*?</section>", re.I | re.S)
# Emphasis markup is replaced by a space, not deleted. Formex loses the space at
# an italic boundary -- the OJ reads "artículos 97 octies y 97 nonies" but the
# XML holds `97 <HT TYPE="ITALIC">octies</HT>y 97`, so plain tag-stripping yields
# "97 octiesy 97 nonies". Since EU drafting italicises exactly the Latin ordinals
# that suffix article numbers ("97 nonies", "13 bis"), that damage lands squarely
# on the identifiers this pipeline resolves. Emphasis always wraps whole words
# here, so a space is the faithful reconstruction; other inline tags are still
# removed outright so that markup inside a word does not gain a space.
_EMPHASIS_RE = re.compile(r"</?(?:em|i|b|strong)\b[^>]*>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")


def html_to_text(fragment: str) -> str:
    """Plain text for a parser HTML fragment, with block structure as newlines.

    Reference char spans are recorded against exactly this output, so the
    transform has to stay deterministic -- changing it invalidates every span
    already extracted.
    """
    if not fragment:
        return ""
    text = _FOOTNOTE_BODY_RE.sub(" ", fragment)
    text = _FOOTNOTE_REF_RE.sub("", text)
    text = _BLOCK_RE.sub("\n", text)
    text = _EMPHASIS_RE.sub(" ", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    text = _SPACE_RUN_RE.sub(" ", text)
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def token_count(text: str) -> int:
    """Whitespace tokens. A stable proxy, not a model tokenizer count."""
    return len((text or "").split())


# -- structural flags ------------------------------------------------------ #

# Computed on English only and copied to the other languages by the structural
# join, for the same reason references are: per-language detection drifts, and a
# flag that disagrees across languages would silently split the corpus.
_DEFINITIONS_TITLE_RE = re.compile(r"\bdefinitions?\b", re.I)
_FINAL_TITLE_RE = re.compile(
    r"\b(entry into force|addressees|final provisions?|transitional provisions?"
    r"|transposition|repeal|repeals|application|implementation)\b", re.I)
_AMENDING_TITLE_RE = re.compile(r"\bamend(ment|ments|ing)?\b", re.I)
_AMENDING_BODY_RE = re.compile(
    r"\b(is|are) (hereby )?amended as follows\b"
    r"|\bshall be (replaced by|deleted|inserted)\b"
    r"|\bis (replaced by|deleted|inserted)\b", re.I)


def structural_flags(*, title: str, text: str, definition_articles: set[str],
                     article_number: str) -> dict[str, bool]:
    title = title or ""
    return {
        "is_definitions": bool(_DEFINITIONS_TITLE_RE.search(title))
                          or article_number in definition_articles,
        "is_final_provision": bool(_FINAL_TITLE_RE.search(title)),
        "is_amending": bool(_AMENDING_TITLE_RE.search(title))
                       or bool(_AMENDING_BODY_RE.search(text)),
    }


# -- parse ----------------------------------------------------------------- #

def run_parse_fmx(xml: bytes, out_path: Path) -> dict | None:
    """Parse one Formex ACT document. `xml` is handed over on stdin.

    Feeding stdin rather than a path is what lets the whole corpus run without
    extracting every zip: at ~120k act and annex documents, extraction would
    have cost more files than the cache itself.
    """
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except json.JSONDecodeError:
            out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            ["node", str(parse_fmx_path())],
            input=xml, capture_output=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        print(f"    parse-fmx timed out on {out_path.stem}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"    parse-fmx failed on {out_path.stem}: "
              f"{proc.stderr.decode('utf-8', 'replace').strip()[:160]}", file=sys.stderr)
        return None
    try:
        parsed = json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None
    out_path.write_text(json.dumps(parsed, ensure_ascii=False))
    return parsed


_ANNEX_TITLE_RE = re.compile(r"<TI>(.*?)</TI>", re.S | re.I)
_XML_TAG_RE = re.compile(r"<[^>]+>")

# The act's own title, e.g. "Regulation (EC) No 1223/2009 ... on cosmetic
# products (recast)". `parse-fmx` reports only the short designation
# ("Regulation (EC) No 1223/2009"), which names the instrument but not its
# subject matter -- and the subject matter is the useful half when the title is
# shown to a question generator as document metadata.
#
# TITLE is not unique in Formex: divisions carry one too (the GDPR has 27 TITLE
# elements for 26 divisions). The act's title is the first, appearing as a direct
# child of ACT before PREAMBLE, so document order picks it out.
_ACT_TITLE_RE = re.compile(r"<TITLE>(.*?)</TITLE>", re.S | re.I)


def act_title(xml: bytes | str) -> str:
    raw = xml.decode("utf-8", "replace") if isinstance(xml, bytes) else xml
    match = _ACT_TITLE_RE.search(raw)
    if not match:
        return ""
    text = html.unescape(_XML_TAG_RE.sub(" ", match.group(1)))
    return re.sub(r"\s+", " ", text).strip()

# "ANNEX I", "ANNEX VIIa", "ANNEX 2". The label is what the enacting text cites
# ("as set out in Annex III"), so it -- not the document's position in the zip --
# has to be the identifier.
#
# The heading word is language-specific and the numeral is not, which is the
# whole point: parsing "ANHANG II" and "ANNEX II" both to "2" is what makes the
# annex ELI identical across language versions. Matching only the English word
# left German and Spanish with positional fallback ids, and every annex edge
# then failed the cross-language join.
_ANNEX_WORDS = {
    "en": r"ANNEX(?:ES)?",
    "fr": r"ANNEXES?",
    "de": r"ANH[ÄA]NGE?",
    "es": r"ANEXOS?",
}
# French and Spanish insert subdivisions with Latin ordinals where English and
# German use a letter suffix: "ANNEXE 13 bis" is "ANNEX 13a". The mapping was
# confirmed against the parallel corpus -- bis/ter/quinquies matched a/b/d with
# identical counts on identical numbers -- and without it the same annex takes
# two different ELI ids and the cross-language join drops the edge.
# The full paradigm, confirmed by matched counts across the parallel corpus:
# French/Spanish ordinal totals equal English letter-suffix totals almost
# exactly (bis 107 = a 107, ter 46 = b 46, quater 36 = c 36, sexdecies 13 = o 13,
# septdecies 10 = p 10, vicies 4 = s 4, tervicies 3 = v 3, ...). A partial map
# silently mis-identifies the higher ordinals, which are not rare: the pilot
# alone uses sixteen distinct ones.
LATIN_ORDINAL_SUFFIX = {
    "bis": "a", "ter": "b", "quater": "c", "quinquies": "d", "sexies": "e",
    "septies": "f", "octies": "g", "nonies": "h", "novies": "h", "decies": "i",
    "undecies": "j", "duodecies": "k", "terdecies": "l", "quaterdecies": "m",
    "quindecies": "n", "sexdecies": "o", "septdecies": "p", "octodecies": "q",
    "novodecies": "r", "vicies": "s", "unvicies": "t", "duovicies": "u",
    "tervicies": "v", "quatervicies": "w",
}
# Longest first: a plain "|".join puts "ter" before "terdecies", and the
# regex alternation is first-match, so "13 terdecies" would parse as "13b".
_ORDINAL_ALT = "|".join(sorted(LATIN_ORDINAL_SUFFIX, key=len, reverse=True))
_ANNEX_LABEL_RES = {
    lang: re.compile(
        rf"^\W*{word}\s+([IVXLC]+|\d+)"
        rf"(?:\s*(?P<ordinal>{_ORDINAL_ALT})\b|\s*(?P<letter>[a-z])\b)?",
        re.I)
    for lang, word in _ANNEX_WORDS.items()
}
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(value: str) -> int | None:
    total = previous = 0
    for char in reversed(value.upper()):
        digit = _ROMAN_VALUES.get(char)
        if digit is None:
            return None
        total = total - digit if digit < previous else total + digit
        previous = max(previous, digit)
    return total or None


def annex_label(title: str, language: str = "en") -> str | None:
    """Canonical annex id from its heading: "ANNEX VIIa" -> "7a".

    Returns None when the document carries no annex label at all. That happens:
    a Formex zip may ship a "Preamble to Annexes II to VI" document alongside the
    annexes themselves, and numbering annexes by their position in the zip put
    every annex after it under the wrong identifier -- 38% of annex records in
    the pilot pointed at the wrong document before this existed.
    """
    pattern = _ANNEX_LABEL_RES.get(language) or _ANNEX_LABEL_RES["en"]
    match = pattern.match((title or "").strip())
    if not match:
        return None
    core = match.group(1)
    ordinal = (match.group("ordinal") or "").lower()
    suffix = LATIN_ORDINAL_SUFFIX.get(ordinal, (match.group("letter") or "").lower())
    number = int(core) if core.isdigit() else roman_to_int(core)
    return f"{number}{suffix}" if number else None


def annex_text(xml: bytes | str) -> tuple[str, str]:
    """(title, text) for a standalone Formex ANNEX document.

    `parse-fmx` returns an empty body when ANNEX is the root element, so annex
    text is taken straight from the XML here.
    """
    raw = xml.decode("utf-8", "replace") if isinstance(xml, bytes) else xml
    match = _ANNEX_TITLE_RE.search(raw)
    title = html.unescape(_XML_TAG_RE.sub(" ", match.group(1))).strip() if match else ""
    body = _XML_TAG_RE.sub("\n", raw)
    body = html.unescape(body).replace("\xa0", " ")
    body = _SPACE_RUN_RE.sub(" ", body)
    body = _TRAILING_SPACE_RE.sub("\n", body)
    body = _MULTI_NEWLINE_RE.sub("\n\n", body).strip()
    return re.sub(r"\s+", " ", title), body


def records_for(celex: str, language: str, act_eli: str, meta: dict,
                parsed: dict, annex_docs: list[bytes],
                flags_by_article: dict[str, dict] | None,
                act_title_text: str = "") -> list[dict]:
    """Every stored unit for one (act, language): articles, recitals, annexes."""
    definition_articles = {str(d.get("sourceArticle")) for d in parsed.get("definitions", [])
                           if d.get("sourceArticle")}
    rows: list[dict] = []

    for article in parsed.get("articles", []):
        number = str(article.get("article_number", "")).strip()
        if not number:
            continue
        title = (article.get("article_title") or "").strip()
        text = html_to_text(article.get("article_html", ""))
        division = article.get("division") or {}
        chapter = (division.get("chapter") or {}) or {}
        section = (division.get("section") or {}) or {}
        part = (division.get("part") or {}) or {}
        title_div = (division.get("title") or {}) or {}

        flags = (flags_by_article or {}).get(number) or structural_flags(
            title=title, text=text,
            definition_articles=definition_articles, article_number=number)

        rows.append({
            "eli_id": cellar.article_eli(act_eli, number),
            "act_eli": act_eli,
            "act_title": act_title_text,
            "celex_id": celex,
            "language": language,
            "unit_type": "article",
            "article_number": number,
            "heading": title,
            "text": text,
            "part": (part.get("title") or part.get("number") or "") or "",
            "title_division": (title_div.get("title") or title_div.get("number") or "") or "",
            "chapter": (chapter.get("title") or "") or "",
            "chapter_number": (chapter.get("number") or "") or "",
            "section": (section.get("title") or "") or "",
            "section_number": (section.get("number") or "") or "",
            "token_count": token_count(text),
            "char_count": len(text),
            "doc_type": meta.get("doc_type", ""),
            "year": meta.get("year", ""),
            **flags,
            "is_preamble": False,
            "is_recital": False,
            "is_annex": False,
        })

    for recital in parsed.get("recitals", []):
        number = str(recital.get("recital_number", "")).strip()
        text = recital.get("recital_text") or html_to_text(recital.get("recital_html", ""))
        rows.append({
            "eli_id": f"{act_eli[:-len('/oj')]}/rct_{number}/oj" if act_eli.endswith("/oj") else "",
            "act_eli": act_eli, "act_title": act_title_text, "celex_id": celex, "language": language,
            "unit_type": "recital", "article_number": "", "heading": f"Recital {number}",
            "text": text, "part": "", "title_division": "", "chapter": "",
            "chapter_number": "", "section": "", "section_number": "",
            "token_count": token_count(text), "char_count": len(text),
            "doc_type": meta.get("doc_type", ""), "year": meta.get("year", ""),
            "is_definitions": False, "is_final_provision": False, "is_amending": False,
            "is_preamble": False, "is_recital": True, "is_annex": False,
            "recital_number": number,
        })

    # Annexes shipped inside the ACT document, if any.
    for position, annex in enumerate(parsed.get("annexes", []), 1):
        heading = annex.get("annex_title") or ""
        text = html_to_text(annex.get("annex_html", "")) or (annex.get("annex_text") or "")
        declared = str(annex.get("annex_id") or annex.get("annex_number") or "").strip()
        rows.append(_annex_row(celex, language, act_eli, meta,
                               annex_label(heading, language) or declared or str(position),
                               heading, text,
                               labelled=annex_label(heading, language) is not None,
                               act_title_text=act_title_text))

    # Annexes shipped as their own Formex documents.
    for position, document in enumerate(annex_docs, 1):
        heading, text = annex_text(document)
        label = annex_label(heading, language)
        rows.append(_annex_row(celex, language, act_eli, meta,
                               label or f"pos{position}", heading, text,
                               labelled=label is not None, act_title_text=act_title_text))

    return rows


def _annex_row(celex, language, act_eli, meta, identifier, heading, text, *,
               labelled: bool = True, act_title_text: str = "") -> dict:
    return {
        "eli_id": f"{act_eli[:-len('/oj')]}/anx_{identifier}/oj" if act_eli.endswith("/oj") else "",
        "act_eli": act_eli, "act_title": act_title_text, "celex_id": celex, "language": language,
        "unit_type": "annex", "article_number": "", "heading": heading,
        "text": text, "part": "", "title_division": "", "chapter": "",
        "chapter_number": "", "section": "", "section_number": "",
        "token_count": token_count(text), "char_count": len(text),
        "doc_type": meta.get("doc_type", ""), "year": meta.get("year", ""),
        "is_definitions": False, "is_final_provision": False, "is_amending": False,
        "is_preamble": False, "is_recital": False, "is_annex": True,
        "annex_id": identifier,
        # False when the document carried no "ANNEX <n>" heading, so the id is a
        # positional fallback and nothing should resolve a citation to it.
        "annex_labelled": labelled,
    }


def segment_act(celex: str, entry: dict, meta: dict, *, profile=None, transport=None,
                parser=None) -> tuple[list[dict], dict]:
    """Every stored record for one act, across the languages that have Formex."""
    profile_paths = profile or paths
    act_eli = entry.get("eli")
    if not act_eli:
        return [], {"celex": celex, "skipped": "no-eli"}

    # English first: its flags are the ones every language will carry, so that a
    # flag never disagrees across language versions of the same article.
    flags_by_article: dict[str, dict] | None = None
    per_language: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for language in ("en",) + tuple(l for l in ACT_LANGUAGES if l != "en"):
        documents = (cellar.read_documents(celex, language, profile=profile, transport=transport)
                     if profile is not None else cellar.read_documents(celex, language))
        if documents.act_xml is None:
            errors[language] = documents.error or "no-act"
            continue
        output = profile_paths.PARSED_DIR / celex[1:5] / f"{celex}.{language}.json"
        try:
            parsed = (_profile_parse(documents.act_xml, output, parser or run_parse_fmx)
                      if profile is not None else (parser or run_parse_fmx)(documents.act_xml, output))
        except FormexArticleError as error:
            errors[language] = str(error)
            continue
        if parsed is None:
            errors[language] = "parse-failed"
            continue
        rows = records_for(celex, language, act_eli, meta, parsed,
                           documents.annex_xml, flags_by_article,
                           act_title_text=act_title(documents.act_xml))
        if language == "en":
            flags_by_article = {
                r["article_number"]: {k: r[k] for k in
                                      ("is_definitions", "is_final_provision", "is_amending")}
                for r in rows if r["unit_type"] == "article"
            }
        per_language[language] = rows

    rows = [r for language_rows in per_language.values() for r in language_rows]
    summary = {
        "celex": celex, "eli": act_eli,
        "articles": {lg: sum(1 for r in v if r["unit_type"] == "article")
                     for lg, v in per_language.items()},
        "errors": errors,
    }
    return rows, summary


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_version():
    result = subprocess.run(["node", "--version"], capture_output=True, text=True,
                            check=False, timeout=15)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("cannot record the Formex parser's Node version")
    return result.stdout.strip()


def _tool_git(tool_root, *arguments):
    # Read only this explicit tool checkout. Disabling optional locks and
    # configured filesystem monitors avoids writes or monitor subprocesses.
    result = subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(tool_root),
         *arguments], capture_output=True, check=False, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot record the Formex parser checkout's provenance")
    return result.stdout


def parser_provenance():
    """Fingerprint the parser entrypoint, runtime and its tracked dependencies.

    A custom entrypoint outside the expected checkout layout records adjacent
    package files without searching parent repositories or inspecting their diffs.
    """
    entrypoint = parse_fmx_path().resolve()
    backend = entrypoint.parent.parent if entrypoint.parent.name == "bin" else entrypoint.parent
    package_files = {name: _file_hash(backend / name) if (backend / name).is_file() else None
                     for name in ("package.json", "package-lock.json")}
    result = {"entrypoint_sha256": _file_hash(entrypoint), "node_version": _node_version(),
              "package_json_sha256": package_files["package.json"],
              "package_lock_sha256": package_files["package-lock.json"],
              "git_head": None, "git_dirty": None, "tracked_backend_diff_sha256": None,
              "dependency_scope": "entrypoint_and_adjacent_packages"}
    tool_root = backend.parent
    if (entrypoint.parent.name == "bin" and backend.name == "backend"
            and (tool_root / ".git").exists() and not (tool_root / ".git").is_symlink()):
        declared_root = Path(_tool_git(tool_root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        if declared_root != tool_root:
            raise RuntimeError("Formex parser checkout resolved to an unexpected repository")
        head = _tool_git(tool_root, "rev-parse", "HEAD").decode().strip()
        diff = _tool_git(tool_root, "diff", "--no-ext-diff", "--no-textconv", "--binary",
                         "HEAD", "--", "backend")
        result.update(git_head=head, git_dirty=bool(diff),
                      tracked_backend_diff_sha256=hashlib.sha256(diff).hexdigest(),
                      dependency_scope="tracked_backend_and_adjacent_packages")
    return result


def _parser_identity(parser):
    if parser is run_parse_fmx:
        return hashlib.sha256(json.dumps(parser_provenance(), sort_keys=True).encode()).hexdigest()
    code = getattr(parser, "__code__", None)
    source = Path(code.co_filename) if code is not None else None
    if source is not None and source.is_file():
        return _file_hash(source)
    return f"{getattr(parser, '__module__', '')}:{getattr(parser, '__qualname__', type(parser).__name__)}"


class FormexArticleError(ValueError):
    """A parsed article inventory cannot be proved against the official XML."""


def _profile_article_xml(xml):
    """Normalize operative GR.SEQ divisions and read the original article IDs.

    The external parser only walks DIVISION, while newer Formex also uses
    GR.SEQ for the same chapter/section hierarchy. Only those operative
    containers are renamed. Article content, quoted amendments, tables of
    contents and annexes are never traversed for normalization or promoted into
    the act's article inventory. The original source bytes remain in the cache.
    """
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise FormexArticleError("invalid-formex-article-xml") from error

    def name(node):
        return node.tag.rsplit("}", 1)[-1]

    body = next((node for node in root if name(node) == "ENACTING.TERMS"), None)
    if body is None:
        return xml, ()
    identifiers = []
    changed = False

    def number(article):
        explicit = article.get("IDENTIFIER", "").strip()
        identifier = re.fullmatch(r"0*(\d+)([a-z]*)", explicit, re.I)
        title = next((node for node in article if name(node) == "TI.ART"), None)
        label = "".join(title.itertext()).strip() if title is not None else ""
        labelled = re.fullmatch(r"(?:Article|Artikel|Art[ií]culo)\s+(\d+)([a-z]*)", label, re.I)
        if identifier is not None:
            result = str(int(identifier[1])) + identifier[2].lower()
            if labelled is not None:
                declared = str(int(labelled[1])) + labelled[2].lower()
                if declared != result:
                    raise FormexArticleError("conflicting-formex-article-number")
            return result
        if labelled is not None:
            return str(int(labelled[1])) + labelled[2].lower()
        raise FormexArticleError("missing-formex-article-number")

    def walk(container, normalize):
        nonlocal changed
        for child in container:
            kind = name(child)
            if kind == "ARTICLE":
                identifiers.append(number(child))
                # A quoted replacement ARTICLE belongs to this article's body.
                continue
            if kind in {"TOC", "ANNEX", "TITLE", "PREAMBLE", "FINAL"}:
                continue
            operative = normalize and kind in {"DIVISION", "GR.SEQ"}
            if operative and kind == "GR.SEQ":
                child.tag = child.tag[:-len("GR.SEQ")] + "DIVISION"
                changed = True
            # Unknown containers remain unchanged, but their original article
            # IDs still enter the inventory so unsupported layouts fail closed.
            walk(child, operative)

    walk(body, True)
    if len(set(identifiers)) != len(identifiers):
        raise FormexArticleError("duplicate-formex-article-number")
    return (ElementTree.tostring(root, encoding="utf-8") if changed else xml), tuple(identifiers)


def _validate_profile_articles(result, expected, *, strict):
    actual = [str(article.get("article_number", "")).strip()
              for article in result.get("articles", [])]
    if (expected and (len(actual) != len(expected) or set(actual) != set(expected))
            or strict and not expected):
        raise FormexArticleError("formex-article-inventory-mismatch")


def _profile_parse(xml, destination, parser):
    """Bind a cached parse to its exact source XML before reusing it."""
    parser_xml, expected = _profile_article_xml(xml)
    digest = hashlib.sha256(xml).hexdigest()
    parser_identity = _parser_identity(parser)
    adapter_identity = _file_hash(Path(__file__))
    marker = destination.with_suffix(".source.json")
    if marker.exists() and destination.exists():
        try:
            metadata = json.loads(marker.read_text())
            if metadata == {"xml_sha256": digest, "parsed_sha256": _file_hash(destination),
                            "parser_identity": parser_identity,
                            "segmentation_sha256": adapter_identity}:
                cached = json.loads(destination.read_text())
                _validate_profile_articles(cached, expected, strict=parser is run_parse_fmx)
                return cached
        except (OSError, ValueError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".parsing")
    temporary.unlink(missing_ok=True)
    try:
        result = parser(parser_xml, temporary)
        if not isinstance(result, dict):
            return None
        _validate_profile_articles(result, expected, strict=parser is run_parse_fmx)
        cellar._atomic_json(destination, result)
        cellar._atomic_json(marker, {"xml_sha256": digest, "parsed_sha256": _file_hash(destination),
                                     "parser_identity": parser_identity,
                                     "segmentation_sha256": adapter_identity})
        return result
    finally:
        temporary.unlink(missing_ok=True)


def _profile_segment(celexes, *, workers, resume, profile, transport, parser):
    """Resume from complete per-act bundles, never from a partial aggregate."""
    profile.ensure_dirs()
    fetched = json.loads(profile.FETCH_MANIFEST_JSON.read_text())
    metadata = {}
    if profile.CANDIDATES_CSV.exists():
        with profile.CANDIDATES_CSV.open(encoding="utf-8") as handle:
            metadata = {row["celex_id"]: row for row in csv.DictReader(handle)}
    tool_identity = _parser_identity(parser or run_parse_fmx)

    def one(celex):
        entry = fetched.get(celex, {})
        sources = {"eli": entry.get("eli"), "parser": tool_identity,
                   "segmentation": _file_hash(Path(__file__))}
        for language in ACT_LANGUAGES:
            path = entry.get("languages", {}).get(language, {}).get("zip")
            sources[language] = _file_hash(path) if path and Path(path).exists() else ""
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        output = profile.PARSED_DIR / "acts" / celex[1:5] / f"{celex}.json"
        if resume and output.exists():
            try:
                saved = json.loads(output.read_text())
                rows = saved["rows"]
                row_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
                if (saved.get("fingerprint") == fingerprint and saved.get("complete") is True
                        and saved.get("rows_sha256") == row_hash
                        and {r["language"] for r in rows if r.get("unit_type") == "article"} == set(ACT_LANGUAGES)):
                    return rows, saved["summary"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
        rows, summary = segment_act(celex, entry, metadata.get(celex, {}), profile=profile,
                                    transport=transport, parser=parser)
        complete = (not summary.get("errors") and not summary.get("skipped")
                    and {r["language"] for r in rows if r.get("unit_type") == "article"} == set(ACT_LANGUAGES))
        cellar._atomic_json(output, {"fingerprint": fingerprint, "complete": complete,
                                     "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                                     "rows": rows, "summary": summary})
        return rows, summary

    article_tmp = profile.ARTICLES_JSONL.with_suffix(".jsonl.tmp")
    summary_path = profile.STRUCTURE_DIR / "segment_summary.jsonl"
    summary_tmp = summary_path.with_suffix(".jsonl.tmp")
    try:
        with article_tmp.open("w", encoding="utf-8") as article_handle, \
                summary_tmp.open("w", encoding="utf-8") as summary_handle, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            for rows, summary in pool.map(one, sorted(set(celexes))):
                for row in rows:
                    article_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                summary_handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
        article_tmp.replace(profile.ARTICLES_JSONL)
        summary_tmp.replace(summary_path)
    finally:
        article_tmp.unlink(missing_ok=True)
        summary_tmp.unlink(missing_ok=True)


def segment(celexes: list[str], *, workers: int = 12, resume: bool = True,
            profile=None, transport=None, parser=None) -> None:
    """Segment every act, streaming records to disk as they are produced.

    Output is streamed rather than accumulated: the full corpus is on the order
    of a million records, and holding them in memory to write once at the end
    would be gigabytes for no reason. Streaming also makes the run resumable --
    a re-run skips acts already present in the output.
    """
    if profile is not None:
        return _profile_segment(celexes, workers=workers, resume=resume, profile=profile,
                                transport=transport, parser=parser)
    paths.ensure_dirs()
    manifest = json.loads((paths.STRUCTURE_DIR / "fetch_manifest.json").read_text())

    meta_by_celex: dict[str, dict] = {}
    if paths.CANDIDATES_CSV.exists():
        with paths.CANDIDATES_CSV.open(encoding="utf-8") as fh:
            meta_by_celex = {row["celex_id"]: row for row in csv.DictReader(fh)}

    done: set[str] = set()
    if resume and paths.ARTICLES_JSONL.exists():
        with paths.ARTICLES_JSONL.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["celex_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[segment] resuming: {len(done):,} acts already segmented", flush=True)

    todo = [c for c in celexes if c not in done and c in manifest]
    print(f"[segment] {len(todo):,} acts to segment, {workers} workers", flush=True)

    summary_path = paths.STRUCTURE_DIR / "segment_summary.jsonl"
    started = time.monotonic()
    written = failed = 0

    mode = "a" if (resume and paths.ARTICLES_JSONL.exists()) else "w"
    with paths.ARTICLES_JSONL.open(mode, encoding="utf-8") as out, \
            summary_path.open(mode, encoding="utf-8") as summaries, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(segment_act, c, manifest[c], meta_by_celex.get(c, {})): c
                   for c in todo}
        for index, future in enumerate(as_completed(futures), 1):
            celex = futures[future]
            try:
                rows, summary = future.result()
            except Exception as error:
                failed += 1
                summaries.write(json.dumps({"celex": celex, "error": str(error)[:200]}) + "\n")
                continue
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += len(rows)
            summaries.write(json.dumps(summary) + "\n")
            if index % 200 == 0 or index == len(todo):
                out.flush()
                summaries.flush()
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0
                print(f"  {index:>6,}/{len(todo):,}  {written:,} records  "
                      f"{rate:.1f} acts/s  eta {(len(todo) - index) / rate / 3600:.1f}h"
                      if rate else f"  {index:,}/{len(todo):,}", flush=True)

    print(f"[segment] wrote {written:,} records ({failed} act failures) "
          f"-> {paths.ARTICLES_JSONL}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--from-year", type=int, default=2004)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("celex", nargs="*")
    args = parser.parse_args()
    celexes = list(args.celex)
    if args.pilot:
        with paths.PILOT_CSV.open(encoding="utf-8") as fh:
            celexes += [row["celex_id"] for row in csv.DictReader(fh)]
    if args.all:
        with paths.CANDIDATES_CSV.open(encoding="utf-8") as fh:
            celexes += [row["celex_id"] for row in csv.DictReader(fh)
                        if int(row["year"]) >= args.from_year]
    if not celexes:
        parser.error("give CELEX ids, --pilot or --all")
    segment(celexes, workers=args.workers, resume=not args.no_resume)


if __name__ == "__main__":
    main()
