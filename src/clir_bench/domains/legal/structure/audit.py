"""
Stage 7 -- validation: an audit sample, a recall probe, and a second opinion.

Three independent checks, because each catches something the others cannot:

*Precision* -- a stratified sample of internal edges, each written out with the
surrounding text so the target can be confirmed by reading. Stratified across
acts and across the rule that fired, so no single rule can hide behind volume.

*Recall* -- the extractor consumes every ``Article N`` head in the text, so recall
inside that pattern is 100% by construction and measuring it would prove nothing.
What can actually be missed is a reference that never uses the word: ``that
Article``, ``the said Article``, a bare ``Art. 5``. The probe counts those
directly.

*A second opinion* -- ``noworneverev/eurlex-parser`` scrapes EUR-Lex HTML and
reports per-article external act references. It reports no article numbers and no
internal edges, so it can only corroborate the external file, but that is an
independent pipeline over an independent source format, which makes disagreement
informative.

Usage:
    python -m clir_bench.domains.legal.structure.audit sample --n 100
    python -m clir_bench.domains.legal.structure.audit recall
    python -m clir_bench.domains.legal.structure.audit crosscheck --python <interpreter>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict

from clir_bench.domains.legal.structure import paths

CONTEXT = 110


def _load(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _article_text(*, profile=None) -> dict[tuple[str, str], str]:
    profile_paths = profile or paths
    out = {}
    for row in _load(profile_paths.ARTICLES_JSONL):
        if row["unit_type"] == "article" and row["language"] == "en":
            out[(row["celex_id"], row["article_number"])] = row["text"]
    return out


# -- precision sample ------------------------------------------------------- #

def sample(n: int, seed: int, *, profile=None) -> None:
    profile_paths = profile or paths
    edges = _load(profile_paths.INTERNAL_EDGES_JSONL)
    texts = _article_text(profile=profile) if profile else _article_text()

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for edge in edges:
        strata[(edge["source_celex"], edge["expansion"])].append(edge)

    def rank(edge):
        key = f"{seed}:{edge['source_article_id']}:{edge['target_article_number']}:{edge['char_start']}"
        return hashlib.sha256(key.encode()).hexdigest()

    for bucket in strata.values():
        bucket.sort(key=rank)

    picked, depth, order = [], 0, sorted(strata)
    while len(picked) < n:
        progressed = False
        for key in order:
            if len(picked) >= n:
                break
            if depth < len(strata[key]):
                picked.append(strata[key][depth])
                progressed = True
        if not progressed:
            break
        depth += 1

    out = profile_paths.REPORTS_DIR / "audit_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["celex", "source_article", "target_article", "rule",
                         "expansion", "surface_form", "source_is_amending",
                         "context", "verdict", "note"])
        for edge in picked:
            text = texts.get((edge["source_celex"], edge["source_article_number"]), "")
            lo = max(0, edge["char_start"] - CONTEXT)
            hi = min(len(text), edge["char_end"] + CONTEXT)
            context = re.sub(r"\s+", " ", text[lo:hi])
            writer.writerow([
                edge["source_celex"], edge["source_article_number"],
                edge["target_article_number"], edge["rule"], edge["expansion"],
                edge["surface_form"], edge.get("source_is_amending", False),
                context, "", "",
            ])
    print(f"wrote {out}  ({len(picked)} edges)")
    print("  by rule     :", dict(Counter(e["rule"] for e in picked)))
    print("  by expansion:", dict(Counter(e["expansion"] for e in picked)))
    print("  by act      :", len({e["source_celex"] for e in picked}), "acts")


# -- recall probe ----------------------------------------------------------- #

# Reference-shaped phrases that never contain "Article <number>", so the
# extractor cannot see them by construction.
_MISSABLE = {
    "anaphoric-this-article": r"\bthis\s+Article\b",
    "anaphoric-that-article": r"\b(that|the\s+said|the\s+same)\s+Article\b",
    "abbreviated-art": r"\bArt\.\s*\d+",
    "preceding-following": r"\b(preceding|following)\s+Article\b",
    "paragraph-only": r"\bparagraphs?\s+\d+",
    "annex-reference": r"\bAnnexe?s?\s+[IVXLC]+\b",
}


def recall(*, profile=None) -> None:
    profile_paths = profile or paths
    from clir_bench.domains.legal.structure import references as refs

    texts = _article_text(profile=profile) if profile else _article_text()
    blob = "\n".join(texts.values())

    # Account for every "Article <n>" head in the corpus. A head is either the
    # start of an enumeration or absorbed into one that began earlier -- a long
    # citation chain repeats the word ("Articles 110 and 111, Article 125(1),
    # ...") and is deliberately parsed as a single enumeration. Comparing head
    # count against enumeration count without this split reads like a 30% miss
    # rate when nothing has been missed at all.
    total_heads = len(re.findall(r"\bArticles?\s+\d", blob, re.I))
    enumerations = absorbed = 0
    for text in texts.values():
        found = refs.find_enumerations(text)
        enumerations += len(found)
        for enumeration in found:
            absorbed += len(re.findall(r"\bArticles?\s+\d", enumeration["raw"], re.I)) - 1

    print(f"English article text                 {len(blob):>10,} chars")
    print(f"'Article <n>' heads in text          {total_heads:>10,}")
    print(f"  heads starting an enumeration      {enumerations:>10,}")
    print(f"  heads absorbed into a longer chain {absorbed:>10,}")
    print(f"  unaccounted for                    {total_heads - enumerations - absorbed:>10,}"
          "   <- genuine misses of the 'Article N' pattern")
    print("\nreference-shaped phrases the extractor cannot match by construction:")
    rows = []
    for name, pattern in _MISSABLE.items():
        count = len(re.findall(pattern, blob, re.I))
        rows.append({"pattern": name, "count": count})
        print(f"  {name:<26} {count:>6,}")
    print("\n'paragraph N' and 'Annex N' are out of scope by design -- this pipeline")
    print("resolves at article granularity only. The first four rows are the real")
    print("recall ceiling for article-to-article edges.")

    (profile_paths.REPORTS_DIR / "recall_probe.json").write_text(json.dumps({
        "chars": len(blob), "article_heads": total_heads,
        "enumerations": enumerations, "heads_absorbed": absorbed,
        "unaccounted": total_heads - enumerations - absorbed,
        "missable": rows,
    }, indent=2))


# -- external cross-check --------------------------------------------------- #

_ACT_KEY_RE = re.compile(
    r"(Regulation|Directive|Decision)\s*(?:\(\s*(?:EU|EC|EEC|Euratom)[^)]*\)\s*)?"
    r"(?:No\s*)?(\d+/\d+)", re.I)


def act_keys(surface: str) -> set[str]:
    """Normalise an act citation to {type}:{number/year} for comparison."""
    return {f"{m.group(1).lower()}:{m.group(2)}" for m in _ACT_KEY_RE.finditer(surface or "")}


def crosscheck(python_bin: str, *, profile=None) -> None:
    profile_paths = profile or paths
    ours = _load(profile_paths.EXTERNAL_REFS_JSONL)
    by_article: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in ours:
        by_article[(row["source_celex"], row["source_article_number"])] |= \
            act_keys(row["target_act_surface"])

    celexes = sorted({row["source_celex"] for row in ours})
    script = r"""
import json, sys, warnings
warnings.filterwarnings('ignore')
import eurlex
out = {}
for celex in sys.argv[1:]:
    try:
        data = eurlex.get_data_by_celex_id(celex, language='en')
    except Exception as exc:
        out[celex] = {'_error': str(exc)[:200]}
        continue
    out[celex] = {
        str(a.get('id', '')).replace('Article', '').strip(): list(a.get('references') or [])
        for a in data.get('articles', [])
    }
print(json.dumps(out))
"""
    proc = subprocess.run([python_bin, "-c", script, *celexes],
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise SystemExit(f"eurlex-parser run failed:\n{proc.stderr[-2000:]}")
    theirs_raw = json.loads(proc.stdout.strip().splitlines()[-1])

    texts = _article_text(profile=profile) if profile else _article_text()
    agree = only_ours = only_theirs_bare = only_theirs_qualified = 0
    failed = []
    for celex in celexes:
        entry = theirs_raw.get(celex) or {}
        if "_error" in entry:
            failed.append(celex)
            continue
        articles = set(entry) | {a for (c, a) in by_article if c == celex}
        for article in articles:
            mine = by_article.get((celex, article), set())
            yours: set[str] = set()
            for surface in entry.get(article, []):
                yours |= act_keys(surface)
            agree += len(mine & yours)
            only_ours += len(mine - yours)
            # The two tools do not have the same scope. eurlex-parser lists every
            # act *named anywhere* in an article; this pipeline records only acts
            # that govern an article citation ("Article 5 of Directive X"). So a
            # citation only they report is a real disagreement solely when the
            # text actually qualifies that act with an article.
            text = texts.get((celex, article), "")
            for key in yours - mine:
                number = key.split(":")[1]
                qualified = re.search(
                    rf"Articles?\s+[\d(),\s]+\s*(?:of|thereof)[^.;]{{0,60}}{re.escape(number)}",
                    text, re.I)
                if qualified:
                    only_theirs_qualified += 1
                else:
                    only_theirs_bare += 1

    ours_total = agree + only_ours
    corroborated = agree / ours_total if ours_total else 0.0
    print(f"acts compared                         {len(celexes) - len(failed)}"
          f"{f' ({len(failed)} failed)' if failed else ''}")
    print(f"identifiable act citations we report  {ours_total:,}")
    print(f"  corroborated by eurlex-parser       {agree:,}  ({corroborated:.1%})")
    print(f"  reported only by us                 {only_ours:,}")
    print(f"reported only by eurlex-parser        {only_theirs_bare + only_theirs_qualified:,}")
    print(f"  bare act mention, no article given  {only_theirs_bare:,}  (out of scope here)")
    print(f"  article-qualified, missed by us     {only_theirs_qualified:,}  <- real misses")
    (profile_paths.REPORTS_DIR / "external_crosscheck.json").write_text(json.dumps(
        {"corroborated_rate": corroborated, "agree": agree, "only_ours": only_ours,
         "only_theirs_bare": only_theirs_bare,
         "only_theirs_article_qualified": only_theirs_qualified,
         "failed": failed}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample"); p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260803)
    sub.add_parser("recall")
    c = sub.add_parser("crosscheck")
    c.add_argument("--python", default=sys.executable,
                   help="interpreter with eurlex-parser installed")
    args = parser.parse_args()

    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "sample":
        sample(args.n, args.seed)
    elif args.command == "recall":
        recall()
    else:
        crosscheck(args.python)


if __name__ == "__main__":
    main()
