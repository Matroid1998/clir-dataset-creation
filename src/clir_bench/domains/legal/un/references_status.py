"""
Stage: classify and resolve every citation in the UN corpus, persist the verdict.

``qac/un_references`` is the pure library -- it decides, per citation, whether
the surface pins exactly one target and names the reason when it does not.
This stage runs that decision over every target block once and writes it down,
so question generation can gate on it instead of re-deciding per payload:

    reference_status_en.jsonl   one row per target block WITH >= 1 citation.
                                A target block absent from the file cites
                                nothing and is complete by definition.
    un_citation_edges.jsonl     every resolved external citation, as an edge
                                (citing block -> cited document, with the
                                anchored paragraph/annex part when known)
    un_unresolved.jsonl         every citation that carries a reason --
                                blocking and merely-recorded alike
    reports/un_references/un_reference_stats.json
                                per-kind resolution table, per-reason counts,
                                symbol-map health, gate pool sizes

The stage also owns the human review harness (``--sample``), which used to
live in ``un_references`` itself.

Usage:
    python -m clir_bench.domains.legal.un.references_status
    python -m clir_bench.domains.legal.un.references_status --limit-docs 500
    python -m clir_bench.domains.legal.un.references_status --sample 400
    python -m clir_bench.domains.legal.un.references_status --audit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from clir_bench.domains.legal.un import paths
from clir_bench.domains.legal.qac import un_references as refs


def _genre(doc_id: str, title: str = "") -> str:
    if "/res/" in doc_id or "/dec/" in doc_id:
        return "resolution"
    if "sr_" in doc_id or "pv_" in doc_id:
        return "meeting"
    if title.lower().startswith("letter"):
        return "letter"
    return "other"


def _load_docs(docs_path: Path) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    with open(docs_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            docs[row["doc_id"]] = row
    return docs


def _doc_citations(doc_id: str, doc: dict, blocks: list[dict],
                   index: refs.SymbolIndex,
                   docs: dict[str, dict]) -> dict[int, list[refs.Citation]]:
    """Citations per target block index, fully classified and resolved."""
    texts = [b["text"] for b in blocks]
    parts = [b.get("part_id", "") for b in blocks]
    annexes = doc.get("annexes", [])
    out: dict[int, list[refs.Citation]] = {}
    for idx in doc.get("target_idxs", []):
        citations = refs.extract_citations(texts[idx], doc_id=doc_id)
        if not citations:
            continue
        refs.resolve_citations(citations, index, citing_doc_id=doc_id)
        refs.resolve_external_annexes(
            citations, lambda d: docs.get(d, {}).get("annexes", []))
        refs.resolve_internal(citations, block_index=idx, block_texts=texts,
                              block_parts=parts, annexes=annexes)
        out[idx] = citations
    return out


def run(*, blocks_path: Path | None = None, docs_path: Path | None = None,
        status_out: Path | None = None, edges_out: Path | None = None,
        unresolved_out: Path | None = None, stats_out: Path | None = None,
        limit_docs: int = 0) -> dict:
    blocks_path = blocks_path or paths.BLOCKS_JSONL
    docs_path = docs_path or paths.DOCS_JSONL
    status_out = status_out or paths.REFERENCE_STATUS_JSONL
    edges_out = edges_out or paths.CITATION_EDGES_JSONL
    unresolved_out = unresolved_out or paths.UNRESOLVED_JSONL
    stats_out = stats_out or paths.STATS_JSON
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    docs = _load_docs(docs_path)
    index = refs.SymbolIndex.from_docs(docs)
    if index.n_collisions:
        print(f"WARNING: {index.n_collisions} symbol-map collisions dropped as "
              "ambiguous", file=sys.stderr)

    stats: Counter = Counter()
    kinds_found: Counter = Counter()
    kinds_resolved: Counter = Counter()
    reasons: Counter = Counter()
    incomplete_by_genre: Counter = Counter()
    n_docs = n_targets = n_citing = n_complete_citing = 0

    with open(blocks_path, encoding="utf-8") as blocks_fh, \
            open(status_out, "w", encoding="utf-8") as status_fh, \
            open(edges_out, "w", encoding="utf-8") as edges_fh, \
            open(unresolved_out, "w", encoding="utf-8") as unresolved_fh:
        # Docs rows appear in blocks-file order, so one sequential read of the
        # blocks file serves every document without seeking.
        for doc_id, doc in docs.items():
            rows = [json.loads(blocks_fh.readline()) for _ in range(doc["n_blocks"])]
            n_docs += 1
            n_targets += len(doc.get("target_idxs", []))
            if limit_docs and n_docs >= limit_docs:
                per_block = _doc_citations(doc_id, doc, rows, index, docs)
                _write(doc_id, doc, per_block, status_fh, edges_fh, unresolved_fh,
                       stats, kinds_found, kinds_resolved, reasons,
                       incomplete_by_genre)
                n_citing += len(per_block)
                n_complete_citing += sum(
                    1 for cites in per_block.values()
                    if not any(refs.is_blocking(c) for c in cites))
                break
            if not doc.get("n_targets"):
                continue
            per_block = _doc_citations(doc_id, doc, rows, index, docs)
            n_citing += len(per_block)
            n_complete_citing += sum(
                1 for cites in per_block.values()
                if not any(refs.is_blocking(c) for c in cites))
            _write(doc_id, doc, per_block, status_fh, edges_fh, unresolved_fh,
                   stats, kinds_found, kinds_resolved, reasons, incomplete_by_genre)

    summary = {
        "n_docs": n_docs,
        "n_target_blocks": n_targets,
        "n_citing_target_blocks": n_citing,
        "n_no_citation": n_targets - n_citing,
        "n_complete_with_citations": n_complete_citing,
        "n_incomplete": n_citing - n_complete_citing,
        "complete_share_of_targets":
            round((n_targets - n_citing + n_complete_citing) / max(n_targets, 1), 4),
        "citations_found": sum(kinds_found.values()),
        "citations_resolved": sum(kinds_resolved.values()),
        "kinds": {k: {"found": v, "resolved": kinds_resolved.get(k, 0)}
                  for k, v in kinds_found.most_common()},
        "reasons": dict(reasons.most_common()),
        "incomplete_by_genre": dict(incomplete_by_genre),
        "symbol_map": {
            "keys": len(index.map),
            "collisions_dropped": index.n_collisions,
            "dirty_keys_normalised": index.n_dirty_keys,
            "ga_special_sessions": sorted(index.ga_special_sessions),
            "hrc_special_sessions": sorted(index.hrc_special_sessions),
            "special_session_overlap": sorted(
                index.ga_special_sessions & index.hrc_special_sessions),
        },
        **{f"rows:{k}": v for k, v in sorted(stats.items())},
    }
    stats_out.write_text(json.dumps(summary, indent=2))
    return summary


def _write(doc_id, doc, per_block, status_fh, edges_fh, unresolved_fh,
           stats, kinds_found, kinds_resolved, reasons, incomplete_by_genre) -> None:
    for idx, citations in per_block.items():
        block_id = f"{doc_id}#{idx}"
        status = refs.citation_status(citations)
        status_fh.write(json.dumps(
            {"block_id": block_id, "doc_id": doc_id, **status},
            ensure_ascii=False) + "\n")
        stats["status"] += 1
        if not status["complete"]:
            incomplete_by_genre[_genre(doc_id, doc.get("title", ""))] += 1
        for c in citations:
            kinds_found[c.kind] += 1
            clean = c.reason is None and (
                c.doc_id or c.scope != refs.SCOPE_EXTERNAL)
            if clean:
                kinds_resolved[c.kind] += 1
            if c.reason:
                reasons[c.reason] += 1
                unresolved_fh.write(json.dumps({
                    "block_id": block_id, "doc_id": doc_id, "kind": c.kind,
                    "scope": c.scope, "raw": c.raw, "char_start": c.start,
                    "char_end": c.end, "reason": c.reason,
                    "blocking": refs.is_blocking(c)}, ensure_ascii=False) + "\n")
                stats["unresolved"] += 1
            if c.doc_id and c.reason is None:
                edges_fh.write(json.dumps({
                    "block_id": block_id, "doc_id": doc_id,
                    "target_doc_id": c.doc_id, "symbol": c.symbol,
                    "kind": c.kind, "paragraph": c.paragraph,
                    "part_id": c.part_id, "char_start": c.start,
                    "char_end": c.end}, ensure_ascii=False) + "\n")
                stats["edges"] += 1


# --------------------------------------------------------------------------- #
# Review harness (moved here from un_references) and audit
# --------------------------------------------------------------------------- #

def _context(text: str, start: int, end: int, width: int = 80) -> str:
    before = text[max(0, start - width):start].replace("\n", " ")
    after = text[end:end + width].replace("\n", " ")
    return f"…{before}«{text[start:end]}»{after}…"


def write_review_sample(n: int, seed: int, out: Path) -> None:
    import random
    from clir_bench.domains.legal.qac.un_context import BlockIndex

    random.seed(seed)
    index = BlockIndex()
    symbols = refs.SymbolIndex.from_docs(index.docs)
    doc_ids = [d for d, row in index.docs.items() if row.get("n_targets", 0)]
    random.shuffle(doc_ids)

    kinds: Counter = Counter()
    resolved_kinds: Counter = Counter()
    outcome: Counter = Counter()
    sections: list[str] = []
    blocks_seen = 0
    for doc_id in doc_ids:
        if len(sections) >= n:
            break
        doc = index.docs[doc_id]
        blocks = index.blocks_for(doc_id)
        texts = [b.texts["en"] for b in blocks]
        parts = [b.part_id for b in blocks]
        for idx in doc.get("target_idxs", []):
            if len(sections) >= n:
                break
            blocks_seen += 1
            cites = refs.extract_citations(texts[idx], doc_id=doc_id)
            if not cites:
                continue
            refs.resolve_citations(cites, symbols, citing_doc_id=doc_id)
            refs.resolve_external_annexes(
                cites, lambda d: index.docs.get(d, {}).get("annexes", []))
            refs.resolve_internal(cites, block_index=idx, block_texts=texts,
                                  block_parts=parts,
                                  annexes=doc.get("annexes", []))
            lines = [f"## {doc_id}#{idx}", ""]
            for c in cites:
                kinds[c.kind] += 1
                if c.doc_id:
                    resolved_kinds[c.kind] += 1
                    outcome["resolved"] += 1
                    title = index.docs[c.doc_id]["title"][:90]
                    status = f"→ **{c.symbol}** — {title}"
                    if c.paragraph is not None:
                        status += f"  ·  ¶{c.paragraph}"
                    if c.part_id:
                        status += f"  ·  {c.part_id}"
                elif c.scope == refs.SCOPE_INTERNAL and c.reason is None:
                    resolved_kinds[c.kind] += 1
                    outcome["internal_resolved"] += 1
                    status = f"→ INTERNAL {c.anchor_blocks or '(self)'}"
                    if c.part_id:
                        status += f" {c.part_id}"
                else:
                    outcome[c.reason or "unresolved"] += 1
                    status = f"→ {'BLOCKS' if refs.is_blocking(c) else 'recorded'} ({c.reason})"
                lines.append(f"- `{c.kind}` {_context(texts[idx], c.start, c.end)}")
                lines.append(f"  {status}")
            lines.append("")
            sections.append("\n".join(lines))

    total = sum(kinds.values())
    resolved = sum(resolved_kinds.values())
    header = [
        "# UN cross-reference classification — manual review sample", "",
        f"Scanned {blocks_seen:,} target blocks (seed {seed}); {len(sections)} "
        f"citing blocks shown; {total} citations, {resolved} resolved "
        f"({resolved / max(total, 1):.0%}).", "",
        "| kind | found | resolved |", "|---|---|---|",
    ]
    header += [f"| {k} | {v} | {resolved_kinds.get(k, 0)} |"
               for k, v in kinds.most_common()]
    header.append("\n| outcome | n |\n|---|---|")
    header += [f"| {k} | {v} |" for k, v in outcome.most_common()]
    out.write_text("\n".join(header) + "\n\n" + "\n".join(sections),
                   encoding="utf-8")
    print(f"wrote {len(sections)} citing blocks -> {out}", file=sys.stderr)


def audit() -> None:
    docs = _load_docs(paths.DOCS_JSONL)
    index = refs.SymbolIndex.from_docs(docs)
    print(f"symbols               {len(index.map):,}")
    print(f"collisions dropped    {index.n_collisions:,}")
    print(f"dirty keys normalised {index.n_dirty_keys:,}")
    print(f"GA special sessions   {sorted(index.ga_special_sessions)}")
    print(f"HRC special sessions  {sorted(index.hrc_special_sessions)}")
    print(f"session overlap       {sorted(index.ga_special_sessions & index.hrc_special_sessions)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0,
                        help="write a manual-review markdown instead of the full run")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--audit", action="store_true",
                        help="print symbol-map health only")
    args = parser.parse_args()
    paths.ensure_dirs()

    if args.audit:
        audit()
        return
    if args.sample:
        write_review_sample(args.sample, args.seed, paths.REVIEW_MD)
        return
    summary = run(limit_docs=args.limit_docs)
    for key in ("n_target_blocks", "n_citing_target_blocks", "n_no_citation",
                "n_complete_with_citations", "n_incomplete",
                "complete_share_of_targets", "citations_found",
                "citations_resolved"):
        print(f"{key:<28} {summary[key]:,}" if isinstance(summary[key], int)
              else f"{key:<28} {summary[key]}")
    print(f"-> {paths.REFERENCE_STATUS_JSONL}")
    print(f"-> {paths.STATS_JSON}")


if __name__ == "__main__":
    main()
