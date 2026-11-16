"""The status stage: one pass, four artifacts, absence-means-complete."""

from __future__ import annotations

import json

from clir_bench.domains.legal.un import references_status as st


def _doc(doc_id, symbol, blocks, annexes=(), targets=None):
    return ({"doc_id": doc_id, "symbol": symbol, "title": f"title {symbol}",
             "n_blocks": len(blocks), "n_targets": len(targets or range(len(blocks))),
             "target_idxs": list(targets or range(len(blocks))),
             "annexes": list(annexes), "offset": 0},
            [{"block_id": f"{doc_id}#{i}", "doc_id": doc_id, "text": t,
              "part_id": ""} for i, t in enumerate(blocks)])


def _corpus(tmp_path):
    docs = []
    blocks = []
    d, b = _doc("1999/s/res/1265_1999_", "S/RES/1265(1999)",
                ["1. Decides to protect civilians in armed conflict;"])
    docs.append(d); blocks += b
    # complete: cites a resolvable resolution
    d, b = _doc("2000/s/res/1296_2000_", "S/RES/1296(2000)",
                [("Recalling its resolution 1265 (1999), and paragraph 1 above,\n"
                  "1. Reaffirms the framework;")])
    docs.append(d); blocks += b
    # incomplete: treaty article
    d, b = _doc("2001/a/56/100_", "A/56/100",
                ["Considering Article 19 of the Covenant, the Committee met"])
    docs.append(d); blocks += b
    # no citations at all
    d, b = _doc("2002/a/57/1_", "A/57/1",
                ["The Assembly held a general debate on many topics today"])
    docs.append(d); blocks += b
    docs_path = tmp_path / "docs.jsonl"
    blocks_path = tmp_path / "blocks.jsonl"
    docs_path.write_text("".join(json.dumps(r) + "\n" for r in docs))
    blocks_path.write_text("".join(json.dumps(r) + "\n" for r in blocks))
    return docs_path, blocks_path


def test_run_writes_all_artifacts_with_correct_verdicts(tmp_path):
    docs_path, blocks_path = _corpus(tmp_path)
    outs = {k: tmp_path / f"{k}.jsonl" for k in ("status", "edges", "unresolved")}
    summary = st.run(blocks_path=blocks_path, docs_path=docs_path,
                     status_out=outs["status"], edges_out=outs["edges"],
                     unresolved_out=outs["unresolved"],
                     stats_out=tmp_path / "stats.json")
    status = {json.loads(l)["block_id"]: json.loads(l)
              for l in outs["status"].read_text().splitlines()}
    # the no-citation block is ABSENT (absence == complete)
    assert "2002/a/57/1_#0" not in status
    assert status["2000/s/res/1296_2000_#0"]["complete"] is True
    assert status["2000/s/res/1296_2000_#0"]["n_external_resolved"] == 1
    assert status["2001/a/56/100_#0"]["complete"] is False
    assert status["2001/a/56/100_#0"]["reasons"] == {"treaty_article_unmodelled": 1}
    edges = [json.loads(l) for l in outs["edges"].read_text().splitlines()]
    assert [(e["block_id"], e["target_doc_id"]) for e in edges] == [
        ("2000/s/res/1296_2000_#0", "1999/s/res/1265_1999_")]
    unresolved = [json.loads(l) for l in outs["unresolved"].read_text().splitlines()]
    assert any(u["reason"] == "treaty_article_unmodelled" and u["blocking"]
               for u in unresolved)
    assert summary["n_incomplete"] == 1
    assert summary["symbol_map"]["collisions_dropped"] == 0
    assert (tmp_path / "stats.json").exists()


def test_internal_anchor_counts_as_resolved(tmp_path):
    docs_path, blocks_path = _corpus(tmp_path)
    outs = {k: tmp_path / f"{k}.jsonl" for k in ("status", "edges", "unresolved")}
    st.run(blocks_path=blocks_path, docs_path=docs_path,
           status_out=outs["status"], edges_out=outs["edges"],
           unresolved_out=outs["unresolved"], stats_out=tmp_path / "stats.json")
    row = next(json.loads(l) for l in outs["status"].read_text().splitlines()
               if l and json.loads(l)["block_id"] == "2000/s/res/1296_2000_#0")
    # "paragraph 1 above" anchors to the block's own paragraph 1 line
    assert row["n_internal_resolved"] == 1 and row["n_internal_unresolved"] == 0


def test_wrong_body_cannot_create_an_edge_or_pass_the_completeness_gate(tmp_path):
    ga, ga_blocks = _doc("1996/a/res/51/1", "A/RES/51/1",
                         ["1. Decides to grant observer status;"])
    citing, citing_blocks = _doc("2023/a/78/1", "A/78/1",
                                 ["Recalling Human Rights Council resolution 51/1,"])
    docs_path, blocks_path = tmp_path / "docs.jsonl", tmp_path / "blocks.jsonl"
    docs_path.write_text("".join(json.dumps(r) + "\n" for r in [ga, citing]))
    blocks_path.write_text("".join(json.dumps(r) + "\n"
                                   for r in ga_blocks + citing_blocks))
    outs = {k: tmp_path / f"{k}.jsonl" for k in ("status", "edges", "unresolved")}

    st.run(blocks_path=blocks_path, docs_path=docs_path,
           status_out=outs["status"], edges_out=outs["edges"],
           unresolved_out=outs["unresolved"], stats_out=tmp_path / "stats.json")

    (status,) = [json.loads(line) for line in outs["status"].read_text().splitlines()]
    assert status["block_id"] == "2023/a/78/1#0"
    assert status["complete"] is False
    assert status["n_external_resolved"] == 0
    assert outs["edges"].read_text() == ""
    (unresolved,) = [json.loads(line) for line in outs["unresolved"].read_text().splitlines()]
    assert unresolved["reason"] == "out_of_corpus" and unresolved["blocking"]
