"""One-shot generation, out-of-order batch replies, and quota refill."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from dataclasses import asdict, replace
from datetime import date

import pytest
import requests

from clir_bench.cli import main, qac_batch_content, qac_batch_sources
from clir_bench.cli.qac_batch_eurlex import _official_inventory
from clir_bench.core import batch_api, grading
from clir_bench.domains.legal.qac.evidence import EvidenceSource
from clir_bench.domains.legal.structure import cellar, eurlex_resources
from clir_bench.domains.legal.structure.profile import ProfileArticleIndex
from test_eurlex_resources import baseline, fake_parser, formex_zip
from test_legal_batch_replay_integration import case


def ready_legal(source, item, language, mode):
    _, _, _, example, _ = case(source)
    target = replace(item.value, language=language, mode=mode)
    if source == "eurlex":
        body = example.target.texts["en"]
        unit = replace(example.target, eli_id=item.key, celex_id=item.parent,
                       texts={"en": body, language: body})
        ref, kind = target.article_number, "article"
    else:
        body = example.target.texts["en"]
        unit = replace(example.target, block_id=item.key, doc_id=item.parent, line_start=1,
                       texts={"en": body, language: body})
        ref, kind = "target", "block"
    spans = {(ref, lang): EvidenceSource(
        ref, item.key, lang, body, body, kind, item.parent, 1, 2
    ) for lang in unit.texts}
    options = {"target": unit, "references": [], "text": body, "evidence_sources": spans}
    if source == "eurlex":
        options["dropped_references"] = []
    payload = replace(example, **options)
    return asdict(target), qac_batch_content.freeze_payload(source, payload, language), {}


class Provider:
    def __init__(self, source):
        self.source = source
        self.jobs = []
        self.requests = Counter()
        self.failed_grade = False

    def __call__(self, method, path="", body=None):
        if method == "POST":
            parsed = json.loads(body)
            assert parsed["endpoint"] == "/v1/chat/completions"
            self.jobs.append(parsed)
            return {"id": f"batch-{len(self.jobs)}"}
        number = int(path.split("-")[-1])
        request_data = self.jobs[number - 1]
        results = []
        for request in reversed(request_data["requests"]):
            key = request["custom_id"]
            stage = key.split("-", 1)[0]
            self.requests[key] += 1
            system, user = [message["content"] for message in request["body"]["messages"]]
            if stage == "generation":
                result = self.generation(system, user, key)
            elif stage == "faithfulness":
                result = self.faithfulness(user)
            elif not self.failed_grade:
                self.failed_grade = True
                result = {"malformed": True}
            else:
                result = self.quality(system, user)
            results.append({
                "custom_id": key, "error": None, "response": {
                    "status_code": 200, "body": {
                        "choices": [{"message": {"content": json.dumps(result)}}],
                        "usage": {"prompt_tokens": 42, "completion_tokens": 18, "cost": 0.01},
                    },
                },
            })
        return {"id": path[1:], "status": "completed", "results": results}

    def generation(self, system, user, key):
        snippets = json.loads(user.split("Verbatim evidence excerpts:\n", 1)[1])
        excerpt = snippets[0]
        if self.source == "eurlex":
            assert "exactly 3 distinct" in system
            size = 2 if sum(name.startswith("generation-") for name in self.requests) == 1 else 3
        else:
            size = 1
        rows = []
        for index in range(size):
            row = {
                "question": f"Which entity performs duty {index}?",
                "answer": excerpt["text"].split()[0], "answer_is_translation": False,
                "anchor": "entity",
            }
            if self.source == "gp":
                row.update(evidence=excerpt["text"], evidence_language=excerpt["language"],
                           question_type="reported_outcome")
            else:
                ref = excerpt["source_ref"]
                row["evidence_spans"] = [
                    {"source_ref": ref, "language": excerpt["language"], "text": row["answer"]}
                ]
                if self.source == "eurlex":
                    row["articles_involved"] = [ref]
            rows.append(row)
        return rows

    def faithfulness(self, user):
        if self.source == "gp":
            assert json.loads(user)["candidates"][0]["evidence"]
            count = 1
        else:
            assert "evidence_spans" in user
            count = 3 if self.source == "eurlex" else 1
        return [{"index": i, "grounding": 4, "precision": 4, "numerical_fidelity": 4,
                 "reason": "The evidence supports the answer."} for i in range(count)]

    def quality(self, system, user):
        candidates = json.loads(user)["candidates"]
        keys = grading.rubric_keys(system)
        return {
            "candidates": [{
                "index": index, "candidate_id": candidate["candidate_id"],
                "scores": dict.fromkeys(keys, 4 if index == 1 else 3),
                "score_notes": dict.fromkeys(keys, "Supported by the supplied source."),
                "checks": dict.fromkeys(("mode", "support", "metadata"), "pass"), "problems": [],
            } for index, candidate in enumerate(candidates)],
            "batch_diversity": "pass" if len(candidates) > 1 else "not_applicable",
        }


@pytest.mark.parametrize("source,questions", [("eurlex", 6), ("un", 6), ("gp", 7)])
def test_batch_cli_uses_existing_parsers_and_finishes_quotas(tmp_path, monkeypatch, source, questions):
    monkeypatch.setenv("CLIR_PROMPT_SOURCE", "local")
    monkeypatch.delenv("CLIR_PROMPT_MANIFEST", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-batch-key")
    run_dir = tmp_path / source
    provider = Provider(source)
    monkeypatch.setattr(batch_api, "openrouter", provider)

    prototype = case(source)[2] if source != "gp" else None
    rounds = 0

    def groups(context, config, profiles, used, scan_limit):
        nonlocal rounds
        rounds += 1
        languages = tuple(config["languages"])
        items = []
        for i in range(24):
            if rounds == 1 and source == "eurlex" and i not in {0, 1, 2, 12, 13, 14}:
                continue
            key = f"{source}-target-{i}"
            if key in used:
                continue
            published = "2021-01-01" if i < 12 else "2020-12-31"
            if source == "eurlex":
                parent = "32021R0001" if i < 12 else "32020R0001"
                value = replace(prototype, eli_id=key, celex_id=parent)
            elif source == "gp":
                parent = key
                value = {lang: {"language": lang, "publication_number": parent,
                                "id": f"{key}_{lang}", "context": "Quartz is the disclosed material.",
                                "abstract": "Quartz is the disclosed material."}
                         for lang in languages}
            else:
                parent = prototype.doc_id
                value = replace(prototype, block_id=key)
            items.append(qac_batch_sources.SourceTarget(key, parent, published, languages, value))
        yield items, None

    def prepare(context, corpus, item, index, language, mode):
        if corpus == "gp":
            return {}, qac_batch_content.patent_payload(context, item.value[language]), {}
        return ready_legal(corpus, item, language, mode)

    monkeypatch.setattr(qac_batch_sources, "groups", groups)
    monkeypatch.setattr(qac_batch_sources, "assess", lambda *args, **kwargs: {
        "selected_modes": ["conceptual" if source == "un" else "lookup"],
        "probabilities_yes": {"conceptual" if source == "un" else "lookup": 1},
        "threshold": 0.5, "model": "test/jev",
    })
    monkeypatch.setattr(qac_batch_sources, "prepare", prepare)
    monkeypatch.setattr(qac_batch_sources, "load_un_languages", lambda *args: None)

    prefix = ["--domain", "chemistry" if source == "gp" else "legal", "qac", "batch"]
    route = ["--source", source, "--questions", str(questions), "--workers", "2"]
    if source == "eurlex":
        profile = tmp_path / "manifest.json"
        profile.write_text("{}")
        route += ["--no-act-cap", "--candidates-per-target", "3",
                  "--generation-attempts-per-target", "1", "--eurlex-reuse-profile", str(profile)]
    if source != "un":
        route += ["--recent-since", "2021-01-01", "--min-recent-share", "0.5"]
    destination = ["--run-dir", str(run_dir)]
    poll = ["--poll-seconds", "0.001"]

    assert main(prefix + ["route"] + route + destination) == 0
    assert main(prefix + ["generate"] + destination + poll) == 0
    assert main(prefix + ["generate"] + destination + poll) == 0
    assert main(prefix + ["verify", "--fill"] + destination + poll) == 0
    jobs = len(provider.jobs)
    assert main(prefix + ["verify", "--fill"] + destination + poll) == 0
    assert len(provider.jobs) == jobs
    assert all(count == 1 for count in provider.requests.values())
    assert sum(key.endswith("-2") for key in provider.requests) == 1
    assert next(key for key in provider.requests if key.endswith("-2")).startswith("quality-")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["complete"] and summary["kept"] == questions
    assert set(summary["by_language"].values()) <= {1, 2}
    if source != "un":
        assert summary["recent_questions"] * 2 >= questions
    with (run_dir / "best.csv").open() as stream:
        winners = list(csv.DictReader(stream))
    assert len({row["target_id"] for row in winners}) == questions
    assert all(json.loads(row["evidence_spans"]) for row in winners)
    if source == "eurlex":
        assert rounds > 1
        assert max(Counter(row["document_id"] for row in winners).values()) > 2
        assert all(row["question"].endswith("1?") for row in winners)
        with sqlite3.connect(run_dir / "run.sqlite") as db:
            assert db.execute("SELECT count(*) FROM targets WHERE state='failed'").fetchone()[0] == 1


def test_official_snapshot_pins_post_2022_acts(tmp_path, monkeypatch):
    def discover(start, end):
        assert start.year == end.year == 2024
        return {"results": {"bindings": [{
            "celex": {"value": "32024R0123"}, "date": {"value": "2024-02-01"},
        }]}}

    monkeypatch.setattr("clir_bench.cli.qac_batch_eurlex._discover", discover)
    manifest = _official_inventory(tmp_path / "official", "2024-01-01", date(2024, 3, 1), 42)
    assert eurlex_resources.audit(manifest)["canonical_acts"] == 1
    row = json.loads((manifest.parent / "inventory.jsonl").read_text())
    assert row["date"] == "2024-02-01"
    assert row["celex_id"] == "32024R0123"

    act = "http://data.europa.eu/eli/reg/2024/123/oj"
    names = {name: code for code, name in cellar.LANG_CODE.items()}

    def transport(url, accept, *, language=None):
        if language is None:
            return f"<identifiers><uri>{act}</uri></identifiers>".encode()
        return formex_zip(row["celex_id"], names[language])

    original = tmp_path / "baseline"
    baseline(original)
    profile = tmp_path / "prepared"
    eurlex_resources.prepare(
        manifest, original, profile, pilot_size=1, workers=1,
        parser=fake_parser([]), transport=transport,
    )
    index = ProfileArticleIndex(profile / "manifest.json")
    assert index.target_acts == {"32024R0123"}
    identifier = cellar.article_eli(act, "1")
    assert index.is_target_eligible(identifier)


def test_lost_submission_response_can_resume_without_a_duplicate(monkeypatch):
    submitted = []

    def provider(method, path="", body=None):
        if method == "POST":
            submitted.append(json.loads(body))
            raise requests.ConnectionError("connection lost after submit")
        assert path == "/remote-1"
        return {"id": "remote-1", "status": "completed", "results": [{
            "custom_id": "request-1", "response": {"status_code": 200, "body": {"choices": []}},
        }]}

    monkeypatch.setattr(batch_api, "openrouter", provider)
    db = sqlite3.connect(":memory:")
    jobs = batch_api.BatchJobs(db)
    with db:
        jobs.add("request-1", "generation", "anthropic/claude-haiku-5.5", {"messages": []})
    with pytest.raises(RuntimeError, match="unresolved"):
        jobs.execute(stages=("generation",), poll_seconds=0.001)
    with pytest.raises(RuntimeError, match="recover-job"):
        jobs.execute(stages=("generation",), poll_seconds=0.001)
    jobs.recover(1, "remote-1")
    jobs.execute(stages=("generation",), poll_seconds=0.001)
    assert len(submitted) == 1
    assert jobs.result("request-1") == {"response": {"choices": []}}
