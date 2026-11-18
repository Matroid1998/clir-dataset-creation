"""Prepare additional recent EUR-Lex targets using the existing official profile builder."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path

import requests

from clir_bench.domains.legal.structure import eurlex_resources as resources


def _discover(start, end):
    query = f"""
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT DISTINCT ?celex ?date WHERE {{
  ?work cdm:resource_legal_id_celex ?celex ;
        cdm:work_date_document ?date .
  FILTER (?date >= "{start}"^^xsd:date && ?date <= "{end}"^^xsd:date)
  FILTER regex(str(?celex), "^3[0-9]{{4}}[RL][0-9]{{4,5}}$")
}}
ORDER BY ?celex ?date
"""
    with requests.get(
        resources.CELLAR_SPARQL, params={"query": query, "format": "application/sparql-results+json"},
        headers={"Accept": "application/sparql-results+json"}, timeout=(15, 120),
    ) as response:
        response.raise_for_status()
        return response.json()


def _official_inventory(directory, after, through, seed):
    """Freeze official act IDs and document dates; Formex supplies all article text."""
    if (directory / "manifest.json").exists():
        return directory / "manifest.json"
    period_start = date.fromisoformat(after)
    pages = []
    for year in range(period_start.year, through.year + 1):
        start = max(period_start, date(year, 1, 1))
        end = min(through, date(year, 12, 31))
        pages.append({"from": start.isoformat(), "to": end.isoformat(),
                      "response": _discover(start, end)})
    seen = defaultdict(set)
    for page in pages:
        for binding in page["response"]["results"]["bindings"]:
            celex = binding["celex"]["value"]
            if resources._canonical(celex, to_year=through.year):
                published = date.fromisoformat(binding["date"]["value"][:10])
                if period_start <= published <= through:
                    seen[celex].add(published.isoformat())
    records = []
    for celex, dates in sorted(seen.items()):
        if len(dates) == 1:
            records.append({
                "celex_id": celex, "year": int(celex[1:5]),
                "doc_type": "regulation" if celex[5] == "R" else "directive",
                "date": next(iter(dates)), "title": "", "proxy_articles": 0,
                "size_bin": "unknown", "chars_en": 0, "eligible": True,
            })
    directory.parent.mkdir(parents=True, exist_ok=True)
    output, temp = resources._fresh_work(directory)
    try:
        resources._write_json(temp / "cellar-discovery.json", pages)
        (temp / "inventory.jsonl").write_text(
            "".join(resources._json(row) + "\n" for row in records), encoding="utf-8"
        )
        resources._write_json(temp / "pilot.json", resources.stratified_sample(records, 1, seed))
        artifacts = ("cellar-discovery.json", "inventory.jsonl", "pilot.json")
        manifest = {
            "format": resources.INVENTORY_FORMAT, "version": resources.VERSION,
            "source": {
                "repository": resources.CELLAR_SPARQL, "mode": resources.CELLAR_DISCOVERY,
                "revision": resources.sha256_file(temp / "cellar-discovery.json"),
                "date_field": "work_date_document",
                "date_semantics": "official CELLAR work date",
            },
            "config": {"from_year": resources.FROM_YEAR, "to_year": through.year,
                       "seed": seed, "pilot_size": 1},
            "counts": {"canonical_acts": len(records)},
            "files": resources._hashes(temp, artifacts),
        }
        resources._write_json(temp / "manifest.json", manifest)
        resources.audit(temp / "manifest.json")
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return output / "manifest.json"


def _prepare(config, profiles, directory, baseline, act_limit):
    if not config["inventory"]:
        return False
    parent = directory / "eurlex-supplements"
    number = len(list(parent.glob("*/manifest.json"))) + 1
    output = parent / f"{number:03d}"
    saved = parent / f"{number:03d}-selection.json"
    pinned = resources._manifest_path(Path(config["inventory"]))
    excluded = set()
    for manifest in profiles:
        selected = Path(manifest).parent / "selected_acts.json"
        excluded.update(row["celex_id"] for row in json.loads(selected.read_text(encoding="utf-8")))
    _, base_acts, _ = resources._baseline_info(baseline)
    excluded.update(base_acts)
    before = list(resources._rows(pinned.parent / "inventory.jsonl"))

    def choose(rows):
        recent = [row for row in rows if row["date"] >= config["recent_since"]
                  and row["celex_id"] not in excluded]
        return resources.article_budget_order(recent, config["seed"])[:act_limit]

    if saved.exists():
        plan = json.loads(saved.read_text(encoding="utf-8"))
        inventory = Path(plan["inventory"])
        selected = plan["selected"]
    else:
        selected = choose(before)
        inventory = pinned
        if not selected and config["official_fallback"]:
            # Include the rest of the last year represented in the pinned inventory.
            start = max(config["recent_since"], max(row["date"] for row in before))
            inventory = _official_inventory(
                directory / "cellar-inventory", start, datetime.now(timezone.utc).date(), config["seed"]
            )
            rows = resources._rows(inventory.parent / "inventory.jsonl")
            selected = choose(rows)
        if selected:
            parent.mkdir(parents=True, exist_ok=True)
            resources._write_json(saved, {"inventory": str(inventory), "selected": selected})
    if not selected:
        return False

    cache_candidates = [
        pinned.parent.parent / "prep-5000-cache" / "cache",
        *[Path(path).parent / "cache" for path in profiles],
    ]
    cache = [path for path in cache_candidates if path.is_dir()]
    print(f"Preparing {len(selected)} additional recent EUR-Lex acts -> {output}", flush=True)
    resources._prepare_acts(
        inventory, baseline, output, pilot_size=len(selected),
        seed=config["seed"], workers=min(8, config["workers"]),
        cache_from=cache, _selected_rows=selected,
    )
    return True


def supplement_task(runner, missing):
    config = runner.config
    return partial(
        _prepare, config, runner.profiles(), runner.directory,
        runner.context.workspace.data("eurlex_structure"),
        max(100, min(400, math.ceil(missing / 3))),
    )


def supplement(runner):
    return supplement_task(runner, runner.config["questions"] - len(runner.winners()))()
