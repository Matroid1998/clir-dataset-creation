"""Legal QAC commands, using the existing article and block batch pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .batch_recording import RunState, read_run_metadata


def _digest(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _batches():
    from . import eurlex_batch, un_batch
    return {"eurlex": eurlex_batch, "un": un_batch}


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


_IDENTITY_FIELDS = ("corpus", "document_id", "target_id", "document_text_scope",
                    "question_language", "mode", "generator_model_name", "generator_model_id",
                    "candidate_id", "candidate_rank", "is_best", "question", "answer",
                    "faithfulness_verifier_model", "quality_verifier_model", "grading_status",
                    "grading_error", "total_score", "document_text")


def _write_csv(path, rows, fieldnames=()):
    fields = list(dict.fromkeys((*_IDENTITY_FIELDS, *fieldnames,
                                *(key for row in rows for key in row))))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _group_key(row):
    return (row["corpus"], row["target_id"], row["question_language"], row["mode"],
            row["generator_model_id"], row.get("faithfulness_verifier_model", ""),
            row.get("quality_verifier_model", ""))


def _rank(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    for key, group in groups.items():
        def score(row):
            value = row.get("total_score")
            return float(value) if value not in (None, "") else float("-inf")
        ranked = sorted(group, key=score, reverse=True)
        chosen = False
        for position, row in enumerate(ranked, 1):
            valid = row.get("grading_status", row.get("regrade_status", "completed")) == "completed"
            valid = valid and row.get("total_score") not in (None, "")
            row["candidate_rank"] = position if valid else ""
            eligible = valid and (key[0] != "un" or float(row.get("faith_grounding") or 0) >= 3)
            row["is_best"] = bool(eligible and not chosen)
            chosen = chosen or eligible
            row["best_selection_rule"] = "highest_total_grounding_at_least_3" if key[0] == "un" else "highest_total"
            row["best_selection_semantics"] = "Score ranking; check quality_audit_status separately for validity."


def _locations(args, context, sources=None):
    from clir_bench.core.runs import make_run_id
    sources = list(dict.fromkeys(sources if sources is not None else (getattr(args, "source", None) or [])))
    if len(sources) != 1:
        raise ValueError("a run folder belongs to exactly one source")
    root = context.workspace.qac_dir(sources[0])
    directory = getattr(args, "run_dir", None)
    directory = Path(directory) if directory else root / make_run_id()
    filename = Path(getattr(args, "output", None) or "results.csv")
    if filename.name != str(filename) or filename.suffix.lower() != ".csv":
        raise ValueError("--output must be a CSV filename within --run-dir")
    output = directory / filename
    resume = getattr(args, "resume", False)
    if resume and not (directory / "run.sqlite").is_file():
        raise ValueError("--resume requires an existing run.sqlite")
    if not resume and directory.exists() and any(directory.iterdir()):
        raise ValueError(f"run directory is not empty; choose another or use --resume: {directory}")
    saved = read_run_metadata(directory) if resume else {}
    if saved.get("imported") and resume:
        raise ValueError("historical imports have no resumable stage checkpoints; use --targets-from or regrade")
    if saved.get("output") and filename.name != saved["output"]:
        if getattr(args, "output", None):
            raise ValueError("--resume cannot change the output filename")
        output = directory / saved["output"]
    return directory, output, saved


def _run_locations(args, context, sources):
    """A shared experiment ID, with a separate resumable run for each corpus."""
    if len(sources) == 1:
        return {sources[0]: _locations(args, context, sources)}
    from clir_bench.core.runs import make_run_id
    run_id = make_run_id()
    parent = getattr(args, "run_dir", None)
    folders = {source: (Path(parent) / ("un_parallel" if source == "un" else source)
                        if parent else context.workspace.qac_dir(source) / run_id) for source in sources}
    return {source: _locations(SimpleNamespace(**(vars(args) | {"run_dir": directory})), context, [source])
            for source, directory in folders.items()}


def _joined_metadata(records):
    present = [record for record in records if record]
    if not present:
        return {}
    targets, seen = [], set()
    for record in present:
        for entry in record.get("targets", []):
            key = _digest([entry["corpus"], entry["target"]])
            if key not in seen:
                targets.append(entry)
                seen.add(key)
    return dict(present[0], targets=targets)


def _target_metadata(path, sources):
    def read(path):
        directory = path.parent if path.is_file() else path
        database = path if path.suffix == ".sqlite" else directory / "run.sqlite"
        record = read_run_metadata(database) if database.is_file() else {}
        if record.get("targets") and path.suffix != ".csv":
            return record
        csv_path = path if path.suffix == ".csv" else directory / record.get("output", "results.csv")
        rows, _ = _read_csv(csv_path)
        batches, groups = _batches(), defaultdict(list)
        source_hint = record.get("config", {}).get("sources") or sources
        for row in rows:
            source = row.get("corpus") or (source_hint[0] if len(source_hint) == 1 else "")
            if source not in sources:
                if source not in batches:
                    raise ValueError("saved results need corpus=eurlex/un, or a single --source")
                continue
            target = batches[source].target_from_rows([row])
            groups[(source, _digest(asdict(target)))].append(row)
        targets = []
        for (source, _), group in groups.items():
            entry = {"corpus": source, "target": asdict(batches[source].target_from_rows(group))}
            hashes = {row["source_payload_sha256"] for row in group if row.get("source_payload_sha256")}
            if len(hashes) > 1:
                raise ValueError("saved results disagree on source payload")
            if hashes:
                entry["payload_sha256"] = next(iter(hashes))
            targets.append(entry)
        return dict(record, targets=targets)

    path = Path(path)
    if path.is_file() or (path / "run.sqlite").is_file() or (path / "results.csv").is_file():
        record = read(path)
        records = [record]
        available = {entry["corpus"] for entry in record.get("targets", [])}
        related = record.get("related_runs", {})
        related = related if isinstance(related, dict) else {}
        for source in sources:
            sibling = related.get(source)
            if source not in available and sibling:
                records.append(read(Path(sibling)))
        return _joined_metadata(records)
    return _joined_metadata(read(path / ("un_parallel" if source == "un" else source))
                            for source in sources)


def _options(args, context, saved=None, *, operation="generate"):
    old = (saved or {}).get("config", {})
    configured_models = context.setting("generation_model_candidates")
    if configured_models is None:
        configured_models = context.setting("generation_model", "gpt-5.6-luna")
    if isinstance(configured_models, str):
        configured_models = [configured_models]
    elif isinstance(configured_models, (list, tuple)):
        configured_models = list(configured_models)
    else:
        raise ValueError("generation_model_candidates must be a model ID or list of model IDs")  # noqa: TRY004 - invalid configuration
    defaults = {"questions": 100, "questions_per_mode": None, "seed": 42, "keep": 3, "retries": 3,
                "max_references": 6, "context_chars": 30000,
                "reference_chars": None,
                "generation_model": configured_models,
                "verifier_model": context.setting("verifier_model", "google/gemini-3.8-flash"),
                "langs": None, "modes": None, "decider_model": None}
    options = {key: getattr(args, key, None) if getattr(args, key, None) is not None
               else old.get(key, default) for key, default in defaults.items()}
    initial_counts = getattr(args, "persona_initial_counts", None)
    if initial_counts is None:
        initial_counts = old.get("persona_initial_counts")
    if initial_counts is not None:
        options["persona_initial_counts"] = initial_counts
    if options["decider_model"] is None:
        options.pop("decider_model")  # Keep existing run fingerprints stable.
    elif options["decider_model"] not in ("generator", "jev"):
        raise ValueError("--decider-model must be generator or jev")
    elif options["modes"] or options["questions_per_mode"] is not None:
        raise ValueError("--decider-model cannot be combined with --modes or --questions-per-mode; "
                         "the decider selects modes from the supplied prompts")
    if options["questions_per_mode"] is None:
        options.pop("questions_per_mode")  # Preserve fingerprints of existing runs.
    elif options["questions_per_mode"] < 1:
        raise ValueError("--questions-per-mode must be positive")
    elif getattr(args, "questions", None) is not None:
        raise ValueError("--questions and --questions-per-mode cannot be combined")
    for key in ("questions", "keep", "retries"):
        if options[key] < 1:
            raise ValueError(f"--{key} must be positive")
    if options["max_references"] < 0 or options["context_chars"] < 0:
        raise ValueError("context limits must be nonnegative")
    if not options["verifier_model"] or not options["generation_model"] or any(
            not isinstance(model, str) or not model.strip() for model in options["generation_model"]):
        raise ValueError("generator and verifier model IDs must be nonempty")
    if old.get("persona_selection_policy"):
        from . import persona_routing
        if old["persona_selection_policy"] != persona_routing.POLICY:
            raise ValueError("unsupported saved persona selection policy")
        options["persona_selection_policy"] = old["persona_selection_policy"]
        options["persona_weights"] = persona_routing.resolve_weights(old["persona_weights"])
    elif (operation == "generate" and options.get("decider_model") == "jev"
          and not (saved or {}).get("fingerprint") and not getattr(args, "resume", False)):
        from . import persona_routing
        options["persona_selection_policy"] = persona_routing.POLICY
        options["persona_weights"] = persona_routing.resolve_weights(context.setting("persona_weights"))
    return options


def _indexes(sources, supplied=None, *, context=None):
    batches = _batches()
    indexes = {}
    for source in sources:
        if (supplied or {}).get(source) is not None:
            indexes[source] = supplied[source]
        elif source == "eurlex":
            directory = context.workspace.data("eurlex_structure")
            indexes[source] = batches[source].ctx.ArticleIndex(
                articles_path=directory / "articles.jsonl", edges_path=directory / "internal_edges.jsonl",
                external_edges_path=directory / "external_edges.jsonl", status_path=directory / "reference_status.jsonl",
                quarantine_path=directory / "quarantine.jsonl")
        else:
            directory = context.workspace.data("un_blocks")
            indexes[source] = batches[source].ctx.BlockIndex(
                blocks_path=directory / "blocks_en.jsonl", docs_path=directory / "docs_en.jsonl",
                status_path=directory / "reference_status_en.jsonl", sixway_dir=context.workspace.data("un_parallel"))
    return indexes


def _select(sources, indexes, options):
    batches, selections = _batches(), []
    for position, source in enumerate(sources):
        batch = batches[source]
        languages = options["langs"] or (["en", "fr", "de", "es"] if source == "eurlex" else ["en", "fr", "es", "zh"])
        modes = (["lookup"] if options.get("decider_model") else options["modes"]
                 or (batch.gen.DEFAULT_MODES if source == "eurlex" else batch.DEFAULT_MODES))
        supported_langs = batch.ACT_LANGUAGES if source == "eurlex" else batch.UN_LANGUAGES
        supported_modes = batch.gen.MODES if source == "eurlex" else batch.SUPPORTED_MODES
        # Mixed-source selections may name the union of their supported personas/languages.
        active_languages = [language for language in languages if language in supported_langs]
        active_modes = [mode for mode in modes if mode in supported_modes]
        if not active_languages or not active_modes:
            raise ValueError(f"no selected languages/personas are supported by {source}")
        count = (options["questions_per_mode"] * len(active_modes) if options.get("questions_per_mode")
                 else options["questions"] // len(sources) + (position < options["questions"] % len(sources)))
        if count:
            targets = batch.select(indexes[source], n=count, seed=options["seed"],
                                   languages=active_languages, modes=active_modes)
            selections.extend({"corpus": source, "target": asdict(target)} for target in targets)
    all_modes = set().union(*(set(batches[s].gen.MODES if s == "eurlex" else batches[s].SUPPORTED_MODES) for s in sources))
    all_langs = set().union(*(set(batches[s].ACT_LANGUAGES if s == "eurlex" else batches[s].UN_LANGUAGES) for s in sources))
    if set(options["modes"] or []) - all_modes or set(options["langs"] or []) - all_langs:
        raise ValueError("unsupported language or persona in the selected sources")
    return selections


def _prepare(selections, indexes, options, operation):
    batches = _batches()
    targets = [(entry, batches[entry["corpus"]].Target(**entry["target"])) for entry in selections]
    if "un" in indexes:
        wanted = defaultdict(set)
        for entry, target in targets:
            if entry["corpus"] == "un" and target.language != "en":
                wanted[target.language].add(target.doc_id)
                wanted[target.language].update(batches["un"]._cited_doc_ids(indexes["un"], target))
        for language, docs in sorted(wanted.items()):
            indexes["un"].preload_translations([language], docs)
    prepared, prompts = [], {}
    for entry, target in targets:
        source, batch = entry["corpus"], batches[entry["corpus"]]
        supported_modes = batch.gen.MODES if source == "eurlex" else batch.SUPPORTED_MODES
        supported_languages = batch.ACT_LANGUAGES if source == "eurlex" else batch.UN_LANGUAGES
        automatic = operation == "generate" and bool(options.get("decider_model"))
        if (target.mode not in supported_modes and not (automatic and target.mode == "auto")
                or target.language not in supported_languages):
            raise ValueError(f"unsupported saved target persona/language for {source}")
        limits = {"max_references": options["max_references"]} if source == "eurlex" else {
            "context_chars": options["context_chars"], "reference_chars": options["reference_chars"]}
        payload = batch.prepare_payload(target, indexes[source], **limits)
        if payload is None:
            raise ValueError(f"target cannot be reconstructed: {entry['target']}")
        if hasattr(payload, "target") and not payload.target.texts.get(target.language):
            raise ValueError(f"target has no source text in {target.language}: {entry['target']}")
        digest = _digest(payload.text.encode())
        if entry.get("payload_sha256") and digest != entry["payload_sha256"]:
            raise ValueError(f"source payload changed: {entry['target']}")
        entry["payload_sha256"] = digest
        pack = batch.gen.PROMPTS
        prompts[f"{source}/faithfulness"] = pack.faithfulness("batch")
        modes = [target.mode]
        if automatic:
            from . import decider
            if options.get("persona_selection_policy"):
                if f"{source}/decider" not in prompts:
                    decider.build_eligibility_request(source, "preflight validation")
                    decider.eligibility_policy(source)
                modes = [decider.generation_mode(mode) for mode in decider.MODES[source]]
                prompts[f"{source}/decider"] = decider.eligibility_prompt_text(source)
            else:
                modes = [decider.generation_mode(mode) for mode in decider.prompt_modes(source)]
                prompts[f"{source}/decider"] = decider.prompt_text(source, options["decider_model"])
        for mode in modes:
            prompts[f"{source}/quality/{mode}"] = pack.quality(mode, "batch")
            if operation == "generate":
                prompts[f"{source}/generation/{mode}/{target.language}"] = pack.generation(mode, target.language)
        prepared.append((entry, target, payload))
    return prepared, prompts


def generate(args, context, *, selections=None, supplied_indexes=None):
    from clir_bench.core.prompts import load_prompt
    load_prompt.cache_clear()
    if getattr(args, "append", False):
        raise ValueError("legal runs use a new folder or --resume instead of --append")
    if (getattr(args, "exclude_from", None) or getattr(args, "pool", None)
            or getattr(args, "priority_langs", None) or getattr(args, "plan", "balanced") != "balanced"):
        raise ValueError("legal selection uses its batch samplers or --targets-from")
    sources = list(dict.fromkeys(args.source))
    if not sources or set(sources) - {"eurlex", "un"}:
        raise ValueError("legal sources are eurlex and un")
    locations = _run_locations(args, context, sources)
    saved = _joined_metadata(location[2] for location in locations.values())
    origin = _target_metadata(args.targets_from, sources) if getattr(args, "targets_from", None) else saved
    inherited = {key: value for key, value in origin.get("config", {}).items()
                 if key in ("max_references", "context_chars", "reference_chars", "seed")}
    options = _options(args, context, saved or {"config": inherited})
    if selections is None:
        selections = [entry for entry in origin.get("targets", []) if entry["corpus"] in sources] or None
        if origin and selections is None:
            raise ValueError("input run has no matching saved targets")
        if origin and set(sources) - {entry["corpus"] for entry in selections or []}:
            raise ValueError("saved target plan does not cover every requested source")
        if selections is not None and not saved:
            for key, field in (("modes", "mode"), ("langs", "language")):
                requested = getattr(args, key, None)
                if options.get("questions_per_mode"):
                    if requested:
                        selections = [entry for entry in selections if entry["target"][field] in requested]
                    continue
                expected = origin.get("config", {}).get(key) or {entry["target"][field] for entry in selections}
                if requested and set(requested) != set(expected):
                    raise ValueError(f"--{key} cannot change the selections in a saved target plan")
            requested_count = getattr(args, "questions", None)
            if requested_count is not None and requested_count != len(selections):
                raise ValueError("--questions differs from the saved target count")
    indexes = _indexes(sources, supplied_indexes, context=context)
    selections = selections if selections is not None else _select(sources, indexes, options)
    if options.get("decider_model"):
        # A fixed-mode comparison plan can contain the same target several times.
        # Route each source/target/language once for each generator instead.
        unique = {}
        for entry in selections:
            entry = dict(entry, target=dict(entry["target"], mode="auto"))
            identity = (entry["corpus"], entry["target"].get("eli_id")
                        or entry["target"].get("block_id"), entry["target"]["language"])
            unique.setdefault(identity, entry)
        selections = list(unique.values())
    if getattr(args, "limit", None):
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selections = selections[:args.limit]
    if not selections:
        raise ValueError("selection produced no targets")
    if options.get("questions_per_mode"):
        batches = _batches()
        for source in sources:
            batch = batches[source]
            supported = batch.gen.MODES if source == "eurlex" else batch.SUPPORTED_MODES
            selected_modes = {entry["target"]["mode"] for entry in selections if entry["corpus"] == source}
            wanted = set(options["modes"] or selected_modes) & set(supported)
            if not wanted or wanted - selected_modes:
                raise ValueError(f"saved target pool does not cover every requested mode for {source}")
        all_modes = set().union(*(set(batches[s].gen.MODES if s == "eurlex" else batches[s].SUPPORTED_MODES) for s in sources))
        all_langs = set().union(*(set(batches[s].ACT_LANGUAGES if s == "eurlex" else batches[s].UN_LANGUAGES) for s in sources))
        if set(options["modes"] or []) - all_modes or set(options["langs"] or []) - all_langs:
            raise ValueError("unsupported language or persona in the selected sources")
    if options.get("persona_selection_policy"):
        from . import persona_routing
        persona_routing.require_safe_targets(selections, indexes)
    prepared, prompts = _prepare(selections, indexes, options, "generate")
    models = list(dict.fromkeys(options["generation_model"]))
    options["generation_model"] = models
    cases = [(entry, target, payload, model, None)
             for entry, target, payload in prepared for model in models]
    return _execute_sources(args, context, "generate", locations, options, selections,
                            prompts, indexes, cases)


def regrade(args, context):
    from clir_bench.core.prompts import load_prompt
    load_prompt.cache_clear()
    rows, fields = _read_csv(args.input)
    if not rows:
        raise ValueError("input CSV has no generated candidates")
    source_hint = getattr(args, "source", None) or []
    input_sources = list(dict.fromkeys(row.get("corpus") or (source_hint[0] if len(source_hint) == 1 else "")
                                      for row in rows))
    if set(input_sources) - {"eurlex", "un"}:
        raise ValueError("input needs corpus=eurlex/un, or a single --source")
    locations = _run_locations(args, context, input_sources)
    saved = _joined_metadata(location[2] for location in locations.values())
    input_hash = _digest(Path(args.input).read_bytes())
    parent = read_run_metadata(Path(args.input).parent) if (Path(args.input).parent / "run.sqlite").exists() else {}
    # Reuse the original source limits, but a new regrade uses today's judge
    # default. A resumed regrade must retain its own saved verifier instead.
    inherited = {key: value for key, value in parent.get("config", {}).items()
                 if key != "verifier_model"}
    options = _options(args, context, saved or {"config": inherited}, operation="regrade")
    batches, groups = _batches(), defaultdict(list)
    selected_sources = getattr(args, "source", None)
    for position, row in enumerate(rows, 1):
        row["corpus"] = row.get("corpus") or (selected_sources[0] if selected_sources and len(selected_sources) == 1 else "")
        source = row["corpus"]
        if source not in batches:
            raise ValueError("input needs corpus=eurlex/un, or a single --source")
        if selected_sources and source not in selected_sources:
            raise ValueError("--source must cover every input row; regrading never silently removes rows")
        row["target_id"] = row.get("target_id") or row.get("target_article_id") or row.get("block_id")
        if not row.get("generator_model_id") or not row.get("question") or not row.get("answer") or not row["target_id"]:
            raise ValueError(f"input row {position} lacks generator identity, target, question, or answer")
        row["candidate_id"] = row.get("candidate_id") or "q_" + _digest([input_hash, position])[:24]
        row["generator_model_name"] = row.get("generator_model_name") or row["generator_model_id"].rsplit("/", 1)[-1]
        groups[_group_key(row)[:5]].append(row)
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id must be unique in the input CSV")
    sources = list(dict.fromkeys(row["corpus"] for row in rows))
    selections, seen = [], set()
    inherited = {(e["corpus"], e["target"].get("eli_id") or e["target"].get("block_id"),
                  e["target"]["language"], e["target"]["mode"]): e for e in parent.get("targets", [])}
    for key, group in groups.items():
        identity = key[:4]
        hashes = {row["source_payload_sha256"] for row in group if row.get("source_payload_sha256")}
        if len(hashes) > 1:
            raise ValueError("input rows disagree on source payload")
        if identity not in seen:
            entry = dict(inherited.get(identity) or {"corpus": key[0], "target": asdict(batches[key[0]].target_from_rows(group))})
            if hashes:
                if entry.get("payload_sha256") and entry["payload_sha256"] not in hashes:
                    raise ValueError("input source payload differs from its parent run")
                entry["payload_sha256"] = next(iter(hashes))
            selections.append(entry)
            seen.add(identity)
        elif hashes:
            previous = next(e for e in selections if (e["corpus"],
                e["target"].get("eli_id") or e["target"].get("block_id"),
                e["target"]["language"], e["target"]["mode"]) == identity)
            if previous.get("payload_sha256") and previous["payload_sha256"] not in hashes:
                raise ValueError("different generators received different source payloads")
            previous["payload_sha256"] = next(iter(hashes))
    indexes = _indexes(sources, context=context)
    prepared, prompts = _prepare(selections, indexes, options, "regrade")
    contexts = {(e["corpus"], t.eli_id if e["corpus"] == "eurlex" else t.block_id, t.language, t.mode): (e,t,p)
                for e,t,p in prepared}
    cases = [(*contexts[key[:4]], key[4], group) for key, group in groups.items()]
    options["input_sha256"] = input_hash
    return _execute_sources(args, context, "regrade", locations, options, selections,
                            prompts, indexes, cases, original_rows=rows, original_fields=fields)


def _execute_sources(args, context, operation, locations, options, selections,
                     prompts, indexes, cases, *, original_rows=None, original_fields=()):
    jobs = []
    for source, (directory, output, saved) in locations.items():
        selected = [entry for entry in selections if entry["corpus"] == source]
        if not selected:
            continue
        parameters = (args, context, operation, directory, output, saved, options, selected,
                      {key: value for key, value in prompts.items() if key.startswith(source + "/")},
                      {source: indexes[source]}, [case for case in cases if case[0]["corpus"] == source])
        keywords = {"original_rows": ([row for row in original_rows if row["corpus"] == source]
                                      if original_rows is not None else None),
                    "original_fields": original_fields,
                    "related_runs": {key: str(value[0].resolve()) for key, value in locations.items()
                                     if any(entry["corpus"] == key for entry in selections)}}
        # Check every corpus before any billable work or output is started.
        _execute(*parameters, **keywords, validate_only=True)
        jobs.append((parameters, keywords))
    status = 0
    for parameters, keywords in jobs:
        status = max(status, _execute(*parameters, **keywords))
    return status


def _execute(args, context, operation, directory, output, saved, options, selections,
             prompts, indexes, cases, *, original_rows=None, original_fields=(),
             related_runs=None, validate_only=False):
    config = dict(options, operation=operation, targets=selections,
                  prompts_sha256={key: _digest(text.encode()) for key, text in prompts.items()})
    weighted = operation == "generate" and bool(options.get("persona_selection_policy"))
    if weighted:
        # Wait for all generators on one target before verifying that target.
        # This lets duplicate candidates share grades without delaying every
        # verifier until the entire corpus has finished generation.
        config["generation_barrier_policy"] = "per_target_streaming_before_verification-v4"
        config["generation_target_window"] = 8
        from .evidence import uses_contract
        evidence_contract = any(uses_contract(text) for text in prompts.values())
        config["verification_deduplication_policy"] = (
            "full_candidate_with_evidence-v1" if evidence_contract
            else "exact_question_within_target-v1")
    from clir_bench.core.prompt_registry import manifest_metadata
    registry = manifest_metadata()
    if registry:
        config["prompt_registry"] = registry
    cache_path = getattr(args, "generation_cache", None) or saved.get("config", {}).get("generation_cache")
    if cache_path:
        config["generation_cache"] = str(Path(cache_path).resolve())
        config["generation_cache_sha256"] = _digest(Path(cache_path).read_bytes())
    fingerprint = _digest(config)
    if saved and saved.get("fingerprint") != fingerprint:
        raise ValueError("resume settings, prompts, or source inputs changed; start a new run")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if validate_only:
        return 0
    print(f"{operation}: {len(selections)} shared targets, {len(cases)} model/target cases -> {output}")
    if getattr(args, "dry_run", False):
        stages = (3 + bool(options.get("decider_model"))) if operation == "generate" else 2
        calls = len(selections) + 3 * len(cases) if weighted else len(cases) * stages
        print(f"no model calls; at most {calls} initial stage calls")
        return 0
    batches = _batches()
    originals = {row["candidate_id"]: row for row in original_rows or []}
    quota = options.get("questions_per_mode") if operation == "generate" else None
    def task_id(case):
        entry, target, _, model, candidates = case
        return _digest([entry["corpus"], asdict(target), model, [c["candidate_id"] for c in candidates or []]])

    groups = defaultdict(list)
    for case in cases:
        groups[(case[0]["corpus"], case[1].mode, case[3])].append(case)
    plan_order = {task_id(case): position for position, case in enumerate(cases)}
    with RunState(directory / "run.sqlite") as state:
        # A second process may have completed after the preflight path check.
        if saved:
            if state.get("fingerprint") != fingerprint:
                raise ValueError("run state changed since preflight; start a new run")
        elif state.get("schema_version") is not None or state.outcomes():
            raise ValueError("run directory acquired an existing run; choose another directory")
        if cache_path:
            state.load_replay(Path(cache_path), options["generation_model"])
        for key, value in {"schema_version": 1, "config": config, "fingerprint": fingerprint,
                           "targets": selections, "prompts": prompts, "output": output.name,
                           "status": "running"}.items():
            state.put(key, value)
        if original_rows is not None:
            state.put("input", {"path": str(Path(args.input).resolve()), "sha256": options["input_sha256"]})
        if related_runs:
            state.put("related_runs", (state.get("related_runs") or {}) | related_runs)
        completed = {record["task"]: len(record["rows"]) for record in state.outcomes()
                     if record["status"] in ("completed", "no_candidates")}
        routing_decisions, routing_errors = {}, {}
        def work(case, *, generation_only=False, generated_rows=None):
            entry, target, payload, model, candidates = case
            source = entry["corpus"]
            task = task_id(case)
            if task in completed:
                return task, None if generation_only else completed[task]
            limits = {"max_references": options["max_references"]} if source == "eurlex" else {
                "context_chars": options["context_chars"], "reference_chars": options["reference_chars"]}
            decision = None
            try:
                checkpoint = state.checkpoint(task, retries=options["retries"])
                if operation == "generate" and options.get("decider_model"):
                    from . import decider
                    if weighted:
                        from . import persona_routing
                        key = persona_routing.task_id(entry)
                        if key in routing_errors:
                            raise RuntimeError(routing_errors[key])
                        decision = routing_decisions[key]
                    else:
                        decision = decider.decide(source, payload, backend=options["decider_model"],
                                                  model=model, checkpoint=checkpoint)
                    if decision["mode"] == "skip":
                        state.outcome(task, [])
                        return task, None if generation_only else 0
                    target = replace(target, mode=decision["generation_mode"])
                result = batches[source].run_one(target, indexes[source], gen_model=model,
                    grade_model=options["verifier_model"],
                    keep=options["keep"] if candidates is None else len(candidates),
                    existing_candidates=generated_rows if generated_rows is not None else candidates,
                    generation_only=generation_only,
                    deduplicate_exact_questions=weighted,
                    checkpoint=checkpoint,
                    retries=options["retries"], payload=payload, **limits)
                if generation_only:
                    return task, result
                for row in result:
                    if decision is not None:
                        row.update(decider.row_metadata(decision))
                    row.setdefault("generator_model_name", model.rsplit("/", 1)[-1])
                    row["source_payload_sha256"] = entry["payload_sha256"]
                    for phase, prompt_key in (("faithfulness", f"{source}/faithfulness"),
                                              ("quality", f"{source}/quality/{target.mode}")):
                        row[f"{phase}_verifier_prompt_sha256"] = config["prompts_sha256"][prompt_key]
                    if originals:
                        original = originals[row["candidate_id"]]
                        old = {k: v for k,v in original.items() if not (k.startswith(("faith_", "qual_", "quality_", "faithfulness_", "regrade_", "grading_"))
                               or k in ("total_score", "candidate_rank", "is_best"))}
                        old.update(row)
                        # Metadata supplied by the caller remains authoritative.
                        for key in ("question", "answer", "generator_model_name", "generator_model_id", "candidate_id"):
                            old[key] = original[key]
                        row.clear()
                        row.update(old)
                        row["regrade_status"] = row.get("grading_status", "completed")
                        row["regrade_error"] = row.get("grading_error", "")
                        row["regraded_at_utc"] = datetime.now(timezone.utc).isoformat()
                        row["previous_comparison_sha256"] = options["input_sha256"]
                from .batch_recording import _redact
                for row in result:
                    for diagnostic in ("grading_error", "regrade_error"):
                        if diagnostic in row:
                            row[diagnostic] = _redact(row[diagnostic], state._secrets)
                error = "; ".join(sorted({r.get("grading_error", "") for r in result if r.get("grading_status") == "failed"}))
                state.outcome(task, result, error)
            except Exception as exc:  # noqa: BLE001 - provider and parser failures must preserve candidates
                # Regrading must retain every original question even when context/transport fails.
                from .batch_recording import _redact
                diagnostic = _redact(str(exc), state._secrets)
                result = []
                fallback_rows = generated_rows if generated_rows is not None else (candidates or [])
                for original in fallback_rows:
                    row = {k: v for k, v in original.items()
                           if not k.startswith(("faith_", "qual_", "quality_", "faithfulness_"))}
                    row.update(corpus=source, target_id=(target.block_id if source == "un" else target.eli_id),
                               document_id=(target.doc_id if source == "un" else target.celex_id),
                               mode=target.mode, question_language=target.language,
                               candidate_id=original.get("candidate_id", ""),
                               generator_model_id=model,
                               generator_model_name=model.rsplit("/", 1)[-1])
                    if decision is not None:
                        row.update(decider.row_metadata(decision))
                    row.update(grading_status="failed", grading_error=diagnostic, regrade_status="failed",
                               regrade_error=diagnostic, total_score="", candidate_rank="", is_best=False)
                    result.append(row)
                state.outcome(task, result, str(exc))
            return task, None if generation_only else len(result)

        def export():
            outcomes = state.outcomes()
            outcomes.sort(key=lambda record: plan_order[record["task"]])
            result, saved_counts = [], Counter()
            for record in outcomes:
                rows = record["rows"]
                if quota:
                    # A short final batch is chosen by score; original outcomes remain intact.
                    rows = sorted(rows, key=lambda row: float(row["total_score"])
                                  if row.get("total_score") not in (None, "") else float("-inf"), reverse=True)
                for row in rows:
                    key = (row["corpus"], row["mode"], row["generator_model_id"])
                    if not quota or saved_counts[key] < quota:
                        result.append(row)
                        saved_counts[key] += 1
            if originals:
                positions = {key: i for i,key in enumerate(originals)}
                result.sort(key=lambda row: positions[row["candidate_id"]])
            _rank(result)
            _write_csv(output, result, original_fields)
            return outcomes, result

        executor = ThreadPoolExecutor(max_workers=args.workers)
        try:
            if weighted:
                from . import persona_routing
                targets = {}
                for entry, _, payload, _, _ in cases:
                    targets.setdefault(persona_routing.task_id(entry), (entry, payload))
                for number, (key, decision, error) in enumerate(persona_routing.route_targets(
                        state, list(targets.values()), weights=options["persona_weights"],
                        seed=options["seed"], retries=options["retries"], executor=executor,
                        initial_counts=options.get("persona_initial_counts")), 1):
                    if error is not None:
                        routing_errors[key] = error
                    else:
                        routing_decisions[key] = decision
                    if number % 25 == 0 or number == len(targets):
                        print(f"completed eligibility/routing {number}/{len(targets)} targets", flush=True)
                # Generate every model's questions for a target before grading
                # that target. A small target window keeps both generation and
                # verification parallel without putting all verifier jobs behind
                # the full corpus generation queue.
                generated = {}
                grading_futures = []
                target_cases = defaultdict(list)
                for case in cases:
                    target_cases[persona_routing.task_id(case[0])].append(case)
                target_keys = list(target_cases)
                target_window = config["generation_target_window"]
                generation_finished = 0
                generation_futures = {}
                active_targets = set()
                finished_by_target = Counter()
                next_target = 0

                def launch_target(key):
                    active_targets.add(key)
                    for case in target_cases[key]:
                        future = executor.submit(work, case, generation_only=True)
                        generation_futures[future] = key

                while next_target < len(target_keys) and len(active_targets) < target_window:
                    launch_target(target_keys[next_target])
                    next_target += 1
                while generation_futures:
                    done, _ = wait(generation_futures, return_when=FIRST_COMPLETED)
                    finished_targets = []
                    for future in done:
                        key = generation_futures.pop(future)
                        task, rows = future.result()
                        if rows is not None:
                            generated[task] = rows
                        finished_by_target[key] += 1
                        generation_finished += 1
                        if finished_by_target[key] == len(target_cases[key]):
                            for case in target_cases[key]:
                                task = task_id(case)
                                if task in completed or task not in generated:
                                    continue
                                grading_futures.append(executor.submit(
                                    work, case, generated_rows=generated[task]))
                            finished_targets.append(key)
                        if generation_finished % 10 == 0 or generation_finished == len(cases):
                            print(f"completed generation {generation_finished}/{len(cases)} model/target cases",
                                  flush=True)
                    for key in finished_targets:
                        active_targets.remove(key)
                    while next_target < len(target_keys) and len(active_targets) < target_window:
                        launch_target(target_keys[next_target])
                        next_target += 1
                for number, future in enumerate(as_completed(grading_futures), 1):
                    future.result()
                    if number % 10 == 0 or number == len(grading_futures):
                        print(f"completed verification {number}/{len(grading_futures)} model/target cases", flush=True)
            elif quota:
                # Every generator sees the same target rounds, including rounds needed
                # by a model that skipped or returned fewer questions. Keep surplus
                # outcomes in SQLite while limiting only the final CSV.
                rounds = defaultdict(dict)
                for case in cases:
                    mode_key = (case[0]["corpus"], case[1].mode)
                    target_key = _digest(asdict(case[1]))
                    rounds[mode_key].setdefault(target_key, []).append(case)
                pending = {key: list(targets.values()) for key, targets in rounds.items()}
                with state.transaction() as db:
                    started = {row[0] for row in db.execute(
                        "SELECT task FROM outcomes UNION SELECT task FROM stages UNION SELECT task FROM requests")}
                # Resume must finish every previously started shared target even if
                # earlier retried calls now satisfy the quota.
                required = {key: max((position + 1 for position, target_round in enumerate(targets)
                                     if any(task_id(case) in started for case in target_round)), default=0)
                            for key, targets in pending.items()}
                produced, positions, number = Counter(), Counter(), 0
                while pending:
                    futures = {}
                    for key in list(pending):
                        deficit = max(0, max(quota - produced[group] for group in groups if group[:2] == key))
                        # At most --keep candidates can be returned per target, so
                        # these common targets are necessary even with perfect output.
                        needed = max((deficit + options["keep"] - 1) // options["keep"],
                                     required[key] - positions[key])
                        batch = pending[key][positions[key]:positions[key] + needed]
                        if not batch:
                            del pending[key]
                            continue
                        positions[key] += len(batch)
                        for target_round in batch:
                            for case in target_round:
                                futures[executor.submit(work, case)] = (key[0], key[1], case[3])
                    if not futures:
                        break
                    for future in as_completed(futures):
                        _, count = future.result()
                        produced[futures[future]] += count
                    number += 1
                    print(f"completed shared target batch {number}: {len(futures)} model/target cases", flush=True)
            else:
                futures = [executor.submit(work, case) for case in cases]
                for number, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if number % 10 == 0 or number == len(futures):
                        print(f"completed {number}/{len(futures)} cases", flush=True)
        except BaseException:
            state.cancelled.set()
            executor.shutdown(wait=True, cancel_futures=True)
            export()
            state.put("status", "interrupted")
            if getattr(args, "trace", False):
                from .batch_recording import export_traces
                export_traces(directory)
            raise
        else:
            executor.shutdown(wait=True)
        outcomes, rows = export()
        counts = dict(Counter(record["status"] for record in outcomes))
        summary = dict(counts, candidates=len(rows), best=sum(bool(r["is_best"]) for r in rows))
        if weighted:
            summary["persona_balance"] = persona_routing.balance_summary(
                selections[0]["corpus"], routing_decisions.values(), options["persona_weights"],
                failures=len(routing_errors))
        shortages = False
        if quota:
            generated = Counter((row["corpus"], row["mode"], row["generator_model_id"])
                                for record in outcomes for row in record["rows"])
            saved_counts = Counter((row["corpus"], row["mode"], row["generator_model_id"]) for row in rows)
            summary["generated_candidates"] = sum(generated.values())
            summary["quotas"] = [{"corpus": key[0], "mode": key[1], "generator_model_id": key[2],
                                  "requested": quota, "generated": generated[key], "saved": saved_counts[key],
                                  "shortage": max(0, quota - saved_counts[key])} for key in groups]
            shortages = any(item["shortage"] for item in summary["quotas"])
        state.put("summary", summary)
        state.put("status", "completed_with_errors" if counts.get("failed") or shortages else "completed")
        state.put("results_sha256", _digest(output.read_bytes()))
        if getattr(args, "trace", False):
            from .batch_recording import export_traces
            export_traces(directory)
        print(f"wrote {len(rows)} candidates; outcomes {counts}; state -> {state.path}")
        if shortages:
            print(f"target pool exhausted before filling every quota: {summary['quotas']}")
        return 1 if counts.get("failed") or shortages else 0


def best(args, context):
    rows, fields = _read_csv(args.input)
    _rank(rows)
    selected = [row for row in rows if row["is_best"]]
    output = Path(args.output) if args.output else Path(args.input).with_stem(Path(args.input).stem + "_best")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    _write_csv(output, selected, fields)
    print(f"wrote {len(selected)} best candidates -> {output}")
    return 0


def legacy_main(args, *, index, corpus, targets):
    """Adapt the original batch arguments after their corpus-specific selection."""
    from clir_bench.core.config import load_settings
    from clir_bench.core.context import AppContext
    from clir_bench.domains.legal import SPEC
    context = AppContext.build(load_settings(), SPEC)
    output = Path(args.out) if args.out else None
    options = SimpleNamespace(
        source=[corpus], questions=args.n, generation_model=[args.gen_model],
        verifier_model=args.grade_model, workers=args.workers, seed=args.seed,
        keep=args.keep, retries=3, dry_run=args.dry_run,
        langs=None, modes=None, targets_from=None, resume=False,
        run_dir=output.parent if output and str(output.parent) != "." else None,
        output=Path(output.name) if output else None,
        max_references=getattr(args, "max_references", 6),
        context_chars=getattr(args, "context_chars", 30000),
        reference_chars=(_batches()["un"].refs.DEFAULT_REFERENCE_CHARS
                         if corpus == "un" and getattr(args, "no_fit", False) else None),
        generation_cache=getattr(args, "generation_cache", None),
        decider_model=getattr(args, "decider_model", None),
    )
    if getattr(args, "generation_records", None):
        print("generation recording is included in run.sqlite; separate recording files are no longer needed")
    return generate(options, context,
                    selections=[{"corpus": corpus, "target": asdict(target)} for target in targets],
                    supplied_indexes={corpus: index})
