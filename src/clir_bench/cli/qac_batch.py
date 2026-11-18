"""Route, generate and verify one corpus with resumable OpenRouter batch jobs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import sqlite3
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import date
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile

from clir_bench.core.batch_api import BatchJobs, ChatCalls, MissingReply, encode
from clir_bench.core.generation_models import select_generator, selection_policy
from clir_bench.domains.legal.structure.profile import sha256_file

from . import qac_batch_content as content
from . import qac_batch_sources as sources

VERIFIERS = ("faithfulness", "quality")
EXPECTED_YIELD = {"eurlex": 0.72, "un": 0.72, "gp": 0.86}


def register(subparsers, context):
    available = tuple(source for source in ("eurlex", "un", "gp")
                      if context is None or source in context.domain.source_names)
    if not available:
        return
    parser = subparsers.add_parser("batch", help="Run corpus QAC with OpenRouter batch jobs")
    stages = parser.add_subparsers(dest="batch_stage", required=True)
    route = stages.add_parser("route", help="Screen safe targets with Jev and assign languages")
    route.add_argument("--source", choices=available, required=True)
    route.add_argument("--questions", type=int, required=True, help="Number of verified winners")
    route.add_argument("--decider-model", choices=("jev",), default="jev")
    route.add_argument("--generation-model", default="auto")
    route.add_argument("--verifier-model", default="google/gemini-3.8-flash")
    route.add_argument("--seed", type=int, default=42)
    route.add_argument("--workers", type=int, default=16, help="Concurrent Decisions requests")
    route.add_argument("--recent-since", type=date.fromisoformat)
    route.add_argument("--min-recent-share", type=float)
    route.add_argument("--no-act-cap", action="store_true")
    route.add_argument("--candidates-per-target", type=int, choices=(1, 2, 3))
    route.add_argument("--generation-attempts-per-target", type=int, choices=(1,), default=1)
    route.add_argument("--eurlex-reuse-profile", type=Path, action="append", default=[])
    route.add_argument("--eurlex-inventory", type=Path)
    route.add_argument("--eurlex-official-fallback", action="store_true")
    route.add_argument("--dry-run", action="store_true", help="Count safe targets locally")
    generate = stages.add_parser("generate", help="Submit and import generation batches")
    verify = stages.add_parser("verify", help="Submit both verifiers and export the winners")
    verify_mode = verify.add_mutually_exclusive_group()
    verify_mode.add_argument("--fill", action="store_true", help="Repeat the stages for quota deficits")
    verify_mode.add_argument("--after-generation", action="store_true",
                             help="Wait for accepted generation jobs and verify available candidates")
    for stage in (route, generate, verify):
        stage.add_argument("--run-dir", required=True, type=Path)
        stage.set_defaults(handler=run)
    for stage in (generate, verify):
        stage.add_argument("--poll-seconds", type=float, default=30)
        recovery = stage.add_mutually_exclusive_group()
        recovery.add_argument("--recover-job", metavar="LOCAL_ID=OPENROUTER_ID")
        recovery.add_argument("--retire-job", type=int, metavar="LOCAL_ID",
                              help="Abandon an unresolved submission and replace its targets")


def config_for(args, context):
    if args.questions < 1 or args.workers < 1:
        raise ValueError("--questions and --workers must be positive")
    recent = args.min_recent_share
    if recent is not None and not 0 <= recent <= 1:
        raise ValueError("--min-recent-share must be between 0 and 1")
    if (recent is None) != (args.recent_since is None):
        raise ValueError("--recent-since and --min-recent-share are used together")
    if args.source == "un" and args.recent_since is not None:
        raise ValueError("UN block sampling has no date quota")
    profiles = [str(path.expanduser().resolve()) for path in args.eurlex_reuse_profile]
    eurlex_options = (profiles or args.eurlex_inventory or args.eurlex_official_fallback
                      or args.no_act_cap)
    if args.source != "eurlex" and eurlex_options:
        raise ValueError("EUR-Lex options require --source eurlex")
    if args.source == "eurlex" and not profiles:
        raise ValueError("Supply at least one validated --eurlex-reuse-profile manifest")
    if (args.eurlex_inventory or args.eurlex_official_fallback) and not args.recent_since:
        raise ValueError("EUR-Lex source expansion requires --recent-since")
    if args.eurlex_official_fallback and not args.eurlex_inventory:
        raise ValueError("Official fallback first requires the pinned --eurlex-inventory")
    corpus = context.domain.source(args.source)
    languages = [lang for lang in context.languages.working if lang in corpus.languages]
    if not languages:
        raise ValueError(f"No supported languages for {args.source}")
    quotient, remainder = divmod(args.questions, len(languages))
    return {
        "version": 1, "domain": context.domain.name, "source": args.source,
        "questions": args.questions, "languages": languages, "seed": args.seed,
        "quotas": {lang: quotient + (i < remainder) for i, lang in enumerate(languages)},
        "recent_since": args.recent_since.isoformat() if args.recent_since else None,
        "min_recent_share": recent or 0,
        "min_recent": math.ceil(args.questions * Fraction(str(recent or 0))),
        "generation_model": args.generation_model, "verifier_model": args.verifier_model,
        "model_selection": selection_policy() if args.generation_model == "auto" else None,
        "prompt_sha256": content.prompt_fingerprint(context, args.source, languages),
        "generation_reasoning_effort": context.settings.llm.generation_reasoning_effort,
        "grading_reasoning_effort": context.settings.llm.grading_reasoning_effort,
        "persona_weights": context.setting("persona_weights"), "workers": args.workers,
        "retries": context.settings.llm.retries,
        "no_act_cap": args.no_act_cap,
        "candidates_per_target": args.candidates_per_target or (3 if args.source == "eurlex" else None),
        "profiles": list(dict.fromkeys(profiles)),
        "inventory": (str(args.eurlex_inventory.expanduser().resolve())
                      if args.eurlex_inventory else None),
        "official_fallback": args.eurlex_official_fallback,
    }


@contextmanager
def run_database(directory, *, wait=False):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another batch command is using {directory}") from error
        with closing(sqlite3.connect(directory / "run.sqlite")) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS run_config (config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS source_profiles (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS screening (
                    key TEXT PRIMARY KEY, language TEXT NOT NULL, eligibility TEXT
                );
                CREATE TABLE IF NOT EXISTS targets (
                    key TEXT PRIMARY KEY, parent TEXT NOT NULL, language TEXT NOT NULL,
                    published TEXT NOT NULL, mode TEXT NOT NULL,
                    state TEXT NOT NULL, data TEXT, candidates TEXT, rows TEXT, error TEXT
                );
            """)
            yield db


class Run:
    def __init__(self, db, directory, config, context, poll_seconds=30):
        self.db = db
        self.directory = directory
        self.config = config
        self.context = context
        self.jobs = BatchJobs(db)
        self.poll_seconds = poll_seconds

    @property
    def source(self):
        return self.config["source"]

    def profiles(self):
        profiles = list(self.config["profiles"])
        profiles.extend(str(path) for path in sorted(
            self.directory.glob("eurlex-supplements/*/manifest.json")
        ))
        for path in profiles:
            observed = sha256_file(Path(path))
            row = self.db.execute("SELECT sha256 FROM source_profiles WHERE path=?", (path,)).fetchone()
            if row is not None and row[0] != observed:
                raise ValueError(f"EUR-Lex source profile changed during the run: {path}")
            if row is None:
                with self.db:
                    self.db.execute("INSERT INTO source_profiles VALUES (?,?)", (path, observed))
        return profiles

    def pending(self):
        return bool(self.db.execute(
            "SELECT 1 FROM targets WHERE state IN ('routed','generated') LIMIT 1"
        ).fetchone())

    def winners(self):
        best_by_target = []
        for (data,) in self.db.execute("SELECT rows FROM targets WHERE state='graded' ORDER BY key"):
            rows = json.loads(data)
            passing = [row for row in rows if content.passing(row, self.source)]
            if passing:
                best_by_target.append(max(passing, key=lambda row: float(row["total_score"])))
        return sources.winner_selection(best_by_target, self.config)

    def route(self, *, recent_only=False):
        config = self.config
        if config["model_selection"] and config["model_selection"] != selection_policy():
            raise ValueError("The automatic model-selection table changed since routing started")
        if config["persona_weights"] != self.context.setting("persona_weights"):
            raise ValueError("Persona weights changed since routing started")
        if self.pending():
            print("Existing targets are ready for generation or verification", flush=True)
            return 0

        winners = self.winners()
        counts = Counter(row["question_language"] for row in winners)
        deficits = {lang: goal - counts[lang] for lang, goal in config["quotas"].items()}
        remaining = sum(deficits.values())
        if remaining == 0:
            return 0
        cutoff = config["recent_since"]
        kept_recent = sum(bool(cutoff and row["source_date"] >= cutoff) for row in winners)
        goal_recent = config["min_recent"]
        recent_remaining = max(0, goal_recent - kept_recent)
        older_remaining = min(remaining, config["questions"] - goal_recent
                              - (len(winners) - kept_recent))
        age_goals = (
            {True: remaining, False: 0} if recent_only else
            {True: recent_remaining, False: older_remaining}
        )
        reserve = EXPECTED_YIELD[self.source]
        language_left = {lang: math.ceil(need / reserve) for lang, need in deficits.items()}
        age_left = {bucket: math.ceil(need / reserve) for bucket, need in age_goals.items()}

        used = {row[0] for row in self.db.execute("SELECT key FROM targets")}
        screened = {key: (lang, data) for key, lang, data in self.db.execute(
            "SELECT key,language,eligibility FROM screening WHERE key NOT IN (SELECT key FROM targets)"
        )}
        modes = Counter(dict(self.db.execute(
            "SELECT mode, COUNT(*) FROM targets WHERE data IS NOT NULL GROUP BY mode"
        )))
        act_counts = Counter(dict(self.db.execute(
            "SELECT parent, COUNT(*) FROM targets WHERE data IS NOT NULL GROUP BY parent"
        )))
        staged = 0
        scan_limit = max(512, 4 * sum(language_left.values()))
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            for pool, index in sources.groups(
                    self.context, config, self.profiles(), used, scan_limit):
                routed = self._screen_group(
                    pool, index, executor, language_left, age_left, used, screened, modes, act_counts
                )
                staged += routed
                # Release a complete profile before loading the next large index.
                del index
        print(f"Routed {staged} new {self.source} targets", flush=True)
        return staged

    def _screen_group(self, pool, index, executor, language_left, age_left, used,
                      screened, modes, act_counts):
        config = self.config
        cutoff = config["recent_since"]
        queue, assigned = deque(pool), []
        while queue and sum(age_left.values()) and sum(language_left.values()):
            reserved = Counter()
            reserved_age = Counter()
            window = []
            deferred = []
            limit = min(config["workers"] * 3, sum(language_left.values()), sum(age_left.values()))
            while queue and len(window) < limit:
                item = queue.popleft()
                is_recent = bool(cutoff and item.published >= cutoff)
                if item.key in used or age_left[is_recent] <= 0:
                    continue
                if self.source == "eurlex" and not config["no_act_cap"] and act_counts[item.parent] >= 2:
                    continue
                available = item.languages
                saved = screened.get(item.key)
                if saved is not None and saved[0]:
                    available = (saved[0],)
                options = [lang for lang in available if language_left[lang] > 0]
                if not options:
                    continue
                room = [lang for lang in options if language_left[lang] > reserved[lang]]
                if not room or age_left[is_recent] <= reserved_age[is_recent]:
                    deferred.append(item)
                    continue
                language = max(room, key=lambda lang: language_left[lang] - reserved[lang])
                reserved[language] += 1
                reserved_age[is_recent] += 1
                window.append((item, language, is_recent, saved))

            if not window:
                break
            def decide(entry):
                item, language, _, saved = entry
                if saved is None:
                    return sources.assess(
                        self.context, self.source, item, language, index, retries=config["retries"]
                    )
                return json.loads(saved[1]) if saved[1] else None

            for (item, planned, is_recent, saved), eligibility in zip(window, executor.map(decide, window)):
                if saved is None:
                    with self.db:
                        self.db.execute("INSERT INTO screening VALUES (?,?,?)", (
                            item.key, planned if self.source == "gp" else "", encode(eligibility)
                        ))
                if (eligibility is None or not eligibility["selected_modes"]):
                    self._skip(item, "No eligible persona")
                    used.add(item.key)
                    continue
                if self.source == "eurlex" and not config["no_act_cap"] and act_counts[item.parent] >= 2:
                    continue
                mode, audit = sources.persona(self.context, config, eligibility, modes, item.key)
                # For patents the tentative language identifies the exact source endpoint
                # Jev screened. Legal targets cover all supported languages at routing time.
                language = planned if self.source == "gp" else max(
                    item.languages, key=language_left.__getitem__
                )
                assigned.append((item, language, mode, audit))
                used.add(item.key)
                language_left[language] -= 1
                age_left[is_recent] -= 1
                modes[mode] += 1
                act_counts[item.parent] += 1
            queue.extendleft(reversed(deferred))

        if self.source == "un" and assigned:
            sources.load_un_languages(index, assigned)
        staged = 0
        for item, language, mode, audit in assigned:
            data = sources.prepare(self.context, self.source, item, index, language, mode)
            if data is None:
                self._skip(item, f"Selected source has no valid {language} payload")
                continue
            details, payload, metadata = data
            model = select_generator(self.source, mode, language, model=config["generation_model"])
            target = {
                "key": item.key, "parent": item.parent, "published": item.published,
                "language": language, "mode": mode, "model": model.model,
                "target": details, "payload": payload, "metadata": audit | metadata | model.metadata(),
            }
            with self.db:
                self.db.execute(
                    "INSERT INTO targets(key,parent,language,published,mode,state,data) "
                    "VALUES (?,?,?,?,?,'routed',?)",
                    (item.key, item.parent, language, item.published, mode, encode(target)),
                )
            staged += 1
        return staged

    def _skip(self, item, error):
        with self.db:
            self.db.execute(
                "INSERT INTO targets(key,parent,language,published,mode,state,error) "
                "VALUES (?,?,'',?,'','skipped',?)", (item.key, item.parent, item.published, error)
            )

    def _generation_calls(self, target, replies=None):
        return ChatCalls(replies, generation_prompt=content.generation_prompt(
            self.source, target["payload"], target["language"], self.config["candidates_per_target"]
        ))

    def _ids(self, key, stage, attempt=1):
        return f"{stage}-{content.digest(key)[:32]}-{attempt}"

    def _target_rows(self, state):
        # Yield keys first so writes during processing cannot disturb a live cursor.
        keys = self.db.execute("SELECT key FROM targets WHERE state=? ORDER BY key", (state,)).fetchall()
        for (key,) in keys:
            data, candidates = self.db.execute(
                "SELECT data,candidates FROM targets WHERE key=?", (key,)
            ).fetchone()
            yield key, json.loads(data), json.loads(candidates) if candidates else None

    def generate(self):
        for key, target, _ in self._target_rows("routed"):
            calls = self._generation_calls(target)
            try:
                content.generate(self.context, self.config, target, calls)
            except MissingReply:
                pass
            model, body = calls.requests["generation"]
            with self.db:
                self.jobs.add(self._ids(key, "generation"), "generation", model, body)
        task = self._first_wave_supplement()
        if task:
            with ThreadPoolExecutor(max_workers=1) as executor:
                preparation = executor.submit(task)
                self.jobs.execute(stages=("generation",), poll_seconds=self.poll_seconds)
                error = preparation.exception()
                if error:
                    print(f"EUR-Lex supplement will retry during --fill: {error}", flush=True)
        else:
            self.jobs.execute(stages=("generation",), poll_seconds=self.poll_seconds)
        self.import_generation()

    def import_generation(self, *, available_only=False):
        for key, target, _ in self._target_rows("routed"):
            identity = self._ids(key, "generation")
            if available_only and not self.db.execute(
                    "SELECT 1 FROM batch_requests WHERE id=? AND result IS NOT NULL", (identity,)
            ).fetchone():
                continue
            calls = self._generation_calls(target, {"generation": self.jobs.result(identity)})
            try:
                candidates = content.generate(self.context, self.config, target, calls)
                error = "" if candidates else "Generation abstained"
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                candidates, error = [], str(exc)
            self.jobs.reply(identity, calls.requests["generation"])
            with self.db:
                self.db.execute("UPDATE targets SET state=?,candidates=?,error=? WHERE key=?", (
                    "generated" if candidates else "failed", encode(candidates), error, key
                ))

    def _first_wave_supplement(self):
        if self.source != "eurlex" or not self.config["inventory"]:
            return None
        if self.directory.joinpath("eurlex-supplements").exists():
            return None
        routed, recent = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(published >= ?),0) FROM targets WHERE state='routed'",
            (self.config["recent_since"],),
        ).fetchone()
        if not routed or recent >= self.config["min_recent"]:
            return None
        if self.db.execute("SELECT 1 FROM targets WHERE candidates IS NOT NULL LIMIT 1").fetchone():
            return None
        from .qac_batch_eurlex import supplement_task

        return supplement_task(self, self.config["min_recent"] - recent)

    def verify(self, *, available_only=False):
        max_attempts = max(1, self.config["retries"])
        while True:
            outstanding = False
            for key, target, candidates in self._target_rows("generated"):
                attempts = {}
                replies = {}
                for stage in VERIFIERS:
                    ids = [self._ids(key, stage, attempt) for attempt in range(1, max_attempts + 1)]
                    placeholders = ",".join("?" for _ in ids)
                    latest = self.db.execute(
                        f"SELECT r.id,r.result,j.state FROM batch_requests AS r "
                        f"LEFT JOIN batch_jobs AS j ON r.job=j.id WHERE r.id IN ({placeholders}) "
                        "ORDER BY r.rowid DESC LIMIT 1", ids,
                    ).fetchone()
                    if latest is not None:
                        identity, saved, job_state = latest
                        attempts[stage] = (ids.index(identity) + 1, saved is None, job_state)
                        if saved is not None:
                            replies[stage] = (identity, json.loads(saved))

                calls = ChatCalls({stage: reply for stage, (_, reply) in replies.items()})
                rows = content.grade(self.context, self.config, target, candidates, calls)
                for stage, (identity, _) in replies.items():
                    self.jobs.reply(identity, calls.requests[stage])
                waiting = held = False
                with self.db:
                    for stage in calls.errors:
                        attempt, pending, job_state = attempts.get(stage, (0, False, None))
                        if pending and available_only and job_state == "submitting":
                            held = True
                        elif pending:
                            waiting = True
                        elif attempt < max_attempts:
                            model, body = calls.requests[stage]
                            self.jobs.add(self._ids(key, stage, attempt + 1), stage, model, body)
                            waiting = True
                if waiting:
                    outstanding = True
                elif not held:
                    if not rows:
                        rows = [dict(row, grading_status="failed") for row in candidates]
                    self._save_grades(key, target, rows, encode(calls.errors) if calls.errors else "")
            if not outstanding:
                return
            self.jobs.execute(stages=VERIFIERS, poll_seconds=self.poll_seconds,
                              allow_unresolved=available_only)

    def _save_grades(self, key, target, rows, error=""):
        for row in rows:
            row.update(target["metadata"])
            row.update(
                document_id=target["parent"], target_id=(target["target"].get("eli_id", key)),
                question_language=target["language"], mode=target["mode"],
                generator_model_id=target["model"], corpus=self.source,
                source_date=target["published"], target_key=key,
            )
        rows.sort(key=lambda row: float(row["total_score"]) if row.get("total_score") is not None
                  else float("-inf"), reverse=True)
        with self.db:
            self.db.execute("UPDATE targets SET rows=?,state='graded',error=? WHERE key=?",
                            (encode(rows), error, key))

    def export(self):
        best = self.winners()
        selected = {(row["target_key"], row["candidate_id"]) for row in best}
        fields = dict.fromkeys((
            "corpus", "target_id", "document_id", "source_date", "question_language", "mode",
            "question", "answer", "candidate_id", "generator_model_id",
            "total_score", "is_best", "evidence_spans",
        ))
        def rows():
            for (data,) in self.db.execute(
                    "SELECT rows FROM targets WHERE state='graded' ORDER BY key"):
                for row in json.loads(data):
                    row["is_best"] = (row["target_key"], row["candidate_id"]) in selected
                    yield row

        for row in rows():
            fields.update(dict.fromkeys(row))
        _write_csv(self.directory / "results.csv", fields, rows())
        _write_csv(self.directory / "best.csv", fields,
                   (dict(row, is_best=True) for row in best))
        config = self.config
        cutoff = config["recent_since"]
        recent = [row for row in best if cutoff and row["source_date"] >= cutoff]
        languages = Counter(row["question_language"] for row in best)
        parents = {row["document_id"] for row in best}
        recent_parents = {row["document_id"] for row in recent}
        required_recent_parents = math.ceil(len(parents) * Fraction(str(config["min_recent_share"])))
        complete = (len(best) == config["questions"] and len(recent) >= config["min_recent"]
                    and (self.source != "eurlex" or not cutoff
                         or len(recent_parents) >= required_recent_parents))
        if complete:
            complete = all(languages[lang] == goal for lang, goal in config["quotas"].items())
        summary = {
            "complete": complete, "source": self.source, "target": config["questions"],
            "kept": len(best), "language_goals": config["quotas"], "by_language": dict(languages),
            "minimum_recent_questions": config["min_recent"], "recent_questions": len(recent),
            "parent_documents": len(parents), "recent_parent_documents": len(recent_parents),
            "state_counts": dict(self.db.execute("SELECT state,count(*) FROM targets GROUP BY state")),
        }
        _write(self.directory / "summary.json", json.dumps(summary, indent=2) + "\n")
        print(f"{self.source}: {len(best)}/{config['questions']} verified winners -> {self.directory}",
              flush=True)
        return complete


def _write(path, text):
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(text)
    tmp.replace(path)


def _write_csv(path, fields, rows):
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8",
                            newline="", delete=False) as stream:
        tmp = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: encode(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    tmp.replace(path)


def run(args: argparse.Namespace, context):
    route = args.batch_stage == "route"
    wanted = config_for(args, context) if route else None
    if route and args.dry_run:
        by_age = Counter()
        used = set()
        for pool, index in sources.groups(
                context, wanted, wanted["profiles"], used, args.questions * 2):
            for item in pool:
                if item.key not in used:
                    cutoff = wanted["recent_since"]
                    by_age["recent" if cutoff and item.published >= cutoff else "other"] += 1
                    used.add(item.key)
            del index
        print(json.dumps({"source": wanted["source"], "eligible": by_age,
                          "final_language_quotas": wanted["quotas"]}, indent=2))
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is not set")
    directory = args.run_dir.expanduser().resolve()
    if not route and not (directory / "run.sqlite").is_file():
        raise ValueError(f"Route targets first: {directory}")
    poll_seconds = getattr(args, "poll_seconds", 30)
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    with run_database(directory, wait=getattr(args, "after_generation", False)) as db:
        saved = db.execute("SELECT config FROM run_config").fetchone()
        if saved:
            config = json.loads(saved[0])
            if route and config != wanted:
                raise ValueError("Route settings changed. Use the original flags or a new run directory.")
        elif route:
            config = wanted
            with db:
                db.execute("INSERT INTO run_config VALUES (?)", (encode(config),))
        else:
            raise ValueError(f"Route targets first: {directory}")
        if config["domain"] != context.domain.name:
            raise ValueError(f"Use --domain {config['domain']} for this run")
        if not route and config["prompt_sha256"] != content.prompt_fingerprint(
                context, config["source"], config["languages"]):
            raise ValueError("The active prompts changed since routing. Restore the routed prompt bundle.")
        runner = Run(db, directory, config, context, poll_seconds)
        if recovery := getattr(args, "recover_job", None):
            local, separator, remote = recovery.partition("=")
            if not separator or not local.isdecimal():
                raise ValueError("--recover-job expects LOCAL_ID=OPENROUTER_ID")
            runner.jobs.recover(int(local), remote)
        elif getattr(args, "retire_job", None) is not None:
            runner.jobs.retire(args.retire_job)

        if route:
            runner.route()
            return 0
        if args.batch_stage == "generate":
            runner.generate()
            return 0
        if args.after_generation:
            if db.execute(
                "SELECT 1 FROM batch_requests AS r LEFT JOIN batch_jobs AS j ON r.job=j.id "
                "WHERE r.stage='generation' AND (j.id IS NULL OR j.state='planned') LIMIT 1"
            ).fetchone():
                raise ValueError("Run qac batch generate to submit pending generation jobs first")
            runner.jobs.execute(stages=("generation",), poll_seconds=poll_seconds,
                                allow_unresolved=True)
            runner.import_generation(available_only=True)
        elif db.execute("SELECT 1 FROM targets WHERE state='routed' LIMIT 1").fetchone():
            if not args.fill:
                raise ValueError("Run qac batch generate before verification")
            runner.generate()
        runner.verify(available_only=args.after_generation)
        complete = runner.export()
        while args.fill and not complete:
            if not runner.route() and not (config["recent_since"] and runner.route(recent_only=True)):
                from .qac_batch_eurlex import supplement

                if config["source"] != "eurlex" or not supplement(runner):
                    print(f"Eligible targets exhausted. Remaining quotas are in {directory / 'summary.json'}")
                    return 2
                continue
            runner.generate()
            runner.verify()
            complete = runner.export()
        return 0 if complete or not args.fill else 2
