"""Durable OpenRouter jobs for chat requests prepared by the QAC pipeline."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from types import SimpleNamespace

import requests

from .llm import OPENROUTER_BASE_URL

TERMINAL = {"completed", "failed", "expired", "cancelled"}
MAX_REQUESTS = 500
MAX_BYTES = 8 * 1024 * 1024


class SubmissionUnresolved(RuntimeError):
    """The provider may have accepted a job whose receipt was lost."""


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class MissingReply(ValueError):
    """A captured API call has yet to run."""


class ChatCalls:
    """Capture or replay the normal chat transport, using the existing stage parsers."""

    def __init__(self, replies=None, generation_prompt=None):
        self.replies = replies or {}
        self.generation_prompt = generation_prompt
        self.requests = {}
        self.errors = {}

    def run(self, stage, function):
        try:
            return function()
        except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
            self.errors[stage] = str(error)
            raise

    def client(self, stage, model):
        def create(*, model, messages, extra_body=None, **kwargs):
            if stage == "generation" and self.generation_prompt:
                messages = self.generation_prompt(messages)
            body = {**kwargs, **(extra_body or {}), "messages": messages}
            self.requests[stage] = (model, body)
            if stage not in self.replies:
                raise MissingReply(f"{stage} response pending")
            reply = self.replies[stage]
            if reply.get("error"):
                raise ValueError(reply["error"])
            content = reply["response"]["choices"][0]["message"].get("content")
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content))])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def openrouter(method, path="", body=None):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    with requests.request(
        method, f"{OPENROUTER_BASE_URL}/batches{path}", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=(15, 120),
    ) as response:
        response.raise_for_status()
        return response.json()


class BatchJobs:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        db.executescript("""
            CREATE TABLE IF NOT EXISTS batch_jobs (
                id INTEGER PRIMARY KEY, stage TEXT NOT NULL, model TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'planned', remote_id TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS batch_requests (
                id TEXT PRIMARY KEY, stage TEXT NOT NULL, model TEXT NOT NULL,
                body TEXT NOT NULL, job INTEGER REFERENCES batch_jobs(id), result TEXT
            );
            CREATE INDEX IF NOT EXISTS batch_requests_job ON batch_requests(job);
        """)

    def add(self, identity, stage, model, body):
        self.db.execute(
            "INSERT OR IGNORE INTO batch_requests(id, stage, model, body) VALUES (?,?,?,?)",
            (identity, stage, model, encode(body)),
        )

    def reply(self, identity, call):
        row = self.db.execute(
            "SELECT model, body, result FROM batch_requests WHERE id=?", (identity,)
        ).fetchone()
        if row is None or row[2] is None:
            raise RuntimeError(f"Missing saved result for {identity}")
        if (row[0], json.loads(row[1])) != call:
            raise RuntimeError(f"Request changed during replay for {identity}")
        return json.loads(row[2])

    def result(self, identity):
        row = self.db.execute("SELECT result FROM batch_requests WHERE id=?", (identity,)).fetchone()
        if row is None or row[0] is None:
            raise RuntimeError(f"Missing saved result for {identity}")
        return json.loads(row[0])

    def _plan(self, stage_filter, stages):
        while (group := self.db.execute(
            f"SELECT stage, model FROM batch_requests WHERE job IS NULL AND {stage_filter} "
            "ORDER BY stage, model LIMIT 1", stages,
        ).fetchone()):
            stage, model = group
            ids, size = [], 0
            rows = self.db.execute(
                "SELECT id, length(CAST(body AS BLOB)) FROM batch_requests "
                "WHERE job IS NULL AND stage=? AND model=? ORDER BY id LIMIT ?",
                (stage, model, MAX_REQUESTS),
            )
            for identity, length in rows:
                if ids and size + length + 128 > MAX_BYTES:
                    break
                ids.append(identity)
                size += length + 128
            rows.close()
            with self.db:
                job = self.db.execute(
                    "INSERT INTO batch_jobs(stage,model) VALUES (?,?)", (stage, model)
                ).lastrowid
                self.db.executemany(
                    "UPDATE batch_requests SET job=? WHERE id=?", ((job, key) for key in ids)
                )

    def _submit(self, job, model):
        rows = self.db.execute(
            "SELECT id, body FROM batch_requests WHERE job=? ORDER BY id", (job,)
        ).fetchall()
        body = encode({
            "endpoint": "/v1/chat/completions", "model": model,
            "requests": [{"custom_id": key, "body": json.loads(data)} for key, data in rows],
        }).encode()
        with self.db:
            self.db.execute("UPDATE batch_jobs SET state='submitting' WHERE id=?", (job,))
        try:
            reply = openrouter("POST", body=body)
        except requests.HTTPError as error:
            status = error.response.status_code
            if status in {400, 401, 402, 403, 404, 413, 422, 429}:
                # These responses reject the submission before creating a provider job.
                with self.db:
                    self.db.execute("UPDATE batch_jobs SET state='planned' WHERE id=?", (job,))
                if status == 429:
                    return False
                raise RuntimeError(
                    f"OpenRouter rejected batch {job} ({status}): {error.response.text[:500]}"
                ) from error
            raise SubmissionUnresolved(
                f"OpenRouter submission for local batch {job} is unresolved "
                f"(HTTP {status}): {error.response.text[:350]}"
            ) from error
        except (requests.Timeout, requests.ConnectionError, ValueError) as error:
            raise SubmissionUnresolved(
                f"OpenRouter submission for local batch {job} is unresolved: {type(error).__name__}"
            ) from error
        remote = reply.get("id")
        if not isinstance(remote, str) or not remote:
            raise SubmissionUnresolved(f"Batch {job} returned no ID. Check OpenRouter before resuming.")
        with self.db:
            self.db.execute(
                "UPDATE batch_jobs SET state='submitted', remote_id=? WHERE id=?", (remote, job)
            )
        print(f"Submitted {model} batch {job} as {remote} ({len(rows)} requests)", flush=True)
        return True

    def recover(self, local_id, remote_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", remote_id):
            raise ValueError("Invalid OpenRouter job ID")
        row = self.db.execute("SELECT state FROM batch_jobs WHERE id=?", (local_id,)).fetchone()
        if row is None or row[0] != "submitting":
            raise ValueError(f"Batch {local_id} has no unresolved submission")
        data = openrouter("GET", f"/{remote_id}")
        if data.get("status") != "completed":
            raise ValueError("Wait for the recovered batch to finish so its request IDs can be checked")
        self._store(local_id, data, require_complete_ids=True)

    def retire(self, local_id):
        with self.db:
            updated = self.db.execute(
                "UPDATE batch_jobs SET state='finished' WHERE id=? AND state='submitting'",
                (local_id,),
            )
            if not updated.rowcount:
                raise ValueError(f"Batch {local_id} has no unresolved submission")
            self.db.execute("UPDATE batch_requests SET result=? WHERE job=?", (
                encode({"error": "Unresolved submission retired by operator"}), local_id
            ))

    def _store(self, job, data, *, require_complete_ids=False):
        ids = {row[0] for row in self.db.execute(
            "SELECT id FROM batch_requests WHERE job=?", (job,)
        )}
        results = {}
        for record in data.get("results") or ():
            identity = record.get("custom_id")
            if identity not in ids or identity in results:
                raise ValueError(f"Unexpected or duplicate request ID in batch {job}: {identity}")
            response = record.get("response")
            response = response if isinstance(response, dict) else {}
            body = response.get("body")
            status = response.get("status_code", 0)
            if (record.get("error") or type(status) is not int or not 200 <= status < 300
                    or not isinstance(body, dict)):
                error = record.get("error") or (body.get("error") if isinstance(body, dict) else None)
                error = error or "Empty response"
                results[identity] = {"error": encode(error)}
            else:
                results[identity] = {"response": body}
        if (require_complete_ids or data["status"] == "completed") and ids != results.keys():
            raise ValueError(f"OpenRouter batch {job} returned incomplete request IDs")
        for identity in ids - results.keys():
            results[identity] = {
                "error": f"Batch {data['status']}: {encode(data.get('error'))}"
            }
        with self.db:
            self.db.executemany(
                "UPDATE batch_requests SET result=? WHERE id=?",
                ((encode(result), identity) for identity, result in results.items()),
            )
            self.db.execute(
                "UPDATE batch_jobs SET state='finished', remote_id=? WHERE id=?",
                (data["id"], job),
            )
        print(f"Imported {data['status']} batch {data['id']} ({len(results)} results)", flush=True)

    def execute(self, *, stages, poll_seconds, allow_unresolved=False):
        stage_filter = "stage IN (" + ",".join("?" for _ in stages) + ")"
        self._plan(stage_filter, stages)
        refused = 0
        not_found = {}
        while True:
            planned = self.db.execute(
                f"SELECT id, model FROM batch_jobs WHERE {stage_filter} AND state='planned' "
                "ORDER BY id", stages,
            ).fetchall()
            for job, model in planned:
                try:
                    submitted = self._submit(job, model)
                except SubmissionUnresolved as error:
                    print(str(error), flush=True)
                    continue
                if not submitted:
                    refused += 1
                    break
                refused = 0
            active = self.db.execute(
                f"SELECT id, remote_id FROM batch_jobs WHERE {stage_filter} AND state='submitted' "
                "ORDER BY id", stages,
            ).fetchall()
            for job, remote_id in active:
                try:
                    data = openrouter("GET", f"/{remote_id}")
                except (requests.Timeout, requests.ConnectionError):
                    continue
                except requests.HTTPError as error:
                    status = error.response.status_code
                    if status == 404:
                        not_found[job] = not_found.get(job, 0) + 1
                        if not_found[job] < 5:
                            continue
                        raise RuntimeError(f"Cannot retrieve OpenRouter batch {remote_id}") from error
                    if status in {429, 500, 502, 503, 504}:
                        continue
                    raise
                not_found.pop(job, None)
                if data.get("status") in TERMINAL:
                    if data.get("id") != remote_id:
                        raise ValueError(f"OpenRouter returned a different batch ID for {remote_id}")
                    self._store(job, data)
            pending = self.db.execute(
                f"SELECT 1 FROM batch_jobs WHERE {stage_filter} "
                "AND state IN ('planned','submitted') LIMIT 1",
                stages,
            ).fetchone()
            if not pending:
                unresolved = self.db.execute(
                    f"SELECT id FROM batch_jobs WHERE {stage_filter} AND state='submitting' ORDER BY id",
                    stages,
                ).fetchall()
                if unresolved and not allow_unresolved:
                    jobs = ", ".join(str(row[0]) for row in unresolved)
                    raise RuntimeError(
                        f"OpenRouter batches {jobs} remain unresolved. Resume with "
                        "--recover-job LOCAL_ID=OPENROUTER_ID after confirming the receipt, "
                        "or use --retire-job LOCAL_ID to replace the affected targets."
                    )
                return
            if refused >= 4 and not active:
                raise RuntimeError("OpenRouter batch capacity is full. Resume after other jobs finish.")
            time.sleep(poll_seconds)
