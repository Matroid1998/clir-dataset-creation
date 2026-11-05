"""Durable, per-call CSV checkpoints for full multilingual evidence alignment.

SQLite is authoritative. ``calls.csv`` records every completed API attempt and
``qa_updates.csv`` records every changed QA after a task completes. Both journals
are flushed and fsynced before the corresponding method returns. ``export()``
writes current full snapshots; reopening repairs CSVs from committed transactions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "terra_alignment_store_v1"
LANGUAGES = frozenset({"en", "de", "es", "fr", "zh"})
QA_FIELDS = [
    "qa_id", "corpus", "question", "answer", "question_language", "source_unit_id",
    "source_language", "anchor_id", "source_evidence_text", "source_evidence_origin",
    "eligibility_status", "metadata_json", "evidence_all_languages",
    "evidence_alignment_status", "export_schema_version",
]
CALL_FIELDS = [
    "call_revision", "task_id", "request_hash", "attempt", "response_id", "model",
    "api_model", "provider", "source_unit_id", "target_unit_id", "source_language",
    "target_language", "status", "started_at", "elapsed_seconds", "usage_json", "cost",
    "results_json", "response_text", "raw_response_path", "error_json", "export_schema_version",
]
RESULT_FIELDS = [
    "alignment_id", "task_id", "request_hash", "anchor_id", "corpus", "source_unit_id",
    "target_unit_id", "source_language", "target_language", "source_text", "target_text",
    "status", "exact_match", "qa_ids", "model", "api_model", "provider", "attempt_count",
    "reason", "validation_status", "export_schema_version",
]
UPDATE_FIELDS = ["revision", "task_id", "request_hash", *QA_FIELDS]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _read_index(value: Path | Iterable[dict] | dict) -> list[dict]:
    if isinstance(value, (Path, str)):
        rows = [json.loads(line) for line in Path(value).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    elif isinstance(value, dict):
        rows = list(value["qas"] if "qas" in value else value.values())
    else:
        rows = list(value)
    if not rows:
        raise ValueError("qa_index must contain the frozen QA inventory")
    if len({row["qa_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate qa_id in frozen inventory")
    return sorted(rows, key=lambda row: row["qa_id"])


def _csv_cell(value: Any) -> Any:
    return _json(value) if isinstance(value, (dict, list, tuple)) else value


class AlignmentStore:
    """Thread-safe durable journal and CSV exporter for a single frozen run."""

    def __init__(self, run_dir: Path, qa_index: Path | Iterable[dict] | dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._closed = False
        self._csv_dirty = False
        self._last_call_revision = 0
        self._last_qa_revision = 0
        rows = _read_index(qa_index)
        self._qas = {row["qa_id"]: row for row in rows}
        for row in rows:
            self._validate_qa(row)
        self._connection = sqlite3.connect(self.run_dir / "alignment.sqlite", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS qas (qa_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                revision INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL, request_hash TEXT NOT NULL, attempt INTEGER NOT NULL,
                envelope TEXT NOT NULL, parsed_rows TEXT NOT NULL,
                UNIQUE(task_id, request_hash, attempt)
            );
            CREATE TABLE IF NOT EXISTS task_results (
                task_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, fingerprint TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results (
                alignment_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS results_task ON results(task_id);
            CREATE TABLE IF NOT EXISTS result_qas (
                alignment_id TEXT NOT NULL REFERENCES results(alignment_id) ON DELETE CASCADE,
                qa_id TEXT NOT NULL REFERENCES qas(qa_id),
                PRIMARY KEY(alignment_id, qa_id)
            );
            CREATE INDEX IF NOT EXISTS result_qas_qa ON result_qas(qa_id);
            CREATE TABLE IF NOT EXISTS qa_updates (
                revision INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                request_hash TEXT NOT NULL, qa_id TEXT NOT NULL, payload TEXT NOT NULL
            );
        """)
        try:
            metadata = dict(self._connection.execute("SELECT key, value FROM metadata"))
            expected = {"schema_version": SCHEMA_VERSION, "qa_index_sha256": _digest(rows)}
            if metadata and any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError("checkpoint schema or frozen QA inventory does not match this run")
            with self._connection:
                if not metadata:
                    self._connection.executemany("INSERT INTO metadata VALUES (?, ?)", expected.items())
                    self._connection.executemany("INSERT INTO qas VALUES (?, ?)",
                                                 [(row["qa_id"], _json(row)) for row in rows])
            self._rebuild_exports()
        except Exception:
            self._connection.close()
            raise

    @staticmethod
    def _validate_qa(qa: dict) -> None:
        if not qa.get("qa_id"):
            raise ValueError("empty QA identifier")
        evidence = qa.get("evidence_all_languages") or {}
        if not isinstance(evidence, dict):
            raise TypeError("evidence_all_languages must be a language-keyed object")
        for language, entries in evidence.items():
            if language not in LANGUAGES or not isinstance(entries, list):
                raise ValueError("unsupported original evidence language or invalid span list")
            for entry in entries:
                if (not isinstance(entry.get("text"), str) or not entry["text"].strip()
                        or entry.get("match_status", "exact") != "exact"):
                    raise ValueError("original evidence must be a nonempty exact source quotation")
        for target in qa.get("expected_targets", []):
            if target["language"] not in LANGUAGES:
                raise ValueError("Arabic/Russian or other unsupported target in frozen inventory")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("alignment store is closed")
        if self._csv_dirty:
            self._rebuild_exports()

    def record_attempt(self, envelope: dict, rows: list[dict] | None = None) -> dict:
        """Commit one API response/error and immediately flush it to calls.csv."""
        required = (envelope.get("task_id"), envelope.get("request_hash"))
        if not all(required) or type(envelope.get("attempt")) is not int or envelope["attempt"] < 1:
            raise ValueError("attempt requires task_id, request_hash, and a positive attempt number")
        for field in ("source_language", "target_language"):
            if envelope.get(field) and envelope[field] not in LANGUAGES:
                raise ValueError(f"unsupported attempt {field}")
        raw, parsed = _json(envelope), _json(rows or [])
        with self._lock:
            self._ensure_open()
            previous = self._connection.execute(
                "SELECT * FROM attempts WHERE task_id=? AND request_hash=? AND attempt=?",
                (*required, envelope["attempt"]),
            ).fetchone()
            if previous:
                if previous["envelope"] != raw or previous["parsed_rows"] != parsed:
                    raise ValueError("conflicting payload for an already journaled API attempt")
                self._sync_journals()
                return {"call_revision": previous["revision"], "inserted": False}
            with self._connection:
                cursor = self._connection.execute(
                    "INSERT INTO attempts(task_id, request_hash, attempt, envelope, parsed_rows) "
                    "VALUES (?, ?, ?, ?, ?)", (*required, envelope["attempt"], raw, parsed),
                )
                revision = cursor.lastrowid
            self._sync_journals()
            return {"call_revision": revision, "inserted": True}

    def _validated_results(self, task: dict, rows: list[dict]) -> list[dict]:
        if task["source_language"] not in LANGUAGES or task["target_language"] not in LANGUAGES:
            raise ValueError("unsupported alignment task language")
        if task["source_language"] == task["target_language"]:
            raise ValueError("target language must differ from source language")
        anchors = {anchor["anchor_id"]: anchor for anchor in task["anchors"]}
        if len(anchors) != len(task["anchors"]):
            raise ValueError("duplicate source anchor in task")
        if {row["anchor_id"] for row in rows} != set(anchors) or len(rows) != len(anchors):
            raise ValueError("final task results must cover each source anchor exactly once")
        hashes = {row.get("request_hash") for row in rows}
        if len(hashes) != 1 or not next(iter(hashes), None):
            raise ValueError("final task results require one request_hash")
        normalized = []
        for original in rows:
            row = dict(original)
            anchor = anchors[row["anchor_id"]]
            if not anchor["text"].strip() or anchor["text"] not in task["source_text"]:
                raise ValueError("source anchor is not an exact supplied-source quotation")
            if row.get("source_text", anchor["text"]) != anchor["text"]:
                raise ValueError("source anchor text changed in result")
            row["source_text"] = anchor["text"]
            for key in ("task_id", "corpus", "source_unit_id", "target_unit_id",
                        "source_language", "target_language"):
                if row.get(key, task[key]) != task[key]:
                    raise ValueError(f"result identity mismatch: {key}")
                row[key] = task[key]
            identity = task["task_id"] + ":" + anchor["anchor_id"]
            if row.get("alignment_id", identity) != identity:
                raise ValueError("result alignment ID does not match its task and anchor")
            row["alignment_id"] = identity
            ids = row.get("qa_ids", anchor["qa_ids"])
            if set(ids) != set(anchor["qa_ids"]) or len(ids) != len(set(ids)):
                raise ValueError("QA links do not match the frozen source anchor")
            row["qa_ids"] = sorted(ids)
            for qa_id in ids:
                qa = self._qas.get(qa_id)
                if (qa is None or qa["source_unit_id"] != task["source_unit_id"]
                        or qa.get("anchor_id") != anchor["anchor_id"]
                        or qa.get("source_evidence_text") != anchor["text"]):
                    raise ValueError("result QA provenance differs from the frozen inventory")
            quote = row.get("target_text") or ""
            exact = isinstance(quote, str) and bool(quote.strip()) and quote in task["target_text"]
            row["exact_match"] = exact
            if row.get("status") == "matched" and not exact:
                row["status"] = "invalid"
                row["reason"] = "Returned quotation is not a nonempty exact target substring."
            row["validation_status"] = "pending_review" if row.get("status") == "matched" and exact else "not_accepted"
            normalized.append(row)
        return sorted(normalized, key=lambda row: row["alignment_id"])

    def record_results(self, task: dict, rows: list[dict]) -> dict:
        """Replace final task outcomes and fsync latest affected QA CSV revisions."""
        with self._lock:
            self._ensure_open()
            normalized = self._validated_results(task, rows)
            request_hash = normalized[0]["request_hash"]
            fingerprint = _digest(normalized)
            previous = self._connection.execute("SELECT * FROM task_results WHERE task_id=?",
                                                 (task["task_id"],)).fetchone()
            if previous and previous["fingerprint"] == fingerprint:
                self._sync_journals()
                return {"inserted": False, "qa_updates": 0}
            old_qa_ids = {record[0] for record in self._connection.execute(
                "SELECT DISTINCT q.qa_id FROM result_qas q JOIN results r USING(alignment_id) "
                "WHERE r.task_id=?", (task["task_id"],),
            )}
            affected = old_qa_ids | {qa_id for row in normalized for qa_id in row["qa_ids"]}
            with self._connection:
                self._connection.execute("DELETE FROM results WHERE task_id=?", (task["task_id"],))
                for row in normalized:
                    self._connection.execute("INSERT INTO results VALUES (?, ?, ?)",
                                             (row["alignment_id"], task["task_id"], _json(row)))
                    self._connection.executemany("INSERT INTO result_qas VALUES (?, ?)",
                                                 [(row["alignment_id"], qa_id) for qa_id in row["qa_ids"]])
                self._connection.execute(
                    "INSERT INTO task_results VALUES (?, ?, ?) ON CONFLICT(task_id) DO UPDATE SET "
                    "request_hash=excluded.request_hash, fingerprint=excluded.fingerprint",
                    (task["task_id"], request_hash, fingerprint),
                )
                for qa_id in sorted(affected):
                    self._connection.execute(
                        "INSERT INTO qa_updates(task_id,request_hash,qa_id,payload) VALUES (?,?,?,?)",
                        (task["task_id"], request_hash, qa_id, _json(self._qa_row(qa_id))),
                    )
            self._sync_journals()
            return {"inserted": True, "qa_updates": len(affected),
                    "revision": self._last_qa_revision}

    def load_results(self, task_id: str, request_hash: str | None = None) -> list[dict]:
        with self._lock:
            self._ensure_open()
            state = self._connection.execute("SELECT request_hash FROM task_results WHERE task_id=?",
                                              (task_id,)).fetchone()
            if not state or request_hash is not None and state[0] != request_hash:
                return []
            return [json.loads(row[0]) for row in self._connection.execute(
                "SELECT payload FROM results WHERE task_id=? ORDER BY alignment_id", (task_id,),
            )]

    def _qa_row(self, qa_id: str, results: list[dict] | None = None) -> dict:
        qa = self._qas[qa_id]
        evidence = json.loads(_json(qa.get("evidence_all_languages") or {}))
        statuses: dict[str, list[dict]] = {}
        for language, entries in evidence.items():
            for entry in entries:
                entry.setdefault("anchor_id", qa.get("anchor_id"))
                entry.setdefault("alignment_id", f"source:{qa['source_unit_id']}:{qa.get('anchor_id')}")
                entry["validation_status"] = "source_exact"
                entry["is_original"] = True
            statuses[language] = [{"source_unit_id": entry["source_unit_id"],
                                   "anchor_id": entry["anchor_id"], "status": "source_exact"}
                                  for entry in entries]
        for target in qa.get("expected_targets", []):
            statuses.setdefault(target["language"], []).append({
                "source_unit_id": target["source_unit_id"], "task_id": target["task_id"],
                "anchor_id": qa.get("anchor_id"), "status": "pending",
            })
        if results is None:
            results = [json.loads(row[0]) for row in self._connection.execute(
                "SELECT r.payload FROM results r JOIN result_qas q USING(alignment_id) "
                "WHERE q.qa_id=? ORDER BY r.alignment_id", (qa_id,),
            )]
        for row in results:
            language = row["target_language"]
            statuses[language] = [status for status in statuses.get(language, [])
                                  if not (status.get("task_id") == row["task_id"]
                                          and status.get("anchor_id") == row["anchor_id"])]
            provenance = {key: row.get(key) for key in (
                "alignment_id", "anchor_id", "task_id", "request_hash", "model", "api_model", "provider",
            )}
            statuses[language].append({**provenance, "source_unit_id": row["target_unit_id"],
                                       "status": row["status"], "exact_match": row["exact_match"],
                                       "validation_status": row["validation_status"],
                                       "reason": row.get("reason", "")})
            if row["status"] == "matched" and row["exact_match"]:
                evidence.setdefault(language, []).append({
                    **provenance, "source_unit_id": row["target_unit_id"], "text": row["target_text"],
                    "origin": "cross_language_alignment", "validation_status": "pending_review",
                    "match_status": "exact", "is_original": False,
                })
        for language, entries in evidence.items():
            unique = {(entry["source_unit_id"], entry.get("anchor_id")): entry for entry in entries}
            evidence[language] = [unique[key] for key in sorted(unique)]
        row = {field: qa.get(field, "") for field in QA_FIELDS}
        row.update(evidence_all_languages=evidence, evidence_alignment_status=statuses,
                   export_schema_version=SCHEMA_VERSION)
        return row

    @staticmethod
    def _call_row(record: sqlite3.Row) -> dict:
        envelope = json.loads(record["envelope"])
        response = envelope.get("response") or {}
        usage = response.get("usage") or {}
        text = "".join(content.get("text", "") for output in response.get("output", [])
                       for content in output.get("content") or [] if content.get("type") == "output_text")
        row = {field: envelope.get(field, "") for field in CALL_FIELDS}
        row.update(call_revision=record["revision"], response_id=response.get("id", ""),
                   model=response.get("model", envelope.get("model", "")),
                   status="error" if envelope.get("error") else response.get("status", "unknown"),
                   usage_json=usage, cost=usage.get("cost"), results_json=json.loads(record["parsed_rows"]),
                   response_text=text, error_json=envelope.get("error") or {},
                   export_schema_version=SCHEMA_VERSION)
        return row

    @staticmethod
    def _update_row(record: sqlite3.Row) -> dict:
        return {"revision": record["revision"], "task_id": record["task_id"],
                "request_hash": record["request_hash"], **json.loads(record["payload"])}

    def _append_csv(self, path: Path, fields: list[str], rows: list[dict]) -> None:
        if not rows:
            return
        try:
            with path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                writer.writerows({key: _csv_cell(value) for key, value in row.items()} for row in rows)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            self._csv_dirty = True
            raise

    def _atomic_csv(self, path: Path, fields: list[str], rows: Iterable[dict]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({key: _csv_cell(value) for key, value in row.items()} for row in rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _sync_journals(self) -> None:
        calls = self._connection.execute("SELECT * FROM attempts WHERE revision>? ORDER BY revision",
                                         (self._last_call_revision,)).fetchall()
        self._append_csv(self.run_dir / "calls.csv", CALL_FIELDS, [self._call_row(row) for row in calls])
        if calls:
            self._last_call_revision = calls[-1]["revision"]
        updates = self._connection.execute("SELECT * FROM qa_updates WHERE revision>? ORDER BY revision",
                                           (self._last_qa_revision,)).fetchall()
        self._append_csv(self.run_dir / "qa_updates.csv", UPDATE_FIELDS,
                         [self._update_row(row) for row in updates])
        if updates:
            self._last_qa_revision = updates[-1]["revision"]

    def _write_snapshots(self) -> dict:
        results = [json.loads(row[0]) for row in self._connection.execute(
            "SELECT payload FROM results ORDER BY alignment_id")]
        by_qa: dict[str, list[dict]] = {}
        for row in results:
            for qa_id in row["qa_ids"]:
                by_qa.setdefault(qa_id, []).append(row)
        self._atomic_csv(self.run_dir / "results.csv", RESULT_FIELDS,
                         ({**row, "export_schema_version": SCHEMA_VERSION} for row in results))
        self._atomic_csv(self.run_dir / "qa_evidence.csv", QA_FIELDS,
                         (self._qa_row(qa_id, by_qa.get(qa_id, [])) for qa_id in sorted(self._qas)))
        return {"qas": len(self._qas), "alignments": len(results),
                "call_revision": self._last_call_revision, "qa_revision": self._last_qa_revision}

    def _rebuild_exports(self) -> None:
        calls = self._connection.execute("SELECT * FROM attempts ORDER BY revision").fetchall()
        updates = self._connection.execute("SELECT * FROM qa_updates ORDER BY revision").fetchall()
        self._atomic_csv(self.run_dir / "calls.csv", CALL_FIELDS, (self._call_row(row) for row in calls))
        self._atomic_csv(self.run_dir / "qa_updates.csv", UPDATE_FIELDS,
                         (self._update_row(row) for row in updates))
        self._last_call_revision = calls[-1]["revision"] if calls else 0
        self._last_qa_revision = updates[-1]["revision"] if updates else 0
        self._write_snapshots()
        self._csv_dirty = False

    def export(self) -> dict:
        """Atomically refresh complete results/QA snapshots from durable state."""
        with self._lock:
            self._ensure_open()
            self._sync_journals()
            return self._write_snapshots()

    checkpoint = export

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                try:
                    self.export()
                finally:
                    self._connection.close()
                    self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


AlignmentCSVStore = AlignmentStore
