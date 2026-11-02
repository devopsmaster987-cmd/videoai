"""Persistent render jobs (SQLite) with real progress and cancellation.

The web UI enqueues work and polls jobs; jobs survive an app restart because all
state lives in SQLite.  Progress values are written by the pipeline as it works
(measured frames / total frames, TTS lines completed, etc.) — never a fake
indeterminate animation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

STATUSES = ("queued", "running", "done", "failed", "cancelled")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    error TEXT,
    params TEXT,
    result TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at DESC);
"""


@dataclass
class Job:
    id: str
    project_id: str
    kind: str
    status: str
    stage: str | None = None
    progress: float = 0.0
    message: str | None = None
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    cancel_requested: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # -- writes ------------------------------------------------------------
    def create(self, project_id: str, kind: str, params: dict[str, Any] | None = None) -> Job:
        now = time.time()
        job = Job(id=uuid.uuid4().hex[:12], project_id=project_id, kind=kind,
                  status="queued", progress=0.0, params=params or {},
                  created_at=now, updated_at=now)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, project_id, kind, status, progress, params, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (job.id, job.project_id, job.kind, job.status, 0.0,
                 json.dumps(job.params), now, now))
        return job

    def update(self, job_id: str, **fields: Any) -> Job | None:
        allowed = {"status", "stage", "progress", "message", "error", "result",
                   "cancel_requested", "started_at", "finished_at"}
        sets, vals = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in ("params", "result") and value is not None:
                value = json.dumps(value)
            sets.append(f"{key}=?")
            vals.append(value)
        if not sets:
            return self.get(job_id)
        sets.append("updated_at=?")
        vals.append(time.time())
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", (*vals, job_id))
        return self.get(job_id)

    def set_progress(self, job_id: str, stage: str, progress: float,
                     message: str | None = None) -> None:
        p = max(0.0, min(1.0, float(progress)))
        self.update(job_id, stage=stage, progress=p, message=message)

    def request_cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        self.update(job_id, cancel_requested=1)
        if job.status == "queued":
            self.update(job_id, status="cancelled", stage="cancelled",
                        message="Cancelled before it started", finished_at=time.time())
        return True

    def mark_running(self, job_id: str) -> None:
        self.update(job_id, status="running", stage="starting", started_at=time.time())

    def mark_done(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        self.update(job_id, status="done", stage="complete", progress=1.0,
                    message="Finished", result=result or {}, finished_at=time.time())

    def mark_failed(self, job_id: str, error: str) -> None:
        self.update(job_id, status="failed", stage="failed", message=error[:500],
                    error=error, finished_at=time.time())

    def mark_cancelled(self, job_id: str) -> None:
        self.update(job_id, status="cancelled", stage="cancelled", message="Cancelled",
                    finished_at=time.time())

    def recover_stale(self) -> int:
        """Mark jobs left 'running' by a crash as failed (honest recovery)."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status='failed', stage='interrupted', "
                "error='The application stopped while this job was running.', "
                "message='Interrupted by restart', finished_at=? "
                "WHERE status='running'", (time.time(),))
            return cur.rowcount

    # -- reads -------------------------------------------------------------
    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        d = dict(row)
        d["params"] = json.loads(d.get("params") or "{}")
        d["result"] = json.loads(d["result"]) if d.get("result") else None
        d["cancel_requested"] = bool(d.get("cancel_requested"))
        return Job(**{k: v for k, v in d.items() if k in Job.__dataclass_fields__})

    def get(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def is_cancelled(self, job_id: str) -> bool:
        job = self.get(job_id)
        return bool(job and (job.cancel_requested or job.status == "cancelled"))

    def list(self, *, project_id: str | None = None, limit: int = 50,
             active_only: bool = False) -> list[Job]:
        sql = "SELECT * FROM jobs"
        where, args = [], []
        if project_id:
            where.append("project_id=?")
            args.append(project_id)
        if active_only:
            where.append("status IN ('queued','running')")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row_to_job(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status").fetchall()
        out = {s: 0 for s in STATUSES}
        for r in rows:
            out[r["status"]] = int(r["c"])
        return out


class JobCancelled(RuntimeError):
    pass


class JobContext:
    """Passed to worker functions for progress reporting and cancellation."""

    def __init__(self, store: JobStore, job_id: str, *, min_interval: float = 0.35):
        self.store = store
        self.job_id = job_id
        self._last = 0.0
        self._min_interval = min_interval
        self._lock = threading.Lock()

    def check_cancelled(self) -> None:
        if self.store.is_cancelled(self.job_id):
            raise JobCancelled()

    def progress(self, stage: str, fraction: float, message: str | None = None,
                 *, force: bool = False) -> None:
        self.check_cancelled()
        now = time.time()
        with self._lock:
            if not force and now - self._last < self._min_interval:
                return
            self._last = now
        self.store.set_progress(self.job_id, stage, fraction, message)

    def log(self, message: str) -> None:
        self.check_cancelled()
        self.store.update(self.job_id, message=message)


class JobRunner:
    """Runs jobs on daemon threads, updating the store as they go."""

    def __init__(self, store: JobStore):
        self.store = store
        self.store.recover_stale()
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def submit(self, project_id: str, kind: str,
               fn: Callable[[JobContext, dict[str, Any]], dict[str, Any]],
               params: dict[str, Any] | None = None, *,
               job_id: str | None = None) -> Job:
        job = self.store.create(project_id, kind, params) if job_id is None else self.store.get(job_id)
        assert job is not None
        self.start(job.id, fn, job.params)
        return self.store.get(job.id) or job

    def start(self, job_id: str,
              fn: Callable[[JobContext, dict[str, Any]], dict[str, Any]],
              params: dict[str, Any]) -> None:
        with self._lock:
            if job_id in self._threads and self._threads[job_id].is_alive():
                return
            t = threading.Thread(target=self._run, args=(job_id, fn, params),
                                 name=f"job-{job_id}", daemon=True)
            self._threads[job_id] = t
            t.start()

    def _run(self, job_id: str, fn: Callable[[JobContext, dict[str, Any]], dict[str, Any]],
             params: dict[str, Any]) -> None:
        ctx = JobContext(self.store, job_id)
        try:
            if self.store.is_cancelled(job_id):
                self.store.mark_cancelled(job_id)
                return
            self.store.mark_running(job_id)
            result = fn(ctx, params) or {}
            if self.store.is_cancelled(job_id):
                self.store.mark_cancelled(job_id)
            else:
                self.store.mark_done(job_id, result)
        except JobCancelled:
            self.store.mark_cancelled(job_id)
        except Exception as exc:  # noqa: BLE001 — record the real failure
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.store.mark_failed(job_id, detail)
        finally:
            with self._lock:
                self._threads.pop(job_id, None)

    def is_running(self, job_id: str) -> bool:
        t = self._threads.get(job_id)
        return bool(t and t.is_alive())

    def cancel(self, job_id: str) -> bool:
        return self.store.request_cancel(job_id)

    def wait(self, job_id: str, timeout: float | None = None) -> Job | None:
        t = self._threads.get(job_id)
        if t:
            t.join(timeout)
        return self.store.get(job_id)
