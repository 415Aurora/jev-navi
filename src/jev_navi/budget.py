"""Persistent HTTP attempt budgets; this module performs no network requests.

Phase configuration is immutable after creation: reopening must supply the same
limit, initial accounting, and live permission. In particular, constructing an
existing offline phase with ``live_enabled=True`` cannot enable it. A future
permission or limit change requires a separately reviewed ledger migration, not
an ordinary application restart. Every attempt must call ``consume`` *before*
HTTP; exceptions and retries never refund a committed debit.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator

from .types import ServiceError

# Historical accounting source: docs/stop_diagnostic_2026-10-27.md records
# 467 previous requests + 16 diagnostic requests = 483 of the first 500.
_HISTORICAL_SOURCE = "docs/stop_diagnostic_2026-10-27.md: 467 + 16 = 483 requests"
_OFFLINE_SOURCE = "docs/plans/2026-10-27-r2rce-offline-hierarchy.md: Task 2, zero live HTTP"
_OFFLINE_VALUES = {"1", "true", "yes", "on"}


class PhaseBudget:
    """SQLite-backed, process-safe budget with explicit per-phase live consent.

    ``path`` identifies the shared ledger, independently of an experiment output
    directory. Callers must reuse that path and phase across runs and workers.
    Only counters and configuration are stored; credentials are never accepted.
    The ``OFFLINE`` environment flag is checked anew on every debit; 1, true,
    yes and on (case-insensitive) prohibit requests even for a live phase.
    """

    persistent = True

    def __init__(
        self,
        path: str | Path,
        phase_id: str,
        *,
        limit: int,
        live_enabled: bool = False,
        initial_used: int = 0,
    ):
        if type(limit) is not int or limit < 0:
            raise ValueError("Budget limit must be a nonnegative integer")
        if type(initial_used) is not int or not 0 <= initial_used <= limit:
            raise ValueError("Initial budget accounting must be between zero and limit")
        if type(live_enabled) is not bool:
            raise ValueError("Live permission must be an explicit boolean")
        if not isinstance(phase_id, str) or not phase_id.strip():
            raise ValueError("Budget phase_id must be nonempty")
        self.path = Path(path).expanduser().resolve()
        self.phase_id = phase_id
        self._limit = limit
        self._initial_used = initial_used
        self._live_enabled = live_enabled
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create before SQLite opens it, so the database and rollback journal
        # inherit private permissions even under a permissive process umask.
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        with self._transaction() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS phase_budgets (
                    phase_id TEXT PRIMARY KEY,
                    call_limit INTEGER NOT NULL CHECK(call_limit >= 0),
                    used INTEGER NOT NULL CHECK(used >= 0 AND used <= call_limit),
                    initial_used INTEGER NOT NULL
                        CHECK(initial_used >= 0 AND initial_used <= used),
                    live_enabled INTEGER NOT NULL CHECK(live_enabled IN (0, 1)),
                    source TEXT NOT NULL DEFAULT ''
                )"""
            )
            connection.execute(
                """INSERT INTO phase_budgets
                   (phase_id, call_limit, used, initial_used, live_enabled)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT(phase_id) DO NOTHING""",
                (phase_id, limit, initial_used, initial_used, int(live_enabled)),
            )
            row = self._row(connection)
            if (row["call_limit"], row["initial_used"], bool(row["live_enabled"])) != (
                limit,
                initial_used,
                live_enabled,
            ):
                raise ValueError("Existing phase configuration differs; ledger was not changed")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM phase_budgets WHERE phase_id = ?", (self.phase_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Budget phase is missing from ledger")
        return row

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        with self._connection() as connection:
            return int(self._row(connection)["used"])

    def consume(self) -> None:
        """Atomically predebit one HTTP attempt, or fail before any HTTP occurs."""
        if os.environ.get("OFFLINE", "").strip().lower() in _OFFLINE_VALUES:
            raise ServiceError("offline_mode")
        with self._transaction() as connection:
            row = self._row(connection)
            if not self._live_enabled or not row["live_enabled"]:
                raise ServiceError("offline_mode")
            if row["call_limit"] != self._limit or row["initial_used"] != self._initial_used:
                raise ValueError("Existing phase configuration differs; ledger was not changed")
            if row["used"] >= row["call_limit"]:
                raise ServiceError("call_budget")
            connection.execute(
                "UPDATE phase_budgets SET used = used + 1 WHERE phase_id = ?",
                (self.phase_id,),
            )

    def snapshot(self) -> dict:
        """Read a consistent current-phase audit snapshot from the shared database."""
        with self._connection() as connection:
            row = self._row(connection)
            return {
                "path": str(self.path),
                "phase_id": self.phase_id,
                "persistent": True,
                "limit": row["call_limit"],
                "used": row["used"],
                "initial_used": row["initial_used"],
                "remaining": row["call_limit"] - row["used"],
                "live_enabled": bool(row["live_enabled"]),
                "source": row["source"],
            }


def initialize_ledger(path: str | Path) -> dict:
    """Idempotently seed historical and current offline phases without resets.

    Returns snapshots keyed by phase ID. Configuration conflicts raise instead
    of overwriting existing configuration or accounting. Neither phase grants
    live permission, including the unused 17 historical requests.
    """
    phases = (
        ("first-round", 500, 483, _HISTORICAL_SOURCE),
        ("r2rce-offline-v1", 0, 0, _OFFLINE_SOURCE),
    )
    snapshots = {}
    for phase_id, limit, initial_used, source in phases:
        budget = PhaseBudget(path, phase_id, limit=limit, initial_used=initial_used)
        with budget._transaction() as connection:
            connection.execute(
                "UPDATE phase_budgets SET source = ? WHERE phase_id = ? AND source = ''",
                (source, phase_id),
            )
        snapshots[phase_id] = budget.snapshot()
    return snapshots


class RunBudget:
    """A per-run attempt cap layered over an authoritative persistent phase.

    Share this instance across every policy and episode in one run. A new run
    starts its local counter at zero, while the underlying phase total persists.
    Failed HTTP attempts remain debited at both levels; denied phase attempts
    never increment the local count. This local counter is not a crash ledger:
    after a crash the phase remains the authoritative accounting source.
    """

    persistent = True

    def __init__(self, phase: PhaseBudget, limit: int):
        if not isinstance(phase, PhaseBudget):
            raise ValueError("RunBudget requires a persistent PhaseBudget")
        if type(limit) is not int or limit < 0:
            raise ValueError("Run limit must be a nonnegative integer")
        self.phase = phase
        self._limit = limit
        self._used = 0
        self._lock = Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def consume(self) -> None:
        with self._lock:
            if self._used >= self._limit:
                raise ServiceError("call_budget")
            self.phase.consume()
            self._used += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "persistent": True,
                "run_limit": self._limit,
                "run_used": self._used,
                "phase": self.phase.snapshot(),
            }
