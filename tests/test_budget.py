"""Ledger behavior tests use SQLite and local processes, never model requests."""

import multiprocessing
import os
import sqlite3
import stat
from concurrent.futures import ProcessPoolExecutor

import pytest

from jev_navi.types import ServiceError


@pytest.fixture(autouse=True)
def isolated_offline_environment(monkeypatch):
    # Each ledger test owns its switch; monkeypatch restores the caller's value.
    monkeypatch.delenv("OFFLINE", raising=False)


def _consume_worker(path, attempts):
    from jev_navi.budget import PhaseBudget

    budget = PhaseBudget(path, "parallel", limit=17, live_enabled=True)
    accepted = 0
    for _ in range(attempts):
        try:
            budget.consume()
            accepted += 1
        except ServiceError as exc:
            if exc.code != "call_budget":
                raise
    return accepted


def test_ledger_shares_counts_across_instances_and_restart(tmp_path):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "budgets.sqlite3"
    first = PhaseBudget(path, "run", limit=3, live_enabled=True)
    second = PhaseBudget(path, "run", limit=3, live_enabled=True)
    first.consume()
    assert second.used == 1
    second.consume()
    del first, second
    restarted = PhaseBudget(path, "run", limit=3, live_enabled=True)
    assert restarted.used == 2
    restarted.consume()
    with pytest.raises(ServiceError, match="call_budget"):
        restarted.consume()
    assert restarted.used == 3
    assert restarted.persistent is True


def test_budget_counts_each_attempt_without_refunding_failure(tmp_path):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "ledger.sqlite3"
    budget = PhaseBudget(path, "retry", limit=2, live_enabled=True)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            budget.consume()
            raise ConnectionError("HTTP failed after predebit")
    with pytest.raises(ServiceError, match="call_budget"):
        budget.consume()
    assert PhaseBudget(path, "retry", limit=2, live_enabled=True).used == 2


def test_multiprocess_race_cannot_exceed_cap(tmp_path):
    from jev_navi.budget import PhaseBudget

    path = str(tmp_path / "parallel.sqlite3")
    with ProcessPoolExecutor(
        max_workers=6, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        accepted = list(pool.map(_consume_worker, [path] * 6, [15] * 6))
    assert sum(accepted) == 17
    assert PhaseBudget(path, "parallel", limit=17, live_enabled=True).used == 17


@pytest.mark.parametrize("changes", [{"limit": 7}, {"initial_used": 0}, {"live_enabled": True}])
def test_existing_phase_configuration_cannot_reset_or_open_ledger(tmp_path, changes):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "ledger.sqlite3"
    original = {"limit": 8, "initial_used": 3, "live_enabled": False}
    PhaseBudget(path, "immutable", **original)
    with pytest.raises(ValueError, match="configuration"):
        PhaseBudget(path, "immutable", **(original | changes))
    assert PhaseBudget(path, "immutable", **original).used == 3


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_offline_environment_checked_again_on_every_consume(tmp_path, monkeypatch, flag):
    from jev_navi.budget import PhaseBudget

    budget = PhaseBudget(tmp_path / "ledger.sqlite3", "run", limit=4, live_enabled=True)
    budget.consume()
    monkeypatch.setenv("OFFLINE", flag)
    with pytest.raises(ServiceError, match="offline_mode"):
        budget.consume()
    assert budget.used == 1
    monkeypatch.delenv("OFFLINE")
    budget.consume()
    assert budget.used == 2


def test_default_offline_and_zero_cap_fail_closed(tmp_path):
    from jev_navi.budget import PhaseBudget

    offline = PhaseBudget(tmp_path / "ledger.sqlite3", "offline", limit=0)
    with pytest.raises(ServiceError, match="offline_mode"):
        offline.consume()
    live_zero = PhaseBudget(tmp_path / "ledger.sqlite3", "live-zero", limit=0, live_enabled=True)
    with pytest.raises(ServiceError, match="call_budget"):
        live_zero.consume()
    assert offline.used == live_zero.used == 0


def test_database_revocation_applies_to_already_constructed_instance(tmp_path):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "ledger.sqlite3"
    budget = PhaseBudget(path, "run", limit=3, live_enabled=True)
    budget.consume()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE phase_budgets SET live_enabled=0 WHERE phase_id='run'")
    with pytest.raises(ServiceError, match="offline_mode"):
        budget.consume()
    assert budget.used == 1


def test_phase_counts_are_isolated(tmp_path):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "ledger.sqlite3"
    first = PhaseBudget(path, "one", limit=3, live_enabled=True)
    second = PhaseBudget(path, "two", limit=3, live_enabled=True)
    first.consume()
    assert first.used == 1
    assert second.used == 0


def test_initialize_ledger_preserves_historical_483_of_500(tmp_path):
    from jev_navi.budget import PhaseBudget, initialize_ledger

    path = tmp_path / "ledger.sqlite3"
    initial = initialize_ledger(path)
    assert initial["first-round"]["used"] == 483
    assert initial["first-round"]["limit"] == 500
    assert initial["first-round"]["live_enabled"] is False
    assert "stop_diagnostic_2026-10-27.md" in initial["first-round"]["source"]
    current = initial["r2rce-offline-v1"]
    assert (current["limit"], current["used"], current["live_enabled"]) == (0, 0, False)
    with pytest.raises(ValueError, match="configuration"):
        PhaseBudget(path, "first-round", limit=500)
    history = PhaseBudget(path, "first-round", limit=500, initial_used=483)
    with pytest.raises(ServiceError, match="offline_mode"):
        history.consume()
    assert initialize_ledger(path) == initial
    assert history.used == 483


def test_snapshot_reads_database_and_reports_remaining(tmp_path):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "ledger.sqlite3"
    first = PhaseBudget(path, "run", limit=4, initial_used=1, live_enabled=True)
    second = PhaseBudget(path, "run", limit=4, initial_used=1, live_enabled=True)
    second.consume()
    snapshot = first.snapshot()
    assert snapshot["phase_id"] == "run"
    assert snapshot["initial_used"] == 1
    assert snapshot["used"] == 2
    assert snapshot["remaining"] == 2
    assert snapshot["persistent"] is True
    assert snapshot["live_enabled"] is True


def test_nested_path_and_reopened_database_are_private(tmp_path):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "nested" / "directory" / "ledger.sqlite3"
    budget = PhaseBudget(path, "private", limit=2, live_enabled=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    os.chmod(path, 0o644)
    PhaseBudget(path, "private", limit=2, live_enabled=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with sqlite3.connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("UPDATE phase_budgets SET used=used+1 WHERE phase_id='private'")
        journal = path.with_name(path.name + "-journal")
        assert journal.exists()
        assert stat.S_IMODE(journal.stat().st_mode) == 0o600
        connection.rollback()
    assert budget.used == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": -1},
        {"limit": True},
        {"limit": 1.5},
        {"limit": 2, "initial_used": -1},
        {"limit": 2, "initial_used": 3},
        {"limit": 2, "initial_used": True},
        {"limit": 2, "live_enabled": "true"},
    ],
)
def test_invalid_configuration_is_rejected_without_creating_database(tmp_path, kwargs):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "ledger.sqlite3"
    with pytest.raises(ValueError):
        PhaseBudget(path, "run", **kwargs)
    assert not path.exists()


def test_blank_phase_is_rejected(tmp_path):
    from jev_navi.budget import PhaseBudget

    with pytest.raises(ValueError):
        PhaseBudget(tmp_path / "ledger.sqlite3", " ", limit=0)


def test_shared_ledger_is_independent_of_experiment_working_directory(tmp_path, monkeypatch):
    from jev_navi.budget import PhaseBudget

    path = tmp_path / "shared" / "ledger.sqlite3"
    first_directory = tmp_path / "experiment-one"
    second_directory = tmp_path / "experiment-two"
    first_directory.mkdir()
    second_directory.mkdir()
    monkeypatch.chdir(first_directory)
    first = PhaseBudget(path, "same-phase", limit=1, live_enabled=True)
    first.consume()
    monkeypatch.chdir(second_directory)
    restarted = PhaseBudget(path, "same-phase", limit=1, live_enabled=True)
    with pytest.raises(ServiceError, match="call_budget"):
        restarted.consume()
    assert restarted.used == first.used == 1


def test_run_budget_caps_each_instance_and_shares_phase_total(tmp_path):
    from jev_navi.budget import PhaseBudget, RunBudget

    path = tmp_path / "ledger.sqlite3"
    phase = PhaseBudget(path, "live", limit=3, live_enabled=True)
    first = RunBudget(phase, limit=2)
    second = RunBudget(PhaseBudget(path, "live", limit=3, live_enabled=True), limit=2)
    assert first.used == second.used == 0
    first.consume()
    first.consume()
    with pytest.raises(ServiceError, match="call_budget"):
        first.consume()
    second.consume()
    with pytest.raises(ServiceError, match="call_budget"):
        second.consume()
    assert (first.limit, first.used, second.used, phase.used) == (2, 2, 1, 3)
    assert first.persistent is True


def test_run_budget_does_not_refund_failed_http_attempts(tmp_path):
    from jev_navi.budget import PhaseBudget, RunBudget

    phase = PhaseBudget(tmp_path / "ledger.sqlite3", "live", limit=5, live_enabled=True)
    budget = RunBudget(phase, limit=2)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            budget.consume()
            raise ConnectionError("HTTP attempt failed after budget reservation")
    with pytest.raises(ServiceError, match="call_budget"):
        budget.consume()
    assert budget.used == phase.used == 2


def test_run_budget_phase_rejection_leaves_run_counter_unchanged(tmp_path):
    from jev_navi.budget import PhaseBudget, RunBudget

    phase = PhaseBudget(tmp_path / "ledger.sqlite3", "offline", limit=0)
    budget = RunBudget(phase, limit=2)
    with pytest.raises(ServiceError, match="offline_mode"):
        budget.consume()
    assert budget.used == phase.used == 0
    zero_run = RunBudget(phase, limit=0)
    with pytest.raises(ServiceError, match="call_budget"):
        zero_run.consume()
    assert zero_run.used == phase.used == 0


def test_run_snapshot_combines_local_attempts_and_current_phase(tmp_path):
    from jev_navi.budget import PhaseBudget, RunBudget

    phase = PhaseBudget(tmp_path / "ledger.sqlite3", "live", limit=5, live_enabled=True)
    first = RunBudget(phase, limit=2)
    second = RunBudget(phase, limit=3)
    first.consume()
    second.consume()
    snapshot = first.snapshot()
    assert snapshot["run_limit"] == 2
    assert snapshot["run_used"] == 1
    assert snapshot["phase"]["used"] == 2
    assert snapshot["phase"]["limit"] == 5


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_run_budget_rejects_invalid_caps(tmp_path, limit):
    from jev_navi.budget import PhaseBudget, RunBudget

    phase = PhaseBudget(tmp_path / "ledger.sqlite3", "offline", limit=0)
    with pytest.raises(ValueError):
        RunBudget(phase, limit=limit)


def test_run_budget_requires_a_persistent_phase_budget():
    from jev_navi.budget import RunBudget
    from jev_navi.providers import CallBudget

    with pytest.raises(ValueError, match="PhaseBudget"):
        RunBudget(CallBudget(2), limit=2)
