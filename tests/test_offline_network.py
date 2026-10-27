import httpx
import numpy as np
import pytest

from jev_navi.types import Observation, PolicyInput, ServiceError


def test_unbudgeted_provider_never_reaches_real_transport(monkeypatch):
    from jev_navi.providers import StartLuxPolicy

    calls = []
    monkeypatch.setattr(httpx.Client, "request", lambda *a, **k: calls.append(k))
    p = StartLuxPolicy("sentinel", endpoint="https://example.invalid/decisions")
    try:
        with pytest.raises(ServiceError) as caught:
            p.decide(
                PolicyInput("stop", (Observation(np.zeros((8, 8, 3), dtype=np.uint8), 0),), ())
            )
        assert caught.value.code == "offline_mode"
        assert calls == []
        assert p.request_count == 0
    finally:
        p.close()


def test_persistent_phase_predebits_each_provider_retry(tmp_path, monkeypatch):
    from jev_navi.budget import PhaseBudget, RunBudget
    from jev_navi.providers import StartLuxPolicy

    monkeypatch.delenv("OFFLINE", raising=False)
    phase = PhaseBudget(tmp_path / "phase.sqlite3", "authorized-test", limit=2, live_enabled=True)
    budget = RunBudget(phase, 2)
    seen = []

    class LocalTransport(httpx.BaseTransport):
        def handle_request(self, request):
            seen.append(phase.used)
            if len(seen) == 1:
                return httpx.Response(500)
            return httpx.Response(
                200,
                json={
                    "model": "fixture",
                    "answers": {
                        "action": {
                            "probabilities": {
                                "MOVE_FORWARD": 0,
                                "TURN_LEFT": 0,
                                "TURN_RIGHT": 0,
                                "STOP": 1,
                            }
                        }
                    },
                },
            )

    client = httpx.Client(transport=LocalTransport())
    policy = StartLuxPolicy("fixture", client=client, budget=budget)
    try:
        policy.decide(
            PolicyInput("stop", (Observation(np.zeros((8, 8, 3), dtype=np.uint8), 0),), ())
        )
        assert seen == [1, 2]
        assert budget.used == phase.used == policy.request_count == 2
        with pytest.raises(ServiceError, match="call_budget"):
            policy.decide(
                PolicyInput("stop", (Observation(np.zeros((8, 8, 3), dtype=np.uint8), 0),), ())
            )
        assert seen == [1, 2]
    finally:
        client.close()
