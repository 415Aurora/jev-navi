import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from jev_navi.providers import D3Policy
from jev_navi.types import Action, Feedback, Observation, PolicyInput, ServiceError


def route_view():
    return PolicyInput(
        "Turn left. Walk through the door. Stop beside the chair.",
        tuple(Observation(np.full((8, 8, 3), n, np.uint8), n) for n in (4, 5, 6)),
        (Feedback(Action.MOVE_FORWARD, None, "blocked", False),),
        memory={"goal_position": "PRIVATE_GOAL", "gt_locations": "PRIVATE_GT"},
    )


def completed(*, model="d3-mini", probabilities=None):
    return {
        "data": {
            "id": "task-1",
            "status": 10,
            "output": {
                "model": model,
                "choices": {
                    "action": {
                        "probabilities": probabilities
                        or {
                            "MOVE_FORWARD": 0.25,
                            "TURN_LEFT": 0.25,
                            "TURN_RIGHT": 0.25,
                            "STOP": 0.25,
                        }
                    }
                },
            },
        }
    }


def policy(handler, **kwargs):
    return D3Policy(
        "fixture-secret",
        prompt_version="r2rce-direct-v1",
        poll_interval=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_route_wire_context_keeps_full_instruction_order_and_no_private_memory():
    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completed())

    p = policy(serve)
    decision = p.decide(route_view())
    body = requests[0]
    assert body["app"] == "vllm-sr/d3-mini"
    assert (
        "Route instruction: Turn left. Walk through the door. Stop beside the chair."
        in body["input"]["state"]
    )
    assert "Local instruction:" not in body["input"]["state"]
    assert "did not move" in body["input"]["choices"][0]["instructions"]
    options = body["input"]["choices"][0]["options"]
    assert [o["name"] for o in options] == ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
    assert "entire" in options[-1]["description"]
    assert "0.25" in options[0]["description"] and "15" in options[1]["description"]
    pixels = [
        np.asarray(Image.open(io.BytesIO(base64.b64decode(v.split(",", 1)[1]))))[0, 0, 0]
        for v in body["input"]["images"]
    ]
    assert pixels == [4, 5, 6]
    assert all(f"step={n}" in body["input"]["state"] for n in (4, 5, 6))
    assert '"executed": null' in body["input"]["state"]
    assert "PRIVATE_" not in json.dumps(body)
    assert decision.action == Action.MOVE_FORWARD  # Stable enum-order tie break.
    assert p.prompt_version == "r2rce-direct-v1"
    assert decision.raw["identity"] == {
        "requested_app": "vllm-sr/d3-mini",
        "requested_model": "d3-mini",
        "returned_model": "d3-mini",
        "provider_revision": None,
    }
    assert decision.usage["reported_cost"] is None


def test_legacy_d3_keeps_local_prompt():
    sent = []
    p = D3Policy(
        "fixture",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: (
                    sent.append(json.loads(r.content)) or httpx.Response(200, json=completed())
                )
            )
        ),
    )
    p.decide(route_view())
    assert "Local instruction:" in sent[0]["input"]["state"]
    assert "local destination" in sent[0]["input"]["choices"][0]["options"][-1]["description"]


def test_route_prompt_rejects_unknown_version_before_transport():
    with pytest.raises(ValueError, match="prompt"):
        D3Policy("fixture", prompt_version="unknown")


@pytest.mark.parametrize("model", ["other-model", "", None])
def test_d3_rejects_wrong_identity_and_retains_response(model):
    p = policy(lambda r: httpx.Response(200, json=completed(model=model)))
    with pytest.raises(ServiceError) as error:
        p.decide(route_view())
    assert error.value.code == ("model_mismatch" if model else "invalid_response")
    assert p.response_events[-1]["raw"]["response"]["data"]["output"]["model"] == model


@pytest.mark.parametrize(
    "probabilities",
    [
        {"STOP": 1},
        {"MOVE_FORWARD": True, "TURN_LEFT": 0, "TURN_RIGHT": 0, "STOP": 0},
        {"MOVE_FORWARD": 0.5, "TURN_LEFT": 0.5, "TURN_RIGHT": 0.5, "STOP": 0.5},
        {"MOVE_FORWARD": -0.1, "TURN_LEFT": 0.1, "TURN_RIGHT": 0, "STOP": 1},
    ],
)
def test_d3_invalid_probabilities_leave_auditable_failure(probabilities):
    p = policy(lambda r: httpx.Response(200, json=completed(probabilities=probabilities)))
    with pytest.raises(ServiceError) as error:
        p.decide(route_view())
    assert error.value.code == "invalid_response"
    assert p.response_events and p.request_audit()["model_submissions"] == 1


def test_poll_and_retry_predebit_each_actual_attempt(tmp_path, monkeypatch):
    from jev_navi.budget import PhaseBudget, RunBudget

    monkeypatch.delenv("OFFLINE", raising=False)
    phase = PhaseBudget(
        tmp_path / "ledger.sqlite3", "local-transport-test", limit=4, live_enabled=True
    )
    seen = []

    class LocalTransport(httpx.BaseTransport):
        def handle_request(self, request):
            seen.append((request.method, phase.used))
            if len(seen) in (1, 3):
                return httpx.Response(503)
            return httpx.Response(
                200,
                json=(
                    {"data": {"id": "task-1", "status": 1}}
                    if request.method == "POST"
                    else completed()
                ),
            )

    p = D3Policy(
        "fixture",
        client=httpx.Client(transport=LocalTransport()),
        budget=RunBudget(phase, 4),
        poll_interval=0,
        prompt_version="r2rce-direct-v1",
    )
    result = p.decide(route_view())
    assert result.action == Action.MOVE_FORWARD
    assert seen == [("POST", 1), ("POST", 2), ("GET", 3), ("GET", 4)]
    assert p.request_audit() == {
        "http_attempts": 4,
        "model_submissions": 2,
        "poll_attempts": 2,
        "retry_attempts": 2,
    }
    assert p.response_events[0]["stage"] == "submission"
    assert p.response_events[-1]["stage"] == "poll"
    with pytest.raises(ServiceError, match="call_budget"):
        p.decide(route_view())
    assert len(seen) == 4


def test_poll_deadline_is_audited_without_more_requests(monkeypatch):
    import jev_navi.providers as providers

    now = [0.0]
    monkeypatch.setattr(providers.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        providers.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration)
    )

    def pending(request):
        now[0] += 0.3
        return httpx.Response(200, json={"data": {"id": "task-1", "status": 1}})

    p = policy(pending, timeout=1)
    with pytest.raises(ServiceError, match="timeout"):
        p.decide(route_view())
    assert p.request_audit()["http_attempts"] == 4
    assert p.request_audit()["poll_attempts"] == 3
    assert len(p.response_events) == 4


def test_offline_d3_cannot_reach_real_transport(tmp_path, monkeypatch):
    from jev_navi.budget import PhaseBudget, RunBudget

    phase = PhaseBudget(tmp_path / "ledger.sqlite3", "disabled", limit=1, live_enabled=False)
    reached = []
    monkeypatch.setattr(httpx.Client, "request", lambda *a, **k: reached.append(k))
    p = D3Policy("fixture", budget=RunBudget(phase, 1), prompt_version="r2rce-direct-v1")
    try:
        with pytest.raises(ServiceError, match="offline_mode"):
            p.decide(route_view())
        assert not reached and p.request_count == phase.used == 0
    finally:
        p.close()
