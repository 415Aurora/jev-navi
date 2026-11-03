import json

import httpx
import pytest
from test_d3_direct import route_view

from jev_navi.official_d3 import OfficialD3Policy
from jev_navi.providers import CallBudget
from jev_navi.types import ServiceError


def response(model="vllm-sr/d3-mini"):
    return {
        "model": model,
        "answers": {
            "action": {
                "type": "choice",
                "choice": "STOP",
                "probabilities": {
                    "MOVE_FORWARD": 0.0,
                    "TURN_LEFT": 0.0,
                    "TURN_RIGHT": 0.0,
                    "STOP": 1.0,
                },
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def test_wire_and_audit():
    requests = []

    def handler(req):
        requests.append(req)
        return httpx.Response(200, json=response())

    p = OfficialD3Policy(
        "fixture-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        budget=CallBudget(2),
    )
    d = p.decide(route_view())
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://decision.vllm-sr.ai/v1/systemone"
    assert requests[0].headers["authorization"] == "Bearer fixture-key"
    assert body["model"] == "vllm-sr/d3-mini" and len(body["images"]) == 3
    assert set(body["questions"]) == {"action"} and "PRIVATE_" not in json.dumps(body)
    assert d.action.value == "STOP" and d.usage["reported_usage"]["input_tokens"] == 100
    assert p.request_audit() == {
        "http_attempts": 1,
        "model_submissions": 1,
        "poll_attempts": 0,
        "retry_attempts": 0,
    }
    assert p.response_events[0]["raw"]["model"] == "vllm-sr/d3-mini"
    assert p.response_events[0]["request"]["image_count"] == 3


@pytest.mark.parametrize("status", [401, 429, 503])
def test_http_error_single_attempt(status):
    b = CallBudget(3)
    p = OfficialD3Policy(
        "fixture-key",
        budget=b,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(status, json={"echo": "fixture-key"})
            )
        ),
    )
    with pytest.raises(ServiceError):
        p.decide(route_view())
    assert b.used == 1 and p.request_count == 1
    assert "fixture-key" not in json.dumps(p.response_events)


@pytest.mark.parametrize("mutation", ["model", "nan", "missing", "type"])
def test_bad_responses_are_rejected_and_audited(mutation):
    r = response()
    if mutation == "model":
        r["model"] = "other"
    elif mutation == "nan":
        r["answers"]["action"]["probabilities"]["STOP"] = float("nan")
    elif mutation == "missing":
        del r["answers"]
    else:
        r["answers"]["action"]["type"] = "score"
    p = OfficialD3Policy(
        "fixture-key",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda q: httpx.Response(200, content=json.dumps(r)))
        ),
    )
    with pytest.raises(ServiceError):
        p.decide(route_view())
    assert p.response_events
    json.dumps(p.response_events, allow_nan=False)


def test_factory_and_credentials(tmp_path, monkeypatch):
    from jev_navi.credentials import load_local_env
    from jev_navi.experiments import make_policy

    path = tmp_path / "env"
    path.write_text("DECISION_API_KEY=fixture-key\n")
    monkeypatch.delenv("DECISION_API_KEY", raising=False)
    load_local_env(path)
    p = make_policy({"kind": "d3", "provider": "official"}, CallBudget(1))
    assert isinstance(p, OfficialD3Policy)
    p.close()


def test_real_transport_requires_persistent_budget():
    p = OfficialD3Policy("fixture-key", budget=CallBudget(1))
    try:
        with pytest.raises(ServiceError, match="persistent"):
            p.decide(route_view())
        assert p.budget.used == 0
    finally:
        p.close()
