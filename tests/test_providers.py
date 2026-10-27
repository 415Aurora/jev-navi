import json

import httpx
import numpy as np
import pytest

from jev_navi.types import Action, Observation, PolicyInput, ServiceError


def view():
    return PolicyInput(
        "go left to the door",
        tuple(Observation(np.full((16, 16, 3), i * 70, np.uint8), i) for i in range(3)),
        (),
    )


def startlux_response():
    return {
        "model": "test-model-v1",
        "answers": {
            "action": {
                "choice": "STOP",
                "probabilities": {
                    "MOVE_FORWARD": 0.7,
                    "TURN_LEFT": 0.1,
                    "TURN_RIGHT": 0.1,
                    "STOP": 0.1,
                },
                "none_probability": 0.2,
            }
        },
        "usage": {"input_tokens": 99},
    }


def test_startlux_sends_time_ordered_images_and_uses_probability_argmax():
    from jev_navi.providers import CallBudget, StartLuxPolicy

    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=startlux_response())

    p = StartLuxPolicy(
        "secret", client=httpx.Client(transport=httpx.MockTransport(serve)), budget=CallBudget(3)
    )
    decision = p.decide(view())
    assert decision.action == Action.MOVE_FORWARD
    assert len(requests[0]["images"]) == 3
    assert len(set(requests[0]["images"])) == 3
    assert requests[0]["think_mode"] == "never"
    assert requests[0]["model"] == "startlux-decision-27b-autothink"
    assert requests[0]["state"].count("<image>") == len(requests[0]["images"])
    for i in range(3):
        assert f"Image {i + 1}: step={i} <image>" in requests[0]["state"]
    assert "step=0" in requests[0]["state"]
    assert "step=2" in requests[0]["state"]
    assert decision.raw["answers"]["action"]["none_probability"] == 0.2
    assert p.calls == p.request_count == 1


@pytest.mark.parametrize(
    "status,code,attempts",
    [
        (401, "authentication", 1),
        (402, "quota", 1),
        (422, "unsupported_input", 1),
        (429, "rate_limit", 2),
        (503, "http_error", 2),
    ],
)
def test_errors_and_bounded_retry(status, code, attempts):
    from jev_navi.providers import CallBudget, StartLuxPolicy

    p = StartLuxPolicy(
        "secret",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(status, json={"error": "x"}))
        ),
        budget=CallBudget(3),
    )
    with pytest.raises(ServiceError) as exc:
        p.decide(view())
    assert exc.value.code == code
    assert "secret" not in str(exc.value)
    assert p.calls == attempts


def test_retry_consumes_global_budget():
    from jev_navi.providers import CallBudget, StartLuxPolicy

    budget = CallBudget(1)
    p = StartLuxPolicy(
        "secret",
        budget=budget,
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
    )
    with pytest.raises(ServiceError, match="call_budget"):
        p.decide(view())
    assert p.calls == 1
    assert budget.used == 1


@pytest.mark.parametrize(
    "probabilities",
    [
        {"STOP": 1},
        {a.value: float("nan") for a in Action},
        {a.value: -0.1 for a in Action},
        {a.value: 0 for a in Action},
        {a.value: 0.8 for a in Action},
    ],
)
def test_invalid_distributions_fail(probabilities):
    # NaN isn't legal HTTP JSON; inject it at the parser boundary for that case.
    from jev_navi.providers import parse_probabilities

    with pytest.raises(ServiceError):
        parse_probabilities(probabilities)


def test_d3_handles_async_and_preserves_image_order():
    from jev_navi.providers import CallBudget, D3Policy

    requests = []

    def serve(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"data": {"id": "task-1", "status": 1}})
        return httpx.Response(
            200,
            json={
                "data": {
                    "status": 10,
                    "output": {
                        "model": "d3-mini",
                        "choices": {
                            "action": {
                                "choice": "TURN_LEFT",
                                "probabilities": {
                                    "MOVE_FORWARD": 0.1,
                                    "TURN_LEFT": 0.7,
                                    "TURN_RIGHT": 0.1,
                                    "STOP": 0.1,
                                },
                            }
                        },
                    },
                }
            },
        )

    p = D3Policy(
        "secret",
        budget=CallBudget(5),
        poll_interval=0,
        client=httpx.Client(transport=httpx.MockTransport(serve)),
    )
    result = p.decide(view())
    assert result.action == Action.TURN_LEFT
    body = json.loads(requests[0].content)
    assert body["app"] == "vllm-sr/d3-mini"
    assert len(body["input"]["images"]) == 3
    assert requests[1].url.path == "/tasks/task-1"
    assert p.calls == 1
    assert p.request_count == p.budget.used == 2


def test_d3_terminal_failure():
    from jev_navi.providers import D3Policy

    p = D3Policy(
        "secret",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"data": {"id": "1", "status": 11}})
            )
        ),
    )
    with pytest.raises(ServiceError) as exc:
        p.decide(view())
    assert exc.value.code == "task_failed"


def test_transport_timeout_retry_and_non_json():
    from jev_navi.providers import StartLuxPolicy

    def timeout(request):
        raise httpx.ReadTimeout("contains-private-header", request=request)

    p = StartLuxPolicy("secret", client=httpx.Client(transport=httpx.MockTransport(timeout)))
    with pytest.raises(ServiceError) as exc:
        p.decide(view())
    assert exc.value.code == "timeout"
    assert p.calls == 2
    assert "private-header" not in str(exc.value)
    p = StartLuxPolicy(
        "secret",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text="not json"))
        ),
    )
    with pytest.raises(ServiceError) as exc:
        p.decide(view())
    assert exc.value.code == "invalid_response"
