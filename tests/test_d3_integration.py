import json

import pytest
from test_d3_direct import completed, route_view
from test_offline_validation import catalog

from jev_navi.environments.synthetic import SyntheticEnvironment
from jev_navi.experiments import make_policy, run_suite
from jev_navi.providers import CallBudget, D3Policy
from jev_navi.runner import run_episode
from jev_navi.safety import DepthGuard
from jev_navi.types import Camera


def exchange(action="STOP", *, method="POST", path="/apps/run", pending=False, status=200):
    probabilities = {
        name: float(name == action) for name in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP")
    }
    response = (
        {"data": {"id": "task-1", "status": 1}}
        if pending
        else completed(probabilities=probabilities)
    )
    return {"method": method, "path": path, "status_code": status, "json": response}


def fixture(tmp_path, records):
    path = tmp_path / "responses.json"
    path.write_text(json.dumps(records))
    return {"kind": "d3_fixture", "responses": str(path), "prompt_version": "r2rce-direct-v1"}


def test_d3_fixture_runs_actual_async_parser_and_counts_attempts(tmp_path):
    settings = fixture(
        tmp_path, [exchange(pending=True), exchange(method="GET", path="/tasks/task-1")]
    )
    p = make_policy(settings, CallBudget(5))
    assert isinstance(p, D3Policy)
    result = p.decide(route_view())
    assert result.action.value == "STOP" and result.model == "d3-mini"
    assert p.is_fixture
    assert p.request_audit() == {
        "http_attempts": 2,
        "model_submissions": 1,
        "poll_attempts": 1,
        "retry_attempts": 0,
    }
    assert p.response_events[0]["request"]["image_count"] == 3
    assert "Route instruction:" in p.response_events[0]["request"]["state"]
    p.close()
    assert p.client.is_closed


def test_d3_fixture_rejects_exchange_mismatch(tmp_path):
    from jev_navi.types import ServiceError

    p = make_policy(fixture(tmp_path, [exchange(method="GET")]), CallBudget(5))
    with pytest.raises(ServiceError, match="fixture_mismatch"):
        p.decide(route_view())


def test_runner_keeps_bounded_history_counters_and_stops_at_500(tmp_path):
    from jev_navi.types import Action

    records = [exchange("TURN_LEFT")] * 499 + [exchange("STOP"), exchange("MOVE_FORWARD")]
    p = make_policy(fixture(tmp_path, records), CallBudget(600))
    camera = Camera(width=8, height=8)
    result = run_episode(
        SyntheticEnvironment(camera),
        p,
        DepthGuard(camera),
        "Turn and then stop.",
        tmp_path / "run",
        max_decisions=500,
        max_steps=500,
        protocol_id="r2rce_v13_dev",
    )
    p.close()
    assert result["decisions"] == result["executed_steps"] == 500
    assert result["semantic_vetoes"] == 0 and result["end_reason"] == "stop"
    assert result["http_requests"] == 0 and result["fixture_requests"] == 500
    assert result["provider_attempts"] == {
        "http_attempts": 500,
        "model_submissions": 500,
        "poll_attempts": 0,
        "retry_attempts": 0,
    }
    rows = [json.loads(s) for s in (tmp_path / "run/steps.jsonl").read_text().splitlines()]
    request = rows[-1]["response_events"][0]["request"]
    assert request["image_count"] == 3
    assert rows[-1]["input_frames"] == [497, 498, 499]
    assert len(rows[-1]["history"]) == 4
    assert "step=496" not in request["state"]
    assert rows[-1]["executed_action"] == Action.STOP
    assert rows[-1]["provider_attempts"]["http_attempts"] == 500


def test_d3_fixture_integrates_suite_reporting_and_resets_every_episode(tmp_path):
    from jev_navi.datasets import prepare_r2rce

    data, scenes = catalog(tmp_path)
    route = data["routes"][0]
    source = tmp_path / "episodes.json"
    source.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "fixture",
                        "scene_id": "fixture.glb",
                        "trajectory_id": "trajectory",
                        "instruction": {"instruction_text": "Follow the route and stop."},
                        "start_position": route["start_position"],
                        "start_rotation": route["start_rotation"],
                        "goals": [{"position": route["goal_position"]}],
                    }
                ]
            }
        )
    )
    suite = prepare_r2rce(source, scenes, "train", source_kind="fixture")
    config = {
        "protocol_id": "r2rce_v13_dev",
        "environment": {"backend": "synthetic"},
        "policy": fixture(tmp_path, [exchange()]),
        "conditions": ["full", "no_guard"],
        "execution": {"mode": "offline", "phase_id": "r2rce-offline-v1"},
        "run": {"max_decisions": 5, "call_limit": 10, "seed": 17},
    }
    rows = run_suite(config, suite, tmp_path / "suite-run")
    assert len(rows) == 2 and all(r["decisions"] == 1 for r in rows)
    assert all(r["metadata"]["prompt_version"] == "r2rce-direct-v1" for r in rows)
    assert all(r["http_requests"] == 0 and r["models"] == ["d3-mini"] for r in rows)
    for path in (tmp_path / "suite-run").glob("[0-9]*/steps.jsonl"):
        first = json.loads(path.read_text().splitlines()[0])
        assert first["history"] == [] and first["input_frames"] == [0]
        assert first["response_events"][0]["request"]["image_count"] == 1
    summary = json.loads((tmp_path / "suite-run/summary.json").read_text())
    assert summary["fixture_rows"] == 2
    assert not summary["model_comparisons"]
    assert all(g["provider_attempts"]["model_submissions"] == 1 for g in summary["groups"])


@pytest.mark.parametrize("change", ["recovery", "prompt", "live_fixture"])
def test_direct_baseline_rejects_mixed_contracts_before_environment(tmp_path, change):
    from jev_navi.datasets import prepare_r2rce

    data, scenes = catalog(tmp_path)
    r = data["routes"][0]
    source = tmp_path / "episodes.json"
    source.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "fixture",
                        "scene_id": "fixture.glb",
                        "instruction": {"instruction_text": "Follow the route."},
                        "start_position": r["start_position"],
                        "start_rotation": r["start_rotation"],
                        "goals": [{"position": r["goal_position"]}],
                    }
                ]
            }
        )
    )
    suite = prepare_r2rce(source, scenes, "train", source_kind="fixture")
    config = {
        "protocol_id": "r2rce_v13_dev",
        "environment": {"backend": "synthetic"},
        "policy": fixture(tmp_path, [exchange()]),
        "conditions": ["full"],
    }
    if change == "recovery":
        config["recovery"] = {"mode": "feedback"}
    if change == "prompt":
        config["policy"]["prompt_version"] = "local-navigation-v3-recovery"
    if change == "live_fixture":
        config["execution"] = {"mode": "live", "phase_id": "forbidden"}
    with pytest.raises(ValueError, match="direct|fixture|prompt"):
        run_suite(config, suite, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_shipped_offline_configuration_runs_without_network(tmp_path):
    from pathlib import Path

    import yaml

    from jev_navi.datasets import prepare_r2rce

    data, scenes = catalog(tmp_path)
    r = data["routes"][0]
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "fixture",
                        "scene_id": "fixture.glb",
                        "instruction": {"instruction_text": "Follow the route."},
                        "start_position": r["start_position"],
                        "start_rotation": r["start_rotation"],
                        "goals": [{"position": r["goal_position"]}],
                    }
                ]
            }
        )
    )
    suite = prepare_r2rce(source, scenes, "train", source_kind="fixture")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/r2rce_d3_offline.yaml").read_text())
    config["environment"]["backend"] = "synthetic"
    config["policy"]["responses"] = str(root / config["policy"]["responses"])
    results = run_suite(config, suite, tmp_path / "shipped")
    assert len(results) == 1 and results[0]["end_reason"] == "stop"
    assert results[0]["http_requests"] == 0 and results[0]["fixture_requests"] == 1


@pytest.mark.parametrize("stage", ["submission", "poll"])
@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        b"null",
        b'{"data":',
        b'{"data":{"status":10,"output":{"model":"d3-mini","choices":{"action":{"probabilities":{"MOVE_FORWARD":1e999,"TURN_LEFT":0,"TURN_RIGHT":0,"STOP":0}}}}}}',
        b'{"data":{"status":10,"output":{"model":"d3-mini","usage":{"tokens":1e999},"choices":{"action":{"probabilities":{"MOVE_FORWARD":1,"TURN_LEFT":0,"TURN_RIGHT":0,"STOP":0}}}}}}',
    ],
)
def test_bad_provider_body_preserves_started_episode_and_response(tmp_path, stage, body):
    import hashlib

    import httpx

    def serve(request):
        if stage == "poll" and request.method == "POST":
            return httpx.Response(200, json={"data": {"id": "task-1", "status": 1}})
        return httpx.Response(200, content=body)

    p = D3Policy(
        "fixture-secret",
        prompt_version="r2rce-direct-v1",
        poll_interval=0,
        client=httpx.Client(transport=httpx.MockTransport(serve)),
    )
    camera = Camera(width=8, height=8)
    result = run_episode(
        SyntheticEnvironment(camera),
        p,
        DepthGuard(camera),
        "Follow the route and stop.",
        tmp_path / "bad-body",
        max_decisions=5,
        max_steps=5,
        protocol_id="r2rce_v13_dev",
    )
    p.close()
    assert result["end_reason"] == "service_error" and result["error_code"] == "invalid_response"
    assert result["decisions"] == 1 and result["executed_steps"] == 0
    assert result["benchmark_metrics"]["sr"] == result["benchmark_metrics"]["spl"] == 0
    assert (tmp_path / "bad-body/result.json").is_file()
    row = json.loads((tmp_path / "bad-body/steps.jsonl").read_text())
    events = row["response_events"]
    assert len(events) == (1 if stage == "submission" else 2)
    assert events[-1]["stage"] == stage
    assert events[-1]["raw"]["body_sha256"] == hashlib.sha256(body).hexdigest()
    assert events[-1]["raw"]["http_status"] == 200
    assert events[0]["request"]["image_count"] == 1
    assert row["provider_attempts"]["http_attempts"] == len(events)
    json.dumps(events, allow_nan=False)


def test_bad_body_audit_redacts_echoed_credentials(tmp_path):
    import httpx

    body = b'{"echo":"Bearer fixture-secret"'
    p = D3Policy(
        "fixture-secret",
        prompt_version="r2rce-direct-v1",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body))
        ),
    )
    from jev_navi.types import ServiceError

    with pytest.raises(ServiceError):
        p.decide(route_view())
    serialized = json.dumps(p.response_events, allow_nan=False)
    assert "fixture-secret" not in serialized
    assert "REDACTED" in serialized
    assert p.response_events[0]["raw"]["body_bytes"] == len(body)


def test_oversized_bad_body_audit_is_bounded():
    import httpx

    p = D3Policy(
        "fixture-secret",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, content=b"[" + b"x" * 150000)
            )
        ),
    )
    from jev_navi.types import ServiceError

    with pytest.raises(ServiceError):
        p.decide(route_view())
    audit = p.response_events[-1]["raw"]
    assert audit["body_truncated"]
    assert len(audit["body_excerpt"].encode()) <= 65536


def test_successful_provider_echo_is_redacted_in_all_saved_fields():
    import httpx

    body = completed()
    body["data"]["output"]["usage"] = {"echo": "Bearer fixture-secret"}
    p = D3Policy(
        "fixture-secret",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
        ),
    )
    decision = p.decide(route_view())
    serialized = json.dumps([decision.raw, decision.usage, p.response_events], allow_nan=False)
    assert "fixture-secret" not in serialized
    assert "REDACTED" in serialized
