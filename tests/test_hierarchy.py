import json

import httpx
import numpy as np
import pytest

from jev_navi.types import Action, Observation, PolicyInput, ServiceError


def view(instruction="Walk to the door. Then turn left and stop."):
    return PolicyInput(instruction, (Observation(np.zeros((8, 8, 3), dtype=np.uint8), 0),), ())


def response(questions, choices):
    return {
        "model": "startlux-fixture",
        "answers": {
            name: {"probabilities": {k: float(k == choices[name]) for k in q["criteria"]}}
            for name, q in questions.items()
        },
    }


def test_hierarchical_progress_uses_real_choice_wire_and_never_executes_old_action():
    from jev_navi.hierarchy import HierarchicalPolicy

    requests = []
    sequence = [
        {"progress": "complete", "action": "MOVE_FORWARD"},
        {"progress": "continue", "action": "STOP"},
        {"progress": "complete", "action": "STOP"},
    ]

    def serve(request):
        payload = json.loads(request.content)
        requests.append(payload)
        questions = payload["questions"]
        if "progress" not in questions:
            choices = {k: "split" for k in questions}
        else:
            choices = sequence.pop(0)
        return httpx.Response(200, json=response(questions, choices))

    policy = HierarchicalPolicy(
        "fixture", client=httpx.Client(transport=httpx.MockTransport(serve))
    )
    # A period and "Then" at the same source boundary must not create an empty fragment.
    first = policy.decide(view("Walk to the door. Turn left and stop."))
    assert first.action == Action.MOVE_FORWARD
    assert not first.execute
    assert first.control_reason == "subgoal_advanced"
    assert first.semantic_state["current_index"] == 1
    second = policy.decide(view("Walk to the door. Turn left and stop."))
    assert not second.execute and second.control_reason == "stop_not_confirmed"
    third = policy.decide(view("Walk to the door. Turn left and stop."))
    assert third.execute and third.action == Action.STOP
    assert policy.calls == 4
    assert all(p["think_mode"] == "auto" for p in requests)
    assert all(p["state"].count("<image>") == len(p["images"]) for p in requests)
    assert "Current subgoal" in requests[-1]["state"]


def test_source_spans_cover_original_and_complex_instructions_are_not_truncated():
    from jev_navi.hierarchy import candidate_boundaries, source_segments

    instruction = "Go forward; then turn left. Stop."
    boundaries = candidate_boundaries(instruction)
    segments = source_segments(instruction, boundaries)
    assert "".join(instruction[a:b] for a, b in segments) == instruction
    assert all(a < b for a, b in segments)
    with pytest.raises(ValueError, match="boundary"):
        candidate_boundaries("Turn. " * 40)
    with pytest.raises(ValueError):
        source_segments(instruction, (len(instruction) + 1,))


def test_hierarchy_rejects_invalid_progress_distribution():
    from jev_navi.hierarchy import HierarchicalPolicy

    def serve(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json=response(payload["questions"], {"progress": "complete", "action": "STOP"})
            | {"answers": {"progress": {"probabilities": {"made_up": 1}}}},
        )

    p = HierarchicalPolicy("fixture", client=httpx.Client(transport=httpx.MockTransport(serve)))
    with pytest.raises(ServiceError, match="invalid_response"):
        p.decide(view("Stop here"))


def test_fixture_questions_must_match_declared_responses(tmp_path):
    from jev_navi.hierarchy import make_fixture_policy

    path = tmp_path / "responses.json"
    path.write_text(json.dumps([{"choices": {"progress": "complete", "action": "STOP"}}]))
    p = make_fixture_policy({"kind": "startlux_hierarchical_fixture", "responses": str(path)})
    assert p.decide(view("Stop here")).execute
    assert p.is_fixture
    assert p.request_count == 1


def test_evaluator_canary_never_enters_wire_or_memory(tmp_path):
    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.hierarchy import HierarchicalPolicy
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Camera

    payloads = []
    canary = "PRIVATE_EVALUATION_SENTINEL_104729"

    class PrivateEnvironment(SyntheticEnvironment):
        def evaluate(self, stopped):
            return {**super().evaluate(stopped), "private_annotation": canary}

    def serve(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(
            200, json=response(payload["questions"], {"progress": "complete", "action": "STOP"})
        )

    camera = Camera(width=8, height=8)
    policy = HierarchicalPolicy(
        "fixture", client=httpx.Client(transport=httpx.MockTransport(serve))
    )
    run_episode(
        PrivateEnvironment(camera), policy, DepthGuard(camera), "Stop here", tmp_path / "private"
    )
    assert canary in (tmp_path / "private/steps.jsonl").read_text()
    assert canary not in json.dumps(payloads)
    assert canary not in json.dumps(policy.memory_snapshot())


def test_depth_rejection_does_not_complete_current_subgoal(tmp_path):
    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.hierarchy import make_fixture_policy
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Camera

    fixture = tmp_path / "blocked.json"
    fixture.write_text(
        json.dumps(
            [{"choices": {"boundary_0": "split"}}]
            + [{"choices": {"progress": "continue", "action": "MOVE_FORWARD"}}] * 3
        )
    )
    policy = make_fixture_policy(
        {"kind": "startlux_hierarchical_fixture", "responses": str(fixture)}
    )
    camera = Camera(width=8, height=8)
    result = run_episode(
        SyntheticEnvironment(camera, obstacle_distance=0.3),
        policy,
        DepthGuard(camera),
        "Approach the door. Stop.",
        tmp_path / "blocked",
    )
    assert result["end_reason"] == "stuck"
    assert result["executed_steps"] == 0
    assert result["blocked_actions"] == 3
    assert policy.memory_snapshot()["current_index"] == 0
    assert policy.memory_snapshot()["completed_indices"] == []


def test_leading_whitespace_and_terminal_connector_do_not_create_empty_stages():
    from jev_navi.hierarchy import candidate_boundaries, source_segments

    instruction = "  Then go left.  "
    assert candidate_boundaries(instruction) == ()
    spans = source_segments(instruction, candidate_boundaries(instruction))
    assert "".join(instruction[a:b] for a, b in spans) == instruction
    assert all(instruction[a:b].strip() for a, b in spans)


@pytest.mark.parametrize("action_fails", [False, True])
def test_planning_response_and_model_survive_action_failure(tmp_path, action_fails):
    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.hierarchy import HierarchicalPolicy
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Camera

    def serve(request):
        payload = json.loads(request.content)
        questions = payload["questions"]
        if "progress" not in questions:
            raw = response(questions, {key: "merge" for key in questions})
            raw["model"] = "planner-version-a"
        elif action_fails:
            return httpx.Response(503)
        else:
            raw = response(questions, {"progress": "complete", "action": "STOP"})
            raw["model"] = "actor-version-b"
        return httpx.Response(200, json=raw)

    camera = Camera(width=8, height=8)
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        policy = HierarchicalPolicy("fixture", client=client)
        result = run_episode(
            SyntheticEnvironment(camera),
            policy,
            DepthGuard(camera),
            "Turn left. Stop here.",
            tmp_path / "versions",
            protocol_id="r2rce_v13_dev",
        )
    expected = ["planner-version-a"] if action_fails else ["actor-version-b", "planner-version-a"]
    assert result["models"] == expected
    record = json.loads((tmp_path / "versions/steps.jsonl").read_text().splitlines()[0])
    assert record["response_events"][0]["raw"]["model"] == "planner-version-a"
    if action_fails:
        assert result["end_reason"] == "service_error"
        assert result["benchmark_metrics"]["sr"] == result["benchmark_metrics"]["spl"] == 0
