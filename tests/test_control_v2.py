import json

from jev_navi.environments.synthetic import SyntheticEnvironment
from jev_navi.policies import ScriptedPolicy
from jev_navi.runner import run_episode
from jev_navi.safety import DepthGuard
from jev_navi.types import Action, Camera


def test_r2rce_can_stop_on_500th_action_and_keeps_legacy_cap(tmp_path):
    camera = Camera(width=8, height=8)
    result = run_episode(
        SyntheticEnvironment(camera),
        ScriptedPolicy(["TURN_LEFT"] * 499 + ["STOP"]),
        DepthGuard(camera),
        "turn and stop",
        tmp_path / "long",
        max_decisions=500,
        protocol_id="r2rce_v13_dev",
        max_steps=500,
    )
    assert result["decisions"] == result["executed_steps"] == 500
    assert result["end_reason"] == "stop"


def test_semantic_transition_does_not_step_or_count_as_depth_block(tmp_path):
    from jev_navi.hierarchy import make_fixture_policy

    fixture = tmp_path / "responses.json"
    fixture.write_text(
        json.dumps(
            [
                {"choices": {"boundary_0": "split"}},
                {"choices": {"progress": "complete", "action": "MOVE_FORWARD"}},
                {"choices": {"progress": "complete", "action": "STOP"}},
            ]
        )
    )
    policy = make_fixture_policy(
        {"kind": "startlux_hierarchical_fixture", "responses": str(fixture)}
    )
    camera = Camera(width=8, height=8)
    result = run_episode(
        SyntheticEnvironment(camera),
        policy,
        DepthGuard(camera),
        "Turn left. Stop here.",
        tmp_path / "transition",
    )
    assert result["executed_steps"] == 1
    assert result["blocked_actions"] == 0
    assert result["http_requests"] == 0
    assert result["fixture_requests"] == 3
    rows = [json.loads(s) for s in (tmp_path / "transition/steps.jsonl").read_text().splitlines()]
    assert rows[0]["executed_action"] is None
    assert rows[0]["guard_reason"] == "subgoal_advanced"
    assert rows[0]["evaluation"]["position"] == rows[1]["evaluation"]["position"]


def test_environment_termination_prevents_further_decisions(tmp_path):
    camera = Camera(width=8, height=8)

    class ShortEnvironment(SyntheticEnvironment):
        def evaluate(self, stopped):
            return {**super().evaluate(stopped), "environment_done": self.step >= 2}

    result = run_episode(
        ShortEnvironment(camera),
        ScriptedPolicy(["TURN_LEFT"] * 4),
        DepthGuard(camera),
        "turn",
        tmp_path / "terminal",
    )
    assert result["executed_steps"] == 2
    assert result["end_reason"] == "environment_budget"


def test_native_distinct_rgb_depth_and_configurable_horizon():
    from pathlib import Path

    import pytest

    from jev_navi.environments.habitat import HabitatEnvironment

    root = Path(__file__).resolve().parents[1]
    if not (root / ".habitat/bin/python").exists():
        pytest.skip("Native environment absent")
    state = json.loads((root / "environment/smoke-result.json").read_text())
    start = state["start_position"]
    env = HabitatEnvironment(
        {
            "id": "v2-camera",
            "scene": str(root / "data/scene_datasets/skokloster-castle.glb"),
            "start_position": start,
            "start_rotation": [0, 0, 0, 1],
            "goal_position": start,
            "success_radius": 3.0,
        },
        Camera(width=224, height=224),
        depth_camera={"width": 256, "height": 256, "hfov_deg": 90, "height_m": 1.25},
        max_episode_steps=500,
        success_strict=True,
    )
    try:
        assert env.reset().rgb.shape == (224, 224, 3)
        assert env.depth().shape == (256, 256)
        for _ in range(62):
            env.execute(Action.TURN_LEFT)
        assert not env.evaluate(False)["environment_done"]
        env.execute(Action.STOP)
        assert env.evaluate(True)["environment_done"]
        with pytest.raises(RuntimeError, match="terminated"):
            env.execute(Action.STOP)
    finally:
        env.close()


def test_synthetic_depth_camera_is_independent():
    env = SyntheticEnvironment(
        Camera(width=224, height=224), depth_camera=Camera(width=32, height=32)
    )
    assert env.reset().rgb.shape == (224, 224, 3)
    assert env.depth().shape == (32, 32)
