import numpy as np
import pytest


def test_depth_obstacle_blocks_forward_but_not_turn():
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Action, Camera

    guard = DepthGuard(Camera(width=32, height=32))
    depth = np.full((32, 32), 0.30, dtype=np.float32)
    assert not guard.check(Action.MOVE_FORWARD, depth).allowed
    assert guard.check(Action.TURN_LEFT, depth).allowed
    assert guard.check(Action.STOP, depth).allowed


def test_depth_clear_and_invalid():
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Action, Camera

    guard = DepthGuard(Camera(width=32, height=32))
    assert guard.check(Action.MOVE_FORWARD, np.full((32, 32), 3.0)).allowed
    assert not guard.check(Action.MOVE_FORWARD, np.full((32, 32), np.nan)).allowed
    assert not guard.check(Action.MOVE_FORWARD, np.zeros((32, 32))).allowed
    with pytest.raises(ValueError):
        guard.check(Action.MOVE_FORWARD, np.ones((16, 32)))


@pytest.mark.parametrize("field", ["radius_m", "agent_height_m", "forward_m", "margin_m"])
def test_nonfinite_guard_geometry_is_rejected(field):
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Camera

    with pytest.raises(ValueError):
        DepthGuard(Camera(), **{field: float("nan")})


def test_policy_view_has_no_ground_truth():
    from dataclasses import fields

    from jev_navi.types import Observation, PolicyInput

    forbidden = {"position", "goal", "depth", "distance_to_goal", "success", "shortest_path"}
    assert not forbidden.intersection(f.name for f in fields(PolicyInput))
    assert not forbidden.intersection(f.name for f in fields(Observation))


def test_camera_geometry_matches_native_integer_configuration():
    from jev_navi.types import Camera

    with pytest.raises(ValueError):
        Camera(hfov_deg=90.5)
    with pytest.raises(ValueError):
        Camera(width=256.5)


def test_only_stop_can_succeed_and_blocked_actions_do_not_move(tmp_path):
    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.policies import ScriptedPolicy
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Action, Camera

    env = SyntheticEnvironment(obstacle_distance=0.30, target=[0, 0, 0])
    result = run_episode(
        env,
        ScriptedPolicy([Action.MOVE_FORWARD] * 3),
        DepthGuard(Camera()),
        "go to door",
        tmp_path / "blocked",
    )
    assert result["end_reason"] == "stuck"
    assert result["success"] is False
    assert result["path_length"] == 0
    assert result["decisions"] == 3


def test_budget_at_goal_is_not_stop(tmp_path):
    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.policies import ScriptedPolicy
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Action, Camera

    env = SyntheticEnvironment(target=[0, 0, 0])
    result = run_episode(
        env,
        ScriptedPolicy([Action.TURN_LEFT]),
        DepthGuard(Camera()),
        "go to door",
        tmp_path / "budget",
        max_decisions=1,
    )
    assert not result["success"]
    assert result["ever_reached"]
    assert result["end_reason"] == "decision_budget"


def test_stop_and_log_history(tmp_path):
    import json

    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.policies import ScriptedPolicy
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Action, Camera

    seen = []

    class Spy(ScriptedPolicy):
        def decide(self, view):
            seen.append(view)
            return super().decide(view)

    result = run_episode(
        SyntheticEnvironment(target=[0, 0, -0.5]),
        Spy([Action.MOVE_FORWARD, Action.MOVE_FORWARD, Action.STOP]),
        DepthGuard(Camera()),
        "door",
        tmp_path / "ok",
    )
    assert result["success"]
    assert [x.step for x in seen[2].frames] == [0, 1, 2]
    assert len(seen[2].history) == 2
    assert result["model_calls"] == 0
    records = [json.loads(x) for x in (tmp_path / "ok" / "steps.jsonl").read_text().splitlines()]
    assert records[-1]["proposed_action"] == "STOP"
    assert records[-1]["executed_action"] == "STOP"
    assert (tmp_path / "ok" / "frames" / "0000.png").exists()


def test_veto_history_uses_decision_frame_ids_and_local_cap(tmp_path):
    import json

    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.policies import ScriptedPolicy
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Action, Camera

    result_dir = tmp_path / "veto-turn"
    run_episode(
        SyntheticEnvironment(obstacle_distance=0.3),
        ScriptedPolicy([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.STOP]),
        DepthGuard(Camera()),
        "door",
        result_dir,
    )
    records = [json.loads(x) for x in (result_dir / "steps.jsonl").read_text().splitlines()]
    assert records[-1]["input_frames"] == [0, 1, 2]
    with pytest.raises(ValueError):
        run_episode(
            SyntheticEnvironment(),
            ScriptedPolicy([Action.STOP]),
            DepthGuard(Camera()),
            "door",
            tmp_path / "too-long",
            max_decisions=61,
        )


def test_failed_service_latency_is_recorded(tmp_path):
    import json

    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Camera, ServiceError

    class Failed:
        calls = 1
        request_count = 1

        def decide(self, view):
            raise ServiceError("timeout")

    root = tmp_path / "timeout"
    result = run_episode(SyntheticEnvironment(), Failed(), DepthGuard(Camera()), "door", root)
    assert len(result["latency_s"]) == 1
    row = json.loads((root / "steps.jsonl").read_text())
    assert row["latency_s"] >= 0


def test_pitched_depth_reprojects_floor_and_low_obstacles():
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Action, Camera

    camera = Camera(width=32, height=32)
    guard = DepthGuard(camera, depth_pitch_deg=45)
    rows, _ = np.indices((32, 32))
    y_per_z = -(rows - 15.5) / 16
    theta = np.deg2rad(45)
    floor = camera.height_m / (np.sin(theta) - y_per_z * np.cos(theta))
    assert guard.check(Action.MOVE_FORWARD, floor).allowed
    obstacle = 0.3 / (np.cos(theta) + y_per_z * np.sin(theta))
    assert not guard.check(Action.MOVE_FORWARD, obstacle).allowed
    for invalid in (float("nan"), -1, 90):
        with pytest.raises(ValueError):
            DepthGuard(camera, depth_pitch_deg=invalid)
