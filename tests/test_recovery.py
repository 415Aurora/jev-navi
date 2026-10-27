import pytest

from jev_navi.types import Action, Feedback


def test_loop_counts_failed_forward_across_turns_but_resets_on_progress():
    from jev_navi.recovery import LoopMonitor

    monitor = LoopMonitor(mode="terminate", failed_forward_limit=4)
    blocked = Feedback(Action.MOVE_FORWARD, None, "depth_obstacle")
    turn = Feedback(Action.TURN_LEFT, Action.TURN_LEFT, "not_forward")
    for _ in range(3):
        assert monitor.update(blocked) is None
        assert monitor.update(turn) is None
    assert monitor.update(blocked) == "terminate"
    monitor.update(Feedback(Action.MOVE_FORWARD, Action.MOVE_FORWARD, "depth_clear"))
    assert monitor.failed_forwards == 0


def test_feedback_recovery_is_bounded_and_collisions_count_as_failures():
    from jev_navi.recovery import LoopMonitor

    monitor = LoopMonitor(mode="feedback", failed_forward_limit=4)
    collision = Feedback(Action.MOVE_FORWARD, Action.MOVE_FORWARD, "depth_clear", True)
    events = [monitor.update(collision) for _ in range(8)]
    assert events == [None, None, None, "recover", None, None, None, "terminate"]
    assert monitor.recovery_attempts == 1


@pytest.mark.parametrize("mode,limit", [("unknown", 4), ("terminate", 0), ("feedback", True)])
def test_invalid_loop_settings(mode, limit):
    from jev_navi.recovery import LoopMonitor

    with pytest.raises(ValueError):
        LoopMonitor(mode=mode, failed_forward_limit=limit)


def test_runner_recovery_context_and_loop_end_do_not_become_stop(tmp_path):
    from jev_navi.environments.synthetic import SyntheticEnvironment
    from jev_navi.policies import ScriptedPolicy
    from jev_navi.runner import run_episode
    from jev_navi.safety import DepthGuard
    from jev_navi.types import Camera

    class CapturePolicy(ScriptedPolicy):
        def __init__(self):
            super().__init__([Action.MOVE_FORWARD, Action.TURN_LEFT] * 8)
            self.views = []

        def decide(self, view):
            self.views.append(view)
            return super().decide(view)

    policy = CapturePolicy()
    result = run_episode(
        SyntheticEnvironment(obstacle_distance=0.3),
        policy,
        DepthGuard(Camera()),
        "go to door",
        tmp_path / "feedback",
        max_decisions=30,
        recovery={"mode": "feedback", "failed_forward_limit": 4},
    )
    assert result["end_reason"] == "loop_detected"
    assert result["recovery_attempts"] == 1
    assert result["decisions"] == 15
    assert not result["success"]
    assert not policy.views[0].recovery_context
    assert policy.views[7].recovery_context


def test_recovery_context_is_sent_to_model_without_replacing_instruction():
    import numpy as np

    from jev_navi.providers import state_text
    from jev_navi.types import Observation, PolicyInput

    view = PolicyInput(
        "original goal",
        (Observation(np.zeros((8, 8, 3), dtype=np.uint8), 0),),
        (),
        recovery_context="Four forward failures; change approach.",
    )
    text = state_text(view, image_markers=True)
    assert "original goal" in text
    assert "Four forward failures; change approach." in text
