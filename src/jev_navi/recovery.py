"""Bounded loop detection from action feedback only; no evaluator access."""

from .types import Action

RECOVERY_MESSAGE = (
    "Recovery attempt 1: four forward attempts have failed since the last non-colliding "
    "forward movement. Turns did not clear the blockage. Repeated forward attempts and "
    "alternating left/right turns have not worked. Use the current visual evidence to "
    "choose a different approach and turn sufficiently to find a route around the obstacle. "
    "Keep pursuing the original instruction; being blocked does not mean the goal is reached. "
    "Choose only one of the same four actions each time."
)


class LoopMonitor:
    def __init__(self, mode="off", failed_forward_limit=4):
        if mode not in ("off", "terminate", "feedback"):
            raise ValueError("Unknown recovery mode")
        if type(failed_forward_limit) is not int or failed_forward_limit != 4:
            raise ValueError("This experiment fixes failed_forward_limit at 4")
        self.mode = mode
        self.failed_forwards = 0
        self.recovery_attempts = 0
        self.context = ""

    def update(self, feedback):
        if self.mode == "off" or feedback.proposed != Action.MOVE_FORWARD:
            return None
        if feedback.executed == Action.MOVE_FORWARD and not feedback.collided:
            self.failed_forwards = 0
            self.context = ""
            return None
        self.failed_forwards += 1
        if self.failed_forwards < 4:
            return None
        if self.mode == "feedback" and self.recovery_attempts == 0:
            self.recovery_attempts = 1
            self.failed_forwards = 0
            self.context = RECOVERY_MESSAGE
            return "recover"
        return "terminate"
