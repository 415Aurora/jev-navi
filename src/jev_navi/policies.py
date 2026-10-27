"""Local scripted and action-replay policies for software validation."""

import json
from pathlib import Path

from .types import Action, Decision, ServiceError


class ScriptedPolicy:
    calls = 0
    request_count = 0

    def __init__(self, actions):
        self.actions = [Action(x) for x in actions]
        if not self.actions:
            raise ValueError("Script must contain at least one action")
        self.cursor = 0

    def decide(self, view):
        if self.cursor >= len(self.actions):
            raise ServiceError("script_exhausted")
        action = self.actions[self.cursor]
        self.cursor += 1
        return Decision(action=action, model="scripted_fixture", raw={"action": action.value})


class ReplayPolicy:
    calls = 0
    request_count = 0

    def __init__(self, path):
        self.records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
        self.cursor = 0

    def decide(self, view):
        if self.cursor >= len(self.records):
            raise ServiceError("replay_exhausted")
        record = self.records[self.cursor]
        self.cursor += 1
        action = record.get("proposed_action")
        if action is None:
            raise ServiceError(record.get("error_code", "replay_error"))
        # This replays decisions, not a new model measurement. No API calls are made.
        return Decision(
            Action(action), "replay_fixture", record.get("probabilities"), record.get("raw", {})
        )
