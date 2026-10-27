"""Narrow policy interfaces: evaluator state must never enter PolicyInput."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np


class Action(StrEnum):
    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"


@dataclass(frozen=True)
class Camera:
    width: int = 256
    height: int = 256
    hfov_deg: float = 90.0
    height_m: float = 1.25

    def __post_init__(self):
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or self.width < 8
            or self.height < 8
            or not 0 < self.hfov_deg < 180
            or int(self.hfov_deg) != self.hfov_deg
        ):
            raise ValueError("Invalid camera dimensions or HFOV")
        if not np.isfinite(self.height_m) or self.height_m <= 0:
            raise ValueError("Camera height must be positive")


@dataclass(frozen=True)
class Observation:
    rgb: np.ndarray
    step: int


@dataclass(frozen=True)
class Feedback:
    proposed: Action
    executed: Action | None
    reason: str
    collided: bool = False


@dataclass(frozen=True)
class PolicyInput:
    instruction: str
    frames: tuple[Observation, ...]
    history: tuple[Feedback, ...]
    recovery_context: str = ""


@dataclass(frozen=True)
class Decision:
    action: Action
    model: str
    probabilities: dict[str, float] | None = None
    raw: dict = field(default_factory=dict)
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)


class ServiceError(Exception):
    """Stable failure code; messages must not contain credentials or request headers."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


class Policy(Protocol):
    calls: int
    request_count: int

    def decide(self, view: PolicyInput) -> Decision: ...


class Environment(Protocol):
    kind: str

    def reset(self) -> Observation: ...
    def depth(self) -> np.ndarray: ...
    def execute(self, action: Action) -> tuple[Observation, dict]: ...
    def evaluate(self, stopped: bool) -> dict: ...
    def close(self) -> None: ...
