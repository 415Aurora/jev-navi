"""Sensor-only forward swept-corridor heuristic, not a safety certificate."""

from dataclasses import dataclass

import numpy as np

from .types import Action, Camera


@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    reason: str


class DepthGuard:
    def __init__(
        self,
        camera: Camera,
        radius_m: float = 0.20,
        agent_height_m: float = 1.5,
        forward_m: float = 0.25,
        margin_m: float = 0.10,
        enabled: bool = True,
        depth_pitch_deg: float = 0.0,
    ):
        if (
            not all(np.isfinite(x) for x in (radius_m, agent_height_m, forward_m, margin_m))
            or min(radius_m, agent_height_m, forward_m) <= 0
            or margin_m < 0
        ):
            raise ValueError("Invalid geometry parameters")
        if not np.isfinite(depth_pitch_deg) or not 0 <= depth_pitch_deg <= 60:
            raise ValueError("Depth downward pitch must be between 0 and 60 degrees")
        self.depth_pitch_deg = depth_pitch_deg
        self.camera = camera
        self.radius = radius_m
        self.agent_height = agent_height_m
        self.forward = forward_m
        self.margin = margin_m
        self.enabled = enabled

    def check(self, action: Action, depth: np.ndarray) -> SafetyResult:
        if action != Action.MOVE_FORWARD or not self.enabled:
            return SafetyResult(True, "not_forward" if self.enabled else "guard_disabled")
        c = self.camera
        z = np.asarray(depth, dtype=float)
        if z.shape != (c.height, c.width):
            raise ValueError("Depth shape does not match camera; expected axial depth in metres")
        fx = c.width / (2 * np.tan(np.deg2rad(c.hfov_deg) / 2))
        rows, cols = np.indices(z.shape)
        # Habitat's pinhole depth is axial distance, not Euclidean ray range.
        valid = np.isfinite(z) & (z > 0)
        with np.errstate(invalid="ignore", over="ignore"):
            x = (cols - (c.width - 1) / 2) * z / fx
            y = -(rows - (c.height - 1) / 2) * z / fx
            theta = np.deg2rad(self.depth_pitch_deg)
            height = c.height_m + y * np.cos(theta) - z * np.sin(theta)
            forward = z * np.cos(theta) + y * np.sin(theta)
        corridor = (np.abs(x) <= self.radius + self.margin) & (height >= 0.08)
        corridor &= height <= self.agent_height
        blocked = (
            valid & corridor & (forward > 0) & (forward <= self.forward + self.radius + self.margin)
        )
        if np.any(blocked):
            return SafetyResult(False, "depth_obstacle")
        # Fail closed on missing central evidence; floor/ceiling need not be valid.
        central = valid[c.height // 3 : 2 * c.height // 3, c.width // 3 : 2 * c.width // 3]
        if central.mean() < 0.8:
            return SafetyResult(False, "depth_invalid")
        return SafetyResult(True, "depth_clear")
