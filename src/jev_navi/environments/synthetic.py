"""Deterministic software fixture. Never use its results as navigation evidence."""

import numpy as np

from ..types import Action, Camera, Observation


class SyntheticEnvironment:
    kind = "synthetic_fixture"

    def __init__(
        self,
        camera: Camera | None = None,
        obstacle_distance: float = 5.0,
        target=None,
        success_radius: float = 0.20,
        forward_m: float = 0.25,
        turn_deg: float = 15.0,
    ):
        self.camera = camera or Camera()
        self.obstacle_distance = obstacle_distance
        self.target = np.array(target if target is not None else [0, 0, -0.5], dtype=float)
        self.success_radius = success_radius
        self.forward_m = forward_m
        self.turn_deg = turn_deg
        self.reset()

    def _observe(self):
        c = self.camera
        rgb = np.full((c.height, c.width, 3), 100, dtype=np.uint8)
        rgb[:, c.width // 3 : 2 * c.width // 3] = [80, 130, 180]
        return Observation(rgb, self.step)

    def reset(self):
        self.position = np.zeros(3, dtype=float)
        self.yaw = 0.0
        self.step = 0
        self.path_length = 0.0
        self.collisions = 0
        self.ever_reached = False
        self._track()
        return self._observe()

    def _track(self):
        distance = float(np.linalg.norm((self.position - self.target)[[0, 2]]))
        self.ever_reached |= distance <= self.success_radius
        return distance

    def depth(self):
        return np.full(
            (self.camera.height, self.camera.width), self.obstacle_distance, dtype=np.float32
        )

    def execute(self, action):
        collided = False
        if action == Action.MOVE_FORWARD:
            collided = self.obstacle_distance <= self.forward_m
            if not collided:
                self.position += self.forward_m * np.array(
                    [-np.sin(self.yaw), 0, -np.cos(self.yaw)]
                )
                self.path_length += self.forward_m
        elif action == Action.TURN_LEFT:
            self.yaw += np.deg2rad(self.turn_deg)
        elif action == Action.TURN_RIGHT:
            self.yaw -= np.deg2rad(self.turn_deg)
        self.collisions += int(collided)
        self.step += 1
        self._track()
        return self._observe(), {"collided": collided}

    def evaluate(self, stopped):
        distance = self._track()
        return {
            "success": bool(stopped and distance <= self.success_radius),
            "ever_reached": bool(self.ever_reached),
            "distance_to_goal": distance,
            "path_length": self.path_length,
            "collisions": self.collisions,
            "position": self.position.tolist(),
        }

    def close(self):
        pass
