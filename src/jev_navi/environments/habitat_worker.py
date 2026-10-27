"""Standalone Python 3.9 worker for unmodified official Habitat-Lab."""

import base64
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Executing this file puts its directory first; sibling habitat.py would shadow
# the official top-level Habitat package. Keep only native environment imports.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).parent.resolve()]


class HabitatEnvironment:
    kind = "habitat"

    def __init__(
        self,
        episode,
        camera,
        *,
        radius_m=0.20,
        agent_height_m=1.5,
        forward_m=0.25,
        turn_deg=15.0,
        allow_sliding=False,
        seed=17,
        depth_pitch_deg=0.0,
    ):
        # Optional native dependencies are imported only when this backend is selected.
        import habitat
        from habitat.config import read_write
        from habitat.core.dataset import Dataset
        from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

        if not np.isfinite(depth_pitch_deg) or not 0 <= depth_pitch_deg <= 60:
            raise ValueError("Depth downward pitch must be between 0 and 60 degrees")
        scene = str(Path(episode["scene"]).expanduser().resolve())
        if not Path(scene).is_file():
            raise ValueError(f"Scene does not exist: {scene}")
        config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
        with read_write(config):
            h = config.habitat
            h.seed = seed
            h.environment.max_episode_steps = 61  # outer loop enforces decision cap incl vetoes
            h.environment.iterator_options.shuffle = False
            h.simulator.forward_step_size = forward_m
            h.simulator.turn_angle = int(turn_deg)
            h.simulator.habitat_sim_v0.allow_sliding = allow_sliding
            h.simulator.habitat_sim_v0.gpu_gpu = False
            agent = h.simulator.agents.main_agent
            agent.radius = radius_m
            agent.height = agent_height_m
            for sensor in agent.sim_sensors.values():
                sensor.width, sensor.height = camera.width, camera.height
                sensor.hfov = int(camera.hfov_deg)
                sensor.position = [0.0, camera.height_m, 0.0]
            agent.sim_sensors.depth_sensor.orientation = [
                -float(np.deg2rad(depth_pitch_deg)),
                0.0,
                0.0,
            ]
            agent.sim_sensors.depth_sensor.normalize_depth = False
            agent.sim_sensors.depth_sensor.min_depth = 0.0
            agent.sim_sensors.depth_sensor.max_depth = 10.0
            # No GPS/point-goal/semantic/oracle sensors. Even the backend does not
            # need a goal observation; the private task measures handle evaluation.
            h.task.lab_sensors = {}
            h.task.measurements.success.success_distance = episode["success_radius"]
            h.task.measurements = {
                k: v for k, v in h.task.measurements.items() if k in ("distance_to_goal", "success")
            }
        dataset = Dataset()
        dataset.episodes = [
            NavigationEpisode(
                episode_id=episode["id"],
                scene_id=scene,
                scene_dataset_config=episode.get("scene_dataset_config", "default"),
                start_position=episode["start_position"],
                start_rotation=episode["start_rotation"],
                goals=[
                    NavigationGoal(
                        position=episode["goal_position"], radius=episode["success_radius"]
                    )
                ],
            )
        ]
        self.env = habitat.Env(config=config, dataset=dataset)
        self.camera = camera
        self.radius = episode["success_radius"]
        self.step = 0

    def _observe(self, raw):
        self._depth = np.asarray(raw["depth"], dtype=np.float32).squeeze().copy()
        return {"rgb": np.asarray(raw["rgb"], dtype=np.uint8)[..., :3].copy(), "step": self.step}

    def _track(self):
        position = np.asarray(self.env.sim.get_agent_state().position, dtype=float)
        if self._last_position is not None:
            self._path_length += float(np.linalg.norm(position - self._last_position))
        self._last_position = position
        distance = float(self.env.get_metrics()["distance_to_goal"])
        self._ever_reached |= np.isfinite(distance) and distance <= self.radius

    def reset(self):
        self.step = 0
        self._path_length, self._collisions, self._ever_reached = 0.0, 0, False
        self._last_position = None
        raw = self.env.reset()
        state = self.env.sim.get_agent_state()
        if not self.env.sim.pathfinder.is_navigable(state.position):
            raise ValueError("Episode start is not navigable for this agent/navmesh")
        distance = float(self.env.get_metrics()["distance_to_goal"])
        if not np.isfinite(distance):
            raise ValueError("Goal is unreachable on this navmesh")
        self._track()
        return self._observe(raw)

    def depth(self):
        return self._depth.copy()

    def execute(self, action):
        action_id = {
            "STOP": "stop",
            "MOVE_FORWARD": "move_forward",
            "TURN_LEFT": "turn_left",
            "TURN_RIGHT": "turn_right",
        }[action]
        raw = self.env.step({"action": action_id})
        collided = bool(self.env.sim.previous_step_collided) if action == "MOVE_FORWARD" else False
        self._collisions += int(collided)
        self.step += 1
        self._track()
        return self._observe(raw), {"collided": collided}

    def evaluate(self, stopped):
        metrics = self.env.get_metrics()
        distance = float(metrics["distance_to_goal"])
        return {
            "success": bool(stopped and distance <= self.radius),
            "ever_reached": bool(self._ever_reached),
            "distance_to_goal": distance,
            "path_length": self._path_length,
            "collisions": self._collisions,
            "position": self._last_position.tolist(),
        }

    def close(self):
        self.env.close()


def pack(value):
    if isinstance(value, np.ndarray):
        return {
            "_array": base64.b64encode(value.tobytes()).decode(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [pack(v) for v in value]
    return value


def main():
    protocol = sys.stdout
    env = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    operation = request["operation"]
                    if operation == "init":
                        env = HabitatEnvironment(
                            request["episode"],
                            SimpleNamespace(**request["camera"]),
                            **request["settings"],
                        )
                        import habitat
                        import habitat_sim

                        result = {
                            "ready": True,
                            "python": sys.version.split()[0],
                            "habitat_lab": habitat.__version__,
                            "habitat_sim": habitat_sim.__version__,
                            "numpy": np.__version__,
                        }
                    elif operation == "reset":
                        result = env.reset()
                    elif operation == "depth":
                        result = env.depth()
                    elif operation == "execute":
                        result = env.execute(request["action"])
                    elif operation == "evaluate":
                        result = env.evaluate(request["stopped"])
                    elif operation == "close":
                        result = None
                    else:
                        raise ValueError("Unknown worker operation")
                    protocol.write(
                        "JEV_RESPONSE "
                        + json.dumps({"result": pack(result)}, allow_nan=False)
                        + "\n"
                    )
                    protocol.flush()
                    if operation == "close":
                        break
                except Exception as error:
                    protocol.write(
                        "JEV_RESPONSE "
                        + json.dumps({"error": type(error).__name__, "message": str(error)})
                        + "\n"
                    )
                    protocol.flush()
    finally:
        if env is not None:
            with contextlib.redirect_stdout(sys.stderr):
                env.close()


if __name__ == "__main__":
    main()
