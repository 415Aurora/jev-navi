"""Validate the actual Habitat-Lab Env task API, independent of policy code."""
import json
from pathlib import Path

import habitat
import habitat_sim
import numpy as np
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

root = Path(__file__).resolve().parents[1]
scene = root / "data/scene_datasets/skokloster-castle.glb"
pathfinder = habitat_sim.PathFinder()
pathfinder.load_nav_mesh(str(scene.with_suffix(".navmesh")))
pathfinder.seed(17)
start = np.asarray(pathfinder.get_random_navigable_point()).tolist()
goal = np.asarray(pathfinder.get_random_navigable_point()).tolist()
config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
with habitat.read_write(config):
    config.habitat.dataset.type = ""
    config.habitat.simulator.scene = str(scene)
    config.habitat.simulator.habitat_sim_v0.enable_physics = False
    config.habitat.task.lab_sensors = {}
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = False
episode = NavigationEpisode(
    episode_id="isolated-smoke",
    scene_id=str(scene),
    start_position=start,
    start_rotation=[0, 0, 0, 1],
    goals=[NavigationGoal(position=goal, radius=0.2)],
    info={"geodesic_distance": float(np.linalg.norm(np.asarray(goal) - start))},
)
with habitat.Env(config=config) as env:
    env.current_episode = episode
    obs = env.reset()
    assert set(obs) == {"rgb", "depth"}, set(obs)
    before = env.sim.get_agent_state().position.copy()
    env.step("turn_left")
    for _ in range(4):
        env.step("move_forward")
    after = env.sim.get_agent_state().position.copy()
    assert float(np.linalg.norm(after - before)) > 0.01
    metrics = env.get_metrics()
    result = {
        "habitat_lab_version": habitat.__version__,
        "observation_keys": sorted(obs),
        "rgb_shape": list(obs["rgb"].shape),
        "depth_shape": list(obs["depth"].shape),
        "movement_m": float(np.linalg.norm(after - before)),
        "metrics": metrics,
    }
    (root / "environment/smoke-lab-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
