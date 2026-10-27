"""Headless RGB-D/navmesh smoke, run with .habitat/bin/python (Python 3.9)."""
import json
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
scene = root / "data/scene_datasets/skokloster-castle.glb"
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = str(scene)
cfg.enable_physics = False
cfg.gpu_device_id = 0
agent = habitat_sim.agent.AgentConfiguration()
agent.sensor_specifications = []
for name, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = name
    spec.sensor_type = kind
    spec.resolution = [256, 256]
    spec.position = [0.0, 1.5, 0.0]
    agent.sensor_specifications.append(spec)
agent.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
    "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)),
}
with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
    sim.seed(17)
    assert sim.pathfinder.is_loaded, "Scene navmesh was not loaded"
    initial = habitat_sim.AgentState()
    initial.position = sim.pathfinder.get_random_navigable_point()
    sim.initialize_agent(0, initial)
    obs = sim.get_sensor_observations()
    assert obs["rgb"].shape == (256, 256, 4)
    assert obs["depth"].shape == (256, 256)
    assert np.isfinite(obs["depth"]).all() and float(obs["depth"].max()) > 0
    assert float(np.std(obs["rgb"][..., :3])) > 1
    before = sim.get_agent(0).get_state().position.copy()
    sim.step("turn_left")
    for _ in range(4):
        sim.step("move_forward")
    after = sim.get_agent(0).get_state().position.copy()
    distance = float(np.linalg.norm(after - before))
    assert distance > 0.01, "Agent did not move"
    output = root / "environment/smoke-rgb.png"
    Image.fromarray(obs["rgb"][..., :3]).save(output)
    result = {
        "scene": str(scene.relative_to(root)),
        "habitat_sim_version": habitat_sim.__version__,
        "rgb_shape": list(obs["rgb"].shape),
        "depth_shape": list(obs["depth"].shape),
        "depth_min_m": float(obs["depth"].min()),
        "depth_max_m": float(obs["depth"].max()),
        "movement_m": distance,
        "start_position": np.asarray(before).tolist(),
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "final_position": np.asarray(after).tolist(),
        "navmesh_area_m2": float(sim.pathfinder.navigable_area),
    }
    (root / "environment/smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
