#!/usr/bin/env python3
"""Native Habitat annotation preview; never marks annotations verified automatically.

Run with .habitat/bin/python3.9 scripts/annotate_tasks.py survey, or preview.
Survey outputs camera poses and four cardinal views at sampled reachable positions.
Preview renders frozen start/goal views and checks navmesh reachability for review.
"""
import argparse
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image, ImageDraw
from habitat_sim.utils.common import quat_from_coeffs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/annotation-preview"
SCENES = ("skokloster-castle", "van-gogh-room", "apartment_1")


def simulator(scene):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(ROOT / "data/scene_datasets" / (scene + ".glb"))
    cfg.enable_physics = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.radius = 0.2
    agent.height = 1.5
    sensors = []
    for name, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = name
        spec.sensor_type = kind
        spec.resolution = [256, 256]
        spec.position = [0.0, 1.25, 0.0]
        spec.hfov = 90
        sensors.append(spec)
    agent.sensor_specifications = sensors
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = 0.2
    settings.agent_height = 1.5
    assert sim.recompute_navmesh(sim.pathfinder, settings), "navmesh recompute failed"
    sim.seed(421)
    return sim


def rotation(yaw):
    a = math.radians(yaw) / 2
    return [0.0, math.sin(a), 0.0, math.cos(a)]


def render(sim, position, quaternion):
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = quat_from_coeffs(quaternion)
    sim.get_agent(0).set_state(state)
    return Image.fromarray(sim.get_sensor_observations()["rgb"][..., :3])


def montage(images, labels, destination, columns=4):
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * 256, rows * 282), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(images, labels)):
        x, y = (i % columns) * 256, (i // columns) * 282
        sheet.paste(im, (x, y))
        draw.text((x + 3, y + 258), label, fill="black")
    sheet.save(destination)


def survey():
    OUT.mkdir(parents=True, exist_ok=True)
    record = {}
    for scene in SCENES:
        with simulator(scene) as sim:
            pts = []
            for _ in range(2000):
                p = np.array(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
                if p[1] > {"skokloster-castle": 0.4, "van-gogh-room": 0.3, "apartment_1": -1.3}[scene]:
                    continue
                if sim.pathfinder.island_radius(p) < 0.8:
                    continue
                if all(np.linalg.norm(p - q) > (0.7 if scene == "van-gogh-room" else 2) for q in pts):
                    pts.append(p)
                if len(pts) == 6:
                    break
            if not pts:
                pts = [np.array(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)]
            images, labels, poses = [], [], []
            for i, p in enumerate(pts):
                for yaw in (0, 90, 180, 270):
                    q = rotation(yaw)
                    images.append(render(sim, p, q))
                    labels.append(f"{i} yaw={yaw} pos=" + ",".join(f"{v:.1f}" for v in p))
                    poses.append(dict(index=i, yaw=yaw, position=p.tolist(), rotation=q))
            montage(images, labels, OUT / (scene + "-survey.jpg"))
            record[scene] = dict(area=sim.pathfinder.navigable_area, bounds=[list(v) for v in sim.pathfinder.get_bounds()], poses=poses)
            print(scene, "area", sim.pathfinder.navigable_area, "points", len(pts))
    (OUT / "survey.json").write_text(json.dumps(record, indent=2) + "\n")


def preview(config):
    import yaml
    OUT.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(Path(config).read_text())
    report = []
    for scene in SCENES:
        rows = [r for r in data["episodes"] if r["scene_id"] == scene]
        if not rows:
            continue
        with simulator(scene) as sim:
            images, labels = [], []
            for row in rows:
                start = np.array(row["start_position"], dtype=np.float32)
                goal = np.array(row["goal_position"], dtype=np.float32)
                path = habitat_sim.ShortestPath()
                path.requested_start, path.requested_end = start, goal
                found = sim.pathfinder.find_path(path)
                images.append(render(sim, start, row["start_rotation"]))
                labels.append(row["id"][len(scene) + 1:] + " start")
                images.append(render(sim, goal, row.get("goal_rotation", row["start_rotation"])))
                labels.append(row["id"][len(scene) + 1:] + " goal")
                # Offline authoring oracle only: goal coordinates never go to a policy.
                render(sim, start, row["start_rotation"])
                oracle = {"oracle_actions": None, "oracle_within_60_decisions": False}
                try:
                    follower = sim.make_greedy_follower(goal_radius=float(row["success_radius"]))
                    actions = follower.find_path(goal)
                    for action in actions:
                        if action is not None:
                            sim.get_agent(0).act(action)
                    final = habitat_sim.ShortestPath()
                    final.requested_start = sim.get_agent(0).get_state().position
                    final.requested_end = goal
                    final_found = sim.pathfinder.find_path(final)
                    oracle = dict(oracle_actions=len(actions), oracle_within_60_decisions=bool(len(actions) <= 60 and final_found and final.geodesic_distance <= row["success_radius"]), oracle_final_geodesic=float(final.geodesic_distance))
                except habitat_sim.errors.GreedyFollowerError:
                    oracle["oracle_error"] = "GreedyFollowerError"
                report.append(dict(id=row["id"], **oracle, start_navigable=sim.pathfinder.is_navigable(start), goal_navigable=sim.pathfinder.is_navigable(goal), path_found=found, geodesic_distance=float(path.geodesic_distance), start_outside_success_radius=bool(np.linalg.norm(start - goal) > row["success_radius"]), start_goal_distance=float(np.linalg.norm(start - goal))))
            montage(images, labels, OUT / (scene + "-tasks.jpg"))
    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(r["start_navigable"] and r["goal_navigable"] and r["path_found"] and r["start_outside_success_radius"] for r in report):
        raise SystemExit("Annotation geometry validation failed; inspect validation.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("survey", "preview"))
    parser.add_argument("--config", default=str(ROOT / "configs/local_tasks.yaml"))
    args = parser.parse_args()
    survey() if args.command == "survey" else preview(args.config)
