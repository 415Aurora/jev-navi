"""Offline privileged STOP audit; run with the isolated Habitat interpreter."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_coeffs
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def audit(source, output):
    spec = importlib.util.spec_from_file_location(
        "worker", ROOT / "src/jev_navi/environments/habitat_worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "suite_manifest.json").read_text())
    episodes = {e["id"]: e for e in manifest["suite"]["episodes"]}
    records = []
    for folder in sorted(source.glob("[0-9]*")):
        result = json.loads((folder / "result.json").read_text())
        if result["end_reason"] != "stop":
            continue
        ep = episodes[result["metadata"]["episode_id"]]
        dest = output / ep["id"]
        dest.mkdir()
        settings = dict(manifest["config"]["environment"])
        settings.pop("backend")
        env = worker.HabitatEnvironment(
            ep, SimpleNamespace(**manifest["config"]["camera"]), **settings
        )
        try:
            obs = env.reset()
            for line in (folder / "steps.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["executed_action"] not in (None, "STOP"):
                    obs, feedback = env.execute(row["executed_action"])
                    assert feedback["collided"] == row["collided"]
            sim = env.env.sim
            assert np.allclose(sim.get_agent_state().position, result["position"])
            stop_rgb = obs["rgb"]
            goal = np.array(ep["goal_position"], dtype=np.float32)
            follower = sim.make_greedy_follower(goal_radius=float(ep["success_radius"]))
            actions = follower.find_path(goal)
            route = []
            for i, action in enumerate(actions):
                if action is None:
                    continue
                current = sim.get_sensor_observations()
                np.save(dest / f"route-{i:03d}-depth.npy", current["depth"].squeeze())
                before = sim.get_agent_state().position.copy()
                sim.step(action)
                route.append(
                    {
                        "action": str(action),
                        "displacement_m": float(
                            np.linalg.norm(sim.get_agent_state().position - before)
                        ),
                        "collided": bool(sim.previous_step_collided),
                    }
                )
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = sim.get_agent_state().position, goal
            assert sim.pathfinder.find_path(path)
            arrival = sim.get_sensor_observations()["rgb"][..., :3]
            state = habitat_sim.AgentState()
            state.position, state.rotation = goal, quat_from_coeffs(ep["goal_rotation"])
            sim.get_agent(0).set_state(state)
            reference = sim.get_sensor_observations()["rgb"][..., :3]
            panel = Image.new("RGB", (768, 284), "white")
            draw = ImageDraw.Draw(panel)
            for j, (label, rgb) in enumerate(
                (
                    ("model STOP", stop_rgb),
                    ("oracle arrival", arrival),
                    ("annotation goal", reference),
                )
            ):
                Image.fromarray(rgb).save(dest / (label.replace(" ", "-") + ".png"))
                panel.paste(Image.fromarray(rgb), (j * 256, 28))
                draw.text((j * 256 + 5, 5), label, fill="black")
            panel.save(dest / "comparison.jpg")
            record = {
                "episode_id": ep["id"],
                "instruction": ep["instruction"],
                "stop_distance_m": result["distance_to_goal"],
                "success_radius_m": ep["success_radius"],
                "oracle_actions": route,
                "oracle_final_distance_m": float(path.geodesic_distance),
                "oracle_reaches_region": bool(path.geodesic_distance <= ep["success_radius"]),
                "action_space": {
                    str(k): v.name for k, v in sim.get_agent(0).agent_config.action_space.items()
                },
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        finally:
            env.close()
    (output / "audit.json").write_text(
        json.dumps(
            {
                "api_calls": 0,
                "source": str(source.resolve()),
                "records": records,
                "note": "Privileged goals for offline evaluation only; oracle is not model capability.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/startlux-depth30-pilot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.source, args.output)
