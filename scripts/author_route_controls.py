#!/usr/bin/env python3
"""Freeze privileged route controls with .habitat/bin/python (never model evidence)."""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace


def map_actions(actions):
    mapping = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
    names = []
    for action in actions:
        if action is None:
            names.append("STOP")
        elif type(action) is int and action in mapping:
            names.append(mapping[action])
        else:
            raise ValueError("Unknown native action code")
    if not 2 <= len(names) <= 499 or names[-1] != "STOP" or "STOP" in names[:-1]:
        raise ValueError("Need a terminal STOP and room for the semantic transition")
    return names


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def author_routes(catalog_path, scenes_root, output):
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    catalog_path, scenes_root, output = (
        Path(catalog_path),
        Path(scenes_root).resolve(),
        Path(output),
    )
    if output.exists():
        raise FileExistsError(output)
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("kind") != "privileged_software_fixture" or not catalog.get("routes"):
        raise ValueError("Expected privileged software fixture targets")
    worker_path = root / "src/jev_navi/environments/habitat_worker.py"
    spec = importlib.util.spec_from_file_location("route_author_worker", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    records, seen = [], set()
    for route in catalog["routes"]:
        identity = route.get("id", "")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
            raise ValueError("Route id must be a safe filename component")
        if identity in seen:
            raise ValueError("Route IDs must be unique")
        seen.add(identity)
        relative = route.get("scene_id", "")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("Invalid scene path")
        scene = (scenes_root / relative).resolve()
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not scene.is_relative_to(scenes_root)
        ):
            raise ValueError("Scene path escapes scenes root")
        env = worker.HabitatEnvironment(
            {
                "id": identity,
                "scene": str(scene),
                "start_position": route["start_position"],
                "start_rotation": route["start_rotation"],
                "goal_position": route["goal_position"],
                "success_radius": 3.0,
            },
            SimpleNamespace(width=224, height=224, hfov_deg=90, height_m=1.25),
            radius_m=0.1,
            agent_height_m=1.5,
            allow_sliding=True,
            seed=17,
            max_episode_steps=500,
            success_strict=True,
            depth_pitch_deg=0,
            depth_camera={"width": 256, "height": 256, "hfov_deg": 90, "height_m": 1.25},
        )
        try:
            env.reset()
            distance = env.evaluate(False)["distance_to_goal"]
            if not math.isfinite(distance) or distance <= 3:
                raise ValueError("Control needs a reachable initial distance greater than 3m")
            native_actions = env.env.sim.make_greedy_follower(goal_radius=2.5).find_path(
                np.array(route["goal_position"], dtype=np.float32)
            )
            actions = map_actions(native_actions)
            # Explicit reset avoids relying on the follower's internal restoration behavior.
            env.reset()
            points = [env.evaluate(False)["position"]]
            for action in actions:
                env.execute(action)
                points.append(env.evaluate(action == "STOP")["position"])
            final = env.evaluate(True)
            if not final["success"]:
                raise ValueError("Frozen oracle replay did not reach the strict success radius")
            records.append(
                {
                    **route,
                    "scene_sha256": sha(scene),
                    "actions": actions,
                    "gt_locations": points,
                    "initial_geodesic_distance": distance,
                    "final_geodesic_distance": final["distance_to_goal"],
                    "author_collisions": final["collisions"],
                }
            )
        finally:
            env.close()
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
    )
    result = {
        "schema_version": 1,
        "kind": "privileged_software_fixture",
        "protocol_id": "r2rce_v13_dev",
        "routes": records,
        "provenance": {
            "git": {"commit": commit, "dirty": dirty},
            "author_sha256": sha(__file__),
            "native_worker_sha256": sha(worker_path),
            "target_catalog_sha256": sha(catalog_path),
            "follower_goal_radius": 2.5,
            "environment": json.loads((root / "environment/source-manifest.json").read_text())[
                "installed"
            ],
            "note": "Privileged native greedy-follower authoring. Actions and dense positions "
            "are software references, not human annotations or model-generated trajectories.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"written": str(output), "routes": len(records), "http_requests": 0}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--scenes-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    author_routes(args.catalog, args.scenes_root, args.output)


if __name__ == "__main__":
    main()
