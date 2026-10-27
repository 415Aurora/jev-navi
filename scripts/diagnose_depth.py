#!/usr/bin/env python3
"""Replay frozen pilot actions; collision labels are evaluator-only, never guard input."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jev_navi.environments.habitat import HabitatEnvironment
from jev_navi.experiments import implementation_digest
from jev_navi.safety import DepthGuard
from jev_navi.types import Action, Camera


def diagnose(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "suite_manifest.json").read_text())
    config = manifest["config"]
    camera = Camera(**config["camera"])
    episodes = {e["id"]: e for e in manifest["suite"]["episodes"]}
    records = []
    for pitch in (0, 30):
        for folder in sorted(source.glob("[0-9]*")):
            if not folder.is_dir():
                continue
            original = json.loads((folder / "manifest.json").read_text())
            episode = episodes[original["metadata"]["episode_id"]]
            settings = dict(config["environment"])
            settings.pop("backend")
            settings["depth_pitch_deg"] = pitch
            env = HabitatEnvironment(episode, camera, **settings)
            guard = DepthGuard(
                camera,
                depth_pitch_deg=pitch,
                radius_m=settings.get("radius_m", 0.2),
                agent_height_m=settings.get("agent_height_m", 1.5),
                margin_m=config.get("safety", {}).get("margin_m", 0.1),
            )
            try:
                observation = env.reset()
                for line in (folder / "steps.jsonl").read_text().splitlines():
                    old = json.loads(line)
                    depth = env.depth()
                    proposed = Action(old["proposed_action"])
                    safety = guard.check(proposed, depth)
                    saved_rgb = np.asarray(Image.open(folder / old["image"]))
                    record = {
                        "episode_id": episode["id"],
                        "step": old["step"],
                        "depth_pitch_deg": pitch,
                        "proposed": proposed.value,
                        "allowed": safety.allowed,
                        "reason": safety.reason,
                        "rgb_matches_original": bool(np.array_equal(observation.rgb, saved_rgb)),
                        "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
                        "original_executed": old["executed_action"],
                        "collision_label": None,
                    }
                    if pitch == 0:
                        record["depth_matches_original"] = bool(
                            np.array_equal(depth, np.load(folder / old["depth"]))
                        )
                    if old["executed_action"] is not None:
                        observation, feedback = env.execute(Action(old["executed_action"]))
                        assert feedback["collided"] == old["collided"], "Replay collision mismatch"
                        if old["executed_action"] == Action.MOVE_FORWARD:
                            record["collision_label"] = feedback["collided"]
                    records.append(record)
                    # Keep first collision evidence independent of the source trajectory.
                    if episode["id"] == "apartment_1-turn-01" and old["step"] == 2:
                        np.save(output / f"first-collision-depth-pitch-{pitch}.npy", depth)
                print(f"pitch={pitch} {episode['id']} replayed", flush=True)
            finally:
                env.close()
    assert all(r["rgb_matches_original"] for r in records), "RGB replay mismatch"
    assert all(r.get("depth_matches_original", True) for r in records), "Depth replay mismatch"
    groups = []
    for pitch in (0, 30):
        labelled = [
            r for r in records if r["depth_pitch_deg"] == pitch and r["collision_label"] is not None
        ]
        groups.append(
            {
                "depth_pitch_deg": pitch,
                "executed_forward_samples": len(labelled),
                "collision_blocked": sum(
                    r["collision_label"] and not r["allowed"] for r in labelled
                ),
                "collision_allowed": sum(r["collision_label"] and r["allowed"] for r in labelled),
                "noncollision_blocked": sum(
                    not r["collision_label"] and not r["allowed"] for r in labelled
                ),
                "noncollision_allowed": sum(
                    not r["collision_label"] and r["allowed"] for r in labelled
                ),
            }
        )
    (output / "checks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = {
        "source": str(source.resolve()),
        "source_implementation_sha256": manifest["implementation_sha256"],
        "implementation_sha256": implementation_digest(),
        "api_calls": 0,
        "all_rgb_match_original": True,
        "horizontal_depth_matches_original": True,
        "groups": groups,
        "limitations": "Same frozen poses/actions, not a new policy rollout. Labels only for "
        "originally executed forward actions; blocked actions lack collision labels. Repeated "
        "collisions and scene reuse are correlated. Noncollision blocking may be conservative "
        "margin rather than an error. Pitch30 uses additional geometric coverage and changes "
        "the sensor protocol. No navigation success improvement is measured.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/startlux-pilot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    diagnose(args.source, args.output)
