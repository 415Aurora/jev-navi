#!/usr/bin/env python3
"""Counterfactual termination replay. No fresh model navigation evidence."""

import argparse
import json
from pathlib import Path

from jev_navi.experiments import make_environment
from jev_navi.policies import ReplayPolicy
from jev_navi.reporting import write_report
from jev_navi.runner import run_episode, write_json
from jev_navi.safety import DepthGuard
from jev_navi.types import Camera


def diagnose(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "suite_manifest.json").read_text())
    config = manifest["config"]
    camera = Camera(**config["camera"])
    episodes = {e["id"]: e for e in manifest["suite"]["episodes"]}
    rows = []
    for folder in sorted(source.glob("[0-9]*")):
        original = json.loads((folder / "result.json").read_text())
        episode = episodes[original["metadata"]["episode_id"]]
        env = make_environment(config["environment"], episode, camera, 17)
        try:
            result = run_episode(
                env,
                ReplayPolicy(folder / "steps.jsonl"),
                DepthGuard(camera, depth_pitch_deg=config["environment"].get("depth_pitch_deg", 0)),
                episode["instruction"],
                output / folder.name,
                max_decisions=30,
                recovery={"mode": "terminate", "failed_forward_limit": 4},
                metadata={**original["metadata"], "policy_kind": "replay"},
            )
            source_steps = [
                json.loads(x) for x in (folder / "steps.jsonl").read_text().splitlines()
            ]
            new_steps = [
                json.loads(x)
                for x in (output / folder.name / "steps.jsonl").read_text().splitlines()
            ]
            for old, new in zip(source_steps, new_steps):
                assert old["executed_action"] == new["executed_action"]
                assert old["guard_reason"] == new["guard_reason"]
                assert old["evaluation"] == new["evaluation"]
            rows.append(
                {
                    "episode_id": episode["id"],
                    "baseline_decisions": original["decisions"],
                    "replay_decisions": result["decisions"],
                    "end_reason": result["end_reason"],
                    "counterfactual_avoided_requests": original["decisions"] - result["decisions"],
                }
            )
        finally:
            env.close()
    write_json(
        output / "termination_comparison.json",
        {
            "source": str(source.resolve()),
            "api_calls": 0,
            "rows": rows,
            "counterfactual_avoided_requests": sum(
                r["counterfactual_avoided_requests"] for r in rows
            ),
            "note": "Frozen response replay; request savings are counterfactual, not fresh model measurements.",
        },
    )
    write_report(output)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/startlux-depth30-pilot")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    diagnose(args.source, args.output)
