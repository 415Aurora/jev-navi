"""Screen existing visible tasks for direct approaches, without model calls."""

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from jev_navi.environments.habitat import HabitatEnvironment
from jev_navi.safety import DepthGuard
from jev_navi.types import Action, Camera


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    suite = yaml.safe_load(Path("configs/local_tasks.yaml").read_text())
    camera = Camera()
    guard = DepthGuard(camera, depth_pitch_deg=30)
    records = []
    for episode in suite["episodes"]:
        if episode["task_type"] != "visible":
            continue
        folder = output / episode["id"]
        folder.mkdir()
        env = HabitatEnvironment(episode, camera, depth_pitch_deg=30)
        try:
            initial = env.reset()
            Image.fromarray(initial.rgb).save(folder / "start.png")
            attempts = []
            for turns in (0, -1, 1, -2, 2):
                observation = env.reset()
                for _ in range(abs(turns)):
                    observation, _ = env.execute(
                        Action.TURN_LEFT if turns > 0 else Action.TURN_RIGHT
                    )
                count = 0
                reason = "step_limit"
                while count < 16:
                    if env.evaluate(False)["distance_to_goal"] <= episode["success_radius"]:
                        reason = "reached"
                        break
                    safety = guard.check(Action.MOVE_FORWARD, env.depth())
                    if not safety.allowed:
                        reason = safety.reason
                        break
                    observation, feedback = env.execute(Action.MOVE_FORWARD)
                    count += 1
                    if feedback["collided"]:
                        reason = "collision"
                        break
                distance = env.evaluate(False)["distance_to_goal"]
                if reason == "step_limit" and distance <= episode["success_radius"]:
                    reason = "reached"
                Image.fromarray(observation.rgb).save(folder / f"end-{turns}.png")
                attempts.append(
                    {
                        "initial_turn_steps": turns,
                        "forward_steps": count,
                        "end_reason": reason,
                        "distance_m": distance,
                    }
                )
            passed = [a for a in attempts if a["end_reason"] == "reached"]
            record = {
                "episode_id": episode["id"],
                "instruction": episode["instruction"],
                "geometry_pass": bool(passed),
                "attempts": attempts,
                "visual_review": "pending",
            }
            records.append(record)
            panel = Image.new("RGB", (512, 284), "white")
            panel.paste(Image.fromarray(initial.rgb), (0, 28))
            choice = passed[0] if passed else attempts[0]
            panel.paste(Image.open(folder / f"end-{choice['initial_turn_steps']}.png"), (256, 28))
            draw = ImageDraw.Draw(panel)
            draw.text((5, 5), "start", fill="black")
            draw.text(
                (261, 5),
                f"{choice['end_reason']} turns={choice['initial_turn_steps']}",
                fill="black",
            )
            panel.save(folder / "review.jpg")
            print(json.dumps(record), flush=True)
            (output / "screen.json").write_text(
                json.dumps(
                    {
                        "api_calls": 0,
                        "records": records,
                        "note": "Offline oracle evaluation only. Geometry does not establish visual clarity.",
                    },
                    indent=2,
                )
            )
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
