"""Frozen terminal observations: image history x execution feedback diagnostic."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from jev_navi.credentials import load_local_env
from jev_navi.experiments import make_policy
from jev_navi.providers import CallBudget
from jev_navi.runner import write_json
from jev_navi.types import Action, Feedback, Observation, PolicyInput

ROOT = Path(__file__).resolve().parents[1]


def main():
    load_local_env(ROOT / ".env")
    source = ROOT / "runs/startlux-depth30-pilot"
    output = ROOT / "runs/stop-history-probe-01"
    output.mkdir(parents=True, exist_ok=False)
    budget = CallBudget(8)
    policy = make_policy({"kind": "startlux"}, budget)
    records = []
    try:
        with (output / "responses.jsonl").open("w") as handle:
            for folder in sorted(source.glob("[0-9]*")):
                result = json.loads((folder / "result.json").read_text())
                if result["end_reason"] != "stop":
                    continue
                rows = [
                    json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()
                ]
                last = rows[-1]
                frames = tuple(
                    Observation(np.asarray(Image.open(folder / rows[i]["image"]).convert("RGB")), i)
                    for i in last["input_frames"]
                )
                history = tuple(
                    Feedback(
                        Action(x["proposed"]),
                        Action(x["executed"]) if x["executed"] else None,
                        x["reason"],
                        x["collided"],
                    )
                    for x in last["history"]
                )
                instruction = json.loads((folder / "manifest.json").read_text())["instruction"]
                for images in (False, True):
                    for actions in (False, True):
                        view = PolicyInput(
                            instruction,
                            frames if images else frames[-1:],
                            history if actions else (),
                        )
                        decision = policy.decide(view)
                        record = {
                            "episode_id": result["metadata"]["episode_id"],
                            "image_history": images,
                            "action_feedback": actions,
                            "frame_steps": [f.step for f in view.frames],
                            "action": decision.action.value,
                            "probabilities": decision.probabilities,
                            "raw": decision.raw,
                            "matches_original_probabilities": decision.probabilities
                            == last["probabilities"],
                        }
                        records.append(record)
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        print(
                            json.dumps({k: v for k, v in record.items() if k != "raw"}), flush=True
                        )
    finally:
        policy.close()
        write_json(
            output / "summary.json",
            {
                "http_attempts": budget.used,
                "records": records,
                "note": "Two selected failed STOP states, four input combinations each; no navigation success measurement.",
            },
        )


if __name__ == "__main__":
    main()
