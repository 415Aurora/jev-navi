"""Static completion diagnostic; positive goal images never enter navigation runs."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from jev_navi.credentials import load_local_env
from jev_navi.experiments import make_policy
from jev_navi.providers import CallBudget, encode_images
from jev_navi.runner import write_json
from jev_navi.types import Observation, PolicyInput

ROOT = Path(__file__).resolve().parents[1]


def main():
    load_local_env(ROOT / ".env")
    source = ROOT / "runs/stop-audit-01"
    output = ROOT / "runs/stop-judgment-probe-01"
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((source / "audit.json").read_text())
    budget = CallBudget(8)
    policy = make_policy({"kind": "startlux"}, budget)
    records = []
    try:
        with (output / "responses.jsonl").open("w") as handle:
            for task in audit["records"]:
                for name in ("model-STOP", "annotation-goal"):
                    path = source / task["episode_id"] / (name + ".png")
                    view = PolicyInput(
                        task["instruction"],
                        (Observation(np.asarray(Image.open(path).convert("RGB")), 0),),
                        (),
                    )
                    decision = policy.decide(view)
                    record = {
                        "episode_id": task["episode_id"],
                        "image_case": name,
                        "probe": "four_actions",
                        "raw": decision.raw,
                        "action": decision.action.value,
                    }
                    records.append(record)
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    payload = {
                        "model": "startlux-decision-27b-autothink",
                        "state": "<image>\nLocal instruction: " + task["instruction"],
                        "images": encode_images(view),
                        "questions": {
                            "completed": {
                                "type": "noul",
                                "instructions": "Does this view show that the robot has already completed the instruction "
                                "and is in a suitable nearby floor position immediately in front of the "
                                "requested landmark? Merely seeing the landmark is not completion. Being "
                                "blocked by furniture is not completion; the approach may require turning "
                                "and going around it. Answer yes only if no further approach is needed.",
                            }
                        },
                        "think_mode": "never",
                        "allow_none": False,
                        "none_check": True,
                    }
                    raw = policy._request("POST", policy.endpoint, payload, inference=True)
                    value = raw["answers"]["completed"]["noul"]
                    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                        raise ValueError("Invalid completion probability")
                    record = {
                        "episode_id": task["episode_id"],
                        "image_case": name,
                        "probe": "completion",
                        "raw": raw,
                        "completed_probability": value,
                        "image_path": str(path),
                        "state": payload["state"],
                        "question": payload["questions"]["completed"],
                    }
                    records.append(record)
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    print(task["episode_id"], name, decision.action.value, value, flush=True)
    finally:
        policy.close()
        write_json(
            output / "summary.json",
            {
                "http_attempts": budget.used,
                "records": records,
                "note": "Static two-task diagnostic without history; not a navigation result.",
            },
        )


if __name__ == "__main__":
    main()
