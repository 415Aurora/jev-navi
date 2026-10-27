"""Static visual-input checks; a successful HTTP response is not proof of grounding."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .experiments import dependencies, make_policy, make_run_budget
from .providers import PROMPT_VERSION
from .runner import write_json
from .types import Observation, PolicyInput, ServiceError


def run_contract(config, image_paths, instruction, output):
    if config.get("free_quota_confirmed") is not True:
        raise ValueError("Confirm free quota before a live contract check")
    if not 1 <= len(image_paths) <= 3:
        raise ValueError("Provide one to three time-ordered images, current image last")
    images = [np.asarray(Image.open(path).convert("RGB")) for path in image_paths]
    frames = tuple(Observation(image, i) for i, image in enumerate(images))
    blank = (Observation(np.zeros_like(images[-1]), len(images) - 1),)
    cases = [
        ("current_only", (frames[-1],)),
        ("chronological", frames),
        ("reversed", tuple(reversed(frames))),
        ("blank_current", blank),
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    budget = make_run_budget(config, 24)
    policy = make_policy(config["policy"], budget)
    records = []
    try:
        with (output / "responses.jsonl").open("w") as handle:
            for name, selected in cases:
                record = {
                    "case": name,
                    "frame_steps": [f.step for f in selected],
                    "rgb_sha256": [hashlib.sha256(f.rgb.tobytes()).hexdigest() for f in selected],
                }
                try:
                    decision = policy.decide(PolicyInput(instruction, selected, ()))
                    record.update(
                        action=decision.action.value,
                        probabilities=decision.probabilities,
                        raw=decision.raw,
                        model=decision.model,
                        latency_s=decision.latency_s,
                    )
                except ServiceError as error:
                    record["error_code"] = error.code
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                if "error_code" in record:
                    break
    finally:
        if hasattr(policy, "close"):
            policy.close()
    result = {
        "cases": len(records),
        "http_attempts": budget.used,
        "image_grounding_verified": False,
        "note": "Compare labelled navigation decisions across images and ablations before "
        "claiming grounding. Unchanged actions alone neither prove nor refute image use.",
        "prompt_version": PROMPT_VERSION,
        "dependencies": dependencies(),
        "instruction": instruction,
        "images": [str(Path(p).resolve()) for p in image_paths],
        "errors": [r["error_code"] for r in records if "error_code" in r],
    }
    write_json(output / "contract.json", result)
    return result
