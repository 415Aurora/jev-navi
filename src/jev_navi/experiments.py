"""Validated manifests and deterministic paired experiments."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .policies import ReplayPolicy, ScriptedPolicy
from .providers import PROMPT_VERSION, CallBudget, D3Policy, StartLuxPolicy
from .recovery import LoopMonitor
from .runner import run_episode, write_json
from .safety import DepthGuard
from .types import Camera

CONDITIONS = {"full", "no_image_history", "shuffled_instruction", "no_guard"}
TASK_TYPES = {"visible", "turn", "disambiguate"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def validate_suite(suite, diagnostic=False):
    if suite.get("schema_version") != 1 or not suite.get("episodes"):
        raise ValueError("Expected schema_version=1 and nonempty episodes")
    seen = set()
    families = defaultdict(set)
    scenes = defaultdict(list)
    for row in suite["episodes"]:
        required = {
            "id",
            "scene_id",
            "scene_family",
            "split",
            "task_type",
            "instruction",
            "scene",
            "start_position",
            "start_rotation",
            "goal_position",
            "success_radius",
            "annotation_verified",
        }
        if not required <= row.keys():
            raise ValueError(f"Missing episode fields: {sorted(required - row.keys())}")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError("Episode IDs must be unique nonempty strings")
        seen.add(row["id"])
        if row["split"] not in ("dev", "holdout") or row["task_type"] not in TASK_TYPES:
            raise ValueError("Invalid split or task_type")
        if not isinstance(row["instruction"], str) or not row["instruction"].strip():
            raise ValueError("Instruction must be nonempty")
        for field, length in [("start_position", 3), ("goal_position", 3), ("start_rotation", 4)]:
            value = np.asarray(row[field], dtype=float)
            if value.shape != (length,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {field}")
        if not np.isclose(np.linalg.norm(row["start_rotation"]), 1, atol=1e-3):
            raise ValueError("start_rotation must be normalized [x,y,z,w] quaternion")
        radius = row["success_radius"]
        if not isinstance(radius, (float, int)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("success_radius must be positive and finite")
        if diagnostic and row["annotation_verified"] is not True:
            raise ValueError("Diagnostic suite requires verified semantic annotations")
        families[row["scene_family"]].add(row["split"])
        scenes[row["scene_id"]].append(row)
    if any(len(splits) > 1 for splits in families.values()):
        raise ValueError("Scene family overlaps dev and holdout splits")
    if diagnostic:
        if len(suite["episodes"]) != 36 or len(scenes) != 3:
            raise ValueError("Diagnostic suite requires 3 scenes and 36 episodes")
        scene_splits = []
        for rows in scenes.values():
            if len(rows) != 12 or Counter(x["task_type"] for x in rows) != dict.fromkeys(
                TASK_TYPES, 4
            ):
                raise ValueError("Each scene requires 12 episodes, 4 per task type")
            if len({x["split"] for x in rows}) != 1:
                raise ValueError("Scene overlaps splits")
            scene_splits.append(rows[0]["split"])
        if Counter(scene_splits) != {"dev": 1, "holdout": 2}:
            raise ValueError("Expected one development scene and two holdout scenes")


def instruction_permutation(episodes, seed=17):
    """Derange instructions within each scene/split, never borrowing holdout text for dev."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in episodes:
        groups[(row["scene_id"], row["split"])].append(row)
    mapping = {}
    for rows in groups.values():
        rows = sorted(rows, key=lambda x: x["id"])
        if len(rows) < 2:
            raise ValueError(
                "Shuffled condition needs at least two distinct instructions per scene"
            )
        found = False
        for _ in range(1000):
            other = rows.copy()
            rng.shuffle(other)
            if all(
                a["id"] != b["id"]
                and a["instruction"] != b["instruction"]
                and (
                    not a.get("target_region_id")
                    or a["target_region_id"] != b.get("target_region_id")
                )
                for a, b in zip(rows, other)
            ):
                mapping.update({a["id"]: b["id"] for a, b in zip(rows, other)})
                found = True
                break
        if not found:
            raise ValueError("Cannot derange duplicated instructions within scene")
    return mapping


def make_policy(settings, budget):
    kind = settings.get("kind", "scripted")
    if kind == "scripted":
        return ScriptedPolicy(settings["actions"])
    if kind == "replay":
        return ReplayPolicy(settings["path"])
    if kind == "startlux":
        endpoint = settings.get("endpoint") or os.environ.get("STARTLUX_ENDPOINT")
        if not endpoint:
            raise ValueError("Set STARTLUX_ENDPOINT to the documented deployment URL")
        return StartLuxPolicy(
            os.environ.get("BAMBOO_API_KEY", ""), endpoint=endpoint, budget=budget
        )
    if kind == "d3":
        return D3Policy(
            os.environ.get("INFERENCE_API_KEY", ""),
            budget=budget,
            base_url=settings.get("base_url", "https://api.inference.sh"),
        )
    raise ValueError(f"Unsupported policy: {kind}")


def make_environment(settings, episode, camera, seed):
    common = {k: settings[k] for k in ("forward_m", "turn_deg") if k in settings}
    if settings.get("backend", "habitat") == "synthetic":
        from .environments.synthetic import SyntheticEnvironment

        return SyntheticEnvironment(
            camera,
            obstacle_distance=settings.get("obstacle_distance", 5),
            target=episode["goal_position"],
            success_radius=episode["success_radius"],
            **common,
        )
    if settings.get("backend", "habitat") != "habitat":
        raise ValueError("Unknown environment backend")
    from .environments.habitat import HabitatEnvironment

    return HabitatEnvironment(
        episode,
        camera,
        radius_m=settings.get("radius_m", 0.2),
        agent_height_m=settings.get("agent_height_m", 1.5),
        allow_sliding=settings.get("allow_sliding", False),
        depth_pitch_deg=settings.get("depth_pitch_deg", 0.0),
        seed=seed,
        python=settings.get("python"),
        **common,
    )


def dependencies():
    versions = {"python": platform.python_version()}
    for name in ("jev-navi", "numpy", "Pillow", "httpx", "habitat-sim", "habitat-lab"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def implementation_digest():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})


def run_suite(config, suite, output):
    validate_suite(suite, diagnostic=config.get("diagnostic", False))
    LoopMonitor(**config.get("recovery", {}))
    settings = config.get("run", {})
    seed = settings.get("seed", 17)
    conditions = config.get("conditions", ["full"])
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - CONDITIONS:
        raise ValueError("Invalid or duplicate conditions")
    policy_settings = config["policy"]
    live = policy_settings.get("kind") in ("startlux", "d3")
    if live and config.get("free_quota_confirmed") is not True:
        raise ValueError("Set free_quota_confirmed=true only after checking the provider quota")
    if live and any(x["annotation_verified"] is not True for x in suite["episodes"]):
        raise ValueError("Live navigation requires verified episode annotations")
    limit = settings.get("call_limit", 500)
    cap = 7000 if config.get("diagnostic") else 500
    if type(limit) is not int or not 1 <= limit <= cap:
        raise ValueError(f"call_limit must be 1..{cap}")
    camera = Camera(**config.get("camera", {}))
    environment = config.get("environment", {})
    if environment.get("forward_m", 0.25) != 0.25 or environment.get("turn_deg", 15) != 15:
        raise ValueError("This protocol uses fixed 0.25m forward and 15 degree turn actions")
    budget = CallBudget(limit)
    shuffled = (
        instruction_permutation(suite["episodes"], seed)
        if "shuffled_instruction" in conditions
        else {}
    )
    index = {x["id"]: x for x in suite["episodes"]}
    # Fail early on missing credentials/configuration before creating run artifacts.
    policy = make_policy(policy_settings, budget)
    if hasattr(policy, "close"):
        policy.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "config": config,
        "suite": suite,
        "suite_sha256": digest(suite),
        "config_sha256": digest(config),
        "instruction_permutation": shuffled,
        "prompt_version": PROMPT_VERSION,
        "dependencies": dependencies(),
        "implementation_sha256": implementation_digest(),
        "protocol_sha256": digest(
            {
                "environment": environment,
                "camera": asdict(camera),
                "safety": config.get("safety", {}),
                "recovery": config.get("recovery", {"mode": "off"}),
                "max_decisions": settings.get("max_decisions", 60),
                "seed": seed,
                "call_limit": limit,
                "prompt_version": PROMPT_VERSION,
                "implementation_sha256": implementation_digest(),
            }
        ),
        "budget": {"limit": limit, "used": 0},
        "status": "running",
    }
    native_manifest = Path(__file__).resolve().parents[2] / "environment/source-manifest.json"
    if environment.get("backend", "habitat") == "habitat" and native_manifest.exists():
        manifest["native_environment"] = json.loads(native_manifest.read_text())
    write_json(output / "suite_manifest.json", manifest)
    jobs = [
        (episode, condition)
        for episode in suite["episodes"]
        for condition in conditions
        if condition != "no_guard" or episode["split"] == "dev"
    ]
    random.Random(seed).shuffle(jobs)
    results = []
    try:
        for job_id, (episode, condition) in enumerate(jobs):
            if live and budget.used >= budget.limit:
                manifest["status"] = "partial_call_budget"
                break
            policy = make_policy(policy_settings, budget)
            env = None
            try:
                env = make_environment(environment, episode, camera, seed)
                guard = DepthGuard(
                    camera,
                    radius_m=environment.get("radius_m", 0.2),
                    agent_height_m=environment.get("agent_height_m", 1.5),
                    forward_m=environment.get("forward_m", 0.25),
                    margin_m=config.get("safety", {}).get("margin_m", 0.1),
                    enabled=condition != "no_guard",
                    depth_pitch_deg=environment.get("depth_pitch_deg", 0.0),
                )
                instruction_source = (
                    shuffled[episode["id"]]
                    if condition == "shuffled_instruction"
                    else episode["id"]
                )
                result = run_episode(
                    env,
                    policy,
                    guard,
                    index[instruction_source]["instruction"],
                    output / f"{job_id:04d}",
                    max_decisions=settings.get("max_decisions", 60),
                    recovery=config.get("recovery", {}),
                    image_history=0 if condition == "no_image_history" else 2,
                    metadata={
                        "episode_id": episode["id"],
                        "scene_id": episode["scene_id"],
                        "scene_family": episode["scene_family"],
                        "split": episode["split"],
                        "task_type": episode["task_type"],
                        "condition": condition,
                        "policy_kind": policy_settings["kind"],
                        "instruction_source": instruction_source,
                        "annotation_verified": episode["annotation_verified"],
                        "seed": seed,
                        "suite_sha256": digest(suite),
                        "config_sha256": digest(config),
                        "prompt_version": PROMPT_VERSION,
                        "implementation_sha256": manifest["implementation_sha256"],
                        "protocol_sha256": manifest["protocol_sha256"],
                    },
                )
                results.append(result)
                if result["error_code"] in (
                    "authentication",
                    "quota",
                    "unsupported_input",
                    "call_budget",
                ):
                    manifest["status"] = "partial_" + result["error_code"]
                    break
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    if hasattr(policy, "close"):
                        policy.close()
        else:
            manifest["status"] = "completed"
    finally:
        manifest["budget"]["used"] = budget.used
        manifest["completed_jobs"] = len(results)
        manifest["planned_jobs"] = len(jobs)
        if manifest["status"] == "running":
            manifest["status"] = "interrupted_or_error"
        write_json(output / "suite_manifest.json", manifest)
        from .reporting import write_report

        write_report(output)
    return results
