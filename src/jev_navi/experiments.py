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
from .runner import EpisodeInitializationError, run_episode, write_json
from .safety import DepthGuard
from .types import Camera

CONDITIONS = {"full", "no_image_history", "shuffled_instruction", "no_guard"}
TASK_TYPES = {"visible", "turn", "disambiguate"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def validate_suite(suite, diagnostic=False):
    if suite.get("schema_version") == 2:
        from .datasets import validate_r2rce_suite

        if diagnostic:
            raise ValueError("Legacy 36-task diagnostic is not an R2R-CE protocol")
        validate_r2rce_suite(suite)
        if not suite["episodes"]:
            raise ValueError("Suite must have episodes")
        return
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
    if kind == "d3_fixture":
        from .d3_fixture import make_d3_fixture_policy

        return make_d3_fixture_policy(settings, budget)
    if kind in ("startlux_fixture", "startlux_hierarchical_fixture"):
        from .hierarchy import make_fixture_policy

        return make_fixture_policy(settings, budget)
    if kind in ("startlux", "startlux_hierarchical"):
        endpoint = settings.get("endpoint") or os.environ.get("STARTLUX_ENDPOINT")
        if not endpoint:
            raise ValueError("Set STARTLUX_ENDPOINT to the documented deployment URL")
        cls = StartLuxPolicy
        if kind == "startlux_hierarchical":
            from .hierarchy import HierarchicalPolicy

            cls = HierarchicalPolicy
        return cls(
            os.environ.get("BAMBOO_API_KEY", ""),
            endpoint=endpoint,
            budget=budget,
            think_mode=settings.get(
                "think_mode", "auto" if kind == "startlux_hierarchical" else "never"
            ),
        )
    if kind == "d3":
        provider = settings.get("provider", "inference_sh")
        if provider == "official":
            from .official_d3 import OfficialD3Policy

            if "base_url" in settings:
                raise ValueError("Official provider uses a fixed endpoint; omit base_url")
            return OfficialD3Policy(
                os.environ.get("DECISION_API_KEY", ""),
                budget=budget,
                prompt_version=settings.get("prompt_version", "r2rce-direct-v1"),
            )
        if provider != "inference_sh":
            raise ValueError("Unsupported d3 provider")
        return D3Policy(
            os.environ.get("INFERENCE_API_KEY", ""),
            budget=budget,
            base_url=settings.get("base_url", "https://api.inference.sh"),
            prompt_version=settings.get("prompt_version", PROMPT_VERSION),
        )
    raise ValueError(f"Unsupported policy: {kind}")


def make_environment(settings, episode, camera, seed, depth_camera=None):
    common = {k: settings[k] for k in ("forward_m", "turn_deg") if k in settings}
    native = {
        k: settings[k]
        for k in (
            "radius_m",
            "agent_height_m",
            "allow_sliding",
            "depth_pitch_deg",
            "max_episode_steps",
            "success_strict",
        )
        if k in settings
    }
    if settings.get("backend", "habitat") == "synthetic":
        from .environments.synthetic import SyntheticEnvironment

        extra = {}
        if settings.get("success_strict"):
            extra = {
                "start_position": episode["start_position"],
                "start_rotation": episode["start_rotation"],
            }
        return SyntheticEnvironment(
            camera,
            obstacle_distance=settings.get("obstacle_distance", 5),
            target=episode["goal_position"],
            success_radius=episode["success_radius"],
            depth_camera=depth_camera,
            max_episode_steps=settings.get("max_episode_steps"),
            success_strict=settings.get("success_strict", False),
            **extra,
            **common,
        )
    if settings.get("backend", "habitat") != "habitat":
        raise ValueError("Unknown environment backend")
    from .environments.habitat import HabitatEnvironment

    return HabitatEnvironment(
        episode,
        camera,
        seed=seed,
        python=settings.get("python"),
        depth_camera=asdict(depth_camera) if depth_camera else None,
        **native,
        **common,
    )


def make_run_budget(config, limit):
    """Campaign permissions are independent of run directories and local attempt caps."""
    from .budget import PhaseBudget, RunBudget, initialize_ledger

    execution = config.get("execution", {})
    path = Path(execution.get("ledger", "runs/request-ledger.sqlite3")).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    initialize_ledger(path)
    phase_id = execution.get("phase_id", "r2rce-offline-v1")
    enabled = execution.get("mode", "offline") == "live"
    phase_limit = execution.get("phase_limit", 0)
    initial = execution.get("initial_used", 0)
    if phase_id == "first-round":
        phase_limit, initial = 500, 483
    phase = PhaseBudget(
        path, phase_id, limit=phase_limit, initial_used=initial, live_enabled=enabled
    )
    return RunBudget(phase, limit)


def source_git():
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def dependencies():
    versions = {"python": platform.python_version()}
    for name in ("jev-navi", "numpy", "Pillow", "httpx", "fastdtw", "habitat-sim", "habitat-lab"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def implementation_digest():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*.py"))})


def run_suite(config, suite, output):
    from .profiles import resolve_config

    config = resolve_config(config)
    validate_suite(suite, diagnostic=config.get("diagnostic", False))
    modern = suite.get("schema_version") == 2
    if modern and "selection" in suite:
        from .task_selection import verify_selection

        verify_selection(suite, suite["selection"])
    if modern and config["protocol_id"] == "local_v1":
        raise ValueError("R2R-CE suite requires an explicitly selected R2R-CE protocol")
    if not modern and config["protocol_id"] != "local_v1":
        raise ValueError("R2R-CE protocol requires a schema v2 suite")
    LoopMonitor(**config.get("recovery", {}))
    settings = config["run"]
    seed = settings["seed"]
    conditions = config.get("conditions", ["full"])
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - CONDITIONS:
        raise ValueError("Invalid or duplicate conditions")
    policy_settings = config["policy"]
    kind = policy_settings.get("kind", "scripted")
    if kind in {"d3", "d3_fixture"} and modern:
        from .action_context import ROUTE_PROMPT_VERSION

        policy_settings.setdefault("prompt_version", ROUTE_PROMPT_VERSION)
        if (
            config["protocol_id"] != "r2rce_v13_dev"
            or policy_settings["prompt_version"] != ROUTE_PROMPT_VERSION
            or config.get("recovery", {}).get("mode", "off") != "off"
        ):
            raise ValueError(
                "D3 direct baseline requires the fixed protocol, route prompt and recovery off"
            )
        allowed = {"full", "no_guard"} if kind == "d3_fixture" else {"full"}
        if set(conditions) - allowed:
            raise ValueError("D3 direct baseline does not enable input or memory ablations")
        if kind == "d3_fixture" and config["execution"].get("mode") != "offline":
            raise ValueError("D3 fixture execution must remain offline")
    live = kind in ("startlux", "startlux_hierarchical", "d3")
    if live and config.get("free_quota_confirmed") is not True:
        raise ValueError("Set free_quota_confirmed=true only after checking the provider quota")
    if live and config["execution"].get("mode", "offline") != "live":
        raise ValueError("Offline execution forbids a real model policy")
    if live and modern and suite["source"]["kind"] == "fixture":
        raise ValueError("Fixture tasks cannot be used as a live R2R-CE benchmark")
    if live and any(x["annotation_verified"] is not True for x in suite["episodes"]):
        raise ValueError("Live navigation requires verified episode annotations")
    limit = settings["call_limit"]
    cap = 7000 if config.get("diagnostic") or modern else 500
    if type(limit) is not int or not 1 <= limit <= cap:
        raise ValueError(f"call_limit must be 1..{cap}")
    camera = Camera(**config["camera"])
    depth_camera = Camera(**config.get("depth_camera", config["camera"]))
    environment = config["environment"]
    if environment.get("forward_m", 0.25) != 0.25 or environment.get("turn_deg", 15) != 15:
        raise ValueError("This protocol uses fixed 0.25m forward and 15 degree turn actions")
    native = environment.get("backend", "habitat") == "habitat"
    if modern:
        from .datasets import environment_episode

        if any(not row["evaluable"] for row in suite["episodes"]):
            raise ValueError("Local evaluation unavailable for episodes without private goals")
        missing = sorted(
            {
                row["scene"]
                for row in suite["episodes"]
                if native and not Path(row["scene"]).is_file()
            }
        )
        if missing:
            raise ValueError("Unrunnable: missing scene assets: " + ", ".join(missing))
    campaign = make_run_budget(config, limit)
    budget = campaign if live else CallBudget(limit)
    shuffled = (
        instruction_permutation(suite["episodes"], seed)
        if "shuffled_instruction" in conditions
        else {}
    )
    index = {x["id"]: x for x in suite["episodes"]}
    policy = make_policy(policy_settings, budget)
    if hasattr(policy, "close"):
        policy.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    version = getattr(policy, "prompt_version", PROMPT_VERSION)
    physics = {
        "protocol_id": config["protocol_id"],
        "environment": environment,
        "camera": asdict(camera),
        "depth_camera": asdict(depth_camera),
        "safety": config.get("safety", {}),
        "recovery": config.get("recovery", {"mode": "off"}),
        "max_decisions": settings["max_decisions"],
        "max_steps": settings.get("max_steps"),
        "seed": seed,
        "implementation_sha256": implementation_digest(),
    }
    if not modern:
        physics.update(prompt_version=version, call_limit=limit)
    jobs = [
        (episode, condition)
        for episode in suite["episodes"]
        for condition in conditions
        if condition != "no_guard" or episode["split"] in ("dev", "train")
    ]
    manifest = {
        "config": config,
        "suite": suite,
        "suite_sha256": digest(suite),
        "config_sha256": digest(config),
        "instruction_permutation": shuffled,
        "prompt_version": version,
        "dependencies": dependencies(),
        "git": source_git(),
        "implementation_sha256": implementation_digest(),
        "protocol_sha256": digest(physics),
        "protocol": physics,
        "budget": {"limit": limit, "used": 0},
        "phase_budget": campaign.snapshot(),
        "status": "running",
        "measurement_kind": "model" if live else "fixture",
        "planned_jobs": len(jobs),
        "completed_jobs": 0,
        "unrunnable_jobs": [],
    }
    native_manifest = Path(__file__).resolve().parents[2] / "environment/source-manifest.json"
    if native and native_manifest.exists():
        manifest["native_environment"] = json.loads(native_manifest.read_text())
    write_json(output / "suite_manifest.json", manifest)
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
                env_episode = environment_episode(episode) if modern else episode
                try:
                    env = make_environment(environment, env_episode, camera, seed, depth_camera)
                except (ValueError, RuntimeError) as error:
                    manifest["unrunnable_jobs"].append(
                        {
                            "episode_id": episode["id"],
                            "repeat_id": episode.get("repeat_id", 0),
                            "condition": condition,
                            "reason": str(error),
                            "started": False,
                        }
                    )
                    continue
                guard = DepthGuard(
                    depth_camera,
                    radius_m=environment["radius_m"],
                    agent_height_m=environment["agent_height_m"],
                    forward_m=environment["forward_m"],
                    margin_m=config.get("safety", {}).get("margin_m", 0.1),
                    enabled=condition != "no_guard",
                    depth_pitch_deg=environment["depth_pitch_deg"],
                )
                instruction_source = (
                    shuffled[episode["id"]]
                    if condition == "shuffled_instruction"
                    else episode["id"]
                )
                metadata = {
                    "episode_id": episode["id"],
                    "repeat_id": episode.get("repeat_id", 0),
                    "trajectory_id": episode.get("trajectory_id"),
                    "scene_id": episode["scene_id"],
                    "scene_family": episode["scene_family"],
                    "split": episode["split"],
                    "task_type": episode["task_type"],
                    "condition": condition,
                    "policy_kind": kind,
                    "think_mode": (
                        None
                        if modern and kind in {"d3", "d3_fixture"}
                        else policy_settings.get(
                            "think_mode", "auto" if "hierarchical" in kind else "never"
                        )
                    ),
                    "reasoning_mode": "direct_choice" if kind in {"d3", "d3_fixture"} else None,
                    "measurement_kind": "model" if live else "fixture",
                    "protocol_id": config["protocol_id"],
                    "task_family": suite.get("task_family", "local"),
                    "instruction_source": instruction_source,
                    "annotation_verified": episode["annotation_verified"],
                    "seed": seed,
                    "suite_sha256": manifest["suite_sha256"],
                    "config_sha256": manifest["config_sha256"],
                    "prompt_version": version,
                    "implementation_sha256": manifest["implementation_sha256"],
                    "protocol_sha256": manifest["protocol_sha256"],
                    "request_budget": {
                        "run_limit": limit,
                        "phase_limit": campaign.phase.limit,
                    },
                }
                result = run_episode(
                    env,
                    policy,
                    guard,
                    index[instruction_source]["instruction"],
                    output / f"{job_id:04d}",
                    max_decisions=settings["max_decisions"],
                    recovery=config.get("recovery", {}),
                    image_history=0 if condition == "no_image_history" else 2,
                    metadata=metadata,
                    protocol_id=config["protocol_id"],
                    max_steps=settings.get("max_steps"),
                    evaluation_spec=episode.get("evaluation"),
                    ndtw_algorithm=config.get("evaluation", {}).get("ndtw_algorithm", "fastdtw"),
                )
                results.append(result)
                if result["error_code"] in (
                    "authentication",
                    "quota",
                    "unsupported_input",
                    "call_budget",
                    "offline_mode",
                    "model_mismatch",
                ):
                    manifest["status"] = "partial_" + result["error_code"]
                    break
            except EpisodeInitializationError as error:
                manifest["unrunnable_jobs"].append(
                    {
                        "episode_id": episode["id"],
                        "repeat_id": episode.get("repeat_id", 0),
                        "condition": condition,
                        "reason": str(error),
                        "started": False,
                    }
                )
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    if hasattr(policy, "close"):
                        policy.close()
        else:
            manifest["status"] = (
                "partial_unrunnable" if manifest["unrunnable_jobs"] else "completed"
            )
    finally:
        manifest["budget"]["used"] = budget.used if live else 0
        manifest["fixture_requests"] = budget.used if not live else 0
        manifest["phase_budget"] = campaign.snapshot()
        manifest["completed_jobs"] = len(results)
        if manifest["status"] == "running":
            manifest["status"] = "interrupted_or_error"
        write_json(output / "suite_manifest.json", manifest)
        from .reporting import write_report

        write_report(output)
    return results
