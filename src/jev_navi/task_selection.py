"""Deterministic train-only task freeze. No simulator, model, or budget access."""

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .datasets import prepare_r2rce, validate_r2rce_suite

VERSION = "train-scene-trajectory-sha256-v1"


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _definition(row):
    # Absolute asset paths can move; all semantic/init/private fields stay bound.
    return {key: value for key, value in row.items() if key != "scene"}


def _raw_suite(suite):
    validate_r2rce_suite(suite)
    source = suite["source"]
    for path_key, hash_key in (("path", "sha256"), ("gt_path", "gt_sha256")):
        path, expected = source.get(path_key), source.get(hash_key)
        if path is None and path_key == "gt_path" and expected is None:
            continue
        if not path or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("source integrity mismatch")
    return prepare_r2rce(
        source["path"],
        source["scenes_root"],
        source["split"],
        source.get("gt_path"),
        source_kind=source["kind"],
    )


def _select(raw, scenes, per_scene, seed):
    if raw["source"]["split"] != "train":
        raise ValueError("Only train development tasks may be selected")
    if any(type(x) is not int or x < 1 for x in (scenes, per_scene)):
        raise ValueError("scenes and per_scene must be positive integers")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a nonempty string")
    groups = defaultdict(lambda: defaultdict(list))
    for row in raw["episodes"]:
        if not row["trajectory_id"]:
            raise ValueError("Every candidate needs a trajectory identity")
        if row["repeat_id"] != 0:
            raise ValueError("Freeze source must contain original tasks without repeats")
        groups[row["scene_id"]][row["trajectory_id"]].append(row)
    eligible = [scene for scene, routes in groups.items() if len(routes) >= per_scene]
    if len(eligible) < scenes:
        raise ValueError("Insufficient eligible scenes/distinct trajectories")

    def rank(kind, *parts):
        return _digest([VERSION, seed, kind, *parts]), parts

    rows = []
    for scene in sorted(eligible, key=lambda s: rank("scene", s))[:scenes]:
        for trajectory in sorted(groups[scene], key=lambda t: rank("trajectory", scene, t))[
            :per_scene
        ]:
            candidates = groups[scene][trajectory]
            rows.append(min(candidates, key=lambda r: rank("episode", scene, trajectory, r["id"])))
    source = raw["source"]
    manifest = {
        "schema_version": 1,
        "selection_version": VERSION,
        "seed": seed,
        "scenes": scenes,
        "per_scene": per_scene,
        "purpose": "train_development",
        "protocol_id": raw["protocol_id"],
        "source_identity": {k: source[k] for k in ("kind", "split", "sha256", "gt_sha256")},
        "tasks": [
            {
                "id": r["id"],
                "repeat_id": r["repeat_id"],
                "scene_id": r["scene_id"],
                "trajectory_id": r["trajectory_id"],
                "definition_sha256": _digest(_definition(r)),
            }
            for r in rows
        ],
    }
    selected = copy.deepcopy(raw)
    selected["episodes"] = copy.deepcopy(rows)
    selected["missing_scenes"] = sorted(
        {r["scene"] for r in rows if not Path(r["scene"]).is_file()}
    )
    selected["selection"] = copy.deepcopy(manifest)
    return selected, manifest


def freeze_tasks(suite, *, scenes=4, per_scene=6, seed="jev-train-dev-v1"):
    """Freeze a full imported train source, never filtering by asset availability."""
    raw = _raw_suite(suite)
    provided = sorted(
        (_definition(r) for r in suite["episodes"]), key=lambda r: (r["id"], r["repeat_id"])
    )
    original = sorted(
        (_definition(r) for r in raw["episodes"]), key=lambda r: (r["id"], r["repeat_id"])
    )
    if provided != original:
        raise ValueError("Prepared source task definitions differ from original source")
    return _select(raw, scenes, per_scene, seed)


def verify_selection(suite, manifest):
    """Rebuild selection from raw annotations; hashes are reproducibility, not signatures."""
    if not isinstance(manifest, dict) or manifest.get("selection_version") != VERSION:
        raise ValueError("Unsupported selection manifest")
    raw = _raw_suite(suite)
    expected, identity = _select(
        raw, manifest.get("scenes"), manifest.get("per_scene"), manifest.get("seed")
    )
    if identity != manifest or suite.get("selection", manifest) != manifest:
        raise ValueError("Frozen selection manifest mismatch")
    if [_definition(r) for r in suite["episodes"]] != [
        _definition(r) for r in expected["episodes"]
    ]:
        raise ValueError("Frozen selection task definition or membership mismatch")
    return {
        "verified": True,
        "manifest_sha256": _digest(manifest),
        "tasks": len(expected["episodes"]),
        "source_reimported": True,
    }
