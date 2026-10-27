"""Import official R2R-CE records without exposing evaluation data to policies.

Format: https://jacobkrantz.github.io/vlnce/data (R2R_VLNCE_v1-3).
Pass the *data/scene_datasets* directory as scenes_root. An official scene_id
such as mp3d/scan/scan.glb is appended intact; passing the mp3d directory would
duplicate that prefix. Absolute paths, traversal and symlink escapes are rejected.

annotation_verified means an annotation was imported from the declared source,
not that a human visually verified it. source.kind='fixture' marks synthetic
format examples, which must never be reported as model evaluation evidence.
TaskSpec groups task/initializer metadata, not a model-input whitelist; EvaluationSpec and environment_episode
are private environment/evaluator inputs. GT actions are deliberately discarded.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SPLITS = {"train", "val_seen", "val_unseen", "test"}
PROTOCOL = "r2rce_v13_dev"
PRIVATE_KEYS = {
    "evaluation",
    "goal_position",
    "goals",
    "success_radius",
    "reference_path",
    "gt_locations",
    "initial_distance",
    "actions",
    "info",
}


@dataclass(frozen=True)
class TaskSpec:
    id: str
    scene_id: str
    scene_family: str
    split: str
    task_type: str
    instruction: str
    scene: str
    start_position: list[float]
    start_rotation: list[float]
    trajectory_id: str | None
    repeat_id: int
    annotation_verified: bool


@dataclass(frozen=True)
class EvaluationSpec:
    goal_position: list[float] | None
    success_radius: float
    reference_path: list[list[float]] | None
    gt_locations: list[list[float]] | None
    initial_distance: float | None


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _vector(value: Any, size: int, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} coordinates")
    for number in value:
        _number(number, name)


def _path(value: Any, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty path or null")
    for position in value:
        _vector(position, 3, name)


def _scene_path(scene_id: Any, root: Path) -> Path:
    if not isinstance(scene_id, str) or not scene_id or "\\" in scene_id:
        raise ValueError("scene_id must be a relative POSIX scene path")
    relative = PurePosixPath(scene_id)
    if relative.is_absolute() or any(part in {"..", "."} for part in scene_id.split("/")):
        raise ValueError("scene_id must not be absolute or contain traversal")
    target = (root / scene_id).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("scene path escapes scenes_root")
    return target


def _read(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def task_spec(row: dict) -> TaskSpec:
    """Copy task/initializer metadata. Neither poses nor scene paths enter the policy."""
    try:
        return TaskSpec(**{key: copy.deepcopy(row[key]) for key in TaskSpec.__dataclass_fields__})
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid task fields: {error}") from error


def evaluation_spec(row: dict) -> EvaluationSpec:
    """Copy the evaluator-only specification, with no GT actions."""
    try:
        private = row["evaluation"]
        return EvaluationSpec(
            **{key: copy.deepcopy(private[key]) for key in EvaluationSpec.__dataclass_fields__}
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid evaluation fields: {error}") from error


def environment_episode(row: dict) -> dict:
    """Worker-compatible environment input; never use this as PolicyInput."""
    task = asdict(task_spec(row))
    private = evaluation_spec(row)
    if private.goal_position is None:
        raise ValueError("episode has no private goal; local evaluation is unavailable")
    task.update(goal_position=private.goal_position, success_radius=private.success_radius)
    return task


def prepare_r2rce(
    episodes_path,
    scenes_root,
    split,
    gt_path=None,
    *,
    source_kind="official",
) -> dict:
    """Read json[.gz], record raw-file hashes, and report missing licensed assets.

    No scene assets are downloaded. Missing dense GT and geodesic metadata stay
    null; sparse reference_path is never substituted for dense GT. Official test
    episodes without goals are retained with evaluable=False for inference.
    """
    if split not in SPLITS:
        raise ValueError(f"unsupported R2R-CE split: {split}")
    if source_kind not in {"official", "fixture"}:
        raise ValueError("source_kind must be official or fixture")
    episodes_path = Path(episodes_path).expanduser().resolve()
    root = Path(scenes_root).expanduser().resolve()
    data = _read(episodes_path)
    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list):
        raise ValueError("source must contain an episodes list")
    gt_file = Path(gt_path).expanduser().resolve() if gt_path is not None else None
    gt = _read(gt_file) if gt_file is not None else {}
    if not isinstance(gt, dict):
        raise ValueError("GT source must map episode IDs to locations")
    rows = []
    missing = set()
    for episode in data["episodes"]:
        if not isinstance(episode, dict):
            raise ValueError("source episode must be an object")
        try:
            scene_id = episode["scene_id"]
            scene = _scene_path(scene_id, root)
            if not scene.is_file():
                missing.add(str(scene))
            identifier = episode["episode_id"]
            if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
                raise ValueError("episode_id must be a string or integer")
            identifier = str(identifier)
            instruction = episode["instruction"]["instruction_text"]
            goals = episode.get("goals")
            if goals is not None and (not isinstance(goals, list) or len(goals) > 1):
                raise ValueError("R2R-CE supports exactly one goal, or no test goal")
            goal = goals[0]["position"] if goals else None
            info = episode.get("info") or {}
            gt_record = gt.get(identifier)
            if gt_record is not None and not isinstance(gt_record, dict):
                raise ValueError("GT episode must be an object")
            dense = gt_record.get("locations") if gt_record is not None else None
            trajectory = episode.get("trajectory_id")
            if trajectory is not None:
                if isinstance(trajectory, bool) or not isinstance(trajectory, (str, int)):
                    raise ValueError("trajectory_id must be a string or integer")
                trajectory = str(trajectory)
            task = TaskSpec(
                id=identifier,
                scene_id=scene_id,
                scene_family=scene_id,
                split=split,
                task_type="route",
                instruction=instruction,
                scene=str(scene),
                start_position=copy.deepcopy(episode["start_position"]),
                start_rotation=copy.deepcopy(episode["start_rotation"]),
                trajectory_id=trajectory,
                repeat_id=0,
                annotation_verified=True,
            )
            private = EvaluationSpec(
                goal_position=copy.deepcopy(goal),
                success_radius=3.0,
                reference_path=copy.deepcopy(episode.get("reference_path")),
                gt_locations=copy.deepcopy(dense),
                initial_distance=info.get("geodesic_distance"),
            )
            rows.append(
                {
                    **asdict(task),
                    "evaluation": asdict(private),
                    "evaluable": goal is not None,
                }
            )
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise ValueError(f"malformed R2R-CE episode: {error}") from error
    source = {
        "path": str(episodes_path),
        "sha256": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
        "kind": source_kind,
        "split": split,
        "scenes_root": str(root),
        "format_url": "https://jacobkrantz.github.io/vlnce/data",
        "annotation_verification": "source import only; no human visual verification",
        "gt_path": str(gt_file) if gt_file else None,
        "gt_sha256": hashlib.sha256(gt_file.read_bytes()).hexdigest() if gt_file else None,
    }
    suite = {
        "schema_version": 2,
        "task_family": "r2rce",
        "protocol_id": PROTOCOL,
        "source": source,
        "missing_scenes": sorted(missing),
        "episodes": rows,
    }
    validate_r2rce_suite(suite)
    return suite


def validate_r2rce_suite(suite: dict) -> None:
    """Validate structure and private/public separation, without requiring assets."""
    if not isinstance(suite, dict):
        raise ValueError("suite must be an object")
    if (
        suite.get("schema_version") != 2
        or suite.get("task_family") != "r2rce"
        or suite.get("protocol_id") != PROTOCOL
    ):
        raise ValueError("expected schema v2 R2R-CE development protocol")
    source = suite.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {"official", "fixture"}:
        raise ValueError("source must declare official or fixture provenance")
    sha = source.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("source sha256 must be a lowercase SHA256 digest")
    if source.get("split") not in SPLITS:
        raise ValueError("invalid source split")
    root = source.get("scenes_root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("source scenes_root must be absolute")
    root = Path(root).resolve()
    missing = suite.get("missing_scenes")
    if not isinstance(missing, list) or any(
        not isinstance(path, str) or not Path(path).is_absolute() for path in missing
    ):
        raise ValueError("missing_scenes must contain absolute scene paths")
    rows = suite.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("episodes must be a list")
    seen = set()
    definitions = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("episode must be an object")
        if (PRIVATE_KEYS - {"evaluation"}) & row.keys():
            raise ValueError("private goals, paths and GT must be inside evaluation")
        task = task_spec(row)
        private = evaluation_spec(row)
        if not isinstance(task.id, str) or not task.id:
            raise ValueError("episode id must be a nonempty string")
        if (
            isinstance(task.repeat_id, bool)
            or not isinstance(task.repeat_id, int)
            or task.repeat_id < 0
        ):
            raise ValueError("repeat_id must be a nonnegative integer")
        identity = (task.id, task.repeat_id)
        if identity in seen:
            raise ValueError("episode id and repeat_id must be unique")
        seen.add(identity)
        if task.split not in SPLITS or task.split != source["split"]:
            raise ValueError("episode split must match source split")
        if task.task_type != "route" or task.annotation_verified is not True:
            raise ValueError("expected a verified imported route annotation")
        if not isinstance(task.instruction, str) or not task.instruction.strip():
            raise ValueError("instruction must be a nonempty original string")
        if task.trajectory_id is not None and not isinstance(task.trajectory_id, str):
            raise ValueError("trajectory_id must be a string or null")
        scene = _scene_path(task.scene_id, root)
        if task.scene_family != task.scene_id:
            raise ValueError("scene_family must equal scene_id")
        if not isinstance(task.scene, str) or not Path(task.scene).is_absolute():
            raise ValueError("scene must be absolute")
        if str(scene) != task.scene:
            raise ValueError("scene does not match scenes_root and scene_id")
        _vector(task.start_position, 3, "start_position")
        _vector(task.start_rotation, 4, "start_rotation")
        if not math.isclose(sum(x * x for x in task.start_rotation), 1, abs_tol=1e-3):
            raise ValueError("start_rotation must be a unit quaternion in xyzw order")
        if set(row["evaluation"]) != set(EvaluationSpec.__dataclass_fields__):
            raise ValueError("evaluation must contain only declared private fields")
        if private.goal_position is not None:
            _vector(private.goal_position, 3, "goal_position")
        elif task.split != "test":
            raise ValueError("non-test episode needs a private goal_position")
        if row.get("evaluable") is not (private.goal_position is not None):
            raise ValueError("evaluable must explicitly reflect private goal availability")
        if _number(private.success_radius, "success_radius") != 3.0:
            raise ValueError("R2R-CE v1-3 success_radius must be 3.0")
        _path(private.reference_path, "reference_path")
        _path(private.gt_locations, "gt_locations")
        if private.initial_distance is not None:
            _number(private.initial_distance, "initial_distance", nonnegative=True)

        definition = {key: value for key, value in row.items() if key != "repeat_id"}
        if task.id in definitions and definitions[task.id] != definition:
            raise ValueError(f"Conflicting task definition for repeated episode {task.id}")
        definitions[task.id] = definition
