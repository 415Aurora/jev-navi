"""R2R-CE shaped fixture tests, never model evaluation evidence."""

import copy
import gzip
import hashlib
import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from jev_navi.datasets import (
    environment_episode,
    evaluation_spec,
    prepare_r2rce,
    task_spec,
    validate_r2rce_suite,
)

INSTRUCTION = "  Walk through the hall.\nTurn left, then STOP!  "


def episode():
    return {
        "episode_id": 7,
        "trajectory_id": 42,
        "scene_id": "mp3d/example/example.glb",
        "instruction": {"instruction_text": INSTRUCTION, "instruction_tokens": [2, 3, 0]},
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [4.0, 0.0, 0.0], "radius": 3.0}],
        "reference_path": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        "info": {"geodesic_distance": 5.0},
    }


def write_gz(path, value):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle)


def prepare(tmp_path, item=None, split="val_unseen", with_gt=False):
    episodes = tmp_path / "episodes.json.gz"
    write_gz(episodes, {"episodes": [episode() if item is None else item]})
    gt = tmp_path / "gt.json.gz"
    if with_gt:
        write_gz(
            gt,
            {
                "7": {
                    "actions": [1, 1, 0],
                    "locations": [[0, 0, 0], [2, 0, 0], [4, 0, 0]],
                    "forward_steps": 2,
                }
            },
        )
    return prepare_r2rce(
        episodes,
        tmp_path / "data/scene_datasets",
        split,
        gt if with_gt else None,
        source_kind="fixture",
    ), episodes


def test_import_real_format_preserves_instruction_and_private_evaluation(tmp_path):
    suite, source = prepare(tmp_path, with_gt=True)
    row = suite["episodes"][0]
    assert suite["schema_version"] == 2
    assert suite["task_family"] == "r2rce"
    assert suite["protocol_id"] == "r2rce_v13_dev"
    assert suite["source"]["kind"] == "fixture"
    assert suite["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert suite["source"]["split"] == "val_unseen"
    assert row["instruction"] == INSTRUCTION
    assert row["id"] == "7"
    assert row["trajectory_id"] == "42"
    assert row["scene_id"] == row["scene_family"] == "mp3d/example/example.glb"
    assert row["repeat_id"] == 0 and row["annotation_verified"] is True
    assert row["evaluation"]["gt_locations"] == [[0, 0, 0], [2, 0, 0], [4, 0, 0]]
    assert row["evaluation"]["initial_distance"] == 5
    assert "actions" not in json.dumps(row)
    task = task_spec(row)
    assert asdict(task)["instruction"] == INSTRUCTION
    assert not (
        {"evaluation", "goal_position", "reference_path", "gt_locations"} & asdict(task).keys()
    )
    assert asdict(evaluation_spec(row))["goal_position"] == [4, 0, 0]
    with pytest.raises(FrozenInstanceError):
        task.instruction = "leak"
    with pytest.raises(FrozenInstanceError):
        evaluation_spec(row).success_radius = 1
    env = environment_episode(row)
    assert env["goal_position"] == [4, 0, 0]
    assert env["success_radius"] == 3
    assert "evaluation" not in env
    validate_r2rce_suite(suite)


def test_missing_scene_list_is_absolute_unique_and_not_validation_error(tmp_path):
    suite, _ = prepare(tmp_path)
    scene = str((tmp_path / "data/scene_datasets/mp3d/example/example.glb").resolve())
    assert suite["missing_scenes"] == [scene]
    assert suite["episodes"][0]["scene"] == scene
    validate_r2rce_suite(suite)
    Path(scene).parent.mkdir(parents=True)
    Path(scene).touch()
    suite, _ = prepare(tmp_path)
    assert suite["missing_scenes"] == []


def test_missing_gt_and_initial_distance_remain_none(tmp_path):
    item = episode()
    del item["info"]
    suite, _ = prepare(tmp_path, item)
    evaluation = suite["episodes"][0]["evaluation"]
    assert evaluation["gt_locations"] is None
    assert evaluation["initial_distance"] is None


def test_official_test_without_goal_is_explicitly_unevaluable(tmp_path):
    item = episode()
    for key in ("goals", "reference_path", "info"):
        del item[key]
    suite, _ = prepare(tmp_path, item, split="test")
    row = suite["episodes"][0]
    assert row["split"] == "test"
    assert row["evaluable"] is False
    assert row["evaluation"]["goal_position"] is None
    assert row["evaluation"]["reference_path"] is None
    validate_r2rce_suite(suite)
    with pytest.raises(ValueError, match="goal"):
        environment_episode(row)


@pytest.mark.parametrize(
    "scene", ["../../outside.glb", "/tmp/outside.glb", "mp3d/../outside.glb", "mp3d\\outside.glb"]
)
def test_scene_path_escape_rejected(tmp_path, scene):
    item = episode()
    item["scene_id"] = scene
    with pytest.raises(ValueError, match="scene"):
        prepare(tmp_path, item)


def test_scene_symlink_escape_rejected(tmp_path):
    root = tmp_path / "data/scene_datasets"
    root.mkdir(parents=True)
    (root / "mp3d").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="scene"):
        prepare(tmp_path)


@pytest.mark.parametrize("mutation", ["duplicate", "split", "pose", "private", "goal", "nan"])
def test_suite_rejects_malformed_records(tmp_path, mutation):
    suite, _ = prepare(tmp_path)
    row = suite["episodes"][0]
    if mutation == "duplicate":
        suite["episodes"].append(copy.deepcopy(row))
    elif mutation == "split":
        row["split"] = "envdrop"
    elif mutation == "pose":
        row["start_rotation"] = [0, 0, 0, 0]
    elif mutation == "private":
        row["goal_position"] = [0, 0, 0]
    elif mutation == "goal":
        row["evaluation"]["goal_position"] = None
    else:
        row["start_position"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_r2rce_suite(suite)


@pytest.mark.parametrize("field", ["instruction", "start_position", "evaluation", "trajectory_id"])
def test_repeats_reject_conflicting_task_definitions(tmp_path, field):
    suite, _ = prepare(tmp_path)
    repeated = copy.deepcopy(suite["episodes"][0])
    repeated["repeat_id"] = 1
    if field == "instruction":
        repeated[field] = "Take a different route."
    elif field == "start_position":
        repeated[field] = [1, 0, 0]
    elif field == "trajectory_id":
        repeated[field] = "different"
    else:
        repeated[field]["goal_position"] = [8, 0, 0]
    suite["episodes"].append(repeated)
    with pytest.raises(ValueError, match="Conflicting.*repeat"):
        validate_r2rce_suite(suite)


def test_repeats_accept_identical_task_definitions(tmp_path):
    suite, _ = prepare(tmp_path)
    repeated = copy.deepcopy(suite["episodes"][0])
    repeated["repeat_id"] = 1
    suite["episodes"].append(repeated)
    validate_r2rce_suite(suite)
