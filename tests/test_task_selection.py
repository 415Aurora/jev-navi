import copy
import json
from pathlib import Path

import pytest

from jev_navi.datasets import prepare_r2rce
from jev_navi.task_selection import freeze_tasks, verify_selection


def source_suite(tmp_path, split="train"):
    rows = []
    for scene in range(5):
        for trajectory in range(7):
            for paraphrase in range(2):
                rows.append(
                    {
                        "episode_id": f"{scene}-{trajectory}-{paraphrase}",
                        "trajectory_id": trajectory,
                        "scene_id": f"mp3d/s{scene}/s{scene}.glb",
                        "instruction": {
                            "instruction_text": f"Walk route {trajectory}, version {paraphrase}."
                        },
                        "start_position": [0, 0, 0],
                        "start_rotation": [0, 0, 0, 1],
                        "goals": [{"position": [0, 0, -5]}],
                        "info": {"geodesic_distance": 5},
                    }
                )
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"episodes": rows}))
    return prepare_r2rce(path, tmp_path / "assets", split, source_kind="fixture")


def test_freeze_is_deterministic_and_distinct_without_assets(tmp_path):
    suite = source_suite(tmp_path)
    original = copy.deepcopy(suite)
    selected, manifest = freeze_tasks(suite)
    assert suite == original and len(selected["episodes"]) == 24
    assert selected["selection"] == manifest
    assert len({r["scene_id"] for r in selected["episodes"]}) == 4
    assert len({(r["scene_id"], r["trajectory_id"]) for r in selected["episodes"]}) == 24
    reversed_suite = copy.deepcopy(suite)
    reversed_suite["episodes"].reverse()
    assert freeze_tasks(reversed_suite) == (selected, manifest)
    assert verify_selection(selected, manifest)["verified"]
    assert "instruction" not in json.dumps(manifest)
    assert "goal_position" not in json.dumps(manifest)


@pytest.mark.parametrize("split", ["val_seen", "val_unseen", "test"])
def test_heldout_splits_cannot_be_selected(tmp_path, split):
    with pytest.raises(ValueError, match="train"):
        freeze_tasks(source_suite(tmp_path, split))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenes": True},
        {"scenes": 0},
        {"per_scene": 0},
        {"seed": ""},
        {"scenes": 6},
        {"per_scene": 8},
    ],
)
def test_invalid_or_insufficient_selection_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        freeze_tasks(source_suite(tmp_path), **kwargs)


def test_repeat_and_missing_trajectory_cannot_be_frozen(tmp_path):
    suite = source_suite(tmp_path)
    suite["episodes"][0]["trajectory_id"] = None
    with pytest.raises(ValueError, match="source"):
        freeze_tasks(suite)
    suite = source_suite(tmp_path)
    suite["episodes"][0]["repeat_id"] = 1
    with pytest.raises(ValueError, match="source"):
        freeze_tasks(suite)


@pytest.mark.parametrize("field", ["instruction", "evaluation", "source", "membership", "manifest"])
def test_frozen_tampering_is_rejected(tmp_path, field):
    selected, manifest = freeze_tasks(source_suite(tmp_path))
    if field == "instruction":
        selected["episodes"][0]["instruction"] += " changed"
    elif field == "evaluation":
        selected["episodes"][0]["evaluation"]["goal_position"][2] -= 1
    elif field == "source":
        Path(selected["source"]["path"]).write_text("{}")
    elif field == "membership":
        selected["episodes"].pop()
    else:
        manifest["tasks"][0]["definition_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_selection(selected, manifest)


def test_cannot_freeze_prepared_rows_changed_from_raw_source(tmp_path):
    suite = source_suite(tmp_path)
    suite["episodes"][0]["instruction"] = "silently rewritten"
    with pytest.raises(ValueError, match="source"):
        freeze_tasks(suite)


def test_paths_can_be_relocated_with_identical_sources(tmp_path):
    suite = source_suite(tmp_path)
    selected, manifest = freeze_tasks(suite)
    new_root = tmp_path / "new-assets"
    selected["source"]["scenes_root"] = str(new_root)
    for row in selected["episodes"]:
        row["scene"] = str(new_root / row["scene_id"])
    assert verify_selection(selected, manifest)["verified"]


def test_raw_missing_trajectory_is_rejected(tmp_path):
    suite = source_suite(tmp_path)
    path = Path(suite["source"]["path"])
    data = json.loads(path.read_text())
    data["episodes"][0]["trajectory_id"] = None
    path.write_text(json.dumps(data))
    suite = prepare_r2rce(path, tmp_path / "assets", "train", source_kind="fixture")
    with pytest.raises(ValueError, match="trajectory"):
        freeze_tasks(suite)


def test_readiness_verifies_embedded_and_external_manifest(tmp_path):
    from jev_navi.readiness import check_r2rce

    selected, manifest = freeze_tasks(source_suite(tmp_path))
    report = check_r2rce(selected, selection=manifest)
    assert report["selection_integrity"]["verified"]
    assert not report["ready"]  # missing scenes, never counted as navigation failures
    assert report["unique_trajectories"] == 24
    assert len(report["scene_inventory"]) == 4
    selected["episodes"][0]["instruction"] = "changed"
    report = check_r2rce(selected)
    assert not report["selection_integrity"]["verified"] and not report["ready"]


def test_run_rejects_changed_selection_before_budget_and_output(tmp_path, monkeypatch):
    from jev_navi import experiments

    selected, _ = freeze_tasks(source_suite(tmp_path))
    selected["episodes"][0]["instruction"] = "changed"

    def forbidden(*args, **kwargs):
        pytest.fail("must reject before budget or environment initialization")

    monkeypatch.setattr(experiments, "make_run_budget", forbidden)
    config = {
        "protocol_id": "r2rce_v13_dev",
        "environment": {"backend": "synthetic"},
        "policy": {"kind": "scripted", "actions": ["STOP"]},
        "run": {"seed": 17, "call_limit": 30},
    }
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="selection"):
        experiments.run_suite(config, selected, output)
    assert not output.exists()


def test_cli_freeze_and_check_have_no_network_or_ledger_side_effects(tmp_path, monkeypatch, capsys):
    import httpx

    from jev_navi import budget
    from jev_navi.cli import main

    def forbidden(*args, **kwargs):
        pytest.fail("freeze and file preflight must have no side effects")

    monkeypatch.setattr(httpx.Client, "request", forbidden)
    monkeypatch.setattr(budget, "initialize_ledger", forbidden)
    original = tmp_path / "suite.json"
    original.write_text(json.dumps(source_suite(tmp_path)))
    output, manifest = tmp_path / "selected.json", tmp_path / "manifest.json"
    args = [
        "freeze-r2rce",
        "--suite",
        str(original),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]
    assert main(args) == 0  # freezes definitions even without assets
    assert json.loads(capsys.readouterr().out)["http_requests"] == 0
    before = output.read_bytes(), manifest.read_bytes()
    assert main(args) == 2
    assert before == (output.read_bytes(), manifest.read_bytes())
    capsys.readouterr()
    assert main(["check-r2rce", "--suite", str(output), "--selection", str(manifest)]) == 2
    assert json.loads(capsys.readouterr().out)["selection_integrity"]["verified"]
