import copy
import json

import pytest


def fixture_suite():
    episodes = []
    for scene, split in [("room-a", "dev"), ("room-b", "holdout"), ("room-c", "holdout")]:
        for index in range(12):
            episodes.append(
                {
                    "id": f"{scene}-{index}",
                    "scene_id": scene,
                    "scene_family": scene,
                    "split": split,
                    "task_type": ["visible", "turn", "disambiguate"][index % 3],
                    "instruction": f"go to door {index}",
                    "scene": "synthetic",
                    "start_position": [0, 0, 0],
                    "start_rotation": [0, 0, 0, 1],
                    "goal_position": [0, 0, -0.5],
                    "success_radius": 0.2,
                    "annotation_verified": True,
                }
            )
    return {"schema_version": 1, "episodes": episodes}


def test_suite_validation_and_permutation():
    from jev_navi.experiments import instruction_permutation, validate_suite

    suite = fixture_suite()
    validate_suite(suite, diagnostic=True)
    permutation = instruction_permutation(suite["episodes"], seed=17)
    assert permutation == instruction_permutation(suite["episodes"], seed=17)
    for row in suite["episodes"]:
        source = next(x for x in suite["episodes"] if x["id"] == permutation[row["id"]])
        assert source["id"] != row["id"]
        assert source["instruction"] != row["instruction"]
        assert source["split"] == row["split"]
        assert source["scene_id"] == row["scene_id"]


def test_holdout_family_overlap_rejected():
    from jev_navi.experiments import validate_suite

    suite = fixture_suite()
    suite["episodes"][12]["scene_family"] = "room-a"
    with pytest.raises(ValueError, match="family"):
        validate_suite(suite, diagnostic=True)


def test_unverified_annotations_cannot_run_as_research():
    from jev_navi.experiments import validate_suite

    suite = fixture_suite()
    suite["episodes"][0]["annotation_verified"] = False
    with pytest.raises(ValueError, match="annotation"):
        validate_suite(suite, diagnostic=True)


def test_run_suite_conditions_and_partial_budget(tmp_path):
    from jev_navi.experiments import run_suite

    config = {
        "environment": {"backend": "synthetic"},
        "policy": {"kind": "scripted", "actions": ["MOVE_FORWARD", "MOVE_FORWARD", "STOP"]},
        "run": {"max_decisions": 3, "seed": 17, "call_limit": 500},
        "conditions": ["full", "no_image_history", "shuffled_instruction"],
    }
    suite = fixture_suite()
    suite["episodes"] = suite["episodes"][:2]
    results = run_suite(config, suite, tmp_path / "suite")
    assert len(results) == 6
    assert all(x["environment_kind"] == "synthetic_fixture" for x in results)
    assert (tmp_path / "suite" / "suite_manifest.json").exists()
    assert (tmp_path / "suite" / "report.md").exists()
    assert "NOT RESEARCH EVIDENCE" in (tmp_path / "suite" / "report.md").read_text()


def test_report_pairing_and_gate_requires_same_model():
    from jev_navi.reporting import summarize

    rows = []
    for cond in ["full", "shuffled_instruction"]:
        for i in range(4):
            rows.append(
                {
                    "success": cond == "full",
                    "ever_reached": True,
                    "end_reason": "stop",
                    "decisions": 3,
                    "blocked_actions": 0,
                    "model_calls": 3,
                    "http_requests": 3,
                    "latency_s": [0.1, 0.2],
                    "models": ["model-a"],
                    "path_length": 0.5,
                    "collisions": 0,
                    "environment_kind": "habitat",
                    "metadata": {
                        "episode_id": str(i),
                        "split": "holdout",
                        "scene_family": "house-b",
                        "condition": cond,
                        "policy_kind": "startlux",
                        "annotation_verified": True,
                    },
                }
            )
    report = summarize(rows)
    assert report["paired"][0]["success_difference"] == 1.0
    assert report["engineering_gate"][0]["passed"] is None
    assert report["engineering_gate"][0]["status"] == "insufficient_or_incomplete"
    changed = copy.deepcopy(rows)
    for r in changed[4:]:
        r["models"] = ["model-b"]
    assert summarize(changed)["paired"] == []


def test_motion_protocol_cannot_diverge_from_prompts(tmp_path):
    from jev_navi.experiments import run_suite

    config = {
        "environment": {"backend": "synthetic", "turn_deg": 90},
        "policy": {"kind": "scripted", "actions": ["STOP"]},
    }
    with pytest.raises(ValueError, match="fixed"):
        run_suite(config, fixture_suite(), tmp_path / "wrong-motion")


def test_cli_smoke_runs_without_habitat(tmp_path, capsys):
    from jev_navi.cli import main

    assert main(["smoke", "--backend", "synthetic", "--output", str(tmp_path / "smoke")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["environment_kind"] == "synthetic_fixture"


def test_cross_model_comparison_requires_frozen_matching_protocol():
    from jev_navi.reporting import summarize

    rows = []
    for model in ["model-a", "model-b"]:
        for i in range(4):
            rows.append(
                {
                    "success": model == "model-a",
                    "ever_reached": True,
                    "end_reason": "stop",
                    "decisions": 1,
                    "blocked_actions": 0,
                    "model_calls": 1,
                    "http_requests": 1,
                    "latency_s": [0.1],
                    "models": [model],
                    "path_length": 1,
                    "collisions": 0,
                    "environment_kind": "habitat",
                    "metadata": {
                        "episode_id": str(i),
                        "split": "holdout",
                        "scene_family": f"house-{i // 2}",
                        "condition": "full",
                        "policy_kind": "startlux",
                        "annotation_verified": True,
                        "suite_sha256": "same-suite",
                        "protocol_sha256": "same-protocol",
                        "config_sha256": model,
                    },
                }
            )
    comparisons = summarize(rows)["model_comparisons"]
    assert len(comparisons) == 1
    assert comparisons[0]["success_difference"] == 1.0
    rows[-1]["metadata"]["protocol_sha256"] = "different-camera"
    comparisons = summarize(rows)["model_comparisons"]
    assert comparisons[0]["paired_episodes"] == 3
    assert comparisons[0]["missing_from_right"] == ["3"]


def test_shuffled_semantic_targets_are_different():
    from jev_navi.experiments import instruction_permutation

    episodes = fixture_suite()["episodes"]
    for row in episodes:
        row["target_region_id"] = str(int(row["id"].rsplit("-", 1)[1]) % 4)
    mapping = instruction_permutation(episodes)
    index = {row["id"]: row for row in episodes}
    assert all(
        index[k]["target_region_id"] != index[v]["target_region_id"] for k, v in mapping.items()
    )
