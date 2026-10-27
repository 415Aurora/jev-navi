import copy
import json

import pytest


def route_suite(tmp_path):
    from jev_navi.datasets import prepare_r2rce

    raw = {
        "episodes": [
            {
                "episode_id": 1,
                "trajectory_id": 9,
                "scene_id": "mp3d/example/example.glb",
                "instruction": {"instruction_text": "Turn left. Move forward. Stop."},
                "start_position": [0, 0, 0],
                "start_rotation": [0, 0, 0, 1],
                "goals": [{"position": [0, 0, -4], "radius": 3}],
                "reference_path": [[0, 0, 0], [0, 0, -4]],
                "info": {"geodesic_distance": 4},
            }
        ]
    }
    source = tmp_path / "episodes.json"
    source.write_text(json.dumps(raw))
    return prepare_r2rce(source, tmp_path / "scenes", "train", source_kind="fixture")


def fixture_config(tmp_path):
    records = [
        {"choices": {"boundary_0": "split", "boundary_1": "split"}},
        {"choices": {"progress": "continue", "action": "TURN_LEFT"}},
        {"choices": {"progress": "complete", "action": "TURN_RIGHT"}},
        {"choices": {"progress": "continue", "action": "MOVE_FORWARD"}},
        {"choices": {"progress": "complete", "action": "MOVE_FORWARD"}},
        {"choices": {"progress": "complete", "action": "STOP"}},
    ]
    path = tmp_path / "responses.json"
    path.write_text(json.dumps(records))
    return {
        "protocol_id": "r2rce_v13_dev",
        "environment": {"backend": "synthetic"},
        "policy": {"kind": "startlux_hierarchical_fixture", "responses": str(path)},
        "run": {"max_decisions": 10, "call_limit": 30},
        "execution": {
            "mode": "offline",
            "phase_id": "r2rce-offline-v1",
            "ledger": str(tmp_path / "shared.sqlite3"),
        },
    }


def test_v2_run_has_separate_repeat_identity_and_fixture_accounting(tmp_path):
    from jev_navi.experiments import run_suite

    suite = route_suite(tmp_path)
    repeated = copy.deepcopy(suite["episodes"][0])
    repeated["repeat_id"] = 1
    suite["episodes"].append(repeated)
    output = tmp_path / "out"
    rows = run_suite(fixture_config(tmp_path), suite, output)
    assert len(rows) == 2
    assert all(row["executed_steps"] == 3 for row in rows)
    assert all(row["http_requests"] == 0 for row in rows)
    assert all(row["fixture_requests"] == 6 for row in rows)
    manifest = json.loads((output / "suite_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["budget"]["used"] == 0
    assert manifest["fixture_requests"] == 12
    assert all(
        row["metadata"]["request_budget"] == {"run_limit": 30, "phase_limit": 0} for row in rows
    )
    assert manifest["phase_budget"]["phase"]["used"] == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["groups"][0]["episodes"] == 2
    assert summary["groups"][0]["metric_coverage"]["ndtw"]["valid"] == 0
    assert summary["engineering_gate"] == []
    assert summary["fixture_rows"] == 2
    assert "NOT RESEARCH EVIDENCE" in (output / "report.md").read_text()


def test_missing_scene_is_preflight_failure_not_strategy_score(tmp_path):
    from jev_navi.experiments import run_suite

    config = fixture_config(tmp_path)
    config["environment"]["backend"] = "habitat"
    output = tmp_path / "missing"
    with pytest.raises(ValueError, match="Unrunnable.*missing"):
        run_suite(config, route_suite(tmp_path), output)
    assert not output.exists()


def test_cli_prepares_missing_assets_without_model_requests(tmp_path, capsys):
    from jev_navi.cli import main

    suite = route_suite(tmp_path)
    output = tmp_path / "prepared.json"
    code = main(
        [
            "prepare-r2rce",
            "--episodes",
            str(tmp_path / "episodes.json"),
            "--scenes-root",
            str(tmp_path / "scenes"),
            "--split",
            "train",
            "--source-kind",
            "fixture",
            "--output",
            str(output),
        ]
    )
    assert code == 2
    assert json.loads(output.read_text())["missing_scenes"] == suite["missing_scenes"]
    assert json.loads(capsys.readouterr().out)["http_requests"] == 0


def test_route_metrics_include_service_failure_and_missing_gt_coverage():
    from jev_navi.reporting import summarize

    def row(condition, repeat, failed):
        return {
            "metadata": {
                "episode_id": "1",
                "repeat_id": repeat,
                "condition": condition,
                "policy_kind": "startlux_hierarchical",
                "measurement_kind": "model",
                "protocol_id": "r2rce_v13_dev",
                "task_family": "r2rce",
                "split": "val_unseen",
                "scene_family": "scene",
                "suite_sha256": "suite",
                "config_sha256": "same",
                "protocol_sha256": "physics",
                "trajectory_id": "9",
                "think_mode": "auto",
                "request_budget": {"run_limit": 64, "phase_limit": 1000},
            },
            "models": [] if failed else ["model-a"],
            "success": not failed,
            "benchmark_metrics": {
                "sr": 0.0 if failed else 1.0,
                "spl": 0.0 if failed else 1.0,
                "ne": 4.0 if failed else 0.5,
                "os": 0.0 if failed else 1.0,
                "ndtw": None,
                "ndtw_reason": "missing_gt_locations",
            },
            "environment_kind": "habitat",
            "end_reason": "service_error" if failed else "stop",
            "path_length": 1.0,
            "collisions": 0,
            "decisions": 3,
            "blocked_actions": 0,
            "http_requests": 2,
            "model_calls": 2,
            "latency_s": [1.0],
            "executed_steps": 2,
        }

    rows = [
        row(c, r, failed=(c == "full" and r == 1))
        for c in ("full", "shuffled_instruction")
        for r in (0, 1)
    ]
    manifest = {
        "suite": {"schema_version": 2, "episodes": [{"id": "1", "repeat_id": r} for r in (0, 1)]},
        "planned_jobs": 4,
        "completed_jobs": 4,
    }
    summary = summarize(rows, manifest)
    full = next(g for g in summary["groups"] if g["condition"] == "full")
    assert full["success_rate"] == 0.5
    assert full["episodes"] == 2
    assert full["metric_coverage"]["ndtw"] == {"valid": 0, "total": 2}
    assert summary["paired"][0]["paired_episodes"] == 2
    assert summary["paired"][0]["planned_pairs"] == 2
    assert summary["paired"][0]["service_failures"] == 1
    assert summary["engineering_gate"] == []


def test_offline_suite_rejects_real_policy_before_output_or_network(tmp_path):
    from jev_navi.experiments import run_suite

    config = fixture_config(tmp_path)
    config["policy"] = {"kind": "startlux"}
    config["free_quota_confirmed"] = True
    with pytest.raises(ValueError, match="Offline"):
        run_suite(config, route_suite(tmp_path), tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()
