import copy
import hashlib

import pytest

from jev_navi.offline_validation import control_cases, run_route_controls


def catalog(tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    asset = scenes / "fixture.glb"
    asset.write_bytes(b"file-level fixture, never native navigation evidence")
    positions = [[0, 0, -0.25 * step] for step in range(10)]
    positions.append(positions[-1])
    return {
        "schema_version": 1,
        "kind": "privileged_software_fixture",
        "routes": [
            {
                "id": "straight-fixture",
                "scene_id": "fixture.glb",
                "scene_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "instruction": "Follow the prepared route to the endpoint. Stop.",
                "start_position": [0, 0, 0],
                "start_rotation": [0, 0, 0, 1],
                "goal_position": [0, 0, -5],
                "actions": ["MOVE_FORWARD"] * 9 + ["STOP"],
                "gt_locations": positions,
            }
        ],
    }, scenes


def test_control_cases_have_distinct_stop_and_semantic_contracts():
    cases = control_cases(["MOVE_FORWARD", "STOP"])
    assert [case["name"] for case in cases] == [
        "premature_stop",
        "arrival_stop",
        "arrival_no_stop",
        "hierarchical_stop",
    ]
    assert cases[0]["responses"] == [{"choices": {"action": "STOP"}}]
    assert cases[2]["max_decisions"] == 1
    hierarchy = cases[3]
    assert hierarchy["responses"][-2]["choices"] == {"progress": "complete", "action": "STOP"}
    assert hierarchy["responses"][-1]["choices"] == {"progress": "complete", "action": "STOP"}
    assert hierarchy["max_decisions"] == 3


@pytest.mark.parametrize(
    "actions",
    [
        [],
        ["STOP"],
        ["MOVE_FORWARD"],
        ["STOP", "MOVE_FORWARD", "STOP"],
        ["TELEPORT", "STOP"],
        ["MOVE_FORWARD"] * 499 + ["STOP"],
    ],
)
def test_invalid_or_unrepresentable_routes_are_rejected(actions):
    with pytest.raises(ValueError):
        control_cases(actions)


def test_real_runner_controls_distinguish_arrival_from_success_and_preserve_budgets(tmp_path):
    data, scenes = catalog(tmp_path)
    report = run_route_controls(
        data, tmp_path / "verification", scenes_root=scenes, backend="synthetic"
    )
    assert report["verified"] is True
    assert report["research_evidence"] is False
    assert report["http_requests"] == 0
    assert report["episodes"] == 8
    assert report["geometry_controls_passed"] == 4
    assert report["phase_budget_unchanged"] is True
    results = {row["control"]: row for row in report["controls"] if row["condition"] == "no_guard"}
    assert results["premature_stop"]["sr"] == results["premature_stop"]["os"] == 0
    assert results["arrival_stop"]["sr"] == results["arrival_stop"]["os"] == 1
    assert results["arrival_no_stop"]["sr"] == 0
    assert results["arrival_no_stop"]["os"] == 1
    assert results["arrival_no_stop"]["end_reason"] == "decision_budget"
    assert results["hierarchical_stop"]["semantic_vetoes"] == 1
    assert all(row["initial_distance"] > 3 for row in results.values())
    assert all(
        row["ndtw"] == pytest.approx(1) for name, row in results.items() if name != "premature_stop"
    )


def test_changed_scene_fails_before_execution(tmp_path):
    data, scenes = catalog(tmp_path)
    (scenes / "fixture.glb").write_bytes(b"modified")
    with pytest.raises(ValueError, match="scene.*hash"):
        run_route_controls(data, tmp_path / "bad", scenes_root=scenes, backend="synthetic")


def test_fixture_marker_is_required(tmp_path):
    data, scenes = catalog(tmp_path)
    data["kind"] = "model"
    with pytest.raises(ValueError, match="fixture"):
        run_route_controls(data, tmp_path / "bad", scenes_root=scenes, backend="synthetic")


def test_duplicate_route_ids_do_not_overwrite_outputs(tmp_path):
    data, scenes = catalog(tmp_path)
    data["routes"].append(copy.deepcopy(data["routes"][0]))
    with pytest.raises(ValueError, match="unique"):
        run_route_controls(data, tmp_path / "bad", scenes_root=scenes, backend="synthetic")


def test_route_ids_cannot_escape_output_directory(tmp_path):
    data, scenes = catalog(tmp_path)
    data["routes"][0]["id"] = "../escaped"
    with pytest.raises(ValueError, match="filename"):
        run_route_controls(data, tmp_path / "bad", scenes_root=scenes, backend="synthetic")
    assert not (tmp_path / "escaped").exists()


def test_successful_run_output_cannot_be_overwritten(tmp_path):
    data, scenes = catalog(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing data")
    with pytest.raises(FileExistsError):
        run_route_controls(data, output, scenes_root=scenes, backend="synthetic")
    assert sentinel.read_text() == "existing data"


def test_native_action_mapping_preserves_rotation_and_stop():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/author_route_controls.py"
    spec = importlib.util.spec_from_file_location("native_author", path)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    assert author.map_actions([2, 1, 3, None]) == [
        "TURN_LEFT",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "STOP",
    ]
    for actions in ([True, None], [1, None, 1], [1, 9, None], [1]):
        with pytest.raises(ValueError):
            author.map_actions(actions)


@pytest.mark.parametrize("fault", ["http", "service", "metric"])
def test_failed_controls_remain_in_audit_summary(tmp_path, monkeypatch, fault):
    import json

    from jev_navi import offline_validation

    data, scenes = catalog(tmp_path)
    original_run = offline_validation.run_suite
    original_check = offline_validation._check_control

    def altered_run(*args, **kwargs):
        rows = original_run(*args, **kwargs)
        if fault == "http":
            rows[0]["http_requests"] = 1
        elif fault == "service":
            rows[0]["error_code"] = "invalid_response"
            rows[0]["end_reason"] = "service_error"
        return rows

    def altered_check(row, control, initial_distance):
        if fault == "metric" and control == "arrival_stop":
            raise ValueError("injected metric mismatch")
        return original_check(row, control, initial_distance)

    monkeypatch.setattr(offline_validation, "run_suite", altered_run)
    monkeypatch.setattr(offline_validation, "_check_control", altered_check)
    output = tmp_path / "failed-audit"
    with pytest.raises(ValueError):
        run_route_controls(data, output, scenes_root=scenes, backend="synthetic")
    report = json.loads((output / "control-summary.json").read_text())
    assert not report["verified"]
    assert report["failure"]
    assert report["episodes"] == (4 if fault == "metric" else 2)
    if fault == "http":
        assert report["http_requests"] == 1
        assert any(row["http_requests"] == 1 for row in report["controls"])
    elif fault == "service":
        assert any(row["error_code"] == "invalid_response" for row in report["controls"])
    else:
        assert any(row["control"] == "arrival_stop" for row in report["controls"])


def test_partial_results_are_recorded_before_missing_pair_failure(tmp_path, monkeypatch):
    import json

    from jev_navi import offline_validation

    data, scenes = catalog(tmp_path)
    original = offline_validation.run_suite

    def partial(*args, **kwargs):
        return original(*args, **kwargs)[:1]

    monkeypatch.setattr(offline_validation, "run_suite", partial)
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="Both guard"):
        run_route_controls(data, output, scenes_root=scenes, backend="synthetic")
    report = json.loads((output / "control-summary.json").read_text())
    assert not report["verified"] and report["failure"]
    assert report["episodes"] == len(report["controls"]) == 1
    assert report["fixture_requests"] > 0


@pytest.mark.parametrize("when", ["initial", "final"])
def test_budget_errors_still_persist_an_unverified_summary(tmp_path, monkeypatch, when):
    import json

    from jev_navi import offline_validation

    data, scenes = catalog(tmp_path)
    original = offline_validation.initialize_ledger
    calls = 0

    def budget_check(path):
        nonlocal calls
        calls += 1
        if calls == (1 if when == "initial" else 2):
            raise RuntimeError("ledger-verification-error")
        return original(path)

    monkeypatch.setattr(offline_validation, "initialize_ledger", budget_check)
    output = tmp_path / "budget-error"
    with pytest.raises(RuntimeError, match="ledger-verification-error"):
        run_route_controls(data, output, scenes_root=scenes, backend="synthetic")
    report = json.loads((output / "control-summary.json").read_text())
    assert not report["verified"]
    assert report["phase_budget_unchanged"] is None
    assert "ledger-verification-error" in report["budget_error"]


def test_secondary_budget_error_does_not_replace_original_control_failure(tmp_path, monkeypatch):
    import json

    from jev_navi import offline_validation

    data, scenes = catalog(tmp_path)
    original = offline_validation.initialize_ledger
    calls = 0

    def budget_check(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("secondary-ledger-error")
        return original(path)

    def invalid_control(*args):
        raise ValueError("primary-control-error")

    monkeypatch.setattr(offline_validation, "initialize_ledger", budget_check)
    monkeypatch.setattr(offline_validation, "_check_control", invalid_control)
    output = tmp_path / "two-errors"
    with pytest.raises(ValueError, match="primary-control-error"):
        run_route_controls(data, output, scenes_root=scenes, backend="synthetic")
    report = json.loads((output / "control-summary.json").read_text())
    assert "primary-control-error" in report["failure"]
    assert "secondary-ledger-error" in report["budget_error"]
    assert report["phase_budget_unchanged"] is None
