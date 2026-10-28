import json

import pytest
from test_offline_validation import catalog

from jev_navi.offline_validation import control_cases, run_route_controls


def test_d3_controls_contain_three_raw_protocol_cases():
    cases = control_cases(["MOVE_FORWARD", "STOP"], provider="d3")
    assert [c["name"] for c in cases] == ["premature_stop", "arrival_stop", "arrival_no_stop"]
    assert all(c["kind"] == "d3_fixture" for c in cases)
    stop = cases[0]["responses"][0]
    assert stop["method"] == "POST" and stop["path"] == "/apps/run"
    assert stop["json"]["data"]["output"]["choices"]["action"]["probabilities"] == {
        "MOVE_FORWARD": 0,
        "TURN_LEFT": 0,
        "TURN_RIGHT": 0,
        "STOP": 1,
    }
    assert len(control_cases(["TURN_LEFT"] * 499 + ["STOP"], provider="d3")[1]["responses"]) == 500
    with pytest.raises(ValueError):
        control_cases(["TURN_LEFT"] * 500 + ["STOP"], provider="d3")


def test_d3_route_controls_use_real_parser_and_exclude_hierarchy(tmp_path):
    data, scenes = catalog(tmp_path)
    output = tmp_path / "controls"
    report = run_route_controls(
        data, output, scenes_root=scenes, backend="synthetic", provider="d3"
    )
    assert report["verified"] and not report["research_evidence"]
    assert report["provider"] == "d3"
    assert report["episodes"] == 6 and report["geometry_controls_passed"] == 3
    assert report["http_requests"] == 0 and report["fixture_requests"] == 40
    assert report["provider_attempts"] == {
        "http_attempts": 40,
        "model_submissions": 40,
        "poll_attempts": 0,
        "retry_attempts": 0,
    }
    assert report["phase_budget_unchanged"]
    assert all(r["semantic_vetoes"] == 0 for r in report["controls"])
    rows = {r["control"]: r for r in report["controls"] if r["condition"] == "no_guard"}
    assert rows["arrival_stop"]["sr"] == 1
    assert rows["arrival_no_stop"]["sr"] == 0 and rows["arrival_no_stop"]["os"] == 1
    for path in output.rglob("suite_manifest.json"):
        manifest = json.loads(path.read_text())
        assert manifest["prompt_version"] == "r2rce-direct-v1"
        assert manifest["measurement_kind"] == "fixture"
    assert "NOT RESEARCH EVIDENCE" in (output / "control-report.md").read_text()
