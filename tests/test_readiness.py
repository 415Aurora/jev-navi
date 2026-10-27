import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from jev_navi.readiness import check_r2rce


def suite(tmp_path, kind="fixture", with_gt=False):
    from jev_navi.datasets import prepare_r2rce

    root = tmp_path / "scenes"
    (root / "mp3d/test").mkdir(parents=True)
    (root / "mp3d/test/test.glb").write_bytes(b"not a real scene; file-check fixture")
    source = tmp_path / "episodes.json"
    source.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "1",
                        "trajectory_id": 7,
                        "scene_id": "mp3d/test/test.glb",
                        "instruction": {"instruction_text": "Walk forward and stop."},
                        "start_position": [0, 0, 0],
                        "start_rotation": [0, 0, 0, 1],
                        "goals": [{"position": [0, 0, -5], "radius": 3}],
                    }
                ]
            }
        )
    )
    gt = tmp_path / "gt.json"
    if with_gt:
        gt.write_text(json.dumps({"1": {"locations": [[0, 0, 0], [0, 0, -5]]}}))
    return prepare_r2rce(source, root, "train", gt if with_gt else None, source_kind=kind)


def test_ready_fixture_is_file_checked_without_claiming_official_or_live_consent(tmp_path):
    report = check_r2rce(suite(tmp_path))
    assert report["ready"] is True
    assert report["source_kind"] == "fixture"
    assert report["declared_official_data_ready"] is False
    assert report["http_requests"] == 0
    assert report["episodes"] == report["scenes"] == 1
    assert report["geometry_verified"] is False
    assert report["live_authorized"] is False


def test_official_is_a_provenance_declaration_not_authorization(tmp_path):
    report = check_r2rce(suite(tmp_path, kind="official", with_gt=True), require_ndtw=True)
    assert report["ready"] and report["declared_official_data_ready"]
    assert report["ndtw_coverage"] == {"available": 1, "total": 1}
    assert report["live_authorized"] is False
    assert "declaration" in report["note"]


def test_missing_scene_fails_readiness_even_when_cached_missing_list_is_empty(tmp_path):
    data = suite(tmp_path)
    Path(data["episodes"][0]["scene"]).unlink()
    report = check_r2rce(data)
    assert not report["ready"]
    assert report["missing_scenes"] == [data["episodes"][0]["scene"]]


@pytest.mark.parametrize("mutation", ["removed", "changed"])
def test_missing_or_changed_source_fails_hash_verification(tmp_path, mutation):
    data = suite(tmp_path)
    source = Path(data["source"]["path"])
    source.unlink() if mutation == "removed" else source.write_text("different")
    report = check_r2rce(data)
    assert not report["ready"]
    assert not report["source_integrity"]["episodes"]["verified"]


def test_changed_gt_fails_integrity_even_when_normalized_locations_remain(tmp_path):
    data = suite(tmp_path, with_gt=True)
    Path(data["source"]["gt_path"]).write_text("different")
    report = check_r2rce(data, require_ndtw=True)
    assert not report["ready"]
    assert not report["source_integrity"]["gt"]["verified"]


def test_missing_gt_is_explicit_and_only_required_when_requested(tmp_path):
    data = suite(tmp_path)
    assert check_r2rce(data)["ready"]
    report = check_r2rce(data, require_ndtw=True)
    assert not report["ready"]
    assert report["ndtw_coverage"] == {"available": 0, "total": 1}
    assert report["missing_gt_episodes"] == [["1", 0]]


def test_unevaluable_test_split_is_not_locally_ready(tmp_path):
    data = suite(tmp_path)
    data["source"]["split"] = data["episodes"][0]["split"] = "test"
    data["episodes"][0]["evaluation"]["goal_position"] = None
    data["episodes"][0]["evaluable"] = False
    report = check_r2rce(data)
    assert not report["ready"]
    assert report["unevaluable_episodes"] == [["1", 0]]


def test_empty_suite_is_not_ready(tmp_path):
    data = suite(tmp_path)
    data["episodes"] = []
    assert not check_r2rce(data)["ready"]


def test_readiness_does_not_mutate_prepared_suite(tmp_path):
    data = suite(tmp_path)
    original = copy.deepcopy(data)
    check_r2rce(data)
    assert data == original


def test_cli_readiness_has_no_model_transport_or_budget_side_effects(tmp_path, monkeypatch, capsys):
    from jev_navi import budget
    from jev_navi.cli import main

    def forbidden(*args, **kwargs):
        pytest.fail("Readiness must not initialize budgets or send HTTP")

    monkeypatch.setattr(httpx.Client, "request", forbidden)
    monkeypatch.setattr(budget, "initialize_ledger", forbidden)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite(tmp_path)))
    assert main(["check-r2rce", "--suite", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["ready"]
    assert main(["check-r2rce", "--suite", str(path), "--require-ndtw"]) == 2
    assert not json.loads(capsys.readouterr().out)["ready"]


def test_source_integrity_reports_expected_and_actual_hash(tmp_path):
    data = suite(tmp_path)
    report = check_r2rce(data)
    check = report["source_integrity"]["episodes"]
    assert (
        check["expected_sha256"]
        == check["actual_sha256"]
        == hashlib.sha256(Path(data["source"]["path"]).read_bytes()).hexdigest()
    )
