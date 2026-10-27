import math
import sys

import pytest

from jev_navi.metrics import navigation_metrics


def test_stop_and_strict_three_meter_boundary():
    for stopped, distance, expected in [(True, 2.999, 1), (True, 3, 0), (False, 0, 0)]:
        result = navigation_metrics(
            [[0, 0, 0]], [distance], stopped=stopped, initial_distance=distance, algorithm="exact"
        )
        assert result["sr"] == expected
        assert result["os"] == int(distance < 3)
        assert result["ne"] == distance


def test_geodesic_errors_euclidean_length_and_spl():
    result = navigation_metrics(
        [[0, 0, 0], [3, 0, 4]], [10, 2], stopped=True, initial_distance=4, algorithm="exact"
    )
    assert result["path_length"] == 5
    assert result["spl"] == 0.8
    assert result["ne"] == 2
    assert result["os"] == 1


def test_service_failure_zeroes_success_but_keeps_oracle_reachability():
    result = navigation_metrics(
        [[0, 0, 0], [2, 0, 0]], [5, 1], stopped=True, initial_distance=5, service_failed=True
    )
    assert result["sr"] == result["spl"] == 0
    assert result["os"] == 1
    assert result["ne"] == 1


def test_missing_initial_distance_does_not_fabricate_spl():
    result = navigation_metrics([[0, 0, 0]], [1], stopped=True)
    assert result["spl"] is None
    assert result["ndtw"] is None
    assert result["ndtw_reason"] == "missing_gt_locations"


def test_failed_episode_spl_is_zero_even_without_initial_distance():
    assert navigation_metrics([[0, 0, 0]], [4], stopped=True)["spl"] == 0


def test_exact_ndtw_known_offset_and_consecutive_repeat_removal():
    # Two unit offset costs => exp(-2 / (2 * 3)).
    gt = [[0, 0, 0], [2, 0, 0]]
    positions = [[0, 1, 0], [0, 1, 0], [2, 1, 0]]
    result = navigation_metrics(
        positions, [4, 4, 2], stopped=True, gt_locations=gt, algorithm="exact"
    )
    assert result["ndtw"] == pytest.approx(math.exp(-1 / 3))
    assert result["ndtw_reason"] is None
    assert result["path_length"] == 2


def test_exact_matching_dense_gt_scores_one():
    gt = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    assert (
        navigation_metrics(gt, [5, 4, 3], stopped=False, gt_locations=gt, algorithm="exact")["ndtw"]
        == 1
    )


def test_missing_fastdtw_returns_reason_without_exact_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastdtw", None)
    result = navigation_metrics([[0, 0, 0]], [0], stopped=True, gt_locations=[[0, 0, 0]])
    assert result["ndtw"] is None
    assert result["ndtw_reason"] == "fastdtw_unavailable"


@pytest.mark.parametrize(
    "positions,distances,kwargs",
    [
        ([], [], {}),
        ([[0, 0, 0]], [], {}),
        ([[0, 0]], [1], {}),
        ([[float("nan"), 0, 0]], [1], {}),
        ([[0, 0, 0]], [float("inf")], {}),
        ([[0, 0, 0]], [-1], {}),
        ([[0, 0, 0]], [1], {"initial_distance": -1}),
        ([[0, 0, 0]], [1], {"success_distance": 0}),
        ([[0, 0, 0]], [1], {"gt_locations": []}),
        ([[0, 0, 0]], [1], {"gt_locations": [[0, 0, float("inf")]]}),
        ([[0, 0, 0]], [1], {"algorithm": "typo"}),
    ],
)
def test_invalid_trajectories_fail_explicitly(positions, distances, kwargs):
    with pytest.raises(ValueError):
        navigation_metrics(positions, distances, stopped=True, **kwargs)


def test_initial_zero_distance_and_stationary_success_have_unit_spl():
    assert navigation_metrics([[0, 0, 0]], [0], stopped=True, initial_distance=0)["spl"] == 1


def test_installed_fastdtw_matches_upstream_nonzero_golden():
    pytest.importorskip("fastdtw")
    from jev_navi.metrics import navigation_metrics

    result = navigation_metrics(
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
        [4, 4, 3, 2],
        stopped=True,
        initial_distance=4,
        gt_locations=[[0, 0, 0], [1, 0, 1], [2, 0, 0]],
        algorithm="fastdtw",
    )
    assert result["ndtw"] == pytest.approx(math.exp(-1 / 9))
    assert result["ndtw_reason"] is None
    assert result["sr"] == result["spl"] == result["os"] == 1.0
