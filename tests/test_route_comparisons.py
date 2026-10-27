import copy

import pytest

from jev_navi.route_reporting import summarize_routes


def result(policy="startlux", config="a", limit=64, phase_limit=1000, condition="full"):
    return {
        "metadata": {
            "episode_id": "1",
            "repeat_id": 0,
            "condition": condition,
            "policy_kind": policy,
            "measurement_kind": "model",
            "protocol_id": "r2rce_v13_dev",
            "task_family": "r2rce",
            "split": "val_unseen",
            "scene_family": "scene",
            "trajectory_id": "9",
            "suite_sha256": "suite",
            "config_sha256": config,
            "protocol_sha256": "physics",
            "think_mode": "auto",
            "request_budget": {"run_limit": limit, "phase_limit": phase_limit},
        },
        "models": ["model-a"],
        "environment_kind": "habitat",
        "end_reason": "stop",
        "benchmark_metrics": {"sr": 1.0, "spl": 1.0, "ne": 0.5, "os": 1.0, "ndtw": None},
        "path_length": 1.0,
        "collisions": 0,
        "http_requests": 2,
    }


@pytest.mark.parametrize("budget", ["run_limit", "phase_limit", "missing"])
def test_unequal_or_unknown_request_caps_do_not_create_method_comparisons(budget):
    left, right = result(), result("startlux_hierarchical", "b")
    if budget == "missing":
        right["metadata"].pop("request_budget")
    else:
        right["metadata"]["request_budget"][budget] += 1
    summary = summarize_routes([left, right])
    assert len(summary["groups"]) == 2
    assert summary["model_comparisons"] == []


def test_comparisons_preserve_all_configs_of_same_policy_without_overwriting():
    rows = [result(config="a"), result(config="b"), result("startlux_hierarchical", "c")]
    summary = summarize_routes(rows)
    assert len(summary["groups"]) == 3
    comparisons = summary["model_comparisons"]
    assert len(comparisons) == 3
    configs = {tuple(sorted((c["left"][2], c["right"][2]))) for c in comparisons}
    assert configs == {("a", "b"), ("a", "c"), ("b", "c")}
    assert all(c["paired_episodes"] == 1 for c in comparisons)


def test_unequal_request_caps_do_not_create_ablation_comparisons():
    left, right = result(), result(condition="no_image_history", limit=63)
    summary = summarize_routes([left, right])
    assert summary["paired"] == []


def test_equal_caps_and_single_model_allow_comparison():
    summary = summarize_routes([result(), result("startlux_hierarchical", "b")])
    assert len(summary["model_comparisons"]) == 1
    assert summary["model_comparisons"][0]["request_budget"] == {
        "run_limit": 64,
        "phase_limit": 1000,
    }


def test_mixed_model_versions_are_not_compared():
    left, right = result(), result("startlux_hierarchical", "b")
    right["models"] = ["model-b"]
    assert summarize_routes([left, right])["model_comparisons"] == []


def test_mixed_budgets_within_method_are_not_silently_overwritten():
    left, repeated, right = result(), result(), result("startlux_hierarchical", "b")
    repeated = copy.deepcopy(repeated)
    repeated["metadata"]["repeat_id"] = 1
    repeated["metadata"]["request_budget"]["run_limit"] = 65
    summary = summarize_routes([left, repeated, right])
    assert summary["model_comparisons"] == []
