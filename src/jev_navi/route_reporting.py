"""R2R-CE summaries: repeats are paired, missing metrics are explicit, failures stay counted."""

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


def _episode(row):
    m = row["metadata"]
    return (m["episode_id"], m.get("repeat_id", 0))


def _fixture(row):
    m = row["metadata"]
    return (
        m.get("measurement_kind") == "fixture"
        or m.get("policy_kind") in ("scripted", "replay")
        or m.get("policy_kind", "").endswith("_fixture")
        or row["environment_kind"] != "habitat"
    )


def _base(row):
    m = row["metadata"]
    return (
        m.get("policy_kind"),
        m.get("protocol_id"),
        m.get("suite_sha256"),
        m.get("config_sha256"),
        m.get("split"),
    )


def _comparable_budget(rows):
    """Unknown, inconsistent or unequal request ceilings cannot support a comparison."""
    budgets = set()
    for row in rows:
        budget = row["metadata"].get("request_budget", {})
        values = (budget.get("run_limit"), budget.get("phase_limit"))
        if any(type(value) is not int or value <= 0 for value in values):
            return None
        budgets.add(values)
    if len(budgets) != 1:
        return None
    run_limit, phase_limit = budgets.pop()
    return {"run_limit": run_limit, "phase_limit": phase_limit}


def _failure(row):
    return row["end_reason"] == "service_error"


def _mean(values):
    return float(np.mean(values)) if values else None


def _comparison(left, right, planned):
    common = sorted(left.keys() & right.keys())
    return {
        "planned_pairs": planned,
        "paired_episodes": len(common),
        "service_failures": sum(_failure(left[k]) + _failure(right[k]) for k in common),
        "success_difference": _mean(
            [
                left[k]["benchmark_metrics"]["sr"] - right[k]["benchmark_metrics"]["sr"]
                for k in common
            ]
        ),
        "missing_from_left": [list(k) for k in sorted(right.keys() - left.keys())],
        "missing_from_right": [list(k) for k in sorted(left.keys() - right.keys())],
        "scene_families": len({left[k]["metadata"].get("scene_family") for k in common}),
        "note": "Descriptive paired results; repeats are not independent scenes.",
    }


def summarize_routes(rows, manifest=None):
    manifest = manifest or {}
    groups = defaultdict(list)
    for row in rows:
        m = row["metadata"]
        groups[(*_base(row), m.get("condition"))].append(row)
    aggregates = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        identities = [_episode(row) for row in members]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate episode/repeat result in route method and condition")
        coverage, metrics = {}, {}
        for metric in ("sr", "spl", "ne", "os", "ndtw"):
            values = [
                r["benchmark_metrics"].get(metric)
                for r in members
                if r.get("benchmark_metrics", {}).get(metric) is not None
            ]
            coverage[metric] = {"valid": len(values), "total": len(members)}
            metrics[metric] = _mean(values)
        latencies = [v for row in members for v in row.get("latency_s", [])]
        models = sorted({model for row in members for model in row.get("models", [])})
        aggregates.append(
            {
                "policy": key[0],
                "protocol_id": key[1],
                "suite_sha256": key[2],
                "config_sha256": key[3],
                "split": key[4],
                "condition": key[5],
                "models": models,
                "model_version_warning": len(models) > 1,
                "episodes": len(members),
                "success_rate": metrics["sr"],
                "metrics": metrics,
                "metric_coverage": coverage,
                "ndtw_unavailable_reasons": dict(
                    Counter(
                        r.get("benchmark_metrics", {}).get("ndtw_reason")
                        for r in members
                        if r.get("benchmark_metrics", {}).get("ndtw_reason")
                    )
                ),
                "end_reasons": dict(Counter(r["end_reason"] for r in members)),
                "service_failures": sum(_failure(r) for r in members),
                "scene_families": len({r["metadata"].get("scene_family") for r in members}),
                "trajectories": len(
                    {
                        (
                            r["metadata"].get("scene_family"),
                            r["metadata"].get("trajectory_id", r["metadata"]["episode_id"]),
                        )
                        for r in members
                    }
                ),
                "mean_path_length": _mean([r["path_length"] for r in members]),
                "collisions": sum(r["collisions"] for r in members),
                "http_requests": sum(r["http_requests"] for r in members),
                "fixture_requests": sum(r.get("fixture_requests", 0) for r in members),
                "mean_model_calls": _mean([r.get("model_calls", 0) for r in members]),
                "latency_median_s": float(np.median(latencies)) if latencies else None,
                "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
            }
        )
    eligible = [row for row in rows if not _fixture(row)]
    paired = []
    planned = len(manifest.get("suite", {}).get("episodes", []))
    by_method = defaultdict(lambda: defaultdict(dict))
    for row in eligible:
        by_method[_base(row)][row["metadata"].get("condition")][_episode(row)] = row
    for key, conditions in sorted(by_method.items(), key=lambda item: str(item[0])):
        full = conditions.get("full", {})
        for condition, other in sorted(conditions.items()):
            if condition == "full" or not full:
                continue
            models = {
                model
                for row in [*full.values(), *other.values()]
                for model in row.get("models", [])
            }
            members = [*full.values(), *other.values()]
            budget = _comparable_budget(members)
            think_modes = {r["metadata"].get("think_mode") for r in members}
            if len(models) != 1 or budget is None or len(think_modes) != 1 or None in think_modes:
                continue
            paired.append(
                {
                    "policy": key[0],
                    "protocol_id": key[1],
                    "split": key[4],
                    "comparison": f"full - {condition}",
                    "config_sha256": key[3],
                    "request_budget": budget,
                    **_comparison(full, other, planned or len(full.keys() | other.keys())),
                }
            )
    comparisons = []
    protocols = defaultdict(lambda: defaultdict(dict))
    for row in eligible:
        m = row["metadata"]
        if m.get("condition") == "full":
            protocols[(m.get("suite_sha256"), m.get("protocol_sha256"), m.get("split"))][
                (m.get("policy_kind"), m.get("think_mode"), m.get("config_sha256"))
            ][_episode(row)] = row
    for protocol, methods in sorted(protocols.items(), key=lambda item: str(item[0])):
        for left_key, right_key in combinations(sorted(methods, key=str), 2):
            if left_key[1] is None or left_key[1] != right_key[1]:
                continue  # Thinking-mode ablations are distinct experiments.
            left, right = methods[left_key], methods[right_key]
            versions = {
                model for r in [*left.values(), *right.values()] for model in r.get("models", [])
            }
            budget = _comparable_budget([*left.values(), *right.values()])
            if len(versions) != 1 or budget is None:
                continue
            comparisons.append(
                {
                    "left": list(left_key),
                    "right": list(right_key),
                    "split": protocol[2],
                    "suite_sha256": protocol[0],
                    "protocol_sha256": protocol[1],
                    "request_budget": budget,
                    **_comparison(left, right, planned or len(left.keys() | right.keys())),
                }
            )
    return {
        "groups": aggregates,
        "paired": paired,
        "model_comparisons": comparisons,
        "engineering_gate": [],
        "fixture_rows": sum(_fixture(row) for row in rows),
        "planned_jobs": manifest.get("planned_jobs", len(rows)),
        "completed_jobs": manifest.get("completed_jobs", len(rows)),
        "unrunnable_jobs": manifest.get("unrunnable_jobs", []),
        "unidentified_model_failures": sum(not r.get("models") and _failure(r) for r in rows),
        "unpaired_or_excluded_research_rows": len(rows) - len(eligible),
    }
