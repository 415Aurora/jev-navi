"""Privileged software controls for execution and metrics; never model evidence."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from .budget import initialize_ledger
from .datasets import environment_episode, prepare_r2rce
from .experiments import (
    dependencies,
    implementation_digest,
    make_environment,
    run_suite,
    source_git,
)
from .profiles import resolve_config
from .reporting import write_report
from .runner import write_json
from .types import Action, Camera

CONTROL_INSTRUCTION = "Follow the prepared route to the endpoint. Stop."


def control_cases(actions):
    """Finite response fixtures share the actual StartLux request/response parser."""
    if (
        not isinstance(actions, list)
        or not 2 <= len(actions) <= 499
        or actions[-1] != "STOP"
        or "STOP" in actions[:-1]
        or any(action not in {member.value for member in Action} for action in actions)
    ):
        raise ValueError(
            "Route needs motion then exactly one final STOP, and room for a transition"
        )
    direct = [{"choices": {"action": action}} for action in actions]
    hierarchy = [{"choices": {"boundary_0": "split"}}]
    hierarchy += [
        {"choices": {"progress": "continue", "action": action}} for action in actions[:-1]
    ]
    hierarchy += [{"choices": {"progress": "complete", "action": "STOP"}} for _ in range(2)]
    return [
        {
            "name": "premature_stop",
            "kind": "startlux_fixture",
            "max_decisions": 1,
            "responses": [{"choices": {"action": "STOP"}}],
        },
        {
            "name": "arrival_stop",
            "kind": "startlux_fixture",
            "max_decisions": len(actions),
            "responses": direct,
        },
        {
            "name": "arrival_no_stop",
            "kind": "startlux_fixture",
            "max_decisions": len(actions) - 1,
            "responses": direct[:-1],
        },
        {
            "name": "hierarchical_stop",
            "kind": "startlux_hierarchical_fixture",
            "max_decisions": len(actions) + 1,
            "responses": hierarchy,
        },
    ]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_catalog(catalog, scenes_root):
    if catalog.get("schema_version") != 1 or catalog.get("kind") != "privileged_software_fixture":
        raise ValueError("Expected a privileged software fixture catalog")
    routes = catalog.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("Fixture catalog needs routes")
    seen = set()
    root = Path(scenes_root).resolve()
    for route in routes:
        identity = route.get("id", "")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
            raise ValueError("Route id must be a safe filename component")
        if identity in seen:
            raise ValueError("Route IDs must be unique")
        seen.add(identity)
        relative = route.get("scene_id", "")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("Invalid scene path")
        path = (root / relative).resolve()
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_relative_to(root)
        ):
            raise ValueError("Scene path escapes scenes root")
        if not path.is_file() or _sha(path) != route.get("scene_sha256"):
            raise ValueError("Fixture scene hash does not match the current asset")
        if route.get("instruction") != CONTROL_INSTRUCTION:
            raise ValueError("Control instruction must use the fixed two-stage software fixture")
        control_cases(route.get("actions"))
        gt = route.get("gt_locations")
        if not isinstance(gt, list) or len(gt) != len(route["actions"]) + 1:
            raise ValueError(
                "Dense fixture locations must align with initial pose and every action"
            )
    return routes


def _check_control(row, control, initial_distance):
    """Assert STOP semantics and dense-path identity for unobstructed oracle replays."""
    metrics = row["benchmark_metrics"]
    if initial_distance <= 3 or row["http_requests"] != 0 or row["error_code"] is not None:
        raise ValueError("Control did not start outside 3m or has request/execution errors")
    if control == "premature_stop":
        expected = metrics["sr"] == metrics["spl"] == metrics["os"] == 0
        expected &= row["end_reason"] == "stop" and metrics["ne"] > 3
    elif control == "arrival_no_stop":
        expected = metrics["sr"] == metrics["spl"] == 0 and metrics["os"] == 1
        expected &= row["end_reason"] == "decision_budget" and metrics["ne"] < 3
    else:
        expected = metrics["sr"] == metrics["os"] == 1 and 0 < metrics["spl"] <= 1
        expected &= row["end_reason"] == "stop" and metrics["ne"] < 3
    if control != "premature_stop":
        expected &= metrics["ndtw"] is not None and math.isclose(
            metrics["ndtw"], 1, rel_tol=0, abs_tol=1e-5
        )
    expected &= row["semantic_vetoes"] == (1 if control == "hierarchical_stop" else 0)
    if not expected:
        raise ValueError(f"Offline control failed: {control}: {metrics}, {row['end_reason']}")


def run_route_controls(catalog, output, *, scenes_root, backend="habitat"):
    """Run both guarded diagnostics and explicit unguarded geometric controls.

    Guarded deviations remain visible; they never replace the unguarded control
    assertions. All strategies are finite privileged fixtures and cannot call a model.
    """
    if backend not in {"habitat", "synthetic"}:
        raise ValueError("Unknown fixture environment backend")
    routes = _validate_catalog(catalog, scenes_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "fixture_catalog.json", catalog)
    ledger = Path(__file__).resolve().parents[2] / "runs/request-ledger.sqlite3"
    before, failure = None, None
    report = {
        "verified": False,
        "research_evidence": False,
        "backend": backend,
        "http_requests": 0,
        "fixture_requests": 0,
        "episodes": 0,
        "geometry_controls_passed": 0,
        "guard_deviations": 0,
        "controls": [],
        "failure": None,
        "budget_error": None,
        "phase_budget_unchanged": None,
        "git": source_git(),
        "implementation_sha256": implementation_digest(),
        "dependencies": dependencies(),
        "catalog_sha256": _sha(output / "fixture_catalog.json"),
        "note": "Privileged software responses. Geometric controls deliberately use no_guard; "
        "full retains the existing guard and separately reports its deviations. No language "
        "understanding or research navigation score is measured. Synthetic uses exact fixture "
        "DTW; native uses the pinned FastDTW mode.",
    }
    try:
        try:
            before = initialize_ledger(ledger)
            report["phase_budget_before"] = before
        except Exception as error:
            report["budget_error"] = f"{type(error).__name__}: {error}"
            raise
        for route in routes:
            folder = output / route["id"]
            folder.mkdir()
            source = folder / "episodes.json"
            gt = folder / "gt.json"
            write_json(
                source,
                {
                    "episodes": [
                        {
                            "episode_id": route["id"],
                            "trajectory_id": route["id"],
                            "scene_id": route["scene_id"],
                            "instruction": {"instruction_text": route["instruction"]},
                            "start_position": route["start_position"],
                            "start_rotation": route["start_rotation"],
                            "goals": [{"position": route["goal_position"], "radius": 3.0}],
                        }
                    ]
                },
            )
            write_json(gt, {route["id"]: {"locations": route["gt_locations"]}})
            suite = prepare_r2rce(source, scenes_root, "train", gt, source_kind="fixture")
            write_json(folder / "suite.json", suite)
            profile = resolve_config(
                {"protocol_id": "r2rce_v13_dev", "environment": {"backend": backend}}
            )
            env = make_environment(
                profile["environment"],
                environment_episode(suite["episodes"][0]),
                Camera(**profile["camera"]),
                17,
                Camera(**profile["depth_camera"]),
            )
            try:
                env.reset()
                initial_distance = env.evaluate(False)["distance_to_goal"]
                runtime = getattr(env, "runtime", {})
            finally:
                env.close()
            if not math.isfinite(initial_distance) or initial_distance <= 3:
                raise ValueError("Every control must start strictly outside the 3m success radius")
            for case in control_cases(route["actions"]):
                case_folder = folder / case["name"]
                case_folder.mkdir()
                responses = case_folder / "responses.json"
                write_json(responses, case["responses"])
                config = {
                    "protocol_id": "r2rce_v13_dev",
                    "environment": {"backend": backend},
                    "policy": {
                        "kind": case["kind"],
                        "responses": str(responses.resolve()),
                        "think_mode": "auto",
                    },
                    "run": {
                        "max_decisions": case["max_decisions"],
                        "max_steps": 500,
                        "call_limit": 7000,
                        "seed": 17,
                    },
                    "conditions": ["full", "no_guard"],
                    "execution": {
                        "mode": "offline",
                        "phase_id": "r2rce-offline-v1",
                        "ledger": str(ledger),
                    },
                    "evaluation": {
                        "ndtw_algorithm": "exact" if backend == "synthetic" else "fastdtw"
                    },
                }
                write_json(case_folder / "config.json", config)
                rows = run_suite(config, suite, case_folder / "run")
                captured = []
                for row in rows:
                    guarded_deviation = (
                        row["metadata"]["condition"] == "full" and row["blocked_actions"] > 0
                    )
                    report["episodes"] += 1
                    report["http_requests"] += row["http_requests"]
                    report["fixture_requests"] += row["fixture_requests"]
                    report["guard_deviations"] += int(guarded_deviation)
                    entry = {
                        "route": route["id"],
                        "control": case["name"],
                        "condition": row["metadata"]["condition"],
                        "initial_distance": initial_distance,
                        **row["benchmark_metrics"],
                        "decisions": row["decisions"],
                        "executed_steps": row["executed_steps"],
                        "semantic_vetoes": row["semantic_vetoes"],
                        "blocked_actions": row["blocked_actions"],
                        "end_reason": row["end_reason"],
                        "http_requests": row["http_requests"],
                        "fixture_requests": row["fixture_requests"],
                        "error_code": row["error_code"],
                        "guard_deviation": guarded_deviation,
                        "assertion_passed": None if guarded_deviation else False,
                        "native_runtime": runtime,
                    }
                    report["controls"].append(entry)
                    captured.append((row, entry))
                if len(rows) != 2:
                    raise ValueError("Both guard conditions must finish and produce a result")
                for row, entry in captured:
                    if row["http_requests"] != 0 or row["error_code"] is not None:
                        raise ValueError(
                            "Offline fixture encountered an execution or request error"
                        )
                    if not entry["guard_deviation"]:
                        try:
                            _check_control(row, case["name"], initial_distance)
                        except ValueError as error:
                            entry["assertion_error"] = str(error)
                            raise
                        entry["assertion_passed"] = True
                    if row["metadata"]["condition"] == "no_guard":
                        report["geometry_controls_passed"] += 1
        report["verified"] = report["geometry_controls_passed"] == len(routes) * 4
    except BaseException as error:
        failure = error
        report["verified"] = False
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if before is not None:
            try:
                after = initialize_ledger(ledger)
                report["phase_budget_after"] = after
                report["phase_budget_unchanged"] = before == after
            except Exception as error:
                report["budget_error"] = f"{type(error).__name__}: {error}"
                if failure is None:
                    failure = error
        if report["phase_budget_unchanged"] is not True:
            report["verified"] = False
            report["failure"] = (
                report["failure"] or report["budget_error"] or "Phase budget changed"
            )
        write_json(output / "control-summary.json", report)
        write_report(output)
        lines = [
            "# Offline route controls",
            "",
            "**NOT RESEARCH EVIDENCE**",
            "",
            report["note"],
            "",
            "| Route | Control | Guard condition | Start NE | Final NE | SR | OS | nDTW | Blocked |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in report["controls"]:
            lines.append(
                f"| {row['route']} | {row['control']} | {row['condition']} | "
                f"{row['initial_distance']:.3f} | {row['ne']:.3f} | "
                f"{row['sr']:.0f} | {row['os']:.0f} | {row['ndtw']} | {row['blocked_actions']} |"
            )
        (output / "control-report.md").write_text("\n".join(lines) + "\n")
    if failure is not None:
        raise failure
    if not report["verified"]:
        raise ValueError("Offline control verification did not pass; inspect control-summary.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--scenes-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["habitat", "synthetic"], default="habitat")
    args = parser.parse_args(argv)
    try:
        report = run_route_controls(
            json.loads(Path(args.fixtures).read_text()),
            args.output,
            scenes_root=args.scenes_root,
            backend=args.backend,
        )
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "controls"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
