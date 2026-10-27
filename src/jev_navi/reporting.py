"""Descriptive diagnostics and matched-episode comparisons; fixtures are excluded."""

import csv
import html
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .runner import write_json


def _identity(row):
    m = row["metadata"]
    return (
        m.get("policy_kind", "unknown"),
        tuple(row.get("models", [])),
        m.get("suite_sha256", "unspecified"),
        m.get("config_sha256", "unspecified"),
    )


def summarize(rows, manifest=None):
    if (manifest or {}).get("suite", {}).get("schema_version") == 2 or any(
        r["metadata"].get("task_family") == "r2rce" for r in rows
    ):
        from .route_reporting import summarize_routes

        return summarize_routes(rows, manifest)
    groups = defaultdict(list)
    for row in rows:
        m = row["metadata"]
        groups[(*_identity(row), m.get("split"), m.get("condition"))].append(row)
    for members in groups.values():
        ids = [r["metadata"].get("episode_id") for r in members]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate episode results in the same model/configuration/condition")
    aggregates = []
    for key, members in sorted(groups.items(), key=lambda x: str(x[0])):
        latencies = [v for r in members for v in r["latency_s"]]
        n = len(members)
        endings = Counter(r["end_reason"] for r in members)
        stopped = [r for r in members if r["end_reason"] == "stop"]
        decisions = sum(r["decisions"] for r in members)
        aggregates.append(
            {
                "policy": key[0],
                "models": list(key[1]),
                "suite_sha256": key[2],
                "config_sha256": key[3],
                "split": key[4],
                "condition": key[5],
                "episodes": n,
                "success_rate": sum(r["success"] for r in members) / n,
                "reached_without_success_rate": sum(
                    r["ever_reached"] and not r["success"] for r in members
                )
                / n,
                "incorrect_stop_rate": sum(not r["success"] for r in stopped) / n,
                "end_reasons": dict(endings),
                "mean_path_length": float(np.mean([r["path_length"] for r in members])),
                "collisions": sum(r["collisions"] for r in members),
                "guard_interception_rate": sum(r["blocked_actions"] for r in members)
                / max(decisions, 1),
                "mean_model_calls": sum(r["model_calls"] for r in members) / n,
                "http_requests": sum(r["http_requests"] for r in members),
                "latency_median_s": float(np.median(latencies)) if latencies else None,
                "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
            }
        )
    eligible = [
        r
        for r in rows
        if r["environment_kind"] == "habitat"
        and r["metadata"].get("annotation_verified") is True
        and r["metadata"].get("policy_kind") not in ("scripted", "replay")
        and r["metadata"].get("measurement_kind") != "fixture"
        and not r["metadata"].get("policy_kind", "").endswith("_fixture")
        and len(r.get("models", [])) == 1
    ]
    paired, gates = [], []
    ids = sorted({_identity(r) for r in eligible}, key=str)
    rng = np.random.default_rng(17)
    for identity in ids:
        for split in ("dev", "holdout"):
            subset = [
                r for r in eligible if _identity(r) == identity and r["metadata"]["split"] == split
            ]
            full = {
                r["metadata"]["episode_id"]: r
                for r in subset
                if r["metadata"]["condition"] == "full"
            }
            for condition in ("shuffled_instruction", "no_image_history", "no_guard"):
                other = {
                    r["metadata"]["episode_id"]: r
                    for r in subset
                    if r["metadata"]["condition"] == condition
                }
                common = sorted(full.keys() & other.keys())
                if not common:
                    continue
                differences = np.array(
                    [int(full[i]["success"]) - int(other[i]["success"]) for i in common]
                )
                # Resample scene families, retaining paired episodes within each family.
                families = sorted({full[i]["metadata"]["scene_family"] for i in common})
                samples = []
                for _ in range(2000):
                    selected = rng.choice(families, size=len(families), replace=True)
                    values = [
                        differences[j]
                        for family in selected
                        for j, i in enumerate(common)
                        if full[i]["metadata"]["scene_family"] == family
                    ]
                    samples.append(np.mean(values))
                result = {
                    "policy": identity[0],
                    "models": list(identity[1]),
                    "split": split,
                    "comparison": f"full - {condition}",
                    "paired_episodes": len(common),
                    "scene_families": len(families),
                    "success_difference": float(differences.mean()),
                    "bootstrap_95_interval": np.percentile(samples, [2.5, 97.5]).tolist(),
                    "interval_warning": "Exploratory only; few independent scene families",
                }
                paired.append(result)
                if split == "holdout" and condition == "shuffled_instruction":
                    sr = float(np.mean([full[i]["success"] for i in common]))
                    expected = {
                        r["id"]
                        for r in (manifest or {}).get("suite", {}).get("episodes", [])
                        if r["split"] == "holdout"
                    }
                    complete = (
                        manifest is not None
                        and manifest.get("status") == "completed"
                        and manifest.get("config", {}).get("diagnostic") is True
                        and manifest.get("completed_jobs") == manifest.get("planned_jobs")
                        and len(expected) == 24
                        and set(common) == expected
                        and len(families) == 2
                        and not any(r["end_reason"] == "service_error" for r in subset)
                    )
                    gates.append(
                        {
                            "policy": identity[0],
                            "models": list(identity[1]),
                            "paired_episodes": len(common),
                            "success_rate": sr,
                            "language_difference": result["success_difference"],
                            "passed": (sr >= 0.60 and result["success_difference"] >= 0.15)
                            if complete
                            else None,
                            "status": "complete" if complete else "insufficient_or_incomplete",
                            "note": "Engineering heuristic, not publication-level evidence",
                        }
                    )
    comparisons = []
    protocol_groups = defaultdict(list)
    for row in eligible:
        m = row["metadata"]
        if m.get("condition") == "full" and m.get("protocol_sha256") and m.get("suite_sha256"):
            protocol_groups[(m["suite_sha256"], m["protocol_sha256"], m["split"])].append(row)
    for (suite_hash, protocol_hash, split), members in sorted(protocol_groups.items()):
        model_groups = defaultdict(dict)
        for row in members:
            label = (row["metadata"]["policy_kind"], row["models"][0])
            episode_id = row["metadata"]["episode_id"]
            if episode_id in model_groups[label]:
                raise ValueError("Duplicate model/episode results for a frozen protocol")
            model_groups[label][episode_id] = row
        for left_label, right_label in combinations(sorted(model_groups), 2):
            left, right = model_groups[left_label], model_groups[right_label]
            common = sorted(left.keys() & right.keys())
            if not common:
                continue
            differences = {i: int(left[i]["success"]) - int(right[i]["success"]) for i in common}
            families = sorted({left[i]["metadata"]["scene_family"] for i in common})
            samples = []
            for _ in range(2000):
                selected = rng.choice(families, len(families), replace=True)
                samples.append(
                    np.mean(
                        [
                            differences[i]
                            for family in selected
                            for i in common
                            if left[i]["metadata"]["scene_family"] == family
                        ]
                    )
                )
            comparisons.append(
                {
                    "left": list(left_label),
                    "right": list(right_label),
                    "split": split,
                    "suite_sha256": suite_hash,
                    "protocol_sha256": protocol_hash,
                    "paired_episodes": len(common),
                    "scene_families": len(families),
                    "success_difference": float(np.mean(list(differences.values()))),
                    "bootstrap_95_interval": np.percentile(samples, [2.5, 97.5]).tolist(),
                    "missing_from_left": sorted(right.keys() - left.keys()),
                    "missing_from_right": sorted(left.keys() - right.keys()),
                    "interval_warning": "Matched subset only; service failures and version changes can remove pairs",
                }
            )
    return {
        "groups": aggregates,
        "paired": paired,
        "engineering_gate": gates,
        "model_comparisons": comparisons,
        "fixture_rows": sum(
            r["environment_kind"] != "habitat"
            or r["metadata"].get("policy_kind") in ("scripted", "replay")
            or r["metadata"].get("measurement_kind") == "fixture"
            or r["metadata"].get("policy_kind", "").endswith("_fixture")
            for r in rows
        ),
        "unidentified_model_failures": sum(
            not r.get("models") and r["end_reason"] == "service_error" for r in rows
        ),
        "unpaired_or_excluded_research_rows": len(eligible)
        - sum(
            2 * x["paired_episodes"]
            for x in paired
            if x["comparison"] == "full - shuffled_instruction"
        ),
    }


def write_report(root):
    root = Path(root)
    paths = sorted(root.rglob("result.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    manifest_path = root / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    summary = summarize(rows, manifest)
    write_json(root / "summary.json", summary)
    with (root / "episodes.csv").open("w", newline="") as handle:
        fields = [
            "episode_id",
            "repeat_id",
            "protocol_id",
            "executed_steps",
            "fixture_requests",
            "sr",
            "spl",
            "ne",
            "os",
            "ndtw",
            "split",
            "condition",
            "task_type",
            "policy_kind",
            "success",
            "ever_reached",
            "end_reason",
            "path_length",
            "collisions",
            "blocked_actions",
            "decisions",
            "model_calls",
            "http_requests",
        ]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row.get("benchmark_metrics", {}), **row["metadata"]})
    lines = [
        "# Navigation diagnostic report",
        "",
        "Small samples are exploratory. Service versions and scene families must be disclosed.",
        "",
    ]
    if summary["fixture_rows"]:
        lines += [
            "**NOT RESEARCH EVIDENCE:** scripted/replay or synthetic fixture runs are present; "
            "they are excluded from paired research comparisons and engineering gates.",
            "",
        ]
    lines += [
        "| Policy / model | Split | Condition | N | SR | Mean path (m) | Collisions |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for g in summary["groups"]:
        lines.append(
            f"| {g['policy']} / {', '.join(g['models'])} | {g['split']} | {g['condition']} | "
            f"{g['episodes']} | {g['success_rate']:.3f} | {g['mean_path_length']:.3f} | {g['collisions']} |"
        )
    if (manifest or {}).get("suite", {}).get("schema_version") == 2:
        lines += [
            "",
            "## Route metrics and coverage",
            "",
            "SR requires STOP and strict distance < 3m. Missing nDTW is null, not zero.",
            "The Habitat 0.3.3 port is engineering validation, not an original 0.1.7 replication.",
            f"Planned jobs: {summary['planned_jobs']}; completed: {summary['completed_jobs']}; "
            f"unrunnable: {len(summary['unrunnable_jobs'])}.",
            "",
            "~~~json",
            json.dumps(
                [
                    {
                        "policy": g["policy"],
                        "condition": g["condition"],
                        "metrics": g["metrics"],
                        "coverage": g["metric_coverage"],
                        "unavailable": g["ndtw_unavailable_reasons"],
                    }
                    for g in summary["groups"]
                ],
                ensure_ascii=False,
                indent=2,
            ),
            "~~~",
            "",
        ]
    lines += [
        "",
        "## Paired comparisons",
        "",
        "```json",
        json.dumps(summary["paired"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Matched comparisons between models",
        "",
        "```json",
        json.dumps(summary["model_comparisons"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Failure annotation",
        "",
        "Review replay.html before assigning perception, direction, distance, stopping, history, "
        "or geometry causes. End reasons alone do not identify semantic failure causes.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    annotations = root / "failure_annotations.csv"
    if not annotations.exists():
        with annotations.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode_id", "condition", "policy", "cause", "review_notes"])
            for row in rows:
                if not row["success"]:
                    m = row["metadata"]
                    writer.writerow(
                        [m.get("episode_id"), m.get("condition"), m.get("policy_kind"), "", ""]
                    )
    cards = []
    for result_path in paths:
        episode_dir = result_path.parent
        log = episode_dir / "steps.jsonl"
        if not log.exists():
            continue
        for line in log.read_text().splitlines():
            record = json.loads(line)
            image = (episode_dir / record["image"]).relative_to(root)
            label = html.escape(
                json.dumps(
                    {
                        k: record.get(k)
                        for k in (
                            "step",
                            "proposed_action",
                            "executed_action",
                            "guard_reason",
                            "error_code",
                            "evaluation",
                        )
                    },
                    ensure_ascii=False,
                )
            )
            cards.append(
                f'<figure><img loading="lazy" src="{html.escape(image.as_posix())}" '
                f'width="256"><figcaption>{html.escape(episode_dir.name)}<pre>{label}</pre></figcaption></figure>'
            )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Navigation replay</title>'
        "<style>body{font:15px system-ui;background:#f7f8fa;color:#172b4d}main{display:flex;"
        "flex-wrap:wrap}figure{background:white;padding:12px;margin:8px;border:1px solid #ddd}"
        "pre{max-width:350px;white-space:pre-wrap}</style><h1>Navigation trajectory replay</h1>"
        "<p>Evaluator truth below is for review only; policy inputs exclude it.</p><main>"
        + "".join(cards)
        + "</main>"
    )
    (root / "replay.html").write_text(page)
    return summary
