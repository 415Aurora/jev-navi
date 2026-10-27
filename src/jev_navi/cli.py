"""Explicit command entry points for fixtures, simulation, experiments and reports."""

import argparse
import getpass
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .credentials import load_local_env, save_bamboo_credentials
from .experiments import dependencies, run_suite, validate_suite
from .reporting import write_report
from .types import ServiceError


def _load(path):
    return yaml.safe_load(Path(path).read_text())


def doctor(habitat_python):
    native = {"available": False, "python": habitat_python}
    if Path(habitat_python).exists():
        try:
            check = subprocess.run(
                [
                    habitat_python,
                    "-c",
                    "import json,sys,habitat,habitat_sim; print(json.dumps({'python':sys.version,"
                    "'habitat_sim':habitat_sim.__version__}))",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            native.update(
                available=check.returncode == 0,
                output=check.stdout.strip(),
                error=check.stderr[-2000:] if check.returncode else None,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            native["error"] = type(error).__name__
    return {
        "dependencies": dependencies(),
        "habitat_worker": native,
        "direct_habitat_import": importlib.util.find_spec("habitat") is not None,
        "free_disk_gb": round(shutil.disk_usage(Path.cwd()).free / 1e9, 2),
        "credentials_configured": {
            name: bool(os.environ.get(name))
            for name in ("BAMBOO_API_KEY", "STARTLUX_ENDPOINT", "INFERENCE_API_KEY")
        },
        "note": "No credential values are printed and no model calls are made.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jev-navi")
    commands = parser.add_subparsers(dest="command", required=True)
    project = Path(__file__).resolve().parents[2]
    load_local_env(project / ".env")
    commands.add_parser("configure-bamboo", help="Save a hidden API key to project-local .env")
    worker_python = str(project / ".habitat" / "bin" / "python")
    p = commands.add_parser("doctor", help="Check dependencies without contacting a model")
    p.add_argument("--habitat-python", default=worker_python)
    p = commands.add_parser("smoke", help="Scripted software/simulator check, never model evidence")
    p.add_argument("--backend", choices=["synthetic", "habitat"], default="synthetic")
    p.add_argument("--output", required=True)
    p.add_argument("--scene")
    p.add_argument("--start", nargs=3, type=float)
    p.add_argument("--goal", nargs=3, type=float)
    p.add_argument("--habitat-python", default=worker_python)
    p = commands.add_parser("run", help="Execute a frozen episode manifest")
    p.add_argument("--config", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("report", help="Regenerate reports and trajectory replay")
    p.add_argument("run_directory")
    p = commands.add_parser("validate-suite", help="Validate annotations and scene splits")
    p.add_argument("suite")
    p.add_argument("--diagnostic", action="store_true")
    p = commands.add_parser("contract", help="At most 24 static model input checks")
    p.add_argument("--config", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser(
        "prepare-r2rce", help="Prepare official-format tasks without model calls"
    )
    p.add_argument("--episodes", required=True)
    p.add_argument("--scenes-root", required=True)
    p.add_argument("--split", required=True, choices=["train", "val_seen", "val_unseen", "test"])
    p.add_argument("--gt")
    p.add_argument("--source-kind", choices=["official", "fixture"], default="official")
    p.add_argument("--output", required=True)
    commands.add_parser("budget-status", help="Show shared phase accounting, no model calls")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-r2rce":
            from .datasets import prepare_r2rce
            from .runner import write_json

            output = Path(args.output)
            if output.exists():
                raise ValueError("Output already exists")
            suite = prepare_r2rce(
                args.episodes, args.scenes_root, args.split, args.gt, source_kind=args.source_kind
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, suite)
            result = {
                "prepared": str(output),
                "episodes": len(suite["episodes"]),
                "missing_scenes": suite["missing_scenes"],
                "http_requests": 0,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2 if suite["missing_scenes"] else 0
        elif args.command == "budget-status":
            from .budget import initialize_ledger

            result = initialize_ledger(project / "runs/request-ledger.sqlite3")
        elif args.command == "configure-bamboo":
            key = os.environ.get("BAMBOO_API_KEY") or getpass.getpass("Bamboo API key: ")
            save_bamboo_credentials(project / ".env", key)
            result = {"saved": True, "path": str(project / ".env"), "mode": "0600"}
        elif args.command == "doctor":
            result = doctor(args.habitat_python)
        elif args.command == "validate-suite":
            suite = _load(args.suite)
            validate_suite(suite, args.diagnostic)
            result = {"valid": True, "episodes": len(suite["episodes"])}
        elif args.command == "report":
            result = write_report(args.run_directory)
        elif args.command == "contract":
            from .contracts import run_contract

            result = run_contract(_load(args.config), args.images, args.instruction, args.output)
        elif args.command == "run":
            config = _load(args.config)
            result = {"results": run_suite(config, _load(args.suite), args.output)}
        else:
            if args.backend == "habitat" and (not args.scene or not args.start or not args.goal):
                parser.error("Habitat smoke requires --scene, --start x y z and --goal x y z")
            episode = {
                "id": "smoke",
                "scene_id": "smoke",
                "scene_family": "smoke",
                "split": "dev",
                "task_type": "visible",
                "instruction": "software smoke test",
                "scene": args.scene or "synthetic",
                "start_position": args.start or [0, 0, 0],
                "start_rotation": [0, 0, 0, 1],
                "goal_position": args.goal or [0, 0, -0.5],
                "success_radius": 0.2,
                "annotation_verified": False,
            }
            config = {
                "environment": {"backend": args.backend, "python": args.habitat_python},
                "policy": {"kind": "scripted", "actions": ["MOVE_FORWARD", "MOVE_FORWARD", "STOP"]},
                "conditions": ["full"],
                "run": {"max_decisions": 3, "call_limit": 500, "seed": 17},
            }
            result = run_suite(config, {"schema_version": 1, "episodes": [episode]}, args.output)[0]
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ServiceError) as error:
        code = getattr(error, "code", type(error).__name__)
        print(
            json.dumps({"error": code, "message": str(error)}, ensure_ascii=False), file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
