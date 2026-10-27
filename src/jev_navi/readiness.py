"""File-level R2R-CE readiness; no environment, policy, ledger or network access."""

import hashlib
from pathlib import Path

from .datasets import validate_r2rce_suite


def _integrity(path, expected):
    check = {
        "path": path,
        "expected_sha256": expected,
        "actual_sha256": None,
        "verified": False,
    }
    if not isinstance(path, str) or not path:
        check["reason"] = "source_path_missing"
        return check
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        check["actual_sha256"] = digest.hexdigest()
        check["verified"] = check["actual_sha256"] == expected
        if not check["verified"]:
            check["reason"] = "sha256_mismatch"
    except OSError as error:
        check["reason"] = type(error).__name__
    return check


def _scene_available(path):
    try:
        with Path(path).open("rb") as handle:
            return bool(handle.read(1))
    except OSError:
        return False


def check_r2rce(suite, *, require_ndtw=False):
    """Check current files rather than trusting a cached missing-assets list.

    Source kind is a user declaration; matching local hashes does not prove
    upstream authenticity or a data license. Asset file checks do not prove
    navigability, scene format compatibility, or model-call authorization.
    """
    if type(require_ndtw) is not bool:
        raise ValueError("require_ndtw must be a boolean")
    validate_r2rce_suite(suite)
    rows, source = suite["episodes"], suite["source"]
    scenes = sorted({row["scene"] for row in rows})
    missing = [path for path in scenes if not _scene_available(path)]
    unevaluable = [[r["id"], r["repeat_id"]] for r in rows if not r["evaluable"]]
    missing_gt = [
        [r["id"], r["repeat_id"]] for r in rows if r["evaluation"]["gt_locations"] is None
    ]
    integrity = {"episodes": _integrity(source.get("path"), source.get("sha256"))}
    if source.get("gt_path") is not None:
        integrity["gt"] = _integrity(source["gt_path"], source.get("gt_sha256"))
    ready = bool(
        rows
        and not missing
        and not unevaluable
        and all(check["verified"] for check in integrity.values())
        and (not require_ndtw or not missing_gt)
    )
    return {
        "ready": ready,
        "source_kind": source["kind"],
        "declared_official_data_ready": ready and source["kind"] == "official",
        "episodes": len(rows),
        "scenes": len(scenes),
        "missing_scenes": missing,
        "unevaluable_episodes": unevaluable,
        "ndtw_required": require_ndtw,
        "ndtw_coverage": {"available": len(rows) - len(missing_gt), "total": len(rows)},
        "missing_gt_episodes": missing_gt,
        "source_integrity": integrity,
        "geometry_verified": False,
        "live_authorized": False,
        "http_requests": 0,
        "note": "File-level checks only. Source kind is a declaration, not proof of upstream "
        "authenticity or a license. Native reset/navigation checks and a separately approved "
        "persistent phase budget are still required for live experiments.",
    }
