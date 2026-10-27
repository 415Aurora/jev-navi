"""Offline VLN-CE metrics, using geodesic errors and Euclidean path lengths.

Semantics: https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/habitat_extensions/measures.py
NDTW drops consecutive repeated agent positions, uses dense GT locations and
normalizes DTW by len(GT) * SUCCESS_DISTANCE. The upstream FDTW mode uses fastdtw;
we never silently replace that optional dependency with exact DTW.
SR requires explicit STOP and strict distance < success_distance. Service failure
zeros SR/SPL while oracle reachability remains descriptive of the observed path.
"""

from __future__ import annotations

import math
from numbers import Real


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _positions(values, name):
    try:
        points = [tuple(point) for point in values]
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence of 3D positions") from error
    if not points:
        raise ValueError(f"{name} must be nonempty")
    for point in points:
        if len(point) != 3 or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in point
        ):
            raise ValueError(f"{name} must contain finite 3D positions")
    return points


def _exact_dtw(positions, reference):
    previous = [0.0] + [math.inf] * len(reference)
    for position in positions:
        current = [math.inf]
        for index, target in enumerate(reference, 1):
            current.append(
                math.dist(position, target) + min(previous[index], current[-1], previous[index - 1])
            )
        previous = current
    return previous[-1]


def navigation_metrics(
    positions,
    distances,
    *,
    stopped,
    success_distance=3.0,
    initial_distance=None,
    gt_locations=None,
    algorithm="fastdtw",
    service_failed=False,
) -> dict:
    """Evaluate an aligned nonempty position/geodesic-distance trace.

    Pass the actual reset geodesic distance as initial_distance for SPL.
    Missing initial distance yields null SPL for successful episodes; failures
    have SPL=0. No distance is inferred from scene coordinates or sparse paths.
    algorithm='exact' is intended for offline fixtures and golden references.
    """
    if algorithm not in {"fastdtw", "exact"}:
        raise ValueError("algorithm must be fastdtw or exact")
    if not isinstance(stopped, bool) or not isinstance(service_failed, bool):
        raise ValueError("stopped and service_failed must be booleans")
    radius = _number(success_distance, "success_distance", positive=True)
    points = _positions(positions, "positions")
    try:
        errors = [_number(value, "distances") for value in distances]
    except TypeError as error:
        raise ValueError("distances must be an aligned nonempty sequence") from error
    if len(errors) != len(points):
        raise ValueError("positions and distances must have equal nonzero lengths")
    initial = (
        _number(initial_distance, "initial_distance") if initial_distance is not None else None
    )
    reference = _positions(gt_locations, "gt_locations") if gt_locations is not None else None
    path_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    if not math.isfinite(path_length):
        raise ValueError("path length is not finite")
    success = float(stopped and errors[-1] < radius and not service_failed)
    if not success:
        spl = 0.0
    elif initial is None:
        spl = None
    else:
        denominator = max(initial, path_length)
        spl = success * initial / denominator if denominator else success
    ndtw = None
    reason = "missing_gt_locations" if reference is None else None
    if reference is not None:
        unique = []
        for point in points:
            if not unique or point != unique[-1]:
                unique.append(point)
        if algorithm == "exact":
            cost = _exact_dtw(unique, reference)
        else:
            try:
                from fastdtw import fastdtw
            except ImportError:
                reason = "fastdtw_unavailable"
                cost = None
            else:
                cost = fastdtw(unique, reference, dist=math.dist)[0]
        if cost is not None:
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("DTW cost must be finite and nonnegative")
            ndtw = math.exp(-cost / (len(reference) * radius))
    return {
        "sr": success,
        "spl": spl,
        "ne": errors[-1],
        "os": float(min(errors) < radius),
        "ndtw": ndtw,
        "ndtw_reason": reason,
        "path_length": path_length,
    }
