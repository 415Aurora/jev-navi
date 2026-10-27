"""Observe/decide/check/execute with explicit semantic, execution and evaluator boundaries."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .recovery import LoopMonitor
from .types import Action, Feedback, Observation, PolicyInput, ServiceError


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class EpisodeInitializationError(RuntimeError):
    """Environment cannot start; no policy decision has been attempted."""


def run_episode(
    env,
    policy,
    guard,
    instruction,
    output,
    max_decisions=60,
    image_history=2,
    metadata=None,
    recovery=None,
    protocol_id="local_v1",
    max_steps=None,
    evaluation_spec=None,
    ndtw_algorithm="fastdtw",
):
    from .profiles import PROTOCOLS

    cap = 60 if protocol_id == "local_v1" else 500
    if (
        protocol_id not in PROTOCOLS
        or type(max_decisions) is not int
        or not 1 <= max_decisions <= cap
        or type(image_history) is not int
        or not 0 <= image_history <= 2
    ):
        raise ValueError("Invalid decision budget or image history")
    if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= cap):
        raise ValueError("Invalid execution-step budget")
    monitor = LoopMonitor(**(recovery or {}))
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    try:
        observation = env.reset()
        initial_evaluation = env.evaluate(False)
    except (ValueError, RuntimeError) as error:
        raise EpisodeInitializationError(str(error)) from error
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    (output / "depth").mkdir()
    start = time.monotonic()
    initial_calls, initial_requests = policy.calls, policy.request_count
    is_fixture = bool(getattr(policy, "is_fixture", False))
    frames, history, latencies, models = [], [], [], set()
    blocked_count = consecutive_blocks = decisions = executed_steps = semantic_vetoes = 0
    end_reason, stopped, error_code = "decision_budget", False, None
    # Evaluation stays local to this function and never becomes policy state.
    trace = [initial_evaluation]
    manifest = {
        "instruction": instruction,
        "max_decisions": max_decisions,
        "max_steps": max_steps,
        "protocol_id": protocol_id,
        "image_history": image_history,
        "environment_kind": env.kind,
        "environment_runtime": getattr(env, "runtime", {}),
        "guard": {
            "camera": asdict(guard.camera),
            "enabled": guard.enabled,
            "radius_m": guard.radius,
            "forward_m": guard.forward,
            "margin_m": guard.margin,
            "agent_height_m": guard.agent_height,
            "depth_pitch_deg": guard.depth_pitch_deg,
        },
        "recovery": recovery or {"mode": "off"},
        "metadata": metadata or {},
        "measurement_kind": "fixture" if is_fixture else (metadata or {}).get("measurement_kind"),
    }
    write_json(output / "manifest.json", manifest)
    with (output / "steps.jsonl").open("w") as log:
        for step in range(max_decisions):
            if trace[-1].get("environment_done"):
                end_reason = "environment_budget"
                break
            if max_steps is not None and executed_steps >= max_steps:
                end_reason = "step_budget"
                break
            decisions += 1
            image_path, depth_path = f"frames/{step:04d}.png", f"depth/{step:04d}.npy"
            Image.fromarray(observation.rgb).save(output / image_path)
            depth = env.depth()
            np.save(output / depth_path, depth)
            frames.append(Observation(observation.rgb.copy(), step))
            frames = frames[-(image_history + 1) :]
            memory = getattr(policy, "memory_snapshot", lambda: {})()
            view = PolicyInput(
                instruction, tuple(frames), tuple(history[-4:]), monitor.context, memory=memory
            )
            record = {
                "step": step,
                "image": image_path,
                "depth": depth_path,
                "input_frames": [f.step for f in frames],
                "history": [asdict(x) for x in history[-4:]],
                "recovery_context": monitor.context,
                "input_memory": memory,
            }
            response_cursor = len(getattr(policy, "response_events", []))
            decision_start = time.monotonic()
            try:
                decision = policy.decide(view)
                duration = time.monotonic() - decision_start
                models.add(decision.model)
                latencies.append(duration)
                executed, collided, loop_event = None, False, None
                if not decision.execute:
                    allowed, reason = False, decision.control_reason
                    semantic_vetoes += 1
                else:
                    safety = guard.check(decision.action, depth)
                    allowed, reason = safety.allowed, safety.reason
                    if allowed:
                        observation, feedback = env.execute(decision.action)
                        executed, collided = decision.action, feedback.get("collided", False)
                        executed_steps += 1
                        consecutive_blocks = 0
                    else:
                        blocked_count += 1
                        consecutive_blocks += 1
                history.append(Feedback(decision.action, executed, reason, collided))
                if decision.execute:
                    loop_event = monitor.update(history[-1])
                record.update(
                    loop_event=loop_event,
                    failed_forwards=monitor.failed_forwards,
                    recovery_attempts=monitor.recovery_attempts,
                    proposed_action=decision.action.value,
                    executed_action=executed.value if executed else None,
                    guard_allowed=allowed if decision.execute else None,
                    guard_reason=reason,
                    collided=collided,
                    probabilities=decision.probabilities,
                    model=decision.model,
                    raw=decision.raw,
                    usage=decision.usage,
                    latency_s=duration,
                    semantic_state=decision.semantic_state,
                    executed_steps=executed_steps,
                )
                if executed == Action.STOP:
                    stopped, end_reason = True, "stop"
                elif consecutive_blocks >= 3:
                    end_reason = "stuck"
                elif loop_event == "terminate":
                    end_reason = "loop_detected"
                elif max_steps is not None and executed_steps >= max_steps:
                    end_reason = "step_budget"
            except ServiceError as error:
                duration = time.monotonic() - decision_start
                latencies.append(duration)
                error_code = error.code
                end_reason = "call_budget" if error.code == "call_budget" else "service_error"
                record.update(
                    proposed_action=None,
                    executed_action=None,
                    error_code=error.code,
                    latency_s=duration,
                )
            events = getattr(policy, "response_events", [])[response_cursor:]
            if events:
                record["response_events"] = events
                models.update(
                    event["raw"]["model"]
                    for event in events
                    if isinstance(event["raw"].get("model"), str) and event["raw"]["model"]
                )
            evaluation = env.evaluate(stopped)
            trace.append(evaluation)
            record["evaluation"] = evaluation
            record["model_calls"] = policy.calls - initial_calls
            attempts = policy.request_count - initial_requests
            record["http_requests"] = 0 if is_fixture else attempts
            record["fixture_requests"] = attempts if is_fixture else 0
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            if stopped or end_reason != "decision_budget":
                break
            if evaluation.get("environment_done"):
                end_reason = "environment_budget"
                break
    attempts = policy.request_count - initial_requests
    result = {
        **env.evaluate(stopped),
        "end_reason": end_reason,
        "error_code": error_code,
        "decisions": decisions,
        "executed_steps": executed_steps,
        "blocked_actions": blocked_count,
        "semantic_vetoes": semantic_vetoes,
        "recovery_attempts": monitor.recovery_attempts,
        "model_calls": policy.calls - initial_calls,
        "http_requests": 0 if is_fixture else attempts,
        "fixture_requests": attempts if is_fixture else 0,
        "latency_s": latencies,
        "models": sorted(models),
        "elapsed_s": time.monotonic() - start,
        "environment_kind": env.kind,
        "metadata": metadata or {},
    }
    if protocol_id != "local_v1":
        from .metrics import navigation_metrics

        private = evaluation_spec or {}
        metrics = navigation_metrics(
            [x["position"] for x in trace],
            [x["distance_to_goal"] for x in trace],
            stopped=stopped,
            success_distance=private.get("success_radius", 3.0),
            initial_distance=trace[0]["distance_to_goal"],
            gt_locations=private.get("gt_locations"),
            algorithm=ndtw_algorithm,
            service_failed=error_code is not None,
        )
        result["benchmark_metrics"] = metrics
        result["success"] = bool(metrics["sr"])
        result["ever_reached"] = bool(metrics["os"])
    write_json(output / "result.json", result)
    return result
