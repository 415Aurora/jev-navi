"""One-action observe/decide/check/execute loop with an explicit truth boundary."""

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
):
    if (
        type(max_decisions) is not int
        or not 1 <= max_decisions <= 60
        or type(image_history) is not int
        or not 0 <= image_history <= 2
    ):
        raise ValueError("Invalid decision budget or image history")
    monitor = LoopMonitor(**(recovery or {}))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    (output / "depth").mkdir()
    start = time.monotonic()
    initial_calls = policy.calls
    initial_requests = policy.request_count
    frames, history, latencies, models = [], [], [], set()
    blocked_count = consecutive_blocks = decisions = 0
    end_reason, stopped, error_code = "decision_budget", False, None
    observation = env.reset()
    manifest = {
        "instruction": instruction,
        "max_decisions": max_decisions,
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
    }
    write_json(output / "manifest.json", manifest)
    with (output / "steps.jsonl").open("w") as log:
        for step in range(max_decisions):
            decisions += 1
            image_path = f"frames/{step:04d}.png"
            depth_path = f"depth/{step:04d}.npy"
            Image.fromarray(observation.rgb).save(output / image_path)
            depth = env.depth()
            np.save(output / depth_path, depth)
            # Retain only the bounded RGB history. No environment or evaluator references.
            frames.append(Observation(observation.rgb.copy(), step))
            frames = frames[-(image_history + 1) :]
            view = PolicyInput(instruction, tuple(frames), tuple(history[-4:]), monitor.context)
            record = {
                "step": step,
                "image": image_path,
                "depth": depth_path,
                "input_frames": [f.step for f in frames],
                "history": [asdict(x) for x in history[-4:]],
                "recovery_context": monitor.context,
            }
            decision_start = time.monotonic()
            try:
                decision = policy.decide(view)
                decision_duration = time.monotonic() - decision_start
                models.add(decision.model)
                latencies.append(decision_duration)
                safety = guard.check(decision.action, depth)
                executed, collided = None, False
                if safety.allowed:
                    observation, feedback = env.execute(decision.action)
                    executed = decision.action
                    collided = feedback.get("collided", False)
                    consecutive_blocks = 0
                else:
                    blocked_count += 1
                    consecutive_blocks += 1
                history.append(Feedback(decision.action, executed, safety.reason, collided))
                loop_event = monitor.update(history[-1])
                record.update(
                    loop_event=loop_event,
                    failed_forwards=monitor.failed_forwards,
                    recovery_attempts=monitor.recovery_attempts,
                    proposed_action=decision.action.value,
                    executed_action=executed.value if executed else None,
                    guard_allowed=safety.allowed,
                    guard_reason=safety.reason,
                    collided=collided,
                    probabilities=decision.probabilities,
                    model=decision.model,
                    raw=decision.raw,
                    usage=decision.usage,
                    latency_s=decision_duration,
                )
                if executed == Action.STOP:
                    stopped, end_reason = True, "stop"
                elif consecutive_blocks >= 3:
                    end_reason = "stuck"
                elif loop_event == "terminate":
                    end_reason = "loop_detected"
            except ServiceError as error:
                decision_duration = time.monotonic() - decision_start
                latencies.append(decision_duration)
                error_code = error.code
                end_reason = "call_budget" if error.code == "call_budget" else "service_error"
                record.update(
                    proposed_action=None,
                    executed_action=None,
                    error_code=error.code,
                    latency_s=decision_duration,
                )
            # Truth is recorded only AFTER policy invocation and is never added to view.
            record["evaluation"] = env.evaluate(stopped)
            record["model_calls"] = policy.calls - initial_calls
            record["http_requests"] = policy.request_count - initial_requests
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            if stopped or end_reason != "decision_budget":
                break
    result = {
        **env.evaluate(stopped),
        "end_reason": end_reason,
        "error_code": error_code,
        "decisions": decisions,
        "blocked_actions": blocked_count,
        "recovery_attempts": monitor.recovery_attempts,
        "model_calls": policy.calls - initial_calls,
        "http_requests": policy.request_count - initial_requests,
        "latency_s": latencies,
        "models": sorted(models),
        "elapsed_s": time.monotonic() - start,
        "environment_kind": env.kind,
        "metadata": metadata or {},
    }
    write_json(output / "result.json", result)
    return result
