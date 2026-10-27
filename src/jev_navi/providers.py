"""Remote API boundaries. No provider has access to simulator or evaluation state."""

import base64
import io
import json
import math
import time
from dataclasses import dataclass

import httpx
from PIL import Image

from .types import Action, Decision, ServiceError

PROMPT_VERSION = "local-navigation-v3-recovery"
ACTION_DESCRIPTIONS = {
    "MOVE_FORWARD": "Move forward 0.25 metres without changing heading.",
    "TURN_LEFT": "Rotate left in place by 15 degrees.",
    "TURN_RIGHT": "Rotate right in place by 15 degrees.",
    "STOP": "End this task now, only if the requested local destination has been reached.",
}
ACTION_QUESTION = (
    "Choose the next single action to achieve the local instruction, using the current image "
    "and any earlier images and executed-action feedback. Images are ordered oldest to newest; "
    "the final image is current. A rejected action did not move the robot. Do not assume an "
    "action succeeded. Stop only when the visual destination and instruction are satisfied."
)


@dataclass
class CallBudget:
    """Conservative global HTTP-attempt cap, including retries and task polls."""

    limit: int = 500
    used: int = 0

    def __post_init__(self):
        if self.limit < 1:
            raise ValueError("Call budget must be positive")

    def consume(self):
        if self.used >= self.limit:
            raise ServiceError("call_budget")
        self.used += 1


def encode_images(view):
    images = []
    for frame in view.frames:
        buffer = io.BytesIO()
        Image.fromarray(frame.rgb).save(buffer, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return images


def state_text(view, *, image_markers=False):
    history = [
        {
            "proposed": x.proposed.value,
            "executed": x.executed.value if x.executed else None,
            "feedback": x.reason,
            "collided": x.collided,
        }
        for x in view.history
    ]
    frames = "\n".join(
        f"Image {i + 1}: step={f.step}" + (" <image>" if image_markers else "")
        for i, f in enumerate(view.frames)
    )
    return (
        f"Local instruction: {view.instruction}\n{frames}\n"
        f"Executed-action history: {json.dumps(history, ensure_ascii=False)}"
        + (f"\nRecovery feedback: {view.recovery_context}" if view.recovery_context else "")
    )


def parse_probabilities(raw):
    if not isinstance(raw, dict) or set(raw) != {x.value for x in Action}:
        raise ServiceError("invalid_response", "Expected one probability for each action")
    try:
        probabilities = {k: float(v) for k, v in raw.items()}
    except (TypeError, ValueError):
        raise ServiceError("invalid_response", "Invalid probability values") from None
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values()):
        raise ServiceError("invalid_response", "Invalid probability range")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.02):
        raise ServiceError("invalid_response", "Action probabilities must sum to one")
    return probabilities


class RemotePolicy:
    def __init__(self, api_key, *, client=None, budget=None, timeout=30.0):
        if not api_key:
            raise ServiceError("authentication", "API key is not configured")
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self._key = api_key
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)
        self._owns_client = client is None
        self.is_fixture = isinstance(getattr(self.client, "_transport", None), httpx.MockTransport)
        self.budget = budget if budget is not None else CallBudget()
        self.timeout = timeout
        self.calls = 0
        self.request_count = 0
        self.response_events = []

    def close(self):
        if self._owns_client:
            self.client.close()

    def _request(self, method, url, payload=None, *, inference=False, deadline=None):
        # A failed inference POST may already have been accepted by a provider. We count
        # every submission, and never claim exactly-once billing across transport retries.
        if not self.is_fixture and not getattr(self.budget, "persistent", False):
            raise ServiceError(
                "offline_mode", "Real HTTP requires an explicitly enabled persistent phase budget"
            )
        for attempt in range(2):
            remaining = self.timeout if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceError("timeout")
            self.budget.consume()
            self.request_count += 1
            self.calls += int(inference)
            try:
                response = self.client.request(
                    method,
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._key}"},
                    timeout=min(remaining, self.timeout),
                )
            except httpx.TimeoutException:
                if attempt == 0:
                    continue
                raise ServiceError("timeout") from None
            except httpx.TransportError:
                if attempt == 0:
                    continue
                raise ServiceError("network") from None
            status = response.status_code
            retryable = status == 429 or status >= 500
            if retryable and attempt == 0:
                continue
            if not 200 <= status < 300:
                code = {
                    401: "authentication",
                    403: "authentication",
                    402: "quota",
                    422: "unsupported_input",
                    429: "rate_limit",
                }.get(status, "http_error")
                # Do not log response error text: it can echo request headers or keys.
                raise ServiceError(code, f"Provider HTTP status {status}")
            try:
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError()
                return data
            except (ValueError, TypeError):
                raise ServiceError("invalid_response", "Expected JSON object") from None
        raise ServiceError("network")


class StartLuxPolicy(RemotePolicy):
    def __init__(
        self,
        api_key,
        *,
        endpoint="http://127.0.0.1:8090/v1/systemone",
        think_mode="never",
        **kwargs,
    ):
        if think_mode not in ("never", "auto", "always"):
            raise ValueError("Invalid think_mode")
        super().__init__(api_key, **kwargs)
        self.endpoint = endpoint
        self.think_mode = think_mode

    def ask(self, view, questions, *, extra_state=""):
        if not 1 <= len(view.frames) <= 8 or not 1 <= len(questions) <= 32:
            raise ValueError("Unsupported image or question count")
        for name, question in questions.items():
            criteria = question.get("criteria")
            if (
                not isinstance(name, str)
                or not isinstance(criteria, dict)
                or not criteria
                or question.get("type") != "choice"
                or any(
                    not isinstance(k, str) or not k or not isinstance(v, str)
                    for k, v in criteria.items()
                )
            ):
                raise ValueError("Expected named finite choice questions")
        payload = {
            "model": "startlux-decision-27b-autothink",
            "state": state_text(view, image_markers=True)
            + ("\n" + extra_state if extra_state else ""),
            "images": encode_images(view),
            "questions": questions,
            "think_mode": self.think_mode,
            "allow_none": False,
            "none_check": True,
        }
        raw = self._request("POST", self.endpoint, payload, inference=True)
        # Persist planning responses and model identity even if a later request fails.
        self.response_events.append(
            {"request_index": self.request_count, "questions": list(questions), "raw": raw}
        )
        try:
            if not isinstance(raw["model"], str) or not raw["model"]:
                raise ValueError()
            if set(raw["answers"]) != set(questions):
                raise ValueError()
            selections, distributions = {}, {}
            for name, question in questions.items():
                options = question["criteria"]
                values = raw["answers"][name]["probabilities"]
                if not isinstance(values, dict) or set(values) != set(options):
                    raise ValueError()
                values = {k: float(v) for k, v in values.items()}
                if any(
                    not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()
                ) or not math.isclose(sum(values.values()), 1.0, abs_tol=0.02):
                    raise ValueError()
                distributions[name] = values
                selections[name] = max(options, key=values.__getitem__)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ServiceError("invalid_response") from None
        return raw, selections, distributions

    def decide(self, view):
        start = time.monotonic()
        raw, selected, distributions = self.ask(
            view,
            {
                "action": {
                    "type": "choice",
                    "instructions": ACTION_QUESTION,
                    "criteria": ACTION_DESCRIPTIONS,
                }
            },
        )
        return Decision(
            Action(selected["action"]),
            raw["model"],
            distributions["action"],
            raw,
            time.monotonic() - start,
            raw.get("usage", {}),
        )


class D3Policy(RemotePolicy):
    def __init__(
        self, api_key, *, base_url="https://api.inference.sh", poll_interval=0.5, **kwargs
    ):
        super().__init__(api_key, **kwargs)
        if poll_interval < 0:
            raise ValueError("Poll interval cannot be negative")
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval

    def decide(self, view):
        start = time.monotonic()
        deadline = start + self.timeout
        payload = {
            "app": "vllm-sr/d3-mini",
            "input": {
                "state": state_text(view),
                "images": encode_images(view),
                "choices": [
                    {
                        "id": "action",
                        "instructions": ACTION_QUESTION,
                        "options": [
                            {"name": k, "description": v} for k, v in ACTION_DESCRIPTIONS.items()
                        ],
                    }
                ],
            },
            "wait": True,
        }
        response = self._request(
            "POST", self.base_url + "/apps/run", payload, inference=True, deadline=deadline
        )
        terminal = response
        try:
            data = response["data"]
            while data["status"] not in (10, 11, 12, 13):
                task_id = data["id"]
                if (
                    not isinstance(task_id, str)
                    or not task_id.replace("-", "").replace("_", "").isalnum()
                ):
                    raise ValueError()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ServiceError("timeout")
                time.sleep(min(self.poll_interval, remaining))
                terminal = self._request(
                    "GET", self.base_url + "/tasks/" + task_id, deadline=deadline
                )
                data = terminal["data"]
            if data["status"] != 10:
                raise ServiceError(
                    {11: "task_failed", 12: "task_cancelled", 13: "task_rejected"}[data["status"]]
                )
            output = data["output"]
            probabilities = parse_probabilities(output["choices"]["action"]["probabilities"])
            model = output["model"]
            if not isinstance(model, str) or not model:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            raise ServiceError("invalid_response", "Invalid inference.sh task response") from None
        action = max(Action, key=lambda x: probabilities[x.value])
        return Decision(
            action,
            model,
            probabilities,
            {"submission": response, "terminal": terminal},
            time.monotonic() - start,
            {"input_tokens": output.get("input_tokens")},
        )
