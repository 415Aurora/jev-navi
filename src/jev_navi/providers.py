"""Remote API boundaries. No provider has access to simulator or evaluation state."""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass

import httpx

from .action_context import (
    ACTION_DESCRIPTIONS,
    ACTION_QUESTION,
    PROMPT_VERSION,
    ROUTE_PROMPT_VERSION,
    action_context,
    encode_images,
    state_text,
)
from .types import Action, Decision, ServiceError


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


def parse_probabilities(raw):
    if not isinstance(raw, dict) or set(raw) != {x.value for x in Action}:
        raise ServiceError("invalid_response", "Expected one probability for each action")
    if any(isinstance(v, bool) for v in raw.values()):
        raise ServiceError("invalid_response", "Boolean action probability")
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
        self.poll_count = 0
        self.retry_count = 0

    def request_audit(self):
        return {
            "http_attempts": self.request_count,
            "model_submissions": self.calls,
            "poll_attempts": self.poll_count,
            "retry_attempts": self.retry_count,
        }

    def _begin_attempt(self, method, payload, inference):
        return None

    def _finish_attempt(self, event, *, response=None, error_code=None):
        pass

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
        for attempt in range(getattr(self, "attempt_limit", 2)):
            remaining = self.timeout if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceError("timeout")
            self.budget.consume()
            self.request_count += 1
            self.calls += int(inference)
            self.poll_count += int(method == "GET")
            self.retry_count += int(attempt > 0)
            event = self._begin_attempt(method, payload, inference)
            try:
                response = self.client.request(
                    method,
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._key}"},
                    timeout=min(remaining, self.timeout),
                )
            except httpx.TimeoutException:
                self._finish_attempt(event, error_code="timeout")
                if attempt == 0:
                    continue
                raise ServiceError("timeout") from None
            except httpx.TransportError:
                self._finish_attempt(event, error_code="network")
                if attempt == 0:
                    continue
                raise ServiceError("network") from None
            self._finish_attempt(event, response=response)
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
    app = "vllm-sr/d3-mini"
    expected_models = {"d3-mini", "vllm-sr/d3-mini"}

    def __init__(
        self,
        api_key,
        *,
        base_url="https://api.inference.sh",
        poll_interval=0.5,
        prompt_version=PROMPT_VERSION,
        **kwargs,
    ):
        if prompt_version not in {PROMPT_VERSION, ROUTE_PROMPT_VERSION}:
            raise ValueError("Unknown direct-action prompt version")
        if not math.isfinite(poll_interval) or poll_interval < 0:
            raise ValueError("Poll interval must be finite and nonnegative")
        super().__init__(api_key, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.prompt_version = prompt_version

    def _redact(self, value):
        # Normalize ASCII JSON escapes before redacting echoed credentials, including
        # malformed JSON that cannot be decoded as a complete object.
        value = re.sub(r"\\u00([0-7][0-9a-fA-F])", lambda m: chr(int(m[1], 16)), value)
        value = value.replace(self._key, "[REDACTED]")
        value = re.sub(r"(?i)(bearer\s+)[^\s\"},]+", r"\1[REDACTED]", value)
        return value.encode("utf-8", errors="replace").decode("utf-8")

    def _safe_audit(self, value):
        invalid_number = False

        def safe(item):
            nonlocal invalid_number
            if isinstance(item, float) and not math.isfinite(item):
                invalid_number = True
                return {"invalid_json_number": str(item)}
            if isinstance(item, str):
                return self._redact(item)
            if isinstance(item, dict):
                return {self._redact(k): safe(v) for k, v in item.items()}
            if isinstance(item, list):
                return [safe(v) for v in item]
            return item

        return safe(value), invalid_number

    def _begin_attempt(self, method, payload, inference):
        event = {
            "request_index": self.request_count,
            "stage": "submission" if inference else "poll",
            "raw": {"model": None, "response": None},
            "_started": time.monotonic(),
        }
        if payload is not None:
            event["request"] = {
                "app": payload["app"],
                "state": self._redact(payload["input"]["state"]),
                "questions": self._safe_audit(payload["input"]["choices"])[0],
                "image_count": len(payload["input"]["images"]),
                "image_uri_sha256": [
                    hashlib.sha256(x.encode()).hexdigest() for x in payload["input"]["images"]
                ],
                "payload_sha256": hashlib.sha256(
                    json.dumps(payload, sort_keys=True).encode()
                ).hexdigest(),
            }
        self.response_events.append(event)
        return event

    def _finish_attempt(self, event, *, response=None, error_code=None):
        event["latency_s"] = time.monotonic() - event.pop("_started")
        raw = event["raw"]
        if response is None:
            raw["transport_error"] = error_code
            return
        raw.update(
            http_status=response.status_code,
            body_bytes=len(response.content),
            body_sha256=hashlib.sha256(response.content).hexdigest(),
        )
        if not 200 <= response.status_code < 300:
            # Keep the existing rule: HTTP error bodies may echo credentials.
            raw["body_omitted"] = "http_error_body"
            return
        preview = self._redact(response.text)
        raw["body_truncated"] = len(preview.encode()) > 65536
        raw["body_excerpt"] = preview.encode()[:65536].decode("utf-8", errors="ignore")
        try:
            parsed = response.json()
            safe, invalid_number = self._safe_audit(parsed)
        except (ValueError, TypeError, RecursionError):
            raw["parse_error"] = "invalid_json"
            raise ServiceError("invalid_response", "Invalid provider JSON") from None
        # Use the redacted parsed representation for valid JSON; raw text may use
        # alternate escaping of a credential. The byte hash preserves body identity.
        rendered = json.dumps(safe, ensure_ascii=False, allow_nan=False).encode()
        raw["body_truncated"] = len(rendered) > 65536
        raw["body_excerpt"] = rendered[:65536].decode("utf-8", errors="ignore")
        raw["response"] = safe if not raw["body_truncated"] else {"audit_omitted": "over_64KiB"}
        data = parsed.get("data") if isinstance(parsed, dict) else None
        output = data.get("output") if isinstance(data, dict) else None
        if isinstance(output, dict) and isinstance(output.get("model"), str):
            raw["model"] = self._redact(output["model"])
        if invalid_number:
            raw["parse_error"] = "nonfinite_json_number"
            raise ServiceError("invalid_response", "Nonfinite provider JSON number")

    def decide(self, view):
        start = time.monotonic()
        deadline = start + self.timeout
        context = action_context(view, self.prompt_version)
        payload = {
            "app": self.app,
            "input": {
                "state": context["state"],
                "images": context["images"],
                "choices": [
                    {
                        "id": "action",
                        "instructions": context["question"],
                        "options": [
                            {"name": k, "description": v} for k, v in context["criteria"].items()
                        ],
                    }
                ],
            },
            "wait": True,
        }
        response = self._request(
            "POST", self.base_url + "/apps/run", payload, inference=True, deadline=deadline
        )
        submission_audit = self.response_events[-1]["raw"]["response"]
        terminal = response
        try:
            data = response["data"]
            while True:
                status = data["status"]
                if type(status) is not int or not 0 <= status <= 13:
                    raise ValueError()
                if status in (10, 11, 12, 13):
                    break
                task_id = data["id"]
                if (
                    not isinstance(task_id, str)
                    or not task_id
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
            if status != 10:
                raise ServiceError(
                    {11: "task_failed", 12: "task_cancelled", 13: "task_rejected"}[status]
                )
            output = data["output"]
            model = output["model"]
            if not isinstance(model, str) or not model:
                raise ValueError()
            if model not in self.expected_models:
                raise ServiceError("model_mismatch", "Provider returned an unexpected model")
            probabilities = parse_probabilities(output["choices"]["action"]["probabilities"])
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            raise ServiceError("invalid_response", "Invalid inference.sh task response") from None
        identity = {
            "requested_app": self.app,
            "requested_model": "d3-mini",
            "returned_model": model,
            "provider_revision": output.get("model_revision", output.get("revision")),
        }
        return Decision(
            max(Action, key=lambda x: probabilities[x.value]),
            model,
            probabilities,
            {
                "submission": submission_audit,
                "terminal": self.response_events[-1]["raw"]["response"],
                "identity": self._safe_audit(identity)[0],
            },
            time.monotonic() - start,
            {
                "input_tokens": self._safe_audit(output.get("input_tokens"))[0],
                "reported_usage": self._safe_audit(output.get("usage"))[0],
                "reported_cost": self._safe_audit(output.get("cost"))[0],
            },
        )
