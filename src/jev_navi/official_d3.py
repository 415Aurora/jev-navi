"""Official synchronous System One boundary; never calls inference.sh."""

import hashlib
import json
import time

from .action_context import ROUTE_PROMPT_VERSION, action_context
from .providers import D3Policy, parse_probabilities
from .types import Action, Decision, ServiceError


class OfficialD3Policy(D3Policy):
    # No automatic retries: callers must not hammer rate-limited free endpoints.
    attempt_limit = 1

    def __init__(self, api_key, **kwargs):
        super().__init__(
            api_key, prompt_version=kwargs.pop("prompt_version", ROUTE_PROMPT_VERSION), **kwargs
        )

    def _begin_attempt(self, method, payload, inference):
        # Reuse bounded/redacted audit rendering; hash the actual wire payload.
        audit_payload = {
            "app": payload["model"],
            "input": {
                "state": payload["state"],
                "images": payload["images"],
                "choices": payload["questions"],
            },
        }
        event = super()._begin_attempt(method, audit_payload, inference)
        event["request"]["payload_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        event["request"]["provider"] = "official_systemone"
        return event

    def _finish_attempt(self, event, *, response=None, error_code=None):
        super()._finish_attempt(event, response=response, error_code=error_code)
        if response is not None and 200 <= response.status_code < 300:
            parsed = response.json()
            if isinstance(parsed, dict) and isinstance(parsed.get("model"), str):
                event["raw"]["model"] = self._redact(parsed["model"])

    def decide(self, view):
        start = time.monotonic()
        context = action_context(view, self.prompt_version)
        payload = {
            "model": self.app,
            "state": context["state"],
            "images": context["images"],
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": context["question"],
                    "criteria": context["criteria"],
                }
            },
        }
        raw = self._request(
            "POST",
            "https://decision.vllm-sr.ai/v1/systemone",
            payload,
            inference=True,
            deadline=start + self.timeout,
        )
        try:
            model = raw["model"]
            if model not in self.expected_models:
                raise ServiceError("model_mismatch", "Official API returned an unexpected model")
            if set(raw["answers"]) != {"action"} or raw["answers"]["action"]["type"] != "choice":
                raise ValueError()
            probabilities = parse_probabilities(raw["answers"]["action"]["probabilities"])
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            raise ServiceError("invalid_response", "Invalid official System One response") from None
        identity = {
            "requested_model": self.app,
            "returned_model": model,
            "provider": "official_systemone",
            "provider_revision": raw.get("model_revision", raw.get("revision")),
        }
        return Decision(
            max(Action, key=lambda x: probabilities[x.value]),
            model,
            probabilities,
            {
                "terminal": self.response_events[-1]["raw"]["response"],
                "identity": self._safe_audit(identity)[0],
            },
            time.monotonic() - start,
            {
                "reported_usage": self._safe_audit(raw.get("usage"))[0],
                "reported_cost": self._safe_audit(raw.get("cost"))[0],
            },
        )
