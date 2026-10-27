"""StartLux-only finite semantic decisions, with identical wire parsing for offline fixtures."""

import json
import re
import time
from pathlib import Path

import httpx

from .providers import ACTION_DESCRIPTIONS, ACTION_QUESTION, CallBudget, StartLuxPolicy
from .types import Action, Decision, ServiceError

PROMPT_VERSION = "r2rce-hierarchy-v1"
MAX_BOUNDARIES = 32  # Engineering limit, not a claimed provider limit.


def candidate_boundaries(instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be nonempty")
    # Candidate generation does not decide whether a boundary is semantically appropriate.
    offsets = {m.end() for m in re.finditer(r"[.!?;](?:\s+|$)", instruction)}
    offsets.update(
        m.start()
        for m in re.finditer(r"\b(?:and then|then|after that|next)\b", instruction, re.IGNORECASE)
    )
    offsets = sorted(
        x
        for x in offsets
        if 0 < x < len(instruction) and instruction[:x].strip() and instruction[x:].strip()
    )
    if len(offsets) > MAX_BOUNDARIES:
        raise ValueError("Instruction exceeds the supported boundary count; no truncation applied")
    return tuple(offsets)


def source_segments(instruction, boundaries):
    points = (0, *boundaries, len(instruction))
    if any(type(x) is not int for x in points) or any(a >= b for a, b in zip(points, points[1:])):
        raise ValueError("Invalid source boundary")
    # Whitespace is retained, so joining spans reconstructs the original instruction exactly.
    if any(not instruction[a:b].strip() for a, b in zip(points, points[1:])):
        raise ValueError("Source boundary creates an empty subgoal")
    return tuple(zip(points, points[1:]))


class HierarchicalPolicy(StartLuxPolicy):
    prompt_version = PROMPT_VERSION

    def __init__(self, *args, think_mode="auto", **kwargs):
        super().__init__(*args, think_mode=think_mode, **kwargs)
        self.instruction = None
        self.spans = ()
        self.current_index = 0
        self.plan_raw = None

    def memory_snapshot(self):
        if self.instruction is None:
            return {}
        return {
            "current_index": self.current_index,
            "subgoals": [
                {"start": a, "end": b, "text": self.instruction[a:b]} for a, b in self.spans
            ],
            "completed_indices": list(range(self.current_index)),
        }

    def _plan(self, view):
        boundaries = candidate_boundaries(view.instruction)
        selected = []
        raw = None
        if boundaries:
            questions = {
                f"boundary_{i}": {
                    "type": "choice",
                    "instructions": (
                        f"Should source offset {point} start a separate ordered subgoal? "
                        "Split only when it separates meaningful route stages. Preserve the "
                        "full instruction and merge fragments that depend on each other."
                    ),
                    "criteria": {
                        "split": "Start a new route stage here.",
                        "merge": "Keep this within the same route stage.",
                    },
                }
                for i, point in enumerate(boundaries)
            }
            raw, choices, _ = self.ask(view, questions)
            selected = [p for i, p in enumerate(boundaries) if choices[f"boundary_{i}"] == "split"]
        self.instruction = view.instruction
        self.spans = source_segments(view.instruction, selected)
        self.current_index = 0
        self.plan_raw = raw

    def decide(self, view):
        started = time.monotonic()
        if self.instruction != view.instruction:
            self._plan(view)
        a, b = self.spans[self.current_index]
        extra = (
            f"Current subgoal [{self.current_index + 1}/{len(self.spans)}]: "
            f"{self.instruction[a:b]}\n"
            f"Route memory: {json.dumps(self.memory_snapshot(), ensure_ascii=False)}\n"
            "Use visual evidence to judge completion. A rejected action, collision, or visible "
            "landmark alone does not prove this stage is complete. STOP ends the entire route."
        )
        questions = {
            "progress": {
                "type": "choice",
                "instructions": "Has the current route stage been completed?",
                "criteria": {
                    "continue": "The current stage still requires navigation.",
                    "complete": "Current visual evidence shows the stage is completed.",
                    "uncertain": "There is insufficient evidence to judge completion.",
                },
            },
            "action": {
                "type": "choice",
                "instructions": ACTION_QUESTION,
                "criteria": ACTION_DESCRIPTIONS,
            },
        }
        raw, choices, probabilities = self.ask(view, questions, extra_state=extra)
        action = Action(choices["action"])
        execute, reason = True, ""
        if choices["progress"] == "complete" and self.current_index < len(self.spans) - 1:
            self.current_index += 1
            execute, reason = False, "subgoal_advanced"
        elif action == Action.STOP and (
            self.current_index != len(self.spans) - 1 or choices["progress"] != "complete"
        ):
            execute, reason = False, "stop_not_confirmed"
        state = self.memory_snapshot()
        state["progress"] = choices["progress"]
        return Decision(
            action,
            raw["model"],
            probabilities["action"],
            {"plan": self.plan_raw, "decision": raw},
            time.monotonic() - started,
            raw.get("usage", {}),
            execute=execute,
            control_reason=reason,
            semantic_state=state,
        )


def make_fixture_policy(settings, budget=None):
    """A finite script chooses answers, but never bypasses the real request/response contract."""
    records = json.loads(Path(settings["responses"]).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Fixture responses must be a nonempty list")
    cursor = 0

    def serve(request):
        nonlocal cursor
        if cursor >= len(records):
            raise ServiceError("fixture_exhausted")
        payload = json.loads(request.content)
        questions = payload["questions"]
        record = records[cursor]
        cursor += 1
        choices = record.get("choices", {})
        if set(choices) != set(questions):
            raise ServiceError(
                "fixture_mismatch", "Fixture question IDs differ from actual request"
            )
        if payload["state"].count("<image>") != len(payload["images"]):
            raise ServiceError("fixture_mismatch", "Image marker mismatch")
        answers = {}
        for name, question in questions.items():
            criteria = question["criteria"]
            if question["type"] != "choice" or choices[name] not in criteria:
                raise ServiceError(
                    "fixture_mismatch", "Fixture choice is not a legal request option"
                )
            answers[name] = {
                "type": "choice",
                "choice": choices[name],
                "probabilities": {k: float(k == choices[name]) for k in criteria},
            }
        return httpx.Response(
            200,
            json={
                "model": "startlux-offline-fixture",
                "answers": answers,
                "usage": {},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(serve), trust_env=False)
    cls = (
        HierarchicalPolicy
        if settings["kind"] == "startlux_hierarchical_fixture"
        else StartLuxPolicy
    )
    policy = cls(
        "offline-fixture",
        endpoint="https://fixture.invalid/v1/decisions",
        client=client,
        budget=budget or CallBudget(2000),
        think_mode=settings.get("think_mode", "auto" if cls is HierarchicalPolicy else "never"),
    )
    # The adapter owns the fixture client, just as it owns a normal internally constructed client.
    policy._owns_client = True
    return policy
