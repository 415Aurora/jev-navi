"""Finite HTTP exchange fixtures using the production D3 adapter, never a real network."""

import json
from pathlib import Path

import httpx

from .action_context import ROUTE_PROMPT_VERSION
from .providers import CallBudget, D3Policy
from .types import Action, ServiceError


def make_d3_fixture_policy(settings, budget=None):
    records = json.loads(Path(settings["responses"]).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("D3 fixture responses must be a nonempty exchange list")
    for record in records:
        if (
            not isinstance(record, dict)
            or not {"method", "path", "status_code", "json"} <= record.keys()
        ):
            raise ValueError("D3 fixture needs method/path/status_code/json exchanges")
    cursor = 0

    def serve(request):
        nonlocal cursor
        if cursor >= len(records):
            raise ServiceError("fixture_exhausted")
        record = records[cursor]
        cursor += 1
        if request.method != record["method"] or request.url.path != record["path"]:
            raise ServiceError("fixture_mismatch", "fixture_mismatch: unexpected HTTP exchange")
        if request.method == "POST":
            payload = json.loads(request.content)
            questions = payload["input"]["choices"]
            names = [o["name"] for o in questions[0]["options"]] if len(questions) == 1 else []
            if payload["app"] != D3Policy.app or names != [a.value for a in Action]:
                raise ServiceError("fixture_mismatch")
        return httpx.Response(record["status_code"], json=record["json"])

    client = httpx.Client(transport=httpx.MockTransport(serve), trust_env=False)
    try:
        policy = D3Policy(
            "offline-fixture",
            base_url="https://fixture.invalid",
            client=client,
            budget=budget or CallBudget(2000),
            poll_interval=0,
            prompt_version=settings.get("prompt_version", ROUTE_PROMPT_VERSION),
        )
    except BaseException:
        client.close()
        raise
    policy._owns_client = True
    return policy
