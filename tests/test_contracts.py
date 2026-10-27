import json

import numpy as np
import pytest
from PIL import Image


def test_contract_matrix_records_images_and_stays_under_24(tmp_path, monkeypatch):
    import jev_navi.contracts as contracts
    from jev_navi.contracts import run_contract
    from jev_navi.policies import ScriptedPolicy

    images = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((16, 16, 3), i * 80, dtype=np.uint8)).save(path)
        images.append(str(path))
    seen = []

    class Fake(ScriptedPolicy):
        def decide(self, view):
            seen.append(view)
            return super().decide(view)

    monkeypatch.setattr(contracts, "make_policy", lambda cfg, budget: Fake(["STOP"] * 24))
    result = run_contract(
        {"free_quota_confirmed": True, "policy": {"kind": "startlux"}},
        images,
        "door",
        tmp_path / "contract",
    )
    assert 4 <= result["cases"] <= 24
    assert len(seen[0].frames) == 1
    assert len(seen[1].frames) == 3
    assert result["image_grounding_verified"] is False
    assert (tmp_path / "contract" / "responses.jsonl").exists()


def test_async_raw_contains_terminal_response():
    import httpx
    from test_providers import view

    from jev_navi.providers import D3Policy

    def serve(request):
        if request.method == "POST":
            return httpx.Response(200, json={"data": {"id": "x", "status": 1}})
        return httpx.Response(
            200,
            json={
                "data": {
                    "status": 10,
                    "output": {
                        "model": "d3-mini",
                        "choices": {
                            "action": {
                                "probabilities": {
                                    "MOVE_FORWARD": 0.1,
                                    "TURN_LEFT": 0.1,
                                    "TURN_RIGHT": 0.1,
                                    "STOP": 0.7,
                                }
                            }
                        },
                        "provider_detail": "preserve-me",
                    },
                }
            },
        )

    policy = D3Policy(
        "key", poll_interval=0, client=httpx.Client(transport=httpx.MockTransport(serve))
    )
    raw = policy.decide(view()).raw
    assert raw["terminal"]["data"]["output"]["provider_detail"] == "preserve-me"


def test_habitat_backend_roundtrip():
    from pathlib import Path

    from jev_navi.environments.habitat import HabitatEnvironment
    from jev_navi.types import Action, Camera

    project = Path(__file__).resolve().parents[1]
    native = project / ".habitat/bin/python"
    scene = project / "data/scene_datasets/skokloster-castle.glb"
    state = project / "environment/smoke-result.json"
    if not native.exists() or not scene.exists() or not state.exists():
        pytest.skip("Native Habitat integration assets not installed")
    metadata = json.loads(state.read_text())
    if "start_position" not in metadata:
        pytest.skip("Native smoke metadata has no reference start")
    start = metadata["start_position"]
    episode = {
        "id": "integration",
        "scene": str(scene),
        "start_position": start,
        "start_rotation": [0, 0, 0, 1],
        "goal_position": start,
        "success_radius": 0.3,
    }
    env = HabitatEnvironment(episode, Camera())
    try:
        obs = env.reset()
        assert obs.rgb.shape == (256, 256, 3)
        assert env.depth().shape == (256, 256)
        assert not env.evaluate(False)["success"]
        env.execute(Action.STOP)
        assert env.evaluate(True)["success"]
    finally:
        env.close()
