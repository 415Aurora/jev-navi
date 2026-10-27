import pytest


def test_profiles_separate_local_and_r2rce_contracts():
    from jev_navi.profiles import resolve_config

    local = resolve_config({"policy": {"kind": "scripted", "actions": ["STOP"]}})
    assert local["run"]["max_decisions"] == 60
    modern = resolve_config({"protocol_id": "r2rce_v13_dev", "policy": {"kind": "scripted"}})
    assert modern["run"]["max_decisions"] == 500
    assert modern["run"]["max_steps"] == 500
    assert modern["camera"]["width"] == 224
    assert modern["depth_camera"]["width"] == 256
    assert modern["environment"]["allow_sliding"] is True
    assert modern["environment"]["radius_m"] == 0.1
    assert modern["execution"]["mode"] == "offline"
    with pytest.raises(ValueError, match="protocol"):
        resolve_config({"protocol_id": "r2rce_v13_dev", "environment": {"depth_pitch_deg": 30}})
