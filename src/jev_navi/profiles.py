"""Versioned engineering protocols; modern Habitat is not the original VLN-CE runtime."""

from copy import deepcopy

LOCAL = "local_v1"
R2RCE = "r2rce_v13_dev"
R2RCE_SAFETY = "r2rce_v13_dev_depth30"
PROTOCOLS = {LOCAL, R2RCE, R2RCE_SAFETY}


def resolve_config(config):
    config = deepcopy(config)
    protocol = config.setdefault("protocol_id", LOCAL)
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown protocol_id")
    modern = protocol != LOCAL
    defaults = {
        "environment": {
            "backend": "habitat",
            "forward_m": 0.25,
            "turn_deg": 15,
            "radius_m": 0.1 if modern else 0.2,
            "agent_height_m": 1.5,
            "allow_sliding": modern,
            "depth_pitch_deg": 30 if protocol == R2RCE_SAFETY else 0,
            "max_episode_steps": 500 if modern else 61,
            "success_strict": modern,
        },
        "camera": {
            "width": 224 if modern else 256,
            "height": 224 if modern else 256,
            "hfov_deg": 90,
            "height_m": 1.25,
        },
        "run": {"max_decisions": 500 if modern else 60, "seed": 17, "call_limit": 500},
    }
    if modern:
        defaults["depth_camera"] = {"width": 256, "height": 256, "hfov_deg": 90, "height_m": 1.25}
        defaults["run"]["max_steps"] = 500
    for section, values in defaults.items():
        dest = config.setdefault(section, {})
        for key, value in values.items():
            dest.setdefault(key, value)
    config.setdefault(
        "execution",
        {
            "mode": "offline",
            "phase_id": "r2rce-offline-v1",
            "ledger": "runs/request-ledger.sqlite3",
        },
    )
    if config["execution"].get("mode", "offline") not in ("offline", "live"):
        raise ValueError("execution.mode must be offline or live")
    if modern:
        # A named protocol fixes the physical contract, not just an experiment label.
        for section in ("environment", "camera", "depth_camera"):
            for key, value in defaults[section].items():
                if key != "backend" and config[section].get(key) != value:
                    raise ValueError(f"Value violates protocol {protocol}: {section}.{key}")
    cap = 500 if modern else 60
    count = config["run"]["max_decisions"]
    if type(count) is not int or not 1 <= count <= cap:
        raise ValueError(f"Protocol decision limit must be 1..{cap}")
    steps = config["run"].get("max_steps")
    if steps is not None and (type(steps) is not int or not 1 <= steps <= cap):
        raise ValueError(f"Protocol execution-step limit must be 1..{cap}")
    return config
