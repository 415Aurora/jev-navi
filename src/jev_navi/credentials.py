"""Project-local credentials, loaded literally without shell evaluation."""

import json
import os
from pathlib import Path

ALLOWED_KEYS = frozenset({"BAMBOO_API_KEY", "STARTLUX_ENDPOINT", "INFERENCE_API_KEY"})
BAMBOO_ENDPOINT = "https://api.bamboonode.dev/v1/decisions"


def load_local_env(path):
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        name, separator, value = line.strip().partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in ALLOWED_KEYS:
            continue
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError("Invalid quoted local credential value") from None
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        if not isinstance(value, str):
            raise ValueError("Local credential values must be strings")
        os.environ.setdefault(name, value)


def save_bamboo_credentials(path, key):
    if not key.strip() or "\n" in key or "\r" in key:
        raise ValueError("API key must be a nonempty single line")
    # Exclusive creation avoids overwriting an existing file or following a symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("BAMBOO_API_KEY=" + json.dumps(key) + "\n")
        handle.write("STARTLUX_ENDPOINT=" + BAMBOO_ENDPOINT + "\n")
