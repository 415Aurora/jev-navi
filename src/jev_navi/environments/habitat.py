"""Modern policy-side adapter for the isolated, official Habitat worker."""

import base64
import json
import os
import selectors
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..types import Observation


def unpack(value):
    if isinstance(value, dict) and "_array" in value:
        return (
            np.frombuffer(base64.b64decode(value["_array"]), dtype=value["dtype"])
            .reshape(value["shape"])
            .copy()
        )
    if isinstance(value, dict):
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


class HabitatEnvironment:
    kind = "habitat"

    def __init__(self, episode, camera, *, python=None, **settings):
        project = Path(__file__).resolve().parents[3]
        executable = (
            python or os.environ.get("JEV_HABITAT_PYTHON") or str(project / ".habitat/bin/python")
        )
        if not Path(executable).is_file():
            raise ValueError("Habitat interpreter missing; run scripts/setup_habitat.sh first")
        self._stderr = tempfile.TemporaryFile(mode="w+")
        self._process = subprocess.Popen(
            [executable, "-u", str(Path(__file__).with_name("habitat_worker.py"))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            env={**os.environ, "HABITAT_SIM_LOG": "quiet", "MAGNUM_LOG": "quiet"},
        )
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            self.runtime = self._rpc(
                "init", episode=episode, camera=asdict(camera), settings=settings
            )
        except Exception:
            self.close()
            raise

    def _rpc(self, operation, **arguments):
        if self._process.poll() is not None:
            raise RuntimeError("Habitat worker exited: " + self._diagnostic())
        try:
            self._process.stdin.write(json.dumps({"operation": operation, **arguments}) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RuntimeError("Habitat worker pipe closed: " + self._diagnostic()) from None
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if not self._selector.select(timeout=min(1, max(deadline - time.monotonic(), 0))):
                if self._process.poll() is not None:
                    break
                continue
            line = self._process.stdout.readline()
            if not line:
                break
            if not line.startswith("JEV_RESPONSE "):
                continue
            response = json.loads(line[len("JEV_RESPONSE ") :])
            if "error" in response:
                raise RuntimeError(f"Habitat {response['error']}: {response['message']}")
            return unpack(response["result"])
        raise RuntimeError("Habitat worker timed out or exited: " + self._diagnostic())

    def _diagnostic(self):
        self._stderr.seek(0)
        return self._stderr.read()[-4000:]

    def reset(self):
        observation = self._rpc("reset")
        return Observation(observation["rgb"], observation["step"])

    def depth(self):
        return self._rpc("depth")

    def execute(self, action):
        observation, feedback = self._rpc("execute", action=action.value)
        return Observation(observation["rgb"], observation["step"]), feedback

    def evaluate(self, stopped):
        return self._rpc("evaluate", stopped=stopped)

    def close(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._selector.close()
        self._process.stdin.close()
        self._process.stdout.close()
        self._stderr.close()
