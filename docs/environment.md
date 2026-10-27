# Local Habitat environment

The working simulator is **Habitat-Sim 0.3.3 + Habitat-Lab 0.3.3 on Python 3.9.23**, installed under `.habitat/`. The main policy application retains its separate Python 3.11+ environment (`.venv/`, Python 3.12 on this machine). It communicates with the simulator worker over JSON on standard input/output. This preserves the application's Python requirement while allowing unmodified upstream Habitat packages to run.

## Install and verify

Run from the project root:

```bash
./scripts/setup_habitat.sh
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet .habitat/bin/python environment/smoke_habitat.py
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet .habitat/bin/python environment/smoke_habitat_lab.py
```

The installer downloads the official micromamba 2.9.0 executable into `.tools/bin/`, checks its SHA-256, disables Conda configuration reads with `--no-rc`, and stores its cache under `.tools/mamba/`. It uses `.habitat/` as the environment prefix. It does not activate an environment or edit `~/.condarc`, the existing Anaconda installation, shell startup files, or other environments. An existing prefix with another Python version or a vendor checkout with unexpected commits/modified tracked files stops setup for inspection.

`environment/conda-linux-64.lock` contains explicit package URLs and hashes for the Conda layer. `environment/pip-linux-64.lock` pins the additional pip distributions, including Pillow 10.4.0 required by Habitat-Sim's metadata. `environment/source-manifest.json` records requested and installed versions, exact source commits, bootstrap hash, dataset URL/hash, GPU and reasons for deviations. These locks target Linux x86_64; the pip overlay has version pins but does not include wheel hashes.

The installer is safe to rerun and verifies both Habitat imports and `pip check`. The two smoke commands perform the actual GPU and task checks. No model API or hosted inference is involved in setup or these checks.

## Why the originally requested versions changed

The original request pinned Python 3.11, Habitat-Sim tag v0.3.4 at `24dfda7a746037f7b6b63730a02e3d05a688fff0`, and Habitat-Lab tag v0.3.4 at `cdbb4880519505adf45fba0f0c0c3a3fd18a2a55`. Both exact source commits were fetched and inspected. As checked on 2026-10-26:

- The official `aihabitat` Conda channel had no `habitat-sim=0.3.4` package.
- The v0.3.4 Habitat-Sim source's `pyproject.toml` reported package version `0.3.3` and required `numpy>=2.0.0,<2.4`. The paired v0.3.4 Habitat-Lab requirements pinned `numpy==1.26.4`, so their declared dependencies conflict.
- The coherent v0.3.3 official headless builds were available for Python 3.9. An attempt to solve them with Python 3.11 failed because the builds require the Python 3.9 ABI.
- A Python 3.11 source-build attempt was started, but Habitat-Lab v0.3.3 import failed immediately with `ValueError: mutable default ... IteratorOptionsConfig ... iterator_options is not allowed: use default_factory`. The build was stopped when the unmodified paired Python 3.9 fallback was chosen. Its build log is retained as diagnostic evidence; no source-built simulator is used.

The installed official headless simulator build is `py3.9_headless_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24`. The paired Habitat-Lab source is pinned to `094d6be2f9d057e4781a68ae792132895fd4d3d0`. Both imported packages report `0.3.3`. This setup does not claim to provide v0.3.4 or a Python 3.11 Habitat runtime.

The user's existing Anaconda base uses Python 3.7 and cannot parse its current `channel_priority=flexible` setting. The project-local micromamba installation bypasses that configuration. The desktop execution sandbox also failed to initialize its loopback network, so installation and GPU smoke commands were run using reviewed host execution; this is an execution-environment limitation, not a Habitat dependency.

## Scene assets and observed results

The public `habitat-test-scenes_v1.0.zip` archive is 93,688,442 bytes (about 89 MiB), with SHA-256 `072c053c00d8e49132e48837eea38f8b6cd19e3079c40fd705ffef12347d7302`. It contains three small scenes and matching navmeshes, extracted into:

```text
data/scene_datasets/skokloster-castle.glb
data/scene_datasets/skokloster-castle.navmesh
data/scene_datasets/van-gogh-room.glb
data/scene_datasets/van-gogh-room.navmesh
data/scene_datasets/apartment_1.glb
data/scene_datasets/apartment_1.navmesh
```

The accessible archive came from the official Facebook public-files host. A Hugging Face test-scene request timed out, so setup uses this fixed archive with a verified checksum. ReplicaCAD was not needed for navigation validation and was not downloaded; these public test scenes require no credentials.

On the local NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB VRAM, driver 590.48.01), the simulator smoke rendered RGB-D through headless EGL at 256×256, loaded a 226.66 m² navigable mesh, and moved 1.00001 m through four forward actions after a turn. Depth values were finite with a maximum of 5.364 m; zero denotes pixels with no measured geometry. Results and a rendered view are saved to `environment/smoke-result.json` and `environment/smoke-rgb.png`.

The independent Habitat-Lab smoke instantiated an actual `habitat.Env` with the PointNav configuration and an in-memory `NavigationEpisode`, called `reset`, stepped the navigation actions, and read task metrics. It produced only `rgb` and `depth` observation keys, with shapes 256×256×3 and 256×256×1, moved 1.00000 m, and reported distance-to-goal, success, SPL and distance reward. Results are recorded in `environment/smoke-lab-result.json`. That small smoke episode validates task execution; its metrics are not a benchmark result.

The test scene has no semantic annotation file. Upstream may emit a semantic-descriptor diagnostic, duplicate plugin notices and Gym's maintenance warning. RGB-D rendering, navigation and task metrics pass; semantic sensing, Bullet physics, training, CUDA tensor interoperability and ReplicaCAD manipulation have not been validated. The chosen build disables Bullet and uses GPU graphics through EGL without requiring a CUDA toolkit.
