#!/usr/bin/env bash
# Project-local, headless Habitat setup. Never activates or modifies another env.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ "$(uname -s)-$(uname -m)" == Linux-x86_64 ]] || { echo 'This lock targets Linux x86_64.' >&2; exit 1; }
mkdir -p .tools/bin .tools/mamba .tools/pip-cache vendor data/scene_datasets environment
export MAMBA_ROOT_PREFIX="$ROOT/.tools/mamba"
export PIP_CACHE_DIR="$ROOT/.tools/pip-cache"
export PYTHONNOUSERSITE=1
MAMBA="$ROOT/.tools/bin/micromamba"
if [[ ! -x "$MAMBA" ]]; then
  curl -fL --retry 2 --connect-timeout 20 --max-time 180 \
    https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64 \
    -o "$MAMBA.download"
  echo '366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3  .tools/bin/micromamba.download' | sha256sum -c -
  mv "$MAMBA.download" "$MAMBA"
  chmod +x "$MAMBA"
fi
echo '366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3  .tools/bin/micromamba' | sha256sum -c -
if [[ -x .habitat/bin/python ]]; then
  .habitat/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 9), "Existing .habitat has a different Python; move it aside explicitly before setup."'
elif [[ -s environment/conda-linux-64.lock ]]; then
  "$MAMBA" --no-rc create -y -p "$ROOT/.habitat" -f environment/conda-linux-64.lock
else
  "$MAMBA" --no-rc create -y -p "$ROOT/.habitat" --override-channels -f environment/environment.yml
fi
checkout_source() {
  local name="$1" commit="$2"
  if [[ ! -d "vendor/$name/.git" ]]; then
    [[ ! -e "vendor/$name" ]] || { echo "Refusing to overwrite vendor/$name" >&2; exit 1; }
    git clone --filter=blob:none --no-checkout "https://github.com/facebookresearch/$name.git" "vendor/$name"
    git -C "vendor/$name" checkout --detach "$commit"
  fi
  [[ "$(git -C "vendor/$name" rev-parse HEAD)" == "$commit" ]] || { echo "Unexpected source commit in vendor/$name" >&2; exit 1; }
  [[ -z "$(git -C "vendor/$name" status --porcelain --untracked-files=no)" ]] || { echo "Modified tracked sources in vendor/$name; inspect before setup." >&2; exit 1; }
}
checkout_source habitat-sim acbe6f4922e68145e401e55c30f9dfea460a3f24
checkout_source habitat-lab 094d6be2f9d057e4781a68ae792132895fd4d3d0
if [[ -s environment/pip-linux-64.lock ]]; then
  .habitat/bin/python -m pip install -r environment/pip-linux-64.lock
else
  .habitat/bin/python -m pip install -r environment/requirements-extra.txt
fi
.habitat/bin/python -m pip install --no-deps --no-build-isolation -e vendor/habitat-lab/habitat-lab
ARCHIVE=data/habitat-test-scenes_v1.0.zip
if [[ ! -f "$ARCHIVE" ]]; then
  curl -fL --retry 2 --connect-timeout 20 --max-time 300 \
    https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes_v1.0.zip -o "$ARCHIVE.download"
  mv "$ARCHIVE.download" "$ARCHIVE"
fi
echo '072c053c00d8e49132e48837eea38f8b6cd19e3079c40fd705ffef12347d7302  data/habitat-test-scenes_v1.0.zip' | sha256sum -c -
if [[ ! -f data/scene_datasets/skokloster-castle.glb || ! -f data/scene_datasets/skokloster-castle.navmesh ]]; then
  .habitat/bin/python -m zipfile -e "$ARCHIVE" data/scene_datasets
fi
.habitat/bin/python -c 'import habitat, habitat_sim, numpy; print("Habitat-Lab", habitat.__version__, "Habitat-Sim", habitat_sim.__version__, "NumPy", numpy.__version__)'
.habitat/bin/python -m pip check
"$MAMBA" --no-rc env export -p "$ROOT/.habitat" --explicit > environment/conda-linux-64.lock
echo 'Setup complete. Use .habitat/bin/python; this installs official Habitat 0.3.3 with Python 3.9 (see docs/environment.md).'
