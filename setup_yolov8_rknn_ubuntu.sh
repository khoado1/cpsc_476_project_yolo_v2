#!/usr/bin/env bash
set -Eeuo pipefail

# Run on Ubuntu x86_64. Recommended: Ubuntu 20.04 with Python 3.8.
# Usage: bash setup_yolov8_rknn_ubuntu.sh
# Optional: PYTHON_BIN=python3.8 bash setup_yolov8_rknn_ubuntu.sh

REPO_URL='https://github.com/khoado1/cpsc_476_project_yolo_v2.git'
PROJECT_DIR="${PROJECT_DIR:-$HOME/cpsc_476_project_yolo_v2}"
PYTHON_BIN="${PYTHON_BIN:-python3.8}"

log() { printf '\n==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nFailed at line %s. See error above.\n" "$LINENO" >&2' ERR

[[ "$(uname -s)" == Linux ]] || fail 'This script requires Linux.'
[[ "$(uname -m)" == x86_64 ]] || fail 'RKNN Toolkit2 host wheels in this guide require x86_64.'
command -v apt-get >/dev/null || fail 'This script expects Ubuntu/Debian apt-get.'

log 'Installing system prerequisites'
sudo apt-get update
sudo apt-get install -y git git-lfs wget curl build-essential python3-pip python3-venv libgl1 libglib2.0-0

command -v "$PYTHON_BIN" >/dev/null || fail "$PYTHON_BIN is missing. Install a compatible Python version (recommended Python 3.8) and rerun with PYTHON_BIN=..."
"$PYTHON_BIN" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 8), 'This setup targets Python 3.8 for legacy RKNN Toolkit2 1.5.0. Choose a Python 3.8 interpreter.'
PY

log 'Cloning project'
if [[ -d "$PROJECT_DIR/.git" ]]; then
  CURRENT_REMOTE="$(git -C "$PROJECT_DIR" remote get-url origin 2>/dev/null || true)"
  case "$CURRENT_REMOTE" in
    *github.com/khoado1/cpsc_476_project_yolo_v2|*github.com/khoado1/cpsc_476_project_yolo_v2.git)
      log 'Fork already cloned; leaving existing changes untouched (no automatic pull)' ;;
    *) fail "Existing directory points to a different origin: $CURRENT_REMOTE. Choose another PROJECT_DIR or inspect it manually." ;;
  esac
elif [[ -e "$PROJECT_DIR" ]]; then
  fail "Directory exists but is not a Git repo: $PROJECT_DIR"
else
  git clone --recursive "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
git submodule update --init --recursive
git lfs install

log 'Creating Python virtual environment'
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip<25' setuptools wheel

log 'Installing YOLOv8 dependencies'
python -m pip install 'numpy==1.23.5' 'ultralytics==8.0.147' 'onnx==1.10.0' 'onnxruntime==1.10.0'

log 'Looking for an RKNN Toolkit2 wheel (1.5.0, cp38, Linux x86_64)'
mapfile -t WHEELS < <(find "$PROJECT_DIR" -type f -name 'rknn_toolkit2-1.5.0*cp38*linux_x86_64.whl' | sort)
if (( ${#WHEELS[@]} == 0 )); then
  cat <<'MSG'
RKNN Toolkit2 1.5.0 cp38 wheel not found in the cloned repository.
Download the compatible Python 3.8 x86_64 wheel from the official RKNN Toolkit2
release/packages and place it anywhere inside the project directory.
Then activate .venv and run: python -m pip install /path/to/wheel.whl
Do not attempt ONNX-to-RKNN conversion until 'from rknn.api import RKNN' succeeds.
MSG
else
  python -m pip install "${WHEELS[0]}"
  python -c 'from rknn.api import RKNN; print("RKNN Toolkit2 import succeeded")'
fi

cat <<MSG

SETUP FINISHED (check any wheel warning above).

NEXT STEPS — these require you to inspect and edit model paths in the scripts:
  cd "$PROJECT_DIR"
  source .venv/bin/activate
  python pytorch2onnx.py                 # only if you need to export an ONNX model
  python onnx2rknn_step1.py             # inspect script and config first
  python onnx2rknn_step2.py             # inspect script and config first

The ONNX model path, calibration images, target platform (rk3588), and
quantization configuration must match your actual model and scripts.
Record terminal output and verify the .rknn file exists for your demo.
MSG
printf '\nActual project directory: %s\n' "$PROJECT_DIR"
