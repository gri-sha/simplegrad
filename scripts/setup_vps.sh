#!/usr/bin/env bash
# setup_vps.sh — bootstrap simplegrad + examples on a Linux VPS (RunPod)
#
# Usage:
#   bash setup_vps.sh
#
# Optional env vars:
#   BASE_DIR   where to clone repos (default: /workspace)

set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y graphviz unzip

BASE_DIR="${BASE_DIR:-$HOME}"
SIMPLEGRAD_REPO="https://github.com/simplegrad/simplegrad.git"
EXAMPLES_REPO="https://github.com/simplegrad/examples.git"

SIMPLEGRAD_DIR="$BASE_DIR/simplegrad"
EXAMPLES_DIR="$BASE_DIR/examples"
DATASETS_DIR="$EXAMPLES_DIR/datasets"

log() { echo "[setup] $*"; }
die() { echo "[error] $*" >&2; exit 1; }

# clean up temp files on exit
TMP_FILES=()
cleanup() { rm -f "${TMP_FILES[@]}"; }
trap cleanup EXIT

for cmd in python3 git curl; do
    command -v "$cmd" &>/dev/null || die "'$cmd' not found — install it and retry."
done

# clone or update a repo
clone_or_pull() {
    local repo="$1" dir="$2" name="$3"
    if [[ -d "$dir/.git" ]]; then
        log "$name already cloned — pulling latest."
        git -C "$dir" pull
    else
        log "Cloning $name..."
        git clone "$repo" "$dir"
    fi
}

clone_or_pull "$SIMPLEGRAD_REPO" "$SIMPLEGRAD_DIR" "simplegrad"
clone_or_pull "$EXAMPLES_REPO"   "$EXAMPLES_DIR"   "examples"

# simplegrad venv
log "Setting up simplegrad venv..."
python3 -m venv "$SIMPLEGRAD_DIR/.venv"
"$SIMPLEGRAD_DIR/.venv/bin/pip" install --upgrade pip --quiet
log "Installing simplegrad[gpu,dev,bench]..."
"$SIMPLEGRAD_DIR/.venv/bin/pip" install -e "$SIMPLEGRAD_DIR[gpu,dev,bench]"

# examples venv
log "Setting up examples venv..."
python3 -m venv "$EXAMPLES_DIR/.venv"
"$EXAMPLES_DIR/.venv/bin/pip" install --upgrade pip --quiet
log "Installing examples requirements..."
"$EXAMPLES_DIR/.venv/bin/pip" install -r "$EXAMPLES_DIR/requirements.txt"

# register kernel
log "Registering Jupyter kernel 'sg-env'..."
"$EXAMPLES_DIR/.venv/bin/python" -m ipykernel install \
    --user --name=sg-env --display-name "Python (sg-env)"

# datasets
mkdir -p "$DATASETS_DIR"

# Shakespeare
if [[ -f "$DATASETS_DIR/shakespeare.txt" ]]; then
    log "Shakespeare: already present, skipping."
else
    log "Downloading Shakespeare (TinyShakespeare)..."
    curl -fsSL \
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
        -o "$DATASETS_DIR/shakespeare.txt"
    log "  -> $DATASETS_DIR/shakespeare.txt"
fi

# MNIST
if [[ -d "$DATASETS_DIR/mnist" && -n "$(ls -A "$DATASETS_DIR/mnist" 2>/dev/null)" ]]; then
    log "MNIST: already present, skipping."
else
    log "Downloading MNIST from Kaggle..."
    tmp=$(mktemp /tmp/mnist_XXXXXX.zip)
    TMP_FILES+=("$tmp")
    curl -fsSL "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset" -o "$tmp"
    mkdir -p "$DATASETS_DIR/mnist"
    unzip -q "$tmp" -d "$DATASETS_DIR/mnist"
    log "  -> $DATASETS_DIR/mnist/"
fi

# CIFAR-10
if [[ -d "$DATASETS_DIR/cifar10" && -n "$(ls -A "$DATASETS_DIR/cifar10" 2>/dev/null)" ]]; then
    log "CIFAR-10: already present, skipping."
else
    log "Downloading CIFAR-10 from Kaggle..."
    tmp=$(mktemp /tmp/cifar10_XXXXXX.zip)
    TMP_FILES+=("$tmp")
    curl -fsSL "https://www.kaggle.com/api/v1/datasets/download/oxcdcd/cifar10" -o "$tmp"
    mkdir -p "$DATASETS_DIR/cifar10"
    unzip -q "$tmp" -d "$DATASETS_DIR/cifar10"
    log "  -> $DATASETS_DIR/cifar10/"
fi

log ""
log "Done."
log "  simplegrad  $SIMPLEGRAD_DIR  (.venv with gpu,dev,bench)"
log "  examples    $EXAMPLES_DIR    (.venv, kernel: 'sg-env')"
log "  datasets    $DATASETS_DIR"
