#!/usr/bin/env bash
# Sync local repo to the coder instance (excludes .venv, .git, caches).
set -euo pipefail
HOST="${1:-main.harel-8g.harel.coder}"
DEST="${2:-/home/user/cleandift}"

rsync -avh --delete \
  --exclude .git --exclude .venv --exclude __pycache__ \
  --exclude '*.pyc' --exclude .DS_Store \
  --exclude data --exclude checkpoints --exclude '.cache' \
  ./ "${HOST}:${DEST}/"
