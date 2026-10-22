#!/bin/sh
set -eu
app_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$app_dir")
export PYTHONDONTWRITEBYTECODE=1
cd "$repo_dir"
exec "$repo_dir/.venv/bin/python" -B -m pp1_demo.app "$@"
