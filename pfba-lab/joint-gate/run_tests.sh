#!/usr/bin/env bash
# CPU tests must use this interpreter only (per task instructions).
set -euo pipefail
cd "$(dirname "$0")"
exec "D:/MINICONDA/envs/py312/python.exe" -m unittest tests.test_joint_gate -v "$@"
