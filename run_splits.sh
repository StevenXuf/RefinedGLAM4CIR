#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/glam-matplotlib}"
PYTHON_BIN="${PYTHON:-python}"
if [[ -x .venv/bin/python && -z "${PYTHON:-}" ]]; then
    PYTHON_BIN=.venv/bin/python
fi
for split in test_id test_ood; do
    "$PYTHON_BIN" pipeline.py --config "${1:-config.yaml}" --split "$split" "${@:2}"
done
