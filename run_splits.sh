#!/usr/bin/env bash
# Run each supported split as a separate end-to-end invocation.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi
CONFIG="${1:-config.yaml}"
status=0
for split in test_ood test_id; do
    printf '\nRunning split: %s\n' "$split"
    if ! "$PYTHON" main.py --config "$CONFIG" --split "$split"; then
        printf 'Split failed: %s\n' "$split" >&2
        status=1
    fi
done
exit "$status"
