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
    if ! log_path=$("$PYTHON" - "$CONFIG" "$split" <<'PY'
import sys
from utils import get_config, output_path
path = output_path(get_config(sys.argv[1], sys.argv[2]), 'run.log')
path.parent.mkdir(parents=True, exist_ok=True)
print(path)
PY
    ); then
        status=1
        continue
    fi
    if ! "$PYTHON" main.py --config "$CONFIG" --split "$split" 2>&1 | tee "$log_path"; then
        printf 'Split failed: %s\n' "$split" >&2
        status=1
    fi
done
exit "$status"
