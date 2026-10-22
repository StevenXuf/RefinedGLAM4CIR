#!/usr/bin/env bash
# Run each supported split as a separate end-to-end invocation.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi
CONFIG="${1:-config.yaml}"
MODIFIERS_DIR="${2:-}"
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
    reuse=()
    if [[ -n "$MODIFIERS_DIR" ]]; then
        if ! source_path=$("$PYTHON" - "$CONFIG" "$split" "$MODIFIERS_DIR" <<'PY'
import sys
from pathlib import Path
from utils import get_config, output_path

cfg = get_config(sys.argv[1], sys.argv[2])
directory = Path(sys.argv[3]).resolve() / sys.argv[2]
source = directory / output_path(cfg, 'modifiers.json', retrieval=False).name
if not source.is_file():
    source = directory / 'modifiers.json'  # Earlier checkpoint naming.
if not source.is_file():
    raise FileNotFoundError(f'No modifier checkpoint in {directory}')
print(source)
PY
        ); then
            status=1
            continue
        fi
        reuse=(--reuse-modifiers "$source_path")
    fi
    if ! "$PYTHON" main.py --config "$CONFIG" --split "$split" "${reuse[@]}" 2>&1 | tee "$log_path"; then
        printf 'Split failed: %s\n' "$split" >&2
        status=1
    fi
done
exit "$status"
