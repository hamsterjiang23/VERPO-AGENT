#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${VERPO_AGENT_PYTHON:-python3}
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/preflight.py"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
fi
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" -c 'import verpo_agent; print("VERPO-AGENT", verpo_agent.__version__)'
echo 'Development package installed. GPU runtime and Agent training are not installed or launched.'
