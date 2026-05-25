#!/usr/bin/env bash
# deploy/run_cc.sh — Cron wrapper for the covered call alert screener.
#
# Schedule (US/Eastern, Mon–Fri):
#   3:25 PM ET — 5 minutes before the put credit spread entry run
#
# Scans VOO, QQQM, EWY, GOOGL, SCHD, DRAM for covered call conditions
# and sends Telegram/Gmail alerts. No orders placed automatically.
# Execute the suggested trades manually in Fidelity.
#
# Logs to: logs/cc_YYYY-MM-DD.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$REPO_DIR/options-screener"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$APP_DIR/logs"
DATE="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/cc_${DATE}.log"

source "$VENV_DIR/bin/activate"

mkdir -p "$LOG_DIR"

{
    echo "========================================"
    echo "CC SCREENER START: $(date)"
    echo "========================================"

    cd "$APP_DIR"

    if [ -f .env ]; then
        set -o allexport
        source .env
        set +o allexport
    fi

    python covered_call_screener.py

    echo ""
    echo "CC SCREENER DONE: $(date)"

} >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    cd "$APP_DIR"
    python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from notifications import notify
notify(
    "🔴 CC SCREENER ERROR",
    "run_cc.sh exited with non-zero status. Check logs/cc_*.log"
)
PYEOF
fi

exit $EXIT_CODE
