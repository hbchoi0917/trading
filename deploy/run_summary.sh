#!/usr/bin/env bash
# deploy/run_summary.sh — Cron wrapper for daily/weekly summary email.
#
# Schedule (US/Eastern, Mon–Fri):
#   4:30 PM ET — after post-close monitor; reads positions.csv only (no broker connection)
#
# Sends:
#   Daily  — every trading day: entries/closes/open positions/MTD P&L/cap warnings
#   Weekly — last trading day of the week (Friday, or Thursday if Friday is a holiday)
#
# Logs to: logs/summary_YYYY-MM-DD_HHMM.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$REPO_DIR/options-screener"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$APP_DIR/logs"
TIMESTAMP="$(date +%Y-%m-%d_%H%M)"
LOG_FILE="$LOG_DIR/summary_${TIMESTAMP}.log"

# Activate virtualenv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

mkdir -p "$LOG_DIR"

{
    echo "========================================"
    echo "SUMMARY START: $(date)"
    echo "========================================"

    cd "$APP_DIR"

    if [ -f .env ]; then
        set -o allexport
        # shellcheck source=/dev/null
        source .env
        set +o allexport
    fi

    python auto_trade.py summary

    echo ""
    echo "SUMMARY DONE: $(date)"

} >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    cd "$APP_DIR"
    python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from notifications import notify
notify(
    "🔴 PIPELINE ERROR — summary phase failed",
    f"run_summary.sh exited with non-zero status.\nCheck logs: logs/summary_*.log"
)
PYEOF
fi

exit $EXIT_CODE
