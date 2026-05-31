#!/usr/bin/env bash
# deploy/run_entry.sh — Cron wrapper for the daily ENTRY phase.
#
# Schedule (US/Eastern, Mon–Fri):
#   9:35 AM ET  — after market open settles (called by crontab)
#
# Steps:
#   1. Run options_premium_screener.py → generates signals_YYYYMMDD.csv
#   2. Run auto_trade.py entry         → places spread orders from signals
#
# Logs to: logs/entry_YYYY-MM-DD.log
# On unexpected exit (non-zero), sends a Telegram/Gmail error alert.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$REPO_DIR/options-screener"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$APP_DIR/logs"
DATE="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/entry_${DATE}.log"

# Activate virtualenv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

mkdir -p "$LOG_DIR"

{
    echo "========================================"
    echo "ENTRY START: $(date)"
    echo "========================================"

    cd "$APP_DIR"

    # Load .env so env vars are available to Python scripts
    if [ -f .env ]; then
        set -o allexport
        # shellcheck source=/dev/null
        source .env
        set +o allexport
    fi

    echo "--- Phase 1: Running screener ---"
    python options_premium_screener.py

    echo ""
    echo "--- Phase 2: Placing orders ---"
    # Pass --live flag only if TT_DRY_RUN=false in .env
    if [ "${TT_DRY_RUN:-true}" = "false" ]; then
        python auto_trade.py entry --live
    else
        python auto_trade.py entry
    fi

    echo ""
    echo "ENTRY DONE: $(date)"

} >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    # Best-effort error alert via Python notify
    cd "$APP_DIR"
    python - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from notifications import notify
notify(
    "🔴 PIPELINE ERROR — entry phase failed",
    f"run_entry.sh exited with non-zero status.\nCheck logs: logs/entry_$(date +%Y-%m-%d).log"
)
PYEOF
fi

exit $EXIT_CODE
