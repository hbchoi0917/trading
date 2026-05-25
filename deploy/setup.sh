#!/usr/bin/env bash
# deploy/setup.sh — One-time server setup for the trading automation pipeline.
#
# Tested on: Ubuntu 22.04 / 24.04 LTS (t3.micro or larger)
# Run once as a non-root user with sudo access:
#   bash deploy/setup.sh
#
# What this does:
#   1. Installs Python 3.11+, pip, git
#   2. Creates a Python virtual environment
#   3. Installs all pip dependencies
#   4. Sets the server timezone to US/Eastern (critical for cron timing)
#   5. Creates logs/ directory with rotation config
#   6. Installs cron jobs from crontab.template
#   7. Prompts you to create .env from .env.example

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$REPO_DIR/options-screener"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$APP_DIR/logs"
CRON_TEMPLATE="$REPO_DIR/deploy/crontab.template"

echo "============================================================"
echo " Trading Automation — Server Setup"
echo " Repo : $REPO_DIR"
echo " App  : $APP_DIR"
echo "============================================================"
echo ""

# ── 1. System packages ───────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3-pip git curl tzdata

# ── 2. Timezone ──────────────────────────────────────────────────────────────
echo "[2/7] Setting timezone to US/Eastern..."
sudo timedatectl set-timezone America/New_York
echo "      Current time: $(date)"

# ── 3. Virtual environment ───────────────────────────────────────────────────
echo "[3/7] Creating Python virtual environment at $VENV_DIR..."
python3.11 -m venv "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ── 4. Dependencies ──────────────────────────────────────────────────────────
echo "[4/7] Installing Python dependencies..."
pip install -r "$APP_DIR/requirements.txt" -q
echo "      Done. Key packages:"
pip show tastytrade yfinance pandas | grep -E "^(Name|Version):" | paste - -

# ── 5. Logs directory ────────────────────────────────────────────────────────
echo "[5/7] Creating logs directory..."
mkdir -p "$LOG_DIR"

# Logrotate config: keep 30 days, compress, no error if missing
sudo tee /etc/logrotate.d/trading-bot > /dev/null <<EOF
$LOG_DIR/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    copytruncate
}
EOF
echo "      Log rotation configured (30 days, daily)"

# ── 6. .env file ─────────────────────────────────────────────────────────────
echo "[6/7] Checking .env file..."
ENV_FILE="$APP_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$APP_DIR/.env.example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo ""
    echo "  ⚠️  Created $ENV_FILE from .env.example"
    echo "  ✏️  Edit it now to add your credentials:"
    echo "       nano $ENV_FILE"
    echo ""
    echo "  Required fields:"
    echo "    TT_PAPER_API_KEY   — from Tastytrade Settings → API → Paper Trading"
    echo "    TG_BOT_TOKEN       — from @BotFather on Telegram"
    echo "    TG_CHAT_ID         — from https://api.telegram.org/bot<TOKEN>/getUpdates"
    echo ""
else
    echo "      .env already exists — skipping"
fi

# ── 7. Cron jobs ─────────────────────────────────────────────────────────────
echo "[7/7] Installing cron jobs..."

# Substitute actual paths into the template
PYTHON_BIN="$VENV_DIR/bin/python"
sed \
    -e "s|__REPO_DIR__|$REPO_DIR|g" \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    -e "s|__PYTHON__|$PYTHON_BIN|g" \
    -e "s|__LOG_DIR__|$LOG_DIR|g" \
    "$CRON_TEMPLATE" > /tmp/trading_cron

# Merge with existing crontab (avoid duplicates)
crontab -l 2>/dev/null | grep -v "auto_trade\|run_entry\|run_monitor\|trading-bot" > /tmp/existing_cron || true
cat /tmp/existing_cron /tmp/trading_cron | crontab -

echo "      Installed cron jobs:"
crontab -l | grep -v "^#" | grep -v "^$"

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Edit credentials:  nano $ENV_FILE"
echo "   2. Run health check:  cd $APP_DIR && $PYTHON_BIN healthcheck.py"
echo "   3. Test dry-run:      cd $APP_DIR && $PYTHON_BIN auto_trade.py entry"
echo ""
echo " Cron logs will appear in: $LOG_DIR/"
echo "============================================================"
