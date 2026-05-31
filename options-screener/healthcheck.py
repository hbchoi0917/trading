"""
healthcheck.py — Pre-flight environment check for the trading pipeline.

Run this before going live or after any server restart to verify that all
required credentials, dependencies, and network paths are working.

Usage:
    python healthcheck.py          # check everything, print report
    python healthcheck.py --notify # also send a Telegram/Gmail test alert
    python healthcheck.py --quiet  # exit 0 (ok) or 1 (failed), minimal output

Exit codes:
    0  All required checks passed
    1  One or more required checks failed
"""

import argparse
import importlib
import os
import sys
import urllib.request
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REQUIRED_PACKAGES = [
    ("tastytrade",    "tastytrade>=12.0.0"),
    ("pandas",        "pandas"),
    ("yfinance",      "yfinance"),
    ("dotenv",        "python-dotenv"),
]

REQUIRED_ENV_VARS = {
    "broker": [
        ("TT_PAPER_TRADING", "Set to 'true' for paper trading (recommended to start)"),
        ("TT_DRY_RUN",       "Set to 'true' to log without submitting orders"),
    ],
    "broker_auth": [
        ("TT_PAPER_API_KEY", "Paper trading API key from Tastytrade Settings → API"),
    ],
    "notifications": [
        ("TG_BOT_TOKEN", "Telegram bot token from @BotFather"),
        ("TG_CHAT_ID",   "Your Telegram chat_id"),
    ],
}

INTERNET_CHECKS = [
    ("api.tastytrade.com", "Tastytrade API"),
    ("api.telegram.org",   "Telegram API"),
    ("query1.finance.yahoo.com", "Yahoo Finance (yfinance)"),
]


# ── Check runners ─────────────────────────────────────────────────────────────

def check_packages() -> list[tuple[bool, str]]:
    results = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            m = importlib.import_module(import_name)
            version = getattr(m, "__version__", "?")
            results.append((True, f"{import_name} ({version})"))
        except ImportError:
            results.append((False, f"{import_name} — NOT INSTALLED  →  pip install {pip_name}"))
    return results


def check_env_vars() -> list[tuple[bool, str, bool]]:
    """Returns list of (ok, message, required)."""
    results = []

    for var, hint in REQUIRED_ENV_VARS["broker"]:
        val = os.getenv(var, "")
        if val:
            results.append((True, f"{var}={val}", True))
        else:
            results.append((False, f"{var} not set  →  {hint}", True))

    paper_key = os.getenv("TT_PAPER_API_KEY", "")
    tt_secret  = os.getenv("TT_SECRET", "")
    tt_refresh = os.getenv("TT_REFRESH", "")

    if paper_key:
        results.append((True, f"TT_PAPER_API_KEY=***{paper_key[-4:]}", True))
    elif tt_secret and tt_refresh:
        results.append((True, "TT_SECRET + TT_REFRESH (OAuth) configured", True))
    else:
        results.append((
            False,
            "No broker auth found — set TT_PAPER_API_KEY (paper) "
            "or TT_SECRET + TT_REFRESH (OAuth/live)",
            True,
        ))

    tg_token = os.getenv("TG_BOT_TOKEN", "")
    tg_chat  = os.getenv("TG_CHAT_ID", "")
    gmail    = os.getenv("GMAIL_SENDER", "")

    if tg_token and tg_chat:
        results.append((True, f"Telegram configured (chat_id={tg_chat[:6]}...)", False))
    elif gmail:
        results.append((True, f"Gmail fallback configured ({gmail})", False))
    else:
        results.append((
            False,
            "No notification channel — set TG_BOT_TOKEN + TG_CHAT_ID (Telegram) "
            "or GMAIL_SENDER + GMAIL_PASSWORD",
            False,
        ))

    return results


def check_network() -> list[tuple[bool, str]]:
    results = []
    for host, label in INTERNET_CHECKS:
        try:
            urllib.request.urlopen(f"https://{host}", timeout=5)
            results.append((True, f"{label} ({host})"))
        except Exception as e:
            results.append((False, f"{label} — {e}"))
    return results


def check_timezone() -> tuple[bool, str]:
    tz = datetime.now().astimezone().tzname()
    now_et = datetime.now().strftime("%H:%M")
    is_eastern = "ET" in tz or "EST" in tz or "EDT" in tz or "Eastern" in tz
    msg = f"Server timezone: {tz}  (current local time: {now_et})"
    if not is_eastern:
        msg += "  ⚠️  Cron schedule assumes US/Eastern — run: sudo timedatectl set-timezone America/New_York"
    return is_eastern, msg


def send_test_notification() -> tuple[bool, str]:
    try:
        from notifications import notify
        notify(
            "✅ Health Check Passed",
            f"Trading pipeline is configured and ready.\n"
            f"Server time: {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}",
        )
        return True, "Test notification sent"
    except Exception as e:
        return False, f"Notification test failed: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def run(notify_test: bool = False, quiet: bool = False) -> bool:
    all_required_passed = True
    lines = []

    def section(title):
        lines.append(f"\n{'─' * 50}")
        lines.append(f"  {title}")
        lines.append('─' * 50)

    def row(ok: bool, msg: str, required: bool = True):
        nonlocal all_required_passed
        icon = "✅" if ok else ("❌" if required else "⚠️ ")
        lines.append(f"  {icon}  {msg}")
        if not ok and required:
            all_required_passed = False

    lines.append(f"\nTrading Pipeline Health Check — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    section("Python packages")
    for ok, msg in check_packages():
        row(ok, msg)

    section("Environment variables")
    for ok, msg, required in check_env_vars():
        row(ok, msg, required)

    section("Timezone")
    ok, msg = check_timezone()
    row(ok, msg, required=False)

    section("Network connectivity")
    for ok, msg in check_network():
        row(ok, msg, required=False)

    if notify_test:
        section("Notification test")
        ok, msg = send_test_notification()
        row(ok, msg, required=False)

    lines.append("")
    if all_required_passed:
        lines.append("  ✅  All required checks passed — pipeline is ready.")
    else:
        lines.append("  ❌  Some required checks FAILED — fix the issues above before going live.")
    lines.append("")

    if not quiet:
        print("\n".join(lines))

    return all_required_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading pipeline health check")
    parser.add_argument("--notify", action="store_true", help="Send a test Telegram/Gmail alert")
    parser.add_argument("--quiet",  action="store_true", help="Suppress output; use exit code only")
    args = parser.parse_args()

    ok = run(notify_test=args.notify, quiet=args.quiet)
    sys.exit(0 if ok else 1)
