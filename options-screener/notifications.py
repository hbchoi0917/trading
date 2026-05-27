"""
notifications.py — Unified alert channel for trading automation.

Supports two backends (configurable via .env):
  1. Telegram Bot  — preferred; instant push to phone, no spam filters
  2. Gmail SMTP    — fallback; requires App Password

Setup (Telegram — recommended):
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message your new bot, then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id (look for "id" inside "chat")
  3. Add to .env:
       TG_BOT_TOKEN=123456789:AABBccDDeeff...
       TG_CHAT_ID=987654321

Setup (Gmail — fallback):
  GMAIL_SENDER=you@gmail.com
  GMAIL_PASSWORD=xxxx xxxx xxxx xxxx   # App Password (16 chars)
  GMAIL_RECEIVER=you@gmail.com

Usage:
    from notifications import notify, notify_entry, notify_close, notify_error

    notify("Test", "Pipeline started successfully")
    notify_entry("NVDA", "put_credit", "2026-06-20", 180, 170, credit=1.45)
    notify_close("NVDA", "profit_target", pnl=116.0)
    notify_error("CRWD", "order rejected: insufficient buying power")
"""

import logging
import os
import smtplib
import ssl
import urllib.request
import urllib.parse
import urllib.error
import json
from datetime import datetime
from email.mime.text import MIMEText
from email.header import Header

logger = logging.getLogger(__name__)

# ── Config (loaded from env) ──────────────────────────────────────────────────

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "")

GMAIL_SENDER   = os.getenv("GMAIL_SENDER", "")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD", "")
GMAIL_RECEIVER = os.getenv("GMAIL_RECEIVER", "")


# ── Core send functions ───────────────────────────────────────────────────────

def _send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        return False
    try:
        url     = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id":    TG_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return False


def _send_gmail(subject: str, body: str) -> bool:
    """Send an email via Gmail SMTP SSL. Returns True on success."""
    if not GMAIL_SENDER or not GMAIL_PASSWORD or not GMAIL_RECEIVER:
        return False
    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From']    = GMAIL_SENDER
        msg['To']      = GMAIL_RECEIVER
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_SENDER, GMAIL_RECEIVER, msg.as_bytes())
        return True
    except Exception as e:
        logger.warning(f"Gmail send failed: {e}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def notify(subject: str, body: str) -> None:
    """
    Send an alert via Telegram (preferred) or Gmail (fallback).
    Falls back to console log if neither is configured.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    tg_message = f"<b>{subject}</b>\n{body}\n<i>{timestamp}</i>"

    if _send_telegram(tg_message):
        logger.info(f"[Telegram] Alert sent: {subject}")
        return

    if _send_gmail(subject, f"[{timestamp}]\n\n{body}"):
        logger.info(f"[Gmail] Alert sent: {subject}")
        return

    # Neither configured — print to console/log so cron captures it
    logger.warning(
        f"[ALERT — no notification channel configured]\n"
        f"  Subject : {subject}\n"
        f"  Body    : {body}"
    )


def notify_entry(
    ticker: str,
    spread_type: str,
    expiry: str,
    short_strike: float,
    long_strike: float,
    credit: float,
    quantity: int = 1,
    account: str = "",
    dry_run: bool = True,
) -> None:
    mode = "DRY-RUN" if dry_run else "LIVE"
    direction = "PUT" if spread_type == "put_credit" else "CALL"
    subject = f"[{mode}] ✅ NEW {direction} SPREAD — {ticker}"
    body = (
        f"Ticker   : {ticker}\n"
        f"Spread   : {direction} Credit Spread\n"
        f"Strikes  : ${short_strike:.0f} / ${long_strike:.0f}\n"
        f"Expiry   : {expiry}\n"
        f"Credit   : ${credit:.2f}/share  (${credit*100*quantity:.0f} total)\n"
        f"Qty      : {quantity} contract(s)\n"
        f"Account  : {account or 'all'}"
    )
    notify(subject, body)


def notify_close(
    ticker: str,
    trigger: str,
    pnl: float,
    account: str = "",
    dry_run: bool = True,
) -> None:
    mode   = "DRY-RUN" if dry_run else "LIVE"
    emoji  = "💰" if pnl >= 0 else "🛑"
    label  = trigger.replace("_", " ").upper()
    subject = f"[{mode}] {emoji} CLOSED — {ticker} ({label})"
    body = (
        f"Ticker   : {ticker}\n"
        f"Trigger  : {label}\n"
        f"P&L      : ${pnl:+,.2f}\n"
        f"Account  : {account or 'all'}"
    )
    notify(subject, body)


def notify_circuit_breaker(monthly_pnl: float, limit: float) -> None:
    subject = "🚨 CIRCUIT BREAKER — New entries paused"
    body = (
        f"Monthly P&L  : ${monthly_pnl:+,.2f}\n"
        f"Drawdown limit: ${limit:,.2f}\n\n"
        "New entry orders are suspended for the rest of this week.\n"
        "Review open positions and resume manually if conditions improve."
    )
    notify(subject, body)


def notify_error(ticker: str, error_msg: str, account: str = "") -> None:
    subject = f"⚠️ ORDER ERROR — {ticker}"
    body = (
        f"Ticker  : {ticker}\n"
        f"Account : {account or 'all'}\n"
        f"Error   : {error_msg}"
    )
    notify(subject, body)


def notify_monitor_summary(
    placed: int,
    skipped: int,
    closed: int,
    monthly_pnl: float,
) -> None:
    subject = f"📊 Daily Summary — {datetime.now().strftime('%Y-%m-%d')}"
    body = (
        f"New positions : {placed}\n"
        f"Skipped       : {skipped}\n"
        f"Auto-closed   : {closed}\n"
        f"MTD P&L       : ${monthly_pnl:+,.2f}"
    )
    notify(subject, body)


# ── Daily / Weekly summary ───────────────────────────────────────────────────

def notify_daily_summary(
    placed: int,
    closed: int,
    open_positions: list[dict],
    mtd_pnl: float,
    drawdown_limit: float = -2000.0,
    expiring_soon: list[dict] | None = None,
    cap_warnings: list[str] | None = None,
) -> None:
    """
    Send end-of-day portfolio summary (4:30 PM ET).

    open_positions : [{ticker, short_strike, long_strike, expiry, dte, contracts}]
    expiring_soon  : positions with DTE ≤ 9
    cap_warnings   : tickers at >80% of portfolio exposure cap
    """
    subject  = f"📊 Daily Summary — {datetime.now().strftime('%b %d')}"
    headroom = mtd_pnl - drawdown_limit   # drawdown_limit is negative (-2000)

    lines = [
        f"진입 {placed} | 클로즈 {closed} | 오픈 {len(open_positions)}개",
        f"MTD P&L: ${mtd_pnl:+,.0f}  (서킷브레이커까지 ${headroom:,.0f} 여유)",
    ]

    if open_positions:
        lines.append("")
        lines.append("오픈 포지션:")
        for p in open_positions:
            dte_str = f"{p['dte']} DTE" if p.get("dte") is not None else "? DTE"
            exp_str = p["expiry"][5:] if len(p.get("expiry", "")) >= 7 else p.get("expiry", "?")
            lines.append(
                f"  {p['ticker']:<6} ${p['short_strike']:.0f}/${p['long_strike']:.0f}P"
                f"  {exp_str}  {dte_str}  {p['contracts']}계약"
            )

    if expiring_soon:
        lines.append("")
        for p in expiring_soon:
            exp_str = p["expiry"][5:] if len(p.get("expiry", "")) >= 7 else p.get("expiry", "?")
            lines.append(f"⚠️  만기임박: {p['ticker']} {exp_str} ({p['dte']} DTE)")

    if cap_warnings:
        lines.append("")
        for w in cap_warnings:
            lines.append(f"⚠️  캡경고: {w}")

    notify(subject, "\n".join(lines))


def notify_weekly_summary(
    week_start: str,
    week_pnl: float,
    mtd_pnl: float,
    week_placed: int,
    week_closed: int,
    next_week_expiring: list[dict] | None = None,
    next_week_earnings: list[dict] | None = None,
    top_winner: dict | None = None,
    top_loser: dict | None = None,
) -> None:
    """
    Send end-of-week recap on the last trading day (4:30 PM ET).

    next_week_expiring : [{ticker, expiry, dte}]
    next_week_earnings : [{ticker, earnings_date, weekday}]
    top_winner/loser   : {ticker, pnl}
    """
    subject = f"📅 Weekly Summary — Week of {week_start}"
    lines = [
        f"주간 P&L: ${week_pnl:+,.0f}  |  MTD: ${mtd_pnl:+,.0f}",
        f"진입 {week_placed}건 | 클로즈 {week_closed}건",
    ]

    if next_week_expiring:
        lines.append("")
        lines.append("다음 주 만기:")
        for p in next_week_expiring:
            lines.append(
                f"  {p['ticker']}  {p['expiry']}  ({p['dte']} DTE) — BTC 검토"
            )

    if next_week_earnings:
        lines.append("")
        lines.append("다음 주 어닝 (스크리너 티커):")
        for e in next_week_earnings:
            lines.append(f"  {e['ticker']:<6} {e['earnings_date']}  ({e['weekday']})")
        lines.append("  → 어닝 당일 진입 여부 수동 판단")

    if top_winner or top_loser:
        lines.append("")
        if top_winner:
            lines.append(f"🏆 이번 주 최고: {top_winner['ticker']} ${top_winner['pnl']:+,.0f}")
        if top_loser:
            lines.append(f"📉 이번 주 최저: {top_loser['ticker']} ${top_loser['pnl']:+,.0f}")

    notify(subject, "\n".join(lines))


# ── Quick setup check ─────────────────────────────────────────────────────────

def check_setup() -> None:
    """Print which notification channels are configured."""
    print("Notification channel status:")
    if TG_BOT_TOKEN and TG_CHAT_ID:
        print(f"  ✅ Telegram  (chat_id={TG_CHAT_ID[:6]}...)")
    else:
        print("  ❌ Telegram  — set TG_BOT_TOKEN and TG_CHAT_ID in .env")
    if GMAIL_SENDER and GMAIL_PASSWORD:
        print(f"  ✅ Gmail     ({GMAIL_SENDER})")
    else:
        print("  ❌ Gmail     — set GMAIL_SENDER / GMAIL_PASSWORD in .env")


if __name__ == "__main__":
    check_setup()
    print()
    notify("Test Alert", "If you see this on Telegram/Gmail, notifications are working!")
    print("Test alert sent. Check your Telegram or email.")
