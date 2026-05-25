"""
Tastytrade session management and account helpers (SDK v12+).

Authentication (two modes):

  LIVE / CERTIFICATION (OAuth):
    1. Register your app at https://developer.tastytrade.com
    2. Complete the OAuth flow to obtain a refresh token:
         python broker/setup_auth.py
    3. Set env vars in .env:
         TT_SECRET=<your-provider-secret>
         TT_REFRESH=<refresh-token>
    4. Set TT_PAPER_TRADING=true to use the certification (sandbox) endpoint.

  PAPER TRADING (API key):
    1. In your Tastytrade app, create a paper trading account
    2. Generate an API key from Settings → API → Paper Trading
    3. Set env vars in .env:
         TT_PAPER_API_KEY=<your-api-key>
         TT_PAPER_TRADING=true

Environment variables:
    TT_SECRET            OAuth provider secret (live/cert)
    TT_REFRESH           OAuth refresh token (live/cert)
    TT_PAPER_TRADING     "true" to use paper endpoint
    TT_PAPER_API_KEY     API key for paper trading (simplest setup)
"""

import asyncio
import os
import logging
from decimal import Decimal
from typing import Optional

from tastytrade.session import Session
from tastytrade.paper import PaperSession
from tastytrade.account import Account, CurrentPosition
from tastytrade.order import InstrumentType

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TastyClient:
    """Wrapper around a Tastytrade session with helpers for multi-account ops."""

    def __init__(self):
        self._session: Optional[Session | PaperSession] = None
        self._accounts: list[Account] = []

    @property
    def session(self) -> Session | PaperSession:
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._session

    async def connect(self) -> None:
        paper      = os.environ.get("TT_PAPER_TRADING", "false").lower() == "true"
        paper_key  = os.environ.get("TT_PAPER_API_KEY", "")

        if paper and paper_key:
            # Simplest setup: paper trading with API key
            self._session = PaperSession(api_key=paper_key)
            mode = "paper (API key)"
        else:
            # OAuth-based session (live or certification)
            is_test = paper
            self._session = Session(is_test=is_test)
            mode = f"{'certification' if is_test else 'live'} (OAuth)"

        async with self._session:
            accounts = await Account.get(self._session)
            self._accounts = accounts if isinstance(accounts, list) else [accounts]

        logger.info(f"Connected to Tastytrade ({mode}): {len(self._accounts)} account(s)")
        for acct in self._accounts:
            logger.info(f"  {acct.account_number}  {acct.account_type_name}  {acct.nickname}")

    async def disconnect(self) -> None:
        self._session = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.disconnect()

    # ── Account helpers ──────────────────────────────────────────────────────

    def get_account(self, account_number: str) -> Account:
        for acct in self._accounts:
            if acct.account_number == account_number:
                return acct
        raise ValueError(f"Account {account_number} not found")

    def list_accounts(self) -> list[dict]:
        return [
            {
                "account_number": a.account_number,
                "type":           a.account_type_name,
                "nickname":       a.nickname,
                "margin_or_cash": a.margin_or_cash,
            }
            for a in self._accounts
        ]

    async def get_balances(self, account_number: str):
        acct = self.get_account(account_number)
        async with self._session:
            return await acct.get_balances(self._session)

    async def get_positions(
        self,
        account_number: str,
        underlying: Optional[str] = None,
    ) -> list[CurrentPosition]:
        acct = self.get_account(account_number)
        kwargs = {}
        if underlying:
            kwargs["underlying_symbols"] = [underlying]
        async with self._session:
            return await acct.get_positions(
                self._session,
                instrument_type=InstrumentType.EQUITY_OPTION,
                **kwargs,
            )

    async def get_live_orders(self, account_number: str):
        acct = self.get_account(account_number)
        async with self._session:
            return await acct.get_live_orders(self._session)

    async def get_all_positions(self) -> dict[str, list[CurrentPosition]]:
        result = {}
        async with self._session:
            for acct in self._accounts:
                result[acct.account_number] = await acct.get_positions(
                    self._session,
                    instrument_type=InstrumentType.EQUITY_OPTION,
                )
        return result

    async def get_net_liquidating_value(self, account_number: str) -> Decimal:
        bal = await self.get_balances(account_number)
        return bal.net_liquidating_value

    async def get_option_buying_power(self, account_number: str) -> Decimal:
        bal = await self.get_balances(account_number)
        return bal.derivative_buying_power


def run_sync(coro):
    """Run a coroutine synchronously (for scripts/cron jobs)."""
    return asyncio.get_event_loop().run_until_complete(coro)
