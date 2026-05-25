"""
One-time OAuth setup helper for Tastytrade (SDK v12+).

Tastytrade moved to OAuth in SDK v12. This script walks through:
  1. Getting your provider secret from https://developer.tastytrade.com
  2. Running the OAuth device flow to get a refresh token
  3. Saving TT_SECRET and TT_REFRESH to your .env file

Run once before using auto_trade.py:
    python broker/setup_auth.py

ALTERNATIVE — Paper Trading (no OAuth needed):
    1. Log in to Tastytrade → Settings → API → Paper Trading
    2. Generate an API key
    3. Add to .env:   TT_PAPER_API_KEY=your_key_here
                      TT_PAPER_TRADING=true
"""

import os
import sys
from pathlib import Path

ENV_FILE = Path(__file__).parent.parent / ".env"


def save_to_env(key: str, value: str) -> None:
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n")
    print(f"  Saved {key} to {ENV_FILE}")


def main():
    print("=" * 60)
    print("Tastytrade OAuth Setup")
    print("=" * 60)
    print()
    print("Choose your setup method:")
    print("  1. Paper Trading (API key) — simplest, recommended to start")
    print("  2. Live / Certification (OAuth) — requires developer registration")
    print()
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print()
        print("Paper Trading Setup:")
        print("  1. Log in to Tastytrade")
        print("  2. Go to Settings → API → Paper Trading")
        print("  3. Generate or copy your paper trading API key")
        print()
        api_key = input("Paste your paper trading API key: ").strip()
        if api_key:
            save_to_env("TT_PAPER_API_KEY", api_key)
            save_to_env("TT_PAPER_TRADING", "true")
            save_to_env("TT_DRY_RUN", "true")
            print()
            print("Done! To test the connection:")
            print("  python auto_trade.py monitor --dry-run")
        else:
            print("No API key entered. Exiting.")

    elif choice == "2":
        print()
        print("OAuth Setup (Live/Certification):")
        print("  1. Go to https://developer.tastytrade.com")
        print("  2. Register your app and get a provider secret")
        print("  3. Complete the OAuth device flow to get a refresh token")
        print()
        secret = input("Paste your TT_SECRET (provider secret): ").strip()
        refresh = input("Paste your TT_REFRESH (refresh token): ").strip()
        is_test = input("Use certification (sandbox) endpoint? [y/N]: ").strip().lower()

        if secret and refresh:
            save_to_env("TT_SECRET", secret)
            save_to_env("TT_REFRESH", refresh)
            save_to_env("TT_PAPER_TRADING", "true" if is_test == "y" else "false")
            save_to_env("TT_DRY_RUN", "true")
            print()
            print("Done! To test the connection:")
            print("  python auto_trade.py monitor --dry-run")
        else:
            print("Incomplete credentials. Exiting.")
    else:
        print("Invalid choice. Run again and enter 1 or 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
