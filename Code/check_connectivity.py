"""
check_connectivity.py

Pre-flight check before spending mobile hotspot data on a real fetch.
Confirms every one of the twelve series in the project is reachable and
returning usable data. Downloads only a short recent window per series,
so the whole thing costs a few hundred kilobytes.

Extends the original single-endpoint test to cover all six Blockchain.com
charts and all six Yahoo tickers, because the sources do not always fail
together and a partial block is the confusing case.

Usage:
    python check_connectivity.py
    python check_connectivity.py --yahoo-only
    python check_connectivity.py --chain-only

Exit code 0 means every series is reachable. Non-zero means at least one
failed, and the summary table says which.
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import requests

CHAIN_BASE = "https://api.blockchain.info/charts"

# Blockchain.com chart slugs, matching the six on-chain metrics in Chapter 3.
CHAIN_CHARTS = {
    "active addresses": "n-unique-addresses",
    "hash rate": "hash-rate",
    "transaction count": "n-transactions",
    "mempool size": "mempool-size",
    "miner revenue": "miners-revenue",
    "difficulty": "difficulty",
}

# Yahoo Finance tickers, matching the six market series in Chapter 3.
YAHOO_TICKERS = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "S&P 500": "^GSPC",
    "Gold": "GC=F",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}

TIMEOUT = 20
results = []


def record(source, label, ok, detail):
    results.append((source, label, ok, detail))
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {label:<20} {detail}")


def check_chain():
    print(f"\nBlockchain.com Charts API  ({CHAIN_BASE})")
    print("-" * 72)
    for label, slug in CHAIN_CHARTS.items():
        url = f"{CHAIN_BASE}/{slug}?timespan=30days&format=json"
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            values = data.get("values", [])
            if not values:
                record("chain", label, False, "responded but returned no values")
                continue
            last = datetime.fromtimestamp(values[-1]["x"], tz=timezone.utc)
            unit = data.get("unit", "?")
            record(
                "chain", label, True,
                f"{len(values):>4} points, latest {last:%Y-%m-%d}, unit {unit}",
            )
        except requests.exceptions.Timeout:
            record("chain", label, False,
                   f"timed out after {TIMEOUT}s, firewall likely dropping silently")
        except requests.exceptions.ConnectionError:
            record("chain", label, False, "connection refused or DNS blocked")
        except requests.exceptions.HTTPError as e:
            record("chain", label, False, f"HTTP error {e.response.status_code}")
        except json.JSONDecodeError:
            snippet = r.text[:60].replace("\n", " ")
            record("chain", label, False,
                   f"not JSON, likely a captive portal: {snippet!r}")
        except Exception as e:
            record("chain", label, False, f"{type(e).__name__}: {e}")


def check_yahoo():
    print("\nYahoo Finance  (via yfinance)")
    print("-" * 72)
    try:
        import yfinance as yf
    except ImportError:
        record("yahoo", "yfinance", False,
               "package not installed, run: pip install yfinance")
        return

    for label, ticker in YAHOO_TICKERS.items():
        try:
            df = yf.download(
                ticker, period="1mo", interval="1d",
                progress=False, auto_adjust=False,
            )
            if df is None or df.empty:
                record("yahoo", label, False,
                       f"{ticker} returned an empty frame")
                continue
            first, last = df.index[0], df.index[-1]
            record(
                "yahoo", label, True,
                f"{len(df):>4} rows, {first:%Y-%m-%d} to {last:%Y-%m-%d}",
            )
        except Exception as e:
            record("yahoo", label, False, f"{type(e).__name__}: {e}")


def summarise():
    print("\n" + "=" * 72)
    failed = [r for r in results if not r[2]]
    chain_failed = [r for r in failed if r[0] == "chain"]
    yahoo_failed = [r for r in failed if r[0] == "yahoo"]

    print(f"{len(results) - len(failed)} of {len(results)} series reachable.")

    if not failed:
        print("\nAll clear. Safe to run: python run_pipeline.py --fetch")
        return 0

    print("\nFailures:")
    for _, label, _, detail in failed:
        print(f"  {label:<20} {detail}")

    if len(chain_failed) == len(CHAIN_CHARTS) and not yahoo_failed:
        print("\nEvery Blockchain.com series failed while Yahoo succeeded.")
        print("That is the university network block. Switch to your mobile")
        print("hotspot and run this again.")
    elif chain_failed and len(chain_failed) < len(CHAIN_CHARTS):
        print("\nSome Blockchain.com charts worked and others did not, which is")
        print("not a firewall pattern. Check the failing slugs against")
        print("https://www.blockchain.com/explorer/charts for renames.")

    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--yahoo-only", action="store_true")
    g.add_argument("--chain-only", action="store_true")
    args = p.parse_args()

    print("=" * 72)
    print("Connectivity pre-flight check")
    print(f"Run at {datetime.now():%Y-%m-%d %H:%M:%S} local")
    print("=" * 72)

    if not args.yahoo_only:
        check_chain()
    if not args.chain_only:
        check_yahoo()

    sys.exit(summarise())


if __name__ == "__main__":
    main()
