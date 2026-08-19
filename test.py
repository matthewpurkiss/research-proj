"""
test_blockchain.py
Single-purpose test: confirm Blockchain.com Charts API is reachable
and returns usable on-chain data. No login, no API key required.

Usage: python3 test_blockchain.py
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import json
from datetime import datetime

URL = "https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json"

print(f"Requesting: {URL}\n")

try:
    r = requests.get(URL, timeout=15)
    print(f"HTTP status:    {r.status_code}")
    print(f"Content-Type:   {r.headers.get('Content-Type', 'unknown')}")
    print(f"Response size:  {len(r.content)} bytes\n")

    r.raise_for_status()
    data = r.json()

    name   = data.get("name", "?")
    unit   = data.get("unit", "?")
    period = data.get("period", "?")
    values = data.get("values", [])

    print(f"Chart name:  {name}")
    print(f"Unit:        {unit}")
    print(f"Period:      {period}")
    print(f"Data points: {len(values)}\n")

    if values:
        print("First 3 points:")
        for p in values[:3]:
            ts = datetime.utcfromtimestamp(p["x"]).strftime("%Y-%m-%d")
            print(f"  {ts}   {p['y']:,.2f}")
        print("\nLast 3 points:")
        for p in values[-3:]:
            ts = datetime.utcfromtimestamp(p["x"]).strftime("%Y-%m-%d")
            print(f"  {ts}   {p['y']:,.2f}")
        print("\n[SUCCESS] Blockchain.com API is reachable and returning data.")
    else:
        print("[WARN] Endpoint responded but returned no values.")

except requests.exceptions.Timeout:
    print("[FAIL] Request timed out (>15s).")
    print("  Likely cause: firewall is silently dropping the connection.")
except requests.exceptions.ConnectionError as e:
    print(f"[FAIL] Connection error: {e}")
    print("  Likely cause: DNS block or network refusing the connection.")
except requests.exceptions.HTTPError as e:
    print(f"[FAIL] HTTP error: {e}")
    print("  The server responded but with an error code.")
except json.JSONDecodeError:
    print("[FAIL] Response was not valid JSON.")
    print("  Likely cause: a captive-portal or proxy is returning an HTML page.")
    print(f"  First 200 chars of response:\n  {r.text[:200]}")
except Exception as e:
    print(f"[FAIL] Unexpected error: {type(e).__name__}: {e}")