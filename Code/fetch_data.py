"""
fetch_data.py

Pulls the twelve raw data series defined in config.py from Yahoo Finance
and the Blockchain.com Charts API, and writes one CSV per series to
data/raw/, in the same file layout make_synthetic.py uses for the test
fixtures (yahoo_{KEY}.csv, chain_{key}.csv).

Raw fetched CSVs are treated as immutable once written (CLAUDE.md). This
script refuses to overwrite any existing target file unless --force is
passed.

The university network blocks api.blockchain.info (see CLAUDE.md). Run
check_connectivity.py first, and fetch the Blockchain.com series
separately (--chain-only) on a mobile hotspot.

Usage:
    python fetch_data.py                  # fetch everything
    python fetch_data.py --yahoo-only
    python fetch_data.py --chain-only
    python fetch_data.py --force          # overwrite existing raw CSVs
"""

import argparse
import os
import sys

import pandas as pd
import requests

import config

TIMEOUT = 30


def _yahoo_path(output_dir, key):
    return os.path.join(output_dir, f"yahoo_{key}.csv")


def _chain_path(output_dir, key):
    return os.path.join(output_dir, f"chain_{key}.csv")


def _existing_targets(output_dir, keys, path_fn):
    return [
        path_fn(output_dir, key) for key in keys if os.path.exists(path_fn(output_dir, key))
    ]


def _guard_overwrite(output_dir, do_yahoo, do_chain, force):
    """Raw CSVs are immutable once written. Refuse to touch any file that
    already exists unless the caller explicitly passes --force."""
    if force:
        return
    existing = []
    if do_yahoo:
        existing += _existing_targets(output_dir, config.YAHOO_TICKERS, _yahoo_path)
    if do_chain:
        existing += _existing_targets(output_dir, config.CHAIN_CHARTS, _chain_path)
    if existing:
        print("Refusing to overwrite existing raw data files:")
        for path in existing:
            print(f"  {path}")
        print("\nRaw CSVs are treated as immutable once written. Pass --force to overwrite.")
        raise SystemExit(1)


def _fetch_yahoo_series(key, ticker):
    import yfinance as yf

    df = yf.download(
        ticker,
        start=config.DATA_START,
        # yfinance's `end` is exclusive of the date itself - passing
        # DATA_END directly silently drops the last day of the range
        # (confirmed: every series ended 2025-12-30, one day short).
        # Pushing it forward one day is what actually includes DATA_END.
        end=(pd.Timestamp(config.DATA_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"{ticker} returned an empty frame")

    # yfinance returns MultiIndex columns (Price, Ticker) even for a
    # single ticker on this version. Drop the ticker level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    frame = df[config.YAHOO_FIELDS[key]].copy()
    frame.index.name = "Date"
    return frame


def _fetch_chain_series(key, slug):
    # sampled=false is required. Blockchain.com's default behaviour
    # downsamples long timespans for chart rendering (confirmed: without
    # it, an 8-year "all" request for hash-rate returned 729 points,
    # roughly one every 4 days, instead of one per day).
    url = f"{config.CHAIN_BASE_URL}/{slug}?timespan=all&format=json&sampled=false"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    values = payload.get("values", [])
    if not values:
        raise ValueError(f"{slug} responded but returned no values")

    frame = pd.DataFrame(values)
    frame["Date"] = pd.to_datetime(frame["x"], unit="s").dt.normalize()
    frame = frame.rename(columns={"y": "value"}).set_index("Date")[["value"]]
    frame = frame.loc[config.DATA_START : config.DATA_END]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise ValueError(f"{slug} returned no values within {config.DATA_START}..{config.DATA_END}")
    return frame


def _describe(frame):
    return (
        f"{len(frame):>4} rows, {frame.index.min():%Y-%m-%d} to "
        f"{frame.index.max():%Y-%m-%d}, {int(frame.isna().sum().sum())} NaNs"
    )


def _print_coverage_report(report):
    print("\n" + "=" * 72)
    print("Coverage report")
    print("=" * 72)
    for label, ok, detail in report:
        flag = "ok  " if ok else "FAIL"
        print(f"  [{flag}] {label:<20} {detail}")
    n_failed = sum(1 for _, ok, _ in report if not ok)
    print(f"\n{len(report) - n_failed} of {len(report)} series fetched successfully.")


def fetch(output_dir, do_yahoo=True, do_chain=True, force=False):
    """Fetches the configured series and writes one CSV per series to
    output_dir. Returns True if every requested series succeeded."""
    _guard_overwrite(output_dir, do_yahoo, do_chain, force)
    os.makedirs(output_dir, exist_ok=True)

    report = []

    if do_yahoo:
        for key, ticker in config.YAHOO_TICKERS.items():
            try:
                frame = _fetch_yahoo_series(key, ticker)
                frame.to_csv(_yahoo_path(output_dir, key))
                report.append((f"yahoo:{key}", True, _describe(frame)))
            except Exception as e:
                report.append((f"yahoo:{key}", False, f"{type(e).__name__}: {e}"))

    if do_chain:
        for key, slug in config.CHAIN_CHARTS.items():
            try:
                frame = _fetch_chain_series(key, slug)
                frame.to_csv(_chain_path(output_dir, key))
                report.append((f"chain:{key}", True, _describe(frame)))
            except Exception as e:
                report.append((f"chain:{key}", False, f"{type(e).__name__}: {e}"))

    _print_coverage_report(report)
    return all(ok for _, ok, _ in report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--yahoo-only", action="store_true")
    group.add_argument("--chain-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw CSVs.")
    parser.add_argument("--output-dir", default=os.path.join("data", "raw"))
    args = parser.parse_args()

    do_yahoo = not args.chain_only
    do_chain = not args.yahoo_only

    ok = fetch(args.output_dir, do_yahoo=do_yahoo, do_chain=do_chain, force=args.force)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
