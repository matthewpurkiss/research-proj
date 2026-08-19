"""
make_synthetic.py

Generates a fake fixture dataset with the same file layout fetch_data.py
produces from the real APIs, so every other script in the pipeline can be
built and run end to end without a single network call.

Not part of the research pipeline itself (see PHASE1_SPEC.md). Used by
run_pipeline.py --synthetic, and by test_audit_teeth.py, which mutates the
output of generate() to check that leakage_audit.py actually catches
known-bad data.
"""

import argparse
import os

import numpy as np
import pandas as pd

import config

SEED = 42

# Starting price/level for each series, chosen to be roughly the right
# order of magnitude for early 2018. The exact values do not matter, only
# that they are positive and give sane log-returns and log-volumes.
STARTING_LEVEL = {
    "BTC": 14000.0,
    "ETH": 750.0,
    "SP500": 2700.0,
    "GOLD": 1300.0,
    "DXY": 92.0,
    "VIX": 18.0,
}

STARTING_VOLUME = {
    "BTC": 8e9,
    "ETH": 3e9,
    "SP500": 3e9,
}

# Daily drift and volatility for the random-walk log-price, tuned so BTC
# and ETH come out meaningfully more volatile than the macro series,
# matching the qualitative picture in Chapter 2 of the proposal.
DAILY_DRIFT = {
    "BTC": 0.0006,
    "ETH": 0.0006,
    "SP500": 0.0003,
    "GOLD": 0.0001,
    "DXY": 0.0,
    "VIX": 0.0,
}
DAILY_VOL = {
    "BTC": 0.035,
    "ETH": 0.045,
    "SP500": 0.010,
    "GOLD": 0.008,
    "DXY": 0.004,
    "VIX": 0.06,
}

# On-chain metrics: modelled as exponential growth plus noise, since real
# hash rate, difficulty, and active addresses all grew substantially over
# 2018-2025. Mempool size and miner revenue are noisier and less trended.
CHAIN_START_LEVEL = {
    "active_addresses": 6.0e5,
    "hash_rate": 3.0e7,
    "transaction_count": 2.5e5,
    "mempool_size": 1.5e7,
    "miner_revenue": 1.5e7,
    "difficulty": 3.0e12,
}
CHAIN_DAILY_GROWTH = {
    "active_addresses": 0.0002,
    "hash_rate": 0.0012,
    "transaction_count": 0.0002,
    "mempool_size": 0.0,
    "miner_revenue": 0.0003,
    "difficulty": 0.0012,
}
CHAIN_DAILY_VOL = {
    "active_addresses": 0.03,
    "hash_rate": 0.02,
    "transaction_count": 0.04,
    "mempool_size": 0.15,
    "miner_revenue": 0.06,
    "difficulty": 0.005,
}

# Fraction of BTC trading days deliberately turned into an outsized move,
# so the jump-detection logic downstream has real jumps to find instead
# of a perfectly smooth series.
JUMP_INJECTION_RATE = 0.01
JUMP_SIZE_STD_MULTIPLIER = 6.0


def _random_walk(rng, n, start_level, daily_drift, daily_vol):
    """Geometric random walk in log space. Returns a positive-valued series."""
    log_returns = rng.normal(loc=daily_drift, scale=daily_vol, size=n)
    log_levels = np.log(start_level) + np.cumsum(log_returns)
    return np.exp(log_levels)


def _inject_jumps(rng, close, rate, std_multiplier):
    """Deliberately blows out a handful of days so jump detection has
    something real to catch. Rescales that day's log-return in place."""
    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    n_jumps = max(1, int(len(close) * rate))
    jump_days = rng.choice(len(close), size=n_jumps, replace=False)
    day_std = np.std(log_returns)
    for day in jump_days:
        sign = rng.choice([-1.0, 1.0])
        log_returns[day] = sign * std_multiplier * day_std
    log_levels = np.log(close[0]) + np.cumsum(log_returns)
    return np.exp(log_levels)


def _calendar_day_index():
    return pd.date_range(config.DATA_START, config.DATA_END, freq="D")


def _business_day_index():
    # Approximates US business days as plain Mon-Fri, with no holiday
    # calendar. Good enough for a fixture whose job is to exercise the
    # pipeline's code paths, not to reproduce the real trading calendar.
    return pd.date_range(config.DATA_START, config.DATA_END, freq="B")


def _make_yahoo_frame(rng, key):
    """Builds one synthetic Yahoo-style OHLCV frame for a single asset key.

    BTC and ETH trade every calendar day, matching real crypto markets.
    Everything else (S&P, Gold, DXY, VIX) trades only on business days,
    matching the real Yahoo feed's US session alignment (Section 3.3).
    """
    index = _calendar_day_index() if key in ("BTC", "ETH") else _business_day_index()
    n = len(index)

    close = _random_walk(rng, n, STARTING_LEVEL[key], DAILY_DRIFT[key], DAILY_VOL[key])
    if key == "BTC":
        close = _inject_jumps(rng, close, JUMP_INJECTION_RATE, JUMP_SIZE_STD_MULTIPLIER)

    fields = config.YAHOO_FIELDS[key]
    frame = pd.DataFrame(index=index)
    frame.index.name = "Date"

    if "Open" in fields or "High" in fields or "Low" in fields:
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        intraday_noise = rng.uniform(0.001, 0.02, size=n)
        frame["Open"] = prev_close * (1 + rng.normal(0, 0.003, size=n))
        frame["High"] = np.maximum(frame["Open"], close) * (1 + intraday_noise)
        frame["Low"] = np.minimum(frame["Open"], close) * (1 - intraday_noise)
    frame["Close"] = close

    if "Volume" in fields:
        base_volume = STARTING_VOLUME.get(key, 1e9)
        frame["Volume"] = base_volume * rng.lognormal(mean=0.0, sigma=0.3, size=n)

    return frame[fields]


def _make_chain_frame(rng, key):
    """Builds one synthetic Blockchain.com-style chart frame. Observed on
    every calendar day, including holidays, matching Section 3.3."""
    index = _calendar_day_index()
    n = len(index)
    value = _random_walk(
        rng, n, CHAIN_START_LEVEL[key], CHAIN_DAILY_GROWTH[key], CHAIN_DAILY_VOL[key]
    )
    frame = pd.DataFrame({"Date": index, "value": value}).set_index("Date")
    return frame


def generate(output_dir, seed=SEED):
    """Writes one CSV per configured data source into output_dir, using
    the same file layout fetch_data.py produces from the real APIs.

    Returns the list of file paths written.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    written = []

    for key in config.YAHOO_TICKERS:
        frame = _make_yahoo_frame(rng, key)
        path = os.path.join(output_dir, f"yahoo_{key}.csv")
        frame.to_csv(path)
        written.append(path)

    for key in config.CHAIN_CHARTS:
        frame = _make_chain_frame(rng, key)
        path = os.path.join(output_dir, f"chain_{key}.csv")
        frame.to_csv(path)
        written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "raw"),
        help="Directory to write the synthetic CSV fixtures into.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    paths = generate(args.output_dir, seed=args.seed)
    print(f"Wrote {len(paths)} synthetic fixture files to {args.output_dir}:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
