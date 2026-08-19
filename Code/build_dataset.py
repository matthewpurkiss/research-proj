"""
build_dataset.py

Reads the twelve raw CSVs from data/raw/, aligns them onto a single
business-day calendar, engineers the 19-dimensional feature vector X_t
plus the three prediction targets, trims the warm-up rows that don't yet
have full rolling-window history, and writes one aligned, feature-complete
CSV to data/processed/ for splits.py to slice into train/val/test.

This file does not scale anything and does not split anything. The
scaler has to be fit on the training slice only, which doesn't exist yet
at this stage - see PHASE1_SPEC.md's file layout for why that is
splits.py's job instead.
"""

import os

import numpy as np
import pandas as pd

import config

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "aligned_dataset.csv")

# Ceiling on how many missing rows in a single on-chain column get
# forward-filled before _align() refuses and fails loudly instead. Real
# fetches have shown gaps of 3-6 rows; this is deliberately far above
# that so a genuine fetch problem (hundreds of missing rows) still trips
# the guard rather than getting silently papered over.
MAX_CHAIN_GAP_ROWS = 20


def _print_synthetic_banner_if_present(raw_dir):
    marker_path = os.path.join(raw_dir, config.SYNTHETIC_MARKER_FILE)
    if os.path.exists(marker_path):
        banner = "SYNTHETIC DATA IN USE - DO NOT REPORT THESE NUMBERS AS RESEARCH RESULTS"
        print("=" * len(banner))
        print(banner)
        print("=" * len(banner))


def _load_yahoo(raw_dir, key):
    path = os.path.join(raw_dir, f"yahoo_{key}.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    if "Volume" in df.columns:
        # A reported volume of exactly 0 is a data error, not a real
        # trading day, for any of these actively-traded assets - confirmed
        # one such case (SP500, 2023-05-24) that fed straight into
        # log(0) = -inf downstream. Treated as missing so it gets the same
        # forward-fill treatment as a holiday gap.
        df.loc[df["Volume"] <= 0, "Volume"] = np.nan
    return df


def _load_chain(raw_dir, key):
    path = os.path.join(raw_dir, f"chain_{key}.csv")
    return pd.read_csv(path, index_col="Date", parse_dates=True)["value"]


def _forward_then_backfill_leading_gap(df, max_leading_gap=3):
    """Forward-fills holiday gaps as usual, then backfills whatever NaN
    is still left over. That remainder is only ever the very first row -
    2018-01-01 is a real New Year's Day market holiday, and it also
    happens to be the first day of the whole dataset, so there is no
    prior value for ffill to reach back to. Capped well above that one
    expected day so a genuinely large gap still fails loudly instead of
    getting silently patched."""
    filled = df.ffill()
    n_leading = int(filled.isna().any(axis=1).sum())
    if n_leading > max_leading_gap:
        raise ValueError(
            f"Unexpectedly large leading gap ({n_leading} rows) after "
            "forward-fill - expected at most a New Year's Day holiday gap."
        )
    return filled.bfill()


def _align(raw_dir):
    """Section 3.3: merge onto one business-day calendar. BTC, ETH, and
    the on-chain metrics update every calendar day, so aligning them onto
    business days just drops their weekend rows. The macro series (S&P,
    Gold, DXY, VIX) already skip weekends but have holiday gaps, which
    get forward-filled from the previous close.

    On-chain metrics were assumed fully observed by the proposal (Section
    3.3), but real fetches have shown small, persistent gaps - a shared
    3-day outage across four metrics in Nov 2025, plus three isolated
    single-day gaps for active_addresses in 2019-2020 (one of them the
    2020-03-13 COVID crash day). Confirmed via a second fetch that these
    are not transient. Forward-filled the same way the macro assets are,
    since the alternative (silently dropping those days pipeline-wide)
    loses more information for no benefit. Still capped below: a gap
    bigger than MAX_CHAIN_GAP_ROWS fails loudly instead of being
    forward-filled, since that would signal a real fetch problem rather
    than a handful of genuinely missing days.
    """
    business_days = pd.date_range(config.DATA_START, config.DATA_END, freq="B")

    btc = _load_yahoo(raw_dir, "BTC").reindex(business_days)
    eth = _load_yahoo(raw_dir, "ETH").reindex(business_days)
    sp500 = _forward_then_backfill_leading_gap(_load_yahoo(raw_dir, "SP500").reindex(business_days))
    gold = _forward_then_backfill_leading_gap(_load_yahoo(raw_dir, "GOLD").reindex(business_days))
    dxy = _forward_then_backfill_leading_gap(_load_yahoo(raw_dir, "DXY").reindex(business_days))
    vix = _forward_then_backfill_leading_gap(_load_yahoo(raw_dir, "VIX").reindex(business_days))

    aligned = pd.DataFrame(index=business_days)
    aligned.index.name = "Date"
    aligned["btc_close"] = btc["Close"]
    aligned["btc_volume"] = btc["Volume"]
    aligned["eth_close"] = eth["Close"]
    aligned["eth_volume"] = eth["Volume"]
    aligned["sp500_close"] = sp500["Close"]
    aligned["sp500_volume"] = sp500["Volume"]
    aligned["gold_close"] = gold["Close"]
    aligned["dxy_close"] = dxy["Close"]
    aligned["vix_close"] = vix["Close"]

    chain_columns = []
    for key in config.CHAIN_CHARTS:
        column = f"chain_{key}"
        aligned[column] = _load_chain(raw_dir, key).reindex(business_days)
        chain_columns.append(column)

    missing = aligned[chain_columns].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        oversized = missing[missing > MAX_CHAIN_GAP_ROWS]
        if not oversized.empty:
            raise ValueError(
                "On-chain columns have gaps larger than "
                f"MAX_CHAIN_GAP_ROWS={MAX_CHAIN_GAP_ROWS}, which looks like a "
                f"fetch problem rather than a handful of missing days:\n{oversized}\n"
                "Re-fetch, or raise MAX_CHAIN_GAP_ROWS deliberately if this is "
                "confirmed to be real."
            )
        print(f"Forward-filling {int(missing.sum())} on-chain gap day(s): {dict(missing)}")
        aligned[chain_columns] = aligned[chain_columns].ffill()

    return aligned


def _log_return(series):
    return np.log(series / series.shift(1))


def _engineer_features(aligned):
    """Section 3.4: builds the 19 columns in config.FEATURE_ORDER. Every
    rolling calculation here is trailing (ends at row t), never centred -
    a centred window would let information from after t leak into row
    t's features (leakage check 2)."""
    df = pd.DataFrame(index=aligned.index)

    df["btc_log_return"] = _log_return(aligned["btc_close"])
    df["eth_log_return"] = _log_return(aligned["eth_close"])
    df["sp500_log_return"] = _log_return(aligned["sp500_close"])
    df["gold_log_return"] = _log_return(aligned["gold_close"])
    df["dxy_log_return"] = _log_return(aligned["dxy_close"])
    df["vix_log_return"] = _log_return(aligned["vix_close"])

    df["btc_realised_vol_14d"] = (
        df["btc_log_return"].rolling(config.REALISED_VOL_WINDOW).std()
    )

    df["corr_btc_eth_30d"] = (
        df["btc_log_return"].rolling(config.CORRELATION_WINDOW).corr(df["eth_log_return"])
    )
    df["corr_btc_sp500_30d"] = (
        df["btc_log_return"].rolling(config.CORRELATION_WINDOW).corr(df["sp500_log_return"])
    )
    df["corr_btc_gold_30d"] = (
        df["btc_log_return"].rolling(config.CORRELATION_WINDOW).corr(df["gold_log_return"])
    )

    for key in config.CHAIN_CHARTS:
        df[f"{key}_roc_7d"] = aligned[f"chain_{key}"].pct_change(config.ONCHAIN_ROC_WINDOW)

    df["btc_volume_log"] = np.log(aligned["btc_volume"])
    df["eth_volume_log"] = np.log(aligned["eth_volume"])
    df["sp500_volume_log"] = np.log(aligned["sp500_volume"])

    return df[config.FEATURE_ORDER]


def _compute_targets(features):
    """Section 5.1-5.3: both agents predict tomorrow's value from today's
    feature row, so every target here is shifted one step into the future
    relative to its feature row (leakage check 4) - the input already
    contains today's log-return and today's realised vol, so a target
    built from today's value would just be handing the model its own
    input back as the answer.

    The jump label compares today's move against a threshold computed
    only from the 30 days strictly BEFORE today (leakage check 3):
    shift(1) on the rolling std is what excludes today's own return from
    the distribution used to judge today, so an unusually large move
    can't inflate the very threshold meant to catch it.
    """
    targets = pd.DataFrame(index=features.index)

    targets["alpha_target"] = features["btc_log_return"].shift(-1)
    targets["beta_target_vol"] = features["btc_realised_vol_14d"].shift(-1)

    trailing_std = (
        features["btc_log_return"]
        .rolling(config.JUMP_THRESHOLD_WINDOW)
        .std()
        .shift(1)
    )
    jump_threshold = config.JUMP_THRESHOLD_STD * trailing_std
    is_jump = (features["btc_log_return"].abs() > jump_threshold).astype(float)
    is_jump[trailing_std.isna()] = np.nan
    targets["beta_target_jump"] = is_jump.shift(-1)

    return targets


def build(raw_dir=RAW_DIR, output_path=OUTPUT_PATH):
    _print_synthetic_banner_if_present(raw_dir)

    aligned = _align(raw_dir)
    features = _engineer_features(aligned)
    targets = _compute_targets(features)

    assert list(features.columns) == config.FEATURE_ORDER, (
        "Feature columns drifted from config.FEATURE_ORDER - fix the "
        "mismatch here, do not relax the pinned order in config.py"
    )
    assert features.shape[1] == config.N_FEATURES, (
        f"Built {features.shape[1]} features, expected d={config.N_FEATURES}"
    )

    combined = pd.concat([features, targets], axis=1)

    # The rolling windows above need history before they produce a real
    # value. Every window used is <= WARMUP_TRIM_DAYS long, computed on
    # the full untrimmed series above, so trimming here (rather than
    # before computing features) is what avoids PHASE1_SPEC.md's "check
    # 6" trap: recomputing a rolling threshold on an already-trimmed
    # series just manufactures a fresh batch of NaNs at the new start.
    before_trim = len(combined)
    combined = combined.iloc[config.WARMUP_TRIM_DAYS :]
    after_warmup_trim = len(combined)

    combined = combined.dropna()
    after_final_trim = len(combined)

    dropped_for_targets = after_warmup_trim - after_final_trim
    if dropped_for_targets > 1:
        raise AssertionError(
            f"Expected to drop exactly 1 trailing row with no forward "
            f"target (the last day has no 'tomorrow'), but dropped "
            f"{dropped_for_targets} - a NaN slipped through the warm-up "
            f"trim somewhere upstream."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path)

    print(
        f"{before_trim} aligned rows -> {after_warmup_trim} after "
        f"{config.WARMUP_TRIM_DAYS}-day warm-up trim -> {after_final_trim} "
        f"after dropping the row with no forward target."
    )
    print(f"Wrote {output_path}")
    return combined


if __name__ == "__main__":
    build()
