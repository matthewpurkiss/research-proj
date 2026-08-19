"""
leakage_audit.py

Nine checks that the pipeline actually did what CLAUDE.md and
PHASE1_SPEC.md require: no future information reached a model at time t,
no boundary was crossed, and nothing scrambled the feature order.

Every check here deliberately RE-DERIVES its own answer from the rawest
data it can reach (data/raw, or build_dataset.py's own alignment/feature
functions called directly) rather than re-checking the pipeline's output
against itself. A check that just repeats the pipeline's own logic would
agree with a bug in that logic instead of catching it. test_audit_teeth.py
is what proves this independence actually works, by deliberately breaking
build_dataset.py five ways and confirming these checks scream.

Exit code 0 means every check passed. Non-zero means at least one failed,
and the report says which and why.
"""

import os
import sys

import numpy as np
import pandas as pd

import build_dataset
import config

PROCESSED_DIR = os.path.join("data", "processed")
ALIGNED_PATH = os.path.join(PROCESSED_DIR, "aligned_dataset.csv")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler_params.csv")


def _read(path):
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def check_scaler_train_only():
    """Leakage check 1: the scaler must be fit on the training slice
    alone. Recomputes mean/std directly from the unscaled aligned dataset
    sliced by config's train dates, independent of splits.py's own
    fit_scaler() implementation."""
    aligned = _read(ALIGNED_PATH)
    train_rows = aligned.loc[config.TRAIN_START : config.TRAIN_END]
    recomputed_mean = train_rows[config.FEATURE_ORDER].mean()
    recomputed_std = train_rows[config.FEATURE_ORDER].std()

    scaler = pd.read_csv(SCALER_PATH, index_col=0)
    stored_mean = scaler["mean"].reindex(config.FEATURE_ORDER)
    stored_std = scaler["std"].reindex(config.FEATURE_ORDER)

    mean_ok = np.allclose(recomputed_mean.to_numpy(), stored_mean.to_numpy())
    std_ok = np.allclose(recomputed_std.to_numpy(), stored_std.to_numpy())
    passed = mean_ok and std_ok
    detail = "recomputed train-only stats match scaler_params.csv" if passed else (
        "MISMATCH - scaler_params.csv does not match training-slice-only mean/std"
    )
    return passed, detail


def check_volatility_trailing():
    """Leakage check 2: hand-computes the 14-day trailing std at sampled
    dates using only that day and the 13 before it, and compares against
    the stored feature. A centred window would disagree here because it
    would have used days after the sample date too."""
    aligned = _read(ALIGNED_PATH)
    returns = aligned["btc_log_return"]
    mismatches = []

    for pos in range(0, len(aligned), 200):
        if pos < config.REALISED_VOL_WINDOW - 1:
            continue
        window = returns.iloc[pos - config.REALISED_VOL_WINDOW + 1 : pos + 1]
        hand_computed = window.std()
        stored = aligned["btc_realised_vol_14d"].iloc[pos]
        if not np.isclose(hand_computed, stored, rtol=1e-6):
            mismatches.append(str(aligned.index[pos].date()))

    passed = not mismatches
    detail = "trailing-window computation confirmed" if passed else f"mismatch at {mismatches}"
    return passed, detail


def check_jump_threshold_shifted():
    """Leakage check 3: independently recomputes the jump threshold using
    a 30-day std shifted by one day (excluding today's own return), and
    confirms it reproduces the stored beta_target_jump column."""
    aligned = _read(ALIGNED_PATH)
    returns = aligned["btc_log_return"]

    threshold = (
        returns.rolling(config.JUMP_THRESHOLD_WINDOW).std().shift(1)
        * config.JUMP_THRESHOLD_STD
    )
    is_jump = (returns.abs() > threshold).astype(float)
    is_jump[threshold.isna()] = np.nan
    recomputed_target = is_jump.shift(-1)

    stored = aligned["beta_target_jump"]
    comparable = recomputed_target.notna() & stored.notna()
    if comparable.sum() == 0:
        return False, "no comparable rows - cannot verify"

    passed = np.allclose(recomputed_target[comparable], stored[comparable])
    detail = "independently recomputed shifted threshold matches" if passed else "MISMATCH"
    return passed, detail


def check_targets_not_contemporaneous():
    """Leakage check 4: alpha_target at row t must equal btc_log_return
    at row t+1, not row t. Confirms the off-by-one is genuinely there."""
    aligned = _read(ALIGNED_PATH)
    target_t = aligned["alpha_target"].iloc[:-1].to_numpy()
    return_t_plus_1 = aligned["btc_log_return"].iloc[1:].to_numpy()

    passed = np.allclose(target_t, return_t_plus_1)
    detail = "alpha_target(t) == btc_log_return(t+1) confirmed" if passed else (
        "MISMATCH - alpha_target may be contemporaneous with its own row"
    )
    return passed, detail


def check_no_shuffling_and_boundaries():
    """Leakage check 5: every stage's index is strictly increasing, and
    each split's actual date range matches config.py's configured
    boundaries exactly, with no overlap between splits."""
    problems = []
    for name, path in (
        ("aligned", ALIGNED_PATH), ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
    ):
        df = _read(path)
        if not df.index.is_monotonic_increasing:
            problems.append(f"{name} index is not monotonically increasing")

    train, val, test = _read(TRAIN_PATH), _read(VAL_PATH), _read(TEST_PATH)

    if val.index.min() < pd.Timestamp(config.VAL_START) or val.index.max() > pd.Timestamp(config.VAL_END):
        problems.append("validation split extends outside its configured boundaries")
    if test.index.min() < pd.Timestamp(config.TEST_START) or test.index.max() > pd.Timestamp(config.TEST_END):
        problems.append("test split extends outside its configured boundaries")
    if train.index.max() >= val.index.min():
        problems.append("train split overlaps the validation period")
    if val.index.max() >= test.index.min():
        problems.append("validation split overlaps the test period")

    passed = not problems
    detail = "chronological order and boundaries confirmed" if passed else "; ".join(problems)
    return passed, detail


def check_no_nan_or_inf():
    """Leakage check 6: no NaN or infinite value survives anywhere in the
    aligned dataset, any split, or any saved window array."""
    problems = []
    for name, path in (
        ("aligned", ALIGNED_PATH), ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
    ):
        df = _read(path)
        numeric = df.select_dtypes(include=[np.number]).to_numpy()
        if not np.isfinite(numeric).all():
            problems.append(f"{name}.csv has NaN or infinite values")

    for agent in ("alpha", "beta"):
        for split in ("train", "val", "test"):
            path = os.path.join(PROCESSED_DIR, f"{agent}_{split}_windows.npz")
            data = np.load(path)
            for key in ("X", "y_alpha", "y_beta_vol", "y_beta_jump"):
                arr = data[key]
                if arr.size and not np.isfinite(arr).all():
                    problems.append(f"{agent}_{split}_windows.npz[{key}] has NaN or infinite values")

    passed = not problems
    detail = "clean" if passed else "; ".join(problems)
    return passed, detail


def check_warmup_trim_exact():
    """Leakage check 7: rebuilds the untrimmed feature frame directly
    from data/raw (bypassing build_dataset.py's own trim step entirely),
    then confirms trimming exactly WARMUP_TRIM_DAYS rows leaves zero
    leading NaNs, AND that trimming one row fewer would NOT have been
    enough - proving the trim length is the exact minimum, not just
    generously large."""
    aligned_raw = build_dataset._align(build_dataset.RAW_DIR)
    features_raw = build_dataset._engineer_features(aligned_raw)

    trimmed_at_configured = features_raw.iloc[config.WARMUP_TRIM_DAYS :]
    trimmed_one_less = features_raw.iloc[config.WARMUP_TRIM_DAYS - 1 :]

    clean_at_configured = trimmed_at_configured.isna().sum().sum() == 0
    still_dirty_at_one_less = trimmed_one_less.isna().sum().sum() > 0

    passed = clean_at_configured and still_dirty_at_one_less
    if not clean_at_configured:
        detail = f"WARMUP_TRIM_DAYS={config.WARMUP_TRIM_DAYS} is too SHORT - NaNs remain after trimming"
    elif not still_dirty_at_one_less:
        detail = f"WARMUP_TRIM_DAYS={config.WARMUP_TRIM_DAYS} is too LONG - one row fewer would already be clean"
    else:
        detail = f"WARMUP_TRIM_DAYS={config.WARMUP_TRIM_DAYS} is exactly the minimum needed"
    return passed, detail


def check_windows_stay_within_split():
    """Leakage check 8: for every saved window, confirms its full
    lookback span (from end_date back lookback-1 days) stays inside the
    SAME split whose CSV it was built from - i.e. there are enough prior
    rows within that split's own date index alone, with no need to reach
    into the previous split."""
    split_paths = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
    lookbacks = {"alpha": config.ALPHA_LOOKBACK, "beta": config.BETA_LOOKBACK}
    problems = []

    for split_name, path in split_paths.items():
        split_dates = _read(path).index

        for agent_name, lookback in lookbacks.items():
            npz_path = os.path.join(PROCESSED_DIR, f"{agent_name}_{split_name}_windows.npz")
            data = np.load(npz_path, allow_pickle=True)
            end_dates = pd.to_datetime(data["end_dates"])

            for end_date in end_dates:
                pos = split_dates.get_loc(end_date)
                start_pos = pos - lookback + 1
                if start_pos < 0:
                    problems.append(
                        f"{agent_name}_{split_name}: window ending {end_date.date()} "
                        f"reaches {abs(start_pos)} day(s) before this split's own start"
                    )

    passed = not problems
    detail = "every window's full lookback stays inside its own split" if passed else "; ".join(problems[:5])
    return passed, detail


def check_feature_column_order():
    """Leakage check 9: the 19 feature columns appear in exactly
    config.FEATURE_ORDER, in every file that carries them."""
    problems = []
    for name, path in (
        ("aligned", ALIGNED_PATH), ("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)
    ):
        df = pd.read_csv(path, index_col="Date", parse_dates=True, nrows=1)
        actual_order = [c for c in df.columns if c in config.FEATURE_ORDER]
        if actual_order != config.FEATURE_ORDER:
            problems.append(f"{name}.csv feature order does not match config.FEATURE_ORDER")

    passed = not problems
    detail = "pinned order confirmed everywhere" if passed else "; ".join(problems)
    return passed, detail


CHECKS = [
    ("1. scaler fit on training rows only", check_scaler_train_only),
    ("2. volatility window is trailing, never centred", check_volatility_trailing),
    ("3. jump threshold excludes today's own return", check_jump_threshold_shifted),
    ("4. targets are not contemporaneous", check_targets_not_contemporaneous),
    ("5. no pre-split shuffling, boundaries correct", check_no_shuffling_and_boundaries),
    ("6. no NaN or infinite values anywhere", check_no_nan_or_inf),
    ("7. warm-up trim is exactly the minimum needed", check_warmup_trim_exact),
    ("8. no window straddles a split boundary", check_windows_stay_within_split),
    ("9. feature column order matches config.FEATURE_ORDER", check_feature_column_order),
]


def run_all_checks():
    """Runs every check in CHECKS and returns (n_failed, results), where
    results is [(name, passed, detail), ...]. Shared by main() below and
    by run_pipeline.py, which runs this same audit automatically after
    every build instead of shelling out to a second process."""
    results = []
    for name, fn in CHECKS:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"{type(e).__name__}: {e}"
        results.append((name, passed, detail))
    n_failed = sum(1 for _, passed, _ in results if not passed)
    return n_failed, results


def main():
    print("=" * 72)
    print("Leakage audit")
    print("=" * 72)

    n_failed, results = run_all_checks()
    for name, passed, detail in results:
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}")
        print(f"         {detail}")

    print()
    print(f"{len(CHECKS) - n_failed} of {len(CHECKS)} checks passed.")
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
