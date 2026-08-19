"""
test_audit_teeth.py

Injects five known faults into a scratch copy of the pipeline's output,
one at a time, and asserts leakage_audit.py's corresponding check fails
on each. PHASE1_SPEC.md's rule: don't trust leakage_audit.py until this
passes - an audit that passes everything is indistinguishable from one
that checks nothing.

Run this after any change to leakage_audit.py or build_dataset.py. If a
mutation stops being caught, the audit has lost a tooth.

Each mutation writes its broken artifacts to a temporary directory (never
touching data/processed), points leakage_audit.py's module-level path
constants at that directory, and calls the ONE check function that
mutation targets. leakage_audit.py's checks were written to read paths
from its own module globals, since its normal job is auditing real files
directly - monkeypatching those globals is how this test reaches around
that without needing to change leakage_audit.py itself.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import build_dataset
import config
import leakage_audit
import splits

REAL_PROCESSED_DIR = splits.OUTPUT_DIR
REAL_ALIGNED_PATH = build_dataset.OUTPUT_PATH


def _point_leakage_audit_at(tmp_dir):
    leakage_audit.PROCESSED_DIR = tmp_dir
    leakage_audit.ALIGNED_PATH = os.path.join(tmp_dir, "aligned_dataset.csv")
    leakage_audit.TRAIN_PATH = os.path.join(tmp_dir, "train.csv")
    leakage_audit.VAL_PATH = os.path.join(tmp_dir, "val.csv")
    leakage_audit.TEST_PATH = os.path.join(tmp_dir, "test.csv")
    leakage_audit.SCALER_PATH = os.path.join(tmp_dir, "scaler_params.csv")


def _restore_leakage_audit_paths():
    leakage_audit.PROCESSED_DIR = REAL_PROCESSED_DIR
    leakage_audit.ALIGNED_PATH = REAL_ALIGNED_PATH
    leakage_audit.TRAIN_PATH = os.path.join(REAL_PROCESSED_DIR, "train.csv")
    leakage_audit.VAL_PATH = os.path.join(REAL_PROCESSED_DIR, "val.csv")
    leakage_audit.TEST_PATH = os.path.join(REAL_PROCESSED_DIR, "test.csv")
    leakage_audit.SCALER_PATH = os.path.join(REAL_PROCESSED_DIR, "scaler_params.csv")


def _load_real_aligned():
    if not os.path.exists(REAL_ALIGNED_PATH):
        raise SystemExit(
            "data/processed/aligned_dataset.csv not found - run build_dataset.py "
            "and splits.py first, this test mutates real pipeline output."
        )
    return pd.read_csv(REAL_ALIGNED_PATH, index_col="Date", parse_dates=True)


def mutation_scaler_on_full_sample(tmp_dir):
    """Fault 1: fits the scaler on train+val+test combined instead of
    training rows only. aligned_dataset.csv is copied through unchanged,
    so the check's own recomputed train-only stats are correct - only
    the stored scaler is wrong."""
    aligned = _load_real_aligned()
    full_mean = aligned[config.FEATURE_ORDER].mean()
    full_std = aligned[config.FEATURE_ORDER].std()
    pd.DataFrame({"mean": full_mean, "std": full_std}).to_csv(
        os.path.join(tmp_dir, "scaler_params.csv")
    )
    shutil.copy(REAL_ALIGNED_PATH, os.path.join(tmp_dir, "aligned_dataset.csv"))


def mutation_volatility_centred(tmp_dir):
    """Fault 2: recomputes btc_realised_vol_14d with a centred window
    instead of trailing, overwriting that one column in an otherwise
    real copy of aligned_dataset.csv."""
    aligned = _load_real_aligned()
    aligned["btc_realised_vol_14d"] = (
        aligned["btc_log_return"].rolling(config.REALISED_VOL_WINDOW, center=True).std()
    )
    aligned.to_csv(os.path.join(tmp_dir, "aligned_dataset.csv"))


def mutation_jump_threshold_unshifted(tmp_dir):
    """Fault 3: recomputes beta_target_jump using a threshold that
    includes today's own return, instead of excluding it via shift(1)."""
    aligned = _load_real_aligned()
    returns = aligned["btc_log_return"]
    threshold = returns.rolling(config.JUMP_THRESHOLD_WINDOW).std() * config.JUMP_THRESHOLD_STD
    is_jump = (returns.abs() > threshold).astype(float)
    is_jump[threshold.isna()] = np.nan
    aligned["beta_target_jump"] = is_jump.shift(-1)
    aligned.to_csv(os.path.join(tmp_dir, "aligned_dataset.csv"))


def mutation_target_contemporaneous(tmp_dir):
    """Fault 4: sets alpha_target to the SAME day's log-return instead of
    tomorrow's."""
    aligned = _load_real_aligned()
    aligned["alpha_target"] = aligned["btc_log_return"]
    aligned.to_csv(os.path.join(tmp_dir, "aligned_dataset.csv"))


def mutation_rows_shuffled(tmp_dir):
    """Fault 5: shuffles the training split's row order before saving.
    aligned_dataset.csv, val.csv, and test.csv are copied through
    unchanged - one broken file is enough to prove the check catches it."""
    shutil.copy(REAL_ALIGNED_PATH, os.path.join(tmp_dir, "aligned_dataset.csv"))
    for name in ("val", "test"):
        shutil.copy(
            os.path.join(REAL_PROCESSED_DIR, f"{name}.csv"), os.path.join(tmp_dir, f"{name}.csv")
        )

    train = pd.read_csv(
        os.path.join(REAL_PROCESSED_DIR, "train.csv"), index_col="Date", parse_dates=True
    )
    shuffled = train.sample(frac=1.0, random_state=0)
    shuffled.to_csv(os.path.join(tmp_dir, "train.csv"))


MUTATIONS = [
    (
        "1. scaler fit on the full sample",
        mutation_scaler_on_full_sample,
        lambda: leakage_audit.check_scaler_train_only(),
    ),
    (
        "2. volatility window centred instead of trailing",
        mutation_volatility_centred,
        lambda: leakage_audit.check_volatility_trailing(),
    ),
    (
        "3. jump threshold not shifted",
        mutation_jump_threshold_unshifted,
        lambda: leakage_audit.check_jump_threshold_shifted(),
    ),
    (
        "4. target made contemporaneous",
        mutation_target_contemporaneous,
        lambda: leakage_audit.check_targets_not_contemporaneous(),
    ),
    (
        "5. rows shuffled before splitting",
        mutation_rows_shuffled,
        lambda: leakage_audit.check_no_shuffling_and_boundaries(),
    ),
]


def main():
    print("=" * 72)
    print("Mutation test: does the audit have teeth?")
    print("=" * 72)

    n_missed = 0
    for name, mutate_fn, check_fn in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mutate_fn(tmp_dir)
            _point_leakage_audit_at(tmp_dir)
            try:
                passed, detail = check_fn()
            except Exception as e:
                passed, detail = False, f"{type(e).__name__}: {e}"
            finally:
                _restore_leakage_audit_paths()

        caught = not passed
        if not caught:
            n_missed += 1
        flag = "CAUGHT" if caught else "MISSED"
        print(f"  [{flag}] {name}")
        print(f"          {detail}")

    print()
    print(f"{len(MUTATIONS) - n_missed} of {len(MUTATIONS)} mutations were caught.")
    if n_missed:
        print("The audit has lost a tooth - do not trust leakage_audit.py until this is fixed.")
    sys.exit(0 if n_missed == 0 else 1)


if __name__ == "__main__":
    main()
