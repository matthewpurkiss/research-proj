"""
splits.py

Slices build_dataset.py's aligned_dataset.csv into chronological
train/validation/test splits, fits the feature scaler on the training
split only, and builds the fixed-length sequence windows both agents
need (60 days for the Alpha Agent, 30 for the Beta Agent).

Windows are built independently per split, after slicing - so by
construction, a window can never contain rows from two different splits.
That is a deliberate, self-imposed rule (see CLAUDE.md), not something
the proposal mandates, and it has a real cost: the first
ALPHA_LOOKBACK-1 days of validation and test get no Alpha window, and
the first BETA_LOOKBACK-1 days get no Beta window, since there aren't
yet enough same-split days behind them to fill one.
"""

import os

import numpy as np
import pandas as pd

import config

INPUT_PATH = os.path.join("data", "processed", "aligned_dataset.csv")
OUTPUT_DIR = os.path.join("data", "processed")

TARGET_COLUMNS = ["alpha_target", "beta_target_vol", "beta_target_jump"]


def load_aligned(input_path=INPUT_PATH):
    return pd.read_csv(input_path, index_col="Date", parse_dates=True)


def chronological_split(df):
    """Section 3.6: three contiguous periods, no shuffling."""
    train = df.loc[config.TRAIN_START : config.TRAIN_END]
    val = df.loc[config.VAL_START : config.VAL_END]
    test = df.loc[config.TEST_START : config.TEST_END]

    for name, split in (("train", train), ("val", val), ("test", test)):
        if not split.index.is_monotonic_increasing:
            raise AssertionError(f"{name} split is not chronologically ordered")

    return train, val, test


def fit_scaler(train_df):
    """Section 3.5: mean and std computed on the training split only."""
    mean = train_df[config.FEATURE_ORDER].mean()
    std = train_df[config.FEATURE_ORDER].std()
    return mean, std


def apply_scaler(df, mean, std):
    """Applies a train-fitted mean/std to any split unchanged. Only the
    19 input features are standardised - the three target columns are
    left as-is, since Section 3.5 scopes normalisation to X_t, not the
    targets."""
    scaled = df.copy()
    scaled[config.FEATURE_ORDER] = (df[config.FEATURE_ORDER] - mean) / std
    return scaled


def make_windows(split_df, lookback):
    """Builds fixed-length sequence windows entirely from split_df's own
    rows. Because split_df has already been sliced to a single split
    before this runs, a window can never reach into a different split -
    the no-straddling rule is satisfied by construction, not by an extra
    check bolted on afterward.

    Window i covers rows [i, i+lookback-1] as the model's input sequence
    (shape: lookback x 19). Its target is read from the LAST row of that
    window, since build_dataset.py already shifted every target column
    to represent "tomorrow" relative to that row.

    Returns (X, y_alpha, y_beta_vol, y_beta_jump, end_dates). A window
    whose target is NaN gets dropped - build_dataset.py's own output
    should never contain one, so this is a defensive check for the
    invariant, not a step that is expected to remove anything.
    """
    features = split_df[config.FEATURE_ORDER].to_numpy()
    targets = split_df[TARGET_COLUMNS].to_numpy()
    dates = split_df.index.to_numpy()

    n_windows = len(split_df) - lookback + 1
    if n_windows <= 0:
        return (
            np.empty((0, lookback, config.N_FEATURES)),
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,)),
            np.empty((0,), dtype=dates.dtype),
        )

    X = np.stack([features[i : i + lookback] for i in range(n_windows)])
    window_targets = targets[lookback - 1 : lookback - 1 + n_windows]
    end_dates = dates[lookback - 1 : lookback - 1 + n_windows]

    valid = ~np.isnan(window_targets).any(axis=1)
    return (
        X[valid],
        window_targets[valid, 0],
        window_targets[valid, 1],
        window_targets[valid, 2],
        end_dates[valid],
    )


def build(input_path=INPUT_PATH, output_dir=OUTPUT_DIR):
    aligned = load_aligned(input_path)
    train_df, val_df, test_df = chronological_split(aligned)

    mean, std = fit_scaler(train_df)
    train_scaled = apply_scaler(train_df, mean, std)
    val_scaled = apply_scaler(val_df, mean, std)
    test_scaled = apply_scaler(test_df, mean, std)

    os.makedirs(output_dir, exist_ok=True)
    train_scaled.to_csv(os.path.join(output_dir, "train.csv"))
    val_scaled.to_csv(os.path.join(output_dir, "val.csv"))
    test_scaled.to_csv(os.path.join(output_dir, "test.csv"))

    scaler_params = pd.DataFrame({"mean": mean, "std": std})
    scaler_params.to_csv(os.path.join(output_dir, "scaler_params.csv"))

    splits = (("train", train_scaled), ("val", val_scaled), ("test", test_scaled))
    agents = (("alpha", config.ALPHA_LOOKBACK), ("beta", config.BETA_LOOKBACK))

    window_report = []
    for agent_name, lookback in agents:
        for split_name, split_df in splits:
            X, y_alpha, y_vol, y_jump, end_dates = make_windows(split_df, lookback)
            out_path = os.path.join(output_dir, f"{agent_name}_{split_name}_windows.npz")
            np.savez(
                out_path,
                X=X,
                y_alpha=y_alpha,
                y_beta_vol=y_vol,
                y_beta_jump=y_jump,
                end_dates=end_dates,
            )
            window_report.append((agent_name, split_name, len(split_df), X.shape[0]))

    print(f"{'agent':<6} {'split':<6} {'split rows':>10} {'usable windows':>15}")
    for agent_name, split_name, split_rows, n_windows in window_report:
        print(f"{agent_name:<6} {split_name:<6} {split_rows:>10} {n_windows:>15}")

    return train_scaled, val_scaled, test_scaled


if __name__ == "__main__":
    build()
