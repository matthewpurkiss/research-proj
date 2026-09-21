"""
experiment_garch_feature.py

THROWAWAY DIAGNOSTIC - not part of the official pipeline, doesn't touch
config.py/build_dataset.py/models.py/train_beta.py. Tests whether giving
the Beta Agent GARCH(1,1)'s own one-step-ahead volatility forecast as an
extra (20th) input feature helps it predict the official absolute-level
volatility target - orthogonal to the delta-target experiment
(experiment_delta_vol.py), which changed WHAT is predicted rather than
what information the model gets to see.

Leakage-safety note (the one real complication here): garch_baseline.py
fits GARCH's parameters once on the FULL training split, which is safe
for producing val/test forecasts (parameters never see val/test data),
but NOT safe for reuse as a feature on training days themselves - a
forecast for an early training day would then secretly be informed by
later training data the fit used.

Fix used here: fit GARCH on only the first ~2 years of train
(GARCH_FIT_START..GARCH_FIT_END), then roll those FIXED parameters
forward one-step-at-a-time through the rest of train, val, and test
(FORECAST_START onward) - same walk-forward technique garch_baseline.py
already uses for val/test, just started earlier. Cost: this experiment's
training set is restricted to FORECAST_START..TRAIN_END (~4 years
instead of the official ~6), since the first ~2 years have no leakage-
safe GARCH feature available. Val/test windows are unaffected - both
already start after FORECAST_START.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from arch import arch_model

import config
import splits
import train_beta
from evaluate import diebold_mariano

DATA_DIR = os.path.join("data", "processed")
SCALE = 100  # matches garch_baseline.py's scaling convention

GARCH_FIT_START = config.TRAIN_START   # 2018-01-01
GARCH_FIT_END = "2019-12-31"           # ~2 years - enough data for a stable GARCH(1,1) fit
FORECAST_START = "2020-01-01"          # everything from here on gets a leakage-safe forecast

LOCAL_FEATURE_ORDER = config.FEATURE_ORDER + ["garch_vol_forecast"]
TARGET_COLUMNS = splits.TARGET_COLUMNS  # ["alpha_target", "beta_target_vol", "beta_target_jump"]
VOL_TARGET_IDX = TARGET_COLUMNS.index("beta_target_vol")


def build_garch_feature_series():
    """One-step-ahead GARCH(1,1) forecasts, fit ONCE on GARCH_FIT_START..
    GARCH_FIT_END and rolled forward through the rest of train/val/test
    with those same fixed parameters - see module docstring."""
    aligned = pd.read_csv(
        os.path.join(DATA_DIR, "aligned_dataset.csv"), index_col="Date", parse_dates=True
    )
    returns = aligned["btc_log_return"]

    fit_returns = returns.loc[GARCH_FIT_START:GARCH_FIT_END] * SCALE
    full_returns = returns.loc[GARCH_FIT_START:config.TEST_END] * SCALE

    am_fit = arch_model(fit_returns, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    fitted = am_fit.fit(disp="off")

    am_full = arch_model(full_returns, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    forecasts = am_full.forecast(
        params=fitted.params, start=FORECAST_START, horizon=1, reindex=False
    )
    forecast_vol = np.sqrt(forecasts.variance.values[:, 0]) / SCALE
    return pd.Series(forecast_vol, index=forecasts.variance.index, name="garch_vol_forecast")


def build_restricted_df():
    """aligned_dataset.csv sliced to FORECAST_START..TEST_END - shrinks
    this experiment's train range to FORECAST_START..TRAIN_END, val/test
    untouched - with the GARCH forecast attached as one more column."""
    aligned = splits.load_aligned()
    garch_feature = build_garch_feature_series()

    restricted = aligned.loc[FORECAST_START:config.TEST_END].copy()
    restricted["garch_vol_forecast"] = garch_feature
    n_missing = restricted["garch_vol_forecast"].isna().sum()
    assert n_missing == 0, f"GARCH feature has {n_missing} gaps in the restricted range"
    return restricted


def chronological_split_restricted(df):
    train = df.loc[FORECAST_START : config.TRAIN_END]
    val = df.loc[config.VAL_START : config.VAL_END]
    test = df.loc[config.TEST_START : config.TEST_END]
    return train, val, test


def fit_scaler_local(train_df):
    mean = train_df[LOCAL_FEATURE_ORDER].mean()
    std = train_df[LOCAL_FEATURE_ORDER].std()
    return mean, std


def apply_scaler_local(df, mean, std):
    scaled = df.copy()
    scaled[LOCAL_FEATURE_ORDER] = (df[LOCAL_FEATURE_ORDER] - mean) / std
    return scaled


def make_windows_local(split_df, lookback=config.BETA_LOOKBACK):
    """Same windowing logic as splits.py's make_windows, generalised to
    the local 20-feature order instead of config.FEATURE_ORDER, and
    returning only the volatility target (this experiment doesn't touch
    the jump head)."""
    features = split_df[LOCAL_FEATURE_ORDER].to_numpy()
    targets = split_df[TARGET_COLUMNS].to_numpy()

    n_windows = len(split_df) - lookback + 1
    X = np.stack([features[i : i + lookback] for i in range(n_windows)])
    window_targets = targets[lookback - 1 : lookback - 1 + n_windows]

    valid = ~np.isnan(window_targets).any(axis=1)
    return X[valid], window_targets[valid, VOL_TARGET_IDX]


def build_beta_datasets():
    restricted = build_restricted_df()
    train_df, val_df, test_df = chronological_split_restricted(restricted)

    mean, std = fit_scaler_local(train_df)
    train_scaled = apply_scaler_local(train_df, mean, std)
    val_scaled = apply_scaler_local(val_df, mean, std)
    test_scaled = apply_scaler_local(test_df, mean, std)

    train_X, train_vol = make_windows_local(train_scaled)
    val_X, val_vol = make_windows_local(val_scaled)
    test_X, test_vol = make_windows_local(test_scaled)

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)
    return (
        to_tensor(train_X), to_tensor(train_vol),
        to_tensor(val_X), to_tensor(val_vol),
        to_tensor(test_X), to_tensor(test_vol),
    )


class VolOnlyGRU(nn.Module):
    """Same single-GRU-layer, softplus-output design as models.BetaAgent's
    volatility head, minus the jump head - this experiment is scoped to
    the volatility question only."""

    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return F.softplus(self.head(out[:, -1, :])).squeeze(-1)


def train_one_seed(seed, train_X, train_vol, val_X, val_vol, test_X, test_vol,
                    hidden_size=32, max_epochs=100, patience=10, lr_patience=5,
                    min_epochs=20, batch_size=64):
    train_beta.set_seed(seed)
    device = train_beta.DEVICE

    train_loader = DataLoader(
        TensorDataset(train_X, train_vol), batch_size=batch_size, shuffle=True
    )

    model = VolOnlyGRU(input_size=train_X.shape[-1], hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X.to(device)).cpu()
            val_loss = loss_fn(val_pred, val_vol).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(test_X.to(device)).cpu().numpy()

    test_actual = test_vol.numpy()
    rmse = float(np.sqrt(np.mean((test_actual - test_pred) ** 2)))
    return {"test_pred": test_pred, "test_actual": test_actual, "rmse": rmse}


if __name__ == "__main__":
    print(f"device: {train_beta.DEVICE}")
    print("GARCH-as-input-feature experiment (official absolute-level target, input_size=20)")
    print("-" * 88)

    train_X, train_vol, val_X, val_vol, test_X, test_vol = build_beta_datasets()
    print(
        f"train windows: {train_X.shape[0]}  "
        f"(restricted to {FORECAST_START} onward, vs official train from {config.TRAIN_START})"
    )
    print(f"val windows:   {val_X.shape[0]}")
    print(f"test windows:  {test_X.shape[0]}")

    results = []
    for seed in train_beta.SEEDS:
        r = train_one_seed(seed, train_X, train_vol, val_X, val_vol, test_X, test_vol)
        results.append(r)
        print(f"seed {seed}: test vol RMSE={r['rmse']:.5f}")

    rmses = np.array([r["rmse"] for r in results])
    print(f"\ntest vol RMSE mean +/- std: {rmses.mean():.5f} +/- {rmses.std():.5f}")
    print("(reference, official Beta Agent, no GARCH feature, full train range: 0.01980 +/- 0.00271)")

    print("\nvs naive persistence (same test set, same target) - DM test per seed:")
    skills, n_significant = [], 0
    for i, r in enumerate(results):
        actual = r["test_actual"]
        persistence_pred = np.concatenate([[val_vol.numpy()[-1]], actual[:-1]])
        persistence_rmse = float(np.sqrt(np.mean((actual - persistence_pred) ** 2)))
        skill = 1 - (r["rmse"] / persistence_rmse)
        skills.append(skill)

        model_errors = actual - r["test_pred"]
        persistence_errors = actual - persistence_pred
        dm_stat, p_value = diebold_mariano(model_errors, persistence_errors, h=1)
        significant = p_value < 0.05
        n_significant += significant

        print(
            f"  seed {i}: model RMSE={r['rmse']:.5f}  persistence RMSE={persistence_rmse:.5f}  "
            f"skill={skill:+.1%}  DM={dm_stat:+.3f}  p={p_value:.4f}  "
            f"({'significant' if significant else 'not significant'})"
        )

    skills = np.array(skills)
    print(f"\nskill score mean +/- std: {skills.mean():+.1%} +/- {skills.std():.1%}")
    print(f"{n_significant} of {len(results)} seeds significant at alpha=0.05")
