"""
experiment_combined.py

THROWAWAY DIAGNOSTIC - not part of the official pipeline. Stacks the two
independent fixes already tested separately:
  - experiment_delta_vol.py: predict the day-over-day CHANGE in vol,
    reconstruct the level, instead of predicting the absolute level
    directly (RMSE 0.01980 -> 0.01153 on its own)
  - experiment_garch_feature.py: add GARCH(1,1)'s one-step-ahead forecast
    as a 20th input feature (RMSE 0.01980 -> 0.01586 on its own)

These target different parts of the problem (what's predicted vs. what
information the model sees), so there's no reason to expect them to be
redundant - this experiment checks whether they actually stack.

Reuses experiment_garch_feature.py's leakage-safe restricted dataset
(GARCH fit on 2018-2019 only, forecast rolled forward from 2020 onward,
train range restricted to 2020..TRAIN_END for the same reason) rather
than rebuilding it, then swaps the target for the delta and the model
for a linear-output GRU (softplus would be wrong for a delta target,
same reasoning as experiment_delta_vol.py).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import experiment_garch_feature as gf
import train_beta
from evaluate import diebold_mariano
from experiment_delta_vol import DeltaVolGRU

VOL_FEATURE_IDX = gf.LOCAL_FEATURE_ORDER.index("btc_realised_vol_14d")


def make_windows_delta(split_df, mean, std, lookback=gf.config.BETA_LOOKBACK):
    """Same windowing as experiment_garch_feature.make_windows_local, but
    also extracts vol_today (un-scaled back to raw units using THIS
    experiment's own local scaler, fit on the restricted train range -
    not the official scaler_params.csv, which was fit on the full
    official train range and would be the wrong mean/std here) and
    returns the delta target instead of the absolute target."""
    features = split_df[gf.LOCAL_FEATURE_ORDER].to_numpy()
    targets = split_df[gf.TARGET_COLUMNS].to_numpy()

    n_windows = len(split_df) - lookback + 1
    X = np.stack([features[i : i + lookback] for i in range(n_windows)])
    window_targets = targets[lookback - 1 : lookback - 1 + n_windows]

    valid = ~np.isnan(window_targets).any(axis=1)
    X = X[valid]
    vol_y = window_targets[valid, gf.VOL_TARGET_IDX]

    vol_today_scaled = X[:, -1, VOL_FEATURE_IDX]
    vol_today = vol_today_scaled * std["btc_realised_vol_14d"] + mean["btc_realised_vol_14d"]
    delta_y = vol_y - vol_today

    return X, vol_today, vol_y, delta_y


def build_datasets():
    restricted = gf.build_restricted_df()
    train_df, val_df, test_df = gf.chronological_split_restricted(restricted)

    mean, std = gf.fit_scaler_local(train_df)
    train_scaled = gf.apply_scaler_local(train_df, mean, std)
    val_scaled = gf.apply_scaler_local(val_df, mean, std)
    test_scaled = gf.apply_scaler_local(test_df, mean, std)

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32)

    train_X, _, _, train_delta = make_windows_delta(train_scaled, mean, std)
    val_X, val_today, val_vol_y, val_delta = make_windows_delta(val_scaled, mean, std)
    test_X, test_today, test_vol_y, test_delta = make_windows_delta(test_scaled, mean, std)

    return {
        "train_X": to_tensor(train_X), "train_delta": to_tensor(train_delta),
        "val_X": to_tensor(val_X), "val_delta": to_tensor(val_delta),
        "val_vol_y": val_vol_y,
        "test_X": to_tensor(test_X), "test_delta": to_tensor(test_delta),
        "test_today": test_today, "test_vol_y": test_vol_y,
    }


def train_one_seed(seed, data, hidden_size=32, max_epochs=100, patience=10,
                    lr_patience=5, min_epochs=20, batch_size=64):
    train_beta.set_seed(seed)
    device = train_beta.DEVICE

    train_loader = DataLoader(
        TensorDataset(data["train_X"], data["train_delta"]), batch_size=batch_size, shuffle=True
    )

    model = DeltaVolGRU(input_size=data["train_X"].shape[-1], hidden_size=hidden_size).to(device)
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
        for xb, delta_b in train_loader:
            xb, delta_b = xb.to(device), delta_b.to(device)
            optimizer.zero_grad()
            pred_delta = model(xb)
            loss = loss_fn(pred_delta, delta_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_delta = model(data["val_X"].to(device)).cpu()
            val_loss = loss_fn(val_pred_delta, data["val_delta"]).item()
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
        test_pred_delta = model(data["test_X"].to(device)).cpu().numpy()

    reconstructed_vol = data["test_today"] + test_pred_delta
    actual_vol = data["test_vol_y"]
    rmse = float(np.sqrt(np.mean((actual_vol - reconstructed_vol) ** 2)))

    return {"reconstructed_vol": reconstructed_vol, "actual_vol": actual_vol, "rmse": rmse}


if __name__ == "__main__":
    print(f"device: {train_beta.DEVICE}")
    print("Combined experiment: delta target + GARCH feature (input_size=20)")
    print("-" * 88)

    data = build_datasets()
    print(f"train windows: {data['train_X'].shape[0]}  (restricted to {gf.FORECAST_START} onward)")
    print(f"val windows:   {data['val_X'].shape[0]}")
    print(f"test windows:  {data['test_X'].shape[0]}")

    results = []
    for seed in train_beta.SEEDS:
        r = train_one_seed(seed, data)
        results.append(r)
        print(f"seed {seed}: reconstructed vol RMSE={r['rmse']:.5f}")

    rmses = np.array([r["rmse"] for r in results])
    print(f"\nreconstructed vol RMSE mean +/- std: {rmses.mean():.5f} +/- {rmses.std():.5f}")
    print("(reference: official=0.01980, delta-only=0.01153, GARCH-feature-only=0.01586)")

    print("\nvs naive persistence (same test set, same target) - DM test per seed:")
    skills, n_significant = [], 0
    for i, r in enumerate(results):
        actual = r["actual_vol"]
        persistence_pred = np.concatenate([[data["val_vol_y"][-1]], actual[:-1]])
        persistence_rmse = float(np.sqrt(np.mean((actual - persistence_pred) ** 2)))
        skill = 1 - (r["rmse"] / persistence_rmse)
        skills.append(skill)

        model_errors = actual - r["reconstructed_vol"]
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
