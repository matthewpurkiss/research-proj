"""
experiment_delta_vol.py

THROWAWAY DIAGNOSTIC - not part of the official Phase 3 pipeline, doesn't
touch config.py/build_dataset.py/models.py. Answers one question before
raising finding 1 with Van Alten: if the Beta Agent's volatility head
predicts the day-over-day CHANGE in realised vol (option B) instead of
the absolute level, does that actually fix the "loses to persistence by
9x" result from skill_scores.py, or is the current architecture/features
just not capturing anything useful about volatility at all?

Mechanism this targets: btc_target_vol[t] (next-day 14d trailing vol) and
btc_realised_vol_14d[t] (today's own 14d trailing vol, already an input
feature) overlap 13/14 days, so predicting the RAW LEVEL lets a model get
"free" accuracy just by regressing toward today's value - except the
current Beta Agent doesn't even do that. Predicting the delta instead
means the naive "do nothing" baseline is delta=0, a genuinely weak
baseline, so any real skill has to come from the model.

Reconstruction for reporting: predicted_vol(t+1) = vol(t) +
predicted_delta(t+1) - same pattern as metrics.py's price reconstruction
from a predicted log-return, applied here to volatility instead of price.

Uses a separate, smaller model (single-head GRU, linear output - no
softplus, since a delta can be negative) rather than reusing
models.BetaAgent, since BetaAgent's vol_head is softplus-activated
specifically because the official target is always positive - that
activation would be actively wrong for a delta target. Ignores the jump
head entirely; this experiment is scoped to the volatility question only.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

import config
import train_beta
from evaluate import diebold_mariano

VOL_FEATURE_IDX = config.FEATURE_ORDER.index("btc_realised_vol_14d")
DEVICE = train_beta.DEVICE
SEEDS = train_beta.SEEDS

# splits.py only standardises the 19 input features (mean/std), never the
# target columns (splits.py's apply_scaler docstring: "Section 3.5 scopes
# normalisation to X_t, not the targets"). X's last-timestep vol feature
# is therefore in SCALED units, while y_beta_vol is RAW - un-scale the
# feature back to raw units before comparing it against the raw target,
# or "vol_today" and "vol_y" live on different, incomparable scales.
_scaler_params = pd.read_csv(
    os.path.join("data", "processed", "scaler_params.csv"), index_col=0
)
VOL_MEAN = float(_scaler_params.loc["btc_realised_vol_14d", "mean"])
VOL_STD = float(_scaler_params.loc["btc_realised_vol_14d", "std"])


class DeltaVolGRU(nn.Module):
    def __init__(self, input_size=19, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def load_split_with_delta(split):
    """Same windows as the official Beta Agent, but the target swapped
    for delta = vol(t+1) - vol(t), where vol(t) is read straight out of
    the input window's own last timestep (today's own realised vol
    feature) - no new data loading, just a different label."""
    X, vol_y, _ = train_beta.load_beta_split(split)
    vol_today_scaled = X[:, -1, VOL_FEATURE_IDX]
    vol_today = vol_today_scaled * VOL_STD + VOL_MEAN
    delta_y = vol_y - vol_today
    return X, vol_today, vol_y, delta_y


def train_one_seed(seed, hidden_size=32, max_epochs=100, patience=10,
                    lr_patience=5, min_epochs=20, batch_size=64, verbose=False):
    train_beta.set_seed(seed)

    train_X, _, _, train_delta = load_split_with_delta("train")
    val_X, val_vol_today, val_vol_y, val_delta = load_split_with_delta("val")
    test_X, test_vol_today, test_vol_y, test_delta = load_split_with_delta("test")

    train_loader = DataLoader(
        TensorDataset(train_X, train_delta), batch_size=batch_size, shuffle=True
    )

    model = DeltaVolGRU(input_size=train_X.shape[-1], hidden_size=hidden_size).to(DEVICE)
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
            xb, delta_b = xb.to(DEVICE), delta_b.to(DEVICE)
            optimizer.zero_grad()
            pred_delta = model(xb)
            loss = loss_fn(pred_delta, delta_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_delta = model(val_X.to(DEVICE)).cpu()
            val_loss = loss_fn(val_pred_delta, val_delta).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(f"  seed {seed} epoch {epoch:>3}: val_delta_mse={val_loss:.6f}")

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_delta = model(test_X.to(DEVICE)).cpu()

    reconstructed_vol = (test_vol_today + test_pred_delta).numpy()
    actual_vol = test_vol_y.numpy()
    reconstructed_rmse = float(np.sqrt(np.mean((actual_vol - reconstructed_vol) ** 2)))

    return {
        "reconstructed_vol": reconstructed_vol,
        "actual_vol": actual_vol,
        "reconstructed_rmse": reconstructed_rmse,
    }


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    print("Delta-volatility experiment: predict vol(t+1) - vol(t), reconstruct level")
    print("-" * 72)

    results = []
    for seed in SEEDS:
        result = train_one_seed(seed, verbose=False)
        results.append(result)
        print(f"seed {seed}: reconstructed vol RMSE={result['reconstructed_rmse']:.5f}")

    rmses = np.array([r["reconstructed_rmse"] for r in results])
    print(f"\nreconstructed vol RMSE mean +/- std: {rmses.mean():.5f} +/- {rmses.std():.5f}")

    print("\nvs naive persistence (same target, same test days) - DM test per seed:")
    skills, n_significant = [], 0
    for i, result in enumerate(results):
        _, _, val_vol_y, _ = load_split_with_delta("val")
        actual_vol = result["actual_vol"]
        prev_last = val_vol_y.numpy()[-1]
        persistence_pred = np.concatenate([[prev_last], actual_vol[:-1]])
        persistence_rmse = float(np.sqrt(np.mean((actual_vol - persistence_pred) ** 2)))

        skill = 1 - (result["reconstructed_rmse"] / persistence_rmse)
        skills.append(skill)

        model_errors = actual_vol - result["reconstructed_vol"]
        persistence_errors = actual_vol - persistence_pred
        dm_stat, p_value = diebold_mariano(model_errors, persistence_errors, h=1)
        significant = p_value < 0.05
        n_significant += significant

        print(
            f"  seed {i}: delta-model RMSE={result['reconstructed_rmse']:.5f}  "
            f"persistence RMSE={persistence_rmse:.5f}  skill={skill:+.1%}  "
            f"DM={dm_stat:+.3f}  p={p_value:.4f}  "
            f"({'significant' if significant else 'not significant'})"
        )

    skills = np.array(skills)
    print(f"\nskill score mean +/- std: {skills.mean():+.1%} +/- {skills.std():.1%}")
    print(f"{n_significant} of {len(results)} seeds significant at alpha=0.05")
