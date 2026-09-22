"""
experiment_single_day_vol.py

THROWAWAY - tests Van Alten's suggestion (email, 2026-09-22): the current
volatility target is a 14-day ROLLING statistic (btc_realised_vol_14d,
config.REALISED_VOL_WINDOW), which is why naive persistence wins so
easily - today's and tomorrow's values share 13 of 14 underlying days.
His question: are we predicting the 14-day average, or just the next
day's own volatility? Confirmed: it's the 14-day average. His suggestion:
predict just that day's own volatility instead.

Single-day volatility here = |btc_log_return| (how big was that day's
move) - the standard single-day realised-volatility proxy, and exactly
what garch_baseline.py already predicts. Two things follow from that:
  1. This should remove the overlap problem at its root (a single day
     shares zero days with any other single day), not just shrink it.
  2. It finally makes a genuinely fair, same-target DM test between the
     Beta Agent and GARCH possible - every previous comparison between
     them (skill_scores.py) had to be descriptive-only because they
     predicted different things.

Reuses the OFFICIAL beta_{split}_windows.npz files unchanged (same 19
features, same 30-day lookback, full 2018-2023 train range - no leakage
complication here, unlike the GARCH-feature experiment, since the new
target is just a different column already fully available in
aligned_dataset.csv, not something that required fitting anything).
Only the target changes: instead of beta_target_vol (14-day rolling
std, shifted), it's next-day's |return|.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import garch_baseline
import train_beta
from evaluate import diebold_mariano

DATA_DIR = os.path.join("data", "processed")


def load_target_series():
    """today_abs_return: |return| on that day itself (used for the
    persistence baseline - forecast(t+1) = |return(t)|, no shifting).
    next_day_abs_return: |return| shifted back one day, i.e. the value
    AT row t is tomorrow's |return| - matches how beta_target_vol
    itself is constructed (build_dataset.py shifts -1), so it lines up
    with the official beta windows' end_dates the same way."""
    aligned = pd.read_csv(
        os.path.join(DATA_DIR, "aligned_dataset.csv"), index_col="Date", parse_dates=True
    )
    today_abs_return = aligned["btc_log_return"].abs()
    next_day_abs_return = today_abs_return.shift(-1)
    return today_abs_return, next_day_abs_return


def load_split_single_day(split, today_series, target_series):
    data = np.load(os.path.join(DATA_DIR, f"beta_{split}_windows.npz"), allow_pickle=True)
    X = torch.tensor(data["X"], dtype=torch.float32)
    end_dates = pd.to_datetime(data["end_dates"])

    today_vals = today_series.reindex(end_dates).to_numpy()
    target_vals = target_series.reindex(end_dates).to_numpy()

    valid = ~np.isnan(target_vals)
    return (
        X[valid],
        torch.tensor(today_vals[valid], dtype=torch.float32),
        torch.tensor(target_vals[valid], dtype=torch.float32),
        end_dates[valid],
    )


class SingleDayVolGRU(nn.Module):
    """Same single-GRU-layer, softplus-output design as models.BetaAgent's
    volatility head - the target is still always positive (it's an
    absolute value), so softplus is still the right activation here."""

    def __init__(self, input_size=19, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return F.softplus(self.head(out[:, -1, :])).squeeze(-1)


def train_one_seed(seed, train_X, train_y, val_X, val_y, test_X, test_y,
                    hidden_size=32, max_epochs=100, patience=10, lr_patience=5,
                    min_epochs=20, batch_size=64):
    train_beta.set_seed(seed)
    device = train_beta.DEVICE

    train_loader = DataLoader(
        TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True
    )

    model = SingleDayVolGRU(input_size=train_X.shape[-1], hidden_size=hidden_size).to(device)
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
            val_loss = loss_fn(val_pred, val_y).item()
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

    test_actual = test_y.numpy()
    rmse = float(np.sqrt(np.mean((test_actual - test_pred) ** 2)))
    return {"test_pred": test_pred, "test_actual": test_actual, "rmse": rmse}


if __name__ == "__main__":
    print(f"device: {train_beta.DEVICE}")
    print("Single-day volatility target: predict tomorrow's |return| (Van Alten's suggestion, 2026-09-22)")
    print("-" * 96)

    today_series, target_series = load_target_series()

    train_X, train_today, train_y, _ = load_split_single_day("train", today_series, target_series)
    val_X, val_today, val_y, _ = load_split_single_day("val", today_series, target_series)
    test_X, test_today, test_y, test_end_dates = load_split_single_day("test", today_series, target_series)

    print(f"train windows: {train_X.shape[0]}  val: {val_X.shape[0]}  test: {test_X.shape[0]}")

    results = []
    for seed in train_beta.SEEDS:
        r = train_one_seed(seed, train_X, train_y, val_X, val_y, test_X, test_y)
        results.append(r)
        print(f"seed {seed}: test RMSE={r['rmse']:.5f}")

    rmses = np.array([r["rmse"] for r in results])
    print(f"\ntest RMSE mean +/- std: {rmses.mean():.5f} +/- {rmses.std():.5f}")

    print("\nvs naive persistence (forecast tomorrow = today's own |return|) - DM test per seed:")
    test_today_np = test_today.numpy()
    skills, n_significant = [], 0
    for i, r in enumerate(results):
        actual = r["test_actual"]
        persistence_pred = test_today_np
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

    print("\nvs GARCH(1,1) - now a genuinely fair, same-target comparison:")
    _, _, garch_test_forecast = garch_baseline.fit_and_forecast()
    aligned_garch = garch_test_forecast.reindex(test_end_dates)
    n_missing = aligned_garch.isna().sum()
    assert n_missing == 0, f"{n_missing} test days have no aligned GARCH forecast"
    garch_pred = aligned_garch.to_numpy()

    actual = test_y.numpy()
    garch_errors = actual - garch_pred
    garch_rmse = float(np.sqrt(np.mean(garch_errors ** 2)))
    print(f"  GARCH(1,1) RMSE on this exact target/days: {garch_rmse:.5f}")

    n_beta_wins = 0
    for i, r in enumerate(results):
        beta_errors = actual - r["test_pred"]
        dm_stat, p_value = diebold_mariano(beta_errors, garch_errors, h=1)
        significant = p_value < 0.05
        better = "Beta Agent" if r["rmse"] < garch_rmse else "GARCH"
        n_beta_wins += r["rmse"] < garch_rmse
        print(
            f"  seed {i}: Beta Agent RMSE={r['rmse']:.5f}  GARCH RMSE={garch_rmse:.5f}  "
            f"DM={dm_stat:+.3f}  p={p_value:.4f}  "
            f"({'significant, ' + better + ' better' if significant else 'not significant'})"
        )
    print(f"\nBeta Agent had lower RMSE than GARCH on {n_beta_wins} of {len(results)} seeds")
