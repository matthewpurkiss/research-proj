"""
train.py

Shared training harness for Phase 2 models: loads the windowed Alpha
Agent data from Phase 1's output, trains a given model under the recipe
fixed in Section 5.5 of the proposal, and reports RMSE on the held-out
test set.

Common training parameters (Section 5.5), same for every model in
Phase 2:
    optimiser: Adam, lr=1e-3, betas=(0.9, 0.999)
    batch size: 64
    max epochs: 100, early stopping after 10 stagnant validation epochs
    LR schedule: halve after 5 stagnant epochs
    gradient clipping: max norm 1.0
    5 fixed random seeds, results reported as mean +/- std

Note: Section 5.6.2 evaluates MAPE against Seabe et al.'s price-based
benchmark table, but our target is a log-return, not a price level.
MAPE computed directly on log-returns (which sit near zero) would be
numerically unstable and wouldn't match that benchmark's scale - getting
a comparable MAPE means reconstructing predicted prices from predicted
returns first. That reconstruction isn't built yet; this file reports
RMSE on the log-return itself for now, which is directly interpretable.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import models

DATA_DIR = os.path.join("data", "processed")
SEEDS = [0, 1, 2, 3, 4]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_alpha_split(split, feature_indices=None):
    """Loads one split's Alpha Agent windows. feature_indices selects a
    subset of the 19 features (e.g. [0] for BTC log-return only, for the
    univariate baselines) - None keeps all 19."""
    data = np.load(os.path.join(DATA_DIR, f"alpha_{split}_windows.npz"))
    X, y = data["X"], data["y_alpha"]
    if feature_indices is not None:
        X = X[:, :, feature_indices]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_one_seed(model_fn, seed, feature_indices=None, max_epochs=100,
                    patience=10, lr_patience=5, batch_size=64, verbose=False):
    """Trains one instance of model_fn() for one seed, using early
    stopping on validation RMSE. Returns (trained_model, test_rmse,
    test_pred, test_y) - the per-day predictions and actuals are what
    the Diebold-Mariano test needs, not just the aggregate RMSE."""
    set_seed(seed)

    train_X, train_y = load_alpha_split("train", feature_indices)
    val_X, val_y = load_alpha_split("val", feature_indices)
    test_X, test_y = load_alpha_split("test", feature_indices)

    train_loader = DataLoader(
        TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True
    )

    model = model_fn().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience
    )
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_rmse = _rmse(model, val_X, val_y)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(f"  seed {seed} epoch {epoch:>3}: val_rmse={val_rmse:.5f}")

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    test_pred = _predict(model, test_X)
    test_rmse = torch.sqrt(torch.mean((test_pred - test_y) ** 2)).item()
    return model, test_rmse, test_pred.numpy(), test_y.numpy()


@torch.no_grad()
def _predict(model, X):
    model.eval()
    return model(X.to(DEVICE)).cpu()


def _rmse(model, X, y):
    return torch.sqrt(torch.mean((_predict(model, X) - y) ** 2)).item()


def run_multi_seed(model_fn, feature_indices=None, seeds=SEEDS, verbose=False):
    """Trains model_fn() across every seed in `seeds` and reports test
    RMSE as mean +/- std, per Section 5.5's reproducibility requirement."""
    rmses = []
    for seed in seeds:
        _, test_rmse, _, _ = train_one_seed(model_fn, seed, feature_indices, verbose=verbose)
        rmses.append(test_rmse)
        print(f"seed {seed}: test RMSE (log-return) = {test_rmse:.5f}")

    rmses = np.array(rmses)
    print(f"\nmean +/- std across {len(seeds)} seeds: {rmses.mean():.5f} +/- {rmses.std():.5f}")
    return rmses


BASELINES = {
    "lstm": ("Vanilla LSTM (univariate)", lambda: models.VanillaLSTM(input_size=1, hidden_size=64), [0]),
    "gru": ("Vanilla GRU (univariate)", lambda: models.VanillaGRU(input_size=1, hidden_size=74), [0]),
    "bilstm": ("Standard Bi-LSTM (multivariate, no attention)", lambda: models.StandardBiLSTM(input_size=19, hidden_size=64), None),
    "alpha": ("Alpha Agent (attention-augmented multivariate Bi-LSTM)", lambda: models.AlphaAgent(input_size=19, hidden_size=64), None),
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(BASELINES.keys()), default="lstm")
    args = parser.parse_args()

    name, model_fn, feature_indices = BASELINES[args.model]
    print(f"device: {DEVICE}")
    print(name)
    run_multi_seed(model_fn, feature_indices=feature_indices)
