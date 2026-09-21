"""
train_beta.py

Training harness for the Beta Agent (Section 5.3) - genuinely different
from train.py's Alpha-side harness, since this model has two outputs
and two loss terms instead of one.

Common training parameters, same Section 5.5 recipe used throughout
Phase 2: Adam (lr=1e-3, betas=(0.9, 0.999)), batch size 64, early
stopping after 10 stagnant validation epochs, LR halved after 5
stagnant epochs, gradient clipping at max norm 1.0, 5 fixed seeds.

The volatility loss (MSE) and jump loss (BCE, with a class-imbalance
pos_weight) sit on very different numeric scales - BCE with a ~36x
pos_weight correction is typically much larger than MSE on a target
that's usually under 0.1. jump_loss_weight below is a rough starting
point, not a carefully derived value - both loss components print
separately every epoch specifically so the real relative scale is
visible and this can be calibrated against actual numbers rather than
guessed blindly.
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


def load_beta_split(split):
    data = np.load(os.path.join(DATA_DIR, f"beta_{split}_windows.npz"))
    X = torch.tensor(data["X"], dtype=torch.float32)
    y_vol = torch.tensor(data["y_beta_vol"], dtype=torch.float32)
    y_jump = torch.tensor(data["y_beta_jump"], dtype=torch.float32)
    return X, y_vol, y_jump


def compute_pos_weight(y_jump_train):
    """BCEWithLogitsLoss's pos_weight, computed directly from the actual
    training window's jump rate - (1 - rate) / rate - rather than reusing
    the ~36.4 figure from Phase 1's EDA, which was measured on the
    unwindowed split and could differ slightly after windowing trims a
    few leading rows."""
    rate = y_jump_train.mean().item()
    return (1 - rate) / rate


@torch.no_grad()
def _eval(model, X, vol_y, jump_y, jump_loss_fn):
    """Returns (vol_mse, jump_bce, pred_vol, pred_jump_prob) on the given
    split. vol_mse, not RMSE - take sqrt when reporting."""
    model.eval()
    pred_vol, pred_jump_logit = model(X.to(DEVICE))
    vol_y_dev, jump_y_dev = vol_y.to(DEVICE), jump_y.to(DEVICE)
    vol_mse = torch.mean((pred_vol - vol_y_dev) ** 2).item()
    jump_bce = jump_loss_fn(pred_jump_logit, jump_y_dev).item()
    pred_jump_prob = torch.sigmoid(pred_jump_logit).cpu()
    return vol_mse, jump_bce, pred_vol.cpu(), pred_jump_prob


def train_one_seed(seed, hidden_size=32, jump_loss_weight=0.01, max_epochs=100,
                    patience=10, lr_patience=5, min_epochs=20, batch_size=64, verbose=False):
    """Trains one BetaAgent instance for one seed. Returns
    (model, test_vol_rmse, test_jump_bce, test_pred_vol, test_pred_jump_prob,
    test_vol_y, test_jump_y).

    min_epochs guards against a real failure mode found while testing:
    val_jump_bce is genuinely noisy in the first several epochs, before
    the model has learned much of anything, and volatility MSE takes
    ~10-20 epochs to collapse to its converged value (see BetaAgent's
    docstring, and open finding 1 - this collapse is the model
    exploiting the target's overlap with its own input). Without a
    warm-up floor, a lucky noisy dip in jump_bce at epoch 0 can get
    selected as "best" before volatility has converged at all, which is
    exactly what happened on seed 0 during testing: it stopped after 11
    epochs having kept epoch 0's weights, with test vol RMSE of 0.47
    instead of the ~0.02 every other seed reached.
    """
    set_seed(seed)

    train_X, train_vol, train_jump = load_beta_split("train")
    val_X, val_vol, val_jump = load_beta_split("val")
    test_X, test_vol, test_jump = load_beta_split("test")

    pos_weight = torch.tensor(compute_pos_weight(train_jump), device=DEVICE)

    train_loader = DataLoader(
        TensorDataset(train_X, train_vol, train_jump), batch_size=batch_size, shuffle=True
    )

    model = models.BetaAgent(input_size=train_X.shape[-1], hidden_size=hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience
    )

    vol_loss_fn = nn.MSELoss()
    jump_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, vol_b, jump_b in train_loader:
            xb, vol_b, jump_b = xb.to(DEVICE), vol_b.to(DEVICE), jump_b.to(DEVICE)
            optimizer.zero_grad()
            pred_vol, pred_jump_logit = model(xb)
            vol_loss = vol_loss_fn(pred_vol, vol_b)
            jump_loss = jump_loss_fn(pred_jump_logit, jump_b)
            loss = vol_loss + jump_loss_weight * jump_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_vol_mse, val_jump_bce, _, _ = _eval(model, val_X, val_vol, val_jump, jump_loss_fn)
        # Checkpoint selection tracks val_jump_bce alone, not the combined
        # weighted loss. Volatility MSE converges to a good value almost
        # immediately (within ~10-20 epochs) regardless of which epoch
        # gets picked - a symptom of open finding 1's structural target
        # overlap, not genuine learning - while jump risk is the genuinely
        # hard, still-improving objective where checkpoint choice matters.
        # The combined metric used previously was already >99% driven by
        # the jump term in practice (the scale mismatch this file's
        # docstring flags), so this makes explicit what was already
        # happening rather than changing the actual selection behaviour.
        scheduler.step(val_jump_bce)

        if val_jump_bce < best_val_loss:
            best_val_loss = val_jump_bce
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"  seed {seed} epoch {epoch:>3}: val_vol_mse={val_vol_mse:.6f} "
                f"(rmse={val_vol_mse ** 0.5:.5f})  val_jump_bce={val_jump_bce:.4f}"
            )

        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    test_vol_mse, test_jump_bce, test_pred_vol, test_pred_jump_prob = _eval(
        model, test_X, test_vol, test_jump, jump_loss_fn
    )
    return (
        model,
        test_vol_mse ** 0.5,
        test_jump_bce,
        test_pred_vol.numpy(),
        test_pred_jump_prob.numpy(),
        test_vol.numpy(),
        test_jump.numpy(),
    )


def run_multi_seed(seeds=SEEDS, verbose=False, **kwargs):
    vol_rmses, jump_bces = [], []
    for seed in seeds:
        _, vol_rmse, jump_bce, _, _, _, _ = train_one_seed(seed, verbose=verbose, **kwargs)
        vol_rmses.append(vol_rmse)
        jump_bces.append(jump_bce)
        print(f"seed {seed}: test vol RMSE={vol_rmse:.5f}  test jump BCE={jump_bce:.4f}")

    vol_rmses, jump_bces = np.array(vol_rmses), np.array(jump_bces)
    print(f"\nvol RMSE mean +/- std: {vol_rmses.mean():.5f} +/- {vol_rmses.std():.5f}")
    print(f"jump BCE mean +/- std: {jump_bces.mean():.4f} +/- {jump_bces.std():.4f}")
    return vol_rmses, jump_bces


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    print("Beta Agent (volatility + jump-risk forecaster)")
    run_multi_seed(verbose=True)
