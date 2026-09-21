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


def naive_jump_bce(train_jump, eval_jump, pos_weight):
    """BCE of always predicting the training set's base rate for every
    single day, regardless of input - the same idea as the MAPE
    naive-baseline check in train.py/metrics.py, applied to the jump
    head. If the trained model's BCE isn't meaningfully better than
    this, it hasn't learned anything beyond the class prior."""
    base_rate = train_jump.mean().item()
    logit_value = float(np.log(base_rate / (1 - base_rate)))
    constant_logits = torch.full_like(eval_jump, logit_value)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(constant_logits, eval_jump).item()


@torch.no_grad()
def _eval(model, X, vol_y, jump_y, jump_loss_fn):
    """Returns (vol_mse, jump_bce, pred_vol, pred_jump_prob, pred_jump_logit)
    on the given split. vol_mse, not RMSE - take sqrt when reporting.
    pred_jump_logit is the RAW (pre-sigmoid) score - needed by Platt
    scaling below, which is defined on the logit, not the probability."""
    model.eval()
    pred_vol, pred_jump_logit = model(X.to(DEVICE))
    vol_y_dev, jump_y_dev = vol_y.to(DEVICE), jump_y.to(DEVICE)
    vol_mse = torch.mean((pred_vol - vol_y_dev) ** 2).item()
    jump_bce = jump_loss_fn(pred_jump_logit, jump_y_dev).item()
    pred_jump_prob = torch.sigmoid(pred_jump_logit).cpu()
    return vol_mse, jump_bce, pred_vol.cpu(), pred_jump_prob, pred_jump_logit.cpu()


def train_one_seed(seed, hidden_size=32, jump_loss_weight=0.01, max_epochs=100,
                    patience=10, lr_patience=5, min_epochs=20, batch_size=64, verbose=False):
    """Trains one BetaAgent instance for one seed. Returns a dict - see
    the return statement at the end of this function for the keys.

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

        val_vol_mse, val_jump_bce, _, _, _ = _eval(model, val_X, val_vol, val_jump, jump_loss_fn)
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
    val_vol_mse, val_jump_bce, val_pred_vol, val_pred_jump_prob, val_pred_jump_logit = _eval(
        model, val_X, val_vol, val_jump, jump_loss_fn
    )
    test_vol_mse, test_jump_bce, test_pred_vol, test_pred_jump_prob, test_pred_jump_logit = _eval(
        model, test_X, test_vol, test_jump, jump_loss_fn
    )
    # Returned as a dict rather than a growing positional tuple - this
    # function's return value has been extended three times now
    # (per-day predictions for the DM test equivalent, then the naive
    # baseline, now Platt scaling), and a dict keeps each addition
    # self-describing at the call site instead of one more silent
    # positional slot to get wrong.
    return {
        "model": model,
        "test_vol_rmse": test_vol_mse ** 0.5,
        "test_jump_bce": test_jump_bce,
        "test_pred_vol": test_pred_vol.numpy(),
        "test_pred_jump_prob": test_pred_jump_prob.numpy(),
        "test_pred_jump_logit": test_pred_jump_logit.numpy(),
        "test_vol_y": test_vol.numpy(),
        "test_jump_y": test_jump.numpy(),
        "val_pred_jump_logit": val_pred_jump_logit.numpy(),
        "val_jump_y": val_jump.numpy(),
    }


def run_multi_seed(seeds=SEEDS, verbose=False, **kwargs):
    """Returns (vol_rmses, jump_bces, pooled_test_probs, pooled_test_labels,
    all_results) - all_results is the full list of per-seed dicts from
    train_one_seed, kept around so downstream analysis (Platt scaling
    calibration below) can reuse the already-trained models' val/test
    logits without retraining from scratch."""
    vol_rmses, jump_bces = [], []
    all_probs, all_labels = [], []
    all_results = []
    for seed in seeds:
        result = train_one_seed(seed, verbose=verbose, **kwargs)
        all_results.append(result)
        vol_rmses.append(result["test_vol_rmse"])
        jump_bces.append(result["test_jump_bce"])
        all_probs.append(result["test_pred_jump_prob"])
        all_labels.append(result["test_jump_y"])
        print(
            f"seed {seed}: test vol RMSE={result['test_vol_rmse']:.5f}  "
            f"test jump BCE={result['test_jump_bce']:.4f}"
        )

    vol_rmses, jump_bces = np.array(vol_rmses), np.array(jump_bces)
    print(f"\nvol RMSE mean +/- std: {vol_rmses.mean():.5f} +/- {vol_rmses.std():.5f}")
    print(f"jump BCE mean +/- std: {jump_bces.mean():.4f} +/- {jump_bces.std():.4f}")

    _, _, train_jump = load_beta_split("train")
    _, _, test_jump = load_beta_split("test")
    pos_weight = torch.tensor(compute_pos_weight(train_jump), device=DEVICE)
    naive_bce = naive_jump_bce(train_jump, test_jump.to(DEVICE), pos_weight)
    print(f"naive baseline (always predict the {train_jump.mean().item():.4%} base rate): jump BCE={naive_bce:.4f}")

    pooled_probs = np.concatenate(all_probs)
    pooled_labels = np.concatenate(all_labels)
    return vol_rmses, jump_bces, pooled_probs, pooled_labels, all_results


def calibration_check(probs, labels, n_buckets=4):
    """Coarse reliability check: for each predicted-probability bucket,
    compares the average predicted probability against the actual
    observed jump frequency in that bucket. Pooled across all 5 seeds'
    test predictions (passed in via probs/labels) rather than one seed,
    and using few, quantile-based buckets rather than a fine-grained
    curve - the test set has only ~260 real days at a ~2.7% jump rate,
    so roughly 7 real jump-days exist per seed. Pooling across 5 seeds
    and keeping buckets coarse is what makes this check meaningful
    rather than pure noise; a fine-grained reliability diagram still
    wouldn't be trustworthy even pooled this way.
    """
    n_real_jump_days = int(round(labels.sum() / len(SEEDS)))
    print(
        f"\nCalibration check, pooled across {len(SEEDS)} seeds "
        f"({len(probs)} predictions, ~{n_real_jump_days} real jump-days per seed):"
    )

    edges = np.quantile(probs, np.linspace(0, 1, n_buckets + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    print(f"{'bucket (predicted prob.)':<28} {'n':>6} {'mean predicted':>16} {'actual freq':>14}")
    for i in range(n_buckets):
        mask = (probs >= edges[i]) & (probs <= edges[i + 1]) if i == n_buckets - 1 else (
            (probs >= edges[i]) & (probs < edges[i + 1])
        )
        n = int(mask.sum())
        mean_pred = probs[mask].mean() if n > 0 else float("nan")
        actual_freq = labels[mask].mean() if n > 0 else float("nan")
        label = f"[{edges[i]:.4f}, {edges[i+1]:.4f})"
        print(f"{label:<28} {n:>6} {mean_pred:>16.4f} {actual_freq:>14.4f}")


def fit_platt_scaling(logits, labels):
    """Fits Platt scaling: calibrated_prob = sigmoid(A * logit + B),
    with A and B found by logistic regression. Operates on the raw
    logit, not the already-sigmoided probability - that's the standard
    Platt scaling formulation, and it avoids the numerical issues of
    trying to invert a sigmoid for probabilities near 0 or 1."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit(logits.reshape(-1, 1), labels)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt_scaling(logits, A, B):
    return 1 / (1 + np.exp(-(A * logits + B)))


def platt_recalibrate(all_results, n_buckets=4):
    """Fits Platt scaling separately PER SEED, on that seed's own
    validation predictions, then applies it to that seed's own test
    predictions - fitting and evaluating on different splits, same
    discipline as everywhere else in this project. Each seed's model
    has its own specific miscalibration, so correcting per-seed (rather
    than pooling raw logits across seeds before fitting) is what keeps
    each correction matched to the model that actually produced it.
    Recalibrated test predictions are pooled afterward for the
    calibration_check.
    """
    all_calibrated_probs, all_labels = [], []
    for i, result in enumerate(all_results):
        A, B = fit_platt_scaling(result["val_pred_jump_logit"], result["val_jump_y"])
        calibrated = apply_platt_scaling(result["test_pred_jump_logit"], A, B)
        all_calibrated_probs.append(calibrated)
        all_labels.append(result["test_jump_y"])
        print(f"seed {i}: Platt scaling A={A:.4f}, B={B:.4f}")

    pooled_calibrated = np.concatenate(all_calibrated_probs)
    pooled_labels = np.concatenate(all_labels)
    print("\nCalibration check AFTER Platt scaling:")
    calibration_check(pooled_calibrated, pooled_labels, n_buckets=n_buckets)
    return pooled_calibrated, pooled_labels


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    print("Beta Agent (volatility + jump-risk forecaster)")
    _, _, pooled_probs, pooled_labels, all_results = run_multi_seed(verbose=True)
    calibration_check(pooled_probs, pooled_labels)
    platt_recalibrate(all_results)
