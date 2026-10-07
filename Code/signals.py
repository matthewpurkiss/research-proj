"""
signals.py

Loads one seed's saved Alpha Agent and Beta Agent (checkpoints/, written by
train.py --model alpha and train_beta.py) and produces the three per-day
inputs the Dynamic Risk Engine consumes, on the validation or test split:

  pred_return  Alpha Agent's predicted next-day BTC log-return
  pred_strength  share of that model's previous STRENGTH_WINDOW predictions
               whose size |pred_return| is smaller than today's (0 to 1).
               The five seeds predict on scales up to ~5x apart, and each
               seed's scale drifts between 2024 and 2025, so the engine's
               "too small to trade" rule compares a prediction with the
               model's own recent output rather than a fixed number. Uses
               only past predictions; the window runs back across split
               boundaries (train -> val -> test), which is fine causally:
               at day t the model's earlier predictions are already known.
  pred_vol     Beta Agent's predicted next-day volatility (|log-return|)
  jump_prob    Beta Agent's jump probability, Platt-scaled with the (A, B)
               fitted on that seed's validation logits by train_beta.py

Indexed by decision date t. The Alpha Agent's 60-day lookback is longer than
the Beta Agent's 30, so only dates where BOTH have a window are kept - the
first ~59 weekdays of each split have no Alpha window (splits.py never lets a
window reach into the previous split).
"""

import os

import numpy as np
import pandas as pd
import torch

import models
import train
import train_beta

DATA_DIR = os.path.join("data", "processed")
STRENGTH_WINDOW = 60
SPLITS = ("train", "val", "test")


def _load_windows(agent, split):
    data = np.load(os.path.join(DATA_DIR, f"{agent}_{split}_windows.npz"), allow_pickle=True)
    X = torch.tensor(data["X"], dtype=torch.float32)
    return X, pd.to_datetime(data["end_dates"])


def prediction_strength(pred_return, window=STRENGTH_WINDOW):
    """For each day, the share of the previous `window` predictions with a
    smaller |pred_return|. Today's own prediction is excluded; NaN until a
    full window of history exists (only ever the start of train)."""
    size = pred_return.abs().to_numpy()
    strength = np.full(len(size), np.nan)
    for i in range(window, len(size)):
        strength[i] = np.mean(size[i - window : i] < size[i])
    return pd.Series(strength, index=pred_return.index)


def load_models(seed, direction_model="alpha"):
    """Seed's saved direction forecaster (direction_model: "alpha" or any key of
    train.BASELINES) and Beta Agent with its Platt (A, B)."""
    _, model_fn, feature_indices = train.BASELINES[direction_model]
    alpha = model_fn()
    alpha.load_state_dict(torch.load(train.checkpoint_path(direction_model, seed), map_location="cpu"))
    alpha.eval()

    beta_ckpt = torch.load(train_beta.checkpoint_path(seed), map_location="cpu")
    beta = models.BetaAgent(input_size=19, hidden_size=32)
    beta.load_state_dict(beta_ckpt["state_dict"])
    beta.eval()
    return alpha, feature_indices, beta, (beta_ckpt["platt_A"], beta_ckpt["platt_B"])


@torch.no_grad()
def from_windows(loaded, alpha_X, alpha_dates, beta_X, beta_dates, keep_dates):
    """Signals for keep_dates. alpha_X/alpha_dates should cover enough earlier
    windows for pred_strength's look-back (STRENGTH_WINDOW predictions)."""
    alpha, feature_indices, beta, (platt_A, platt_B) = loaded
    if feature_indices is not None:
        alpha_X = alpha_X[:, :, feature_indices]
    all_preds = pd.Series(alpha(alpha_X).numpy(), index=alpha_dates).sort_index()
    strength = prediction_strength(all_preds)

    alpha_df = pd.DataFrame({
        "pred_return": all_preds.reindex(keep_dates),
        "pred_strength": strength.reindex(keep_dates),
    })
    pred_vol, jump_logit = beta(beta_X)
    jump_prob = train_beta.apply_platt_scaling(jump_logit.numpy(), platt_A, platt_B)
    beta_df = pd.DataFrame({"pred_vol": pred_vol.numpy(), "jump_prob": jump_prob}, index=beta_dates)
    return alpha_df.join(beta_df, how="inner")[["pred_return", "pred_strength", "pred_vol", "jump_prob"]]


def build(seed, split, direction_model="alpha"):
    """Signals for one split of the frozen dataset. direction_model: which
    forecaster supplies pred_return - "alpha" (the Alpha Agent, default) or any
    other key of train.BASELINES ("bilstm", "lstm", "gru"), for Phase 5's model
    ablations. Needs that model's checkpoints from train.py --model <name>."""
    windows = [_load_windows("alpha", s) for s in SPLITS]
    alpha_X = torch.cat([X for X, _ in windows])
    alpha_dates = pd.DatetimeIndex(np.concatenate([d for _, d in windows]))
    _, keep_dates = _load_windows("alpha", split)
    beta_X, beta_dates = _load_windows("beta", split)

    out = from_windows(load_models(seed, direction_model), alpha_X, alpha_dates, beta_X, beta_dates, keep_dates)
    assert out.notna().all().all(), f"missing signal values for seed {seed}, {split}"
    return out
