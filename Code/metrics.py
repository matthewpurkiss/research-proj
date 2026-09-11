"""
metrics.py

Section 5.6.2 needs MAPE comparable to Seabe et al.'s benchmark (Table
2.1), which is reported on BTC's closing PRICE - but our models predict
log-returns, not price levels. MAPE computed directly on log-returns
would be numerically unstable (they sit near zero, so a tiny absolute
error can blow up into a huge percentage) and wouldn't be on the same
scale as that benchmark anyway.

This reconstructs a predicted price from a predicted log-return:
    P_hat(t+1) = P(t) * exp(predicted_log_return)
using P(t) - today's real closing price, looked up from the aligned
raw data, not anything already log-transformed. The "actual" price for
comparison is reconstructed the same way from the real log-return,
rather than looked up separately by date - since alpha_target was
already built as exactly log(P(t+1)/P(t)) in build_dataset.py, this is
mathematically identical to looking up P(t+1) directly, without needing
to separately work out what "the next business day" is for every
end_date.
"""

import os

import numpy as np
import pandas as pd

import build_dataset

DATA_DIR = os.path.join("data", "processed")


def load_btc_close():
    """The real BTC close price, indexed by date, from the same
    alignment build_dataset.py itself uses - not the log-transformed
    feature column."""
    aligned = build_dataset._align(build_dataset.RAW_DIR)
    return aligned["btc_close"]


def load_test_end_dates(agent="alpha"):
    path = os.path.join(DATA_DIR, f"{agent}_test_windows.npz")
    data = np.load(path, allow_pickle=True)
    return pd.to_datetime(data["end_dates"])


def mape_on_reconstructed_price(end_dates, predicted_log_returns, actual_log_returns, btc_close=None):
    """Returns MAPE (%) between the price implied by the model's
    predicted log-return and the price implied by the real one, both
    reconstructed from today's real close - putting this on the same
    footing as Seabe et al.'s Table 2.1, rather than on the log-return's
    own near-zero, numerically unstable scale."""
    if btc_close is None:
        btc_close = load_btc_close()

    p_today = btc_close.loc[end_dates].to_numpy()
    predicted_price = p_today * np.exp(predicted_log_returns)
    actual_price = p_today * np.exp(actual_log_returns)

    ape = np.abs(actual_price - predicted_price) / np.abs(actual_price)
    return float(ape.mean() * 100)
