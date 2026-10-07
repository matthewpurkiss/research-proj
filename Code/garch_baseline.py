"""
garch_baseline.py

GARCH(1,1) sanity-check baseline for the Beta Agent, per Section 6.1.3.
Fits on the training split's BTC log-returns only, then produces
one-step-ahead volatility forecasts through validation and test using
those FIXED (never refit) parameters against the actual, expanding
return history - standard walk-forward, fixed-parameter evaluation, not
a full periodic refit.

Compares GARCH's forecast against next-day's ABSOLUTE RETURN (the
standard single-day realised-volatility proxy) - which is also the Beta
Agent's own target since the 2026-09-22 switch away from the 14-day
rolling std, so skill_scores.py can DM-test the two head to head.
"""

import os

import numpy as np
import pandas as pd
from arch import arch_model

import config

DATA_DIR = os.path.join("data", "processed")
SCALE = 100  # arch_model is numerically happier with returns as
             # percentage points than raw decimals - undone when reporting.


def load_raw_returns():
    """Raw (unstandardised) BTC log-returns, sliced into the same
    train/val/test boundaries splits.py uses - aligned_dataset.csv, not
    train.csv/val.csv/test.csv, since those hold the SCALED features."""
    aligned = pd.read_csv(
        os.path.join(DATA_DIR, "aligned_dataset.csv"), index_col="Date", parse_dates=True
    )
    returns = aligned["btc_log_return"]
    train = returns.loc[config.TRAIN_START : config.TRAIN_END]
    val = returns.loc[config.VAL_START : config.VAL_END]
    test = returns.loc[config.TEST_START : config.TEST_END]
    return train, val, test


def fit_and_forecast():
    """Fits GARCH(1,1) on the training split only, then produces
    one-step-ahead volatility forecasts for every day from the start of
    validation through the end of test, using those fixed parameters
    against the real, expanding return history (no refitting)."""
    train, val, test = load_raw_returns()

    train_scaled = train * SCALE
    full_scaled = pd.concat([train, val, test]) * SCALE

    am_train = arch_model(train_scaled, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    fitted = am_train.fit(disp="off")

    am_full = arch_model(full_scaled, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    forecasts = am_full.forecast(
        params=fitted.params, start=val.index[0], horizon=1, reindex=False
    )
    forecast_vol = np.sqrt(forecasts.variance.values[:, 0]) / SCALE
    forecast_series = pd.Series(forecast_vol, index=forecasts.variance.index)

    return fitted, forecast_series.loc[forecast_series.index.isin(val.index)], \
        forecast_series.loc[forecast_series.index.isin(test.index)]


def evaluate(forecast_vol, actual_returns):
    """Compares a one-step-ahead volatility forecast against next-day's
    realised |return| - the standard proxy for single-day realised
    volatility. forecast_vol at day t is a forecast FOR day t+1, so it's
    compared against actual_returns shifted back by one day."""
    actual_next_day = actual_returns.shift(-1).abs()
    common = forecast_vol.index.intersection(actual_next_day.dropna().index)
    error = forecast_vol.loc[common] - actual_next_day.loc[common]
    rmse = float(np.sqrt((error ** 2).mean()))
    mae = float(error.abs().mean())
    return rmse, mae


if __name__ == "__main__":
    fitted, val_forecast, test_forecast = fit_and_forecast()
    print(fitted.summary())

    train, val, test = load_raw_returns()
    val_rmse, val_mae = evaluate(val_forecast, val)
    test_rmse, test_mae = evaluate(test_forecast, test)

    print("\nGARCH(1,1), fit on train only, one-step-ahead forecasts:")
    print(f"  validation: RMSE={val_rmse:.5f}  MAE={val_mae:.5f}  (vs next-day |return|)")
    print(f"  test:       RMSE={test_rmse:.5f}  MAE={test_mae:.5f}  (vs next-day |return|)")
