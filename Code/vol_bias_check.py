"""
vol_bias_check.py

Phase 5 diagnostic: how well does the Beta Agent's volatility forecast rank
calm days against wild ones, and is it biased?

RMSE against |r(t+1)| is dominated by the noise in a single day's move, so it
says little about whether the forecast is useful to the Risk Engine, which
only needs it to rank days roughly correctly. On the 2025 backtest days (the
same days the trading results use), this reports:

  - mean forecast vs mean actual move (bias)
  - correlation between forecast and actual move
  - the actual average move within each third of days, sorted by forecast

The forecast is averaged over the 5 seeds. Nothing is trained or tuned.
Writes results/vol_bias_check.md. Run from Code/.
"""

import os

import numpy as np
import pandas as pd

import run_backtest
import signals

OUT_PATH = os.path.join("results", "vol_bias_check.md")


def main():
    sigs = {s: signals.build(s, "test") for s in run_backtest.SEEDS}
    pred = pd.concat([sigs[s]["pred_vol"] for s in run_backtest.SEEDS], axis=1).mean(axis=1)

    data = np.load(os.path.join(signals.DATA_DIR, "beta_test_windows.npz"), allow_pickle=True)
    actual = pd.Series(data["y_beta_vol"], index=pd.to_datetime(data["end_dates"])).reindex(pred.index)

    thirds = pd.qcut(pred, 3, labels=["calmest third", "middle third", "wildest third"])
    by_third = actual.groupby(thirds).mean() * 100

    lines = [
        "# Beta Agent volatility forecast: bias and ranking (2025 backtest days)",
        "",
        f"Days: {len(actual)}. Forecast = average of the 5 seeds.",
        "",
        "| | Value |",
        "|---|---|",
        f"| Mean forecast size of next day's move | {pred.mean() * 100:.2f}% |",
        f"| Mean actual size of next day's move | {actual.mean() * 100:.2f}% |",
        f"| Median actual move | {actual.median() * 100:.2f}% |",
        f"| Correlation, forecast vs actual | {pred.corr(actual):.2f} |",
        "",
        "Actual average move, days split into thirds by forecast:",
        "",
        "| Forecast group | Actual average move |",
        "|---|---|",
    ] + [f"| {name} | {value:.2f}% |" for name, value in by_third.items()]

    report = "\n".join(lines) + "\n"
    with open(OUT_PATH, "w") as f:
        f.write(report)
    print(report)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
