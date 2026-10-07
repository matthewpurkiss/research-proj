"""
direction_baselines.py

Is BTC's next-day direction predictable at all from this kind of data, or is
the Alpha Agent's ~50% hit rate a model failure? Compares the Alpha Agent's
directional hit rate with untrained rules on exactly the days the backtests
use: 2024 validation, 2025 test, and the 2026 forward period.

Rules (nothing fitted or tuned - each just counts how often it was right):
  always up        predict "up" every day
  momentum N       tomorrow's direction = sign of the last N days' summed return
                   (N = 1, 5, 20); uses only returns up to and including today
  mean reversion   tomorrow's direction = opposite of today's

Each hit rate gets a 95% Wilson confidence interval. With ~200 days per period
an interval is roughly +/- 7 points, so a rule needs about 57% before it can be
told apart from a coin flip; pooling the three periods (~600 days) narrows that.

Run from Code/ after forward_test.py has built data/forward_2026/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
import forward_test as ft
import run_backtest as rb
import signals
import splits

RESULTS_DIR = rb.RESULTS_DIR
FIGURES_DIR = rb.FIGURES_DIR
Z = 1.96


def wilson_interval(hits, n):
    p = hits / n
    centre = (p + Z ** 2 / (2 * n)) / (1 + Z ** 2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / (1 + Z ** 2 / n)
    return centre - half, centre + half


def score(pred_sign, actual):
    """Hits and count, skipping days with no prediction (sign 0) or no move."""
    keep = (pred_sign != 0) & (actual != 0)
    return int(np.sum(np.sign(pred_sign[keep]) == np.sign(actual[keep]))), int(keep.sum())


def rule_predictions(returns, dates):
    """returns: btc_log_return by date (today's return at row t). Predictions
    for day t+1 made at day t, using only returns up to and including t."""
    preds = {"Always up": np.ones(len(dates))}
    for n in (1, 5, 20):
        preds[f"Momentum {n}d"] = np.sign(returns.rolling(n).sum().reindex(dates).to_numpy())
    preds["Mean reversion"] = -np.sign(returns.reindex(dates).to_numpy())
    return preds


def forward_period():
    """2026 days and Alpha Agent predictions, from data/forward_2026 (built by
    forward_test.py), scaled with the frozen scaler - read only, not rebuilt."""
    aligned = pd.read_csv(ft.FORWARD_ALIGNED, index_col="Date", parse_dates=True)
    scaler = pd.read_csv(ft.FROZEN_SCALER, index_col=0)
    scaled = splits.apply_scaler(aligned, scaler["mean"], scaler["std"])
    w = {}
    for agent, lookback in (("alpha", config.ALPHA_LOOKBACK), ("beta", config.BETA_LOOKBACK)):
        X, y_alpha, _, _, end_dates = splits.make_windows(scaled, lookback)
        w[agent] = (torch.tensor(X, dtype=torch.float32), pd.DatetimeIndex(end_dates), pd.Series(y_alpha, index=end_dates))
    dates = w["alpha"][1]
    dates = dates[(dates >= ft.FORWARD_START) & (dates <= ft.FORWARD_END)]
    alpha_preds = [
        signals.from_windows(signals.load_models(s), w["alpha"][0], w["alpha"][1], w["beta"][0], w["beta"][1], dates)["pred_return"].to_numpy()
        for s in rb.SEEDS
    ]
    return dates, w["alpha"][2].reindex(dates).to_numpy(), aligned["btc_log_return"], alpha_preds


def frozen_period(split):
    data = np.load(os.path.join(signals.DATA_DIR, f"alpha_{split}_windows.npz"), allow_pickle=True)
    actual = pd.Series(data["y_alpha"], index=pd.to_datetime(data["end_dates"]))
    sig = {s: signals.build(s, split) for s in rb.SEEDS}
    dates = sig[0].index
    returns = pd.read_csv(os.path.join(signals.DATA_DIR, "aligned_dataset.csv"), index_col="Date", parse_dates=True)["btc_log_return"]
    return dates, actual.reindex(dates).to_numpy(), returns, [sig[s]["pred_return"].to_numpy() for s in rb.SEEDS]


def plot(table):
    periods = ["2024 (val)", "2025 (test)", "2026 (forward)", "All three"]
    methods = list(dict.fromkeys(table["method"]))
    x = np.arange(len(methods))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, period in enumerate(periods):
        sub = table[table["period"] == period].set_index("method").reindex(methods)
        rate = sub["hit_rate"].to_numpy() * 100
        err = np.vstack([rate - sub["ci_low"].to_numpy() * 100, sub["ci_high"].to_numpy() * 100 - rate])
        ax.bar(x + (i - 1.5) * width, rate, width, yerr=err, capsize=2, label=period,
               alpha=1.0 if period == "All three" else 0.6)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Next-day direction hit rate (%)")
    ax.set_ylim(35, 70)
    ax.set_title("Can anything call BTC's next-day direction? Hit rate with 95% interval (dashed = coin flip)", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase5_direction_baselines.png"), dpi=200)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    periods = {
        "2024 (val)": frozen_period("val"),
        "2025 (test)": frozen_period("test"),
        "2026 (forward)": forward_period(),
    }

    rows = []
    pooled = {}
    for period, (dates, actual, returns, alpha_preds) in periods.items():
        methods = {"Alpha Agent (5-seed avg)": None, **rule_predictions(returns, dates)}
        for method, pred in methods.items():
            if pred is None:
                per_seed = [score(np.sign(p), actual) for p in alpha_preds]
                hits, n = sum(h for h, _ in per_seed), sum(c for _, c in per_seed)
                hits, n = hits / len(rb.SEEDS), n / len(rb.SEEDS)
            else:
                hits, n = score(pred, actual)
            lo, hi = wilson_interval(hits, n)
            rows.append({"period": period, "method": method, "hit_rate": hits / n, "ci_low": lo, "ci_high": hi, "days": n})
            ph, pn = pooled.get(method, (0, 0))
            pooled[method] = (ph + hits, pn + n)
        rows.append({"period": period, "method": "Share of up days", "hit_rate": np.mean(actual > 0),
                     "ci_low": np.nan, "ci_high": np.nan, "days": len(actual)})
    for method, (hits, n) in pooled.items():
        lo, hi = wilson_interval(hits, n)
        rows.append({"period": "All three", "method": method, "hit_rate": hits / n, "ci_low": lo, "ci_high": hi, "days": n})
    table = pd.DataFrame(rows)

    order = ["Alpha Agent (5-seed avg)", "Always up", "Momentum 1d", "Momentum 5d", "Momentum 20d", "Mean reversion"]
    lines = [
        "# Is BTC's next-day direction predictable? (Phase 5)",
        "",
        "Next-day direction hit rate on the backtest days, with 95% Wilson intervals. No rule is",
        "trained or tuned. Alpha Agent = average over its 5 seeds.",
        "",
        "| Method | 2024 (val) | 2025 (test) | 2026 (forward) | All three pooled |",
        "|---|---|---|---|---|",
    ]
    for method in order:
        cells = []
        for period in ["2024 (val)", "2025 (test)", "2026 (forward)", "All three"]:
            r = table[(table.method == method) & (table.period == period)].iloc[0]
            cells.append(f"{r.hit_rate:.1%} [{r.ci_low:.0%}, {r.ci_high:.0%}]")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    ups = [table[(table.method == "Share of up days") & (table.period == p)].iloc[0].hit_rate
           for p in ["2024 (val)", "2025 (test)", "2026 (forward)"]]
    lines += [
        f"| Share of up days | {ups[0]:.1%} | {ups[1]:.1%} | {ups[2]:.1%} | - |",
        "",
        "An interval that includes 50% means that method can't be told apart from a coin flip.",
        "",
        "Figure: figures/phase5_direction_baselines.png",
    ]

    with open(os.path.join(RESULTS_DIR, "direction_baselines.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    table.to_csv(os.path.join(RESULTS_DIR, "direction_baselines.csv"), index=False)
    plot(table[table.method != "Share of up days"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
