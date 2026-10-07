"""
model_ablation.py

Phase 5 step 5 (proposal Section 6.1.5: "ablations isolate the contribution of
attention and of the multivariate feature set"): does the Alpha Agent's design
matter once its forecasts go through the full framework, or only for forecast
error (Phase 2's H1/H2)?

The full framework - Beta Agent + Dynamic Risk Engine, unchanged - is run with
three different forecasters supplying the direction signal:

  alpha    Alpha Agent: multivariate Bi-LSTM + attention (the proposed model)
  bilstm   same Bi-LSTM stack, multivariate, NO attention   -> isolates attention
  lstm     vanilla LSTM on BTC returns only (univariate)     -> the univariate floor
           (also a different, simpler architecture, so this is not a pure test of
           the feature set alone - Phase 2's H1 test was the clean one)

Each version's Risk Engine settings are retuned on 2024 validation, then run
once on the 2025 test. Also reports each forecaster's directional hit rate on
the test days (share of days where the predicted sign matched the actual next-
day move) - the forecast property that matters most for trading.

Needs checkpoints from: train.py --model alpha / --model bilstm / --model lstm.
"""

import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest
import h3_tests
import risk_engine
import run_backtest as rb
import signals

RESULTS_DIR = rb.RESULTS_DIR
FIGURES_DIR = rb.FIGURES_DIR

FORECASTERS = {
    "alpha": "Alpha Agent (attention)",
    "bilstm": "Bi-LSTM, no attention",
    "lstm": "Univariate LSTM",
}


def dynamic_candidates():
    return [
        risk_engine.DynamicRiskParams(tj, ty, k, c, L)
        for tj, ty, k, c, L in itertools.product(
            rb.JUMP_VETO_GRID, rb.DIRECTION_STRENGTH_GRID, rb.RISK_BUDGET_GRID,
            rb.STOP_MULT_GRID, rb.LEVERAGE_CAP_GRID)
    ]


def actual_next_day_returns(dates):
    data = np.load(os.path.join(signals.DATA_DIR, "alpha_test_windows.npz"), allow_pickle=True)
    y = pd.Series(data["y_alpha"], index=pd.to_datetime(data["end_dates"]))
    return y.reindex(dates)


def hit_rate(pred, actual):
    keep = (pred != 0) & (actual != 0)
    return float(np.mean(np.sign(pred[keep]) == np.sign(actual[keep])))


def plot(table):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = list(table.index)
    x = np.arange(len(labels))
    colors = ["tab:blue", "tab:purple", "tab:grey"][: len(labels)]
    panels = [
        ("mdd_mean", "mdd_std", "Max drawdown (%) - lower is better", 100),
        ("sharpe_mean", "sharpe_std", "Sharpe - higher is better", 1),
        ("hit_rate_mean", "hit_rate_std", "Directional hit rate (%)", 100),
    ]
    for ax, (mean_col, std_col, title, scale) in zip(axes, panels):
        ax.bar(x, table[mean_col].astype(float) * scale, yerr=table[std_col].astype(float) * scale,
               color=colors, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[2].axhline(50, color="black", linestyle="--", linewidth=0.8, label="coin flip")
    axes[2].set_ylim(40, 60)
    axes[2].legend(fontsize=8)
    fig.suptitle("Forecaster ablation inside the full framework (2025 test, mean ± std over 5 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase5_model_ablation.png"), dpi=200)


def main(forecasters=FORECASTERS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    runs, chosen, hits = {}, {}, {}
    for name in forecasters:
        val_sigs = {s: signals.build(s, "val", direction_model=name) for s in rb.SEEDS}
        test_sigs = {s: signals.build(s, "test", direction_model=name) for s in rb.SEEDS}
        val_market = backtest.prepare_market(val_sigs[0].index)
        test_market = backtest.prepare_market(test_sigs[0].index)

        params, val_score = rb.grid_search(dynamic_candidates(), rb.dynamic_decider, val_sigs, val_market)
        chosen[name] = (params, val_score)
        runs[name] = rb.run_all_seeds(params, rb.dynamic_decider, test_sigs, test_market)

        actual = actual_next_day_returns(test_sigs[0].index).to_numpy()
        hits[name] = [hit_rate(test_sigs[s]["pred_return"].to_numpy(), actual) for s in rb.SEEDS]
        print(f"{name:7s} tuned: tau_j={params.tau_j}, tau_y={params.tau_y:.2f}, k={params.k}, "
              f"c={params.c}, L={params.l_max} (val objective {val_score:.3f})")

    ref_returns = {s: h3_tests.daily_returns(runs["alpha"][s]) for s in rb.SEEDS}
    rows = []
    for name, label in forecasters.items():
        m = pd.DataFrame({s: runs[name][s]["metrics"] for s in rb.SEEDS}).T
        row = {
            "forecaster": label,
            "mdd_mean": m["max_drawdown"].mean(), "mdd_std": m["max_drawdown"].std(),
            "sharpe_mean": m["sharpe"].mean(), "sharpe_std": m["sharpe"].std(),
            "final_balance": m["total_return"].mean() * rb.STARTING_CAPITAL,
            "trades": m["n_trades"].mean(),
            "hit_rate_mean": np.mean(hits[name]), "hit_rate_std": np.std(hits[name], ddof=1),
        }
        if name != "alpha":
            other = {s: h3_tests.daily_returns(runs[name][s]) for s in rb.SEEDS}
            _, pooled = h3_tests.bootstrap(ref_returns, other)
            obs = {k: float(np.mean([h3_tests.observed(ref_returns, other, s)[k] for s in rb.SEEDS]))
                   for k in ("dd_reduction", "sharpe_gain")}
            dd = h3_tests.summarise_samples(pooled["dd_reduction"], obs["dd_reduction"])
            sh = h3_tests.summarise_samples(pooled["sharpe_gain"], obs["sharpe_gain"])
            row.update({"alpha_dd_better_pp": dd["observed"] * 100, "dd_ci": (dd["ci_low"] * 100, dd["ci_high"] * 100),
                        "dd_p": dd["p"], "alpha_sharpe_better": sh["observed"],
                        "sharpe_ci": (sh["ci_low"], sh["ci_high"]), "sharpe_p": sh["p"]})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("forecaster")

    lines = [
        "# Forecaster ablation inside the full framework (Phase 5)",
        "",
        "Beta Agent + Dynamic Risk Engine unchanged; only the direction forecaster differs.",
        "Each version retuned on 2024 validation, run once on the 2025 test. Mean ± std over 5 seeds.",
        "",
        "| Forecaster | Max drawdown | Sharpe | Final balance (from R1,000) | Trades | Hit rate |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in table.iterrows():
        lines.append(
            f"| {label} | {r.mdd_mean:.1%} ± {r.mdd_std:.1%} | {r.sharpe_mean:.2f} ± {r.sharpe_std:.2f} "
            f"| {rb.money(r.final_balance)} | {r.trades:.0f} | {r.hit_rate_mean:.1%} ± {r.hit_rate_std:.1%} |"
        )
    if len(table) > 1:
        lines += [
            "",
            "## Is the Alpha Agent version better? (paired block bootstrap, 5-seed average)",
            "",
            "Positive = the Alpha Agent version is better. One-sided p.",
            "",
            "| Compared with | Drawdown lower by (pp) | 95% interval | p | Sharpe higher by | 95% interval | p |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, r in table.iloc[1:].iterrows():
            lines.append(
                f"| {label} | {r.alpha_dd_better_pp:+.1f} | [{r.dd_ci[0]:+.1f}, {r.dd_ci[1]:+.1f}] | {h3_tests.fmt_p(r.dd_p)} "
                f"| {r.alpha_sharpe_better:+.2f} | [{r.sharpe_ci[0]:+.2f}, {r.sharpe_ci[1]:+.2f}] | {h3_tests.fmt_p(r.sharpe_p)} |"
            )
    lines += [
        "",
        "Hit rate: share of test days where the forecast's sign matched the actual next-day move.",
        "The univariate LSTM also differs in architecture, so it isn't a pure test of the features.",
        "",
        "Figure: figures/phase5_model_ablation.png",
    ]

    with open(os.path.join(RESULTS_DIR, "model_ablation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    table.to_csv(os.path.join(RESULTS_DIR, "model_ablation.csv"))
    plot(table)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
