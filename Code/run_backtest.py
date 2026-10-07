"""
run_backtest.py

Phase 4 end-to-end backtest (proposal Sections 5.4, 5.6.3, 6.1.4).

  1. Load each seed's saved Alpha and Beta Agents and build their per-day
     signals for validation (2024) and test (2025) - see signals.py.
  2. Calibrate the Dynamic Risk Engine's settings (tau_j, tau_y, k, c, L_max)
     by grid search on VALIDATION, maximising Sharpe minus max drawdown
     averaged over the 5 seeds. One shared setting for all seeds, so the
     choice isn't tuned to any one seed's quirks.
  3. Do the same for the H3 ablation (fixed size, fixed stop, fixed leverage,
     no Beta Agent), so the comparison is tuned-vs-tuned, not tuned-vs-default.
  4. Run both, once each, on TEST and write results/backtest_report.md,
     results/test_metrics.csv, and the equity chart to
     ../figures/phase4_equity_curves.png (write-up figures live in figures/).

Seed s of the Alpha Agent is paired with seed s of the Beta Agent, giving 5
independent runs; liquidation rate = share of runs with at least one
liquidation (Section 5.6.3).

Run after: train.py --model alpha  and  train_beta.py  (they save checkpoints/).
"""

import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest
import risk_engine
import signals

SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join("..", "figures")

# Display only: the simulation runs on equity = 1.0 (Sharpe, drawdown and the
# grid search don't depend on account size), and every balance is scaled by
# this when reported. Ignores USD/ZAR moves and brokers' minimum trade sizes.
STARTING_CAPITAL = 1000.0
CURRENCY = "R"

DIRECTION_STRENGTH_GRID = [0.0, 1 / 3, 2 / 3]  # tau_y: skip none / weakest third / weakest two thirds
JUMP_VETO_GRID = [0.1, 0.25, 0.5, 1.0]  # 1.0 = never veto (Platt-scaled probs are < 1)
RISK_BUDGET_GRID = [0.005, 0.01, 0.02]  # fraction = k / pred_vol; pred_vol is ~0.02
STOP_MULT_GRID = [1.0, 2.0, 3.0]        # stop = c * pred_vol
LEVERAGE_CAP_GRID = [2.0, 5.0, 10.0]

FIXED_FRACTION_GRID = [0.25, 0.5, 1.0]
FIXED_STOP_GRID = [0.02, 0.04, 0.06]
FIXED_LEVERAGE_GRID = [2.0, 5.0, 10.0]


def objective(m):
    return m["sharpe"] - m["max_drawdown"]


def dynamic_decider(p):
    return lambda r: risk_engine.dynamic_action(r.pred_return, r.pred_strength, r.pred_vol, r.jump_prob, p)


def fixed_decider(p):
    return lambda r: risk_engine.fixed_action(r.pred_return, r.pred_strength, p)


def run_all_seeds(params, make_decider, sigs, market):
    return {s: backtest.run(sigs[s], make_decider(params), market=market) for s in SEEDS}


def grid_search(candidates, make_decider, sigs, market):
    best_score, best_params = -np.inf, None
    for p in candidates:
        runs = run_all_seeds(p, make_decider, sigs, market)
        score = float(np.mean([objective(r["metrics"]) for r in runs.values()]))
        if score > best_score:
            best_score, best_params = score, p
    return best_params, best_score


def buy_and_hold(dates, market):
    """1x long BTC held throughout, as a reference line only."""
    closes = [market[dates[0]][0]] + [market[t][1][-1, 3] for t in dates]
    equity = pd.Series(np.array(closes) / closes[0], index=[dates[0]] + [t + pd.offsets.BDay(1) for t in dates])
    trades = pd.DataFrame({"direction": 1, "liquidated": False}, index=dates)
    return equity, backtest.metrics(equity, trades)


def summarise(runs):
    rows = pd.DataFrame({s: r["metrics"] for s, r in runs.items()}).T.astype(
        {"n_trades": int, "n_liquidations": int})
    summary = {col: (rows[col].mean(), rows[col].std()) for col in
               ["total_return", "max_drawdown", "sharpe", "sortino", "n_trades"]}
    summary["liquidation_rate"] = float((rows["n_liquidations"] > 0).mean())
    finals = np.array([r["equity"].iloc[-1] for r in runs.values()]) * STARTING_CAPITAL
    lows = np.array([r["equity"].min() for r in runs.values()]) * STARTING_CAPITAL
    summary["final_balance"] = (finals.mean(), finals.std(ddof=1))
    summary["lowest_balance"] = (lows.mean(), lows.std(ddof=1))
    return rows, summary


def markdown_table(df):
    header = "| seed | " + " | ".join(df.columns) + " |"
    divider = "|" + "---|" * (len(df.columns) + 1)
    rows = [
        f"| {idx} | " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for idx, row in zip(df.index, df.itertuples(index=False))
    ]
    return "\n".join([header, divider] + rows)


def money(x):
    return f"{CURRENCY}{x:,.0f}"


def write_report(dyn_params, dyn_val, fix_params, fix_val, dyn_rows, dyn_sum, fix_rows, fix_sum, bh_equity, bh_metrics):
    def fmt(summary, key, pct=False):
        mean, std = summary[key]
        return f"{mean:.1%} ± {std:.1%}" if pct else f"{mean:.3f} ± {std:.3f}"

    def fmt_money(summary, key):
        mean, std = summary[key]
        return f"{money(mean)} ± {money(std)}"

    lines = [
        "# Backtest report (Phase 4)",
        "",
        "Test period: 2025 weekdays with both an Alpha and a Beta window. 5 runs (Alpha seed s",
        "paired with Beta seed s). Settings tuned on 2024 validation only. Costs: "
        f"{backtest.DEFAULT_COST_RATE:.2%} of notional per round trip.",
        "",
        "## Chosen settings (validation grid search, objective = Sharpe − max drawdown)",
        "",
        f"- Dynamic Risk Engine: tau_j={dyn_params.tau_j}, tau_y={dyn_params.tau_y:.2f}, "
        f"k={dyn_params.k}, c={dyn_params.c}, L_max={dyn_params.l_max} (validation objective {dyn_val:.3f})",
        f"- Fixed-rule ablation: tau_y={fix_params.tau_y:.2f}, fraction={fix_params.fraction}, "
        f"stop={fix_params.stop_distance}, leverage={fix_params.leverage} (validation objective {fix_val:.3f})",
        "",
        "## Test results (mean ± std over 5 runs)",
        "",
        "| Metric | Dynamic Risk Engine | Fixed-rule ablation | Buy & hold BTC (1x) |",
        "|---|---|---|---|",
        f"| Final balance (from {money(STARTING_CAPITAL)}) | {fmt_money(dyn_sum, 'final_balance')} | {fmt_money(fix_sum, 'final_balance')} | {money(bh_equity.iloc[-1] * STARTING_CAPITAL)} |",
        f"| Lowest balance during test | {fmt_money(dyn_sum, 'lowest_balance')} | {fmt_money(fix_sum, 'lowest_balance')} | {money(bh_equity.min() * STARTING_CAPITAL)} |",
        f"| Total return (multiple) | {fmt(dyn_sum, 'total_return')} | {fmt(fix_sum, 'total_return')} | {bh_metrics['total_return']:.3f} |",
        f"| Max drawdown | {fmt(dyn_sum, 'max_drawdown', True)} | {fmt(fix_sum, 'max_drawdown', True)} | {bh_metrics['max_drawdown']:.1%} |",
        f"| Sharpe | {fmt(dyn_sum, 'sharpe')} | {fmt(fix_sum, 'sharpe')} | {bh_metrics['sharpe']:.3f} |",
        f"| Sortino | {fmt(dyn_sum, 'sortino')} | {fmt(fix_sum, 'sortino')} | {bh_metrics['sortino']:.3f} |",
        f"| Trades | {fmt(dyn_sum, 'n_trades')} | {fmt(fix_sum, 'n_trades')} | - |",
        f"| Liquidation rate | {dyn_sum['liquidation_rate']:.0%} | {fix_sum['liquidation_rate']:.0%} | - |",
        "",
        f"Balances assume a {money(STARTING_CAPITAL)} starting account; every result scales in proportion.",
        "They ignore USD/ZAR moves (BTC is priced in dollars) and brokers' minimum trade",
        "sizes, which a small account may fall below. Statistical comparison (H3): h3_tests.py.",
        "",
        "## Per-run test metrics",
        "",
        "Dynamic Risk Engine:",
        "",
        markdown_table(dyn_rows),
        "",
        "Fixed-rule ablation:",
        "",
        markdown_table(fix_rows),
    ]
    with open(os.path.join(RESULTS_DIR, "backtest_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot(dyn_runs, fix_runs, bh_equity):
    fig, ax = plt.subplots(figsize=(10, 5))
    cap = STARTING_CAPITAL
    for i, s in enumerate(SEEDS):
        ax.plot(dyn_runs[s]["equity"] * cap, color="tab:blue", alpha=0.7, label="Dynamic Risk Engine" if i == 0 else None)
        ax.plot(fix_runs[s]["equity"] * cap, color="tab:orange", alpha=0.7, label="Fixed-rule ablation" if i == 0 else None)
    ax.plot(bh_equity * cap, color="black", linestyle="--", label="Buy & hold BTC (1x)")
    ax.axhline(cap, color="grey", linewidth=0.5)
    ax.set_ylabel(f"Account balance ({CURRENCY}, starting {money(cap)})")
    ax.set_title("Test period (2025): one line per seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase4_equity_curves.png"), dpi=200)


def tune_and_test(verbose=True):
    """Grid-searches both engines on validation, then runs each once on test.
    Shared by main() below and h3_tests.py, so both see exactly the same runs."""
    val_sigs = {s: signals.build(s, "val") for s in SEEDS}
    test_sigs = {s: signals.build(s, "test") for s in SEEDS}
    val_market = backtest.prepare_market(val_sigs[0].index)
    test_market = backtest.prepare_market(test_sigs[0].index)

    dynamic_candidates = [
        risk_engine.DynamicRiskParams(tau_j, tau_y, k, c, l_max)
        for tau_j, tau_y, k, c, l_max in itertools.product(
            JUMP_VETO_GRID, DIRECTION_STRENGTH_GRID, RISK_BUDGET_GRID, STOP_MULT_GRID, LEVERAGE_CAP_GRID)
    ]
    fixed_candidates = [
        risk_engine.FixedRiskParams(tau_y, frac, stop, lev)
        for tau_y, frac, stop, lev in itertools.product(
            DIRECTION_STRENGTH_GRID, FIXED_FRACTION_GRID, FIXED_STOP_GRID, FIXED_LEVERAGE_GRID)
    ]

    if verbose:
        print(f"validation days: {len(val_sigs[0])}, test days: {len(test_sigs[0])}")
        print(f"grid search: {len(dynamic_candidates)} dynamic and {len(fixed_candidates)} fixed settings on validation...")
    dyn_params, dyn_val = grid_search(dynamic_candidates, dynamic_decider, val_sigs, val_market)
    fix_params, fix_val = grid_search(fixed_candidates, fixed_decider, val_sigs, val_market)
    if verbose:
        print(f"dynamic: {dyn_params}  (val objective {dyn_val:.3f})")
        print(f"fixed:   {fix_params}  (val objective {fix_val:.3f})")

    bh_equity, bh_metrics = buy_and_hold(test_sigs[0].index, test_market)
    return {
        "dyn_params": dyn_params, "dyn_val": dyn_val,
        "fix_params": fix_params, "fix_val": fix_val,
        "dyn_runs": run_all_seeds(dyn_params, dynamic_decider, test_sigs, test_market),
        "fix_runs": run_all_seeds(fix_params, fixed_decider, test_sigs, test_market),
        "bh_equity": bh_equity, "bh_metrics": bh_metrics,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    r = tune_and_test()
    dyn_rows, dyn_sum = summarise(r["dyn_runs"])
    fix_rows, fix_sum = summarise(r["fix_runs"])

    write_report(r["dyn_params"], r["dyn_val"], r["fix_params"], r["fix_val"],
                 dyn_rows, dyn_sum, fix_rows, fix_sum, r["bh_equity"], r["bh_metrics"])
    pd.concat({"dynamic": dyn_rows, "fixed": fix_rows}).to_csv(os.path.join(RESULTS_DIR, "test_metrics.csv"))
    with open(os.path.join(RESULTS_DIR, "chosen_params.json"), "w") as f:
        json.dump({"dynamic": r["dyn_params"].__dict__, "fixed": r["fix_params"].__dict__}, f, indent=2)
    plot(r["dyn_runs"], r["fix_runs"], r["bh_equity"])

    print(open(os.path.join(RESULTS_DIR, "backtest_report.md")).read())


if __name__ == "__main__":
    main()
