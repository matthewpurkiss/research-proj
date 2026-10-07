"""
stress_test.py

Phase 5 step 4: leverage stress test for H3's liquidation claim.

At the tuned 2x leverage neither engine is ever liquidated, so liquidation rate
can't separate them. This re-runs both at fixed leverage caps of 2x, 5x, 10x and
20x - retuning every other setting on 2024 validation at each level - and
reports drawdown, Sharpe, final balance and liquidations on the 2025 test, plus
the paired bootstrap of the drawdown and Sharpe differences (h3_tests.py) at each
level.

What to expect, and why (checked before writing this): a liquidation needs the
price to move 1/leverage against the trade before the stop-loss fires. Every stop
in the grids is tighter than that up to 10x, so the stop always fires first; the
only other route is a day that opens beyond the liquidation level, and BTC
trades 24/7 - it opened more than 2% away from the previous close on 3 of 2,922
days. So liquidations only appear once the liquidation distance is shorter than
some stops (20x -> 5%), and then for the DYNAMIC engine, whose stop widens with
forecast volatility (rule 4). That's a real design tension, not a bug. The
simulator also fills stops exactly at the stop price (no slippage), which
understates liquidation risk compared with live markets.

Dynamic engine at cap L: leverage = L * (1 - jump_prob) (rule 5), so its actual
leverage is at or a little below L. Fixed rules: exactly L.
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
LEVERAGE_LEVELS = [2.0, 5.0, 10.0, 20.0]


def tune_at_leverage(L, val_sigs, val_market):
    dyn_cands = [
        risk_engine.DynamicRiskParams(tj, ty, k, c, L)
        for tj, ty, k, c in itertools.product(
            rb.JUMP_VETO_GRID, rb.DIRECTION_STRENGTH_GRID, rb.RISK_BUDGET_GRID, rb.STOP_MULT_GRID)
    ]
    fix_cands = [
        risk_engine.FixedRiskParams(ty, f, st, L)
        for ty, f, st in itertools.product(rb.DIRECTION_STRENGTH_GRID, rb.FIXED_FRACTION_GRID, rb.FIXED_STOP_GRID)
    ]
    dyn_params, _ = rb.grid_search(dyn_cands, rb.dynamic_decider, val_sigs, val_market)
    fix_params, _ = rb.grid_search(fix_cands, rb.fixed_decider, val_sigs, val_market)
    return dyn_params, fix_params


def engine_summary(runs):
    m = pd.DataFrame({s: runs[s]["metrics"] for s in rb.SEEDS}).T
    return {
        "mdd_mean": m["max_drawdown"].mean(), "mdd_std": m["max_drawdown"].std(),
        "sharpe_mean": m["sharpe"].mean(), "sharpe_std": m["sharpe"].std(),
        "final_balance": m["total_return"].mean() * rb.STARTING_CAPITAL,
        "liquidations": int(m["n_liquidations"].sum()),
        "liquidation_rate": float((m["n_liquidations"] > 0).mean()),
    }


def plot(table):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    L = table.index.to_numpy(dtype=float)
    for ax, (key, title, scale) in zip(axes, [
        ("mdd", "Max drawdown on test (%) - lower is better", 100),
        ("sharpe", "Sharpe on test - higher is better", 1),
    ]):
        for engine, color, label in (("dyn", "tab:blue", "Dynamic Risk Engine"), ("fix", "tab:orange", "Fixed-rule ablation")):
            mean = table[f"{engine}_{key}_mean"].to_numpy(dtype=float) * scale
            std = table[f"{engine}_{key}_std"].to_numpy(dtype=float) * scale
            ax.plot(L, mean, marker="o", color=color, label=label)
            ax.fill_between(L, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xscale("log", base=2)
        ax.set_xticks(L)
        ax.set_xticklabels([f"{int(x)}x" for x in L])
        ax.set_xlabel("leverage cap")
        ax.set_title(title, fontsize=10)
    for x, n in zip(L, table["dyn_liquidations"]):
        if n:
            axes[0].annotate(f"{n} liquidation{'s' if n > 1 else ''}", (x, float(table.loc[x, "dyn_mdd_mean"]) * 100),
                             textcoords="offset points", xytext=(-60, 10), fontsize=8, color="tab:blue")
    axes[0].legend(fontsize=8)
    fig.suptitle("Leverage stress test: both engines retuned on 2024 at each cap; mean ± std over 5 seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase5_leverage_stress.png"), dpi=200)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    val_sigs = {s: signals.build(s, "val") for s in rb.SEEDS}
    test_sigs = {s: signals.build(s, "test") for s in rb.SEEDS}
    val_market = backtest.prepare_market(val_sigs[0].index)
    test_market = backtest.prepare_market(test_sigs[0].index)

    rows = {}
    for L in LEVERAGE_LEVELS:
        dyn_params, fix_params = tune_at_leverage(L, val_sigs, val_market)
        dyn_runs = rb.run_all_seeds(dyn_params, rb.dynamic_decider, test_sigs, test_market)
        fix_runs = rb.run_all_seeds(fix_params, rb.fixed_decider, test_sigs, test_market)

        dyn_ret = {s: h3_tests.daily_returns(dyn_runs[s]) for s in rb.SEEDS}
        fix_ret = {s: h3_tests.daily_returns(fix_runs[s]) for s in rb.SEEDS}
        _, pooled = h3_tests.bootstrap(dyn_ret, fix_ret)
        obs = {k: float(np.mean([h3_tests.observed(dyn_ret, fix_ret, s)[k] for s in rb.SEEDS]))
               for k in ("dd_reduction", "sharpe_gain")}
        dd = h3_tests.summarise_samples(pooled["dd_reduction"], obs["dd_reduction"])
        sh = h3_tests.summarise_samples(pooled["sharpe_gain"], obs["sharpe_gain"])

        row = {f"dyn_{k}": v for k, v in engine_summary(dyn_runs).items()}
        row.update({f"fix_{k}": v for k, v in engine_summary(fix_runs).items()})
        row.update({
            "dd_reduction_pp": dd["observed"] * 100, "dd_ci_low": dd["ci_low"] * 100,
            "dd_ci_high": dd["ci_high"] * 100, "dd_p": dd["p"],
            "sharpe_gain": sh["observed"], "sharpe_ci_low": sh["ci_low"],
            "sharpe_ci_high": sh["ci_high"], "sharpe_p": sh["p"],
            "dyn_settings": f"tau_j={dyn_params.tau_j}, tau_y={dyn_params.tau_y:.2f}, k={dyn_params.k}, c={dyn_params.c}",
            "fix_settings": f"tau_y={fix_params.tau_y:.2f}, fraction={fix_params.fraction}, stop={fix_params.stop_distance}",
        })
        rows[L] = row
        print(f"{L:4.0f}x done: dynamic MDD {row['dyn_mdd_mean']:.1%}, fixed MDD {row['fix_mdd_mean']:.1%}, "
              f"liquidations {row['dyn_liquidations']} / {row['fix_liquidations']}")
    table = pd.DataFrame(rows).T
    table.index.name = "leverage"

    lines = [
        "# Leverage stress test (Phase 5)",
        "",
        "Both engines retuned on 2024 validation at each leverage cap, then run once on the",
        "2025 test. Mean over 5 seeds. Liquidations = total across the 5 runs; rate = share",
        "of runs with at least one.",
        "",
        "| Leverage | Engine | Max drawdown | Sharpe | Final balance (from R1,000) | Liquidations (rate) | Settings |",
        "|---|---|---|---|---|---|---|",
    ]
    for L, r in table.iterrows():
        for eng, name in (("dyn", "Risk Engine"), ("fix", "Fixed rules")):
            lines.append(
                f"| {L:.0f}x | {name} | {r[f'{eng}_mdd_mean']:.1%} ± {r[f'{eng}_mdd_std']:.1%} "
                f"| {r[f'{eng}_sharpe_mean']:.2f} ± {r[f'{eng}_sharpe_std']:.2f} | {rb.money(r[f'{eng}_final_balance'])} "
                f"| {r[f'{eng}_liquidations']} ({r[f'{eng}_liquidation_rate']:.0%}) | {r[f'{eng}_settings']} |"
            )
    lines += [
        "",
        "## Risk Engine advantage at each leverage (paired block bootstrap, 5-seed average)",
        "",
        "| Leverage | Drawdown reduction (pp) | 95% interval | p | Sharpe gain | 95% interval | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for L, r in table.iterrows():
        lines.append(
            f"| {L:.0f}x | {r.dd_reduction_pp:+.1f} | [{r.dd_ci_low:+.1f}, {r.dd_ci_high:+.1f}] | {h3_tests.fmt_p(r.dd_p)} "
            f"| {r.sharpe_gain:+.2f} | [{r.sharpe_ci_low:+.2f}, {r.sharpe_ci_high:+.2f}] | {h3_tests.fmt_p(r.sharpe_p)} |"
        )
    lines += [
        "",
        "Why liquidations are rare here: a stop tighter than the liquidation distance (1/leverage)",
        "always fires first, and BTC trades 24/7 so it almost never opens beyond a stop (more",
        "than 2% from the previous close on 3 of 2,922 days). Stops fill exactly at the stop",
        "price (no slippage), which understates liquidation risk compared with live markets.",
        "",
        "Figure: figures/phase5_leverage_stress.png",
    ]

    with open(os.path.join(RESULTS_DIR, "leverage_stress.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    table.to_csv(os.path.join(RESULTS_DIR, "leverage_stress.csv"))
    plot(table)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
