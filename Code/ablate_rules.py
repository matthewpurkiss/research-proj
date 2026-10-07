"""
ablate_rules.py

Phase 5 step 3: which of the Dynamic Risk Engine's Beta-driven rules is doing
the work? The H3 test showed the full engine cuts max drawdown by ~12.6 pp vs
the fixed-rule ablation, but that compares everything-on with everything-off.

Here each version swaps ONE Beta-driven rule for its plain fixed version and
keeps the rest (risk_engine.DynamicRiskParams' rule switches), plus one that
turns off rule 2's filter:

  full            all five rules (reference)
  no_strength     rule 2's "skip weak predictions" filter off (tau_y = 0) - not a
                  Beta rule, but it's a main reason the engine sits out so many
                  days, and the fixed-rule ablation was tuned to skip nothing, so
                  this separates "picky about Alpha's signals" from the Beta rules
  no_jump_veto    rule 1 off: trade even on high-jump-risk days
  fixed_sizing    rule 3 off: same fraction of equity every trade
  fixed_stop      rule 4 off: same stop distance every trade
  fixed_leverage  rule 5 off: leverage not cut when jump risk is high

Every version is retuned on 2024 validation with the same objective as
run_backtest.py, so none is judged with settings chosen for another rule set,
then run once on 2025 test. The paired block bootstrap from h3_tests.py then
asks, for each version: does removing that rule make drawdown / Sharpe
significantly WORSE than the full engine? Positive numbers = the rule helps.
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

VERSIONS = {
    "full": "All five rules",
    "no_strength": "No weak-prediction filter",
    "no_jump_veto": "No jump veto",
    "fixed_sizing": "Fixed position size",
    "fixed_stop": "Fixed stop-loss",
    "fixed_leverage": "Fixed leverage",
}


def candidates(version):
    P = risk_engine.DynamicRiskParams
    tau_y, tau_j = rb.DIRECTION_STRENGTH_GRID, rb.JUMP_VETO_GRID
    k, c, lev = rb.RISK_BUDGET_GRID, rb.STOP_MULT_GRID, rb.LEVERAGE_CAP_GRID
    if version == "full":
        return [P(tj, ty, kk, cc, L) for tj, ty, kk, cc, L in itertools.product(tau_j, tau_y, k, c, lev)]
    if version == "no_strength":
        return [P(tj, 0.0, kk, cc, L) for tj, kk, cc, L in itertools.product(tau_j, k, c, lev)]
    if version == "no_jump_veto":
        return [P(1.0, ty, kk, cc, L, use_jump_veto=False)
                for ty, kk, cc, L in itertools.product(tau_y, k, c, lev)]
    if version == "fixed_sizing":
        return [P(tj, ty, 0.0, cc, L, fixed_fraction=f)
                for tj, ty, f, cc, L in itertools.product(tau_j, tau_y, rb.FIXED_FRACTION_GRID, c, lev)]
    if version == "fixed_stop":
        return [P(tj, ty, kk, 0.0, L, fixed_stop=st)
                for tj, ty, kk, st, L in itertools.product(tau_j, tau_y, k, rb.FIXED_STOP_GRID, lev)]
    if version == "fixed_leverage":
        return [P(tj, ty, kk, cc, L, jump_scaled_leverage=False)
                for tj, ty, kk, cc, L in itertools.product(tau_j, tau_y, k, c, lev)]
    raise ValueError(version)


def describe(p):
    parts = [f"tau_y={p.tau_y:.2f}"]
    parts.append("no veto" if not p.use_jump_veto else f"tau_j={p.tau_j}")
    parts.append(f"fraction={p.fixed_fraction}" if p.fixed_fraction is not None else f"k={p.k}")
    parts.append(f"stop={p.fixed_stop}" if p.fixed_stop is not None else f"c={p.c}")
    parts.append(f"L={p.l_max}" + ("" if p.jump_scaled_leverage else " fixed"))
    return ", ".join(parts)


def plot(table):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = list(table.index)
    x = np.arange(len(labels))
    colors = ["tab:blue"] + ["tab:cyan"] * (len(labels) - 2) + ["tab:orange"]
    for ax, (mean_col, std_col, title, scale) in zip(axes, [
        ("mdd_mean", "mdd_std", "Max drawdown on test (%) - lower is better", 100),
        ("sharpe_mean", "sharpe_std", "Sharpe on test - higher is better", 1),
    ]):
        ax.bar(x, table[mean_col] * scale, yerr=table[std_col] * scale, color=colors, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="grey", linewidth=0.8)
    fig.suptitle("Rule-by-rule ablation: mean ± std over 5 seeds, each version retuned on 2024", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase5_rule_ablation.png"), dpi=200)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    val_sigs = {s: signals.build(s, "val") for s in rb.SEEDS}
    test_sigs = {s: signals.build(s, "test") for s in rb.SEEDS}
    val_market = backtest.prepare_market(val_sigs[0].index)
    test_market = backtest.prepare_market(test_sigs[0].index)

    runs, chosen = {}, {}
    for version in VERSIONS:
        cands = candidates(version)
        params, val_score = rb.grid_search(cands, rb.dynamic_decider, val_sigs, val_market)
        chosen[version] = (params, val_score)
        runs[version] = rb.run_all_seeds(params, rb.dynamic_decider, test_sigs, test_market)
        print(f"{version:15s} {len(cands):3d} settings tried -> {describe(params)} (val objective {val_score:.3f})")

    fix_params, _ = rb.grid_search(
        [risk_engine.FixedRiskParams(ty, f, st, L) for ty, f, st, L in itertools.product(
            rb.DIRECTION_STRENGTH_GRID, rb.FIXED_FRACTION_GRID, rb.FIXED_STOP_GRID, rb.FIXED_LEVERAGE_GRID)],
        rb.fixed_decider, val_sigs, val_market)
    fixed_runs = rb.run_all_seeds(fix_params, rb.fixed_decider, test_sigs, test_market)

    full_returns = {s: h3_tests.daily_returns(runs["full"][s]) for s in rb.SEEDS}

    rows = []
    for version, label in list(VERSIONS.items()) + [("fixed_rules", "All fixed (H3 ablation)")]:
        vr = fixed_runs if version == "fixed_rules" else runs[version]
        m = pd.DataFrame({s: vr[s]["metrics"] for s in rb.SEEDS}).T
        row = {
            "version": label,
            "mdd_mean": m["max_drawdown"].mean(), "mdd_std": m["max_drawdown"].std(),
            "sharpe_mean": m["sharpe"].mean(), "sharpe_std": m["sharpe"].std(),
            "final_balance": m["total_return"].mean() * rb.STARTING_CAPITAL,
            "trades": m["n_trades"].mean(),
            "settings": describe(chosen[version][0]) if version in chosen else "see backtest_report.md",
        }
        if version != "full":
            variant_returns = {s: h3_tests.daily_returns(vr[s]) for s in rb.SEEDS}
            _, pooled = h3_tests.bootstrap(full_returns, variant_returns)
            obs = {
                "dd_reduction": np.mean([h3_tests.observed(full_returns, variant_returns, s)["dd_reduction"] for s in rb.SEEDS]),
                "sharpe_gain": np.mean([h3_tests.observed(full_returns, variant_returns, s)["sharpe_gain"] for s in rb.SEEDS]),
            }
            dd = h3_tests.summarise_samples(pooled["dd_reduction"], obs["dd_reduction"])
            sh = h3_tests.summarise_samples(pooled["sharpe_gain"], obs["sharpe_gain"])
            row.update({
                "rule_cuts_mdd_pp": dd["observed"] * 100, "mdd_ci": (dd["ci_low"] * 100, dd["ci_high"] * 100), "mdd_p": dd["p"],
                "rule_adds_sharpe": sh["observed"], "sharpe_ci": (sh["ci_low"], sh["ci_high"]), "sharpe_p": sh["p"],
            })
        rows.append(row)
    table = pd.DataFrame(rows).set_index("version")

    lines = [
        "# Rule-by-rule ablation (Phase 5)",
        "",
        "Each version swaps one Beta-driven rule for its fixed version, is retuned on 2024",
        "validation, and is run once on the 2025 test. Mean over 5 seeds.",
        "",
        "| Version | Max drawdown | Sharpe | Final balance (from R1,000) | Trades | Settings chosen |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in table.iterrows():
        lines.append(
            f"| {label} | {r.mdd_mean:.1%} ± {r.mdd_std:.1%} | {r.sharpe_mean:.2f} ± {r.sharpe_std:.2f} "
            f"| {rb.money(r.final_balance)} | {r.trades:.0f} | {r.settings} |"
        )
    lines += [
        "",
        "## How much does each rule help? (paired block bootstrap vs the full engine)",
        "",
        "Removing the rule makes drawdown worse by / Sharpe lower by - positive = the rule helps.",
        "Average of 5 seeds, 95% interval, one-sided p.",
        "",
        "| Rule removed | Drawdown worse by (pp) | 95% interval | p | Sharpe lower by | 95% interval | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, r in table.iloc[1:].iterrows():
        lines.append(
            f"| {label} | {r.rule_cuts_mdd_pp:+.1f} | [{r.mdd_ci[0]:+.1f}, {r.mdd_ci[1]:+.1f}] | {h3_tests.fmt_p(r.mdd_p)} "
            f"| {r.rule_adds_sharpe:+.2f} | [{r.sharpe_ci[0]:+.2f}, {r.sharpe_ci[1]:+.2f}] | {h3_tests.fmt_p(r.sharpe_p)} |"
        )
    lines += ["", "Figure: figures/phase5_rule_ablation.png"]

    with open(os.path.join(RESULTS_DIR, "rule_ablation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    table.to_csv(os.path.join(RESULTS_DIR, "rule_ablation.csv"))
    plot(table)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
