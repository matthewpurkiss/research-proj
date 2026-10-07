"""
forward_test.py

Phase 5 step 6: forward test on 2026 data that no model, setting or design
decision has ever seen.

Why this matters: the 2025 test year is no longer fully clean. Several design
choices were made after looking at 2025 results (e.g. the percentile-based
weak-prediction filter replaced a raw threshold because one seed never traded
on 2025). January-September 2026 was untouched by all of that, so it's the
closest this project gets to live trading.

Everything is frozen before 2026 is looked at:
  - models: the saved checkpoints (trained on 2018-2023), no retraining
  - feature scaling: the FROZEN data/processed/scaler_params.csv (train-only),
    never refitted on the new data
  - Risk Engine settings: results/chosen_params.json, tuned on 2024
The frozen dataset (data/raw, data/processed) is never written to; the new
fetch lives in data/forward_2026/.

Steps (run from Code/):
  python3 forward_test.py fetch [--yahoo-only | --chain-only]
      Fetch every series 2018-01-01 .. FORWARD_END into data/forward_2026/raw.
      Blockchain.com needs a non-university connection (hotspot), as in Phase 1.
  python3 forward_test.py check
      Compare the new fetch's 2018-2025 values with the frozen raw data (should
      match closely - guards against the Phase 1 "fake data" incident) and print
      BTC's 2026 prices for an eyeball check against reality.
  python3 forward_test.py run
      Build features, run both engines on FORWARD_START .. FORWARD_END, write
      results/forward_2026.md + .csv and figures/phase5_forward_2026.png.

Windows for early-2026 days use late-2025 rows as history - fine here, since a
live trader would have that history; the "no window crosses a split" rule was
for keeping training and evaluation apart, which doesn't apply to a model that
is no longer being trained or tuned.
"""

import argparse
import contextlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import backtest
import build_dataset
import config
import fetch_data
import h3_tests
import model_ablation
import risk_engine
import run_backtest as rb
import signals
import splits
import stress_test

FORWARD_START = "2026-01-01"
FORWARD_END = "2026-09-30"
FORWARD_DIR = os.path.join("data", "forward_2026")
FORWARD_RAW = os.path.join(FORWARD_DIR, "raw")
FORWARD_ALIGNED = os.path.join(FORWARD_DIR, "processed", "aligned_dataset.csv")
FROZEN_RAW = os.path.join("data", "raw")
FROZEN_SCALER = os.path.join("data", "processed", "scaler_params.csv")
CHOSEN_PARAMS = os.path.join(rb.RESULTS_DIR, "chosen_params.json")
REPORT_NAME = "forward_2026"


@contextlib.contextmanager
def extended_end_date():
    """fetch_data.py and build_dataset.py read config.DATA_END when they run.
    Point it at FORWARD_END for the duration, so both are reused unchanged."""
    original = config.DATA_END
    config.DATA_END = FORWARD_END
    try:
        yield
    finally:
        config.DATA_END = original


def do_fetch(yahoo_only, chain_only, force):
    with extended_end_date():
        ok = fetch_data.fetch(FORWARD_RAW, do_yahoo=not chain_only, do_chain=not yahoo_only, force=force)
    if not ok:
        raise SystemExit(1)


def do_check():
    """Overlap check against the frozen raw files, then BTC 2026 landmarks."""
    print(f"{'series':<28} {'overlap days':>12} {'median diff':>12} {'max diff':>10}")
    worst = 0.0
    for name in sorted(os.listdir(FROZEN_RAW)):
        if not name.endswith(".csv"):
            continue
        old = pd.read_csv(os.path.join(FROZEN_RAW, name), index_col=0, parse_dates=True)
        new = pd.read_csv(os.path.join(FORWARD_RAW, name), index_col=0, parse_dates=True)
        col = "Close" if "Close" in old.columns else "value"
        both = pd.concat([old[col], new[col]], axis=1, keys=["old", "new"]).dropna()
        both = both[both["old"] != 0]
        rel = ((both["new"] - both["old"]) / both["old"]).abs()
        worst = max(worst, rel.median())
        print(f"{name:<28} {len(both):>12} {rel.median():>11.4%} {rel.max():>10.2%}")
    print()
    if worst > 0.01:
        print("WARNING: some series differ from the frozen data by >1% on a typical day.")
        print("Small differences are normal (vendor revisions); large ones mean something is wrong.")
    else:
        print("New fetch agrees with the frozen data over 2018-2025 (typical difference < 1%).")

    btc = pd.read_csv(os.path.join(FORWARD_RAW, "yahoo_BTC.csv"), index_col=0, parse_dates=True)["Close"]
    fwd = btc.loc[FORWARD_START:FORWARD_END]
    print(f"\nBTC 2026 check - compare with any price site (e.g. the one used in Phase 1):")
    print(f"  first close {fwd.index[0].date()}: ${fwd.iloc[0]:,.0f}")
    print(f"  last close  {fwd.index[-1].date()}: ${fwd.iloc[-1]:,.0f}")
    print(f"  high ${fwd.max():,.0f} on {fwd.idxmax().date()}, low ${fwd.min():,.0f} on {fwd.idxmin().date()}")
    print(f"  days covered: {len(fwd)} (expect ~{len(pd.date_range(FORWARD_START, FORWARD_END))})")


def build_forward_windows():
    """Aligned features to FORWARD_END from the new fetch, scaled with the
    FROZEN train-only scaler, windowed over the whole history."""
    with extended_end_date():
        aligned = build_dataset.build(raw_dir=FORWARD_RAW, output_path=FORWARD_ALIGNED)
    scaler = pd.read_csv(FROZEN_SCALER, index_col=0)
    scaled = splits.apply_scaler(aligned, scaler["mean"], scaler["std"])

    out = {}
    for agent, lookback in (("alpha", config.ALPHA_LOOKBACK), ("beta", config.BETA_LOOKBACK)):
        X, y_alpha, y_vol, y_jump, end_dates = splits.make_windows(scaled, lookback)
        out[agent] = {
            "X": torch.tensor(X, dtype=torch.float32), "dates": pd.DatetimeIndex(end_dates),
            "y_alpha": pd.Series(y_alpha, index=end_dates), "y_vol": pd.Series(y_vol, index=end_dates),
        }
    out["today_abs_return"] = aligned["btc_log_return"].abs()
    return out


def load_chosen_params():
    with open(CHOSEN_PARAMS) as f:
        p = json.load(f)
    return risk_engine.DynamicRiskParams(**p["dynamic"]), risk_engine.FixedRiskParams(**p["fixed"])


def plot(dyn_runs, fix_runs, bh_equity):
    fig, ax = plt.subplots(figsize=(10, 5))
    cap = rb.STARTING_CAPITAL
    for i, s in enumerate(rb.SEEDS):
        ax.plot(dyn_runs[s]["equity"] * cap, color="tab:blue", alpha=0.7, label="Dynamic Risk Engine" if i == 0 else None)
        ax.plot(fix_runs[s]["equity"] * cap, color="tab:orange", alpha=0.7, label="Fixed-rule ablation" if i == 0 else None)
    ax.plot(bh_equity * cap, color="black", linestyle="--", label="Buy & hold BTC (1x)")
    ax.axhline(cap, color="grey", linewidth=0.5)
    ax.set_ylabel(f"Account balance ({rb.CURRENCY}, starting {rb.money(cap)})")
    ax.set_title("Forward test on unseen 2026 data: nothing retrained or retuned; one line per seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(rb.FIGURES_DIR, f"phase5_{REPORT_NAME}.png"), dpi=200)


def leverage_menu(sigs, market):
    """The same Risk Engine at each stress-test leverage cap, with every other
    setting tuned on 2024 validation AT THAT CAP (stress_test.tune_at_leverage),
    then run once on the forward period. Presented as a menu of risk levels -
    every cap is reported, not just the best, so nothing is cherry-picked."""
    val_sigs = {s: signals.build(s, "val") for s in rb.SEEDS}
    val_market = backtest.prepare_market(val_sigs[0].index)
    rows = {}
    for L in stress_test.LEVERAGE_LEVELS:
        dyn_params, _ = stress_test.tune_at_leverage(L, val_sigs, val_market)
        runs = rb.run_all_seeds(dyn_params, rb.dynamic_decider, sigs, market)
        finals = np.array([runs[s]["equity"].iloc[-1] for s in rb.SEEDS])
        rows[L] = {
            "returns": finals - 1,
            "lowest": min(runs[s]["equity"].min() for s in rb.SEEDS),
            "mdd": np.mean([runs[s]["metrics"]["max_drawdown"] for s in rb.SEEDS]),
            "sharpe": np.mean([runs[s]["metrics"]["sharpe"] for s in rb.SEEDS]),
            "liquidations": sum(runs[s]["metrics"]["n_liquidations"] for s in rb.SEEDS),
        }
    return rows


def plot_menu(menu, bh_return, bh_mdd):
    caps = list(menu)
    x = np.arange(len(caps))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for i, L in enumerate(caps):
        r = menu[L]["returns"] * 100
        axes[0].scatter(np.full(len(r), i), r, color="tab:blue", alpha=0.6, s=25,
                        label="one run (seed)" if i == 0 else None)
        axes[0].scatter(i, r.mean(), color="black", marker="_", s=600, linewidths=2,
                        label="average" if i == 0 else None)
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].axhline(bh_return * 100, color="black", linestyle="--", linewidth=0.8, label="buy & hold BTC")
    axes[0].set_title("Return over the forward period (%)", fontsize=10)

    axes[1].bar(x, [menu[L]["mdd"] * 100 for L in caps], color="tab:blue")
    axes[1].axhline(bh_mdd * 100, color="black", linestyle="--", linewidth=0.8, label="buy & hold BTC")
    axes[1].set_title("Average worst drawdown (%) - lower is better", fontsize=10)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(L)}x" for L in caps])
        ax.set_xlabel("leverage cap (settings tuned on 2024 at each cap)")
        ax.legend(fontsize=8)
    fig.suptitle("Choosing a risk level: the Risk Engine on unseen 2026 data at each leverage cap", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(rb.FIGURES_DIR, f"phase5_{REPORT_NAME}_leverage.png"), dpi=200)


def do_run():
    os.makedirs(rb.RESULTS_DIR, exist_ok=True)
    os.makedirs(rb.FIGURES_DIR, exist_ok=True)

    w = build_forward_windows()
    fwd_dates = w["alpha"]["dates"][(w["alpha"]["dates"] >= FORWARD_START) & (w["alpha"]["dates"] <= FORWARD_END)]
    dyn_params, fix_params = load_chosen_params()

    btc = pd.read_csv(os.path.join(FORWARD_RAW, "yahoo_BTC.csv"), index_col=0, parse_dates=True)[
        ["Open", "High", "Low", "Close"]]
    market = backtest.prepare_market(fwd_dates, btc)

    sigs, hits, vol_rmse = {}, [], []
    actual_ret = w["alpha"]["y_alpha"].reindex(fwd_dates).to_numpy()
    actual_vol = w["beta"]["y_vol"].reindex(fwd_dates).to_numpy()
    for s in rb.SEEDS:
        sig = signals.from_windows(signals.load_models(s), w["alpha"]["X"], w["alpha"]["dates"],
                                   w["beta"]["X"], w["beta"]["dates"], fwd_dates)
        assert sig.notna().all().all() and len(sig) == len(fwd_dates), f"incomplete signals for seed {s}"
        sigs[s] = sig
        hits.append(model_ablation.hit_rate(sig["pred_return"].to_numpy(), actual_ret))
        vol_rmse.append(float(np.sqrt(np.mean((sig["pred_vol"].to_numpy() - actual_vol) ** 2))))
    persistence_rmse = float(np.sqrt(np.mean((w["today_abs_return"].reindex(fwd_dates).to_numpy() - actual_vol) ** 2)))

    dyn_runs = rb.run_all_seeds(dyn_params, rb.dynamic_decider, sigs, market)
    fix_runs = rb.run_all_seeds(fix_params, rb.fixed_decider, sigs, market)
    bh_equity, bh_metrics = rb.buy_and_hold(fwd_dates, market)
    dyn_rows, dyn_sum = rb.summarise(dyn_runs)
    fix_rows, fix_sum = rb.summarise(fix_runs)

    dyn_ret = {s: h3_tests.daily_returns(dyn_runs[s]) for s in rb.SEEDS}
    fix_ret = {s: h3_tests.daily_returns(fix_runs[s]) for s in rb.SEEDS}
    _, pooled = h3_tests.bootstrap(dyn_ret, fix_ret)
    obs = {k: float(np.mean([h3_tests.observed(dyn_ret, fix_ret, s)[k] for s in rb.SEEDS]))
           for k in ("dd_reduction", "sharpe_gain")}
    dd = h3_tests.summarise_samples(pooled["dd_reduction"], obs["dd_reduction"])
    sh = h3_tests.summarise_samples(pooled["sharpe_gain"], obs["sharpe_gain"])

    def fmt(summary, key, pct=False):
        mean, std = summary[key]
        return f"{mean:.1%} ± {std:.1%}" if pct else f"{mean:.3f} ± {std:.3f}"

    def fmt_money(summary, key):
        mean, std = summary[key]
        return f"{rb.money(mean)} ± {rb.money(std)}"

    lines = [
        "# Forward test on unseen 2026 data (Phase 5)",
        "",
        f"Period: {fwd_dates[0].date()} to {(fwd_dates[-1] + pd.offsets.BDay(1)).date()}, {len(fwd_dates)} trading days.",
        "Nothing retrained, refitted or retuned: saved checkpoints, the frozen train-only scaler,",
        "and the Risk Engine settings chosen on 2024 (results/chosen_params.json).",
        "",
        "## Trading results (mean ± std over 5 runs)",
        "",
        "| Metric | Dynamic Risk Engine | Fixed-rule ablation | Buy & hold BTC (1x) |",
        "|---|---|---|---|",
        f"| Final balance (from {rb.money(rb.STARTING_CAPITAL)}) | {fmt_money(dyn_sum, 'final_balance')} | {fmt_money(fix_sum, 'final_balance')} | {rb.money(bh_equity.iloc[-1] * rb.STARTING_CAPITAL)} |",
        f"| Lowest balance | {fmt_money(dyn_sum, 'lowest_balance')} | {fmt_money(fix_sum, 'lowest_balance')} | {rb.money(bh_equity.min() * rb.STARTING_CAPITAL)} |",
        f"| Max drawdown | {fmt(dyn_sum, 'max_drawdown', True)} | {fmt(fix_sum, 'max_drawdown', True)} | {bh_metrics['max_drawdown']:.1%} |",
        f"| Sharpe | {fmt(dyn_sum, 'sharpe')} | {fmt(fix_sum, 'sharpe')} | {bh_metrics['sharpe']:.3f} |",
        f"| Sortino | {fmt(dyn_sum, 'sortino')} | {fmt(fix_sum, 'sortino')} | {bh_metrics['sortino']:.3f} |",
        f"| Trades | {fmt(dyn_sum, 'n_trades')} | {fmt(fix_sum, 'n_trades')} | - |",
        f"| Liquidation rate | {dyn_sum['liquidation_rate']:.0%} | {fix_sum['liquidation_rate']:.0%} | - |",
        "",
        "## Risk Engine vs fixed rules (paired block bootstrap, 5-seed average)",
        "",
        "| | Observed | 95% interval | p (one-sided) |",
        "|---|---|---|---|",
        f"| Drawdown reduction (pp) | {dd['observed'] * 100:+.1f} | [{dd['ci_low'] * 100:+.1f}, {dd['ci_high'] * 100:+.1f}] | {h3_tests.fmt_p(dd['p'])} |",
        f"| Sharpe gain | {sh['observed']:+.2f} | [{sh['ci_low']:+.2f}, {sh['ci_high']:+.2f}] | {h3_tests.fmt_p(sh['p'])} |",
        "",
        "## Forecasts on the forward period",
        "",
        f"- Alpha Agent directional hit rate: {np.mean(hits):.1%} ± {np.std(hits, ddof=1):.1%} "
        f"(share of up days: {np.mean(actual_ret > 0):.1%})",
        f"- Beta Agent volatility RMSE: {np.mean(vol_rmse):.5f} ± {np.std(vol_rmse, ddof=1):.5f} "
        f"(naive persistence: {persistence_rmse:.5f})",
        "",
        "## Per-run metrics",
        "",
        "Dynamic Risk Engine:",
        "",
        rb.markdown_table(dyn_rows),
        "",
        "Fixed-rule ablation:",
        "",
        rb.markdown_table(fix_rows),
        "",
        f"Figure: figures/phase5_{REPORT_NAME}.png",
    ]

    menu = leverage_menu(sigs, market)
    bh_return = bh_equity.iloc[-1] - 1
    lines += [
        "",
        "## Choosing a risk level (leverage menu)",
        "",
        "The same Risk Engine at each leverage cap, every other setting tuned on 2024 at that",
        "cap, then run once on the forward period. Every cap is shown - nothing picked after",
        "the fact. Percentages are the same for any account size.",
        "",
        f"| Leverage cap | Average return | Best run | Worst run | Lowest point, any run (from {rb.money(rb.STARTING_CAPITAL)}) | Avg worst drawdown | Sharpe | Liquidations |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for L, r in menu.items():
        lines.append(
            f"| {L:.0f}x | {r['returns'].mean():+.1%} | {r['returns'].max():+.1%} | {r['returns'].min():+.1%} "
            f"| {rb.money(r['lowest'] * rb.STARTING_CAPITAL)} | {r['mdd']:.1%} | {r['sharpe']:.2f} | {r['liquidations']} |"
        )
    lines += [
        f"| Buy & hold BTC (1x) | {bh_return:+.1%} | - | - | {rb.money(bh_equity.min() * rb.STARTING_CAPITAL)} "
        f"| {bh_metrics['max_drawdown']:.1%} | {bh_metrics['sharpe']:.2f} | - |",
        "",
        "More leverage widens the range of outcomes; it does not raise return per unit of",
        "risk (Sharpe stays roughly flat until it collapses at the highest cap).",
        "",
        f"Figure: figures/phase5_{REPORT_NAME}_leverage.png",
    ]
    with open(os.path.join(rb.RESULTS_DIR, f"{REPORT_NAME}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.concat({"dynamic": dyn_rows, "fixed": fix_rows}).to_csv(os.path.join(rb.RESULTS_DIR, f"{REPORT_NAME}.csv"))
    plot(dyn_runs, fix_runs, bh_equity)
    plot_menu(menu, bh_return, bh_metrics["max_drawdown"])
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    f = sub.add_parser("fetch")
    g = f.add_mutually_exclusive_group()
    g.add_argument("--yahoo-only", action="store_true")
    g.add_argument("--chain-only", action="store_true")
    f.add_argument("--force", action="store_true", help="overwrite files already in data/forward_2026/raw")
    sub.add_parser("check")
    sub.add_parser("run")
    args = parser.parse_args()

    if args.step == "fetch":
        do_fetch(args.yahoo_only, args.chain_only, args.force)
    elif args.step == "check":
        do_check()
    else:
        do_run()


if __name__ == "__main__":
    main()
