"""
h3_tests.py

Phase 5: is the Dynamic Risk Engine's advantage over the fixed-rule ablation
statistically real, or within what 201 test days and 5 runs could produce by
chance? (H3, proposal Sections 4 and 6.1.5.)

Uses exactly the tuned runs from run_backtest.tune_and_test(), then:

1. Daily-return test, per seed (the proposal's Diebold-Mariano test, adapted).
   DM compares two forecasts' losses day by day; here each engine's "loss" on a
   day is its negative return, so the loss differential is
   d_t = return_fixed(t) - return_dynamic(t). The test asks whether mean(d) is
   zero, with a Newey-West variance so that correlation between nearby days
   doesn't overstate confidence. One test per seed, as in Phase 2.

2. Block bootstrap for max drawdown and Sharpe. Each run yields only one
   drawdown and one Sharpe, so they can't be t-tested. Instead, rebuild the
   test period 5,000 times from randomly chosen 10-day blocks of real test days
   (blocks keep short-run volatility clustering intact). Both engines get the
   SAME resampled days each time - a paired comparison - and since all seeds
   trade the same dates, the same resample is used across seeds too, giving a
   confidence interval for the 5-seed average effect.

Liquidation rate isn't tested: both engines are tuned to 2x leverage and
neither liquidates (see the stress test, Phase 5 step 4).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import backtest
import run_backtest

RESULTS_DIR = run_backtest.RESULTS_DIR
FIGURES_DIR = run_backtest.FIGURES_DIR
N_BOOTSTRAP = 5000
BLOCK_LENGTH = 10
BOOTSTRAP_SEED = 20261007


def daily_returns(run):
    return run["equity"].pct_change().iloc[1:].fillna(0.0).to_numpy()


def newey_west_lags(T):
    """Standard automatic lag choice, floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def return_difference_test(ret_dynamic, ret_fixed):
    """DM-style test on the daily loss differential d = r_fixed - r_dynamic.
    Returns (mean daily advantage of dynamic, t statistic, two-sided p), with t
    signed like the advantage (positive = Risk Engine better)."""
    d = ret_fixed - ret_dynamic
    T = len(d)
    d_c = d - d.mean()
    var = np.mean(d_c ** 2)
    for k in range(1, newey_west_lags(T) + 1):
        weight = 1 - k / (newey_west_lags(T) + 1)
        var += 2 * weight * np.mean(d_c[:-k] * d_c[k:])
    t_stat = d.mean() / np.sqrt(var / T)
    p = 2 * (1 - stats.t.cdf(abs(t_stat), df=T - 1))
    return -d.mean(), -t_stat, p


def block_indices(T, rng):
    n_blocks = int(np.ceil(T / BLOCK_LENGTH))
    starts = rng.integers(0, T - BLOCK_LENGTH + 1, size=(N_BOOTSTRAP, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK_LENGTH)).reshape(N_BOOTSTRAP, -1)
    return idx[:, :T]


def sharpe_rows(R):
    std = R.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = R.mean(axis=1) / std * np.sqrt(backtest.PERIODS_PER_YEAR)
    return np.where(std > 0, s, 0.0)


def max_drawdown_rows(R):
    equity = np.cumprod(1 + R, axis=1)
    equity = np.concatenate([np.ones((R.shape[0], 1)), equity], axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    return ((peak - equity) / peak).max(axis=1)


def bootstrap(dyn_returns, fix_returns):
    """dyn_returns / fix_returns: {seed: daily return array}, same dates.
    Returns per-seed and seed-averaged bootstrap samples of
    drawdown reduction (fixed MDD - dynamic MDD) and Sharpe gain
    (dynamic Sharpe - fixed Sharpe). Positive = Risk Engine better."""
    seeds = sorted(dyn_returns)
    T = len(dyn_returns[seeds[0]])
    idx = block_indices(T, np.random.default_rng(BOOTSTRAP_SEED))

    per_seed = {}
    for s in seeds:
        Rd, Rf = dyn_returns[s][idx], fix_returns[s][idx]
        per_seed[s] = {
            "dd_reduction": max_drawdown_rows(Rf) - max_drawdown_rows(Rd),
            "sharpe_gain": sharpe_rows(Rd) - sharpe_rows(Rf),
        }
    pooled = {
        key: np.mean([per_seed[s][key] for s in seeds], axis=0)
        for key in ("dd_reduction", "sharpe_gain")
    }
    return per_seed, pooled


def observed(dyn_returns, fix_returns, s):
    Rd, Rf = dyn_returns[s][None, :], fix_returns[s][None, :]
    return {
        "dd_reduction": float(max_drawdown_rows(Rf)[0] - max_drawdown_rows(Rd)[0]),
        "sharpe_gain": float(sharpe_rows(Rd)[0] - sharpe_rows(Rf)[0]),
    }


def fmt_p(p):
    """A bootstrap p can't go below 1/N_BOOTSTRAP - say so instead of printing 0."""
    return f"< {1 / N_BOOTSTRAP:.4f}" if p == 0 else f"{p:.4f}"


def summarise_samples(samples, obs):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    p_one_sided = float(np.mean(samples <= 0))  # share of resamples with no advantage
    return {"observed": obs, "ci_low": lo, "ci_high": hi, "p": p_one_sided}


def plot(pooled, pooled_obs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = [
        ("dd_reduction", "Drawdown reduction (fixed MDD − Risk Engine MDD)", True),
        ("sharpe_gain", "Sharpe gain (Risk Engine − fixed)", False),
    ]
    for ax, (key, title, pct) in zip(axes, panels):
        samples, obs = pooled[key], pooled_obs[key]
        lo, hi = np.percentile(samples, [2.5, 97.5])
        scale = 100 if pct else 1
        ax.hist(samples * scale, bins=60, color="tab:blue", alpha=0.6)
        ax.axvline(obs * scale, color="black", label="observed")
        ax.axvline(lo * scale, color="tab:red", linestyle="--", label="95% interval")
        ax.axvline(hi * scale, color="tab:red", linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("percentage points" if pct else "annualised Sharpe")
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"H3 block bootstrap, average over 5 seeds ({N_BOOTSTRAP:,} resamples of the 2025 test days)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase5_h3_bootstrap.png"), dpi=200)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    r = run_backtest.tune_and_test(verbose=False)
    seeds = sorted(r["dyn_runs"])
    dyn = {s: daily_returns(r["dyn_runs"][s]) for s in seeds}
    fix = {s: daily_returns(r["fix_runs"][s]) for s in seeds}
    T = len(dyn[seeds[0]])

    lines = [
        "# H3 significance tests (Phase 5)",
        "",
        f"Dynamic Risk Engine vs fixed-rule ablation, same tuned runs as backtest_report.md. "
        f"{T} test days, 5 runs.",
        "",
        "## 1. Daily-return test per seed (DM-style, Newey-West variance)",
        "",
        f"Loss differential d = return_fixed − return_dynamic; Newey-West lags = {newey_west_lags(T)}.",
        "",
        "| seed | mean daily advantage of Risk Engine | t | p (two-sided) | verdict |",
        "|---|---|---|---|---|",
    ]
    csv_rows = []
    n_sig = 0
    for s in seeds:
        adv, t_stat, p = return_difference_test(dyn[s], fix[s])
        sig = p < 0.05
        n_sig += sig
        verdict = ("significant, Risk Engine better" if adv > 0 else "significant, fixed better") if sig else "not significant"
        lines.append(f"| {s} | {adv:+.4%} | {t_stat:+.2f} | {p:.4f} | {verdict} |")
        csv_rows.append({"test": "daily_return", "seed": s, "estimate": adv, "stat": t_stat, "p": p})
    lines += ["", f"Significant at alpha=0.05 on {n_sig} of {len(seeds)} seeds.", ""]

    per_seed, pooled = bootstrap(dyn, fix)
    per_seed_obs = {s: observed(dyn, fix, s) for s in seeds}
    pooled_obs = {k: float(np.mean([per_seed_obs[s][k] for s in seeds])) for k in ("dd_reduction", "sharpe_gain")}

    lines += [
        f"## 2. Block bootstrap ({N_BOOTSTRAP:,} resamples, {BLOCK_LENGTH}-day blocks)",
        "",
        "Positive = Risk Engine better. p = share of resamples showing no advantage "
        "(one-sided). 95% interval from the resample percentiles.",
        "",
        "| | Drawdown reduction (pp) | 95% interval | p | Sharpe gain | 95% interval | p |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = [(f"seed {s}", per_seed[s], per_seed_obs[s]) for s in seeds] + [("**average of 5**", pooled, pooled_obs)]
    for label, samples, obs in rows:
        dd = summarise_samples(samples["dd_reduction"], obs["dd_reduction"])
        sh = summarise_samples(samples["sharpe_gain"], obs["sharpe_gain"])
        lines.append(
            f"| {label} | {dd['observed'] * 100:+.1f} | [{dd['ci_low'] * 100:+.1f}, {dd['ci_high'] * 100:+.1f}] | {fmt_p(dd['p'])} "
            f"| {sh['observed']:+.2f} | [{sh['ci_low']:+.2f}, {sh['ci_high']:+.2f}] | {fmt_p(sh['p'])} |"
        )
        for name, summ in (("dd_reduction", dd), ("sharpe_gain", sh)):
            csv_rows.append({"test": f"bootstrap_{name}", "seed": label.strip("*").replace("seed ", ""),
                             "estimate": summ["observed"], "ci_low": summ["ci_low"],
                             "ci_high": summ["ci_high"], "p": summ["p"]})

    lines += [
        "",
        "Liquidation rate is not tested: neither engine liquidates at the tuned 2x leverage.",
        "",
        "Figure: figures/phase5_h3_bootstrap.png (seed-averaged bootstrap distributions).",
    ]

    with open(os.path.join(RESULTS_DIR, "h3_tests.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    pd.DataFrame(csv_rows).to_csv(os.path.join(RESULTS_DIR, "h3_tests.csv"), index=False)
    plot(pooled, pooled_obs)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
