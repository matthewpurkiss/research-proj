"""
make_explainer_figures.py

Simple, presentation-friendly versions of the project's results, for explaining
the project to a non-specialist (and for the write-up). Each chart shows one idea
with large labels. Writes figures/explain_*.png.

Numbers come from Code/results/*.csv (the official runs) or are recomputed from
the saved checkpoints with the same functions the official scripts use. Run from
Code/ after run_backtest.py, forward_test.py and direction_baselines.py.
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

FIG = os.path.join("..", "figures")
RES = "results"

INK, MUTED = "#1B2433", "#5F6B7A"
BLUE, ORANGE, GREEN, RED, GREY = "#1F77B4", "#F7931A", "#2E9E6A", "#C0392B", "#9AA5B1"
RE_C, FIX_C, BH_C = BLUE, ORANGE, INK

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
})


def month_axis(ax):
    """Short month labels ('Apr', 'Jun', ...) so dates never run together."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def save(fig, name):
    import textwrap
    for ax in fig.axes:
        t = ax.get_title()
        if t and "\n" not in t:
            ax.set_title(textwrap.fill(t, 62), fontsize=ax.title.get_fontsize())
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)
    print("wrote", name)


def label_bars(ax, bars, fmt, pad=0.6, inside=False):
    for b in bars:
        v = b.get_height()
        y = v - pad if inside else v + pad
        ax.text(b.get_x() + b.get_width() / 2, y, fmt(v), ha="center",
                va="top" if inside else "bottom", fontsize=13, color="white" if inside else INK)


# ── data helpers ──────────────────────────────────────────────────────────────

def runs_2025():
    r = rb.tune_and_test(verbose=False)
    return r["dyn_runs"], r["fix_runs"], r["bh_equity"]


def runs_2026():
    aligned = pd.read_csv(ft.FORWARD_ALIGNED, index_col="Date", parse_dates=True)
    scaler = pd.read_csv(ft.FROZEN_SCALER, index_col=0)
    scaled = splits.apply_scaler(aligned, scaler["mean"], scaler["std"])
    w = {}
    for agent, lb in (("alpha", config.ALPHA_LOOKBACK), ("beta", config.BETA_LOOKBACK)):
        X, _, _, _, d = splits.make_windows(scaled, lb)
        w[agent] = (torch.tensor(X, dtype=torch.float32), pd.DatetimeIndex(d))
    dates = w["alpha"][1]
    fwd = dates[(dates >= ft.FORWARD_START) & (dates <= ft.FORWARD_END)]
    btc = pd.read_csv(os.path.join(ft.FORWARD_RAW, "yahoo_BTC.csv"), index_col=0, parse_dates=True)[
        ["Open", "High", "Low", "Close"]]
    from backtest import prepare_market
    market = prepare_market(fwd, btc)
    dp, fp = ft.load_chosen_params()
    sig = {s: signals.from_windows(signals.load_models(s), *w["alpha"], *w["beta"], fwd) for s in rb.SEEDS}
    dyn = rb.run_all_seeds(dp, rb.dynamic_decider, sig, market)
    fix = rb.run_all_seeds(fp, rb.fixed_decider, sig, market)
    bh, _ = rb.buy_and_hold(fwd, market)
    return dyn, fix, bh


def avg_curve(runs):
    return pd.concat([runs[s]["equity"] for s in rb.SEEDS], axis=1).mean(axis=1) * 1000


def band(runs):
    df = pd.concat([runs[s]["equity"] for s in rb.SEEDS], axis=1) * 1000
    return df.min(axis=1), df.max(axis=1)


# ── charts ────────────────────────────────────────────────────────────────────

def btc_splits():
    btc = pd.read_csv(os.path.join(ft.FORWARD_RAW, "yahoo_BTC.csv"), index_col=0, parse_dates=True)["Close"]
    btc = btc.loc["2018-02-12":"2026-09-30"]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    spans = [("2018-02-12", "2023-12-31", GREY, "TRAIN 2018-2023\nmodels learn here", 118_000),
             ("2024-01-01", "2024-12-31", BLUE, "VALIDATE\n2024\nsettings\ntuned", 30_000),
             ("2025-01-01", "2025-12-31", ORANGE, "TEST\n2025\njudged\nonce", 30_000),
             ("2026-01-01", "2026-09-30", GREEN, "FORWARD\n2026\nnever\nseen", 30_000)]
    for a, b, col, lab, ytxt in spans:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=col, alpha=0.18, lw=0)
        mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
        ax.text(mid, ytxt, lab, ha="center", va="top", fontsize=12, color=INK, fontweight="bold")
    ax.plot(btc.index, btc.values, color=INK, lw=1.4)
    ax.set_ylim(0, 125_000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.set_title("Bitcoin's price, and how the data was split so the models are tested honestly")
    save(fig, "explain_btc_splits.png")


def equity(dyn, fix, bh, year, name):
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    for runs, col, lab in ((fix, FIX_C, "Fixed rules (no Beta Agent)"), (dyn, RE_C, "Risk Engine (full system)")):
        lo, hi = band(runs)
        ax.fill_between(lo.index, lo, hi, color=col, alpha=0.15, lw=0)
        m = avg_curve(runs)
        ax.plot(m.index, m, color=col, lw=2.6, label=f"{lab}: ends R{m.iloc[-1]:,.0f}")
    b = bh * 1000
    ax.plot(b.index, b, color=BH_C, lw=1.6, ls="--", label=f"Buy and hold BTC: ends R{b.iloc[-1]:,.0f}")
    ax.axhline(1000, color=MUTED, lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"R{v:,.0f}"))
    month_axis(ax)
    ax.set_title(f"What happened to R1,000 in {year} (lines = average of 5 runs, shading = best to worst run)",
                 fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)
    save(fig, name)


def drawdown_example(bh):
    b = bh * 1000
    trough_day = (b / b.cummax()).idxmin()
    peak_day = b.loc[:trough_day].idxmax()
    peak, trough = b[peak_day], b[trough_day]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(b.index, b, color=BH_C, lw=2)
    ax.scatter([peak_day], [peak], color=GREEN, s=90, zorder=3)
    ax.scatter([trough_day], [trough], color=RED, s=90, zorder=3)
    ax.annotate(f"Highest point\nR{peak:,.0f}", (peak_day, peak), xytext=(25, 0), textcoords="offset points",
                fontsize=13, color=GREEN, va="center")
    ax.annotate(f"Lowest point after it: R{trough:,.0f}", (trough_day, trough), xytext=(14, -4),
                textcoords="offset points", fontsize=13, color=RED, ha="left", va="center")
    ax.annotate("", xy=(trough_day, trough), xytext=(trough_day, peak),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.5))
    ax.plot([peak_day, trough_day], [peak, peak], color=INK, lw=0.8, ls=":")
    ax.text(trough_day, (peak + trough) / 2 + 40, f"  max drawdown\n  = {(peak - trough) / peak:.1%}",
            fontsize=14, color=INK, va="center")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"R{v:,.0f}"))
    month_axis(ax)
    ax.set_title("Max drawdown = the biggest fall from a high point (example: buying and holding BTC in 2026)",
                 fontsize=14)
    save(fig, "explain_drawdown.png")


def worst_drop():
    t25 = pd.read_csv(os.path.join(RES, "test_metrics.csv"), index_col=[0, 1])
    t26 = pd.read_csv(os.path.join(RES, "forward_2026.csv"), index_col=[0, 1])
    vals = {
        "2025 test": [t25.loc["dynamic"]["max_drawdown"].mean(), t25.loc["fixed"]["max_drawdown"].mean(), 0.318],
        "2026 forward test": [t26.loc["dynamic"]["max_drawdown"].mean(), t26.loc["fixed"]["max_drawdown"].mean(), 0.396],
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(2)
    for i, (lab, col) in enumerate([("Risk Engine", RE_C), ("Fixed rules", FIX_C), ("Buy and hold BTC", BH_C)]):
        bars = ax.bar(x + (i - 1) * 0.27, [vals[k][i] * 100 for k in vals], 0.25, color=col, label=lab)
        label_bars(ax, bars, lambda v: f"{v:.1f}%", pad=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(vals))
    ax.set_ylabel("Worst fall from a high (%)")
    ax.set_ylim(0, 52)
    ax.set_title("Worst loss along the way (max drawdown) - lower is better")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "explain_worst_drop.png")


def vol_forecast_example():
    sig = {s: signals.build(s, "test") for s in rb.SEEDS}
    pred = pd.concat([sig[s]["pred_vol"] for s in rb.SEEDS], axis=1).mean(axis=1) * 100
    data = np.load(os.path.join(signals.DATA_DIR, "beta_test_windows.npz"), allow_pickle=True)
    actual = pd.Series(data["y_beta_vol"], index=pd.to_datetime(data["end_dates"])).reindex(pred.index) * 100
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(actual.index, actual.values, width=1.0, color=GREY, label="What actually happened: size of the next day's move")
    ax.plot(pred.index, pred.values, color=ORANGE, lw=2.6, label="Beta Agent's forecast of that size")
    ax.set_ylabel("Size of daily move (%)")
    month_axis(ax)
    ax.set_title("The Beta Agent forecasts how BIG tomorrow's move will be, not which way (2025 test)", fontsize=14)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "explain_volatility_forecast.png")


def vol_errors():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), gridspec_kw={"width_ratios": [3, 2]})
    groups = [("2025 test", ["Naive guess:\nsame as today", "GARCH\n(classic model)", "Beta Agent"],
               [2.079, 2.246, 1.974], [GREY, GREY, ORANGE]),
              ("2026 forward test", ["Naive guess:\nsame as today", "Beta Agent"], [2.500, 2.038], [GREY, ORANGE])]
    for ax, (title, labels, vals, cols) in zip(axes, groups):
        bars = ax.bar(labels, vals, color=cols, width=0.6)
        label_bars(ax, bars, lambda v: f"{v:.2f}", pad=0.03)
        ax.set_ylim(0, 2.9)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=12)
    axes[0].set_ylabel("Average forecast error (RMSE, %)")
    fig.suptitle("Volatility forecast error - lower is better", fontsize=16)
    save(fig, "explain_volatility_errors.png")


def hit_rate():
    d = pd.read_csv(os.path.join(RES, "direction_baselines.csv"))
    d = d[d.period == "All three"].set_index("method")
    order = [("Alpha Agent (5-seed avg)", "Alpha Agent (my model)"), ("Always up", "Always guess 'up'"),
             ("Momentum 1d", "Same as yesterday"), ("Momentum 20d", "Same as last 20 days"),
             ("Mean reversion", "Opposite of yesterday")]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = np.arange(len(order))[::-1]
    for yy, (key, lab) in zip(y, order):
        r = d.loc[key]
        col = BLUE if key.startswith("Alpha") else GREY
        ax.plot([r.ci_low * 100, r.ci_high * 100], [yy, yy], color=col, lw=4, solid_capstyle="round")
        ax.scatter([r.hit_rate * 100], [yy], color=col, s=120, zorder=3)
        ax.text(r.ci_high * 100 + 0.4, yy, f"{r.hit_rate:.1%}", va="center", fontsize=13)
    ax.axvline(50, color=RED, ls="--", lw=1.5)
    ax.text(50, len(order) - 0.35, "coin flip (50%)", color=RED, ha="center", fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in order])
    ax.set_xlim(40, 62)
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel("How often tomorrow's direction was called right (%), 2024-2026, ~600 days")
    ax.set_title("Nothing beats a coin flip at calling tomorrow's direction (bars = 95% range)", fontsize=14)
    save(fig, "explain_hit_rate.png")


def phase2_rmse():
    models = ["Simple LSTM\n(BTC only)", "Simple GRU\n(BTC only)", "Bi-LSTM\n(19 inputs)", "Alpha Agent\n(19 inputs +\nattention)"]
    mean = [2.337, 2.348, 2.689, 2.372]
    std = [0.005, 0.019, 0.445, 0.051]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    cols = [GREY, GREY, GREY, BLUE]
    bars = ax.bar(models, mean, yerr=std, capsize=6, color=cols, width=0.6)
    for b, m in zip(bars, mean):
        ax.text(b.get_x() + b.get_width() / 2, 1.85, f"{m:.2f}", ha="center", fontsize=13, color="white")
    ax.set_ylim(1.7, 3.3)
    ax.set_ylabel("Prediction error (RMSE, %)")
    ax.set_title("Phase 2: tomorrow's return prediction error (lower is better; whisker = spread over 5 runs)",
                 fontsize=14)
    save(fig, "explain_phase2_rmse.png")


def significance():
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    panels = [("Drawdown reduction (percentage points)", [("2025 test", 12.6, 7.0, 20.9), ("2026 forward", 14.1, 4.5, 19.2)]),
              ("Sharpe improvement", [("2025 test", 1.24, 0.20, 2.45), ("2026 forward", 0.53, -0.44, 1.65)])]
    for ax, (title, rows) in zip(axes, panels):
        for i, (lab, obs, lo, hi) in enumerate(rows[::-1]):
            col = GREEN if lo > 0 else MUTED
            ax.plot([lo, hi], [i, i], color=col, lw=6, solid_capstyle="round")
            ax.scatter([obs], [i], color=INK, s=110, zorder=3)
            ax.text(obs, i + 0.22, f"{obs:+.2f}" if abs(obs) < 5 else f"{obs:+.1f}", ha="center", fontsize=13)
        ax.axvline(0, color=RED, ls="--", lw=1.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows[::-1]])
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.set_title(title, fontsize=14)
    fig.suptitle("How sure are we? Bar = 95% range from 5,000 reshuffles. Green = clear of zero (real), "
                 "grey = could be luck", fontsize=13)
    save(fig, "explain_significance.png")


def rule_ablation():
    rules = [("Volatility-based stop-loss", 3.6), ("Weak-prediction filter", 3.3), ("Volatility-based sizing", 2.2),
             ("Jump-risk veto", 1.0), ("Jump-based leverage cut", 0.3), ("ALL rules switched off", 12.6)]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = np.arange(len(rules))[::-1]
    cols = [BLUE] * 5 + [RED]
    bars = ax.barh(y, [v for _, v in rules], color=cols, height=0.6)
    for yy, (_, v) in zip(y, rules):
        ax.text(v + 0.2, yy, f"+{v} pp", va="center", fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels([r for r, _ in rules])
    ax.set_xlim(0, 15)
    ax.set_xlabel("Extra drawdown when switched off (percentage points)")
    ax.set_title("Which rule matters? Each alone adds a little; all together they matter a lot (2025)",
                 fontsize=14)
    save(fig, "explain_rule_ablation.png")


def leverage_stress():
    lev = [2, 5, 10, 20]
    re_ = [4.6, 12.6, 24.2, 38.2]
    fx = [17.2, 38.3, 42.2, 68.8]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(lev, fx, "o-", color=FIX_C, lw=2.6, ms=9, label="Fixed rules")
    ax.plot(lev, re_, "o-", color=RE_C, lw=2.6, ms=9, label="Risk Engine")
    for x, a, b in zip(lev, re_, fx):
        ax.text(x, a - 4.5, f"{a:.0f}%", ha="center", color=RE_C, fontsize=13)
        ax.text(x, b + 2.5, f"{b:.0f}%", ha="center", color=FIX_C, fontsize=13)
    ax.annotate("2 liquidations\n(Risk Engine only)", (20, 38.2), xytext=(8, 5), fontsize=13, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xscale("log", base=2)
    ax.set_xticks(lev)
    ax.set_xticklabels([f"{v}x" for v in lev])
    ax.set_ylim(0, 80)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Worst fall from a high (%)")
    ax.set_title("Leverage stress test (2025): the Risk Engine always loses less", fontsize=15)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "explain_leverage_stress.png")


def leverage_menu():
    lev = ["2x", "5x", "10x", "20x"]
    avg = [2.4, 5.6, 9.1, -12.1]
    best = [15.4, 17.9, 36.3, 48.3]
    worst = [-7.7, -9.5, -22.3, -72.0]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(lev))
    cols = [GREEN if v > 0 else RED for v in avg]
    ax.bar(x, avg, color=cols, width=0.5)
    ax.vlines(x, worst, best, color=INK, lw=2)
    ax.scatter(x, best, color=INK, marker="_", s=400)
    ax.scatter(x, worst, color=INK, marker="_", s=400)
    for xx, a, b, w in zip(x, avg, best, worst):
        ax.text(xx, b + 2, f"best {b:+.0f}%", ha="center", fontsize=11, color=MUTED)
        ax.text(xx, w - 6, f"worst {w:+.0f}%", ha="center", fontsize=11, color=MUTED)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(-5.8, color=BH_C, ls="--", lw=1.2, label="buy and hold BTC: -5.8%")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\naverage {a:+.1f}%" for l, a in zip(lev, avg)])
    ax.set_xlim(-0.6, 3.5)
    ax.set_ylim(-85, 62)
    ax.set_ylabel("Return over 9 months of 2026 (%)")
    ax.set_title("More leverage = wider spread of outcomes, not a better system (2026)", fontsize=15)
    save(fig, "explain_leverage_menu.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    btc_splits()
    phase2_rmse()
    hit_rate()
    vol_errors()
    worst_drop()
    significance()
    rule_ablation()
    leverage_stress()
    leverage_menu()
    vol_forecast_example()
    d25, f25, b25 = runs_2025()
    equity(d25, f25, b25, "2025", "explain_equity_2025.png")
    d26, f26, b26 = runs_2026()
    equity(d26, f26, b26, "2026", "explain_equity_2026.png")
    drawdown_example(b26)
