"""
make_figures.py

Write-up figures that aren't produced by an experiment script:

  figures/framework_diagram.png       how the Alpha Agent, Beta Agent and Dynamic
                                      Risk Engine fit together
  figures/phase3_volatility_target.png  the finding-1 investigation: skill vs naive
                                      persistence for every volatility-target
                                      variant tried, ending with the single-day
                                      target that was adopted

The volatility-target numbers are the confirmed results recorded in HANDOFF.md
(finding 1). The five 14-day-target variants came from throwaway experiment
scripts retired on 2026-10-07 (in git history at commit 3eb8226); the
single-day row is the official skill_scores.py result.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURES_DIR = os.path.join("..", "figures")

INK = "#1B2433"
BLUE = "#1F77B4"
ORANGE = "#F7931A"
GREEN = "#2E9E6A"
GREY = "#7F8C9A"
LIGHT = "#F1F4F8"


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=color, linewidth=2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=14,
            fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.56, "\n".join(lines), ha="center", va="top",
            fontsize=11, color=INK, linespacing=1.4)


def arrow(ax, start, end, label=None, label_offset=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 color=GREY, linewidth=1.6))
    if label:
        mx, my = (start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8.5, color=INK, style="italic")


def framework_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    box(ax, 0.05, 1.3, 2.25, 2.0, "Daily data", [
        "19 features, 2018-2025",
        "BTC/ETH, S&P 500, gold,", "dollar index, VIX,",
        "volatility, correlations,", "on-chain, volumes",
    ], INK)
    box(ax, 3.0, 2.55, 2.75, 1.55, "Alpha Agent", [
        "Bi-LSTM + attention",
        "60-day window",
        "tomorrow's return (direction)",
    ], BLUE)
    box(ax, 3.0, 0.4, 2.75, 1.55, "Beta Agent", [
        "two independent GRUs",
        "30-day window",
        "tomorrow's volatility + jump risk",
    ], ORANGE)
    box(ax, 7.45, 0.9, 2.65, 2.75, "Dynamic Risk Engine", [
        "fixed rules, not learned:",
        "1  jump-risk veto",
        "2  direction + weak-signal filter",
        "3  size = k / volatility",
        "4  stop = c x volatility",
        "5  leverage = L x (1 - jump prob)",
    ], GREEN)
    box(ax, 10.55, 1.45, 1.4, 1.7, "Trade", [
        "long / short",
        "/ flat; size,",
        "stop, leverage",
    ], INK)

    arrow(ax, (2.3, 2.65), (3.0, 3.3))
    arrow(ax, (2.3, 1.95), (3.0, 1.2))
    arrow(ax, (5.75, 3.3), (7.45, 3.0))
    ax.text(6.6, 3.3, "return forecast", ha="center", va="bottom", fontsize=10, color=INK, style="italic")
    arrow(ax, (5.75, 1.2), (7.45, 1.5))
    ax.text(6.6, 1.15, "volatility, jump risk", ha="center", va="top", fontsize=10, color=INK, style="italic")
    arrow(ax, (10.1, 2.3), (10.55, 2.3))

    ax.text(6.0, 4.55, "Decoupled framework: direction and risk are forecast by separate models,"
            " then combined by deterministic rules", ha="center", va="top", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "framework_diagram.png"), dpi=200)


def volatility_target():
    variants = [
        ("14-day, absolute level (original)", -765.9),
        ("14-day, predict daily change", -404.1),
        ("14-day, daily change (control)", -498.5),
        ("14-day + GARCH as input", -593.9),
        ("14-day, change + GARCH input", -604.3),
        ("Single-day (adopted)", 5.0),
    ]
    labels = [v[0] for v in variants][::-1]
    values = [v[1] for v in variants][::-1]
    colors = [GREEN if v > 0 else GREY for v in values]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color=INK, linewidth=1)
    for bar, v in zip(bars, values):
        x = v - 15 if v < 0 else v + 15
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{v:+.0f}%", va="center",
                ha="right" if v < 0 else "left", fontsize=12, color=INK)
    ax.set_xlim(-900, 160)
    ax.set_xlabel("Skill vs naive persistence (%): right of zero = better than\n'tomorrow looks like today'", fontsize=11)
    ax.set_title("Every 14-day-window variant lost to naive persistence;\n"
                 "the single-day target fixed it", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "phase3_volatility_target.png"), dpi=200)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    framework_diagram()
    volatility_target()
    print("wrote figures/framework_diagram.png and figures/phase3_volatility_target.png")
