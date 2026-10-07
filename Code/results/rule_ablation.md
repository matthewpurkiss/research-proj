# Rule-by-rule ablation (Phase 5)

Each version swaps one Beta-driven rule for its fixed version, is retuned on 2024
validation, and is run once on the 2025 test. Mean over 5 seeds.

| Version | Max drawdown | Sharpe | Final balance (from R1,000) | Trades | Settings chosen |
|---|---|---|---|---|---|
| All five rules | 4.6% ± 1.1% | 0.74 ± 1.09 | R1,045 | 63 | tau_y=0.33, tau_j=0.25, k=0.005, c=1.0, L=2.0 |
| No weak-prediction filter | 7.9% ± 3.3% | 0.39 ± 1.18 | R1,033 | 121 | tau_y=0.00, tau_j=0.25, k=0.005, c=1.0, L=2.0 |
| No jump veto | 5.6% ± 2.1% | 0.47 ± 1.21 | R1,034 | 122 | tau_y=0.33, no veto, k=0.005, c=1.0, L=2.0 |
| Fixed position size | 6.8% ± 1.9% | 0.71 ± 0.98 | R1,063 | 63 | tau_y=0.33, tau_j=0.25, fraction=0.25, c=1.0, L=2.0 |
| Fixed stop-loss | 8.2% ± 3.0% | -0.14 ± 0.85 | R990 | 177 | tau_y=0.00, tau_j=0.5, k=0.005, stop=0.02, L=2.0 |
| Fixed leverage | 4.9% ± 1.1% | 0.74 ± 1.07 | R1,047 | 63 | tau_y=0.33, tau_j=0.25, k=0.005, c=1.0, L=2.0 fixed |
| All fixed (H3 ablation) | 17.2% ± 3.0% | -0.50 ± 0.59 | R931 | 201 | see backtest_report.md |

## How much does each rule help? (paired block bootstrap vs the full engine)

Removing the rule makes drawdown worse by / Sharpe lower by - positive = the rule helps.
Average of 5 seeds, 95% interval, one-sided p.

| Rule removed | Drawdown worse by (pp) | 95% interval | p | Sharpe lower by | 95% interval | p |
|---|---|---|---|---|---|---|
| No weak-prediction filter | +3.3 | [+1.1, +7.3] | 0.0002 | +0.34 | [-0.60, +1.16] | 0.2988 |
| No jump veto | +1.0 | [-0.0, +3.3] | 0.0298 | +0.27 | [-0.13, +0.86] | 0.0966 |
| Fixed position size | +2.2 | [+1.0, +3.3] | < 0.0002 | +0.02 | [-0.13, +0.17] | 0.3350 |
| Fixed stop-loss | +3.6 | [+1.4, +8.5] | 0.0008 | +0.87 | [-0.14, +1.88] | 0.0444 |
| Fixed leverage | +0.3 | [+0.1, +0.8] | 0.0024 | -0.01 | [-0.07, +0.07] | 0.4882 |
| All fixed (H3 ablation) | +12.6 | [+7.0, +20.9] | < 0.0002 | +1.24 | [+0.20, +2.45] | 0.0114 |

Figure: figures/phase5_rule_ablation.png
