# Leverage stress test (Phase 5)

Both engines retuned on 2024 validation at each leverage cap, then run once on the
2025 test. Mean over 5 seeds. Liquidations = total across the 5 runs; rate = share
of runs with at least one.

| Leverage | Engine | Max drawdown | Sharpe | Final balance (from R1,000) | Liquidations (rate) | Settings |
|---|---|---|---|---|---|---|
| 2x | Risk Engine | 4.6% ± 1.1% | 0.74 ± 1.09 | R1,045 | 0 (0%) | tau_j=0.25, tau_y=0.33, k=0.005, c=1.0 |
| 2x | Fixed rules | 17.2% ± 3.0% | -0.50 ± 0.59 | R931 | 0 (0%) | tau_y=0.00, fraction=0.25, stop=0.02 |
| 5x | Risk Engine | 12.6% ± 1.4% | 0.72 ± 0.95 | R1,106 | 0 (0%) | tau_j=0.1, tau_y=0.33, k=0.005, c=2.0 |
| 5x | Fixed rules | 38.3% ± 5.5% | -0.50 ± 0.59 | R810 | 0 (0%) | tau_y=0.00, fraction=0.25, stop=0.02 |
| 10x | Risk Engine | 24.2% ± 2.3% | 0.72 ± 0.95 | R1,214 | 0 (0%) | tau_j=0.1, tau_y=0.33, k=0.005, c=2.0 |
| 10x | Fixed rules | 42.2% ± 12.1% | -0.02 ± 0.80 | R920 | 0 (0%) | tau_y=0.33, fraction=0.25, stop=0.02 |
| 20x | Risk Engine | 38.2% ± 8.2% | 0.77 ± 1.16 | R1,500 | 2 (20%) | tau_j=0.1, tau_y=0.33, k=0.005, c=1.0 |
| 20x | Fixed rules | 68.8% ± 14.9% | -0.02 ± 0.80 | R691 | 0 (0%) | tau_y=0.33, fraction=0.25, stop=0.02 |

## Risk Engine advantage at each leverage (paired block bootstrap, 5-seed average)

| Leverage | Drawdown reduction (pp) | 95% interval | p | Sharpe gain | 95% interval | p |
|---|---|---|---|---|---|---|
| 2x | +12.6 | [+7.0, +20.9] | < 0.0002 | +1.24 | [+0.20, +2.45] | 0.0114 |
| 5x | +25.6 | [+15.7, +44.5] | < 0.0002 | +1.22 | [+0.12, +2.66] | 0.0160 |
| 10x | +18.0 | [+10.8, +46.8] | < 0.0002 | +0.74 | [-0.16, +2.10] | 0.0464 |
| 20x | +30.6 | [+22.4, +62.7] | < 0.0002 | +0.79 | [-0.10, +2.09] | 0.0404 |

Why liquidations are rare here: a stop tighter than the liquidation distance (1/leverage)
always fires first, and BTC trades 24/7 so it almost never opens beyond a stop (more
than 2% from the previous close on 3 of 2,922 days). Stops fill exactly at the stop
price (no slippage), which understates liquidation risk compared with live markets.

Figure: figures/phase5_leverage_stress.png
