# Forward test on unseen 2026 data (Phase 5)

Period: 2026-01-01 to 2026-09-30, 194 trading days.
Nothing retrained, refitted or retuned: saved checkpoints, the frozen train-only scaler,
and the Risk Engine settings chosen on 2024 (results/chosen_params.json).

## Trading results (mean ± std over 5 runs)

| Metric | Dynamic Risk Engine | Fixed-rule ablation | Buy & hold BTC (1x) |
|---|---|---|---|
| Final balance (from R1,000) | R1,024 ± R86 | R986 ± R196 | R942 |
| Lowest balance | R943 ± R57 | R835 ± R120 | R660 |
| Max drawdown | 8.7% ± 3.8% | 22.8% ± 6.4% | 39.6% |
| Sharpe | 0.352 ± 1.018 | -0.179 ± 1.387 | 0.044 |
| Sortino | 0.771 ± 1.866 | -0.101 ± 2.512 | 0.065 |
| Trades | 98.400 ± 35.816 | 194.000 ± 0.000 | - |
| Liquidation rate | 0% | 0% | - |

## Risk Engine vs fixed rules (paired block bootstrap, 5-seed average)

| | Observed | 95% interval | p (one-sided) |
|---|---|---|---|
| Drawdown reduction (pp) | +14.1 | [+4.5, +19.2] | < 0.0002 |
| Sharpe gain | +0.53 | [-0.44, +1.65] | 0.1394 |

## Forecasts on the forward period

- Alpha Agent directional hit rate: 49.3% ± 2.1% (share of up days: 47.4%)
- Beta Agent volatility RMSE: 0.02038 ± 0.00101 (naive persistence: 0.02500)

## Per-run metrics

Dynamic Risk Engine:

| seed | total_return | max_drawdown | sharpe | sortino | n_trades | n_liquidations |
|---|---|---|---|---|---|---|
| 0 | 0.9860 | 0.1205 | -0.1031 | -0.1655 | 138 | 0 |
| 1 | 1.0493 | 0.0318 | 0.9325 | 1.8388 | 42 | 0 |
| 2 | 0.9226 | 0.1250 | -0.9474 | -1.4395 | 104 | 0 |
| 3 | 1.1540 | 0.0810 | 1.7211 | 3.3697 | 116 | 0 |
| 4 | 1.0071 | 0.0783 | 0.1553 | 0.2504 | 92 | 0 |

Fixed-rule ablation:

| seed | total_return | max_drawdown | sharpe | sortino | n_trades | n_liquidations |
|---|---|---|---|---|---|---|
| 0 | 0.8288 | 0.2879 | -1.3341 | -2.1346 | 194 | 0 |
| 1 | 1.2598 | 0.1484 | 1.6770 | 3.3635 | 194 | 0 |
| 2 | 0.8852 | 0.2394 | -0.8131 | -1.3471 | 194 | 0 |
| 3 | 1.1257 | 0.1768 | 0.9085 | 1.7456 | 194 | 0 |
| 4 | 0.8288 | 0.2879 | -1.3341 | -2.1346 | 194 | 0 |

Figure: figures/phase5_forward_2026.png
