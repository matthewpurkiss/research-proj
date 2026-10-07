# H3 significance tests (Phase 5)

Dynamic Risk Engine vs fixed-rule ablation, same tuned runs as backtest_report.md. 201 test days, 5 runs.

## 1. Daily-return test per seed (DM-style, Newey-West variance)

Loss differential d = return_fixed − return_dynamic; Newey-West lags = 4.

| seed | mean daily advantage of Risk Engine | t | p (two-sided) | verdict |
|---|---|---|---|---|
| 0 | +0.0012% | +0.02 | 0.9856 | not significant |
| 1 | +0.0647% | +1.10 | 0.2725 | not significant |
| 2 | +0.0419% | +0.69 | 0.4922 | not significant |
| 3 | +0.0786% | +1.39 | 0.1664 | not significant |
| 4 | +0.0813% | +1.21 | 0.2278 | not significant |

Significant at alpha=0.05 on 0 of 5 seeds.

## 2. Block bootstrap (5,000 resamples, 10-day blocks)

Positive = Risk Engine better. p = share of resamples showing no advantage (one-sided). 95% interval from the resample percentiles.

| | Drawdown reduction (pp) | 95% interval | p | Sharpe gain | 95% interval | p |
|---|---|---|---|---|---|---|
| seed 0 | +11.3 | [+1.2, +21.4] | 0.0124 | +0.22 | [-2.10, +2.34] | 0.4390 |
| seed 1 | +15.4 | [+4.5, +28.1] | 0.0002 | +0.38 | [-1.76, +1.90] | 0.4390 |
| seed 2 | +10.6 | [+2.5, +23.2] | 0.0030 | +1.45 | [-0.70, +3.84] | 0.0794 |
| seed 3 | +10.9 | [+2.6, +24.1] | 0.0026 | +2.00 | [+0.10, +3.78] | 0.0188 |
| seed 4 | +14.7 | [+5.6, +34.4] | < 0.0002 | +2.15 | [+0.40, +4.87] | 0.0098 |
| **average of 5** | +12.6 | [+7.0, +20.9] | < 0.0002 | +1.24 | [+0.20, +2.45] | 0.0114 |

Liquidation rate is not tested: neither engine liquidates at the tuned 2x leverage.

Figure: figures/phase5_h3_bootstrap.png (seed-averaged bootstrap distributions).
