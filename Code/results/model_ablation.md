# Forecaster ablation inside the full framework (Phase 5)

Beta Agent + Dynamic Risk Engine unchanged; only the direction forecaster differs.
Each version retuned on 2024 validation, run once on the 2025 test. Mean ± std over 5 seeds.

| Forecaster | Max drawdown | Sharpe | Final balance (from R1,000) | Trades | Hit rate |
|---|---|---|---|---|---|
| Alpha Agent (attention) | 4.6% ± 1.1% | 0.74 ± 1.09 | R1,045 | 63 | 51.4% ± 1.3% |
| Bi-LSTM, no attention | 3.4% ± 1.4% | 0.36 ± 0.69 | R1,010 | 25 | 50.7% ± 0.8% |
| Univariate LSTM | 9.2% ± 1.6% | -0.44 ± 0.16 | R973 | 79 | 53.6% ± 1.3% |

## Is the Alpha Agent version better? (paired block bootstrap, 5-seed average)

Positive = the Alpha Agent version is better. One-sided p.

| Compared with | Drawdown lower by (pp) | 95% interval | p | Sharpe higher by | 95% interval | p |
|---|---|---|---|---|---|---|
| Bi-LSTM, no attention | -1.2 | [-4.1, +1.3] | 0.8684 | +0.38 | [-1.27, +2.04] | 0.3506 |
| Univariate LSTM | +4.7 | [-1.3, +10.3] | 0.0982 | +1.17 | [-0.59, +3.18] | 0.0972 |

Hit rate: share of test days where the forecast's sign matched the actual next-day move.
The univariate LSTM also differs in architecture, so it isn't a pure test of the features.

Figure: figures/phase5_model_ablation.png
