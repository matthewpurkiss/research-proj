# Backtest report (Phase 4)

Test period: 2025 weekdays with both an Alpha and a Beta window. 5 runs (Alpha seed s
paired with Beta seed s). Settings tuned on 2024 validation only. Costs: 0.10% of notional per round trip.

## Chosen settings (validation grid search, objective = Sharpe − max drawdown)

- Dynamic Risk Engine: tau_j=0.25, tau_y=0.33, k=0.005, c=1.0, L_max=2.0 (validation objective 0.213)
- Fixed-rule ablation: tau_y=0.00, fraction=0.25, stop=0.02, leverage=2.0 (validation objective 0.886)

## Test results (mean ± std over 5 runs)

| Metric | Dynamic Risk Engine | Fixed-rule ablation | Buy & hold BTC (1x) |
|---|---|---|---|
| Total return (multiple) | 1.045 ± 0.060 | 0.931 ± 0.067 | 1.000 |
| Max drawdown | 4.6% ± 1.1% | 17.2% ± 3.0% | 31.8% |
| Sharpe | 0.736 ± 1.086 | -0.503 ± 0.590 | 0.190 |
| Sortino | 1.427 ± 1.837 | -0.773 ± 0.897 | 0.274 |
| Trades | 62.800 ± 20.897 | 201.000 ± 0.000 | - |
| Liquidation rate | 0% | 0% | - |

Statistical comparison of the two engines (H3) is Phase 5, not this report.

## Per-run test metrics

Dynamic Risk Engine:

| seed | total_return | max_drawdown | sharpe | sortino | n_trades | n_liquidations |
|---|---|---|---|---|---|---|
| 0 | 1.0139 | 0.0428 | 0.3147 | 0.5200 | 68 | 0 |
| 1 | 0.9595 | 0.0644 | -0.9852 | -1.3854 | 41 | 0 |
| 2 | 1.0745 | 0.0378 | 1.3667 | 2.5348 | 53 | 0 |
| 3 | 1.1131 | 0.0479 | 1.6316 | 2.9839 | 96 | 0 |
| 4 | 1.0647 | 0.0365 | 1.3501 | 2.4812 | 56 | 0 |

Fixed-rule ablation:

| seed | total_return | max_drawdown | sharpe | sortino | n_trades | n_liquidations |
|---|---|---|---|---|---|---|
| 0 | 1.0019 | 0.1553 | 0.0983 | 0.1653 | 201 | 0 |
| 1 | 0.8350 | 0.2186 | -1.3660 | -2.0683 | 201 | 0 |
| 2 | 0.9788 | 0.1437 | -0.0805 | -0.1333 | 201 | 0 |
| 3 | 0.9428 | 0.1566 | -0.3638 | -0.5926 | 201 | 0 |
| 4 | 0.8967 | 0.1834 | -0.8034 | -1.2377 | 201 | 0 |
