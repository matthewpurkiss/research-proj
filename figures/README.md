# Figures for the write-up

Every chart and diagram made for the project lives here. Each entry says what the
figure shows, which script makes it, and anything a reader needs to interpret it.
Regenerate a figure by re-running its script from `Code/`; don't edit images by hand.

## Phase 4: backtest

### `phase4_equity_curves.png`
- **Shows:** account balance over the 2025 test window for the Dynamic Risk Engine
  (blue, one line per seed), the fixed-rule ablation (orange, one line per seed)
  and 1x buy-and-hold BTC (black dashed), starting from R1,000 (set by
  `STARTING_CAPITAL` in `run_backtest.py`; every result scales in proportion, and
  balances ignore USD/ZAR moves and brokers' minimum trade sizes).
- **Made by:** `Code/run_backtest.py`.
- **Notes for the caption:** test window is 2025-03-25 to 2025-12-31 (201 weekdays;
  the Alpha Agent's 60-day lookback uses up January to March). Settings for both
  engines tuned on 2024 validation only. 0.1% round-trip costs included. Flat
  stretches in the blue lines are days the engine stayed out of the market (jump
  veto or weak prediction). Both engines were tuned to 2x leverage.

## Phase 5: evaluation

### `phase5_h3_bootstrap.png`
- **Shows:** for H3, the bootstrap distribution of how much the Dynamic Risk Engine
  beats the fixed-rule ablation, averaged over the 5 seeds - left: max drawdown
  reduction (percentage points), right: Sharpe gain. Black line = what actually
  happened on the 2025 test; red dashed = 95% interval; grey = zero (no advantage).
  An interval clear of zero means the advantage is unlikely to be luck.
- **Made by:** `Code/h3_tests.py`.
- **Notes for the caption:** 5,000 resamples of the 201 test days in 10-day blocks
  (keeps short-run volatility clustering), the same resample applied to both
  engines and all seeds, so the comparison is paired. Settings as tuned on 2024
  validation by `run_backtest.py`.
