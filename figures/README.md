# Figures for the write-up

Every chart and diagram made for the project lives here. Each entry says what the
figure shows, which script makes it, and anything a reader needs to interpret it.
Regenerate a figure by re-running its script from `Code/`; don't edit images by hand.

## Phase 4: backtest

### `phase4_equity_curves.png`
- **Shows:** account equity over the 2025 test window for the Dynamic Risk Engine
  (blue, one line per seed), the fixed-rule ablation (orange, one line per seed)
  and 1x buy-and-hold BTC (black dashed). Starting capital = 1.0.
- **Made by:** `Code/run_backtest.py`.
- **Notes for the caption:** test window is 2025-03-25 to 2025-12-31 (201 weekdays;
  the Alpha Agent's 60-day lookback uses up January to March). Settings for both
  engines tuned on 2024 validation only. 0.1% round-trip costs included. Flat
  stretches in the blue lines are days the engine stayed out of the market (jump
  veto or weak prediction). Both engines were tuned to 2x leverage.
