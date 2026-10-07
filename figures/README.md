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

### `phase5_rule_ablation.png`
- **Shows:** max drawdown (left) and Sharpe (right) on the 2025 test for the full
  Dynamic Risk Engine (dark blue), versions with one rule switched off (light blue),
  and the all-fixed H3 ablation (orange). Bars = mean over 5 seeds, whiskers = std.
- **Made by:** `Code/ablate_rules.py` (numbers and bootstrap tests in
  `Code/results/rule_ablation.md`).
- **Notes for the caption:** each version is retuned on 2024 validation, so it's
  judged with its own best settings, not the full engine's. Retuning can shift
  other settings too (e.g. the fixed-stop version chose a looser jump veto and no
  weak-prediction filter, so it trades much more) - the bars show each version at
  its best, not a pure one-switch change.

### `phase5_leverage_stress.png`
- **Shows:** max drawdown (left) and Sharpe (right) on the 2025 test for both engines
  as the leverage cap rises from 2x to 20x (log scale). Lines = mean over 5 seeds,
  shaded bands = ± std. The annotation marks the only liquidations (Risk Engine, 20x).
- **Made by:** `Code/stress_test.py` (numbers and bootstrap tests in
  `Code/results/leverage_stress.md`).
- **Notes for the caption:** both engines are retuned on 2024 validation at each cap;
  the Risk Engine's actual leverage is cap × (1 − jump probability). Liquidations
  are rare because a stop tighter than 1/leverage always fires first and BTC (24/7)
  almost never opens past a stop; stops fill without slippage, which understates
  live liquidation risk. At 20x the liquidation distance is 5%, and the Risk
  Engine's volatility-widened stops can sit beyond it - hence its 2 liquidations.

### `phase5_model_ablation.png`
- **Shows:** the full framework (Beta Agent + Dynamic Risk Engine, unchanged) run
  with three different direction forecasters - Alpha Agent (attention), the same
  Bi-LSTM without attention, and the univariate LSTM. Panels: max drawdown, Sharpe,
  and each forecaster's directional hit rate (dashed line = 50%, a coin flip).
  Bars = mean over 5 seeds, whiskers = std.
- **Made by:** `Code/model_ablation.py` (numbers and bootstrap tests in
  `Code/results/model_ablation.md`).
- **Notes for the caption:** each version's Risk Engine settings retuned on 2024
  validation. The univariate LSTM also has a simpler architecture, so it is not a
  pure test of the multivariate features (Phase 2's H1 test is the clean one).

### `phase5_forward_2026.png`
- **Shows:** account balance from R1,000 over January-September 2026 for the
  Dynamic Risk Engine (blue, one line per seed), the fixed-rule ablation (orange)
  and 1x buy-and-hold BTC (black dashed) - data no model, setting or design
  decision had seen.
- **Made by:** `Code/forward_test.py run` (numbers in `Code/results/forward_2026.md`).
- **Notes for the caption:** nothing retrained, refitted or retuned - saved model
  checkpoints, the frozen train-only feature scaler, and Risk Engine settings
  chosen on 2024. The 2025 test year influenced some design choices; 2026 did not,
  so this is the cleanest out-of-sample result in the project.

### `phase5_forward_2026_leverage.png`
- **Shows:** a "choose your risk level" menu on the unseen 2026 data. Left: return
  over Jan-Sep 2026 for each run (dots) and the average (black bar) at leverage
  caps 2x/5x/10x/20x, with buy-and-hold BTC dashed. Right: average worst drawdown
  at each cap, with buy-and-hold dashed.
- **Made by:** `Code/forward_test.py run` (table in `Code/results/forward_2026.md`,
  "Choosing a risk level" section).
- **Notes for the caption:** every other Risk Engine setting tuned on 2024 at each
  cap (as in the leverage stress test), then run once on 2026 - all four caps are
  shown, none picked after the fact. More leverage widens the spread of outcomes
  without raising return per unit of risk; at 20x it turns to an average loss.
