# Project Handoff Document

Written 2026-09-21 to let a new Claude session (or another LLM/account) continue this
project with full context, either for more coding or for helping write the final
research report. Read this whole file before doing anything else in this project.

## Project identity

- **Title:** A Decoupled Attention-Based Bi-LSTM Framework with Volatility-Aware Risk
  Management for AI-Driven CFD Cryptocurrency Trading
- **Student:** Matthew Purkiss, 2697424
- **Supervisor:** Clint Van Alten
- **Institution:** School of Computer Science & Applied Mathematics, University of the
  Witwatersrand. BSc Honours, 40% of final mark, oral defence in Week 20.
- **Project timeline:** 20 weeks, started 19 July 2026. Phase 1 = Weeks 1-3, Phase 2 =
  Weeks 4-8, Phase 3 = Weeks 9-11 (current phase, in progress as of this document).
- **GitHub repo:** https://github.com/matthewpurkiss/research-proj (private)

## Folder structure

```
Research_Proj/                          <- repo root
  Matthew_Purkiss_research_proposal.pdf <- THE key source document, cite section numbers from this
  Matthew_Purkiss_Lit_Review.pdf
  Annotated_Bibliography_2697424.pdf
  Phase1_Progress.pptx                  <- supervisor slide deck, Phase 1
  PHASE3_SUMMARY.pdf                    <- formal Phase 3 results summary
  HANDOFF.md                            <- this file
  figures/                              <- EVERY write-up chart/diagram, indexed in figures/README.md
  Code/
    CLAUDE.md              <- project rules/conventions, READ THIS TOO
    PHASE1_SPEC.md          <- detailed Phase 1 spec, still useful reference
    README.md               <- Phase 1 install/run-order/findings doc
    PHASE2_SUMMARY.pdf       <- formal Phase 2 results summary
    config.py, fetch_data.py, build_dataset.py, splits.py,
    leakage_audit.py, test_audit_teeth.py, make_synthetic.py,
    run_pipeline.py, check_connectivity.py, eda.ipynb   <- Phase 1 pipeline (all done)
    models.py, train.py, evaluate.py, metrics.py         <- Phase 2 (all done)
    train_beta.py, garch_baseline.py, skill_scores.py     <- Phase 3 (done)
    risk_engine.py, backtest.py, signals.py, run_backtest.py  <- Phase 4 (done)
    checkpoints/    <- saved Alpha + Beta weights per seed (beta files also hold Platt A, B)
    results/        <- backtest_report.md, test_metrics.csv, chosen_params.json
    (finding-1 investigation scripts - diagnose_beta_vol.py and the five
     experiment_*.py files - were removed on 2026-10-07 when the single-day
     target became official, since they assume the old 14-day target and would
     print wrong numbers against the rebuilt data. Code is in git history at
     commit 3eb8226; results are recorded under finding 1 below.)
    data/raw/       <- real fetched CSVs (fetched 2026-08-12, frozen 2026-10-07 - see DATA_FREEZE.md)
    data/processed/ <- aligned_dataset.csv, train/val/test.csv, *_windows.npz, scaler_params.csv
```

**`Code/CLAUDE.md` is the authoritative rules file for this project** - read it fully
before writing more code. Key non-negotiables it states: never fabricate a number,
leakage is the primary risk, prefer clear code over clever code (every line must be
explainable at defence), work section by section, explain before implementing.

## How the student likes to work (from memory files, confirmed repeatedly)

1. **The student commits their own code.** Claude writes/edits files and gives the
   exact `git add` / `git commit` / `git push` commands, but does not run them.
   Exception: the very first repo-setup commit was Claude's.
2. **The student runs all scripts/training themselves** and pastes the output back.
   Claude writes code, explains what to expect, and does NOT execute training runs or
   pipeline scripts via Bash. (Claude may run small, quick self-verification checks of
   its own new code before handing it off - e.g. confirming a new function's output
   shape - but the "real" recorded run is always the student's own.)
3. **Explain before implementing.** For any non-trivial concept or design decision,
   explain it in plain English first, confirm understanding, then write code.
4. Work section by section - don't dump multiple files/features unrequested.
5. The student prefers concise, direct answers.

## Critical unresolved issue: AI declaration

The submitted research proposal's AI Declaration page has **box 6 ("Code
Development") and box 3 ("Methods and Experiment Design") NOT checked**, but this
entire project (Phase 1 and Phase 2 pipelines, Phase 3 in progress) has been built
with extensive Claude assistance, including real methodological decisions (e.g. the
finding-1 resolution below). The student's position (stated explicitly): the checked
proposal declaration only covered the proposal document itself, and they will check
the correct boxes on the final thesis/report's own AI declaration. **Whoever continues
this project should be aware this is still an open item the student needs to handle
correctly at final submission - do not let it be forgotten.**

## Phase 1 status: COMPLETE

All 10 files from `PHASE1_SPEC.md` built, tested, and verified against real data.
Leakage audit: **9/9 checks pass**. Mutation test (`test_audit_teeth.py`): **5/5
faults caught**.

### Key config values (see `config.py` for authoritative source)
- Data range: 2018-01-01 to 2025-12-31 (raw fetch), usable range after warm-up trim:
  **2018-02-12 to 2025-12-30**
- Splits: train 2018-01-01/2023-12-31, val 2024, test 2025
- Tickers: BTC-USD, ETH-USD, ^GSPC, GC=F, DX-Y.NYB, ^VIX (Yahoo); 6 Blockchain.com
  charts (active addresses, hash rate, tx count, mempool size, miner revenue,
  difficulty)
- `WARMUP_TRIM_DAYS = 30` (longest rolling window: 30-day correlation / jump
  threshold)
- `ALPHA_LOOKBACK = 60`, `BETA_LOOKBACK = 30`
- **d = 19 features**, pinned order in `config.FEATURE_ORDER`: 6 log-returns (BTC,
  ETH, SP500, Gold, DXY, VIX) + BTC realised vol (14d) + 3 rolling correlations (30d,
  BTC vs ETH/SP500/Gold) + 6 on-chain 7d rate-of-change + 3 log-volumes (BTC, ETH,
  SP500)
- 3 targets per row: `alpha_target` (next-day BTC log-return), `beta_target_vol`
  (next-day 14d trailing realised vol), `beta_target_jump` (next-day binary jump
  label, 3-std-of-trailing-30d threshold)

### Real data-quality bugs found and fixed (not in original spec)
1. `yfinance`'s `end` parameter is exclusive - every Yahoo series silently dropped the
   last day. Fixed in `fetch_data.py` by requesting one day past `DATA_END`.
2. 2018-01-01 is a real market holiday AND the first day of the dataset, so
   forward-fill has nothing to reach back to. Fixed with a backfill capped at 3 rows
   in `build_dataset.py`.
3. SP500 reported `Volume = 0` on 2023-05-24, causing `log(0) = -inf`. Treated as a
   data error, imputed like any other gap.
4. Blockchain.com's Charts API silently downsamples long date ranges unless
   `sampled=false` is passed (confirmed: 729 points instead of ~2900 for an 8-year
   request without it). Fixed in `fetch_data.py`.
5. On-chain data has small persistent gaps (a 3-day outage across 4 metrics in Nov
   2025, 3 isolated single-day gaps in 2019-2020). Forward-filled, capped at
   `MAX_CHAIN_GAP_ROWS=20` so a real fetch problem would still fail loudly.

### Important: the "fake data" incident
Early in Phase 1, `fetch_data.py` succeeded with zero errors but returned **data that
was not real** (BTC-ETH correlation ~0.01 instead of ~0.8, BTC price on its actual
all-time-high date off by 5x). Caught by cross-referencing known historical price
landmarks, not by any automated check. A second fetch returned genuine data,
cross-verified against Statmuse. **Lesson for whoever continues this: fetch success
(HTTP 200, no exceptions) does not guarantee real data - sanity-check against known
values when in doubt.** The current `data/raw` is confirmed real (fetched 2026-08-12,
frozen 2026-10-07 - see `DATA_FREEZE.md`).

### The synthetic-data guard (3 layers, all required together)
1. `run_pipeline.py --synthetic` writes `data/raw/SYNTHETIC_DATA_DO_NOT_TRUST.txt`
2. `build_dataset.py` prints a loud banner whenever that marker exists
3. A bare `run_pipeline.py` run refuses to proceed while the marker exists

### EDA results (from `eda.ipynb`, on the corrected real data)
- ADF: BTC price level non-stationary (p=0.75), log-return stationary (p≈0) -
  justifies the log-return target
- Jump base rate: 2.48% overall (train 2.67%, val 2.67%, test 1.15%)
- Persistence R²: **volatility target = 0.917** (this is open finding 1, see below),
  alpha/return target = -1.055 (expected for a near-unpredictable series, not a bug -
  see conversation for the math explanation)
- Correlation sanity checks (only visible once data was corrected): BTC-ETH return
  correlation 0.82, SP500-VIX -0.74 (textbook "fear" relationship)

### The four open findings from `PHASE1_SPEC.md`
1. **Volatility target overlap** - `beta_target_vol` structurally overlaps 13/14 days
   with its own input feature `btc_realised_vol_14d`, causing artificially high
   persistence R² (0.917). **DECISION MADE (2026-09-21): going with option 2** -
   report skill scores against both the persistence baseline AND a GARCH(1,1) fit
   (already built, see Phase 3 below), rather than option 3 (redefining the target,
   which would ripple into the Dynamic Risk Engine's design in Phase 4). **This
   decision has NOT yet been formally confirmed with Van Alten** - the student's
   stated plan is to raise it next time they meet in person, proceeding under option 2
   in the meantime, willing to change if he prefers option 3. Whoever continues this
   project should check whether that conversation has happened and whether the
   decision still stands.

   **UPDATE (2026-09-21), and this raises the urgency of that conversation:** running
   `skill_scores.py` (option 2's actual implementation) produced a stark result. Naive
   persistence RMSE on the volatility target = **0.00229**; Beta Agent RMSE = **0.01980
   ± 0.00271** - the trained model is roughly **9x worse than doing nothing**, on all 5
   seeds, confirmed by a proper DM test (p≈0.0000 every seed, since this IS a same-target
   comparison - see below). This is not a bug; both numbers were independently checked.
   The mechanism: because 13/14 days of the rolling window are shared between
   consecutive rows, the target barely moves day to day, so "predict yesterday's value"
   is nearly perfect *by construction of the window*, not because persistence is a good
   model. The Beta Agent, trained to regress the target's absolute level, doesn't even
   match that trivial carry-forward. **Practical implication: as currently framed
   (predicting the absolute level of a slow-moving rolling statistic), the Beta Agent's
   volatility head has no demonstrated forecasting value over doing nothing.** This is
   a genuinely strong argument for option 3 (redefine the target - e.g. predict the
   day-over-day *change* in volatility, or use a shorter/non-overlapping window) when
   this gets raised with Van Alten, not just a methodological footnote anymore.

   **Option B tested (2026-09-21), `experiment_delta_vol.py` - THROWAWAY, not part of
   the official pipeline, doesn't touch `config.py`/`build_dataset.py`/`models.py`.**
   Retrained a small GRU to predict the day-over-day CHANGE in volatility
   (`delta = vol(t+1) - vol(t)`, with `vol(t)` read from the input window's own last
   timestep and un-scaled back to raw units first - `splits.py` only standardises
   input features, never target columns, so the raw feature had to be reconstructed
   from `scaler_params.csv`'s mean/std before the delta was meaningful; an earlier
   version of this script skipped that step and produced a badly misleading, much
   worse number as a result). Absolute level reconstructed as `vol(t) +
   predicted_delta`, same pattern as the Alpha Agent's price reconstruction in
   `metrics.py`. **Confirmed result:**
   | Framing | Vol RMSE | Skill vs persistence | Significant (seeds) |
   |---|---|---|---|
   | Option A (official, absolute level) | 0.01980 ± 0.00271 | -765.9% ± 118.6% | 5/5 |
   | Option B (delta, reconstructed) | 0.01153 ± 0.00294 | -404.1% ± 128.6% | 5/5 |

   **Verdict: option B is a real, substantial improvement (RMSE down ~42%) but still
   loses to naive persistence on all 5 seeds.** This is expected, not just a weaker
   version of the same failure: real volatility genuinely clusters day-to-day (a
   well-known stylised fact in the literature, independent of this dataset's specific
   window construction), so "yesterday's vol predicts today's reasonably well" is a
   strong baseline on its own merits, not purely an artifact of the 13/14-day overlap.
   Option B removes the artificial part of persistence's advantage but not the
   genuine part.

   **Diagnosing the official (absolute-level) model further**
   (`diagnose_beta_vol.py` - throwaway): checked whether it was secretly doing a bad
   version of persistence, or collapsing toward a bland average. Neither: predicted
   values have essentially the same variance as the real target (`pred std / actual
   std = 1.004`), but that variance is **uncorrelated with the true future value**
   (`corr(pred, actual)` averaged 0.131 across seeds and flipped sign between them,
   e.g. seed 0 = +0.38, seed 3 = -0.08). Practically: it isn't predicting nothing, and
   it isn't copying anything - it's producing volatility-shaped **noise** with
   realistic magnitude but no real relationship to what actually happens next. Worth
   flagging as a real-world-deployment concern beyond the RMSE number itself: a model
   like this could look convincing (plausible-looking variability) while adding no
   genuine information, which is arguably worse than an obviously-flat prediction.

   **GARCH as an input feature tested (2026-09-21), `experiment_garch_feature.py` -
   THROWAWAY.** Separate idea from option B: instead of changing WHAT is predicted,
   this changes what information the model SEES - GARCH(1,1)'s own one-step-ahead
   forecast added as a 20th input feature, official absolute-level target otherwise
   unchanged. Real complication handled here: `garch_baseline.py`'s GARCH fit uses
   the ENTIRE training split, which is leakage-safe for producing val/test forecasts
   but NOT safe to reuse as an input feature on training days themselves (an early
   training day's "forecast" would then be informed by later training data the fit
   saw). Fixed by fitting GARCH on only the first ~2 years of train (2018-2019) and
   rolling those fixed parameters forward from 2020 onward - same walk-forward
   technique `garch_baseline.py` already uses for val/test, just started earlier.
   Cost: this experiment's usable training range shrinks to 2020-2023 (~4 years
   instead of the official ~6), since the first 2 years have no leakage-safe GARCH
   feature. **Confirmed result: RMSE 0.01586 ± 0.00145** - also a real, significant
   improvement over the official 0.01980, though smaller than option B's, and more
   stable across seeds (std dropped from 0.00271 to 0.00145).

   **Do the two fixes stack? Tested and answered (2026-09-21).** A first combined run
   (delta target + GARCH feature together) scored worse (0.01610) than delta-only
   (0.01153) - but that comparison was confounded, since the combined run inherited
   the GARCH feature's 2020-2023 data restriction while delta-only had used the full
   2018-2023 range. A controlled follow-up (`experiment_delta_restricted.py`, run by
   Claude directly at the student's explicit request) held the training range fixed
   at 2020-2023 for both: **delta-only scored 0.01368, delta+GARCH scored 0.01610 -
   the GARCH feature made it measurably worse, not better, even with the data-range
   confound removed.** Likely explanation: the delta framing may already capture most
   of what GARCH's forecast would add (redundant information), and a 20th input
   dimension adds overfitting risk on a fairly small dataset without a matching
   benefit. **Conclusion: the two fixes do not stack - use option B (delta target)
   alone, not combined with the GARCH feature.**

   **Picture before raising this with Van Alten (5 variants, all still predicting
   the 14-day rolling target in some form):**
   | Variant | Train range | Vol RMSE | Skill vs persistence |
   |---|---|---|---|
   | Official (absolute level) | full 2018-2023 | 0.01980 ± 0.00271 | -765.9% ± 118.6% |
   | Delta target only | full 2018-2023 | 0.01153 ± 0.00294 | -404.1% ± 128.6% |
   | Delta target only (restricted, control) | 2020-2023 | 0.01368 ± 0.00265 | -498.5% ± 116.0% |
   | GARCH feature only (absolute level) | 2020-2023 | 0.01586 ± 0.00145 | -593.9% ± 63.6% |
   | Delta target + GARCH feature | 2020-2023 | 0.01610 ± 0.00272 | -604.3% ± 119.0% |

   All five lose to naive persistence (0.00229), all differences significant by DM
   test on all 5 seeds.

   **BREAKTHROUGH (2026-09-22) - Van Alten's suggestion, confirmed.** Emailed him the
   investigation above; he asked the right clarifying question: "are you predicting
   the 14-day average, or just the next day's own volatility?" Confirmed it's the
   14-day average (`beta_target_vol` = `btc_realised_vol_14d.shift(-1)`, i.e. a
   rolling statistic, not a single day's value) - exactly the root cause already
   diagnosed. His suggestion: predict just that day's own volatility instead. Tested
   as `experiment_single_day_vol.py` (throwaway): retarget to next-day `|btc_log_return|`
   (that day's own absolute move - the standard single-day volatility proxy, and
   exactly what `garch_baseline.py` already predicts, which also finally makes a fair,
   same-target Beta-Agent-vs-GARCH comparison possible for the first time). Same
   official 19 features, 30-day lookback, full 2018-2023 train range - no leakage
   complication this time, since the new target is just a different column already in
   `aligned_dataset.csv`, not something that needed fitting.

   **Confirmed result (2026-09-22):**
   | Comparison | Result |
   |---|---|
   | Test RMSE | 0.01973 ± 0.00092 |
   | vs naive persistence (0.02083) | skill +5.3% ± 4.4% (4/5 seeds beat it, 0/5 individually significant) |
   | vs GARCH(1,1), same target (0.02247) | **Beta Agent wins on 5/5 seeds, DM-significant on all 5 (p<0.05, most p<0.0001)** |

   **This is the first genuinely positive result in the whole investigation.** The
   14-day-average framing was the actual root cause, not just a contributing factor -
   removing it (predict a single day's own volatility instead of a smoothed multi-day
   statistic) doesn't just shrink the gap to persistence, it roughly closes it, and
   produces a clean, significant win over GARCH. Note: GARCH's RMSE here (0.02247)
   differs slightly from the descriptive comparison earlier (0.02220) because this
   evaluates only the ~230 test days that have a full 30-day Beta Agent window behind
   them, a handful fewer than GARCH's full test range - not an inconsistency, just a
   different (smaller, matched) evaluation set for a fair comparison.

   **ADOPTED AS OFFICIAL, 2026-10-07.** `build_dataset.py` now defines
   `beta_target_vol = |btc_log_return|.shift(-1)` (the column name is unchanged).
   Leakage check 4 independently verifies it, and a sixth mutation in
   `test_audit_teeth.py` proves the check catches a same-day version of it.
   `skill_scores.py` now runs a real DM test against GARCH, since both predict the
   same target. `CLAUDE.md`, `README.md` and `PHASE1_SPEC.md` mark finding 1
   resolved. Rebuilt pipeline: leakage audit 9/9, mutation test 6/6.

   **Second change needed, same day: the BetaAgent's heads no longer share a GRU.**
   The first official run on the new target used the original shared-GRU model and
   failed badly: test vol RMSE 0.023-0.479 (mean 0.121), losing to persistence and
   GARCH on every seed. Two causes, both confirmed by direct test:
   - Checkpoint selection on validation jump BCE alone kept epoch-0 weights on seed
     0 (vol RMSE 0.479) - the same failure min_epochs was meant to prevent, which it
     can't, since it only delays early stopping, not best-epoch selection.
   - More fundamentally, the heads interfered: as the jump head trained, validation
     vol RMSE got worse (0.0234 -> 0.0264 on seed 0). The same harness with the jump
     term switched off reached 0.0182 on three seeds. A baseline-normalised combined
     loss was tried first; it fixed the epoch-0 case (seed 0: 0.479 -> 0.030) but not
     the interference.
   Fix: `models.BetaAgent` now holds two independent single-layer 32-unit GRUs
   (`vol_net`, `jump_net`), still returning `(volatility, jump_logit)`, and
   `train_beta.py` trains and checkpoints each branch on its own validation loss.
   `jump_loss_weight` is gone. **This deviates from the proposal's "single GRU layer
   with a dual-head output"** - needs one line of justification in the methodology
   chapter (the interference evidence above); Van Alten has said proposal deviations
   are acceptable.

   **Official results (2026-10-07, student's own run):**
   | | Result |
   |---|---|
   | Test vol RMSE | 0.01974 ± 0.00128 (seeds 0-3 ≈ 0.019, seed 4 = 0.0223) |
   | vs persistence (0.02079) | skill +5.0% ± 6.2%; better on 4/5 seeds, none individually significant |
   | vs GARCH(1,1) (0.02246) | skill +12.1% ± 5.7%; **significantly better on 4/5 seeds** (p<0.0001), seed 4 tied |
   | Test jump BCE | 1.5953 ± 0.2635 vs naive 1.6995 - **better on average but only 3/5 seeds** (seeds 2 and 4 worse) |

   Jump head got weaker than under the old shared setup (5/5 seeds, mean 1.3953):
   sharing with the volatility head seems to have been regularising it. Its
   checkpoint is chosen from validation's ~7 jump days, and the test year has fewer
   jumps (1.15% vs 2.67% in val), so it's noisy across seeds. Possible follow-up:
   regularise the jump branch alone (e.g. weight decay). Calibration also got worse
   - see Phase 3's calibration notes below.

   Original recommendation, kept for history: adopt the single-day volatility
   target (Van Alten's suggestion) as the official Beta Agent target, pending his
   confirmation - this supersedes the delta-target framing as the best result found,
   and is the one with a genuine positive claim behind it (beats GARCH significantly,
   roughly matches persistence) rather than "less bad than before."

   **Practical implication for Phase 4 (Dynamic Risk Engine):** this changes the
   earlier conclusion - if the official target is switched to single-day volatility,
   the Beta Agent's volatility output DOES have genuine, demonstrated value (beats
   GARCH significantly) and can reasonably be used in Phase 4, unlike every 14-day-
   average framing tested before it. Still needs Van Alten's formal sign-off before
   treating this as official, since it changes `build_dataset.py`'s target definition
   (same reason the delta-target/GARCH-feature results were held as "recommended,
   not adopted" - this one is stronger evidence, but the process is the same).
2. **Jump base rate** - measured, 2.48% (see EDA above). Used to compute
   `BCEWithLogitsLoss`'s `pos_weight` for the Beta Agent's jump head (computed exactly
   from the actual training window's rate at runtime, not hardcoded - see
   `train_beta.py`'s `compute_pos_weight`).
3. **Effective start date** - measured, 2018-02-12 (not 2018-01-01 as Chapter 3
   currently states). **Chapter 3 of the proposal still needs correcting to reflect
   this** before final submission.
4. **Proposal text contradiction** - Section 5.1/H1 say "closing price," Section 5.2
   and the actual code use log-return (log-return is correct). **Proposal text still
   needs correcting.** Not yet raised with Van Alten as far as this conversation
   shows.

### Freeze protocol: DONE (2026-10-07)
See `DATA_FREEZE.md` at the repo root: fetch date (2026-08-12), package versions,
and SHA-256 checksums of all 23 raw/processed files. Read-only local copy at
`frozen/data_2026-10-07/` (gitignored - same bytes are in git at tag
`data-freeze-2026-10-07`). Every reported number must come from this dataset; a
re-fetch would need a re-freeze and a full re-run of every result.

## Phase 2 status: COMPLETE (see `Code/PHASE2_SUMMARY.pdf` for the full formal writeup)

Four models built in `models.py`, trained via `train.py`, all using the Section 5.5
recipe: Adam (lr=1e-3, betas=(0.9,0.999)), batch 64, early stopping after 10 stagnant
epochs, LR halved after 5 stagnant epochs, gradient clipping at norm 1.0, 5 fixed
seeds (0-4).

| Model | Architecture | Input | Test RMSE (log-return) |
|---|---|---|---|
| `VanillaLSTM` | 1 layer, 64 units | Univariate (BTC log-return) | 0.02337 ± 0.00005 |
| `VanillaGRU` | 1 layer, 74 units (param-matched to LSTM) | Univariate | 0.02348 ± 0.00019 |
| `StandardBiLSTM` | 2 stacked bidirectional layers, 64 units/direction, dropout 0.2 | Multivariate (19 features) | 0.02689 ± 0.00445 |
| `AlphaAgent` | Same Bi-LSTM stack + single-head self-attention (query = last timestep, keys/values = full 60-step sequence) | Multivariate | 0.02372 ± 0.00051 |

Important implementation note: `StandardBiLSTM` uses `h_n` (final per-direction hidden
state), NOT `out[:, -1, :]`, for pooling - the backward direction's output at the last
timestep has barely started, not actually seen the whole sequence backward. Using
`out[:, -1, :]` would be a real bug. See the class docstring in `models.py`.

### H1 (multivariate beats univariate) - clean test
Used `StandardBiLSTM` on 19 features vs. the same architecture on BTC log-return
alone (architecture held fixed, isolates the input variable cleanly - the
Alpha-vs-LSTM comparison is NOT a clean H1 test since architectures differ too).
Multivariate 0.02689 ± 0.00445 vs univariate 0.03355 ± 0.00983. DM test: 4/5 seeds
significant, but **split 2-2 in direction** (not one-sided). **Conclusion to write up:
"directionally supported on average, not a clean/uniform result" - not "H1
confirmed."**

### H2 (attention beats vanilla Bi-LSTM) - clean test
Alpha Agent vs Standard Bi-LSTM (identical multivariate inputs). 2/5 seeds
significant, BOTH favouring Alpha Agent, and both are the seeds where Bi-LSTM itself
suffered training instability (one Bi-LSTM seed got stuck in a bad local optimum,
RMSE 0.0356, confirmed via a per-epoch diagnostic showing pure noise with no
improvement trend - a known risk of deeper RNNs on a small dataset, not a bug).
**Conclusion: attention gives a real, significant advantage specifically in
unstable-training scenarios, and a small non-significant edge otherwise.**

Secondary (non-clean, informational only) comparisons also run: Alpha vs LSTM (2/5
significant, both favouring LSTM), Alpha vs GRU (1/5 significant, favouring GRU) -
both carry the caveat that LSTM/GRU are univariate and much simpler, so these aren't
fair tests of H2 specifically.

### MAPE (Section 5.6.2, comparable to Seabe et al. Table 2.1)
Predictions are log-returns; MAPE needs price, so `metrics.py` reconstructs
`P_hat(t+1) = P(t) * exp(predicted_log_return)` before computing MAPE.

**Full MAPE table, confirmed 2026-09-21** (naive "predict no change" baseline:
**1.7849%**):
| Model | MAPE (5-seed mean ± std) |
|---|---|
| VanillaLSTM | 1.7755% ± 0.0053% |
| VanillaGRU | 1.7829% ± 0.0166% |
| StandardBiLSTM | 2.1037% ± 0.4328% |
| AlphaAgent | 1.8160% ± 0.0421% |

**Important finding, confirmed across all four models now, not just LSTM:** every
model's MAPE sits close to the naive baseline (1.78-1.82%), except StandardBiLSTM
(2.10%) - which is *worse* than doing nothing, driven by the same seed instability
already noted under H2 (one Bi-LSTM seed got stuck in a bad local optimum, RMSE
0.0356 - visible here too as that seed's individually much higher MAPE, 2.96%, pulling
the mean up and the std wide). This confirms the Phase 1 EDA's near-random-walk
finding (-1.055 persistence R²) holds across every architecture tested, not just the
one originally checked. **RMSE + DM test remain the real basis for conclusions, MAPE
is reported only for literature comparability - this table doesn't change any
existing H1/H2 conclusion**, it just closes out the comparability table for the
write-up.

Phase 2's MAPE loose end is now fully closed - `PHASE2_SUMMARY.pdf` still only shows
the LSTM row and would need regenerating to include the full table above if that
matters for the final write-up (its generator script was discarded, same convention
as Phase 3's; a new one would need writing if this is wanted).

## Phase 3 status: COMPLETE (Beta Agent, Weeks 9-11)

**Current state as of 2026-10-07 - read this first; the bullets below are the
history of how it got here and some numbers in them are superseded.**
- Volatility target: tomorrow's single-day |log-return| (finding 1, resolved).
- Model: two independent 32-unit GRU branches, each trained/checkpointed on its own
  objective (see finding 1's "second change" note for why).
- Official numbers: finding 1's "Official results (2026-10-07)" table.
- Calibration on the current model (pooled test predictions, 4 buckets): raw
  predicted probabilities 21%-94% against actual jump frequencies 0%-3.1%. After
  per-seed Platt scaling: 1.3%-56% predicted against 0%-3.1% actual - roughly
  18-33x too high in the upper three buckets (bottom bucket had no jumps at all),
  WORSE than the old model's 4-7x. Likely cause: each jump branch's checkpoint is
  chosen where it fits validation best, so its validation logits look more
  informative than they turn out to be on test, and Platt scaling (fit on
  validation) over-trusts them (A > 1 on every seed). Ranking is still correct and
  cleaner than before - actual frequency rises 0% -> 0.35% -> 1.73% -> 3.11% across
  the buckets. Same Phase 4 implication as before: use jump outputs to rank days by
  risk, not as literal probabilities.
- The scripts in the bullets below (train_beta.py, skill_scores.py,
  garch_baseline.py) are all current; descriptions of the single shared GRU,
  `jump_loss_weight`, and checkpoint selection on jump BCE alone are history.

### What's built and verified (history)
- **`BetaAgent` class** in `models.py`: single GRU (32 units), 30-day lookback
  (`BETA_LOOKBACK`), dual output head - volatility via softplus (always positive),
  jump-risk via a RAW logit (deliberately not sigmoid-applied in `forward()` - train
  with `BCEWithLogitsLoss` directly on the logit for numerical stability; apply
  `torch.sigmoid()` only when an actual probability is needed).
- **`train_beta.py`**: separate training harness from `train.py` (dual loss: MSE for
  volatility + weighted BCE for jump, `jump_loss_weight=0.01` currently, a rough
  starting point not a carefully tuned value - both loss components print separately
  every epoch to make the real relative scale visible).
  - **Real bug found and fixed:** checkpoint selection based on `val_jump_bce` alone
    (chosen because volatility loss collapses almost immediately - itself a live
    demonstration of finding 1's overlap - so vol shouldn't drive checkpoint choice)
    could pick a very early, undertrained epoch if `jump_bce` had a lucky noisy dip
    before patience ran out. One seed kept epoch-0 weights and scored 0.47 test vol
    RMSE instead of ~0.02. **Fixed with `min_epochs=20`** - a warm-up floor before
    early stopping is allowed to trigger at all.
  - Current results (5 seeds, after the fix): **vol RMSE 0.01980 ± 0.00271, jump BCE
    1.3953 ± 0.1472**.
  - Naive jump baseline added (`naive_jump_bce` function, permanent in
    `run_multi_seed`'s reporting): always predicting the training base rate (2.72%)
    scores BCE **1.6995** - the trained model beats this on **all 5 seeds**, genuine
    evidence the jump head learned something beyond the class prior.
- **`garch_baseline.py`**: GARCH(1,1) fit on training BTC log-returns only, one-step-
  ahead forecasts rolled through val/test using those FIXED parameters (no refitting)
  against the real expanding return history. Compared against next-day's absolute
  return (standard realised-vol proxy), deliberately NOT forced to match the Beta
  Agent's exact 14-day trailing target definition (agreed as acceptable simplification
  for a "sanity check," not a headline comparison). Converged cleanly:
  `alpha[1]+beta[1]=0.908` (healthy, stationary). **Validation RMSE=0.02752, test
  RMSE=0.02220.**
- **Calibration check** (`calibration_check` in `train_beta.py`): pools jump-
  probability predictions across all 5 seeds' test runs (1155 predictions - note: the
  actual pooled count, since Beta test windows = 231/seed, not the raw 260-day split
  count), quantile-bucketed into 4 coarse buckets rather than a fine-grained
  reliability diagram, given the real data constraint in item 3 below.
  **MAJOR FINDING: the model is severely overconfident.** Buckets showed mean
  predicted probabilities of 45%-97%, but actual observed jump frequency in every
  bucket was only 0.35%-2.08% - the highest-confidence bucket (~87% predicted) only
  saw real jumps ~2% of the time. HOWEVER: actual frequency increased monotonically
  with predicted probability across all 4 buckets (0.35% -> 1.04% -> 1.73% -> 2.08%),
  meaning the model **ranks days correctly** (good discrimination) despite being
  **badly calibrated** (wrong absolute probabilities) - these are different
  properties, and only the second one is broken. Root cause: the `pos_weight ~ 36`
  class-imbalance correction in the loss makes missing a jump ~36x costlier than a
  false alarm, giving the model a strong incentive to hedge toward high probabilities
  broadly rather than to output genuinely calibrated ones. Optimizing weighted BCE and
  being calibrated are different objectives that pulled against each other here.
  **This matters directly for Phase 4:** if the Dynamic Risk Engine ever uses these
  probabilities literally (e.g. expected-loss = probability x position size), it would
  be working with numbers ~10-40x too high. If it only uses them to rank/threshold
  days by relative risk, the raw output is fine as-is.
- **Fix built AND CONFIRMED: Platt scaling recalibration** (`fit_platt_scaling`,
  `apply_platt_scaling`, `platt_recalibrate` in `train_beta.py`). Standard 2-parameter
  logistic correction (`calibrated = sigmoid(A*logit + B)`), fit PER SEED on that
  seed's own validation-set logits (never test, to keep test honestly held out), then
  applied to that same seed's test-set logits. Chosen over the more flexible isotonic
  regression alternative because validation has just as few real jump examples as
  test does - Platt scaling's 2-parameter simplicity is more robust to that data
  scarcity than a more flexible non-parametric fit would be.

  **Confirmed results (2026-09-21):** predicted probabilities dropped from the
  45%-97% range to a 2.5%-14% range - much closer to the actual observed frequencies
  (0.35%-2.08% across the same 4 buckets). In ratio terms (predicted/actual, i.e. how
  many times too high):
  | Bucket | Before | After |
  |---|---|---|
  | 1 | 128x | 7.1x |
  | 2 | 62x | 4.1x |
  | 3 | 45x | 4.1x |
  | 4 | 42x | 6.6x |

  **Honest characterization for the writeup:** Platt scaling substantially reduced
  overconfidence (roughly 40-130x too high down to roughly 4-7x too high), but the
  model remains measurably miscalibrated even after correction - this is NOT "fixed,"
  it's "meaningfully improved." Also worth stating explicitly: the "actual frequency"
  numbers in each bucket are themselves estimated from a tiny number of real
  events (~3 real jump-days per seed, ~6-7 unique calendar days total across the test
  set, pooled 5x by re-evaluating with 5 different seeds) - the residual-miscalibration
  ratios above carry real sampling uncertainty and shouldn't be quoted as precise.

### What's still open in Phase 3
1. **Fix built: `skill_scores.py`** (compares Beta Agent against naive persistence -
   proper DM test, same target - and against GARCH - descriptive only, different
   targets, see finding 1 above for why no DM test is run there). Confirmed
   2026-09-21: Beta Agent loses badly to persistence (see finding 1's update above,
   this is the headline result to lead with when raising finding 1 with Van Alten).
   Beta Agent vs GARCH, each against its own target: GARCH RMSE=0.02220, Beta Agent
   RMSE=0.01980 ± 0.00271 - Beta Agent looks slightly better here, but this pairing
   is explicitly NOT a rigorous comparison (different targets) and should not be
   quoted as "Beta Agent beats GARCH" without that caveat attached every time.
2. **Data-scarcity caveat, not yet written up anywhere formal:** the test
   set has only ~231 Beta windows at a ~2.7% jump base rate = a handful of
   actual jump days in the entire test set (single digits). Too few for fine-grained
   per-class metrics beyond what's already been done (precision/recall curves,
   confusion matrices, finer calibration bins). This is NOT a bug or a split-sizing
   mistake - it's inherent to combining a short (1-year) test period with a rare
   target event, and the 1-year test period is already an explicitly disclosed
   limitation in the proposal's own Chapter 7 conclusion. Do not restructure the
   splits to fix this - the tradeoffs aren't obviously better and it would ripple
   through everything already built/audited.
3. **Van Alten conversation on finding 1** - still pending, but now backed by a
   complete, well-controlled investigation (5 variants tested, see finding 1 above)
   rather than a single negative number. Student's stated plan (2026-09-21): raise it
   next time they meet in person, proceeding under option 2 in the meantime, willing
   to change if he prefers option 3 (now with real evidence for what option 3 would
   look like - the delta-target framing). The calibration/Platt scaling finding is
   arguably also worth mentioning, since it's a real limitation of the jump-risk
   output that matters for how Phase 4 should use it.
4. **`PHASE3_SUMMARY.pdf` built (2026-09-21)** - mirrors `PHASE2_SUMMARY.pdf`'s
   format. Covers architecture/training recipe, jump-risk results + calibration/Platt
   scaling, the full finding-1 investigation (all 5 variants), the GARCH baseline,
   and open items for Phase 4. Generator script (`build_phase3_summary.py`) was
   thrown away after running, same pattern as Phase 2's - only the PDF is committed.
5. Phase 3 also needs (per `PHASE1_SPEC.md`'s proposal Chapter 6.1.3): nothing else
   structurally - the three deliverables listed there (trained Beta Agent, calibrated
   forecasts, GARCH verification) are the ones tracked above, all done.

**Phase 3 is now functionally complete** except for the Van Alten conversation on
finding 1 (item 3 above) and the freeze protocol, which depends on that conversation's
outcome.

## Phase 4 status: COMPLETE (Risk Engine + backtest, Weeks 12-13)

Built 2026-10-07, following proposal Section 5.4 / 5.6.3 / 6.1.4.

- **`train.py --model alpha` / `train_beta.py` now save checkpoints** to
  `Code/checkpoints/` (Beta files include each seed's Platt A, B). Re-running both
  reproduced the official numbers exactly (Alpha RMSE 0.02372, Beta vol 0.01974).
- **`risk_engine.py`** - the five Section 5.4 rules as pure functions:
  jump veto (`jump_prob > tau_j` -> flat), direction (`pred_strength < tau_y` -> flat,
  else sign of `pred_return`), size `min(f_max, k / pred_vol)`, stop `c * pred_vol`,
  leverage `L_max * (1 - jump_prob)`. Plus `fixed_action`, the H3 ablation: same
  direction rule, fixed size/stop/leverage, no Beta Agent.
- **`signals.py`** - loads seed s's Alpha + Beta and builds per-day `pred_return`,
  `pred_strength`, `pred_vol`, `jump_prob` (Platt-scaled).
- **`backtest.py`** - leveraged CFD simulation: enter at close t, exit at close of
  the next weekday t' (verified to match how `alpha_target` is built). Stops and
  liquidation are checked against every CALENDAR day's raw BTC Open/High/Low in
  (t, t'] - weekends included, since the model trades weekdays only. Whichever of
  stop / liquidation level is nearer the entry triggers first; gap opens fill at
  the open. Liquidation = a trade loses its whole posted margin (loss capped there -
  negative balance protection); liquidation rate = share of the 5 runs with >= 1.
  Cost 0.1% of notional per round trip. Sharpe/Sortino: rf = 0, annualised x260.
  Simulator checked against 7 hand-worked scenarios (stop, gap, liquidation, short,
  weekend, full Fri->Mon run).
- **`run_backtest.py`** - grid search on 2024 val (one shared setting across seeds,
  objective = mean of Sharpe - max drawdown), then one run on 2025 test, writes
  `Code/results/`, equity chart to `figures/phase4_equity_curves.png`. ~5 s.

**Design decisions taken (all agreed with the student):** Platt-scaled jump
probabilities feed the engine; seed s Alpha paired with seed s Beta = 5 runs;
costs included; objective = Sharpe - MDD.

**One change from the proposal's rule 2, found from results:** `tau_y` is a
percentile level on `pred_strength` (share of the same model's previous 60
predictions that today's |pred_return| exceeds), not a raw return size. With one
shared raw threshold, seed 3 never traded on test and seeds 1 and 4 traded every
day - the five Alpha seeds predict on scales up to ~5x apart, and each drifts
between 2024 and 2025. Causal (past predictions only; window runs back across
split boundaries); checked that changing later predictions doesn't alter earlier
strength values.

**Test results (2025-03-25 to 2025-12-31, 201 weekdays - the Alpha Agent's 60-day
lookback uses up Jan-Mar):**
| | Dynamic Risk Engine | Fixed-rule ablation | Buy & hold (1x) |
|---|---|---|---|
| Total return | 1.045 ± 0.060 | 0.931 ± 0.067 | 1.000 |
| Max drawdown | 4.6% ± 1.1% | 17.2% ± 3.0% | 31.8% |
| Sharpe | 0.736 ± 1.086 | -0.503 ± 0.590 | 0.190 |
| Sortino | 1.427 ± 1.837 | -0.773 ± 0.897 | 0.274 |
| Trades | 62.8 ± 20.9 | 201 | - |
| Liquidation rate | 0% | 0% | - |

Chosen settings - dynamic: tau_j=0.25, tau_y=1/3, k=0.005, c=1.0, L_max=2.0;
fixed: tau_y=0, fraction=0.25, stop=2%, leverage=2.0. The engine beat the
ablation on every seed for drawdown, Sharpe and total return.

**Caveats to carry into Phase 5 and the write-up:**
- Much of the gain is from trading less: ~63 of 201 days vs every day, and the
  equity curves are flat for long stretches (Jun-Aug especially) - likely the jump
  veto firing. Phase 5 should ablate rule by rule (veto / sizing / stops).
- Validation disagreed: on 2024 the fixed rules scored better (objective 0.886 vs
  0.213). Small samples - statistical tests are essential, not a formality.
- Liquidation rate is uninformative: both tuned to 2x leverage, where liquidation
  needs a ~50% move. (2:1 is the EU/UK retail cap for crypto CFDs, so this is
  realistic.) Phase 5 should add a fixed-leverage stress test (e.g. 10x, the
  proposal's own example) for both engines.
- BTC ended the test window almost exactly where it started (87,472 -> 87,509),
  after a large rise and a November fall.

## Phase 5 status: COMPLETE (Evaluation and ablations, Weeks 14-16)

Started early, 2026-10-07. Plan agreed with the student, in order:
1. ~~Rand reporting~~ DONE - `run_backtest.py` reports balances from
   `STARTING_CAPITAL = 1000` (R). Display only; simulation still runs on 1.0.
   Test: Risk Engine R1,000 -> R1,045 ± R60 (lowest R968), fixed rules -> R931 ± R67
   (lowest R884), buy & hold -> R1,000 (lowest R872). Caveats stated in the report:
   ignores USD/ZAR and brokers' minimum trade sizes (a R1,000 account would be
   below some brokers' 0.01 BTC minimum at these settings).
2. ~~H3 significance test~~ DONE - `h3_tests.py`, results in
   `Code/results/h3_tests.md`, figure `figures/phase5_h3_bootstrap.png`.
3. ~~Rule-by-rule ablation~~ DONE - `ablate_rules.py`, results in
   `Code/results/rule_ablation.md`, figure `figures/phase5_rule_ablation.png`.
   `risk_engine.DynamicRiskParams` gained rule switches (defaults = full engine,
   verified to reproduce the official backtest exactly). Each version retuned on
   2024. Drawdown cost of removing each rule (paired bootstrap vs full engine,
   5-seed average): vol-based stop +3.6 pp (p=0.0008; the only one that also hurts
   Sharpe, +0.87, p=0.044), weak-prediction filter +3.3 pp (p=0.0002), vol-based
   sizing +2.2 pp (p<0.0002), jump veto +1.0 pp (p=0.03, interval touches 0),
   jump-scaled leverage +0.3 pp (p=0.002, tiny). All rules off: +12.6 pp.
   **Reading:** no single rule carries the result; removing any one costs <= 3.6 pp
   but all together 12.6 pp, more than the sum (~10.4), so the rules partly back
   each other up - robustness, a finding in itself. Vol-based stops are the
   single most valuable rule (direct credit to the Beta Agent's vol forecast).
   "Sitting out" isn't the whole story: dropping the weak-prediction filter
   doubles trades (63 -> 121) but only adds 3.3 pp. Caveat: retuning shifts other
   settings too (fixed-stop version also loosened the veto and dropped the
   filter, trading 177 days), so each bar is that version at its best, not a pure
   one-switch change.
4. ~~Leverage stress test~~ DONE - `stress_test.py`, results in
   `Code/results/leverage_stress.md`, figure `figures/phase5_leverage_stress.png`.
   Both engines retuned on 2024 at caps 2x/5x/10x/20x, run on 2025:
   | Cap | Risk Engine MDD | Fixed MDD | Risk Engine balance | Fixed balance | Liquidations (RE / fixed) |
   |---|---|---|---|---|---|
   | 2x | 4.6% | 17.2% | R1,045 | R931 | 0 / 0 |
   | 5x | 12.6% | 38.3% | R1,106 | R810 | 0 / 0 |
   | 10x | 24.2% | 42.2% | R1,214 | R920 | 0 / 0 |
   | 20x | 38.2% | 68.8% | R1,500 | R691 | **2 / 0** |
   Drawdown reduction significant at every cap (p < 0.0002; +12.6 / +25.6 / +18.0 /
   +30.6 pp). Sharpe gain significant at 2x and 5x, borderline at 10x and 20x
   (p ~0.04 one-sided, 95% interval just includes 0).
   **Liquidation finding:** zero for both below 20x, because a stop tighter than
   1/leverage always fires first, BTC (24/7) opened >2% from the previous close on
   only 3 of 2,922 days, and the simulator fills stops without slippage (understates
   live risk). At 20x (liquidation distance 5%) the RISK ENGINE liquidated twice:
   its vol-widened stops (rule 4) sat beyond the liquidation level. **So H3's
   liquidation claim is not supported** - untestable below 20x, and reversed at
   20x. Design lesson for the discussion chapter / future work: cap the stop
   distance below 1/leverage (e.g. 80% of it). Deliberately NOT changed now - the
   engine was designed and tuned before seeing these results.
   **Overall H3 reading:** drawdown - strongly supported at every leverage;
   Sharpe - not worse, better on average; liquidation - not supported.
5. ~~Model ablations inside the framework~~ DONE - `model_ablation.py`, results in
   `Code/results/model_ablation.md`, figure `figures/phase5_model_ablation.png`.
   `signals.build(..., direction_model=)` now accepts any `train.BASELINES` key
   (default "alpha", verified unchanged). Bi-LSTM and LSTM checkpoints trained by
   the student; both reproduced Phase 2 exactly (0.02689, 0.02337).
   | Forecaster | MDD | Sharpe | Balance from R1,000 | Trades | Hit rate |
   |---|---|---|---|---|---|
   | Alpha Agent | 4.6% | 0.74 | R1,045 | 63 | 51.4% ± 1.3% |
   | Bi-LSTM, no attention | 3.4% | 0.36 | R1,010 | 25 | 50.7% ± 0.8% |
   | Univariate LSTM | 9.2% | -0.44 | R973 | 79 | 53.6% ± 1.3% |
   Bootstrap, is the Alpha version better? vs Bi-LSTM: no (drawdown -1.2 pp,
   p=0.87; Sharpe +0.38, p=0.35). vs univariate LSTM: better on both but not
   significant (+4.7 pp, p=0.098; Sharpe +1.17, p=0.097).
   **Reading:** attention adds nothing measurable once forecasts go through the
   framework (consistent with Phase 2's H2: only helped on unstable seeds); the
   multivariate model does better than the univariate one but not significantly.
   **Key finding for the write-up:** directional skill is near zero. 2025 test was
   50.2% up days. Several models predict ONE direction nearly every day: Alpha seed
   1 predicts down on 100% of test days, seed 4 up on 100%; Bi-LSTM seeds 1 and 0
   likewise (0% / 98% up); the univariate LSTM leans up 71-96% of days. So the
   framework's results come from the Risk Engine deciding WHEN and HOW MUCH to be
   exposed (veto, strength filter, vol sizing and stops), not from forecasting
   direction. Say this plainly rather than crediting the Alpha Agent's signal.
6. ~~2026 forward test~~ DONE (2026-10-07) - results `Code/results/forward_2026.md`,
   figure `figures/phase5_forward_2026.png`, dataset recorded in `DATA_FREEZE.md`
   (addendum). Fetch: 12/12 series; 11/12 identical to the frozen data over
   2018-2025, gold 0.10% typical difference (futures roll adjustments).
   **Result on 194 unseen trading days (2026-01-01..2026-09-30), nothing retuned:**
   | | Risk Engine | Fixed rules | Buy & hold |
   |---|---|---|---|
   | Balance from R1,000 | R1,024 ± R86 | R986 ± R196 | R942 |
   | Lowest balance | R943 | R835 | R660 |
   | Max drawdown | 8.7% ± 3.8% | 22.8% ± 6.4% | 39.6% |
   | Sharpe | 0.35 | -0.18 | 0.04 |
   Drawdown reduction +14.1 pp [+4.5, +19.2], p < 0.0002, lower on all 5 seeds -
   **the drawdown claim replicates on fully clean data.** Sharpe gain +0.53, not
   significant (p = 0.14) - "comparable", as H3 states. Seed 1: fixed rules made
   more (R1,260 vs R1,049) - lower risk can cost upside. Beta Agent vol RMSE
   0.0204 vs persistence 0.0250 (~18% better, stronger than 2025's ~5%). Alpha
   Agent hit rate 49.3% (47.4% up days) - no directional skill out of sample.
   Zero liquidations (2x). This is the cleanest result in the project: 2025
   influenced design choices, 2026 influenced nothing.
   **Leverage menu (added at the student's request, wanting bigger returns
   honestly):** same Risk Engine at each stress-test cap, settings tuned on 2024 at
   that cap, run once on 2026, ALL caps reported (`forward_test.py`, figure
   `figures/phase5_forward_2026_leverage.png`): 2x +2.4% (worst run -7.7%, avg max
   DD 8.7%); 5x +5.6% (-9.5%, 14.8%); 10x +9.1% (-22.3%, 27.7%); 20x -12.1% (-72.0%,
   55.2%); buy & hold -5.8% (39.6%). Sharpe ~0.35-0.41 until it collapses at 20x -
   leverage widens outcomes, doesn't add edge. The student was told plainly that
   retuning on 2025/2026, changing the objective, or reporting only the best cap
   would be cherry-picking, and agreed to the menu instead. Also discussed: the
   Alpha Agent has no directional edge (49.3% hit rate), so real gains would need
   a better signal, not more leverage - good future-work material, as is a
   long-only "volatility dial" using only the Beta Agent's forecast.
7. **Direction-predictability check** (`direction_baselines.py`, added after the
   student asked whether the Alpha Agent could beat a coin flip): next-day
   direction hit rate on the backtest days of 2024 / 2025 / 2026, 95% Wilson
   intervals, no rule trained or tuned. Pooled over all three (~600 days): Alpha
   Agent 50.2% [46, 54], always-up 49.0%, momentum 1d 47.8%, 5d 47.3%, 20d 52.8%
   [49, 57], mean reversion 52.2% [48, 56]. **Every pooled interval includes 50%.**
   Patterns that looked real in one period reversed in another (mean reversion
   56.7% in 2024 and 2025, 42.8% in 2026). Conclusion for the write-up: BTC's
   next-day direction is not predictable from this kind of data by the Alpha Agent
   or any simple rule - a property of the problem, not a flaw specific to the
   model. Future-work ideas discussed with the student (NOT to be done before
   submission - 2025 and 2026 have both been looked at, so no clean test set is
   left): train the Alpha Agent as an up/down classifier instead of MSE (MSE on
   near-noise targets pushes predictions to ~0, explaining the constant-sign
   seeds), predict a longer horizon (weekly momentum - Liu & Tsyvinski 2021),
   add derivatives / order-book / intraday data, ensemble the seeds.
   Implementation notes: `forward_test.py`
   (`fetch` / `check` / `run`) fetches 2018-01-01..2026-09-30 into
   `Code/data/forward_2026/` (frozen data never written), checks the new fetch's
   2018-2025 values against the frozen raw data and prints BTC 2026 landmarks for
   an eyeball check (Phase 1 fake-data lesson), then runs both engines on
   2026-01-01..2026-09-30 with NOTHING retrained/refitted/retuned (checkpoints,
   frozen train-only scaler, 2024-tuned settings from results/chosen_params.json).
   Verified offline by pointing it at the frozen data with a Jul-Dec 2025 stand-in
   period: rebuilt features identical to the frozen aligned dataset, and all four
   signals identical to signals.build on all 131 shared days. `signals.py` was
   split into load_models / from_windows / build for this (build output verified
   identical before and after). Why it matters: several design choices were made
   after seeing 2025 test results, so 2026 is the only fully clean test.

**H3 result (2026-10-07):**
- Daily-return test per seed (proposal's DM test adapted: loss = negative daily
  return, Newey-West variance, 4 lags): Risk Engine ahead on every seed
  (+0.001% to +0.081% per day) but **not significant on any (0/5)**. Expected - it
  sits out ~2/3 of days, so most daily differences are just the fixed rules'
  noise. Checked: on pure noise the test's false-positive rate is 4.75%.
- Block bootstrap (5,000 resamples, 10-day blocks, paired across engines and seeds):
  | | Drawdown reduction | 95% interval | Sharpe gain | 95% interval |
  |---|---|---|---|---|
  | average of 5 seeds | **+12.6 pp** | [+7.0, +20.9], p < 0.0002 | **+1.24** | [+0.20, +2.45], p = 0.011 |
  | per seed | significant on **5/5** | | significant on 2/5 (seeds 3, 4) | |
- **Reading for the write-up:** H3 predicts lower drawdown and liquidation rate
  with "comparable" Sharpe/Sortino. Drawdown: supported, strongly and on every
  seed. Sharpe: not worse - better on average, significantly so when pooled.
  Liquidation rate: untested until the 10x stress test (step 4). So: "supported
  for drawdown and risk-adjusted return; liquidation pending".
- **Limits to state:** the bootstrap captures uncertainty from which market days
  occurred, not from model training or tuning (the settings are fixed at their
  2024-tuned values). Validation favoured the fixed rules, which the bootstrap on
  test can't speak to. Much of the drawdown gain comes from being out of the
  market more (step 3 will show which rule does it).

## Immediate next steps, in likely order

As of 2026-10-07, Phases 1-4 are complete and the data is frozen (`DATA_FREEZE.md`).
Findings 2-4 need no further supervisor input: Van Alten has said proposal-vs-final
differences are fine, so finding 3 (start date 2018-02-12) and finding 4 ("closing
price" wording) are just corrections to make in the final report's text.

1. **Phase 5's planned steps are all done** (see its status section). Remaining before
   Phase 6 (writing, from ~2026-11-08): the four write-up figures (framework diagram,
   volatility-target investigation, jump calibration, Phase 2 model comparison).
2. Optional follow-up: regularise the Beta Agent's jump branch on its own (e.g.
   weight decay) to see if it gets back to beating the base rate on all seeds.
3. Optional: regenerate `Code/PHASE2_SUMMARY.pdf` with the full MAPE table.
4. Student's own task: tick boxes 3 and 6 on the final thesis's AI Declaration.

## Working conventions worth preserving

- Every chart or diagram made for the project goes in `figures/` (repo root), with
  an entry in `figures/README.md` (what it shows, which script made it, caption
  notes), and gets committed - the student reuses them in the write-up. Scripts
  run from `Code/` write straight to `../figures/`.
- One file, one job (see the whole Phase 1 file layout for the pattern)
- Every non-obvious design choice gets a comment explaining WHY, not just what
- Real bugs found during testing get documented in commit messages, not silently
  fixed and forgotten
- Never fabricate a number - if something isn't measured, say so and stop
- Diebold-Mariano test used throughout for "is this difference real" questions, not
  just eyeballing which mean is smaller
- Naive-baseline checks (MAPE vs "predict no change", jump BCE vs "predict base
  rate") have repeatedly caught metrics that looked good but weren't - keep doing this
  for any new metric introduced
