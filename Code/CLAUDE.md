# CLAUDE.md

Context for Claude Code working in this repository. Read this before touching anything.

## What this is

Honours capstone research project, University of the Witwatersrand, School of
Computer Science and Applied Mathematics. Worth 40% of the final mark.
Supervisor: Clint Van Alten.

Title: *A Decoupled Attention-Based Bi-LSTM Framework with Volatility-Aware Risk
Management for AI-Driven CFD Cryptocurrency Trading.*

The proposal has been submitted and accepted. This repository is the
implementation. The proposal is the contract: if code and proposal disagree, that
is a finding to raise, not a licence to silently pick one.

## Non-negotiables

**This work is defended orally in Week 20.** Every line has to be explainable by
the student. Prefer the clear implementation over the clever one. When a design
choice has a real alternative, leave a comment saying why this one was picked.

**Never fabricate a number.** No placeholder metrics, no illustrative results, no
"approximately" values invented to fill a table. If a number is not yet measured,
write `TODO: measure` and stop. A fabricated result in a research report is
misconduct, not a rough draft.

**Leakage is the primary risk.** This is a time series project with a trading
backtest attached. Any operation that lets information from time `t+k` reach a
model at time `t` invalidates the entire result. Assume every new transform is
guilty until the audit clears it. See `PHASE1_SPEC.md` for the specific
assertions.

**Synthetic data must never be mistaken for real data.** The pipeline stamps a
marker file after a dry run and refuses subsequent plain runs while it exists.
Do not remove that guard.

## Architecture

Three components, deliberately decoupled.

| Component | Model | Predicts | Lookback |
|---|---|---|---|
| Alpha Agent | Attention-augmented stacked Bi-LSTM | Directional signal, log-returns | 60 steps |
| Beta Agent | Dual-output GRU | Volatility and jump risk | 30 steps |
| Dynamic Risk Engine | Deterministic rules, no learning | Sized and vetoed actions | n/a |

The Risk Engine consumes both agents' outputs and applies volatility-conditioned
position sizing and stop-loss placement. It is rules-based on purpose. Do not
turn it into a learned policy.

The decoupling is a design commitment, justified in the methodology by
multi-task learning limitations. A unified single-network comparison was
deliberately deferred to future work. Do not build one and do not reintroduce it
as a hypothesis.

## Hypotheses

- **H1**: Multivariate input outperforms univariate for directional forecasting.
- **H2**: Bi-LSTM outperforms GRU and unidirectional LSTM on identical inputs.
- **H3**: Volatility-conditioned risk management reduces maximum drawdown
  relative to fixed position sizes and fixed stop-losses.

H4 (decoupled versus unified architecture) was removed from the hypotheses. It is
now a methodology paragraph only.

Every hypothesis needs a significance test, not just a favourable mean.
Diebold-Mariano at alpha = 0.05 across multiple seeds is what the project plan
commits to.

## Data

Daily resolution. USD denominated. Nominal start 2018-01-01, though the true
effective start is later (see open findings).

Yahoo Finance via `yfinance`: BTC-USD OHLCV, ETH-USD, S&P 500 (`^GSPC`),
Gold (`GC=F`), DXY (`DX-Y.NYB`), VIX (`^VIX`).

Blockchain.com Charts API, no key required: active addresses
(`n-unique-addresses`), hash rate (`hash-rate`), transaction count
(`n-transactions`), mempool size (`mempool-size`), miner revenue
(`miners-revenue`), difficulty (`difficulty`).

**The university network blocks `api.blockchain.info`.** Real fetches must run on
a mobile hotspot. Because hotspot data is finite, always dry-run first.

Engineered feature vector dimension is **d = 19**, asserted in the build step.
If that assertion fires, something upstream changed and the fix is upstream, not
a relaxed assertion.

Chronological split, never shuffled: train 2018-2023, validation 2024, test 2025.

## Repository conventions

- Python. All constants live in `config.py` and nothing else hard-codes them.
- Scalers fit on training data only, then applied to validation and test.
- Windowing must respect split boundaries. No window may straddle two splits.
- Random seeds recorded for every run. Reproducibility is an explicit success
  criterion.
- Raw fetched CSVs are treated as immutable once written. `fetch_data.py`
  refuses to overwrite without `--force`.

## Working style

The student prefers concise answers and will say when output is too long. He asks
for plain-English explanations of concepts before agreeing to code or LaTeX
changes, and that is the right order: explain, agree, then implement.

Work section by section. Do not deliver ten changed files when one was asked for.

## Prose conventions

For anything that lands in the report, the proposal, or documentation:

- No em-dashes.
- No three-item parallel lists used for rhetorical effect.
- No "rather than X, Y" constructions.
- Plain `\cite{}` style, hand-written `thebibliography` block, `\chapter`
  sectioning, Wits `csam-template` report class.
- Citation keys are PascalCase with full years, for example `Seabe2023`,
  `ChenHuang2021`. Note that `Azizi2023` is authored by Pourmohammadazizi et al.

## Open findings carried from Phase 1

These four are unresolved and must not be forgotten. Full detail in
`PHASE1_SPEC.md`.

1. **Volatility target overlap.** The Beta Agent's volatility target
   structurally overlaps its own input feature, so naive persistence scores a
   high R-squared by construction rather than by learning. Beta Agent results are
   meaningless unless reported against that persistence baseline.
2. **Jump base rate unmeasured.** Needed to set BCE class weighting before any
   Phase 3 training run.
3. **Effective start date.** Rolling-window warm-up consumes early rows, so the
   usable dataset begins later than 2018-01-01. Chapter 3 currently misstates
   this.
4. **Proposal contradiction.** Sections 5.1 and H1 say "closing price". Section
   5.2 and the implemented loss function use log-return. Log-return is correct;
   the proposal text needs fixing.

## Phase map

| Phase | Weeks | Content |
|---|---|---|
| 1 | 1-3 | Data acquisition, alignment, feature engineering, splits, leakage audit, EDA |
| 2 | 4-7 | Alpha Agent, H1 and H2 comparisons |
| 3 | 8-10 | Beta Agent, volatility and jump heads |
| 4 | 11-13 | Dynamic Risk Engine, end-to-end backtest |
| 5 | 14-16 | Evaluation, ablations, H3, significance tests |
| 6 | 17-19 | Writing and revision |
| 7 | 20 | Submission and defence |

Phase 1 is being rebuilt from scratch. Start there.
