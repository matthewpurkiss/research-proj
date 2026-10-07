# Phase 1: Data Acquisition and Preprocessing

Rebuild of the Phase 1 pipeline described in `PHASE1_SPEC.md`, for the honours
research project specified in the proposal (`Matthew_Purkiss_research_proposal.pdf`).
Read `CLAUDE.md` first if you haven't - it has the non-negotiables (leakage,
synthetic-data handling, prose conventions) that everything below assumes.

## Install

Python 3.9. Package versions this pipeline was last verified against:

```
pandas 2.3.3
numpy 1.23.5
requests 2.32.5
yfinance 1.2.0
statsmodels 0.14.6
matplotlib 3.9.4
```

`nbformat`, `nbclient`, and `ipykernel` are only needed if you regenerate
`eda.ipynb` (see below) - not required to run the rest of the pipeline.

## Quick start

```bash
python run_pipeline.py --synthetic   # fake data, no network, ~4s - do this first
python run_pipeline.py --fetch       # real data - only on a connection that can
                                      # reach api.blockchain.info (mobile hotspot,
                                      # not the university network)
python run_pipeline.py               # rebuild from data/raw already on disk
python test_audit_teeth.py           # re-run after touching leakage_audit.py
                                      # or build_dataset.py - don't trust the
                                      # audit until this passes
```

`run_pipeline.py` always ends by running the 9-check leakage audit and exits
non-zero if any check fails.

## File-by-file

| File | Job |
|---|---|
| `config.py` | Every constant - tickers, chart slugs, windows, split dates, pinned feature order |
| `make_synthetic.py` | Fake fixture generator, so everything else can be tested without a network call |
| `fetch_data.py` | Pulls Yahoo + Blockchain.com, one CSV per source, refuses to overwrite without `--force` |
| `build_dataset.py` | Alignment, feature engineering, target construction, warm-up trim, `d==19` assertion |
| `splits.py` | Chronological split, train-only scaler, boundary-safe sequence windowing |
| `leakage_audit.py` | 9 independent checks that nothing above leaked information across time or splits |
| `test_audit_teeth.py` | Mutation test - breaks the pipeline 5 known ways, confirms the audit catches each |
| `eda.ipynb` | ADF tests, jump base rate, moments, redundancy, persistence baselines, regime coverage |
| `run_pipeline.py` | Orchestrates all of the above behind `--synthetic` / `--fetch` / bare |
| `check_connectivity.py` | Pre-flight check before spending mobile hotspot data on `--fetch` |

`eda.ipynb` was built and executed programmatically (via `nbformat`/`nbclient`), not
hand-edited in Jupyter. To regenerate it after a pipeline change, re-run it with any
notebook environment, or ask for it to be rebuilt the same way.

## Findings that affect later phases

### Open findings carried from `PHASE1_SPEC.md`

1. **Volatility target overlap. Resolved.** Confirmed with real numbers in
   `eda.ipynb`: on the original target (tomorrow's 14-day rolling std), naive
   persistence scored an R² of **0.917** by construction. On Van Alten's suggestion
   the target was changed to tomorrow's single-day realised volatility,
   `|log-return(t+1)|`, which shares no days with the input window. `eda.ipynb`'s
   stored outputs predate this change; re-run it to see the new target's
   persistence R². Full investigation in `../HANDOFF.md` and `../PHASE3_SUMMARY.pdf`.
2. **Jump base rate.** Measured: **2.48% overall** (train 2.67%, val 2.67%, test
   1.15%). Suggested BCE `pos_weight` for the Phase 3 jump head, from the training
   split: **~36.4**.
3. **Effective start date.** Measured: after the 30-day warm-up trim, the usable
   dataset starts **2018-02-12**, not 2018-01-01. Chapter 3 needs correcting before
   submission.
4. **Proposal contradiction (closing price vs log-return).** Code uses log-return
   throughout, per Section 5.2's loss function. Section 5.1/H1's "closing price"
   wording still needs to be flagged to Van Alten and fixed in the text.

### Real data-quality issues found and fixed during this rebuild

These weren't in the original `PHASE1_SPEC.md` because the previous build's files
were lost - worth knowing about if anything gets refetched or refactored later.

- **`yfinance`'s `end` parameter is exclusive.** Passing `config.DATA_END` directly
  silently dropped the last calendar day from every Yahoo series. Fixed in
  `fetch_data.py` by requesting one day past `DATA_END`.
- **2018-01-01 is a real market holiday** for the S&P 500, Gold, DXY, and VIX, and
  it's also the first day of the whole dataset - forward-fill has nothing before it
  to reach back to. `build_dataset.py` now forward-fills as usual, then backfills
  only that leading gap, capped at 3 rows so a genuinely larger gap still fails
  loudly instead of being silently patched.
- **SP500 reported `Volume = 0`** on 2023-05-24, which fed straight into
  `log(0) = -inf`. Treated as a data error (real trading volume is never exactly
  zero for an actively-traded index) and imputed the same way as any other gap.
- **Blockchain.com's Charts API silently downsamples long date ranges** unless
  `sampled=false` is passed - an 8-year request without it returned roughly one
  point every 4 days instead of daily. Fixed in `fetch_data.py`.
- **On-chain data has small, persistent gaps**, contrary to Section 3.3's "fully
  observed" assumption: a shared 3-day outage across four metrics (2025-11-13 to
  11-15), plus three isolated single-day gaps for `active_addresses` in 2019-2020
  (one is the 2020-03-13 COVID crash day). Confirmed via a second fetch that these
  are not transient. Forward-filled, capped at `MAX_CHAIN_GAP_ROWS=20` in
  `build_dataset.py` so a real fetch problem would still fail loudly rather than
  being papered over.

### Boundary-safe windowing cost

`splits.py` enforces "no window may straddle a split boundary" by building each
agent's sequence windows independently per split. Cost, measured on the real
dataset: the Alpha Agent (60-day lookback) loses its first 59 days of both
validation and test - about 200 usable windows remain in each split out of ~260
rows. The Beta Agent (30-day lookback) loses proportionally less. This was a
deliberate choice (see conversation history / CLAUDE.md), not a bug - flagged here
so it isn't rediscovered as one.

## Freeze protocol (not yet done)

Once you're satisfied the current `data/processed/` is final:

1. Copy `data/processed/` to a read-only location.
2. Record the fetch date (the current raw data was fetched/corrected on
   2026-08-12) - Yahoo revises history, so a later refetch will not reproduce
   today's numbers exactly.
3. Record the exact package versions above.

Every number in the report needs to trace back to one frozen dataset - do this
before starting Phase 2.
