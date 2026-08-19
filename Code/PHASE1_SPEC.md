# Phase 1 Specification: Data Acquisition and Preprocessing (Weeks 1-3)

Rebuild spec. A previous version of this pipeline was built and tested against a
synthetic fixture, but the files were lost. This document captures what that
version did and what it concluded, so the rebuild starts from Phase 1's findings
instead of from zero.

**Status note.** Items marked `[CONFIRM]` are values that must be read back out
of Chapter 3 of the proposal before coding, because they were not fully recovered.
Do not guess them.

## Deliverables

Three, per the project plan:

1. A frozen aligned dataset.
2. A reproducible preprocessing pipeline, runnable end to end from raw fetch.
3. An EDA pass with an explicit leakage audit that exits non-zero on failure.

## File layout

Ten files, each with one job.

| File | Week | Responsibility |
|---|---|---|
| `config.py` | - | Every constant. Tickers, chart slugs, windows, split dates, pinned feature order. Nothing else hard-codes these. |
| `fetch_data.py` | 1 | Pull Yahoo and Blockchain.com. One CSV per source. Coverage report. Refuses overwrite without `--force`. |
| `build_dataset.py` | 2 | Alignment, feature engineering, `d == 19` assertion, warm-up trim. |
| `splits.py` | 3 | Chronological split, train-only scaler, boundary-safe windowing. |
| `leakage_audit.py` | 3 | Nine assertions. Exit 0 means all passed. |
| `eda.py` | 3 | ADF, jump base rate, moments, redundancy, persistence baselines, regime coverage. |
| `run_pipeline.py` | - | Orchestrator. Flags `--fetch`, `--synthetic`, or bare rebuild from disk. |
| `make_synthetic.py` | - | Test fixture generator. Not part of the research pipeline. |
| `test_audit_teeth.py` | - | Mutation test. Injects known faults, confirms the audit catches each. |
| `README.md` | - | Install, run order, findings that affect later phases. |

## Entry points

```bash
python run_pipeline.py --synthetic   # dry run, no network, ~4 seconds
python run_pipeline.py --fetch       # pull real data then build everything
python run_pipeline.py               # rebuild from data/raw already on disk
python test_audit_teeth.py           # verify the audit has real sensitivity
```

## The synthetic-data guard

This exists because of a real footgun. After a `--synthetic` dry run, `data/raw`
holds fake CSVs, and a later bare `run_pipeline.py` would happily build a
"real" dataset from them and report meaningless numbers that look plausible.

Required behaviour:

- `--synthetic` writes `data/raw/SYNTHETIC_DATA_DO_NOT_TRUST.txt` after copying
  the fixture files, containing a plain-English explanation of what the directory
  holds.
- `build_dataset.py` prints a loud full-width banner whenever that marker is
  present, on every single run.
- A bare `run_pipeline.py` raises `SystemExit` while the marker exists, telling
  the user to delete `data/` and use `--fetch`, or pass `--synthetic` to dry-run
  again.

Keep all three layers. Any one alone is insufficient.

## The nine leakage assertions

Five of these are independently verified by the mutation test and are known
precisely. The remaining four were part of the original audit but their exact
formulations were not recovered; reconstruct them from the list of candidates
below and from Chapter 3.

**Known and mutation-verified:**

1. **Scaler fit on training rows only.** Fitting on the full sample leaks test
   distribution into training. Assert that scaler statistics computed from the
   training slice alone reproduce the stored scaler parameters exactly.
2. **Volatility window is trailing, never centred.** A centred rolling window at
   time `t` includes returns from after `t`. Assert the window is backward-looking
   by checking a hand-computed trailing value against the stored feature.
3. **Jump threshold is shifted.** The threshold that classifies a jump at time `t`
   must be computed from data strictly before `t`. An unshifted threshold uses
   the very observation it is labelling.
4. **Targets are not contemporaneous.** The Alpha target at row `t` must be the
   forward log-return, not the current one. Assert a deliberate off-by-one is
   caught.
5. **No pre-split shuffling.** Assert the index is monotonically increasing at
   every stage and that split boundaries fall on the configured dates.

**Candidates for the remaining four, to confirm against Chapter 3:**

- No NaN or infinite values survive into any split.
- Warm-up trim actually removed every row whose features depend on unavailable
  history.
- No window straddles a split boundary.
- Feature column order matches the pinned order in `config.py`.
- Row counts across the three splits sum to the trimmed dataset length.

### A note on check 6

In the previous build, check 6 was itself buggy. It compared jump labels against
a threshold recomputed on the already-trimmed dataset, so the first 30 rows were
compared against NaN. **The pipeline was correct and the check was wrong.** When
reconstructing, compute any comparison threshold from the untrimmed series, then
align to the trimmed index.

This is worth internalising: an audit that fails is not proof the pipeline is
broken. Debug the check before touching the pipeline.

## The mutation test

`test_audit_teeth.py` is what makes the audit trustworthy. An audit that passes
everything is indistinguishable from an audit that checks nothing.

It injects five faults, one at a time, and asserts the audit fails on each:

1. Scaler fit on the full sample.
2. Volatility window centred instead of trailing.
3. Jump threshold not shifted.
4. Target made contemporaneous.
5. Rows shuffled before splitting.

Run this after any change to `leakage_audit.py` or `build_dataset.py`. If a
mutation stops being caught, the audit has lost a tooth.

## EDA required sections

At minimum:

- **ADF stationarity tests** on returns and on levels, to justify the log-return
  target.
- **Jump base rate.** Class balance for the Beta Agent's jump head. Record the
  number.
- **Distributional moments** per feature. Skew and kurtosis matter for a
  fat-tailed asset.
- **Redundancy / correlation** across the 19 features.
- **Persistence baselines** for both targets. See below, this one is critical.
- **Regime coverage.** Confirm train, validation, and test each contain both bull
  and bear stretches. A test set that is one directional regime cannot support
  H3.

### Section 11f, persistence baselines

The most important output of Phase 1's EDA. It exists because of open finding 1.

Report the naive-persistence R-squared for the volatility target, where the
prediction at `t+1` is simply the value at `t`. Because the target window
overlaps the input feature window, this baseline scores high **by construction**.

For contrast, the directional target shows near-zero persistence, which is
expected for log-returns and means the Alpha Agent faces no equivalent problem.

## Open findings

### 1. Volatility target overlap

The Beta Agent's volatility target structurally overlaps its own input feature.
A naive persistence predictor achieves a high R-squared without learning
anything. Reporting a raw R-squared for the Beta Agent would therefore overstate
its performance dramatically.

Three options, in ascending order of cost and cleanliness:

1. Report the naive-persistence R-squared alongside every Beta result and judge
   the agent on its improvement over that baseline.
2. Report skill scores relative to both persistence and a GARCH(1,1) fit, which
   the project plan already schedules as a Phase 3 sanity check.
3. Change the target to a non-overlapping forward window, for example realised
   volatility of returns from `t+1` to `t+14`, and amend Section 3.4.2. Cleanest,
   but it changes what the Dynamic Risk Engine consumes, so it needs Van Alten's
   agreement.

**This is a supervisor conversation, not a coding decision.** Do not pick one
unilaterally.

### 2. Jump base rate

Must be measured in Week 3. It sets the BCE loss weighting for the Beta Agent's
jump head in Phase 3. Training on an unweighted BCE against a rare positive class
produces a model that predicts "no jump" always and reports high accuracy.

### 3. Effective start date

Rolling-window warm-up consumes early rows, so the usable dataset begins later
than 2018-01-01. Measure the actual first usable date and correct Chapter 3,
which currently claims 2018. `[CONFIRM]` the longest rolling window in the
feature set, since that determines the trim length.

### 4. Proposal contradiction

Section 5.1 and H1 say "closing price". Section 5.2 and the implemented loss
function use log-return. Log-return is correct and is what the code should do.
The proposal text needs correcting. Flag to Van Alten rather than quietly
diverging from a submitted document.

## Freeze protocol

Once the audit passes on real data:

1. Copy `data/processed` to a read-only location.
2. Record the fetch date. Yahoo revises history, so a refetch in November will
   not reproduce an August dataset.
3. Record the exact package versions used for the fetch.

Every number in the report must trace back to one frozen dataset.

## Run order for the rebuild

1. Reconstruct `config.py` first, from Chapter 3. Everything else depends on it.
2. Write `make_synthetic.py` second, so every later file can be tested without
   network access.
3. Then `fetch_data.py`, `build_dataset.py`, `splits.py`.
4. Then `leakage_audit.py` and immediately `test_audit_teeth.py`. Do not trust the
   audit before the mutation test passes.
5. Then `eda.py`.
6. Then `run_pipeline.py` and the synthetic guard.
7. Dry run. Fix. Dry run again.
8. Only then, on a mobile hotspot, `--fetch`.
