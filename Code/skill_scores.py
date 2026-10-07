"""
skill_scores.py

How much better is the Beta Agent's volatility forecast than (a) naive
persistence and (b) the GARCH(1,1) baseline? Both comparisons are on the
SAME target - tomorrow's single-day realised volatility, |log-return(t+1)|
(build_dataset.py's beta_target_vol) - on the same test days, so both get
a proper Diebold-Mariano test per seed.

Before the target switch (open finding 1, resolved 2026-09-22) the Beta
Agent predicted a 14-day rolling std while GARCH predicted next-day
|return|, so the GARCH comparison here could only be descriptive. Now that
both forecast the same quantity, it's a real paired test.
"""

import os

import numpy as np
import pandas as pd

import garch_baseline
import train_beta
from evaluate import diebold_mariano

DATA_DIR = os.path.join("data", "processed")


def load_test_context():
    """Each test window's end date, and that day's own |log-return| - the
    naive persistence forecast for tomorrow's single-day volatility.
    Read straight from aligned_dataset.csv by date rather than by shifting
    the target array, because the first test window sits BETA_LOOKBACK-1
    days into the test split: shifting would borrow a value from ~29 days
    earlier for that first day instead of the actual previous day."""
    aligned = pd.read_csv(
        os.path.join(DATA_DIR, "aligned_dataset.csv"), index_col="Date", parse_dates=True
    )
    data = np.load(os.path.join(DATA_DIR, "beta_test_windows.npz"), allow_pickle=True)
    end_dates = pd.to_datetime(data["end_dates"])
    persistence_pred = aligned["btc_log_return"].abs().reindex(end_dates).to_numpy()
    return end_dates, persistence_pred


def skill_score(model_rmse, baseline_rmse):
    return 1 - (model_rmse / baseline_rmse)


def compare_to_baseline(all_results, baseline_name, baseline_pred):
    """Per-seed RMSE, skill score and DM test of the Beta Agent against
    one baseline's per-day forecasts on the identical test days."""
    actual = all_results[0]["test_vol_y"]
    baseline_errors = actual - baseline_pred
    baseline_rmse = float(np.sqrt(np.mean(baseline_errors ** 2)))

    print(f"\nBeta Agent vs {baseline_name} (baseline RMSE={baseline_rmse:.5f})")
    print("-" * 72)

    skills, n_significant_better, n_significant_worse = [], 0, 0
    for i, result in enumerate(all_results):
        assert np.allclose(result["test_vol_y"], actual), "test targets differ between seeds"
        model_errors = actual - result["test_pred_vol"]
        skill = skill_score(result["test_vol_rmse"], baseline_rmse)
        skills.append(skill)

        dm_stat, p_value = diebold_mariano(model_errors, baseline_errors, h=1)
        if p_value < 0.05:
            if result["test_vol_rmse"] < baseline_rmse:
                n_significant_better += 1
                verdict = "significant, Beta Agent better"
            else:
                n_significant_worse += 1
                verdict = f"significant, {baseline_name} better"
        else:
            verdict = "not significant"

        print(
            f"  seed {i}: Beta RMSE={result['test_vol_rmse']:.5f}  skill={skill:+.1%}  "
            f"DM={dm_stat:+.3f}  p={p_value:.4f}  ({verdict})"
        )

    skills = np.array(skills)
    print(f"\nskill score mean +/- std: {skills.mean():+.1%} +/- {skills.std():.1%}")
    print(
        f"significant at alpha=0.05: Beta Agent better on {n_significant_better}, "
        f"{baseline_name} better on {n_significant_worse}, of {len(all_results)} seeds"
    )


if __name__ == "__main__":
    print(f"device: {train_beta.DEVICE}")
    _, _, _, _, all_results = train_beta.run_multi_seed(verbose=False)

    end_dates, persistence_pred = load_test_context()
    assert len(persistence_pred) == len(all_results[0]["test_vol_y"]), (
        "persistence baseline and Beta Agent test windows don't line up"
    )
    compare_to_baseline(all_results, "naive persistence", persistence_pred)

    _, _, garch_test_forecast = garch_baseline.fit_and_forecast()
    garch_pred = garch_test_forecast.reindex(end_dates)
    assert garch_pred.notna().all(), "some test windows have no aligned GARCH forecast"
    compare_to_baseline(all_results, "GARCH(1,1)", garch_pred.to_numpy())
