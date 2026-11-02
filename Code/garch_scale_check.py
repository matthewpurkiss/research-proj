"""
garch_scale_check.py

Is the Beta Agent vs GARCH(1,1) comparison fair on scale?

garch_baseline.py's forecast is GARCH's conditional standard deviation
sigma(t+1). The target is |log-return(t+1)|. These are not the same
quantity: if returns are roughly normal, E|r| = sqrt(2/pi) * sigma, about
0.80 * sigma. So the raw GARCH forecast is about 25% too high for this
target, and its RMSE is inflated by a scale mismatch rather than by poor
forecasting.

The Beta Agent doesn't have this problem, because it was trained directly
on |r(t+1)|. This script re-runs skill_scores.py's paired comparison with
two rescaled GARCH baselines, neither of which uses the test year:

  1. GARCH x sqrt(2/pi)      - the theoretical normal-distribution factor,
                               nothing fitted
  2. GARCH x s_val           - s_val fitted by least squares on the 2024
                               validation year only

The Beta Agent's forecasts are also biased (too high in 2025), so for a
fair comparison it then gets the same treatment: each seed's forecasts are
multiplied by a least-squares scale fitted on that seed's 2024 predictions,
and the scaled Beta Agent is compared with the scaled GARCH.

Same test days, same per-seed Diebold-Mariano tests as skill_scores.py.
Like skill_scores.py, it retrains the Beta Agent's 5 seeds (same fixed
seeds, so the same models). Run from Code/.
"""

import numpy as np

import garch_baseline
import skill_scores
import train_beta


def validation_scale():
    """Least-squares factor s minimising sum (s * sigma - |r_next|)^2 on 2024."""
    _, val_returns, _ = garch_baseline.load_raw_returns()
    _, val_forecast, _ = garch_baseline.fit_and_forecast()
    actual_next = val_returns.shift(-1).abs()
    common = val_forecast.index.intersection(actual_next.dropna().index)
    f, a = val_forecast.loc[common], actual_next.loc[common]
    return float((f * a).sum() / (f ** 2).sum())


if __name__ == "__main__":
    print(f"device: {train_beta.DEVICE}")
    _, _, _, _, all_results = train_beta.run_multi_seed(verbose=False)

    end_dates, persistence_pred = skill_scores.load_test_context()
    _, _, garch_test_forecast = garch_baseline.fit_and_forecast()
    garch_pred = garch_test_forecast.reindex(end_dates).to_numpy()
    assert not np.isnan(garch_pred).any(), "some test windows have no aligned GARCH forecast"

    theory = np.sqrt(2 / np.pi)
    s_val = validation_scale()
    print(f"\nscale factors: sqrt(2/pi) = {theory:.3f}, fitted on 2024 validation = {s_val:.3f}")

    skill_scores.compare_to_baseline(all_results, "naive persistence", persistence_pred)
    skill_scores.compare_to_baseline(all_results, "GARCH(1,1) raw sigma", garch_pred)
    skill_scores.compare_to_baseline(all_results, "GARCH(1,1) x sqrt(2/pi)", garch_pred * theory)
    skill_scores.compare_to_baseline(all_results, "GARCH(1,1) x validation scale", garch_pred * s_val)

    # Fair version: give the Beta Agent the same correction. Each seed gets
    # its own least-squares scale fitted on its 2024 validation predictions,
    # applied unchanged to its test predictions.
    scaled_results = []
    for result in all_results:
        v_pred, v_y = result["val_pred_vol"].ravel(), result["val_vol_y"].ravel()
        s_beta = float((v_pred * v_y).sum() / (v_pred ** 2).sum())
        test_pred = result["test_pred_vol"] * s_beta
        rmse = float(np.sqrt(np.mean((result["test_vol_y"] - test_pred) ** 2)))
        print(f"Beta Agent validation scale: {s_beta:.3f} -> scaled test RMSE {rmse:.5f}")
        scaled_results.append({**result, "test_pred_vol": test_pred, "test_vol_rmse": rmse})
    rmses = np.array([r["test_vol_rmse"] for r in scaled_results])
    print(f"scaled Beta Agent test vol RMSE mean +/- std: {rmses.mean():.5f} +/- {rmses.std():.5f}")

    skill_scores.compare_to_baseline(scaled_results, "naive persistence [Beta scaled]", persistence_pred)
    skill_scores.compare_to_baseline(scaled_results, "GARCH(1,1) x validation scale [Beta scaled]", garch_pred * s_val)
