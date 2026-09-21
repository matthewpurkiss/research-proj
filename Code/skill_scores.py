"""
skill_scores.py

Resolves Phase 3's "what next" item after Platt scaling: putting a number
on how much better the Beta Agent's volatility forecast is than (a) a
naive persistence baseline on the SAME target, and (b) the GARCH(1,1)
sanity-check baseline.

This is NOT a Diebold-Mariano test against GARCH. DM requires both
models' errors to be computed against the same target series on the
same days (see evaluate.py's diebold_mariano). The Beta Agent predicts
14-day trailing realised volatility (config.REALISED_VOL_WINDOW); GARCH
predicts next-day |return| - open finding 1, still unresolved with Van
Alten. Forcing a DM test between them would silently paper over that
mismatch by pretending it's a fair paired comparison when it isn't.

What this script does instead - the "option 2" resolution the user
already decided on for finding 1: report each model's own skill score
(1 - RMSE / baseline_RMSE) against a baseline that shares ITS target,
and place them side by side. The persistence-vs-Beta-Agent comparison
below is a real, valid same-target comparison and gets a proper DM
test. The GARCH row is descriptive only, clearly labelled as such.
"""

import numpy as np

import garch_baseline
import train_beta
from evaluate import diebold_mariano


def naive_persistence_vol(vol_y, prev_last_value):
    """Predicts today's 14-day trailing realised vol as simply
    yesterday's value of the same series - the standard "nothing
    changed" naive baseline used throughout this project (metrics.py's
    MAPE baseline, train_beta.py's naive_jump_bce). prev_last_value
    carries the last value from the split immediately before this one,
    so the very first test-day prediction isn't left undefined."""
    shifted = np.concatenate([[prev_last_value], vol_y[:-1]])
    return shifted


def skill_score(model_rmse, baseline_rmse):
    return 1 - (model_rmse / baseline_rmse)


def beta_vs_persistence(all_results):
    """Same-target comparison: Beta Agent vs lag-1 persistence, on each
    seed's own test predictions. Valid DM test, since both series here
    are forecasts of the identical target on the identical days."""
    _, val_vol, _ = train_beta.load_beta_split("val")
    val_vol = val_vol.numpy()

    print("Beta Agent vs naive persistence (same target: 14-day trailing realised vol)")
    print("-" * 72)

    skills, n_significant = [], 0
    for i, result in enumerate(all_results):
        test_vol_y = result["test_vol_y"]
        test_pred_vol = result["test_pred_vol"]
        persistence_pred = naive_persistence_vol(test_vol_y, val_vol[-1])

        model_errors = test_vol_y - test_pred_vol
        persistence_errors = test_vol_y - persistence_pred
        persistence_rmse = float(np.sqrt(np.mean(persistence_errors ** 2)))

        skill = skill_score(result["test_vol_rmse"], persistence_rmse)
        skills.append(skill)

        dm_stat, p_value = diebold_mariano(model_errors, persistence_errors, h=1)
        significant = p_value < 0.05
        n_significant += significant

        print(
            f"  seed {i}: Beta RMSE={result['test_vol_rmse']:.5f}  "
            f"persistence RMSE={persistence_rmse:.5f}  skill={skill:+.1%}  "
            f"DM={dm_stat:+.3f}  p={p_value:.4f}  "
            f"({'significant' if significant else 'not significant'})"
        )

    skills = np.array(skills)
    print(f"\nskill score mean +/- std: {skills.mean():+.1%} +/- {skills.std():.1%}")
    print(f"{n_significant} of {len(all_results)} seeds significant at alpha=0.05")


def beta_vs_garch(all_results):
    """Descriptive only - different targets (see module docstring), so
    no DM test here, just each model's RMSE against its own target and
    the resulting skill scores placed side by side."""
    _, _, test_forecast = garch_baseline.fit_and_forecast()
    _, _, test_returns = garch_baseline.load_raw_returns()
    garch_rmse, _ = garch_baseline.evaluate(test_forecast, test_returns)

    beta_rmses = np.array([r["test_vol_rmse"] for r in all_results])

    print("\nBeta Agent vs GARCH(1,1) - DIFFERENT TARGETS, descriptive comparison only")
    print("-" * 72)
    print("  Beta Agent target: 14-day trailing realised volatility")
    print("  GARCH target:      next-day |return|")
    print(f"  GARCH(1,1) test RMSE (vs its own target):        {garch_rmse:.5f}")
    print(
        f"  Beta Agent test RMSE mean +/- std (vs its own target): "
        f"{beta_rmses.mean():.5f} +/- {beta_rmses.std():.5f}"
    )
    print(
        "  No single skill score is reported here - the two RMSEs aren't on the "
        "same target, so dividing one by the other would imply a rigor this "
        "comparison doesn't have. Kept here as a sanity-check reference point only."
    )


if __name__ == "__main__":
    print(f"device: {train_beta.DEVICE}")
    _, _, _, _, all_results = train_beta.run_multi_seed(verbose=False)
    beta_vs_persistence(all_results)
    beta_vs_garch(all_results)
