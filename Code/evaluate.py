"""
evaluate.py

Diebold-Mariano significance testing for Phase 2 (Section 5.6.4): turns
"model A's mean RMSE looks lower than model B's" into a statistically
defensible statement, accounting for the fact that forecast errors on
consecutive days are not independent of each other.

Run once per seed (the same seed trains both models on identical data),
rather than pooling all seeds into one combined test - each seed is an
independently-trained instance of the same architecture, and collapsing
them into a single test would blur "the architectures genuinely differ"
together with "this seed happened to train differently."
"""

import numpy as np
from scipy import stats

from train import BASELINES, SEEDS, train_one_seed


def diebold_mariano(errors_a, errors_b, h=1):
    """errors_a, errors_b: paired per-day forecast errors (actual minus
    predicted) for two models on the SAME test days, same order.

    h: forecast horizon - 1 for our one-day-ahead setup. The classic DM
    formula only corrects for autocorrelation up to lag h-1, which is
    empty when h=1, so no extra lag terms get added here for our case.

    Returns (dm_statistic, p_value). Applies the Harvey-Leybourne-
    Newbold small-sample correction and assesses significance against a
    Student's t distribution with T-1 degrees of freedom rather than
    the standard normal - the plain DM statistic can over-reject with a
    test set as small as ours (~201 days), and this correction is the
    standard fix for that.
    """
    loss_a = errors_a ** 2
    loss_b = errors_b ** 2
    d = loss_a - loss_b
    T = len(d)
    d_bar = d.mean()

    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for k in range(1, h):
        gamma_k = np.mean((d[:-k] - d_bar) * (d[k:] - d_bar))
        var_d += 2 * gamma_k
    var_d /= T

    dm_stat = d_bar / np.sqrt(var_d)

    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat = dm_stat * correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=T - 1))
    return dm_stat, p_value


def compare(name_a, name_b, seeds=SEEDS, alpha=0.05):
    """Trains both named models (from train.BASELINES) across the same
    seeds, and runs a paired DM test per seed."""
    label_a, model_fn_a, feat_a = BASELINES[name_a]
    label_b, model_fn_b, feat_b = BASELINES[name_b]

    print(f"{label_a}\n  vs\n{label_b}")
    print("-" * 72)

    n_significant = 0
    for seed in seeds:
        _, rmse_a, pred_a, y_a = train_one_seed(model_fn_a, seed, feat_a)
        _, rmse_b, pred_b, y_b = train_one_seed(model_fn_b, seed, feat_b)

        # Both models predict the same target regardless of which
        # features feed their input, so the test targets must match
        # exactly - if they don't, something upstream is broken.
        assert np.allclose(y_a, y_b), "test targets differ between models - not comparable"

        errors_a = y_a - pred_a
        errors_b = y_b - pred_b
        dm_stat, p_value = diebold_mariano(errors_a, errors_b, h=1)

        significant = p_value < alpha
        n_significant += significant
        verdict = "not significant"
        if significant:
            better = name_a if rmse_a < rmse_b else name_b
            verdict = f"significant, {better} better"

        print(
            f"  seed {seed}: RMSE {name_a}={rmse_a:.5f}  {name_b}={rmse_b:.5f}  "
            f"DM={dm_stat:+.3f}  p={p_value:.4f}  ({verdict})"
        )

    print(f"\n{n_significant} of {len(seeds)} seeds significant at alpha={alpha}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, choices=list(BASELINES.keys()))
    parser.add_argument("--b", required=True, choices=list(BASELINES.keys()))
    args = parser.parse_args()

    compare(args.a, args.b)
