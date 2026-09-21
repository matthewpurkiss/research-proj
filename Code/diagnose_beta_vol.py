"""
diagnose_beta_vol.py

THROWAWAY DIAGNOSTIC - checks WHAT the official Beta Agent (models.BetaAgent,
absolute-level target, trained exactly as train_beta.train_one_seed already
does) is actually doing wrong, rather than just measuring that it's wrong
(already established by skill_scores.py). Doesn't touch config.py/
build_dataset.py/models.py/train_beta.py - reuses train_one_seed unmodified
and only inspects its outputs.

Three competing hypotheses, each with its own diagnostic:
1. Is it tracking the real target at all, even loosely?
   -> correlation(predicted, actual next-day vol). Near 0 = no real signal.
2. Is it secretly doing persistence, just worse (a laggy/noisy copy)?
   -> correlation(predicted, today's own actual vol - what persistence uses).
   High correlation here would mean it learned a persistence-like strategy
   but executes it poorly; low correlation rules that out.
3. Is it collapsing toward a bland "safe average" instead of tracking real
   day-to-day fluctuation?
   -> std(predicted) vs std(actual). A model that mostly outputs something
   close to the training-set mean, regardless of input, would show a much
   smaller predicted std than the real target's std.
"""

import numpy as np

import train_beta
from experiment_delta_vol import VOL_FEATURE_IDX, VOL_MEAN, VOL_STD


def vol_today_raw(X):
    """Same un-scaling as experiment_delta_vol.py: X's last-timestep vol
    feature is in SCALED units (splits.py standardises inputs, never
    targets), so it has to be converted back to raw units before it's
    comparable to anything else here."""
    return X[:, -1, VOL_FEATURE_IDX].numpy() * VOL_STD + VOL_MEAN


def diagnose_one_seed(seed):
    result = train_beta.train_one_seed(seed)
    test_X, _, _ = train_beta.load_beta_split("test")
    _, train_vol_y, _ = train_beta.load_beta_split("train")

    pred = result["test_pred_vol"]
    actual = result["test_vol_y"]
    today = vol_today_raw(test_X)
    train_mean = train_vol_y.numpy().mean()

    return {
        "seed": seed,
        "corr_pred_actual": float(np.corrcoef(pred, actual)[0, 1]),
        "corr_pred_today": float(np.corrcoef(pred, today)[0, 1]),
        "pred_std": float(pred.std()),
        "actual_std": float(actual.std()),
        "pred_mean": float(pred.mean()),
        "train_mean": float(train_mean),
    }


if __name__ == "__main__":
    print(f"device: {train_beta.DEVICE}")
    print("Diagnosing the OFFICIAL Beta Agent's volatility predictions (absolute-level target)")
    print("-" * 92)

    rows = [diagnose_one_seed(seed) for seed in train_beta.SEEDS]

    print(
        f"{'seed':>4} {'corr(pred,actual)':>18} {'corr(pred,today)':>18} "
        f"{'pred std':>10} {'actual std':>11} {'pred mean':>10} {'train mean':>11}"
    )
    for r in rows:
        print(
            f"{r['seed']:>4} {r['corr_pred_actual']:>18.3f} {r['corr_pred_today']:>18.3f} "
            f"{r['pred_std']:>10.5f} {r['actual_std']:>11.5f} {r['pred_mean']:>10.5f} "
            f"{r['train_mean']:>11.5f}"
        )

    avg_corr_actual = np.mean([r["corr_pred_actual"] for r in rows])
    avg_corr_today = np.mean([r["corr_pred_today"] for r in rows])
    avg_pred_std = np.mean([r["pred_std"] for r in rows])
    avg_actual_std = np.mean([r["actual_std"] for r in rows])

    print(f"\naverage corr(pred, actual next-day vol): {avg_corr_actual:.3f}")
    print(f"average corr(pred, today's actual vol):  {avg_corr_today:.3f}")
    print(f"average pred std:                        {avg_pred_std:.5f}")
    print(f"average actual std:                       {avg_actual_std:.5f}")
    print(
        f"pred std / actual std ratio:              {avg_pred_std / avg_actual_std:.3f}  "
        "(near 0 = collapsed toward a bland average, near 1 = tracking real fluctuation)"
    )
