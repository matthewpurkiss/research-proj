"""
risk_engine.py

The Dynamic Risk Engine (proposal Section 5.4): a deterministic decision
layer, not a learned model, so its behaviour is auditable and adjustable
without retraining. Each day it turns the two agents' forecasts into one
trading action.

Inputs per day:
  pred_return  Alpha Agent's predicted next-day BTC log-return (y_hat)
  pred_strength  how big today's prediction is relative to the same model's
               previous 60 (0 to 1, see signals.py)
  pred_vol     Beta Agent's predicted next-day volatility, |log-return| (sigma_hat)
  jump_prob    Beta Agent's next-day jump probability, Platt-scaled (p_hat_j)

The rules, applied in order (Section 5.4):
  1. Jump-risk veto:    jump_prob > tau_j            -> flat
  2. Direction:         pred_strength < tau_y        -> flat, else sign(pred_return)
  3. Position sizing:   fraction = min(f_max, k / pred_vol)
  4. Stop-loss:         stop distance = c * pred_vol  (fraction of entry price)
  5. Leverage:          leverage = L_max * (1 - jump_prob)

tau_y is a percentile level, not a raw return size: the proposal's "flat if
the magnitude falls below a noise threshold", but with the threshold set
relative to each model's own recent predictions. A fixed number can't work
here - the five Alpha seeds predict on scales up to ~5x apart and drift
between 2024 and 2025 (with one shared cut-off, one seed never traded on
test and two always did).

FixedRiskParams is the H3 ablation (Section 6.1.5): the same direction rule
on the same forecasts, but fixed size, fixed stop and fixed leverage, and no
use of the Beta Agent at all.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Action:
    direction: int        # +1 long, -1 short, 0 flat
    fraction: float       # share of current equity posted as margin
    stop_distance: float  # stop-loss distance as a fraction of the entry price
    leverage: float


FLAT = Action(direction=0, fraction=0.0, stop_distance=0.0, leverage=0.0)


@dataclass(frozen=True)
class DynamicRiskParams:
    tau_j: float   # jump-probability veto threshold
    tau_y: float   # minimum pred_strength (percentile level) worth trading
    k: float       # risk budget: fraction = k / pred_vol
    c: float       # stop-loss multiplier on pred_vol
    l_max: float   # leverage cap
    f_max: float = 1.0  # hard cap on fraction of equity per position
    # Rule switches for the Phase 5 rule-by-rule ablation (ablate_rules.py).
    # Defaults = the full engine; each one swaps a single Beta-driven rule for
    # its plain fixed version.
    use_jump_veto: bool = True             # rule 1 off -> never veto
    fixed_fraction: Optional[float] = None  # rule 3 off -> this fraction every trade
    fixed_stop: Optional[float] = None      # rule 4 off -> this stop distance every trade
    jump_scaled_leverage: bool = True       # rule 5 off -> leverage = l_max always


@dataclass(frozen=True)
class FixedRiskParams:
    tau_y: float
    fraction: float
    stop_distance: float
    leverage: float


def _direction(pred_return, pred_strength, tau_y):
    if pred_strength < tau_y or pred_return == 0:
        return 0
    return 1 if pred_return > 0 else -1


def dynamic_action(pred_return, pred_strength, pred_vol, jump_prob, p):
    if p.use_jump_veto and jump_prob > p.tau_j:
        return FLAT

    direction = _direction(pred_return, pred_strength, p.tau_y)
    if direction == 0:
        return FLAT

    return Action(
        direction=direction,
        fraction=p.fixed_fraction if p.fixed_fraction is not None else min(p.f_max, p.k / pred_vol),
        stop_distance=p.fixed_stop if p.fixed_stop is not None else p.c * pred_vol,
        leverage=p.l_max * (1 - jump_prob) if p.jump_scaled_leverage else p.l_max,
    )


def fixed_action(pred_return, pred_strength, p):
    direction = _direction(pred_return, pred_strength, p.tau_y)
    if direction == 0:
        return FLAT
    return Action(direction, p.fraction, p.stop_distance, p.leverage)
