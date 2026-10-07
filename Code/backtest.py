"""
backtest.py

Simulates leveraged CFD trading in BTC driven by a risk-engine decision
function, and computes the survival metrics of proposal Section 5.6.3.

Mechanics, one decision per weekday t (the bars the agents are trained on):
  - Enter at BTC's close on day t, exit at the close of the next weekday t'.
    (Verified: alpha_target(t) == log(close(t') / close(t)) for every row.)
  - The position is exposed to every calendar day in (t, t'], weekends
    included - Friday's position holds through Saturday and Sunday. Stops and
    liquidation are checked against each of those days' raw BTC Open/High/Low
    in date order, not just the weekday bars.
  - Margin posted = fraction * equity; notional = margin * leverage.
  - Exit triggers, whichever level is closer to the entry price (prices move
    continuously, so the nearer level is reached first):
      stop-loss  at stop_distance from entry
      liquidation at 1/leverage from entry - the move that wipes out the margin
    If a day OPENS beyond the trigger level, the fill is at that open (gap).
    Exiting at or beyond the liquidation level counts as a liquidation and
    loses exactly the posted margin - retail CFD negative-balance protection
    caps the loss there.
  - Cost: cost_rate * notional per round trip (spread/commission).
  - Sharpe and Sortino use a zero risk-free rate and annualise by
    PERIODS_PER_YEAR weekday bars.
"""

import os

import numpy as np
import pandas as pd

RAW_BTC_PATH = os.path.join("data", "raw", "yahoo_BTC.csv")
PERIODS_PER_YEAR = 260  # weekday bars: 52 weeks x 5
DEFAULT_COST_RATE = 0.001


def load_btc_ohlc():
    return pd.read_csv(RAW_BTC_PATH, index_col=0, parse_dates=True)[["Open", "High", "Low", "Close"]]


def prepare_market(dates, btc=None):
    """For each decision date t: (entry close, path), where path is an array
    of the calendar days (t, t'] with columns Open, High, Low, Close. Built
    once and reused, since the grid search replays the same dates many times."""
    if btc is None:
        btc = load_btc_ohlc()
    market = {}
    for t in dates:
        t_next = t + pd.offsets.BDay(1)
        path = btc.loc[t + pd.Timedelta(days=1) : t_next, ["Open", "High", "Low", "Close"]]
        market[t] = (btc.at[t, "Close"], path.to_numpy())
    return market


def _simulate_trade(direction, entry, stop_distance, leverage, path):
    """Returns (price_return, liquidated) for one held position.
    price_return is the simple return of the BTC price from entry to exit,
    signed so that a profitable move is positive for this direction.
    path: array of the calendar days (t, t'], columns Open, High, Low, Close."""
    liq_distance = 1.0 / leverage
    has_stop = 0 < stop_distance < liq_distance
    trigger_distance = stop_distance if has_stop else liq_distance

    if direction == 1:
        trigger_px = entry * (1 - trigger_distance)
        for open_, high, low, _ in path:
            if open_ <= trigger_px:
                return _exit(direction, entry, open_, liq_distance)
            if low <= trigger_px:
                return _exit(direction, entry, trigger_px, liq_distance)
    else:
        trigger_px = entry * (1 + trigger_distance)
        for open_, high, low, _ in path:
            if open_ >= trigger_px:
                return _exit(direction, entry, open_, liq_distance)
            if high >= trigger_px:
                return _exit(direction, entry, trigger_px, liq_distance)

    return _exit(direction, entry, path[-1, 3], liq_distance)


def _exit(direction, entry, exit_px, liq_distance):
    price_return = direction * (exit_px / entry - 1)
    liquidated = price_return <= -liq_distance + 1e-12
    return price_return, liquidated


def run(signals, decide, market=None, initial_equity=1.0, cost_rate=DEFAULT_COST_RATE):
    """signals: DataFrame indexed by decision date t, whatever columns
    `decide` reads. decide: row -> risk_engine.Action. market: output of
    prepare_market for these dates (built here if not passed).
    Returns a dict with the equity curve, a per-day trade log, and metrics."""
    if market is None:
        market = prepare_market(signals.index)

    equity = initial_equity
    curve, log = [], []
    for row in signals.itertuples():
        t = row.Index
        action = decide(row)
        t_next = t + pd.offsets.BDay(1)
        liquidated = False
        pnl = 0.0

        if action.direction != 0 and equity > 0:
            margin = min(action.fraction, 1.0) * equity
            notional = margin * action.leverage
            entry, path = market[t]
            price_return, liquidated = _simulate_trade(
                action.direction, entry, action.stop_distance, action.leverage, path
            )
            pnl = -margin if liquidated else notional * price_return
            pnl -= cost_rate * notional
            equity = max(equity + pnl, 0.0)

        curve.append((t_next, equity))
        log.append({
            "date": t, "direction": action.direction, "fraction": action.fraction,
            "leverage": action.leverage, "stop_distance": action.stop_distance,
            "pnl": pnl, "liquidated": liquidated, "equity_after": equity,
        })

    equity_curve = pd.Series(
        [initial_equity] + [e for _, e in curve],
        index=[signals.index[0]] + [d for d, _ in curve],
    )
    trades = pd.DataFrame(log).set_index("date")
    return {"equity": equity_curve, "trades": trades, "metrics": metrics(equity_curve, trades)}


def metrics(equity_curve, trades):
    # Equity can't recover from exactly 0, so 0 -> 0 after a wipe-out gives
    # 0/0 = NaN: count those days as 0% return, not missing.
    daily = equity_curve.pct_change().iloc[1:].fillna(0.0)
    std = daily.std()
    downside = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    running_peak = equity_curve.cummax()
    drawdown = (running_peak - equity_curve) / running_peak

    return {
        "total_return": equity_curve.iloc[-1] / equity_curve.iloc[0],
        "max_drawdown": float(drawdown.max()),
        "sharpe": float(daily.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else 0.0,
        "sortino": float(daily.mean() / downside * np.sqrt(PERIODS_PER_YEAR)) if downside > 0 else 0.0,
        "n_trades": int((trades["direction"] != 0).sum()),
        "n_liquidations": int(trades["liquidated"].sum()),
    }
