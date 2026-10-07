# Is BTC's next-day direction predictable? (Phase 5)

Next-day direction hit rate on the backtest days, with 95% Wilson intervals. No rule is
trained or tuned. Alpha Agent = average over its 5 seeds.

| Method | 2024 (val) | 2025 (test) | 2026 (forward) | All three pooled |
|---|---|---|---|---|
| Alpha Agent (5-seed avg) | 49.9% [43%, 57%] | 51.4% [45%, 58%] | 49.3% [42%, 56%] | 50.2% [46%, 54%] |
| Always up | 49.3% [42%, 56%] | 50.2% [43%, 57%] | 47.4% [41%, 54%] | 49.0% [45%, 53%] |
| Momentum 1d | 43.3% [37%, 50%] | 43.3% [37%, 50%] | 57.2% [50%, 64%] | 47.8% [44%, 52%] |
| Momentum 5d | 46.3% [40%, 53%] | 48.8% [42%, 56%] | 46.9% [40%, 54%] | 47.3% [43%, 51%] |
| Momentum 20d | 50.2% [43%, 57%] | 52.7% [46%, 60%] | 55.7% [49%, 62%] | 52.8% [49%, 57%] |
| Mean reversion | 56.7% [50%, 63%] | 56.7% [50%, 63%] | 42.8% [36%, 50%] | 52.2% [48%, 56%] |
| Share of up days | 49.3% | 50.2% | 47.4% | - |

An interval that includes 50% means that method can't be told apart from a coin flip.

Figure: figures/phase5_direction_baselines.png
