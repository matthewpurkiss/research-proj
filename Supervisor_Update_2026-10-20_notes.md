# Speaker notes - supervisor meeting, 20 October 2026

Companion to `Supervisor_Update_2026-10-20.pdf`. One section per slide.

## Slide 1 - Title

Goal for today: walk through everything since the Phase 1 deck - Phases 2 to 5 - and get your view on a few decisions before I start writing (slide 17). About 30 minutes, then questions.

## Slide 2 - Phases 1–5 are complete, about 1.5 weeks ahead of plan

Week 14 of 20 today. Phase 5 was planned for weeks 14-16 and is already done, so I have extra time for writing. Everything is committed to GitHub; the dataset is frozen with checksums (DATA_FREEZE.md), and every number in this deck traces back to it.

## Slide 3 - Risk management works; direction prediction does not

The one-line summary. The Beta Agent's volatility forecast and the Risk Engine genuinely work - they held losses down on unseen data. The Alpha Agent can't predict direction, and I'll show nothing else can either from this data, so it's a property of the problem. H3 is supported for drawdown and Sharpe, not for liquidation.

## Slide 4 - The framework as built

Same decoupled design as the proposal, with two changes I'll come back to: the Beta Agent now has two separate GRUs instead of one shared one, and rule 2's noise threshold is relative to the model's own recent predictions rather than a fixed number. Everything downstream is deterministic rules, tuned on 2024 validation only.

## Slide 5 - Phase 2: no model beats a naive forecast of BTC returns

Test RMSE on next-day log-return, 5 seeds each. MAPE is there for comparability with Seabe et al., but a 'predict no change' baseline gets 1.785%, so MAPE flatters every model. H1: multivariate beat univariate on average with the same architecture, but significance split 2-2 in direction across seeds, so 'directionally supported', not confirmed. H2: attention only helped significantly on the two seeds where the plain Bi-LSTM trained badly.

## Slide 6 - Phase 3: your single-day volatility target fixed finding 1

Recap of our email exchange. The original target was tomorrow's 14-day rolling volatility, which shares 13 of 14 days with an input feature - so 'tomorrow looks like today' was almost unbeatable. I tried five variants of that target; all lost by 400-770%. Your question - is it the 14-day average or the day's own volatility? - identified the root cause. Switching to tomorrow's |return| fixed it, and also made GARCH directly comparable for the first time.

## Slide 7 - Phase 3: the two heads needed separate networks

The one architecture change I want you to be aware of. With one shared GRU, training the jump head made the volatility forecasts worse (validation vol error rose as jump loss fell), and one seed kept untrained epoch-0 weights. With the jump term switched off, volatility reached 0.018. So each head now has its own GRU and is checkpointed on its own loss. Same outputs, so the Risk Engine is unaffected. Trade-off: the jump head lost the regularising effect and is weaker (3/5 seeds beat the base rate, not 5/5).

## Slide 8 - Phase 4: five fixed rules plus a realistic CFD backtest

The five rules are exactly as in proposal Section 5.4. Settings tuned by grid search on 2024 only, one shared setting across seeds, objective Sharpe minus max drawdown. Simulation choices on the right - the weekend one matters: the models trade weekdays, but BTC moves at weekends, so stops are checked on every calendar day. One change from the proposal: rule 2's noise threshold. A fixed number failed because the five Alpha seeds predict on scales up to 5x apart (one seed never traded), so the threshold is now a percentile of each model's own last 60 predictions.

## Slide 9 - 2025 test: same forecasts, a quarter of the drawdown

Starting from R1,000. Blue = Risk Engine, orange = the H3 ablation (same direction signal, fixed size, stop and leverage, no Beta Agent), black dashed = buy and hold. The Risk Engine beat the fixed rules on every seed for drawdown, Sharpe and return. Caveat: on 2024 validation the fixed rules actually scored better, so significance testing matters. Test window starts 25 March because the Alpha Agent needs 60 days of history.

## Slide 10 - H3: the drawdown cut is significant, Sharpe is not worse

The proposal's DM test compares forecast errors, so I adapted it: per seed, DM on the two engines' daily returns with Newey-West variance - not significant on any seed, expected because the engine sits out about two-thirds of days. Drawdown and Sharpe are one number per run, so I used a paired block bootstrap: 5,000 resamples of 10-day blocks, same days for both engines. Drawdown reduction significant on all 5 seeds; Sharpe gain significant pooled, 2/5 per seed.

## Slide 11 - No single rule carries the result: they back each other up

Each version switches off one rule and is retuned on 2024. Removing any one rule costs at most 3.6 points of drawdown, but removing all of them costs 12.6 - more than the sum, so the rules partly cover for each other. Volatility-based stops matter most and are the only rule whose removal also hurts Sharpe significantly - direct credit to the Beta Agent's volatility forecast. Caveat: retuning shifts other settings too.

## Slide 12 - Liquidation claim not supported; a design lesson instead

Both engines retuned at each leverage cap. Drawdown advantage is significant at every level. But no liquidations for either below 20x: a stop tighter than the liquidation distance always fires first, and BTC trades 24/7 so it almost never gaps (more than 2% on 3 of 2,922 days). At 20x the Risk Engine was liquidated twice: its volatility-widened stops sat past the 5% liquidation line. So H3's liquidation claim isn't supported. Fix for future work: cap stops below 1/leverage. I deliberately didn't change the engine after seeing this.

## Slide 13 - Nothing predicts BTC's next-day direction, not just my model

Untrained rules versus the Alpha Agent on the same days, 95% intervals. Pooled over 2024-2026 every interval includes 50%. Mean reversion looked real at 56.7% in 2024 and 2025, then fell to 42.8% in 2026 - the signature of noise. Inside the full framework, swapping the Alpha Agent for the plain Bi-LSTM or the univariate LSTM made no significant difference to trading results either. So the framework's value comes from risk management, not direction.

## Slide 14 - Unseen 2026: the drawdown protection held

The cleanest result in the project. Some design choices were made after seeing 2025, so 2025 isn't fully clean. January to September 2026 influenced nothing: saved models, frozen train-only scaler, settings chosen on 2024. Fresh fetch, checked against the frozen data (11 of 12 series identical). Drawdown 8.7% vs 22.8%, lower on every seed, p < 0.0002. Sharpe better on average but not significant - which is what H3 claims ('comparable'). Beta Agent's vol forecast was 18% better than persistence, stronger than on 2025.

## Slide 15 - More leverage widens outcomes but adds no edge

Same engine at each leverage cap, every other setting tuned on 2024 at that cap, run once on 2026. All caps shown, none picked afterwards. Return per unit of risk stays at a Sharpe of about 0.35-0.41 until it collapses at 20x. Presented in the report as a risk dial an investor chooses, not as 'the system makes 9%'.

## Slide 16 - Where the implementation differs from the proposal

You said differences from the proposal are fine, but I want you to have seen each one, since they'll be justified in the methodology chapter. The first was your suggestion. The second and third came from evidence in the results. The fourth adapts the proposal's DM test to trading results.

## Slide 17 - Questions for you, and what's next

Main things I need from today: your OK on the deviations, your view on the framing, and where the forward test should go. Then I can write with confidence. Draft to you in week 18 as planned.
