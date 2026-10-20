# Speaking notes - project walk-through, 20 October 2026

Companion to `Project_Explained_2026-10-20.pdf`. For each slide: what to say, and answers to likely questions.

## Slide 1 - Title

SAY: This is a walk-through of the whole project so far, from the question I asked to what I found. I'll explain each idea as I go.

## Slide 2 - The whole project in one slide

SAY: I wanted to know whether AI can trade Bitcoin with leverage without blowing up the account. I built two AI models - one guesses which way the price goes, one guesses how risky tomorrow is - plus a rulebook that turns their guesses into trades. The direction guesser turned out to be a coin flip, and so is everything else I tried. But the risk side genuinely works: on data it had never seen, it lost far less than simply holding Bitcoin.

IF CLINT ASKS: Why is that still a good result? Because H3, the main hypothesis, is about cutting losses, not about making money - and that part held up on unseen 2026 data.

## Slide 3 - The setting: leveraged CFD trading, and why it is dangerous

SAY: A CFD lets you bet on Bitcoin's price without owning Bitcoin. Leverage means the broker lets you control more than you put in. With 10x leverage, R1,000 controls R10,000 of Bitcoin. If Bitcoin rises 10% you make R1,000 - you doubled your money. But if it falls 10%, you lose R1,000 - everything - and the broker closes the trade. That's called liquidation. Bitcoin can move 10% in a day, so getting the risk right matters as much as getting the direction right.

IF CLINT ASKS: Why CFDs and not just buying Bitcoin? The proposal is about leveraged retail trading, where risk management matters most. In the EU, retail crypto CFDs are capped at 2x leverage (the UK bans them for retail traders), and 2x is what my tuning chose.

## Slide 4 - Trading words, in plain English

SAY: Quick glossary so the rest makes sense. Long means you profit if the price rises, short if it falls. A stop-loss is an automatic exit to cap a loss. Margin is the money you put down; liquidation is when losses eat all of it. A backtest replays history to see how a strategy would have done.

## Slide 5 - How the results are measured

SAY: These are the numbers I'll keep quoting. Volatility is how big the daily moves are, up or down. Max drawdown is the biggest fall from a high point - the worst pain along the way. Sharpe is return divided by bumpiness: how much you earn per unit of risk. RMSE is the average size of a forecast's mistakes. Significance tells us whether a difference is real or could just be luck.

IF CLINT ASKS: Why Sharpe and not just return? Because you can always get a bigger return with more leverage - Sharpe shows whether you are being paid for the extra risk.

## Slide 6 - The three claims I set out to test

SAY: From my proposal. H1: giving the model lots of market information beats giving it only Bitcoin's own price history. H2: adding an 'attention' layer - which lets the model focus on the most relevant past days - beats the same model without it. H3 is the main one: risk rules driven by a volatility forecast cut losses compared with fixed rules, without making risk-adjusted returns worse.

IF CLINT ASKS: Which hypothesis matters most? H3 - it is the 'volatility-aware risk management' in the title.

## Slide 7 - Seven phases; the first five are done

SAY: The project runs 20 weeks from 19 July. Phases 1 to 5 - all the building and testing - are complete, about a week and a half early. What's left is writing and the defence.

## Slide 8 - What the models see: 19 signals per day, 2018–2025

SAY: Every trading day the models get 19 numbers, called features. They fall into four groups: crypto prices, the wider market, Bitcoin's network activity, and trading volumes. Prices come from Yahoo Finance, network data from Blockchain.com. About 2,057 trading days in total.

IF CLINT ASKS: Why on-chain data? It measures how much the Bitcoin network is actually being used, which some studies link to price moves. Why weekdays only? The stock market data only exists on weekdays.

## Slide 9 - Train on the past, test on the future

SAY: This is the most important rule in the project. The models only learn from 2018 to 2023. Settings are chosen on 2024. The final test is 2025, used once. And on top of the plan, I tested on January to September 2026, data nothing had ever seen. Like a student: learn from past papers, check with a mock exam, then write the real exam once.

IF CLINT ASKS: Why not shuffle the days randomly? Because then the model would learn from the future to predict the past - that's cheating, and it makes results look much better than they really are.

## Slide 10 - Checks that nothing cheats

SAY: I built automatic checks for leakage - any way future information could sneak into the past. Nine checks, all pass. Then a 'mutation test': I deliberately broke the pipeline six ways and confirmed the checks catch every one. Early on, a download returned fake-looking data - Bitcoin's price was 5 times wrong on a known date - which I caught by checking against a price website. Lesson: a successful download isn't necessarily real data.

## Slide 11 - Three parts: two forecasters and a rulebook

SAY: Think of a ship. The Alpha Agent is a navigator guessing which direction the wind will blow. The Beta Agent is a weather forecaster warning how stormy tomorrow will be. The Risk Engine is the captain's rulebook: how far to sail, how much cargo to risk, when to stay in port. Keeping direction and risk in separate models is the 'decoupled' part of the title.

## Slide 12 - The Alpha Agent: guessing tomorrow's direction

SAY: The Alpha Agent reads the last 60 days of the 19 features and predicts tomorrow's Bitcoin return. It's built from an LSTM, a type of network designed for sequences - it keeps a running memory as it reads day by day. Bidirectional means it reads the 60 days forwards and backwards. Attention lets it pick out the most relevant days instead of relying only on its memory. All 60 days are in the past, so reading 'backwards' isn't seeing the future.

IF CLINT ASKS: Isn't bidirectional cheating? No - both directions only read the 60 past days in the window; tomorrow is never in the input.

## Slide 13 - The Beta Agent: forecasting how risky tomorrow is

SAY: The Beta Agent doesn't care about direction. It predicts how big tomorrow's move will be, and the chance of an extreme jump - a move more than 3 standard deviations, roughly 3 times the usual daily wobble. The grey bars are what actually happened each day in 2025, the orange line is its forecast. It can't predict each spike, but it does track calm and wild periods.

IF CLINT ASKS: What's a GRU? A simpler cousin of the LSTM - fewer parts, faster, suits a narrower task.

## Slide 14 - The Risk Engine: five fixed rules

SAY: Every day the rules run in order. One: if jump risk is high, don't trade. Two: go long or short in the Alpha Agent's direction, but skip weak predictions. Three: trade smaller when volatility is high. Four: put the stop-loss further away when volatility is high, so normal wobbles don't trigger it. Five: use less leverage when jump risk rises. The settings for these rules were chosen on 2024 only.

IF CLINT ASKS: Why fixed rules and not another AI? So every decision can be explained and checked - that's a deliberate design choice in the proposal.

## Slide 15 - Predicting tomorrow's return: no model does well

SAY: Phase 2 compared four models at predicting tomorrow's Bitcoin return. Lower error is better. They're all about the same, around 2.3-2.7%, and none beats simply guessing 'no change' by much. The Bi-LSTM's big whisker means it was unstable - one run trained badly. H1 is partly supported: more information helped on average but not consistently. H2 is partly supported: attention only helped when the plain model trained badly.

IF CLINT ASKS: What's the Diebold-Mariano test? The standard test for whether one forecast is really better than another, not just lucky.

## Slide 16 - Can anything call tomorrow's direction? No

SAY: This is a key finding. Over about 600 days from 2024 to 2026, my model called the direction right 50.2% of the time - a coin flip. I then tried simple rules like 'same as yesterday' or 'opposite of yesterday'. None are reliably better than 50%: every bar crosses the red line. One rule looked great for two years then flipped. So this isn't my model failing - Bitcoin's daily direction just isn't predictable from this data.

IF CLINT ASKS: Does that make the Alpha Agent useless? For direction, yes - and that's an honest, defensible finding. It means the system's value comes from the risk side.

## Slide 17 - A trap in the original volatility target, and your fix

SAY: The proposal had the Beta Agent predict tomorrow's 14-day average volatility. But today's 14-day average and tomorrow's share 13 of the same 14 days, so 'tomorrow looks like today' was almost unbeatable. It's like predicting tomorrow's 14-day average temperature: just say today's. My model lost to that guess by about 9 times. I tried five fixes - all failed. Your suggestion, predicting the day's own volatility instead, fixed it.

IF CLINT ASKS: Why did the 14-day target fail so badly? 13 of its 14 days were already known to the model as an input, so persistence wins by construction.

## Slide 18 - The volatility forecast beats the standard model

SAY: After the fix, the Beta Agent beats both the naive 'same as today' guess and GARCH, the classic textbook volatility model, by 12%, significantly on 4 of 5 runs. On unseen 2026 data it was even better: 18% less error than the naive guess. One more change: the two forecasts - volatility and jump risk - fought each other when they shared one network, so each now has its own.

IF CLINT ASKS: What's GARCH? A standard statistical model of volatility from finance, used for decades - the natural benchmark.

## Slide 19 - How the trading test works, step by step

SAY: The backtest replays each trading day. At the close it reads the forecasts, the rules decide the trade, and the position is held to the next close. Stop-losses are checked against every day's real high and low, including weekends, since Bitcoin trades 24/7. Every trade costs 0.1%. I compare three things: the full Risk Engine, a 'fixed rules' version that uses the same direction signal but no Beta Agent, and simply buying and holding Bitcoin.

## Slide 20 - Max drawdown: the worst pain along the way

SAY: Max drawdown is the biggest fall from a high point to a later low. Here's buying and holding Bitcoin in 2026: R1,000 rose to R1,092, then fell to R660 - a 39.6% drawdown, even though it ended at R942. It measures the worst moment an investor would have lived through, which is exactly what risk management is supposed to reduce.

## Slide 21 - Sharpe ratio: return per unit of risk

SAY: Imagine two investors who both make 10% in a year. One had a smooth ride, the other swung wildly up and down. The smooth one is better - they got the same reward for less risk. Sharpe captures that: return divided by how much it bounced around. Below zero means losing money. Around 1 is good, above 2 is very good. Sortino, which also appears in my results, is the same idea but only counts the downward bumps.

## Slide 22 - 2025: what happened to R1,000

SAY: Lines are the average of 5 independent runs; shading shows best to worst. The Risk Engine ended at R1,045, the fixed rules at R931, and buy-and-hold at R1,000 after a big rise and fall. The Risk Engine's line is flat for long stretches - that's it choosing to stay out of the market. Its worst drop was 4.6% against 17.2% for the fixed rules.

IF CLINT ASKS: Why 5 runs? Neural networks start from random settings. Running 5 times with different random starts shows results aren't a fluke of one lucky run.

## Slide 23 - The Risk Engine's worst loss is a fraction of the others'

SAY: This is the heart of H3. In both test years the Risk Engine's worst fall was a fraction of the fixed rules' and of simply holding Bitcoin. In 2026, Bitcoin itself fell almost 40% from its high; the Risk Engine's worst was under 9%.

## Slide 24 - Is the improvement real, or luck?

SAY: To check, I reshuffled the test days 5,000 times in 10-day chunks and re-measured each time - that's called a bootstrap. The bar shows the range the true improvement probably lies in. If the whole bar is above zero, the improvement is real. The drawdown cut is clearly real in both years. The Sharpe gain is real in 2025 but could be luck in 2026 - which matches H3, because H3 only claims Sharpe stays comparable.

IF CLINT ASKS: Why not the proposal's Diebold-Mariano test? That compares forecast errors, not trading results, and drawdown is one number per run. I did adapt it to daily returns too: no significant difference there, because the Risk Engine sits out about two-thirds of days.

## Slide 25 - Which rule does the work?

SAY: I switched off one rule at a time and re-ran the test. Each rule alone only adds a few points of drawdown when removed - the stop-loss rule matters most. But switching all of them off adds 12.6 points. So the rules back each other up: remove one and the others still catch most of the risk. That's a robustness finding.

## Slide 26 - Pushing leverage up: the liquidation surprise

SAY: I re-ran both systems at 2, 5, 10 and 20 times leverage. The Risk Engine always had the smaller drawdown. But H3 also predicted fewer liquidations, and that didn't happen: below 20x neither system was liquidated at all, and at 20x only the Risk Engine was - twice. The reason: on wild days it widens its stop-loss, and at 20x the liquidation line is only 5% away, so the wider stop ended up past it. A design lesson: cap the stop below the liquidation level.

IF CLINT ASKS: Why didn't you fix it? Changing the engine after seeing the test results would be cheating - it goes in the discussion as future work.

## Slide 27 - The cleanest test: 2026, data nothing had seen

SAY: Some of my design choices were made after seeing 2025 results, so 2025 isn't perfectly clean. January to September 2026 influenced nothing: same saved models, same settings. Starting with R1,000: the Risk Engine ended at R1,024, fixed rules R986, buy-and-hold R942. Bitcoin crashed twice - February and June - and the Risk Engine barely moved through both.

IF CLINT ASKS: Was the 2026 data checked? Yes - 11 of 12 series exactly matched the frozen data for the overlapping years, and I checked Bitcoin's 2026 prices against a price website.

## Slide 28 - More leverage: bigger swings, not a better system

SAY: If you want bigger returns you can use more leverage - here's what each level would have done in 2026. 10x averaged +9%, but one run lost 22%. At 20x it averaged a loss, and one run lost 72%. The return per unit of risk stays the same; leverage just widens the spread of outcomes. So I present it as a risk dial an investor chooses, not as 'the system makes 9%'.

## Slide 29 - Scorecard: what was supported

SAY: Bringing it together. H1 and H2 are partly supported. H3 is supported for drawdown, strongly and on unseen data, and for Sharpe, which was never worse. It's not supported for liquidation. The honest bottom line: a system that protects capital well but can't predict direction.

## Slide 30 - What I changed from the proposal, and why

SAY: You said changes from the proposal are fine, but I want you to have seen each one. One: the volatility target - your suggestion. Two: separate networks for the two Beta forecasts, because they interfered. Three: the 'skip weak predictions' threshold is relative to each model's recent predictions, because one fixed number made one run never trade. Four: the significance test adapted to trading results. Five: small text fixes.

## Slide 31 - Limitations I will state in the report

SAY: Every project has limits, and examiners respect stating them. Short test periods with few extreme jumps. No directional edge. The backtest assumes stops fill exactly and ignores overnight financing fees. And 2025 was used to make some design choices, which is why the 2026 test matters.

## Slide 32 - Questions for you, and what's next

SAY: The main things I need from today: your OK on the changes, your view on the framing, and where the 2026 test should go in the report. Then I can write with confidence.
