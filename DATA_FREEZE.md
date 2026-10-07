# Data freeze record

Frozen **2026-10-07**, per the freeze protocol in `Code/PHASE1_SPEC.md`. Every
number in the report must trace back to this dataset.

## What is frozen

- `Code/data/raw/` (12 files) and `Code/data/processed/` (11 files), as of git tag
  `data-freeze-2026-10-07`.
- A read-only local copy is at `frozen/data_2026-10-07/` (not committed to git; the
  same bytes are in the tagged commit). To check either copy against this record:
  `cd frozen/data_2026-10-07 && shasum -a 256 -c ../../DATA_FREEZE.md.sums`, or
  compare by eye against the table below.

## Fetch date

**2026-08-12** - the raw files' modification times, first committed to git on
2026-08-19 (commit 1447a3b). Yahoo Finance revises history, so a refetch will not
reproduce this data. Never re-run `run_pipeline.py --fetch` without a reason that
justifies re-freezing and re-running every result.

## What was built from it

`data/processed/` was rebuilt on 2026-10-07 with a bare `python3 run_pipeline.py`
(no re-fetch) after the volatility target changed to tomorrow's single-day
`|log-return|` (open finding 1). Leakage audit: 9/9 checks passed. Mutation test:
6/6 faults caught. 2,057 aligned rows, 2018-02-12 to 2025-12-30.

## Package versions

Recorded 2026-10-07 from the environment that ran the rebuild. The 2026-08-12
fetch used this same machine; no package changes are known since then, but the
fetch-time versions were not separately recorded.

| Package | Version |
|---|---|
| Python | 3.9.6 |
| yfinance | 1.2.0 |
| requests | 2.32.5 |
| pandas | 2.3.3 |
| numpy | 1.23.5 |
| arch | 7.2.0 |
| torch | 2.8.0 |
| scikit-learn | 1.6.1 |
| scipy | 1.13.1 |

## SHA-256 checksums

```
a80572905fabf68e3e7cd05268d56b591905db848f39b96c4da1615eee66f2e3  processed/aligned_dataset.csv
d7df0e060243c55ac50ca0768e0f9fa81f7e62b3b86b82c580416dcc241c57c5  processed/alpha_test_windows.npz
a7e4b3853ab7d5f0f8b90c35094f436ac044a926096e05607bed919fe5e1f62e  processed/alpha_train_windows.npz
a88390430b769ae1282ba679c7ea3dec91012e62365f4383897f76c6e804f29d  processed/alpha_val_windows.npz
8c3ce634b57869268ffbb5cfd0ef05a8c0863fe41f4c7b20102130f366a40b43  processed/beta_test_windows.npz
5d57dae91dad11c6bd0c076ef0b526112e105d80f63a5225079b5de5406fa269  processed/beta_train_windows.npz
5d788f0c3604c3437230fe99c0dec9e2419987ff40bcda3bcff598405d103e75  processed/beta_val_windows.npz
6a2a3d7cf22530980feedf4e6c1f728c170b47f95d923484e1e1d83e78d137e8  processed/scaler_params.csv
692e33ba693a9c6fd7efd4e1f330f6f3062af2c076729b6f0a89a31ec828b817  processed/test.csv
ebc6f18b278535ef63feeedceda9a8f7fa3a6f72a1b6f25c44c7a4a9be146fa3  processed/train.csv
7b5eb9b37b8647115ec801a31ffa9d77df3ba26df1058c19a763a727b4374fd7  processed/val.csv
59418417f369bbb299fb7ac8efb8bb95e7a5783e342cf02db68c368ae99bf1a6  raw/chain_active_addresses.csv
7a89f695e3c59c4f557aaf6c8b37ce50879834c0d6c407a156be802225ada83c  raw/chain_difficulty.csv
aa6dcf2096592fcdef94cead8818edb23b8a908fd38d24deb52400831f670d8b  raw/chain_hash_rate.csv
6ce36541c74910379caf7d4d817745112e8600b02b807dada237cced7320f90d  raw/chain_mempool_size.csv
206acad7756204757d67df7f49262db16ae53cd0f9c15271edc35febc05a03b4  raw/chain_miner_revenue.csv
89e590c7ba068a89ecbed81f29f601a730e5d373583eadc7030e1f7d0c5c2e28  raw/chain_transaction_count.csv
bd00d2ea52c3ffa546f5e7f0f3c0cbe5cde87474028f96bd3e1b6afff2c87537  raw/yahoo_BTC.csv
3ca231f7c46b1deeff4d1fb612a0dd7b9590900629e7aced1a5556e5c9a6b338  raw/yahoo_DXY.csv
6c5fce9bceec2d06eb782f739420ed0ef4ec9a3b5bbbc577da6eb2189c37ad29  raw/yahoo_ETH.csv
a85a97cd9d78f7883b9d3a0bb1113391517aac22e0f29e7afd3a7bc7b81c2eac  raw/yahoo_GOLD.csv
69d2ddecb33e4af662d63c31d933f168bf23c2a87ebbcb593cbd15046bfbb118  raw/yahoo_SP500.csv
30168be5c3f3d35a5411836f80fa2c383efa5d36ec9b669b05715b983a9c8445  raw/yahoo_VIX.csv
```

---

# Addendum: 2026 forward-test dataset

Fetched **2026-10-07** by `Code/forward_test.py fetch` into `Code/data/forward_2026/`
(series 2018-01-01 to 2026-09-30). Used only for the Phase 5 forward test on
2026-01-01 to 2026-09-30. A separate dataset: the frozen dataset above was not
modified, and every model, scaler and Risk Engine setting used on it comes from
the frozen dataset and 2024 validation.

Agreement with the frozen raw data over 2018-2025 (`forward_test.py check`): 11 of 12
series identical; gold (GC=F futures) differs by 0.10% on a typical day (max 1.96%),
from Yahoo's contract-roll adjustments. 2026 BTC landmarks (close): 2026-01-01 $88,732,
high $96,929 on 2026-01-14, low $58,559 on 2026-06-30, 2026-09-30 $83,554.
Same package versions as above.

```
512891a19aebb023498ac343f8d5c9d507046a560d90c7d728abf68f25de3308  processed/aligned_dataset.csv
cc590d8acb7469de7a5cccad8f2c478aaf4599d9c8572f922cbd169f3d1d6bec  raw/chain_active_addresses.csv
3f75698232c5228e74f1de0a5e278bc9eed27739be7a0e7700de0fde02775877  raw/chain_difficulty.csv
12861e7dbb6fbb172439b20cb391f8093878f4bb4f12ec53c38c73edb1237596  raw/chain_hash_rate.csv
60b793ef347d8c30f5713e9fd87d0f2f58b23fdbe6ac47a08acb870686fac2d5  raw/chain_mempool_size.csv
0a28ea9f44df79d98ef4cea5eb888467f94120c1b681f0909f400251e30d9a17  raw/chain_miner_revenue.csv
11b0c14aea61d77bd91ad60135231de0f85a92aef4208fbb39c80e06d4fe0800  raw/chain_transaction_count.csv
1eaa58b36c012cb8c374752cf230adb991a2eeaa8ee8a69ef54f9a1ffa16e055  raw/yahoo_BTC.csv
6741f1ffcdbe3eea4964d562da995738835e652f439e1d0b7c9bedcab29265f8  raw/yahoo_DXY.csv
e2d10c0bb9f09e3f755d8b496eb9dbfa1a22d6bcfb029394b8de2443c3220d1c  raw/yahoo_ETH.csv
a364b58fcbf8e37ca97c956ac43d7d7b2efa8c06bfb6899c39aa907aef31c9bc  raw/yahoo_GOLD.csv
81fa8d702dba55ea48710d13a06fb038ccd578595334bd7d99f05a967cab3300  raw/yahoo_SP500.csv
1fecee4851a5e62ec474e417a35e812b0aae0bb1579a675b492df900cf6b6aaf  raw/yahoo_VIX.csv
```
