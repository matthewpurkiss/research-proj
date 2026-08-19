"""
config.py

Single source of truth for every constant used by the Phase 1 pipeline.
Nothing else in this pipeline hard-codes a ticker, slug, window, date, or
feature name. See CLAUDE.md and PHASE1_SPEC.md for the rules this file
exists to satisfy.

Values below are taken from Chapter 3 (Data and Preprocessing) of the
research proposal. Where the proposal did not fully pin a value down
(PHASE1_SPEC.md's [CONFIRM] items), that is noted inline.
"""

# --- Date range (Section 3.1) ---
DATA_START = "2018-01-01"
DATA_END = "2025-12-31"

# --- Chronological train / validation / test split (Section 3.6, Table 3.2) ---
TRAIN_START = "2018-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# --- Yahoo Finance tickers (Section 3.1, Table 3.1) ---
YAHOO_TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SP500": "^GSPC",
    "GOLD": "GC=F",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}

# Fields retained per asset (Table 3.1). BTC keeps the full OHLCV tuple
# because High/Low feed the realised-volatility and jump computations.
# Auxiliary assets keep only the fields with established explanatory
# power in the literature.
YAHOO_FIELDS = {
    "BTC": ["Open", "High", "Low", "Close", "Volume"],
    "ETH": ["Close", "Volume"],
    "SP500": ["Close", "Volume"],
    "GOLD": ["Close"],
    "DXY": ["Close"],
    "VIX": ["Close"],
}

# --- Blockchain.com Charts API slugs (Section 3.1, Table 3.1) ---
CHAIN_BASE_URL = "https://api.blockchain.info/charts"
CHAIN_CHARTS = {
    "active_addresses": "n-unique-addresses",
    "hash_rate": "hash-rate",
    "transaction_count": "n-transactions",
    "mempool_size": "mempool-size",
    "miner_revenue": "miners-revenue",
    "difficulty": "difficulty",
}

# --- Feature engineering windows (Section 3.4) ---
REALISED_VOL_WINDOW = 14      # Section 3.4.2, trailing, never centred
CORRELATION_WINDOW = 30       # Section 3.4.3, BTC vs ETH / S&P / Gold
ONCHAIN_ROC_WINDOW = 7        # Section 3.4.4, 7-day rate of change
JUMP_THRESHOLD_WINDOW = 30    # Section 3.4.5, trailing std for jump label
JUMP_THRESHOLD_STD = 3.0      # Section 3.4.5, Chen and Huang (2021) threshold

# Longest rolling window in the feature set, used to size the warm-up trim
# in build_dataset.py (PHASE1_SPEC.md open finding 3, marked [CONFIRM]).
# The 30-day correlation and jump-threshold windows are the longest
# declared in Section 3.4; the volatility window is only 14 days. Revisit
# this if Chapter 3 is revised to add a longer window.
WARMUP_TRIM_DAYS = max(REALISED_VOL_WINDOW, CORRELATION_WINDOW, JUMP_THRESHOLD_WINDOW)

# --- Model input sequence lengths (Section 3.4.6) ---
ALPHA_LOOKBACK = 60   # Alpha Agent input sequence length, in days
BETA_LOOKBACK = 30    # Beta Agent input sequence length, in days

# --- Pinned feature order (Section 3.4.6) ---
# d = 19. build_dataset.py must emit columns in exactly this order;
# leakage_audit.py asserts the pipeline never drifts from it.
FEATURE_ORDER = [
    "btc_log_return",
    "eth_log_return",
    "sp500_log_return",
    "gold_log_return",
    "dxy_log_return",
    "vix_log_return",
    "btc_realised_vol_14d",
    "corr_btc_eth_30d",
    "corr_btc_sp500_30d",
    "corr_btc_gold_30d",
    "active_addresses_roc_7d",
    "hash_rate_roc_7d",
    "transaction_count_roc_7d",
    "mempool_size_roc_7d",
    "miner_revenue_roc_7d",
    "difficulty_roc_7d",
    "btc_volume_log",
    "eth_volume_log",
    "sp500_volume_log",
]
N_FEATURES = len(FEATURE_ORDER)
assert N_FEATURES == 19, (
    f"FEATURE_ORDER has {N_FEATURES} entries; proposal Section 3.4.6 requires d=19"
)

# --- Synthetic-data guard (PHASE1_SPEC.md, "The synthetic-data guard") ---
SYNTHETIC_MARKER_FILE = "SYNTHETIC_DATA_DO_NOT_TRUST.txt"

# --- Currency / resolution (Section 3.1) ---
CURRENCY = "USD"
RESOLUTION = "daily"
