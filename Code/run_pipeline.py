"""
run_pipeline.py

Single entry point for the Phase 1 pipeline. Exactly one of three ways to
get data into data/raw/, followed by the same build -> split -> audit
sequence regardless of which one was used:

    python run_pipeline.py --synthetic   # fake data, no network, ~4s
    python run_pipeline.py --fetch       # real data from Yahoo + Blockchain.com
    python run_pipeline.py               # rebuild from data/raw already on disk

This file is also the third layer of the synthetic-data guard described in
PHASE1_SPEC.md and CLAUDE.md: a bare run refuses to proceed while
data/raw/SYNTHETIC_DATA_DO_NOT_TRUST.txt exists, since building a "real"
dataset from leftover fake fixtures would produce numbers that look
plausible but mean nothing. The other two layers are this file writing
the marker after a --synthetic run, and the loud banner build_dataset.py
prints whenever that marker is present. All three are needed - any one
alone is insufficient.
"""

import argparse
import os
import sys

import build_dataset
import config
import fetch_data
import leakage_audit
import make_synthetic
import splits

MARKER_PATH = os.path.join(build_dataset.RAW_DIR, config.SYNTHETIC_MARKER_FILE)

MARKER_TEXT = """\
This data/raw directory currently holds SYNTHETIC (fake, randomly
generated) data, not real market data. It was written by
`python run_pipeline.py --synthetic` as a test fixture, so the rest of
the pipeline (build_dataset.py, splits.py, leakage_audit.py) could be
exercised end to end without a network call.

Do not use any numbers produced from this data in the dissertation. They
are not real Bitcoin, Ethereum, or on-chain measurements.

To get real data:
  1. Delete this file, or the whole data/ directory.
  2. Run `python run_pipeline.py --fetch` on a connection that can reach
     api.blockchain.info (the university network blocks it - use a
     mobile hotspot).

To run another synthetic dry run instead, just run
`python run_pipeline.py --synthetic` again - it overwrites these
fixtures and this marker together.
"""


def _write_synthetic_marker():
    with open(MARKER_PATH, "w") as f:
        f.write(MARKER_TEXT)


def _refuse_bare_run_on_synthetic_data():
    if os.path.exists(MARKER_PATH):
        raise SystemExit(
            "data/raw holds synthetic fixtures, not real data "
            f"(see {MARKER_PATH}).\n"
            "A bare run refuses to build a dataset from it. Either:\n"
            "  - delete data/ and run --fetch for real data, or\n"
            "  - pass --synthetic to dry-run again on fresh fixtures."
        )


def _run_audit():
    print("-" * 72)
    print("Leakage audit")
    print("-" * 72)
    n_failed, results = leakage_audit.run_all_checks()
    for name, passed, detail in results:
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}")
        if not passed:
            print(f"         {detail}")
    print(f"\n{len(leakage_audit.CHECKS) - n_failed} of {len(leakage_audit.CHECKS)} leakage checks passed.")
    return n_failed == 0


def run(mode):
    if mode == "synthetic":
        make_synthetic.generate(build_dataset.RAW_DIR)
        _write_synthetic_marker()
    elif mode == "fetch":
        ok = fetch_data.fetch(build_dataset.RAW_DIR)
        if not ok:
            raise SystemExit(
                "fetch_data.py did not complete successfully - see the coverage report above."
            )
    else:
        _refuse_bare_run_on_synthetic_data()

    build_dataset.build()
    splits.build()
    print()
    return _run_audit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--synthetic", action="store_true", help="Use fake fixture data, no network."
    )
    group.add_argument(
        "--fetch", action="store_true", help="Fetch real data from Yahoo + Blockchain.com."
    )
    args = parser.parse_args()

    mode = "synthetic" if args.synthetic else ("fetch" if args.fetch else "bare")
    passed = run(mode)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
