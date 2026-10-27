"""Run the complete analysis pipeline in order.

All steps open the source databases read-only and write aggregate outputs to
results/tables and results/figures (created on demand). Individual scripts can
also be run on their own; this runner just fixes the order and reports timing.

    python run_all.py                 # full pipeline
    python run_all.py --list          # show steps
    python run_all.py --only s10_recalibration.py
    python run_all.py --include-meropenem   # also run the meropenem sensitivity analysis
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("s01_incremental_value.py", "primary nested models, increments, decision curves"),
    ("s02_site_decomposition.py", "prior resistance by specimen site"),
    ("s04_mv_window.py", "ventilation-definition sensitivity"),
    ("s05_phenotype_emulation.py", "emulation of the ventilation proxy in MIMIC-IV"),
    ("s06_xgb_replication.py", "gradient-boosted trees instead of logistic regression"),
    ("s07_repeated_cv.py", "repeated patient-grouped cross-validation"),
    ("s08_exclusions.py", "excluding recent cefepime and polymicrobial cultures"),
    ("s09_mimic_severity.py", "adjustment for illness severity (MIMIC-IV)"),
    ("s10_recalibration.py", "recalibration in the large and yearly discrimination"),
    ("s11_cross_database.py", "cross-database transport without refitting"),
    ("s12_recalibrated_thresholds.py", "threshold performance after recalibration"),
    ("s13_coefficients.py", "full coefficients of the primary model"),
    ("s14_cascade.py", "two-drug cascade simulation"),
    ("s15_cascade_paired.py", "paired comparisons between cascade strategies"),
    ("make_figures.py", "all figures (300 dpi)"),
]

OPTIONAL_STEPS = [
    (["s01_incremental_value.py", "--drug", "meropenem", "--tag", "s03_meropenem"],
     "outcome-drug sensitivity analysis (meropenem)"),
]


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, *cmd], cwd=HERE)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed with exit code {result.returncode}")
    print(f"--- {label}: {elapsed:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list the pipeline steps and exit")
    parser.add_argument("--only", help="run a single script by filename")
    parser.add_argument("--include-meropenem", action="store_true",
                        help="also run the meropenem sensitivity analysis")
    args = parser.parse_args()

    if args.list:
        for script, what in STEPS:
            print(f"{script:34s} {what}")
        for cmd, what in OPTIONAL_STEPS:
            print(f"{cmd[0] + ' ' + ' '.join(cmd[1:]):34s} {what} (optional)")
        return

    if args.only:
        known = {s for s, _ in STEPS}
        if args.only not in known:
            raise SystemExit(f"unknown step: {args.only}\ntry --list")
        run([args.only], args.only)
        return

    started = time.time()
    for script, what in STEPS:
        run([script], f"{script} — {what}")

    if args.include_meropenem:
        for cmd, what in OPTIONAL_STEPS:
            run(cmd, f"{cmd[0]} — {what}")

    print(f"\nPipeline finished in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
