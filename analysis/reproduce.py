#!/usr/bin/env python3
"""Regenerate every hardware result in the paper and check it against the paper.

Runs, in order:
  1. whole_cat_stats.py  -> results/trial_metrics.csv, model/sweep/pitch-level CSVs,
                            results/table2.csv + table2.md, results/REPORT.md
  2. fig4_tilt.py        -> results/figures/fig4_tilt.svg, results/tilt_milestones.csv
  3. fig5_whole_cat.py   -> results/figures/fig5_whole_cat.svg
then compares results/table2.csv and results/tilt_milestones.csv with the values
printed in the paper (expected_values.json), at the paper's printed precision.
Exits non-zero on any mismatch.

Usage:
    python analysis/reproduce.py
    python analysis/reproduce.py --results /tmp/meowmentum_results   # leave results/ alone
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

import fig4_tilt
import fig5_whole_cat
import whole_cat_stats

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = Path(__file__).resolve().parent / "expected_values.json"

# Decimal places the paper prints for each quantity.
DECIMALS = {
    "mean_no_tail": 3, "sd_no_tail": 3, "mean_with_tail": 3, "sd_with_tail": 3,
    "tail_effect": 3, "ci_low": 3, "ci_high": 3,
    "p_hc3": 4, "p_holm": 4, "p_permutation": 4,
    "mean_deg": 1, "sd_pop_deg": 1,
}


def compare(label, actual_row, expected):
    """Mismatch messages for one row: integers exactly, floats at printed precision."""
    problems = []
    for key, want in expected.items():
        if key in ("analysis", "morphology", "depth_m"):
            continue
        got = actual_row[key]
        if want is None:
            ok = got is None or (isinstance(got, float) and math.isnan(got))
            shown = "—"
        elif key in DECIMALS:
            shown = f"{got:.{DECIMALS[key]}f}"
            ok = shown == f"{want:.{DECIMALS[key]}f}"
        else:
            shown = str(int(got))
            ok = int(got) == int(want)
        if not ok:
            problems.append(f"{label}: {key} = {shown}, paper says {want}")
    return problems


def check(results_dir, expected):
    problems = []
    table2 = pd.read_csv(results_dir / "table2.csv").set_index("analysis")
    for row in expected["table2"]:
        if row["analysis"] not in table2.index:
            problems.append(f"Table II: missing row {row['analysis']}")
            continue
        problems += compare(f"Table II, {row['analysis']}", table2.loc[row["analysis"]], row)
    milestones = pd.read_csv(results_dir / "tilt_milestones.csv")
    for row in expected["tilt_milestones"]:
        match = milestones[(milestones["morphology"] == row["morphology"])
                           & (milestones["depth_m"] == row["depth_m"])]
        label = f"tilt at {row['depth_m']:g} m, {row['morphology']}"
        if len(match) != 1:
            problems.append(f"{label}: missing")
            continue
        problems += compare(label, match.iloc[0], row)
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "hardware_trials",
                        help="folder of trial CSVs (default: data/hardware_trials)")
    parser.add_argument("--results", type=Path, default=ROOT / "results",
                        help="output folder; figures go to its figures/ subfolder "
                             "(default: results)")
    args = parser.parse_args(argv)
    figures = args.results / "figures"
    timings = {}

    start = time.perf_counter()
    print("== whole_cat_stats.py ==")
    whole_cat_stats.main(["--input", str(args.data), "--output", str(args.results)])
    timings["whole_cat_stats"] = time.perf_counter() - start

    start = time.perf_counter()
    print("\n== fig4_tilt.py ==")
    fig4_tilt.main(["--data", str(args.data), "--out", str(figures / "fig4_tilt.svg"),
                    "--milestones", str(args.results / "tilt_milestones.csv")])
    timings["fig4_tilt"] = time.perf_counter() - start

    start = time.perf_counter()
    print("\n== fig5_whole_cat.py ==")
    fig5_whole_cat.main(["--metrics", str(args.results / "trial_metrics.csv"),
                         "--out", str(figures / "fig5_whole_cat.svg")])
    timings["fig5_whole_cat"] = time.perf_counter() - start

    problems = check(args.results, json.loads(EXPECTED.read_text()))
    print("\n== check against the paper ==")
    print("  " + ", ".join(f"{name} {seconds:.1f} s" for name, seconds in timings.items())
          + f"; total {sum(timings.values()):.1f} s")
    if problems:
        print(f"  FAILED: {len(problems)} value(s) differ from the paper:")
        for problem in problems:
            print(f"    {problem}")
        return 1
    print("  OK: Table II and the 1 m / 2 m tilt values match the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
