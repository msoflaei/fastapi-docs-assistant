"""Average key metrics over several answer-evaluation runs.

Usage: python scripts/summarize_runs.py baseline-v2 baseline-v2-r2 baseline-v2-r3
"""
import json
import sys
from pathlib import Path
from statistics import mean

METRICS = ["fully_correct", "point_coverage", "faithful", "false_refusals"]

runs = [json.loads(Path(f"evals/results/answers_{label}.json").read_text()) for label in sys.argv[1:]]
print(f"{len(runs)} runs: {', '.join(sys.argv[1:])}\n")
for metric in METRICS:
    values = [run["summary"]["in_scope"][metric] for run in runs]
    print(f"  {metric:<16} mean {mean(values):.3f}   range {min(values):.3f} - {max(values):.3f}")