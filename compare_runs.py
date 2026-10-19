#!/usr/bin/env python3
"""
Compare VBench evaluation results across multiple runs.

Each run can be specified as:
  - a run directory containing evaluation_results/final_scores.json
    (e.g. ./sampled_videos/my_model/20260923_153045)
  - a directory containing final_scores.json directly
  - a path to a final_scores.json file

Alternatively, use --base to auto-discover all timestamped runs under a
base output directory.

Usage:
    python compare_runs.py run_dir1 run_dir2 [run_dir3 ...]
    python compare_runs.py --base ./sampled_videos/my_model
"""

import argparse
import json
import os
import sys

ALL_DIMENSIONS = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality",
    "imaging_quality", "object_class", "multiple_objects",
    "human_action", "color", "spatial_relationship",
    "scene", "temporal_style", "appearance_style", "overall_consistency",
]


def find_final_scores(path):
    """Resolve a path to a final_scores.json file (or None)."""
    if os.path.isfile(path):
        return path if path.endswith(".json") else None
    cand = os.path.join(path, "evaluation_results", "final_scores.json")
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(path, "final_scores.json")
    if os.path.isfile(cand):
        return cand
    return None


def run_label(json_path):
    """Human label for a run: the timestamped run dir name when possible."""
    parent = os.path.dirname(os.path.abspath(json_path))
    if os.path.basename(parent) == "evaluation_results":
        return os.path.basename(os.path.dirname(parent))
    return os.path.basename(parent) or json_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare VBench final scores across runs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "runs", nargs="*",
        help="Run dirs or final_scores.json paths (2 or more)",
    )
    parser.add_argument(
        "--base", type=str,
        help="Base dir containing timestamped run subdirs to compare",
    )
    args = parser.parse_args()

    json_paths = []
    if args.base:
        if not os.path.isdir(args.base):
            print(f"ERROR: base dir not found: {args.base}")
            sys.exit(1)
        for name in sorted(os.listdir(args.base)):
            full = os.path.join(args.base, name)
            if os.path.isdir(full):
                jp = find_final_scores(full)
                if jp:
                    json_paths.append(jp)
        if not json_paths:
            print(f"ERROR: no runs with final_scores.json found under {args.base}")
            sys.exit(1)
    else:
        if not args.runs:
            print("ERROR: specify run paths or --base")
            sys.exit(1)
        for r in args.runs:
            jp = find_final_scores(r)
            if jp is None:
                print(f"ERROR: no final_scores.json found for: {r}")
                sys.exit(1)
            json_paths.append(jp)

    runs = []
    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            runs.append((run_label(jp), data))
        except Exception as e:
            print(f"ERROR: could not read {jp}: {e}")
            sys.exit(1)

    # Collect dimensions present in any run (canonical order first)
    dims = [d for d in ALL_DIMENSIONS
            if any(d in s.get("dimension_scores", {}) for _, s in runs)]
    extras = set()
    for _, s in runs:
        extras.update(s.get("dimension_scores", {}).keys())
    dims += sorted(extras - set(ALL_DIMENSIONS))

    labels = [l for l, _ in runs]
    col_w = max(10, max(len(l) for l in labels) + 3)
    row_w = 30 + col_w * len(runs)

    print("=" * row_w)
    print("VBench Score Comparison")
    print("=" * row_w)

    header = f"  {'Dimension':<28s}"
    for l in labels:
        header += f" {l:>{col_w - 1}s}"
    print(header)
    print("-" * row_w)

    def print_row(name, values):
        nums = [v for v in values if isinstance(v, (int, float))]
        best = max(nums) if nums else None
        line = f"  {name:<28s}"
        for v in values:
            if isinstance(v, (int, float)):
                s = f"{v:.4f}"
                if best is not None and v == best and len(nums) > 1:
                    s += " *"
            else:
                s = "-"
            line += f" {s:>{col_w - 1}s}"
        print(line)

    for d in dims:
        values = [s.get("dimension_scores", {}).get(d) for _, s in runs]
        print_row(d, values)

    print("-" * row_w)
    for key, label in (("quality_score", "Quality Score"),
                       ("semantic_score", "Semantic Score"),
                       ("total_score", "Total Score")):
        values = [s.get("summary", {}).get(key) for _, s in runs]
        print_row(label, values)
    print("=" * row_w)
    print("  * = best among compared runs; '-' = dimension not evaluated")


if __name__ == "__main__":
    main()
