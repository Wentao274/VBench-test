#!/usr/bin/env python3
"""
VBench Evaluation Script - Step 2 of the pipeline

Evaluates generated videos using the VBench framework (original evaluate.py)
and calculates final Quality / Semantic / Total scores.

This script does NOT modify any original VBench source code. It calls the
original evaluate.py and static_filter.py as subprocesses.

Usage:
    # Evaluate all 16 dimensions
    python evaluate_videos.py --videos_path ./sampled_videos/my_model --all

    # Evaluate specific dimensions
    python evaluate_videos.py --videos_path ./sampled_videos/my_model \
        --dimensions subject_consistency color

    # Use locally downloaded model weights
    python evaluate_videos.py --videos_path ./sampled_videos/my_model --all \
        --load_ckpt_from_local

    # Re-evaluate only some dimensions: results are MERGED with the existing
    # final_scores.json (other dimensions' results are kept, not overwritten):
    python evaluate_videos.py --videos_path ./sampled_videos/my_model \
        --dimensions object_class multiple_objects \
        --output_path ./sampled_videos/my_model/evaluation_results
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Dimension config (mirrors sample_videos.py / evaluate.sh)
# ---------------------------------------------------------------------------
DIMENSION_TO_PROMPT_FILE = {
    "subject_consistency": "subject_consistency",
    "background_consistency": "scene",
    "temporal_flickering": "temporal_flickering",
    "motion_smoothness": "subject_consistency",
    "dynamic_degree": "subject_consistency",
    "aesthetic_quality": "overall_consistency",
    "imaging_quality": "overall_consistency",
    "object_class": "object_class",
    "multiple_objects": "multiple_objects",
    "human_action": "human_action",
    "color": "color",
    "spatial_relationship": "spatial_relationship",
    "scene": "scene",
    "temporal_style": "temporal_style",
    "appearance_style": "appearance_style",
    "overall_consistency": "overall_consistency",
}
ALL_DIMENSIONS = list(DIMENSION_TO_PROMPT_FILE.keys())

# ---------------------------------------------------------------------------
# Scoring constants (from scripts/constant.py — replicated to avoid imports)
# ---------------------------------------------------------------------------
TASK_INFO = [
    "subject consistency", "background consistency", "temporal flickering",
    "motion smoothness", "dynamic degree", "aesthetic quality",
    "imaging quality", "object class", "multiple objects", "human action",
    "color", "spatial relationship", "scene", "appearance style",
    "temporal style", "overall consistency",
]

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

DIM_WEIGHT = {
    "subject consistency": 1, "background consistency": 1,
    "temporal flickering": 1, "motion smoothness": 1,
    "aesthetic quality": 1, "imaging quality": 1,
    "dynamic degree": 0.5, "object class": 1,
    "multiple objects": 1, "human action": 1,
    "color": 1, "spatial relationship": 1,
    "scene": 1, "appearance style": 1,
    "temporal style": 1, "overall consistency": 1,
}

QUALITY_LIST = [
    "subject consistency", "background consistency", "temporal flickering",
    "motion smoothness", "aesthetic quality", "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    "object class", "multiple objects", "human action",
    "color", "spatial relationship", "scene",
    "appearance style", "temporal style", "overall consistency",
]

QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="VBench Evaluation Script - Step 2: evaluate generated videos",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--videos_path", type=str, default=None,
        help="Directory containing generated .mp4 videos (from sample_videos.py). "
             "Not required when using --download_weights.",
    )
    parser.add_argument(
        "--dimensions", nargs="+",
        help="VBench dimensions to evaluate, e.g. subject_consistency color",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all 16 VBench dimensions",
    )
    parser.add_argument(
        "--output_path", type=str, default="./evaluation_results",
        help="Directory to save evaluation results (default: ./evaluation_results)",
    )
    parser.add_argument(
        "--load_ckpt_from_local", action="store_true",
        help="Use locally downloaded model weights in ~/.cache/vbench",
    )
    parser.add_argument(
        "--skip_static_filter", action="store_true",
        help="Skip static filter for temporal_flickering (not recommended)",
    )
    parser.add_argument(
        "--ngpus", type=int, default=1,
        help="Number of GPUs for evaluation (default: 1)",
    )
    parser.add_argument(
        "--python", type=str, default=sys.executable,
        help="Python executable to use for subprocesses (default: current Python)",
    )
    parser.add_argument(
        "--keep_temp", action="store_true",
        help="Keep temporary clean directory after evaluation",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-evaluate dimensions even if final_scores.json already has "
             "successful results for them (default: reuse and skip)",
    )
    parser.add_argument(
        "--skip_integrity_check", action="store_true",
        help="Skip the pre-evaluation integrity check (missing/corrupted videos)",
    )
    parser.add_argument(
        "--download_weights", action="store_true",
        help="Download VBench evaluation weights for the selected dimensions "
             "(calls the original vbench.utils.init_submodules), then exit",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_integrity(videos_path, dimensions):
    """Check expected-video completeness and detect corrupted files.

    - Missing videos are reported as a WARNING (VBench evaluates with what is
      present; re-running sample_videos.py fills the gaps).
    - Corrupted/unreadable videos are returned so they can be excluded from
      the clean evaluation directory (they would crash VBench otherwise).

    Returns a list of corrupted filenames.
    """
    prompt_files_needed = {}
    for dim in dimensions:
        pf = DIMENSION_TO_PROMPT_FILE.get(dim)
        if pf:
            prompt_files_needed.setdefault(pf, set()).add(dim)

    prompts_dir = os.path.join(SCRIPT_DIR, "prompts", "prompts_per_dimension")

    expected_files = set()
    for pf in prompt_files_needed:
        num = 25 if pf == "temporal_flickering" else 5
        path = os.path.join(prompts_dir, f"{pf}.txt")
        if not os.path.exists(path):
            print(f"  WARNING: prompt file not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    for i in range(num):
                        expected_files.add(f"{p}-{i}.mp4")

    existing = set(
        f for f in os.listdir(videos_path)
        if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(videos_path, f))
    )

    missing = expected_files - existing
    if missing:
        pct = 100.0 * len(missing) / max(1, len(expected_files))
        print(f"  WARNING: {len(missing)}/{len(expected_files)} expected videos missing ({pct:.1f}%).")
        print(f"           VBench will evaluate with the videos present.")
        print(f"           To fill the gaps, re-run sample_videos.py with the same --output_dir (it resumes).")

    # Zero-byte files are detectable without any extra dependency
    corrupted = [f for f in sorted(existing) if os.path.getsize(os.path.join(videos_path, f)) == 0]

    try:
        import cv2
    except ImportError:
        print("  NOTE: cv2 not available, skipping header corruption check")
        return corrupted

    files = sorted(existing)
    for n, f in enumerate(files, 1):
        p = os.path.join(videos_path, f)
        if os.path.getsize(p) == 0:
            continue  # already flagged
        cap = cv2.VideoCapture(p)
        try:
            if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
                if f not in corrupted:
                    corrupted.append(f)
        finally:
            cap.release()
        if n % 1000 == 0:
            print(f"    ... checked {n}/{len(files)} videos")
    return corrupted


def create_clean_eval_dir(videos_path, exclude=None):
    """Create a temp directory containing only .mp4 files via hard links.

    This is necessary because VBench's build_full_info_json() takes the file
    suffix from os.listdir(videos_path)[0]. If a non-mp4 file (e.g.
    _sampling_log.json) appears first, the postfix becomes '.json' and no
    videos are found.

    `exclude` is a set of filenames to leave out (e.g. corrupted videos).
    """
    exclude = exclude or set()
    clean_dir = os.path.join(videos_path, "_vbench_eval_tmp")
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir)
    os.makedirs(clean_dir)

    mp4_files = [
        f for f in os.listdir(videos_path)
        if f.lower().endswith(".mp4") and f not in exclude
        and os.path.isfile(os.path.join(videos_path, f))
    ]
    if not mp4_files:
        raise RuntimeError(f"No .mp4 files found in {videos_path}")

    linked = 0
    for f in mp4_files:
        src = os.path.join(videos_path, f)
        dst = os.path.join(clean_dir, f)
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            shutil.copy2(src, dst)
            linked += 1

    print(f"  Created clean dir with {linked} .mp4 files: {clean_dir}")
    return clean_dir


def run_static_filter(videos_path, result_path, python_exec):
    """Run static_filter.py to filter static videos for temporal_flickering."""
    cmd = [
        python_exec,
        os.path.join(SCRIPT_DIR, "static_filter.py"),
        "--videos_path", videos_path,
        "--result_path", result_path,
    ]
    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    static_filter FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"    stderr: {result.stderr[-1500:]}")
        return None

    filtered_dir = os.path.join(result_path, "filtered_videos")
    if not os.path.isdir(filtered_dir):
        print(f"    Filtered directory not found: {filtered_dir}")
        return None

    filtered_files = [f for f in os.listdir(filtered_dir) if f.endswith(".mp4")]
    print(f"    Static filter produced {len(filtered_files)} videos")
    if len(filtered_files) == 0:
        return None

    return filtered_dir


def run_vbench_dimension(videos_path, dimension, output_path, load_local,
                         ngpus, python_exec):
    """Run the original evaluate.py for a single dimension via subprocess."""
    dim_output = os.path.join(output_path, dimension)
    os.makedirs(dim_output, exist_ok=True)

    cmd = [
        python_exec,
        os.path.join(SCRIPT_DIR, "evaluate.py"),
        "--videos_path", videos_path,
        "--dimension", dimension,
        "--output_path", dim_output,
    ]
    if load_local:
        cmd.extend(["--load_ckpt_from_local", "True"])

    if ngpus > 1:
        cmd = [
            "torchrun", f"--nproc_per_node={ngpus}", "--standalone",
        ] + cmd

    print(f"    Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    evaluate.py FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"    stderr: {result.stderr[-2000:]}")
        return None

    # Find the most recent *_eval_results.json
    eval_jsons = sorted(
        glob.glob(os.path.join(dim_output, "*_eval_results.json")),
        key=os.path.getmtime,
    )
    if not eval_jsons:
        print(f"    No eval_results.json found in {dim_output}")
        return None

    return eval_jsons[-1]


def extract_score(eval_json_path, dimension):
    """Extract the overall dimension score from an eval_results JSON.

    VBench compute functions return (all_results, video_results), so the JSON
    structure is: {"dimension": [score_float, [video_results...]]}
    """
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if dimension not in data:
        return None

    value = data[dimension]
    if isinstance(value, list) and len(value) > 0:
        return float(value[0])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def calculate_scores(dimension_scores, evaluated_dims):
    """Calculate Quality / Semantic / Total scores.

    Only evaluated dimensions contribute to the score. Missing dimensions
    are excluded (not set to 0), so partial evaluations give a fair score.
    """
    normalized = {}
    for dim_key, score in dimension_scores.items():
        dim_name = dim_key.replace("_", " ")
        if dim_name not in NORMALIZE_DIC:
            continue
        min_val = NORMALIZE_DIC[dim_name]["Min"]
        max_val = NORMALIZE_DIC[dim_name]["Max"]
        if max_val > min_val:
            # NOTE: no clamping — mirrors the official scripts/cal_final_score.py,
            # which allows normalized values outside [0, 1].
            norm = (score - min_val) / (max_val - min_val)
        else:
            norm = 0.0
        norm *= DIM_WEIGHT.get(dim_name, 1)
        normalized[dim_name] = norm

    # Quality score
    q_dims = [d for d in QUALITY_LIST if d in normalized]
    q_weight_sum = sum(DIM_WEIGHT[d] for d in q_dims)
    quality_score = (
        sum(normalized[d] for d in q_dims) / q_weight_sum
        if q_weight_sum > 0 else 0.0
    )

    # Semantic score
    s_dims = [d for d in SEMANTIC_LIST if d in normalized]
    s_weight_sum = sum(DIM_WEIGHT[d] for d in s_dims)
    semantic_score = (
        sum(normalized[d] for d in s_dims) / s_weight_sum
        if s_weight_sum > 0 else 0.0
    )

    # Total score
    total_weight = QUALITY_WEIGHT + SEMANTIC_WEIGHT
    total_score = (
        (quality_score * QUALITY_WEIGHT + semantic_score * SEMANTIC_WEIGHT)
        / total_weight
        if total_weight > 0 else 0.0
    )

    return quality_score, semantic_score, total_score, normalized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Determine dimensions
    if args.all:
        dimensions = ALL_DIMENSIONS
    elif args.dimensions:
        dimensions = args.dimensions
    else:
        print("ERROR: specify --dimensions or --all")
        sys.exit(1)

    for dim in dimensions:
        if dim not in DIMENSION_TO_PROMPT_FILE:
            print(f"ERROR: unknown dimension '{dim}'")
            sys.exit(1)

    # Weight pre-download mode: fetch evaluation weights for the selected
    # dimensions via the ORIGINAL vbench.utils.init_submodules (downloads
    # files only, no model loading, no videos needed), then exit.
    if args.download_weights:
        print("=" * 60)
        print("VBench Weight Downloader")
        print("=" * 60)
        print(f"  Dimensions ({len(dimensions)}): {', '.join(dimensions)}")
        print(f"  Local mode (--load_ckpt_from_local): {args.load_ckpt_from_local}")
        print("\n  Downloading via the original vbench.utils.init_submodules ...\n")
        code = (
            "from vbench.utils import init_submodules; "
            f"init_submodules({dimensions!r}, local={bool(args.load_ckpt_from_local)})"
        )
        result = subprocess.run([args.python, "-c", code], cwd=SCRIPT_DIR)
        if result.returncode == 0:
            print("\n  Weight download complete.")
            print("  Verify with: python run_vbench_main.py ... --preflight")
        else:
            print("\nERROR: weight download failed (see output above).")
            print("  Common causes: missing dependencies (torch/decord/PIL), no network, or no wget.")
            sys.exit(1)
        return

    if not args.videos_path:
        print("ERROR: --videos_path is required (unless using --download_weights)")
        sys.exit(1)

    print("=" * 60)
    print("VBench Evaluation - Step 2")
    print("=" * 60)
    print(f"  Videos path:      {args.videos_path}")
    print(f"  Output path:       {args.output_path}")
    print(f"  Dimensions ({len(dimensions)}):  {', '.join(dimensions)}")
    print(f"  Load local ckpt:   {args.load_ckpt_from_local}")
    print(f"  GPUs:              {args.ngpus}")
    print(f"  Skip static filter: {args.skip_static_filter}")

    # Check videos exist
    if not os.path.isdir(args.videos_path):
        print(f"\nERROR: videos directory not found: {args.videos_path}")
        sys.exit(1)

    mp4_count = len([
        f for f in os.listdir(args.videos_path)
        if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(args.videos_path, f))
    ])
    print(f"\n  Found {mp4_count} .mp4 files")
    if mp4_count == 0:
        print("ERROR: no .mp4 files found. Run sample_videos.py first.")
        sys.exit(1)

    # Integrity check: completeness (vs prompt suite) + corrupted video headers
    corrupted = []
    if not args.skip_integrity_check:
        print("\n  Integrity check (expected counts + video headers)...")
        corrupted = check_integrity(args.videos_path, dimensions)
        if corrupted:
            print(f"  WARNING: {len(corrupted)} corrupted/unreadable video(s) will be EXCLUDED:")
            for f in corrupted[:10]:
                print(f"    - {f}")
            if len(corrupted) > 10:
                print(f"    ... and {len(corrupted) - 10} more")
            print(f"  (Delete them or re-run sampling to regenerate; they are excluded so VBench")
            print(f"   does not crash on them.)")

    # Load previous results for resume + merging, so that re-running a subset
    # of dimensions never discards the other dimensions' results.
    results_path = os.path.join(args.output_path, "final_scores.json")
    prev_results = None
    if os.path.exists(results_path):
        try:
            # utf-8-sig: tolerate BOM written by some Windows editors
            with open(results_path, "r", encoding="utf-8-sig") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                prev_results = loaded
        except Exception as e:
            print(f"  WARNING: could not read existing {results_path}: {e}")

    # Step 1: Create clean directory
    print("\n[1/3] Creating clean evaluation directory...")
    clean_dir = create_clean_eval_dir(args.videos_path, exclude=set(corrupted))

    # Step 2: Evaluate each dimension
    print(f"\n[2/3] Running VBench evaluation ({len(dimensions)} dimensions)...")
    dimension_scores = {}
    eval_details = {}

    # Merge first: dims from a previous run that are NOT part of this run's
    # request are kept in the working set from the start, so incremental
    # saves never lose them. Dims attempted this run keep their current
    # outcome (a failed --force re-run must not be masked by the old value).
    if prev_results is not None:
        merged_from_previous = []
        for dim, detail in prev_results.get("details", {}).items():
            if dim in dimensions or dim not in DIMENSION_TO_PROMPT_FILE:
                continue
            if isinstance(detail, dict) and detail.get("status") == "success" \
                    and detail.get("score") is not None:
                dimension_scores[dim] = float(detail["score"])
                eval_details[dim] = dict(detail, reused=True)
                merged_from_previous.append(dim)
        if merged_from_previous:
            print(f"\n  Merged {len(merged_from_previous)} dimension(s) from previous results: "
                  f"{', '.join(merged_from_previous)}")

    def save_results_snapshot():
        """Persist current results to final_scores.json.

        Called after every completed dimension (and once at the end), so an
        interruption loses at most the dimension currently being evaluated.
        """
        quality, semantic, total, normalized = calculate_scores(
            dimension_scores, dimensions
        )
        is_partial = len(dimension_scores) < len(ALL_DIMENSIONS)
        results = {
            "summary": {
                "quality_score": round(quality, 4),
                "semantic_score": round(semantic, 4),
                "total_score": round(total, 4),
                "is_partial_evaluation": is_partial,
                "evaluated_dimensions": len(dimension_scores),
                "total_dimensions": len(ALL_DIMENSIONS),
            },
            "dimension_scores": {
                dim: round(score, 4) for dim, score in dimension_scores.items()
            },
            "normalized_scores": {
                k: round(v, 4) for k, v in normalized.items()
            },
            "details": eval_details,
        }
        os.makedirs(args.output_path, exist_ok=True)
        # Safety backup of the previous results before overwriting
        if os.path.exists(results_path):
            try:
                shutil.copy2(results_path, results_path + ".bak")
            except Exception as e:
                print(f"  WARNING: could not back up previous results: {e}")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    for i, dim in enumerate(dimensions, 1):
        print(f"\n  --- [{i}/{len(dimensions)}] {dim} ---")

        # Resume: reuse successful previous results unless --force
        if prev_results is not None and not args.force:
            prev_detail = prev_results.get("details", {}).get(dim)
            if prev_detail and prev_detail.get("status") == "success" \
                    and prev_detail.get("score") is not None:
                score = float(prev_detail["score"])
                print(f"    [SKIP] Reusing previous result (score={score:.4f}); use --force to re-evaluate")
                dimension_scores[dim] = score
                eval_details[dim] = dict(prev_detail, reused=True)
                save_results_snapshot()
                continue

        eval_videos = clean_dir

        # Special handling for temporal_flickering
        if dim == "temporal_flickering" and not args.skip_static_filter:
            print("    Running static filter (required for temporal_flickering)...")
            filter_result_path = os.path.join(args.output_path, "static_filter_results")
            filtered_dir = run_static_filter(clean_dir, filter_result_path, args.python)
            if filtered_dir is None:
                print(f"    [FAILED] Static filter failed for {dim}")
                eval_details[dim] = {"status": "filter_failed", "score": 0.0}
                save_results_snapshot()
                continue
            eval_videos = filtered_dir
        elif dim == "temporal_flickering" and args.skip_static_filter:
            print("    [WARNING] Skipping static filter (may produce inaccurate results)")

        # Run evaluate.py
        start_time = time.time()
        eval_json = run_vbench_dimension(
            eval_videos, dim, args.output_path,
            args.load_ckpt_from_local, args.ngpus, args.python,
        )
        elapsed = time.time() - start_time

        if eval_json is None:
            print(f"    [FAILED] Evaluation failed for {dim} ({elapsed:.1f}s)")
            eval_details[dim] = {"status": "eval_failed", "score": 0.0}
            save_results_snapshot()
            continue

        # Extract score
        score = extract_score(eval_json, dim)
        if score is None:
            print(f"    [FAILED] Could not extract score for {dim}")
            eval_details[dim] = {"status": "score_extraction_failed", "score": 0.0}
            save_results_snapshot()
            continue

        print(f"    [OK] Score: {score:.4f}  ({elapsed:.1f}s)")
        dimension_scores[dim] = score
        eval_details[dim] = {
            "status": "success",
            "score": score,
            "eval_json": eval_json,
            "elapsed_s": round(elapsed, 1),
        }
        save_results_snapshot()

    # Step 3: Final scores (per-dimension snapshots were already written in
    # [2/3], so an interruption loses at most the dimension in progress)
    print(f"\n[3/3] Calculating final scores...")
    quality_score, semantic_score, total_score, normalized = calculate_scores(
        dimension_scores, dimensions
    )
    is_partial = len(dimension_scores) < len(ALL_DIMENSIONS)
    save_results_snapshot()

    # Print results table (canonical dimension order).
    # NOTE: failed dimensions are shown but do NOT contribute to the scores —
    # this matches the documented design of calculate_scores().
    ordered_dims = [d for d in ALL_DIMENSIONS if d in dimension_scores or d in eval_details]
    ordered_dims += [d for d in dimension_scores if d not in ordered_dims]

    print("\n" + "=" * 65)
    print("                    VBench Evaluation Results")
    print("=" * 65)
    print(f"  {'Dimension':<30s} {'Raw':>10s} {'Normalized':>12s}  Status")
    print("-" * 65)
    for dim in ordered_dims:
        dim_name = dim.replace("_", " ")
        raw = dimension_scores.get(dim)
        raw_s = f"{raw:>10.4f}" if raw is not None else f"{'-':>10s}"
        norm = normalized.get(dim_name)
        norm_s = f"{norm:>12.4f}" if norm is not None else f"{'-':>12s}"
        detail = eval_details.get(dim, {})
        status = detail.get("status", "unknown")
        marker = ""
        if status != "success":
            marker = "  FAIL"
        elif detail.get("reused"):
            marker = "  (reused)"
        print(f"  {dim:<30s} {raw_s} {norm_s}{marker}")
    print("-" * 65)
    print(f"  {'Quality Score':<30s} {quality_score:>10.4f}")
    print(f"  {'Semantic Score':<30s} {semantic_score:>10.4f}")
    print(f"  {'Total Score':<30s} {total_score:>10.4f}")
    print("=" * 65)
    if is_partial:
        print(f"\n  NOTE: Partial evaluation ({len(dimension_scores)}/{len(ALL_DIMENSIONS)} dimensions).")
        print(f"        Scores are computed from evaluated dimensions only.")
        print(f"        For official leaderboard scores, evaluate all 16 dimensions with --all.")

    failed = [d for d, v in eval_details.items() if v["status"] != "success"]
    if failed:
        print(f"\n  {len(failed)} dimension(s) failed: {', '.join(failed)}")
        print(f"  Check error messages above and ensure model weights are downloaded.")
        print(f"  Re-run the same command afterwards — successful dimensions are reused,")
        print(f"  and results are merged (no data loss).")

    if any(v.get("reused") for v in eval_details.values()):
        print(f"\n  NOTE: some dimension results were reused/merged from previous runs (marked '(reused)').")
        print(f"        Use --force to re-evaluate everything from scratch.")

    print(f"\n  Results saved to: {results_path}")

    # Cleanup
    if not args.keep_temp:
        try:
            shutil.rmtree(clean_dir)
            print(f"  Cleaned up: {clean_dir}")
        except Exception as e:
            print(f"  Warning: could not clean up {clean_dir}: {e}")
    else:
        print(f"  Temp directory kept: {clean_dir}")

    print()


if __name__ == "__main__":
    main()
