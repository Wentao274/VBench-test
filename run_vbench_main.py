#!/usr/bin/env python3
"""
VBench Pipeline - Main Entry

End-to-end pipeline for evaluating a local video generation model with VBench:

  Step 1 — sample_videos.py   Generate videos via local model API
  Step 2 — evaluate_videos.py  Evaluate generated videos with VBench
  Step 3 — (built-in)          Display Quality / Semantic / Total scores

Usage:
    # Full pipeline: generate all 16 dimensions + evaluate + score
    python run_vbench_main.py \
        --gateway_url http://localhost:8000 \
        --model minimax-h3 \
        --all \
        --output_dir ./sampled_videos/my_model

    # Only specific dimensions
    python run_vbench_main.py \
        --gateway_url http://localhost:8000 \
        --model minimax-h3 \
        --dimensions subject_consistency color \
        --output_dir ./sampled_videos/my_model

    # Dry run (show plan without calling APIs)
    python run_vbench_main.py \
        --gateway_url http://localhost:8000 \
        --model minimax-h3 --all \
        --output_dir ./sampled_videos/my_model --dry_run

    # Skip sampling (videos already generated), only evaluate latest run
    python run_vbench_main.py \
        --gateway_url http://localhost:8000 \
        --model minimax-h3 --all \
        --output_dir ./sampled_videos/my_model --skip_sample

    # Skip evaluation (only generate videos)
    python run_vbench_main.py \
        --gateway_url http://localhost:8000 \
        --model minimax-h3 --all \
        --output_dir ./sampled_videos/my_model --skip_eval
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_DIMENSIONS = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality",
    "imaging_quality", "object_class", "multiple_objects",
    "human_action", "color", "spatial_relationship",
    "scene", "temporal_style", "appearance_style", "overall_consistency",
]

# Dimension -> prompt file mapping (mirrors sample_videos.py)
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

GRIT_DIMENSIONS = ("object_class", "multiple_objects", "color", "spatial_relationship")

# VBench evaluation weight files under the VBench cache dir
# (~/.cache/vbench or $VBENCH_CACHE_DIR), replicated from vbench/utils.py
# init_submodules() so preflight can verify readiness without importing vbench.
WEIGHTS_ALWAYS = {
    "motion_smoothness": ["amt_model/amt-s.pth"],
    "dynamic_degree": ["raft_model/models/raft-things.pth"],
    "aesthetic_quality": ["aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"],
    "imaging_quality": ["pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"],
    "object_class": ["grit_model/grit_b_densecap_objectdet.pth"],
    "multiple_objects": ["grit_model/grit_b_densecap_objectdet.pth"],
    "color": ["grit_model/grit_b_densecap_objectdet.pth"],
    "spatial_relationship": ["grit_model/grit_b_densecap_objectdet.pth"],
    "human_action": ["umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth"],
    "scene": ["caption_model/tag2text_swin_14m.pth"],
    "temporal_style": ["ViCLIP/ViClip-InternVid-10M-FLT.pth"],
    "overall_consistency": ["ViCLIP/ViClip-InternVid-10M-FLT.pth"],
}

# Extra files needed only in --load_ckpt_from_local mode. In network mode
# DINO/CLIP weights are fetched at runtime by torch.hub / the clip package.
WEIGHTS_LOCAL_ONLY = {
    "subject_consistency": ["dino_model/dino_vitbase16_pretrain.pth",
                            "dino_model/facebookresearch_dino_main"],
    "background_consistency": ["clip_model/ViT-B-32.pt"],
    "aesthetic_quality": ["clip_model/ViT-L-14.pt"],
    "appearance_style": ["clip_model/ViT-B-32.pt"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="VBench Pipeline: sample videos -> evaluate -> score",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Required ---
    parser.add_argument(
        "--gateway_url", type=str, required=True,
        help="Video Gateway base URL, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Base directory for generated videos. A timestamped subdirectory "
             "(YYYYMMDD_HHMMSS) is created under this for each run.",
    )

    # --- Dimension selection ---
    parser.add_argument(
        "--dimensions", nargs="+",
        help="VBench dimensions to evaluate, e.g. subject_consistency color",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Sample and evaluate all 16 VBench dimensions",
    )

    # --- Step 1: Sampling parameters ---
    parser.add_argument("--model", type=str, default="minimax-h3",
                        help="Model name for Gateway (default: minimax-h3)")
    parser.add_argument("--seconds", type=int, default=5,
                        help="Video duration in seconds 4-15 (default: 5)")
    parser.add_argument("--size", type=str, default="1344x768",
                        help="Video size WIDTHxHEIGHT (default: 1344x768)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Frames per second (default: 24)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Inference steps 2-200 (default: 50)")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Base random seed (default: 0)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent generation tasks (default: 4)")
    parser.add_argument("--poll_interval", type=float, default=10.0,
                        help="Seconds between status polls (default: 10)")
    parser.add_argument("--poll_timeout", type=float, default=900.0,
                        help="Max wait per task in seconds (default: 900)")
    parser.add_argument("--max_prompts", type=int, default=0,
                        help="Limit unique prompts to sample (smoke test). 0 = all (default: 0)")
    parser.add_argument("--generate_sound", action="store_true",
                        help="Also generate audio in videos (default: off, VBench does not use audio)")
    parser.add_argument("--max_fail_rate", type=float, default=0.05,
                        help="Abort if sampling failure rate exceeds this fraction (default: 0.05)")

    # --- Step 2: Evaluation parameters ---
    parser.add_argument("--eval_output_path", type=str, default=None,
                        help="Output path for evaluation results "
                             "(default: {output_dir}/evaluation_results)")
    parser.add_argument("--load_ckpt_from_local", action="store_true",
                        help="Use locally downloaded VBench model weights")
    parser.add_argument("--skip_static_filter", action="store_true",
                        help="Skip static filter for temporal_flickering")
    parser.add_argument("--ngpus", type=int, default=1,
                        help="Number of GPUs for evaluation (default: 1)")
    parser.add_argument("--gpu_id", type=str, default=None,
                        help="Set CUDA_VISIBLE_DEVICES for all subprocesses, e.g. '1' or '0,1'")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate dimensions that already have successful results")
    parser.add_argument("--skip_integrity_check", action="store_true",
                        help="Skip pre-evaluation video integrity check")

    # --- Pipeline control ---
    parser.add_argument("--skip_sample", action="store_true",
                        help="Skip step 1, evaluate the latest timestamped run")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip step 2 (only generate videos)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume the LATEST timestamped run under --output_dir: "
                             "finish missing videos (existing files are skipped), "
                             "then re-evaluate with successful-dimension reuse")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show plan without calling APIs or evaluating")
    parser.add_argument("--preflight", action="store_true",
                        help="Only run environment checks (gateway / CUDA / disk) and exit")

    return parser.parse_args()


def run_step(name, cmd, cwd=None):
    """Run a subprocess step and return its exit code."""
    print(f"\n  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def find_latest_timestamp(base_dir):
    """Find the latest timestamp subdirectory under base_dir.

    Matches YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_N (collision-avoidance suffix).
    """
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full) or len(name) < 15 or name[8] != "_":
            continue
        if not (name[:8].isdigit() and name[9:15].isdigit()):
            continue
        suffix = name[15:]
        if suffix and not (suffix.startswith("_") and suffix[1:].isdigit()):
            continue
        candidates.append(name)
    if not candidates:
        return None
    return max(candidates)


def estimate_video_count(dimensions):
    """Estimate the total number of videos for the selected dimensions."""
    prompts_dir = os.path.join(SCRIPT_DIR, "prompts", "prompts_per_dimension")
    seen = set()
    total = 0
    for dim in dimensions:
        pf = DIMENSION_TO_PROMPT_FILE.get(dim)
        if not pf or pf in seen:
            continue
        seen.add(pf)
        num = 25 if pf == "temporal_flickering" else 5
        path = os.path.join(prompts_dir, f"{pf}.txt")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        total += count * num
    return total


def check_weights(dimensions, load_local=False, skip_static_filter=False):
    """Check VBench weight readiness for the given dimensions.

    Returns {dimension: [missing relative paths]} — empty dict = all present.
    """
    cache = os.environ.get("VBENCH_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "vbench")

    missing = {}
    for dim in dimensions:
        rels = list(WEIGHTS_ALWAYS.get(dim, []))
        if load_local:
            rels += WEIGHTS_LOCAL_ONLY.get(dim, [])
        # static_filter (required by temporal_flickering) uses RAFT
        if dim == "temporal_flickering" and not skip_static_filter:
            rels.append("raft_model/models/raft-things.pth")
        miss = [r for r in rels if not os.path.exists(os.path.join(cache, r))]
        if miss:
            missing[dim] = miss
    return missing


def run_preflight(args, num_videos, run_dir):
    """Environment pre-checks before starting the (long) pipeline.

    Returns the number of failed CRITICAL checks (0 = all passed).
    """
    print("\n" + "=" * 65)
    print("  Preflight checks")
    print("=" * 65)
    critical_failures = 0

    # 1. Gateway reachable (needed for sampling)
    if not args.skip_sample:
        try:
            import requests
            resp = requests.get(f"{args.gateway_url.rstrip('/')}/v1/models", timeout=10)
            if resp.status_code == 200:
                print(f"  [OK]   Gateway reachable: {args.gateway_url}")
            else:
                print(f"  [FAIL] Gateway responded HTTP {resp.status_code}: {args.gateway_url}")
                critical_failures += 1
        except Exception as e:
            print(f"  [FAIL] Gateway not reachable: {args.gateway_url} ({e})")
            critical_failures += 1
    else:
        print(f"  [SKIP] Gateway check (--skip_sample)")

    # 2. Disk space (rough estimate: ~15 MB per 5s 1344x768 video)
    if not args.skip_sample:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            est_bytes = num_videos * 15 * 1024 * 1024
            free = shutil.disk_usage(args.output_dir).free
            est_gb, free_gb = est_bytes / (1024 ** 3), free / (1024 ** 3)
            if free < est_bytes:
                print(f"  [WARN] Disk space low: {free_gb:.1f} GB free, "
                      f"~{est_gb:.1f} GB estimated for {num_videos} videos")
            else:
                print(f"  [OK]   Disk space: {free_gb:.1f} GB free "
                      f"(~{est_gb:.1f} GB estimated)")
        except Exception as e:
            print(f"  [WARN] Could not check disk space: {e}")

    # 3. CUDA (needed for evaluation)
    dims = ALL_DIMENSIONS if args.all else (args.dimensions or [])
    if not args.skip_eval:
        try:
            import torch
            if torch.cuda.is_available():
                n_devices = torch.cuda.device_count()
                print(f"  [OK]   CUDA available ({n_devices} device(s))")
                if n_devices < args.ngpus:
                    print(f"  [FAIL] --ngpus={args.ngpus} but only {n_devices} CUDA device(s) visible")
                    critical_failures += 1
                # VRAM per device — evaluation coexists with the serving process
                low_vram = False
                try:
                    for idx in range(n_devices):
                        free, total = torch.cuda.mem_get_info(idx)
                        free_gb, total_gb = free / 1024 ** 3, total / 1024 ** 3
                        name = torch.cuda.get_device_name(idx)
                        if free_gb < 8:
                            print(f"  [WARN] GPU {idx} ({name}): {free_gb:.1f} GB free / {total_gb:.1f} GB "
                                  f"— evaluation may OOM (serving co-located?)")
                            low_vram = True
                        else:
                            print(f"  [OK]   GPU {idx} ({name}): {free_gb:.1f} GB free / {total_gb:.1f} GB")
                    if low_vram:
                        print(f"        Tip: use --gpu_id <id> (or CUDA_VISIBLE_DEVICES) to pick a freer GPU")
                except Exception as e:
                    print(f"  [WARN] Could not query VRAM: {e}")
            else:
                print(f"  [FAIL] CUDA not available — VBench evaluation requires a GPU")
                critical_failures += 1
        except ImportError:
            print(f"  [FAIL] torch not installed — evaluation requires PyTorch")
            critical_failures += 1

        # 4. Evaluation weights readiness (only meaningful when CUDA checks ran)
        missing = check_weights(dims, args.load_ckpt_from_local, args.skip_static_filter)
        if missing:
            n_files = sum(len(v) for v in missing.values())
            print(f"  [WARN] {n_files} evaluation weight file(s) missing:")
            for dim, files in missing.items():
                for f in files:
                    print(f"         - {dim}: {f}")
            if shutil.which("wget") is not None:
                print(f"        They will be auto-downloaded during evaluation (needs network).")
                print(f"        To fetch them now: python evaluate_videos.py --all --download_weights")
            else:
                print(f"  [FAIL] wget not available and weights missing — evaluation will fail")
                critical_failures += 1
        else:
            print(f"  [OK]   Evaluation weights present in VBench cache")

        if not args.load_ckpt_from_local:
            runtime_dl = [d for d in dims if d in (
                "subject_consistency", "background_consistency",
                "appearance_style", "aesthetic_quality")]
            if runtime_dl:
                print(f"  [INFO] network mode: DINO/CLIP models download at runtime "
                      f"(torch.hub / clip package; needs git + network)")

        # 5. detectron2 (required by the 4 GRiT dimensions)
        grit_selected = [d for d in dims if d in GRIT_DIMENSIONS]
        if grit_selected:
            r = subprocess.run([sys.executable, "-c", "import detectron2"],
                                capture_output=True)
            if r.returncode == 0:
                print(f"  [OK]   detectron2 installed (required by GRiT dimensions)")
            else:
                print(f"  [WARN] detectron2 NOT installed — these dimensions will fail: "
                      f"{', '.join(grit_selected)}")
    else:
        print(f"  [SKIP] CUDA check (--skip_eval)")

    # 6. Videos present when skipping sampling (only needed for evaluation)
    if args.skip_sample and not args.skip_eval:
        mp4s = [
            f for f in os.listdir(run_dir)
            if f.lower().endswith(".mp4")
        ] if os.path.isdir(run_dir) else []
        if mp4s:
            print(f"  [OK]   {len(mp4s)} videos found in run dir")
        else:
            print(f"  [FAIL] No .mp4 files in {run_dir}")
            critical_failures += 1

    return critical_failures


def main():
    args = parse_args()

    # --gpu_id must be applied before any torch import / subprocess launch
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Determine dimensions
    if args.all:
        dimensions = ALL_DIMENSIONS
    elif args.dimensions:
        dimensions = args.dimensions
    else:
        print("ERROR: specify --dimensions or --all")
        sys.exit(1)

    # Determine the actual run directory (with timestamp subdirectory)
    if args.resume:
        # Resume: continue the latest existing run — sampling fills in only
        # the missing videos (existing files are skipped), evaluation reuses
        # previously successful dimension results.
        latest = find_latest_timestamp(args.output_dir)
        if latest is None:
            print(f"ERROR: no timestamped run found under {args.output_dir}")
            print(f"  Run without --resume first.")
            sys.exit(1)
        run_dir = os.path.join(args.output_dir, latest)
        existing = [
            f for f in os.listdir(run_dir)
            if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(run_dir, f))
        ] if os.path.isdir(run_dir) else []
        print(f"  Resuming latest run: {latest} ({len(existing)} video(s) already on disk)")
    elif not args.skip_sample:
        # Fresh sampling: create a new timestamp directory.
        # If the exact-second directory already exists (collision), append a
        # numeric suffix instead of mixing videos into the existing run.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, timestamp)
        if os.path.exists(run_dir):
            suffix = 1
            while os.path.exists(os.path.join(args.output_dir, f"{timestamp}_{suffix}")):
                suffix += 1
            run_dir = os.path.join(args.output_dir, f"{timestamp}_{suffix}")
        if args.dry_run:
            # Dry run: don't create dir, just show the plan
            print(f"  (Dry run: would create {run_dir})")
        else:
            os.makedirs(run_dir, exist_ok=True)
    else:
        # Skip sampling: find latest timestamp directory
        latest = find_latest_timestamp(args.output_dir)
        if latest is None:
            print(f"ERROR: no timestamped run found under {args.output_dir}")
            print(f"  Run without --skip_sample first.")
            sys.exit(1)
        run_dir = os.path.join(args.output_dir, latest)
        print(f"  Using latest run: {latest}")

    eval_output_path = args.eval_output_path or os.path.join(run_dir, "evaluation_results")

    print("=" * 65)
    print("           VBench Evaluation Pipeline")
    print("=" * 65)
    print(f"  Gateway URL:    {args.gateway_url}")
    print(f"  Model:          {args.model}")
    print(f"  Base output:    {args.output_dir}")
    print(f"  Run dir:        {run_dir}")
    print(f"  Eval output:    {eval_output_path}")
    print(f"  Dimensions:     {len(dimensions)} ({', '.join(dimensions)})")
    num_videos = estimate_video_count(dimensions)
    print(f"  Videos (est.):  {num_videos}")
    print(f"  Video params:   {args.seconds}s, {args.size}, {args.fps}fps, "
          f"{args.num_inference_steps} steps")
    print(f"  Concurrency:    {args.concurrency}")
    print(f"  GPUs:           {args.ngpus}")
    if args.gpu_id:
        print(f"  GPU id:         {args.gpu_id} (CUDA_VISIBLE_DEVICES)")
    print(f"  Dry run:        {args.dry_run}")
    print(f"  Skip sample:    {args.skip_sample}")
    print(f"  Skip eval:      {args.skip_eval}")

    # ------------------------------------------------------------------
    # Preflight checks
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n  [SKIP] Preflight checks (--dry_run makes no network calls)")
    else:
        failures = run_preflight(args, num_videos, run_dir)
        if failures > 0:
            print(f"\nERROR: {failures} preflight check(s) FAILED — aborting.")
            print(f"  Fix the issues above, or run with --preflight to only check the environment.")
            sys.exit(1)
        if args.preflight:
            print("\n  All preflight checks passed. Exiting (--preflight).")
            return

    # ------------------------------------------------------------------
    # STEP 1: Generate videos
    # ------------------------------------------------------------------
    if not args.skip_sample:
        print("\n" + "=" * 65)
        print("  STEP 1: Video Sampling (sample_videos.py)")
        print("=" * 65)

        sample_cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "sample_videos.py"),
            "--gateway_url", args.gateway_url,
            "--model", args.model,
            "--output_dir", run_dir,
            "--seconds", str(args.seconds),
            "--size", args.size,
            "--fps", str(args.fps),
            "--num_inference_steps", str(args.num_inference_steps),
            "--base_seed", str(args.base_seed),
            "--concurrency", str(args.concurrency),
            "--poll_interval", str(args.poll_interval),
            "--poll_timeout", str(args.poll_timeout),
        ]
        if args.max_prompts > 0:
            sample_cmd.extend(["--max_prompts", str(args.max_prompts)])
        if args.generate_sound:
            sample_cmd.append("--generate_sound")
        sample_cmd.extend(["--max_fail_rate", str(args.max_fail_rate)])
        if args.all:
            sample_cmd.append("--all")
        else:
            sample_cmd.extend(["--dimensions"] + args.dimensions)

        if args.dry_run:
            sample_cmd.append("--dry_run")

        rc = run_step("Sampling", sample_cmd, cwd=SCRIPT_DIR)
        if rc != 0:
            print("\nERROR: Step 1 (sampling) failed.")
            sys.exit(1)

        if args.dry_run:
            print("\n" + "=" * 65)
            print("  Dry run complete. Remove --dry_run to execute.")
            print("=" * 65)
            return
    else:
        print("\n  [SKIP] Step 1: Sampling (--skip_sample)")

    # ------------------------------------------------------------------
    # STEP 2: Evaluate videos
    # ------------------------------------------------------------------
    if not args.skip_eval:
        print("\n" + "=" * 65)
        print("  STEP 2: VBench Evaluation (evaluate_videos.py)")
        print("=" * 65)

        eval_cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "evaluate_videos.py"),
            "--videos_path", run_dir,
            "--output_path", eval_output_path,
            "--python", sys.executable,
            "--ngpus", str(args.ngpus),
        ]
        if args.all:
            eval_cmd.append("--all")
        else:
            eval_cmd.extend(["--dimensions"] + args.dimensions)

        if args.load_ckpt_from_local:
            eval_cmd.append("--load_ckpt_from_local")
        if args.skip_static_filter:
            eval_cmd.append("--skip_static_filter")
        if args.force:
            eval_cmd.append("--force")
        if args.skip_integrity_check:
            eval_cmd.append("--skip_integrity_check")

        rc = run_step("Evaluation", eval_cmd, cwd=SCRIPT_DIR)
        if rc != 0:
            print("\nERROR: Step 2 (evaluation) failed.")
            sys.exit(1)
    else:
        print("\n  [SKIP] Step 2: Evaluation (--skip_eval)")

    # ------------------------------------------------------------------
    # STEP 3: Display final scores
    # ------------------------------------------------------------------
    if not args.skip_eval:
        print("\n" + "=" * 65)
        print("  STEP 3: Final Scores")
        print("=" * 65)

        final_scores_path = os.path.join(eval_output_path, "final_scores.json")
        if os.path.exists(final_scores_path):
            with open(final_scores_path, "r", encoding="utf-8") as f:
                scores = json.load(f)

            summary = scores.get("summary", {})
            print(f"\n  Quality Score:   {summary.get('quality_score', 'N/A')}")
            print(f"  Semantic Score:  {summary.get('semantic_score', 'N/A')}")
            print(f"  Total Score:     {summary.get('total_score', 'N/A')}")

            dim_scores = scores.get("dimension_scores", {})
            if dim_scores:
                print(f"\n  Per-dimension scores:")
                print(f"  {'Dimension':<30s} {'Score':>10s}")
                print(f"  {'-' * 42}")
                for dim, score in dim_scores.items():
                    print(f"  {dim:<30s} {score:>10.4f}")

            if summary.get("is_partial_evaluation"):
                print(f"\n  NOTE: Partial evaluation "
                      f"({summary.get('evaluated_dimensions', 0)}/"
                      f"{summary.get('total_dimensions', 16)} dimensions).")

            print(f"\n  Full results: {final_scores_path}")
        else:
            print(f"\n  Results file not found: {final_scores_path}")
    else:
        print("\n  [SKIP] Step 3: Scores (--skip_eval)")

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
