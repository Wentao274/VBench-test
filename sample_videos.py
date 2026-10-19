#!/usr/bin/env python3
"""
VBench Sampling Bridge Script for MiniMax-H3 Video Gateway

Reads VBench prompt suites, calls the local Video Gateway API to generate videos,
and saves them with VBench's expected naming convention for subsequent evaluation.

Usage:
    python sample_videos.py \
        --gateway_url http://localhost:8000 \
        --model minimax-h3 \
        --dimensions subject_consistency background_consistency \
        --output_dir ./sampled_videos/my_model \
        --seconds 5 \
        --size 1344x768 \
        --num_inference_steps 50 \
        --concurrency 4

    # Generate for all 16 dimensions:
    python sample_videos.py --gateway_url http://localhost:8000 --model minimax-h3 --all --output_dir ./sampled_videos/my_model

    # Dry run (show plan without calling API):
    python sample_videos.py --gateway_url http://localhost:8000 --model minimax-h3 --all --output_dir ./sampled_videos/my_model --dry_run

    # Smoke test (limit prompts for quick end-to-end validation):
    python sample_videos.py --gateway_url http://localhost:8000 --model minimax-h3 --all --output_dir ./sampled_videos/smoke --max_prompts 2

After sampling, evaluate with:
    python evaluate.py --videos_path ./sampled_videos/my_model --dimension subject_consistency background_consistency
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts", "prompts_per_dimension")

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

PROMPT_FILE_NEEDS_25 = {"temporal_flickering"}

VIDEOS_PER_PROMPT = 5
VIDEOS_PER_PROMPT_25 = 25

WINDOWS_INVALID_CHARS = set('<>:"\\|?*')


def parse_args():
    parser = argparse.ArgumentParser(
        description="VBench Sampling Bridge: generate videos via Video Gateway API for VBench evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gateway_url",
        type=str,
        required=True,
        help="Video Gateway base URL, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="minimax-h3",
        help="Model name for the Gateway (default: minimax-h3)",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        help="VBench dimensions to sample videos for, e.g. subject_consistency background_consistency",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sample videos for all 16 VBench dimensions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for generated videos",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=5,
        help="Video duration in seconds, 4-15 (default: 5)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1344x768",
        help="Video size WIDTHxHEIGHT, multiples of 32, max total pixels 768x1344 (default: 1344x768)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (MiniMax-H3 only accepts 24, default: 24)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps, 2-200 (default: 50)",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base random seed; per-video seeds are derived deterministically (default: 0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent video generation tasks (default: 4)",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=10.0,
        help="Seconds between status polls (default: 10)",
    )
    parser.add_argument(
        "--poll_timeout",
        type=float,
        default=900.0,
        help="Max seconds to wait for a single task to complete (default: 900)",
    )
    parser.add_argument(
        "--max_submit_retries",
        type=int,
        default=3,
        help="Max retries for task submission on transient errors (default: 3)",
    )
    parser.add_argument(
        "--max_download_retries",
        type=int,
        default=3,
        help="Max retries for downloading the result video (default: 3)",
    )
    parser.add_argument(
        "--generate_sound",
        action="store_true",
        help="Also generate audio in output videos (default: off — VBench does not evaluate audio)",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=0,
        help="Limit the number of unique prompts to sample (smoke test). 0 = no limit (default: 0)",
    )
    parser.add_argument(
        "--max_fail_rate",
        type=float,
        default=0.05,
        help="Exit with code 1 if the failure rate exceeds this fraction (default: 0.05)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the sampling plan without calling the API",
    )
    parser.add_argument(
        "--list_dimensions",
        action="store_true",
        help="List all available VBench dimensions and exit",
    )
    return parser.parse_args()


def load_prompt_files(dimensions):
    prompt_files_needed = {}
    for dim in dimensions:
        pf = DIMENSION_TO_PROMPT_FILE.get(dim)
        if pf is None:
            print(f"WARNING: Unknown dimension '{dim}', skipping")
            continue
        if pf not in prompt_files_needed:
            prompt_files_needed[pf] = set()
        prompt_files_needed[pf].add(dim)
    return prompt_files_needed


def load_prompts(prompt_files_needed):
    prompts_info = []
    seen_prompts = set()

    for prompt_file, dims in sorted(prompt_files_needed.items()):
        file_path = os.path.join(PROMPTS_DIR, f"{prompt_file}.txt")
        if not os.path.exists(file_path):
            print(f"WARNING: Prompt file not found: {file_path}")
            continue

        num_videos = VIDEOS_PER_PROMPT_25 if prompt_file in PROMPT_FILE_NEEDS_25 else VIDEOS_PER_PROMPT

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if not prompt:
                    continue
                if prompt in seen_prompts:
                    for pi in prompts_info:
                        if pi["prompt"] == prompt:
                            pi["dimensions"].update(dims)
                            pi["num_videos"] = max(pi["num_videos"], num_videos)
                    continue
                seen_prompts.add(prompt)

                invalid = WINDOWS_INVALID_CHARS & set(prompt)
                if invalid:
                    print(f"WARNING: Prompt contains Windows-invalid chars {invalid}: {prompt!r}")

                prompts_info.append({
                    "prompt": prompt,
                    "prompt_file": prompt_file,
                    "dimensions": set(dims),
                    "num_videos": num_videos,
                })
    return prompts_info


def make_seed(base_seed, prompt, index):
    h = hashlib.md5(f"{prompt}-{index}".encode("utf-8")).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2**32)


def make_idempotency_key(prompt, index):
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16]
    return f"vbench-{h}-{index}"


def submit_task(gateway_url, model, prompt, seconds, size, fps, steps, seed,
                idempotency_key, max_retries, generate_sound=False):
    url = f"{gateway_url.rstrip('/')}/v1/videos"
    fields = {
        "model": (None, model),
        "prompt": (None, prompt),
        "seconds": (None, str(seconds)),
        "size": (None, size),
        "fps": (None, str(fps)),
        "num_inference_steps": (None, str(steps)),
        "seed": (None, str(seed)),
    }
    # VBench does not evaluate audio; disable it by default to save
    # server-side compute and reduce download size.
    if not generate_sound:
        fields["generate_sound"] = (None, "false")
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                files=fields,
                headers={"Idempotency-Key": idempotency_key},
                timeout=60,
            )
            if resp.status_code in (200, 202):
                return resp.json().get("id"), None
            if resp.status_code == 429:
                wait = min(60, 10 * attempt)
                time.sleep(wait)
                continue
            if resp.status_code in (502, 503, 504):
                time.sleep(5 * attempt)
                continue
            try:
                err_body = resp.json().get("error", {})
                err_msg = err_body.get("message", resp.text)
                err_code = err_body.get("code", "")
            except Exception:
                err_msg = resp.text
                err_code = ""
            if resp.status_code == 409 and err_code == "idempotency_conflict":
                return None, f"idempotency_conflict: {err_msg}"
            return None, f"submit HTTP {resp.status_code} [{err_code}]: {err_msg}"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(5 * attempt)
                continue
            return None, f"submit request error: {e}"
    return None, "submit exhausted retries"


def poll_task(gateway_url, task_id, poll_interval, poll_timeout):
    url = f"{gateway_url.rstrip('/')}/v1/videos/{task_id}"
    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "")
                if status == "completed":
                    return data, None
                if status in ("failed", "cancelled", "expired"):
                    err = data.get("error", {})
                    return None, f"task {status}: {err.get('message', err)}"
        except requests.exceptions.RequestException:
            pass
        time.sleep(poll_interval)
    return None, f"poll timeout after {poll_timeout}s (task_id={task_id})"


def check_task_once(gateway_url, task_id):
    """Single status check without polling (used for the final sweep)."""
    url = f"{gateway_url.rstrip('/')}/v1/videos/{task_id}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def download_video(gateway_url, task_id, output_path, max_retries):
    url = f"{gateway_url.rstrip('/')}/v1/videos/{task_id}/content"
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=180, stream=True)
            if resp.status_code == 200:
                tmp_path = output_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                if os.path.getsize(tmp_path) > 0:
                    os.replace(tmp_path, output_path)
                    return True, None
                else:
                    os.remove(tmp_path)
                    return False, "downloaded file is empty"
            if resp.status_code == 409:
                time.sleep(5 * attempt)
                continue
            if resp.status_code == 410:
                return False, "video expired"
            return False, f"download HTTP {resp.status_code}"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(5 * attempt)
                continue
            return False, f"download request error: {e}"
    return False, "download exhausted retries"


def generate_one_video(item, gateway_url, model, seconds, size, fps, steps,
                        base_seed, concurrency_params):
    prompt = item["prompt"]
    index = item["index"]
    output_dir = item["output_dir"]

    output_filename = f"{prompt}-{index}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return {
            "prompt": prompt,
            "index": index,
            "status": "skipped",
            "output": output_path,
            "seed": make_seed(base_seed, prompt, index),
        }

    seed = make_seed(base_seed, prompt, index)
    idem_key = make_idempotency_key(prompt, index)

    task_id, err = submit_task(
        gateway_url, model, prompt, seconds, size, fps, steps, seed,
        idem_key, concurrency_params["max_submit_retries"],
        concurrency_params["generate_sound"],
    )
    if task_id is None:
        return {
            "prompt": prompt, "index": index, "status": "failed",
            "error": f"submit: {err}", "seed": seed,
        }

    task_data, err = poll_task(
        gateway_url, task_id,
        concurrency_params["poll_interval"],
        concurrency_params["poll_timeout"],
    )
    if task_data is None:
        return {
            "prompt": prompt, "index": index, "status": "failed",
            "task_id": task_id, "error": f"poll: {err}", "seed": seed,
        }

    ok, err = download_video(
        gateway_url, task_id, output_path,
        concurrency_params["max_download_retries"],
    )
    if not ok:
        return {
            "prompt": prompt, "index": index, "status": "failed",
            "task_id": task_id, "error": f"download: {err}", "seed": seed,
        }

    return {
        "prompt": prompt, "index": index, "status": "generated",
        "task_id": task_id, "output": output_path, "seed": seed,
    }


def main():
    args = parse_args()

    if args.list_dimensions:
        print("Available VBench dimensions (16):")
        for dim in ALL_DIMENSIONS:
            pf = DIMENSION_TO_PROMPT_FILE[dim]
            nv = VIDEOS_PER_PROMPT_25 if pf in PROMPT_FILE_NEEDS_25 else VIDEOS_PER_PROMPT
            print(f"  {dim:30s}  prompt_file={pf:25s}  videos/prompt={nv}")
        return

    if not args.all and not args.dimensions:
        print("ERROR: specify --dimensions or --all")
        sys.exit(1)

    if not args.output_dir:
        print("ERROR: --output_dir is required (unless using --list_dimensions)")
        sys.exit(1)

    dimensions = ALL_DIMENSIONS if args.all else args.dimensions
    print(f"Dimensions ({len(dimensions)}): {', '.join(dimensions)}")

    prompt_files_needed = load_prompt_files(dimensions)
    prompts_info = load_prompts(prompt_files_needed)

    if args.max_prompts > 0 and len(prompts_info) > args.max_prompts:
        print(f"  --max_prompts: keeping first {args.max_prompts} of {len(prompts_info)} unique prompts")
        prompts_info = prompts_info[:args.max_prompts]

    all_tasks = []
    for pi in prompts_info:
        for idx in range(pi["num_videos"]):
            all_tasks.append({
                "prompt": pi["prompt"],
                "index": idx,
                "output_dir": args.output_dir,
            })

    total = len(all_tasks)
    unique_prompts = len(prompts_info)
    print(f"Unique prompts: {unique_prompts}")
    print(f"Total videos to generate: {total}")

    if total == 0:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\n--- Dry Run: Sampling Plan ---")
        for pi in prompts_info:
            print(f"  [{pi['prompt_file']:25s}] {pi['num_videos']:2d}x  {pi['prompt'][:60]}")
        print(f"\nTotal: {total} videos, {unique_prompts} unique prompts")
        print(f"Output dir: {args.output_dir}")
        print(f"Video params: seconds={args.seconds}, size={args.size}, fps={args.fps}, "
              f"steps={args.num_inference_steps}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    concurrency_params = {
        "poll_interval": args.poll_interval,
        "poll_timeout": args.poll_timeout,
        "max_submit_retries": args.max_submit_retries,
        "max_download_retries": args.max_download_retries,
        "generate_sound": args.generate_sound,
    }

    results = []
    succeeded = 0
    skipped = 0
    failed = 0

    print(f"\nStarting video generation (concurrency={args.concurrency}) ...")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for item in all_tasks:
            future = executor.submit(
                generate_one_video,
                item, args.gateway_url, args.model, args.seconds,
                args.size, args.fps, args.num_inference_steps,
                args.base_seed, concurrency_params,
            )
            futures[future] = item

        with tqdm(total=total, desc="Sampling", unit="video") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = result["status"]
                if status == "generated":
                    succeeded += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    tqdm.write(
                        f"  FAILED: [{result['prompt'][:50]}-{result['index']}] "
                        f"{result.get('error', 'unknown')}"
                    )
                pbar.update(1)

    # Final sweep: tasks that timed out during polling may have completed on
    # the server shortly after the deadline. Re-check them once and download
    # any late results instead of counting them as failures.
    timed_out = [
        r for r in results
        if r["status"] == "failed" and r.get("task_id")
        and str(r.get("error", "")).startswith("poll: poll timeout")
    ]
    if timed_out:
        print(f"\nRe-checking {len(timed_out)} timed-out task(s) for late completion...")
        for r in timed_out:
            data = check_task_once(args.gateway_url, r["task_id"])
            if data and data.get("status") == "completed":
                output_path = os.path.join(args.output_dir, f"{r['prompt']}-{r['index']}.mp4")
                ok, dl_err = download_video(
                    args.gateway_url, r["task_id"], output_path,
                    concurrency_params["max_download_retries"],
                )
                if ok:
                    r["status"] = "generated"
                    r["output"] = output_path
                    r["recovered"] = True
                    failed -= 1
                    succeeded += 1
                    print(f"  Recovered late completion: [{r['prompt'][:50]}-{r['index']}]")

    log_path = os.path.join(args.output_dir, "_sampling_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "gateway_url": args.gateway_url,
            "model": args.model,
            "dimensions": dimensions,
            "video_params": {
                "seconds": args.seconds,
                "size": args.size,
                "fps": args.fps,
                "num_inference_steps": args.num_inference_steps,
                "base_seed": args.base_seed,
                "generate_sound": args.generate_sound,
            },
            "summary": {
                "total": total,
                "generated": succeeded,
                "skipped": skipped,
                "failed": failed,
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Sampling complete:")
    print(f"  Generated: {succeeded}")
    print(f"  Skipped:   {skipped} (already existed)")
    print(f"  Failed:    {failed}")
    print(f"  Total:     {total}")
    print(f"  Log:       {log_path}")
    print(f"  Videos:    {args.output_dir}")
    if failed > 0:
        print(f"\n  {failed} video(s) failed. Re-run the same command to retry (completed videos will be skipped).")
        fail_rate = failed / total
        if fail_rate > args.max_fail_rate:
            print(f"\n  ERROR: failure rate {fail_rate:.1%} exceeds --max_fail_rate ({args.max_fail_rate:.1%}).")
            print(f"  Exiting with code 1 so the pipeline stops instead of evaluating incomplete data.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Next step - run VBench evaluation:")
    print(f"  python evaluate_videos.py \\")
    print(f"    --videos_path {args.output_dir} \\")
    if args.all:
        print(f"    --all \\")
    else:
        print(f"    --dimensions {' '.join(dimensions)} \\")
    print(f"    --output_path {os.path.join(args.output_dir, 'evaluation_results')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
