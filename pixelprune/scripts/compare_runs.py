#!/usr/bin/env python3
"""Compare two VLMEvalKit run directories: baseline vs PixelPrune."""

import json
import csv
import os
import glob
import re
import argparse

BASELINE_DIR   = "/home/xilab_program/PixelPrune/eval/outputs/full_baseline_qwen36_moe/Qwen3.5-HF/T20260707-133729/MLVU_MCQ_64frame"
# BASELINE_DIR   = "/data/lijie/PixelPrune/eval/outputs/pixelprune_doc_qwen36/Qwen3.5-HF/T20260510-102147/TextVQA_VAL"
PIXELPRUNE_DIR = "/home/xilab_program/PixelPrune/eval/outputs/pixelprune_doc_qwen36_moe/Qwen3.5-HF/T20260706-190414/MLVU_MCQ_64frame"


def rank_num(path):
    match = re.search(r"\.rank(\d+)\.jsonl$", path)
    return int(match.group(1)) if match else -1


def load_jsonl_sums(paths, fields):
    sums = {f: 0.0 for f in fields}
    count = 0
    per_file_sums = []
    for path in paths:
        file_sums = {f: 0.0 for f in fields}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                count += 1
                for field in fields:
                    sums[field] += rec[field]
                    file_sums[field] += rec[field]
        per_file_sums.append(file_sums)
    return sums, count, per_file_sums


def load_vit_retain(paths):
    total_org = 0
    total_new = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                total_org += sum(rec["org_vit_lens"])
                total_new += sum(rec["new_vit_lens"])
    return total_new / total_org if total_org > 0 else float("nan")


def load_acc(path):
    if path.endswith("_score.json"):
        with open(path) as f:
            return json.load(f)["Final Score Norm"]
    if path.endswith("_rating.json"):
        with open(path) as f:
            d = json.load(f)
        return float(d["overall"]["overall"])
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return float(row["Overall"])


def find_file(run_dir, pattern):
    for fname in os.listdir(run_dir):
        if fname.endswith(pattern):
            return os.path.join(run_dir, fname)
    raise FileNotFoundError(f"No file matching *{pattern} in {run_dir}")


def find_rank_files(run_dir, log_type):
    paths = glob.glob(os.path.join(run_dir, f"*.{log_type}.rank*.jsonl"))
    if not paths:
        paths = glob.glob(os.path.join(run_dir, f"*{log_type}*.rank*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No {log_type} rank jsonl files in {run_dir}")
    return sorted(paths, key=rank_num)


def find_acc_file(run_dir):
    for pattern in ("_acc.csv", "_score.json", "_rating.json"):
        try:
            return find_file(run_dir, pattern)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No acc file (_acc.csv, _score.json, or _rating.json) in {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two VLMEvalKit run directories: baseline vs PixelPrune."
    )
    parser.add_argument("base_dir", nargs="?", default=BASELINE_DIR)
    parser.add_argument("pp_dir", nargs="?", default=PIXELPRUNE_DIR)
    parser.add_argument(
        "--time-mode",
        choices=("sum", "wall"),
        default="sum",
        help=(
            "sum: add all rank records; wall: estimate multi-GPU wall time "
            "by taking the max per-rank summed time"
        ),
    )
    return parser.parse_args()


def aggregate_time(total_sums, per_rank_sums, field, mode):
    if mode == "sum":
        return total_sums[field]
    return max(rank_sums[field] for rank_sums in per_rank_sums)


def main():
    args = parse_args()
    base_dir = args.base_dir
    pp_dir = args.pp_dir

    # --- e2e logs ---
    base_e2e_files = find_rank_files(base_dir, "e2e")
    pp_e2e_files = find_rank_files(pp_dir, "e2e")
    pp_vit_files = find_rank_files(pp_dir, "vit")

    base_e2e, base_e2e_count, base_e2e_per_rank = load_jsonl_sums(
        base_e2e_files,
        ["ttft_ms", "total_time_s"],
    )
    pp_e2e, pp_e2e_count, pp_e2e_per_rank = load_jsonl_sums(
        pp_e2e_files,
        ["ttft_ms", "total_time_s"],
    )

    # Convert ttft ms to seconds. total_time_s is already in seconds.
    base_ttft   = aggregate_time(base_e2e, base_e2e_per_rank, "ttft_ms", args.time_mode) / 1000
    base_total  = aggregate_time(base_e2e, base_e2e_per_rank, "total_time_s", args.time_mode)
    pp_ttft     = aggregate_time(pp_e2e, pp_e2e_per_rank, "ttft_ms", args.time_mode) / 1000
    pp_total    = aggregate_time(pp_e2e, pp_e2e_per_rank, "total_time_s", args.time_mode)

    ttft_speedup  = base_ttft  / pp_ttft
    total_speedup = base_total / pp_total

    # --- vit retain ratio ---
    retain = load_vit_retain(pp_vit_files)

    # --- accuracy ---
    base_acc = load_acc(find_acc_file(base_dir))
    pp_acc   = load_acc(find_acc_file(pp_dir))
    acc_drop = (pp_acc - base_acc) / base_acc * 100

    # --- print table ---
    # col 0: left-aligned label; cols 1-6: right-aligned values
    col_w   = [22, 16, 12, 18, 16, 18, 14]
    headers = ["", "Acc (%)", "TTFT (s)", "TTFT Speedup (%)", "Total Time (s)", "Total Speedup (%)", "Token Retain"]

    def row_str(cells):
        parts = []
        for i, (c, w) in enumerate(zip(cells, col_w)):
            parts.append(str(c).ljust(w) if i == 0 else str(c).rjust(w))
        return "  ".join(parts)

    sep = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    print()
    print(f"Time mode: {args.time_mode}")
    print(f"Baseline logs: {len(base_e2e_files)} e2e rank file(s), {base_e2e_count} record(s)")
    print(f"PixelPrune logs: {len(pp_e2e_files)} e2e rank file(s), {pp_e2e_count} record(s); {len(pp_vit_files)} vit rank file(s)")
    print(row_str(headers))
    print(sep)
    print(row_str([
        "Original",
        f"{base_acc:.2f}",
        f"{base_ttft:.1f}",
        "—",
        f"{base_total:.1f}",
        "—",
        "—",
    ]))
    print(row_str([
        "Qwen3.5-PixelPrune",
        f"{pp_acc:.2f} ({acc_drop:+.2f}%)",
        f"{pp_ttft:.1f}",
        f"{(ttft_speedup - 1) * 100:.1f}%",
        f"{pp_total:.1f}",
        f"{(total_speedup - 1) * 100:.1f}%",
        f"{retain:.4f}",
    ]))
    print()


if __name__ == "__main__":
    main()
