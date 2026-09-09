#!/usr/bin/env python3
"""Compare baseline and PixelPrune MLVU runs using MCQ accuracy."""

import json
import os
import re
import sys

import pandas as pd


BASELINE_DIR = "/home/xilab_program/PixelPrune/eval/outputs/full_baseline_qwen36_moe/Qwen3.5-HF/T20260707-133729/MLVU_MCQ_64frame"
PIXELPRUNE_DIR = "/home/xilab_program/PixelPrune/eval/outputs/pixelprune_doc_qwen36_moe/Qwen3.5-HF/T20260707-191917/MLVU_MCQ_64frame"


def load_jsonl_sums(path, fields):
    sums = {field: 0.0 for field in fields}
    count = 0
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            count += 1
            for field in fields:
                sums[field] += record[field]
    return sums, count


def load_jsonl_rank_stats(paths, fields):
    total_sums = {field: 0.0 for field in fields}
    total_count = 0
    rank_sums = []
    for path in paths:
        sums, count = load_jsonl_sums(path, fields)
        rank_sums.append((path, sums, count))
        total_count += count
        for field in fields:
            total_sums[field] += sums[field]
    if total_count == 0:
        raise ValueError(f"No records found in: {paths}")
    return total_sums, total_count, rank_sums


def load_e2e_stats(run_dir):
    files = find_rank_files(run_dir, "e2e")
    sums, count, rank_sums = load_jsonl_rank_stats(
        files,
        ["ttft_ms", "total_time_s"],
    )
    return {
        "files": files,
        "count": count,
        "wall_ttft_s": max(item[1]["ttft_ms"] for item in rank_sums) / 1000,
        "wall_time_s": max(item[1]["total_time_s"] for item in rank_sums),
        "gpu_time_s": sums["total_time_s"],
    }


def load_vit_retain(paths):
    total_org = 0
    total_new = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                total_org += sum(record["org_vit_lens"])
                total_new += sum(record["new_vit_lens"])
    return total_new / total_org if total_org > 0 else float("nan")


def find_file(run_dir, pattern):
    matches = sorted(
        os.path.join(run_dir, filename)
        for filename in os.listdir(run_dir)
        if filename.endswith(pattern)
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *{pattern} file in {run_dir}, found {len(matches)}"
        )
    return matches[0]


def find_rank_files(run_dir, kind):
    rank_pattern = re.compile(r"\.rank(\d+)\.jsonl$")
    matches = []
    ranks = set()
    for filename in os.listdir(run_dir):
        if f".{kind}.rank" not in filename or not filename.endswith(".jsonl"):
            continue
        match = rank_pattern.search(filename)
        if not match:
            continue
        rank = int(match.group(1))
        if rank in ranks:
            raise ValueError(f"Duplicate {kind} rank{rank} files in {run_dir}")
        ranks.add(rank)
        matches.append(os.path.join(run_dir, filename))
    if not matches:
        raise FileNotFoundError(f"Expected *.{kind}.rank*.jsonl files in {run_dir}")
    return sorted(matches, key=lambda path: int(rank_pattern.search(path).group(1)))


def find_mcq_dir(run_dir):
    run_dir = os.path.abspath(run_dir)
    if os.path.basename(run_dir) == "MLVU_MCQ_64frame":
        return run_dir
    if os.path.basename(run_dir) != "MLVU_64frame":
        raise ValueError(f"Expected an MLVU_64frame directory, got: {run_dir}")
    return os.path.join(os.path.dirname(run_dir), "MLVU_MCQ_64frame")


def ensure_mcq_score(run_dir):
    mcq_dir = find_mcq_dir(run_dir)
    if os.path.isdir(mcq_dir):
        score_files = sorted(
            os.path.join(mcq_dir, filename)
            for filename in os.listdir(mcq_dir)
            if filename.endswith("_score.xlsx")
        )
        if len(score_files) == 1:
            return score_files[0]
        if len(score_files) > 1:
            raise ValueError(
                f"Expected at most one *_score.xlsx in {mcq_dir}, "
                f"found {len(score_files)}"
            )

    from vlmeval.dataset.mlvu import MLVU_MCQ

    prediction_file = find_file(run_dir, "_MLVU_64frame.xlsx")
    data = pd.read_excel(prediction_file)
    required_columns = {
        "SUB_DATASET",
        "index",
        "original_index",
        "prediction",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns in {prediction_file}: {sorted(missing_columns)}"
        )

    mcq_data = data[data["SUB_DATASET"] == "MLVU_MCQ"].copy()
    if mcq_data.empty:
        raise ValueError(f"No MLVU_MCQ samples found in {prediction_file}")
    if mcq_data["prediction"].isna().any():
        raise ValueError(f"Missing MCQ predictions in {prediction_file}")

    mcq_data.pop("index")
    mcq_data["index"] = mcq_data.pop("original_index")
    mcq_data.pop("SUB_DATASET")

    os.makedirs(mcq_dir, exist_ok=True)
    mcq_prediction_file = os.path.join(
        mcq_dir,
        os.path.basename(prediction_file).replace(
            "MLVU_64frame", "MLVU_MCQ_64frame"
        ),
    )
    mcq_data.to_excel(mcq_prediction_file, index=False)

    print(f"MCQ score not found; evaluating {mcq_prediction_file}")
    MLVU_MCQ.evaluate(
        mcq_prediction_file,
        model="exact_matching",
        nproc=1,
        verbose=False,
    )
    return find_file(mcq_dir, "_score.xlsx")


def load_mcq_acc(run_dir):
    score_file = ensure_mcq_score(run_dir)
    data = pd.read_excel(score_file)
    if "score" not in data.columns:
        raise ValueError(f"No 'score' column in {score_file}")
    if data["score"].isna().any():
        raise ValueError(f"Missing scores in {score_file}")
    return float(data["score"].mean() * 100), len(data), int(data["score"].sum())


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else BASELINE_DIR
    pp_dir = sys.argv[2] if len(sys.argv) > 2 else PIXELPRUNE_DIR

    base_e2e = load_e2e_stats(base_dir)
    pp_e2e = load_e2e_stats(pp_dir)
    if base_e2e["count"] != pp_e2e["count"]:
        raise ValueError(
            "E2E sample counts differ: "
            f"baseline={base_e2e['count']}, PixelPrune={pp_e2e['count']}"
        )

    base_ttft = base_e2e["wall_ttft_s"]
    pp_ttft = pp_e2e["wall_ttft_s"]
    base_total = base_e2e["wall_time_s"]
    pp_total = pp_e2e["wall_time_s"]

    ttft_speedup = base_ttft / pp_ttft
    total_speedup = base_total / pp_total
    retain = load_vit_retain(find_rank_files(pp_dir, "vit"))

    base_acc, base_count, base_correct = load_mcq_acc(base_dir)
    pp_acc, pp_count, pp_correct = load_mcq_acc(pp_dir)
    if base_count != pp_count:
        raise ValueError(
            f"MCQ sample counts differ: baseline={base_count}, PixelPrune={pp_count}"
        )
    acc_delta = pp_acc - base_acc

    col_w = [22, 20, 12, 18, 16, 18, 14]
    headers = [
        "",
        "M-Avg (%)",
        "TTFT Wall (s)",
        "TTFT Speedup (%)",
        "Wall Time (s)",
        "Total Speedup (%)",
        "Token Retain",
    ]

    def row_str(cells):
        parts = []
        for index, (cell, width) in enumerate(zip(cells, col_w)):
            parts.append(
                str(cell).ljust(width) if index == 0 else str(cell).rjust(width)
            )
        return "  ".join(parts)

    separator = "-" * (sum(col_w) + 2 * (len(col_w) - 1))

    print()
    print(row_str(headers))
    print(separator)
    print(
        row_str(
            [
                "Original",
                f"{base_acc:.2f} ({base_correct}/{base_count})",
                f"{base_ttft:.1f}",
                "-",
                f"{base_total:.1f}",
                "-",
                "-",
            ]
        )
    )
    print(
        row_str(
            [
                "Qwen3.5-PixelPrune",
                f"{pp_acc:.2f} ({acc_delta:+.2f} pp)",
                f"{pp_ttft:.1f}",
                f"{(ttft_speedup - 1) * 100:.1f}%",
                f"{pp_total:.1f}",
                f"{(total_speedup - 1) * 100:.1f}%",
                f"{retain:.4f}",
            ]
        )
    )
    print()


if __name__ == "__main__":
    main()
