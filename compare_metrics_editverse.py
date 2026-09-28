#!/usr/bin/env python3
"""Compare latency, memory, and token costs from two EditVerse metric runs.

EditVerse writes the same ``sample_id`` (``generate``) for every metric row.
This tool therefore maps rank-local metric order back to task IDs using each
run's ``manifest.json`` and completed ``results/<task-id>/generate.mp4`` files.
It then pairs the common task IDs after Ulysses aggregation, making interrupted
runs with different missing tasks safe to compare.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LATENCY_FIELDS = (
    "preprocess_total_latency_ms",
    "transform_inputs_latency_ms",
    "source_vit_latency_ms",
    "pixelprune_selector_latency_ms",
    "pixelprune_index_conversion_latency_ms",
    "pixelprune_sequence_rebuild_latency_ms",
    "pixelprune_overhead_ms",
    "planner_mllm_forward_latency_ms",
    "connector_latency_ms",
    "vit_decoder_latency_ms",
    "planner_latency_ms",
    "dit_diffusion_latency_ms",
    "vae_encode_latency_ms",
    "vae_decode_latency_ms",
    "e2e_excluding_io_ms",
    "e2e_including_io_ms",
)

MEMORY_FIELDS = ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")
TOKEN_FIELDS = (
    "visual_input_tokens_before",
    "visual_input_tokens_after",
    "visual_input_token_reduction_ratio",
    "mllm_sequence_length_before",
    "mllm_sequence_length_after",
)


@dataclass(frozen=True)
class Record:
    """A metric record plus the key used to pair it with the other method."""

    key: tuple[int, int]
    values: dict[str, Any]


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def numeric_values(records: Iterable[Record], field: str) -> list[float]:
    values = []
    for record in records:
        value = record.values.get(field)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def aggregate(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def load_rank_records(run_dir: Path) -> dict[int, list[Record]]:
    paths = sorted(run_dir.glob("metrics.rank*.jsonl"))
    if not paths:
        raise ValueError(f"No metrics.rank*.jsonl files found in {run_dir}")

    records_by_rank: dict[int, list[Record]] = {}
    for path in paths:
        try:
            rank = int(path.stem.removeprefix("metrics.rank"))
        except ValueError as error:
            raise ValueError(f"Unexpected metric filename: {path.name}") from error
        rank_records = []
        with path.open(encoding="utf-8") as file:
            for ordinal, line in enumerate(file):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}:{ordinal + 1}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object in {path}:{ordinal + 1}")
                rank_records.append(Record((rank, ordinal), value))
        records_by_rank[rank] = rank_records
    return records_by_rank


def load_completed_task_ids(run_dir: Path, dp_rank: int, dp_size: int) -> list[int]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing manifest: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected an object in {manifest_path}")
    try:
        ordered_ids = sorted((int(task_id) for task_id in manifest), key=int)
    except ValueError as error:
        raise ValueError(f"Manifest task IDs must be integers: {manifest_path}") from error
    if len(ordered_ids) != len(manifest) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError(f"Duplicate task IDs in {manifest_path}")

    completed = []
    for task_id in ordered_ids[dp_rank::dp_size]:
        entry = manifest[str(task_id)]
        output = Path(entry.get("output", "")) if isinstance(entry, dict) else Path()
        if output.is_file() and output.stat().st_size > 0:
            completed.append(task_id)
    return completed


def collapse_ulysses_records(run_dir: Path, ulysses: int) -> tuple[list[Record], dict[int, int]]:
    if ulysses < 1:
        raise ValueError(f"ulysses must be >= 1, got {ulysses}")
    records_by_rank = load_rank_records(run_dir)
    ranks = sorted(records_by_rank)
    if ranks != list(range(len(ranks))) or len(ranks) % ulysses:
        raise ValueError(
            f"Ranks must be contiguous and divisible by Ulysses={ulysses}; found {ranks}"
        )

    dp_size = len(ranks) // ulysses
    collapsed = []
    raw_counts = {rank: len(records) for rank, records in records_by_rank.items()}
    for dp_rank in range(dp_size):
        group_ranks = list(range(dp_rank * ulysses, (dp_rank + 1) * ulysses))
        task_ids = load_completed_task_ids(run_dir, dp_rank, dp_size)
        group_records = [records_by_rank[rank] for rank in group_ranks]
        counts = [len(records) for records in group_records]
        if any(count != len(task_ids) for count in counts):
            raise ValueError(
                f"Metric/task count mismatch in DP group {dp_rank} of {run_dir}: "
                f"tasks={len(task_ids)}, ranks={dict(zip(group_ranks, counts))}"
            )
        for ordinal, task_id in enumerate(task_ids):
            rank_records = [records[ordinal] for records in group_records]
            values = dict(rank_records[0].values)
            numeric_keys = {
                field
                for record in rank_records
                for field, value in record.values.items()
                if isinstance(value, (int, float)) and math.isfinite(value)
            }
            for field in numeric_keys:
                values[field] = max(
                    float(record.values[field])
                    for record in rank_records
                    if isinstance(record.values.get(field), (int, float))
                    and math.isfinite(record.values[field])
                )
            values["task_id"] = task_id
            values["ulysses_ranks"] = group_ranks
            collapsed.append(Record((dp_rank, ordinal), values))
    return collapsed, raw_counts


def pair_records(
    baseline: list[Record], pixelprune: list[Record]
) -> tuple[list[tuple[Record, Record]], dict[str, list[int]]]:
    baseline_by_task = {int(record.values["task_id"]): record for record in baseline}
    pixelprune_by_task = {int(record.values["task_id"]): record for record in pixelprune}
    if len(baseline_by_task) != len(baseline) or len(pixelprune_by_task) != len(pixelprune):
        raise ValueError("Duplicate task ID after manifest-based metric mapping")
    only_baseline = sorted(baseline_by_task.keys() - pixelprune_by_task.keys())
    only_pixelprune = sorted(pixelprune_by_task.keys() - baseline_by_task.keys())
    common = sorted(baseline_by_task.keys() & pixelprune_by_task.keys())
    return (
        [(baseline_by_task[task_id], pixelprune_by_task[task_id]) for task_id in common],
        {"only_baseline": only_baseline, "only_pixelprune": only_pixelprune},
    )


def compare_field(
    pairs: list[tuple[Record, Record]], field: str, statistic: str
) -> dict[str, Any] | None:
    baseline_values = []
    pixelprune_values = []
    paired_delta_values = []
    for baseline, pixelprune in pairs:
        baseline_value = baseline.values.get(field)
        pixelprune_value = pixelprune.values.get(field)
        if not (
            isinstance(baseline_value, (int, float))
            and isinstance(pixelprune_value, (int, float))
            and math.isfinite(baseline_value)
            and math.isfinite(pixelprune_value)
        ):
            continue
        baseline_values.append(float(baseline_value))
        pixelprune_values.append(float(pixelprune_value))
        paired_delta_values.append(float(pixelprune_value) - float(baseline_value))

    baseline_summary = aggregate(baseline_values)
    pixelprune_summary = aggregate(pixelprune_values)
    paired_delta = aggregate(paired_delta_values)
    if baseline_summary is None or pixelprune_summary is None or paired_delta is None:
        return None

    baseline_stat = float(baseline_summary[statistic])
    pixelprune_stat = float(pixelprune_summary[statistic])
    result: dict[str, Any] = {
        "baseline": baseline_summary,
        "pixelprune": pixelprune_summary,
        "paired_delta_pixelprune_minus_baseline": paired_delta,
        "delta_at_selected_stat": pixelprune_stat - baseline_stat,
    }
    if baseline_stat:
        result["reduction_ratio_at_selected_stat"] = 1 - pixelprune_stat / baseline_stat
    if pixelprune_stat:
        result["speedup_at_selected_stat"] = baseline_stat / pixelprune_stat
    return result


def selected(summary: dict[str, Any], statistic: str) -> float:
    return float(summary[statistic])


def component_summary(records: list[Record], statistic: str) -> dict[str, Any]:
    components = {
        field: aggregate(numeric_values(records, field))
        for field in (
            "pixelprune_selector_latency_ms",
            "pixelprune_index_conversion_latency_ms",
            "pixelprune_sequence_rebuild_latency_ms",
            "pixelprune_overhead_ms",
        )
    }
    components = {field: value for field, value in components.items() if value is not None}
    return {
        "components": components,
        "selected_statistic": statistic,
        "reported_total_overhead_ms": components.get("pixelprune_overhead_ms"),
    }


def net_vit_path(
    pairs: list[tuple[Record, Record]], statistic: str
) -> dict[str, float] | None:
    baseline_vit = []
    pixelprune_vit = []
    overhead = []
    net_savings = []
    for baseline, pixelprune in pairs:
        baseline_value = baseline.values.get("source_vit_latency_ms")
        pixelprune_value = pixelprune.values.get("source_vit_latency_ms")
        overhead_value = pixelprune.values.get("pixelprune_overhead_ms")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (baseline_value, pixelprune_value, overhead_value)
        ):
            continue
        baseline_vit.append(float(baseline_value))
        pixelprune_vit.append(float(pixelprune_value))
        overhead.append(float(overhead_value))
        net_savings.append(
            float(baseline_value) - float(pixelprune_value) - float(overhead_value)
        )
    summaries = [aggregate(values) for values in (baseline_vit, pixelprune_vit, overhead, net_savings)]
    if any(summary is None for summary in summaries):
        return None
    baseline_summary, pixelprune_summary, overhead_summary, net_summary = summaries
    assert baseline_summary and pixelprune_summary and overhead_summary and net_summary
    baseline_stat = selected(baseline_summary, statistic)
    pixelprune_stat = selected(pixelprune_summary, statistic)
    overhead_stat = selected(overhead_summary, statistic)
    return {
        "baseline_source_vit_ms": baseline_stat,
        "pixelprune_source_vit_ms": pixelprune_stat,
        "gross_source_vit_saving_ms": baseline_stat - pixelprune_stat,
        "pixelprune_overhead_ms": overhead_stat,
        "net_source_vit_path_saving_ms": selected(net_summary, statistic),
        "net_source_vit_path_reduction_ratio": selected(net_summary, statistic) / baseline_stat
        if baseline_stat
        else math.nan,
    }


def build_report(
    baseline_dir: Path, pixelprune_dir: Path, statistic: str, ulysses: int = 2
) -> dict[str, Any]:
    baseline, baseline_raw_counts = collapse_ulysses_records(baseline_dir, ulysses)
    pixelprune, pixelprune_raw_counts = collapse_ulysses_records(pixelprune_dir, ulysses)
    pairs, unpaired = pair_records(baseline, pixelprune)
    if not pairs:
        raise ValueError("No common completed task IDs remain after manifest-based mapping")
    fields = LATENCY_FIELDS + MEMORY_FIELDS + TOKEN_FIELDS
    comparisons = {
        field: comparison
        for field in fields
        if (comparison := compare_field(pairs, field, statistic)) is not None
    }
    ranks = sorted(baseline_raw_counts)
    return {
        "baseline_dir": str(baseline_dir),
        "pixelprune_dir": str(pixelprune_dir),
        "pairing": "manifest task ID after Ulysses aggregation",
        "logical_sample_count": len(pairs),
        "raw_rank_record_count": {
            "baseline": sum(baseline_raw_counts.values()),
            "pixelprune": sum(pixelprune_raw_counts.values()),
        },
        "world_ranks": ranks,
        "ulysses": ulysses,
        "unpaired_logical_samples": unpaired,
        "selected_statistic": statistic,
        "comparisons": comparisons,
        "pixelprune_overhead": component_summary(
            [pixelprune_record for _, pixelprune_record in pairs], statistic
        ),
        "net_source_vit_path": net_vit_path(pairs, statistic),
    }


def ms(value: float) -> str:
    return f"{value:,.1f} ms"


def print_report(report: dict[str, Any]) -> None:
    statistic = report["selected_statistic"]
    print(
        f"Paired {report['logical_sample_count']} logical samples over world ranks "
        f"{report['world_ranks']} (Ulysses={report['ulysses']}; {statistic})."
    )
    unpaired = report["unpaired_logical_samples"]
    if unpaired["only_baseline"] or unpaired["only_pixelprune"]:
        print(
            "Dropped unpaired task IDs: "
            f"baseline-only={len(unpaired['only_baseline'])}, "
            f"pixelprune-only={len(unpaired['only_pixelprune'])}."
        )
    print("\nLatency (PixelPrune - baseline; negative means faster)")
    print(f"{'metric':42} {'baseline':>14} {'pixelprune':>14} {'delta':>14} {'reduction':>10}")
    for field in LATENCY_FIELDS:
        item = report["comparisons"].get(field)
        if item is None:
            continue
        baseline_value = selected(item["baseline"], statistic)
        pixelprune_value = selected(item["pixelprune"], statistic)
        reduction = item.get("reduction_ratio_at_selected_stat")
        reduction_text = f"{reduction:.1%}" if reduction is not None else "n/a"
        print(
            f"{field:42} {ms(baseline_value):>14} {ms(pixelprune_value):>14} "
            f"{ms(float(item['delta_at_selected_stat'])):>14} {reduction_text:>10}"
        )

    print("\nMemory (values shown in GiB)")
    print(f"{'metric':42} {'baseline':>14} {'pixelprune':>14} {'delta':>14} {'reduction':>10}")
    for field in MEMORY_FIELDS:
        item = report["comparisons"].get(field)
        if item is None:
            continue
        baseline_value = selected(item["baseline"], statistic) / 1024**3
        pixelprune_value = selected(item["pixelprune"], statistic) / 1024**3
        delta = pixelprune_value - baseline_value
        reduction = item.get("reduction_ratio_at_selected_stat")
        reduction_text = f"{reduction:.1%}" if reduction is not None else "n/a"
        print(
            f"{field:42} {baseline_value:>12.2f} GiB {pixelprune_value:>12.2f} GiB "
            f"{delta:>+12.2f} GiB {reduction_text:>10}"
        )

    print("\nToken counts")
    print(f"{'metric':42} {'baseline':>14} {'pixelprune':>14} {'delta':>14} {'reduction':>10}")
    for field in (
        "visual_input_tokens_before",
        "visual_input_tokens_after",
        "mllm_sequence_length_before",
        "mllm_sequence_length_after",
    ):
        item = report["comparisons"].get(field)
        if item is None:
            continue
        baseline_value = selected(item["baseline"], statistic)
        pixelprune_value = selected(item["pixelprune"], statistic)
        reduction = item.get("reduction_ratio_at_selected_stat")
        reduction_text = f"{reduction:.1%}" if reduction is not None else "n/a"
        print(
            f"{field:42} {baseline_value:>14,.0f} {pixelprune_value:>14,.0f} "
            f"{pixelprune_value - baseline_value:>+14,.0f} {reduction_text:>10}"
        )

    net = report.get("net_source_vit_path")
    if net:
        print("\nPixelPrune overhead and source-ViT-path effect")
        components = report["pixelprune_overhead"]["components"]
        for field in (
            "pixelprune_selector_latency_ms",
            "pixelprune_index_conversion_latency_ms",
            "pixelprune_sequence_rebuild_latency_ms",
        ):
            component = components.get(field)
            if component:
                print(f"  {field}: {ms(selected(component, statistic))}")
        print(f"  gross source-ViT saving: {ms(net['gross_source_vit_saving_ms'])}")
        print(f"  PixelPrune overhead:     {ms(net['pixelprune_overhead_ms'])}")
        print(f"  net source-ViT saving:   {ms(net['net_source_vit_path_saving_ms'])} "
              f"({net['net_source_vit_path_reduction_ratio']:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path, help="Directory containing baseline metrics.rank*.jsonl")
    parser.add_argument("pixelprune_dir", type=Path, help="Directory containing PixelPrune metrics.rank*.jsonl")
    parser.add_argument(
        "--statistic",
        choices=("mean", "median", "p90"),
        default="mean",
        help="Aggregate used for the headline comparison (default: mean)",
    )
    parser.add_argument(
        "--ulysses",
        type=int,
        default=2,
        help="Ulysses sequence-parallel degree (default: 2)",
    )
    parser.add_argument("--output", type=Path, help="Write the full machine-readable JSON report here")
    args = parser.parse_args()

    try:
        report = build_report(args.baseline_dir, args.pixelprune_dir, args.statistic, args.ulysses)
    except ValueError as error:
        parser.error(str(error))
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nFull JSON report: {args.output}")


if __name__ == "__main__":
    main()
